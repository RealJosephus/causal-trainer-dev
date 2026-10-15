"""Pure functional JAX decoder used by the training step."""

from __future__ import annotations

import operator
from collections.abc import Mapping, Sequence
from functools import partial
from typing import Any, Literal

import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec

from ..kernels.efficient_attention import (
    AttentionKernelProfile,
    AttentionMetadata,
    prepare_attention_metadata,
    resolve_attention_profile,
)
from .attention import AttentionImplementation, _attention_precanonicalized
from .attention_options import AttentionOptions
from .config import ModelConfig
from .lora import lora_linear_delta
from .rotary import (
    _make_position_ids_precanonicalized,
    apply_rotary_cos_sin,
    canonicalize_segment_ids,
    inverse_frequencies,
    rotary_cos_sin,
)
from .token_tiling import (
    tiled_final_norm,
    tiled_norm_mlp,
    tiled_norm_qkv,
    tiled_output_projection,
)

Array = jax.Array
Params = Mapping[str, Any]
RematPolicy = Literal["none", "nothing_saveable"]


def _dense_attention_layer(options, layer_index, layers, step):
    if options is None or not options.needs_dense_backend:
        return False
    if options.method == "dense":
        return True
    selected = (layer_index < options.dense_prefix_layers) | (layer_index >= layers - options.dense_suffix_layers)
    if options.dense_after_step is not None:
        selected = selected | (step >= options.dense_after_step)
    return selected


def _shape(shape: tuple[int, ...], dtype: jnp.dtype) -> jax.ShapeDtypeStruct:
    return jax.ShapeDtypeStruct(shape, dtype)


def parameter_shapes(
    config: ModelConfig,
    param_dtype: jnp.dtype = jnp.bfloat16,
) -> dict[str, Any]:
    """Return the exact parameter PyTree as ``ShapeDtypeStruct`` leaves."""

    dtype = jnp.dtype(param_dtype)

    def projection(in_features: int, out_features: int, *, bias: bool) -> dict[str, Any]:
        result: dict[str, Any] = {"kernel": _shape((in_features, out_features), dtype)}
        if bias:
            result["bias"] = _shape((out_features,), dtype)
        return result

    layers = []
    for _ in range(config.num_hidden_layers):
        layers.append(
            {
                "attention": {
                    "q_proj": projection(config.hidden_size, config.query_width, bias=config.attention_bias),
                    "k_proj": projection(config.hidden_size, config.key_value_width, bias=config.attention_bias),
                    "v_proj": projection(config.hidden_size, config.key_value_width, bias=config.attention_bias),
                    # The checkpoint contract uses bias only for Q/K/V.  The
                    # output projection is always bias-free.
                    "o_proj": projection(config.query_width, config.hidden_size, bias=False),
                },
                "mlp": {
                    "gate_proj": {"kernel": _shape((config.hidden_size, config.intermediate_size), dtype)},
                    "up_proj": {"kernel": _shape((config.hidden_size, config.intermediate_size), dtype)},
                    "down_proj": {"kernel": _shape((config.intermediate_size, config.hidden_size), dtype)},
                },
                "input_layernorm": {"scale": _shape((config.hidden_size,), dtype)},
                "post_attention_layernorm": {"scale": _shape((config.hidden_size,), dtype)},
            }
        )

    return {
        "embed_tokens": {"embedding": _shape((config.vocab_size, config.hidden_size), dtype)},
        "layers": tuple(layers),
        "norm": {"scale": _shape((config.hidden_size,), dtype)},
        "lm_head": {"kernel": _shape((config.hidden_size, config.vocab_size), dtype)},
    }


def init_params(
    config: ModelConfig,
    rng: Array,
    param_dtype: jnp.dtype = jnp.bfloat16,
) -> dict[str, Any]:
    """Initialize the fixed parameter PyTree without module state."""

    dtype = jnp.dtype(param_dtype)
    kernel_keys = iter(jax.random.split(rng, 2 + 7 * config.num_hidden_layers))

    def kernel(shape: tuple[int, ...]) -> Array:
        value = jax.random.normal(next(kernel_keys), shape, dtype=jnp.float32)
        return (value * config.initializer_range).astype(dtype)

    def projection(in_features: int, out_features: int, *, bias: bool) -> dict[str, Array]:
        result = {"kernel": kernel((in_features, out_features))}
        if bias:
            result["bias"] = jnp.zeros((out_features,), dtype=dtype)
        return result

    layers = []
    for _ in range(config.num_hidden_layers):
        layers.append(
            {
                "attention": {
                    "q_proj": projection(config.hidden_size, config.query_width, bias=config.attention_bias),
                    "k_proj": projection(config.hidden_size, config.key_value_width, bias=config.attention_bias),
                    "v_proj": projection(config.hidden_size, config.key_value_width, bias=config.attention_bias),
                    "o_proj": projection(config.query_width, config.hidden_size, bias=False),
                },
                "mlp": {
                    "gate_proj": {"kernel": kernel((config.hidden_size, config.intermediate_size))},
                    "up_proj": {"kernel": kernel((config.hidden_size, config.intermediate_size))},
                    "down_proj": {"kernel": kernel((config.intermediate_size, config.hidden_size))},
                },
                "input_layernorm": {"scale": jnp.ones((config.hidden_size,), dtype=dtype)},
                "post_attention_layernorm": {"scale": jnp.ones((config.hidden_size,), dtype=dtype)},
            }
        )

    embedding = kernel((config.vocab_size, config.hidden_size))
    # Match torch.nn.Embedding(..., padding_idx=...): the padding row is
    # initialized to zero. Loaded checkpoints retain their stored row value;
    # gradient suppression is applied at lookup time below.
    embedding = embedding.at[config.pad_token_id].set(jnp.zeros((config.hidden_size,), dtype=dtype))
    return {
        "embed_tokens": {"embedding": embedding},
        "layers": tuple(layers),
        "norm": {"scale": jnp.ones((config.hidden_size,), dtype=dtype)},
        "lm_head": {"kernel": kernel((config.hidden_size, config.vocab_size))},
    }


def rms_norm(x: Array, scale: Array, eps: float, compute_dtype: jnp.dtype) -> Array:
    """RMS normalization with float32 statistics and learned multiplicative scale."""

    original_dtype = x.dtype
    work = x.astype(jnp.float32)
    normalized = work * jax.lax.rsqrt(jnp.mean(jnp.square(work), axis=-1, keepdims=True) + eps)
    output = normalized.astype(compute_dtype) * scale.astype(compute_dtype)
    return output.astype(original_dtype)


def _embedding_lookup(
    embedding: Array,
    input_ids: Array,
    padding_idx: int,
    compute_dtype: jnp.dtype,
) -> Array:
    """Apply the forward and gradient semantics of ``nn.Embedding.padding_idx``."""

    selected = embedding[input_ids]
    is_padding_index = input_ids == jnp.asarray(padding_idx, dtype=input_ids.dtype)
    # PyTorch does not force a loaded padding row to zero during forward; it
    # only excludes that row from gradient accumulation. Preserve that exact
    # behavior instead of replacing the selected values with zeros.
    selected = jnp.where(
        is_padding_index[..., None],
        jax.lax.stop_gradient(selected),
        selected,
    )
    return selected.astype(compute_dtype)


def linear(
    x: Array,
    params: Params,
    compute_dtype: jnp.dtype,
    adapter: Params | None = None,
) -> Array:
    """Apply a JAX-layout ``[in, out]`` kernel and optional bias."""

    x = x.astype(compute_dtype)
    kernel = params["kernel"].astype(compute_dtype)
    output = jnp.einsum("...i,io->...o", x, kernel, precision=None)
    if adapter is not None:
        output = output + lora_linear_delta(x, adapter, compute_dtype)
    bias = params.get("bias")
    if bias is not None:
        output = output + bias.astype(compute_dtype)
    return output


def _mlp_projection(
    hidden_states: Array,
    mlp_params: Params,
    mlp_adapters: Params | None,
    compute_dtype: jnp.dtype,
) -> Array:
    """Apply the three independent MLP projections to one token tile."""

    def adapter(projection: str) -> Params | None:
        if mlp_adapters is None:
            return None
        return mlp_adapters[projection]

    # Keep gate/up as separate parameter and LoRA paths. In particular, this
    # must not silently introduce a fused gate_up checkpoint contract.
    gate = jax.nn.silu(
        linear(hidden_states, mlp_params["gate_proj"], compute_dtype, adapter("gate_proj"))
    )
    up = linear(hidden_states, mlp_params["up_proj"], compute_dtype, adapter("up_proj"))
    return linear(
        gate * up,
        mlp_params["down_proj"],
        compute_dtype,
        adapter("down_proj"),
    )


def _tiled_mlp_projection(
    hidden_states: Array,
    mlp_params: Params,
    mlp_adapters: Params | None,
    compute_dtype: jnp.dtype,
    chunk_size: int,
) -> Array:
    """Optionally evaluate MLP sequence tiles with a rematerialized scan body."""

    if chunk_size == 0:
        return _mlp_projection(hidden_states, mlp_params, mlp_adapters, compute_dtype)

    batch, sequence, hidden_size = hidden_states.shape
    effective_chunk_size = min(chunk_size, sequence)
    pad = (-sequence) % effective_chunk_size
    padded = jnp.pad(hidden_states, ((0, 0), (0, pad), (0, 0)))
    num_chunks = (sequence + pad) // effective_chunk_size
    chunks = padded.reshape(batch, num_chunks, effective_chunk_size, hidden_size).transpose(1, 0, 2, 3)

    def project_chunk(chunk: Array, params: Params, adapters: Params | None) -> Array:
        return _mlp_projection(chunk, params, adapters, compute_dtype)

    # This inner remat is intentional even when the complete decoder layer is
    # checkpointed: scan stores compact token tiles and recomputes the large
    # gate/up intermediates during backward.
    project_chunk = jax.checkpoint(
        project_chunk,
        policy=jax.checkpoint_policies.nothing_saveable,
        prevent_cse=False,
    )

    def scan_chunk(carry: None, chunk: Array) -> tuple[None, Array]:
        return carry, project_chunk(chunk, mlp_params, mlp_adapters)

    _, projected_chunks = jax.lax.scan(scan_chunk, None, chunks)
    projected = projected_chunks.transpose(1, 0, 2, 3).reshape(
        batch,
        sequence + pad,
        hidden_size,
    )
    return projected[:, :sequence, :]


def _decoder_layer(
    layer_params: Params,
    layer_adapters: Params | None,
    hidden_states: Array,
    attention_mask: Array,
    rotary_cosine: Array,
    rotary_sine: Array,
    segment_ids: Array,
    attention_metadata: AttentionMetadata | None,
    dropout_rng: Array | None,
    layer_index: int | Array = 0,
    *,
    config: ModelConfig,
    implementation: AttentionImplementation,
    mesh: Mesh | None,
    deterministic: bool,
    compute_dtype: jnp.dtype,
    block_q: int,
    block_k: int,
    attention_profile: AttentionKernelProfile | None,
    mlp_chunk_size: int,
    gather_kv_projection_weights: bool,
    attention_options: AttentionOptions | None = None,
    attention_step: int | Array = 0,
) -> Array:
    def adapter(block: str, projection: str) -> Params | None:
        if layer_adapters is None:
            return None
        return layer_adapters[block][projection]

    residual = hidden_states
    normed = rms_norm(
        hidden_states,
        layer_params["input_layernorm"]["scale"],
        config.rms_norm_eps,
        compute_dtype,
    )
    attention_params = layer_params["attention"]
    query = linear(
        normed,
        attention_params["q_proj"],
        compute_dtype,
        adapter("attention", "q_proj"),
    )

    def kv_projection(name: str) -> Array:
        projection_params = attention_params[name]
        projection_adapter = adapter("attention", name)
        if gather_kv_projection_weights:
            if mesh is None:  # Guarded once in decoder_forward; keeps this helper total.
                raise ValueError("gather_kv_projection_weights requires a mesh")
            # Parameters and optimizer state remain in their ordinary TP-sharded
            # layout. Only the small K/V projection weight consumed by this matmul is
            # temporarily replicated over TP, so its activation output does not
            # inherit an output-feature TP partition. The LoRA B matrix has the
            # same output-feature layout and must follow the same constraint.
            kv_weight_sharding = NamedSharding(mesh, PartitionSpec("fsdp", None))
            projection_params = {
                **projection_params,
                "kernel": jax.lax.with_sharding_constraint(
                    projection_params["kernel"],
                    kv_weight_sharding,
                ),
            }
            if projection_adapter is not None:
                projection_adapter = {
                    **projection_adapter,
                    "lora_b": jax.lax.with_sharding_constraint(
                        projection_adapter["lora_b"],
                        kv_weight_sharding,
                    ),
                }
        return linear(normed, projection_params, compute_dtype, projection_adapter)

    key = kv_projection("k_proj")
    value = kv_projection("v_proj")

    batch, sequence = hidden_states.shape[:2]
    query = query.reshape(batch, sequence, config.num_attention_heads, config.head_dim)
    key = key.reshape(batch, sequence, config.num_key_value_heads, config.head_dim)
    value = value.reshape(batch, sequence, config.num_key_value_heads, config.head_dim)
    query, key = apply_rotary_cos_sin(
        query,
        key,
        rotary_cosine,
        rotary_sine,
        rope_style=config.rope_style,
    )
    attended = _attention_precanonicalized(
        query,
        key,
        value,
        attention_mask,
        segment_ids,
        implementation=implementation,
        scale=config.head_dim**-0.5,
        dropout_rate=config.attention_dropout,
        deterministic=deterministic,
        dropout_rng=dropout_rng,
        mesh=mesh,
        block_q=block_q,
        block_k=block_k,
        attention_metadata=attention_metadata,
        attention_profile=attention_profile,
        attention_options=attention_options,
        dense_attention=_dense_attention_layer(
            attention_options, layer_index, config.num_hidden_layers, attention_step,
        ),
    )
    attended = attended.reshape(batch, sequence, config.query_width)
    hidden_states = residual + linear(
        attended,
        attention_params["o_proj"],
        compute_dtype,
        adapter("attention", "o_proj"),
    )

    residual = hidden_states
    normed = rms_norm(
        hidden_states,
        layer_params["post_attention_layernorm"]["scale"],
        config.rms_norm_eps,
        compute_dtype,
    )
    mlp_params = layer_params["mlp"]
    mlp_adapters = None if layer_adapters is None else layer_adapters["mlp"]
    hidden_states = residual + _tiled_mlp_projection(
        normed,
        mlp_params,
        mlp_adapters,
        compute_dtype,
        mlp_chunk_size,
    )
    return hidden_states


def _stack_layer_trees(
    layers: Sequence[Params],
    *,
    expected_layers: int,
    name: str,
) -> Params:
    """Stack an external tuple-of-layers tree for an internal ``lax.scan``."""

    layers = tuple(layers)
    if len(layers) != expected_layers:
        raise ValueError(f"{name} tree has {len(layers)} layers, expected {expected_layers}")
    return jax.tree.map(lambda *leaves: jnp.stack(leaves, axis=0), *layers)


def _tiled_decoder_layer(
    layer_params: Params,
    layer_adapters: Params | None,
    hidden_states: Array,
    attention_mask: Array,
    rotary_cosine: Array,
    rotary_sine: Array,
    segment_ids: Array,
    attention_metadata: AttentionMetadata | None,
    dropout_rng: Array | None,
    layer_index: int | Array = 0,
    *,
    config: ModelConfig,
    implementation: AttentionImplementation,
    mesh: Mesh,
    block_q: int,
    block_k: int,
    attention_profile: AttentionKernelProfile | None,
    chunk_size: int,
    mlp_chunk_size: int,
    attention_options: AttentionOptions | None = None,
    attention_step: int | Array = 0,
) -> Array:
    """Keep sequence/hidden shards across layers; gather only token tiles.

    Attention still sees the original packed sequence order and canonical
    segment IDs. This path changes storage/recomputation, not the causal mask.
    """

    query, key, value = tiled_norm_qkv(
        hidden_states,
        layer_params["input_layernorm"]["scale"],
        layer_params["attention"],
        mesh=mesh,
        rms_norm_eps=config.rms_norm_eps,
        num_attention_heads=config.num_attention_heads,
        num_key_value_heads=config.num_key_value_heads,
        head_dim=config.head_dim,
        chunk_size=chunk_size,
    )
    query, key = apply_rotary_cos_sin(
        query, key, rotary_cosine, rotary_sine, rope_style=config.rope_style,
    )
    attended = _attention_precanonicalized(
        query,
        key,
        value,
        attention_mask,
        segment_ids,
        implementation=implementation,
        scale=config.head_dim**-0.5,
        dropout_rate=0.0,
        deterministic=True,
        mesh=mesh,
        block_q=block_q,
        block_k=block_k,
        attention_metadata=attention_metadata,
        attention_profile=attention_profile,
        attention_options=attention_options,
        dense_attention=_dense_attention_layer(
            attention_options, layer_index, config.num_hidden_layers, attention_step,
        ),
    )
    attended = attended.reshape(*hidden_states.shape[:2], config.query_width)
    hidden_states = hidden_states + tiled_output_projection(
        attended, layer_params["attention"]["o_proj"], mesh=mesh, chunk_size=chunk_size,
    )
    hidden_states = hidden_states + tiled_norm_mlp(
        hidden_states,
        layer_params["post_attention_layernorm"]["scale"],
        layer_params["mlp"],
        mesh=mesh,
        rms_norm_eps=config.rms_norm_eps,
        chunk_size=mlp_chunk_size or hidden_states.shape[1] // int(mesh.shape["sp"]),
    )
    return jax.lax.with_sharding_constraint(
        hidden_states, NamedSharding(mesh, PartitionSpec("dp", "sp", "tp")),
    )


def _grouped_layer_scan(scan_layer, hidden_states: Array, scan_inputs, layers_per_group: int) -> Array:
    """Checkpoint group boundaries, recovering inner layer inputs on demand.

    For L layers in groups of G, reverse mode keeps approximately L/G outer
    boundaries plus G inner boundaries, instead of all L layer inputs. The
    caller checkpoints each layer body as well, so attention/MLP residuals
    are not retained across the group. No host offload is involved.
    """

    layers = jax.tree.leaves(scan_inputs)[0].shape[0]
    if layers % layers_per_group:
        raise ValueError("layers_per_group must divide num_hidden_layers")
    grouped_inputs = jax.tree.map(
        lambda x: x.reshape(layers // layers_per_group, layers_per_group, *x.shape[1:]),
        scan_inputs,
    )

    def scan_group(hidden, group_inputs):
        hidden, _ = jax.lax.scan(scan_layer, hidden, group_inputs)
        return hidden, None

    scan_group = jax.checkpoint(
        scan_group, policy=jax.checkpoint_policies.nothing_saveable, prevent_cse=False,
    )
    hidden_states, _ = jax.lax.scan(scan_group, hidden_states, grouped_inputs)
    return hidden_states


def decoder_forward(
    params: Params,
    config: ModelConfig,
    input_ids: Array,
    attention_mask: Array | None = None,
    position_ids: Array | None = None,
    segment_ids: Array | None = None,
    implementation: AttentionImplementation = "efficient",
    mesh: Mesh | None = None,
    *,
    deterministic: bool = True,
    dropout_rng: Array | None = None,
    lora_params: Params | None = None,
    remat_policy: RematPolicy = "none",
    compute_dtype: jnp.dtype = jnp.bfloat16,
    block_q: int = 128,
    block_k: int = 128,
    mlp_chunk_size: int = 0,
    gather_kv_projection_weights: bool = False,
    scan_layers: bool = False,
    projection_chunk_size: int = 0,
    remat_group_size: int = 1,
    attention_options: AttentionOptions | None = None,
    attention_step: int | Array = 0,
) -> Array:
    """Run the decoder and return final normalized hidden states.

    ``mlp_chunk_size`` is a static sequence-tile size. Zero keeps the ordinary
    full-sequence MLP; a positive value enables ``lax.scan`` tiling with a
    nothing-saveable rematerialized tile body.

    ``scan_layers`` only changes the internal compute representation. The
    checkpoint-facing layer tuple is stacked on entry and autodiff maps its
    gradients back to that unchanged tuple tree.

    ``projection_chunk_size`` tiles QKV/output projections and final norm,
    retaining TP feature shards between layers. Its MLP independently uses
    ``mlp_chunk_size``, with zero covering the full SP-local sequence.
    This projection layout always gathers K/V weights; the
    ``gather_kv_projection_weights`` switch applies to ordinary projections.
    ``remat_group_size`` independently checkpoints groups of scanned layers.
    Their defaults preserve ordinary execution.
    """

    if attention_options is not None:
        attention_options.validate_layers(config.num_hidden_layers)
        if attention_options.method == "dense":
            attention_options = None
    if remat_policy not in ("none", "nothing_saveable"):
        raise ValueError(f"unsupported remat policy: {remat_policy!r}")
    if isinstance(mlp_chunk_size, bool):
        raise TypeError("mlp_chunk_size must be a static non-negative integer")
    try:
        mlp_chunk_size = operator.index(mlp_chunk_size)
    except TypeError as error:
        raise TypeError("mlp_chunk_size must be a static non-negative integer") from error
    if mlp_chunk_size < 0:
        raise ValueError("mlp_chunk_size must be non-negative")
    if not isinstance(gather_kv_projection_weights, bool):
        raise TypeError("gather_kv_projection_weights must be a static bool")
    if not isinstance(scan_layers, bool):
        raise TypeError("scan_layers must be a static bool")
    for name, value, minimum in (
        ("projection_chunk_size", projection_chunk_size, 0), ("remat_group_size", remat_group_size, 1),
    ):
        if isinstance(value, bool):
            raise TypeError(f"{name} must be a static integer")
        try:
            integer = operator.index(value)
        except TypeError as error:
            raise TypeError(f"{name} must be a static integer") from error
        if integer < minimum:
            raise ValueError(f"{name} must be at least {minimum}")
    projection_chunk_size = operator.index(projection_chunk_size)
    remat_group_size = operator.index(remat_group_size)
    if remat_group_size > 1:
        if not scan_layers or remat_policy != "nothing_saveable":
            raise ValueError("remat_group_size>1 requires scan_layers=True and remat_policy=nothing_saveable")
        if config.num_hidden_layers % remat_group_size:
            raise ValueError("remat_group_size must divide num_hidden_layers")
    tiled_execution = projection_chunk_size > 0
    if tiled_execution:
        if mesh is None:
            raise ValueError("tiled execution requires a mesh")
        if any(int(mesh.shape.get(axis, 1)) != 1 for axis in ("dp", "fsdp", "ep")):
            raise ValueError("tiled execution requires DP=FSDP=EP=1")
        if not {"dp", "fsdp", "ep", "tp", "sp"}.issubset(mesh.axis_names):
            raise ValueError("tiled execution requires the dp,fsdp,ep,tp,sp mesh axes")
        if lora_params is not None:
            raise ValueError("tiled execution requires full-parameter training")
        if not deterministic or config.attention_dropout != 0:
            raise ValueError("tiled execution requires deterministic attention with dropout=0")
        if jnp.dtype(compute_dtype) != jnp.dtype(jnp.bfloat16):
            raise ValueError("tiled execution requires bfloat16 compute")
        if any(leaf.dtype != jnp.bfloat16 for leaf in jax.tree.leaves(params)):
            raise ValueError("tiled execution requires bfloat16 parameters")
        # Vanilla is retained as a small CPU numerical reference.
        if implementation not in ("efficient", "vanilla"):
            raise ValueError("tiled execution requires efficient attention (or vanilla for reference tests)")
    if gather_kv_projection_weights:
        if mesh is None:
            raise ValueError("gather_kv_projection_weights requires a mesh")
        if "fsdp" not in mesh.axis_names:
            raise ValueError("gather_kv_projection_weights requires an 'fsdp' mesh axis")
    if (
        mesh is not None
        and int(mesh.shape.get("sp", 1)) > 1
        and mlp_chunk_size > 0
        and not tiled_execution
    ):
        raise ValueError(
            "mlp_chunk_size must be 0 when SP>1 because tiled MLP currently "
            "scans global rather than shard-local sequence chunks"
        )
    input_ids = jnp.asarray(input_ids, dtype=jnp.int32)
    if input_ids.ndim != 2:
        raise ValueError(f"input_ids must have shape [batch, sequence], got {input_ids.shape}")
    batch, sequence = input_ids.shape
    if sequence == 0:
        raise ValueError("input_ids sequence length must be positive")
    if sequence > config.max_position_embeddings:
        raise ValueError(
            f"sequence length {sequence} exceeds max_position_embeddings={config.max_position_embeddings}"
        )

    if attention_mask is None:
        attention_mask = jnp.ones((batch, sequence), dtype=jnp.bool_)
    else:
        attention_mask = jnp.asarray(attention_mask) == 1
    if attention_mask.shape != input_ids.shape:
        raise ValueError(f"attention_mask must have shape {input_ids.shape}, got {attention_mask.shape}")
    if segment_ids is not None and jnp.shape(segment_ids) != input_ids.shape:
        raise ValueError(f"segment_ids must have shape {input_ids.shape}, got {jnp.shape(segment_ids)}")
    canonical_segment_ids = canonicalize_segment_ids(attention_mask, segment_ids)
    if implementation == "efficient" and (attention_options is None or attention_options.needs_dense_backend):
        if mesh is None:
            raise ValueError("efficient decoder attention requires the training Mesh")
        attention_profile = resolve_attention_profile(
            (batch, sequence, config.num_attention_heads, config.head_dim),
            (batch, sequence, config.num_key_value_heads, config.head_dim),
            mesh,
            # Long sequences otherwise replicate each LSE/delta across 128
            # lanes in HBM. Compact row statistics preserve the same arithmetic
            # and reduce the measured v4 TP8/SP4 forward+VJP time at 128K/512K.
            **({"compact_statistics": True} if tiled_execution or sequence >= 131_072 else {}),
        )
        attention_metadata = prepare_attention_metadata(
            canonical_segment_ids,
            mesh=mesh,
            profile=attention_profile,
        )
    else:
        attention_profile = None
        attention_metadata = None
    if position_ids is None:
        position_ids = _make_position_ids_precanonicalized(attention_mask, canonical_segment_ids)
    else:
        position_ids = jnp.asarray(position_ids, dtype=jnp.int32)
        if position_ids.shape != input_ids.shape:
            raise ValueError(f"position_ids must have shape {input_ids.shape}, got {position_ids.shape}")

    compute_dtype = jnp.dtype(compute_dtype)
    # Compute rotary values from per-token positions, including reset and
    # discontinuous logical positions used by packing and EndPrompt.
    rotary_cosine, rotary_sine = rotary_cos_sin(
        position_ids,
        inverse_frequencies(config),
        compute_dtype,
    )
    hidden_states = _embedding_lookup(
        params["embed_tokens"]["embedding"],
        input_ids,
        config.pad_token_id,
        compute_dtype,
    )
    if tiled_execution:
        if batch != 1:
            raise ValueError("tiled execution requires batch size 1")
        hidden_states = jax.lax.with_sharding_constraint(
            hidden_states, NamedSharding(mesh, PartitionSpec("dp", "sp", "tp")),
        )
    parameter_layers = tuple(params["layers"])
    if len(parameter_layers) != config.num_hidden_layers:
        raise ValueError(
            f"parameter tree has {len(parameter_layers)} layers, expected {config.num_hidden_layers}"
        )
    layer_rngs = (
        None
        if dropout_rng is None
        else jax.random.split(dropout_rng, config.num_hidden_layers)
    )
    if lora_params is None:
        layer_adapters = None
    else:
        layer_adapters = tuple(lora_params["layers"])
        if len(layer_adapters) != config.num_hidden_layers:
            raise ValueError(
                f"LoRA parameter tree has {len(layer_adapters)} layers, expected {config.num_hidden_layers}"
            )

    layer_function = partial(
        _decoder_layer,
        config=config,
        implementation=implementation,
        mesh=mesh,
        deterministic=deterministic,
        compute_dtype=compute_dtype,
        block_q=block_q,
        block_k=block_k,
        attention_profile=attention_profile,
        mlp_chunk_size=mlp_chunk_size,
        gather_kv_projection_weights=gather_kv_projection_weights,
        attention_options=attention_options,
        attention_step=attention_step,
    )
    if tiled_execution:
        layer_function = partial(
            _tiled_decoder_layer,
            config=config,
            implementation=implementation,
            mesh=mesh,
            block_q=block_q,
            block_k=block_k,
            attention_profile=attention_profile,
            chunk_size=projection_chunk_size,
            mlp_chunk_size=mlp_chunk_size,
            attention_options=attention_options,
            attention_step=attention_step,
        )
    if remat_policy == "nothing_saveable":
        layer_function = jax.checkpoint(
            layer_function,
            policy=jax.checkpoint_policies.nothing_saveable,
            # The unrolled path needs CSE prevention to preserve explicit
            # layer rematerialization. A scan already provides that structural
            # boundary, and disabling the expensive CSE barrier is the JAX-
            # recommended checkpoint form inside a scan.
            prevent_cse=not scan_layers,
        )

    if scan_layers:
        scan_inputs: dict[str, Any] = {
            "params": _stack_layer_trees(
                parameter_layers,
                expected_layers=config.num_hidden_layers,
                name="parameter",
            )
        }
        if layer_adapters is not None:
            scan_inputs["adapters"] = _stack_layer_trees(
                layer_adapters,
                expected_layers=config.num_hidden_layers,
                name="LoRA parameter",
            )
        if layer_rngs is not None:
            scan_inputs["rng"] = layer_rngs
        if attention_options is not None:
            scan_inputs["layer_index"] = jnp.arange(config.num_hidden_layers, dtype=jnp.int32)

        def scan_layer(hidden: Array, layer_inputs: dict[str, Any]) -> tuple[Array, None]:
            hidden = layer_function(
                layer_inputs["params"],
                layer_inputs.get("adapters"),
                hidden,
                attention_mask,
                rotary_cosine,
                rotary_sine,
                canonical_segment_ids,
                attention_metadata,
                layer_inputs.get("rng"),
                layer_inputs.get("layer_index", 0),
            )
            return hidden, None

        if remat_group_size > 1:
            hidden_states = _grouped_layer_scan(
                scan_layer, hidden_states, scan_inputs, remat_group_size,
            )
        else:
            hidden_states, _ = jax.lax.scan(scan_layer, hidden_states, scan_inputs)
    else:
        adapters = (
            (None,) * config.num_hidden_layers
            if layer_adapters is None
            else layer_adapters
        )
        rngs = (
            (None,) * config.num_hidden_layers
            if layer_rngs is None
            else tuple(layer_rngs)
        )
        for layer_index, (layer_params, layer_adapter, layer_rng) in enumerate(zip(
            parameter_layers,
            adapters,
            rngs,
            strict=True,
        )):
            hidden_states = layer_function(
                layer_params,
                layer_adapter,
                hidden_states,
                attention_mask,
                rotary_cosine,
                rotary_sine,
                canonical_segment_ids,
                attention_metadata,
                layer_rng,
                layer_index,
            )

    if tiled_execution:
        return tiled_final_norm(
            hidden_states, params["norm"]["scale"], mesh=mesh,
            rms_norm_eps=config.rms_norm_eps, chunk_size=projection_chunk_size,
        )
    return rms_norm(hidden_states, params["norm"]["scale"], config.rms_norm_eps, compute_dtype)


def forward(
    params: Params,
    config: ModelConfig,
    input_ids: Array,
    attention_mask: Array | None = None,
    position_ids: Array | None = None,
    segment_ids: Array | None = None,
    implementation: AttentionImplementation = "efficient",
    mesh: Mesh | None = None,
    *,
    deterministic: bool = True,
    dropout_rng: Array | None = None,
    lora_params: Params | None = None,
    remat_policy: RematPolicy = "none",
    compute_dtype: jnp.dtype = jnp.bfloat16,
    block_q: int = 128,
    block_k: int = 128,
    mlp_chunk_size: int = 0,
    gather_kv_projection_weights: bool = False,
    scan_layers: bool = False,
    projection_chunk_size: int = 0,
    remat_group_size: int = 1,
    attention_options: AttentionOptions | None = None,
    attention_step: int | Array = 0,
) -> Array:
    """Return causal language-model logits in ``[batch, sequence, vocab]``."""

    hidden_states = decoder_forward(
        params,
        config,
        input_ids,
        attention_mask,
        position_ids,
        segment_ids,
        implementation,
        mesh,
        deterministic=deterministic,
        dropout_rng=dropout_rng,
        lora_params=lora_params,
        remat_policy=remat_policy,
        compute_dtype=compute_dtype,
        block_q=block_q,
        block_k=block_k,
        mlp_chunk_size=mlp_chunk_size,
        gather_kv_projection_weights=gather_kv_projection_weights,
        scan_layers=scan_layers,
        projection_chunk_size=projection_chunk_size,
        remat_group_size=remat_group_size,
        attention_options=attention_options,
        attention_step=attention_step,
    )
    if config.tie_word_embeddings:
        lm_head = {"kernel": params["embed_tokens"]["embedding"].T}
    else:
        lm_head = params["lm_head"]
    return linear(hidden_states, lm_head, jnp.dtype(compute_dtype))
