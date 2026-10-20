"""Training attention backends for grouped- and multi-query attention."""

from __future__ import annotations

from typing import Literal

import jax
import jax.numpy as jnp
from jax import shard_map
from jax.experimental.pallas.ops.tpu.splash_attention import (
    BlockSizes,
    CausalMask,
    MultiHeadMask,
    SegmentIds,
)
from jax.experimental.pallas.ops.tpu.splash_attention import (
    make_splash_mqa as make_splash_attention,
)
from jax.experimental.pallas.ops.tpu.splash_attention import (
    make_splash_mqa_single_device as make_splash_attention_single_device,
)
from jax.sharding import Mesh, NamedSharding, PartitionSpec

from ..kernels.efficient_attention import (
    AttentionKernelProfile,
    AttentionMetadata,
    efficient_attention,
    prepare_attention_metadata,
    resolve_attention_profile,
)
from .attention_options import AttentionOptions
from .config import validate_splash_block_size
from .rotary import canonicalize_segment_ids

Array = jax.Array
AttentionImplementation = Literal["splash", "vanilla", "efficient"]
_SPLASH_LANES = 128


def _validate_training_qkv(query: Array, key: Array, value: Array) -> None:
    if query.ndim != 4 or key.ndim != 4 or value.ndim != 4:
        raise ValueError("query, key, and value must have shape [batch, sequence, heads, dim]")
    if key.shape != value.shape:
        raise ValueError(f"key and value shapes differ: {key.shape} and {value.shape}")
    if query.shape[:2] != key.shape[:2]:
        raise ValueError("training attention requires matching Q and KV batch/sequence dimensions")
    if query.shape[-1] != key.shape[-1]:
        raise ValueError("query and key head dimensions must match")
    if query.shape[2] % key.shape[2]:
        raise ValueError("query head count must be divisible by key/value head count")


def normalize_segment_ids(attention_mask: Array, segment_ids: Array | None) -> Array:
    """Return leakage-safe one-based IDs for valid contiguous runs."""

    return canonicalize_segment_ids(attention_mask, segment_ids)


def make_causal_segment_mask(attention_mask: Array, segment_ids: Array | None = None) -> Array:
    """Build a leakage-safe ``[B, 1, S, S]`` packed causal mask."""

    valid = jnp.asarray(attention_mask, dtype=jnp.bool_)
    segments = normalize_segment_ids(valid, segment_ids)
    return _make_causal_segment_mask_precanonicalized(valid, segments)


def _make_causal_segment_mask_precanonicalized(
    attention_mask: Array,
    segment_ids: Array,
    attention_end: Array | None = None,
) -> Array:
    """Build the attention mask from canonical segment IDs."""

    valid = jnp.asarray(attention_mask, dtype=jnp.bool_)
    if valid.ndim != 2:
        raise ValueError(f"attention_mask must have shape [batch, sequence], got {valid.shape}")
    segments = jnp.asarray(segment_ids, dtype=jnp.int32)
    if segments.shape != valid.shape:
        raise ValueError(f"segment_ids must have shape {valid.shape}, got {segments.shape}")
    sequence = valid.shape[1]
    row = jnp.arange(sequence)[:, None]
    column = jnp.arange(sequence)[None, :]
    causal = column <= row
    if attention_end is not None:
        ends = jnp.asarray(attention_end, dtype=jnp.int32)
        if ends.shape != valid.shape:
            raise ValueError("attention_end must match attention_mask")
        causal = column[None, :, :] <= ends[:, :, None]
    same_segment = segments[:, :, None] == segments[:, None, :]
    non_padding_segment = segments[:, :, None] != 0
    valid_pairs = valid[:, :, None] & valid[:, None, :]
    return (
        (causal[None, None, :, :] if attention_end is None else causal[:, None, :, :])
        & same_segment[:, None, :, :]
        & non_padding_segment[:, None, :, :]
        & valid_pairs[:, None, :, :]
    )


def vanilla_attention(
    query: Array,
    key: Array,
    value: Array,
    allowed: Array,
    *,
    scale: float,
    dropout_rate: float = 0.0,
    deterministic: bool = True,
    dropout_rng: Array | None = None,
) -> Array:
    """Compute grouped-query attention with a float32 softmax."""

    _validate_training_qkv(query, key, value)

    batch, query_length, query_heads, head_dim = query.shape
    key_length, key_value_heads = key.shape[1:3]
    groups = query_heads // key_value_heads
    grouped_query = query.reshape(batch, query_length, key_value_heads, groups, head_dim)

    # Splash defines the numerical reference order: scale Q in its compute
    # dtype before the dot product, then promote logits for masking/softmax.
    # This is mathematically equivalent to post-scaling QK, but keeping the
    # same order also keeps BF16 rounding aligned with the TPU kernel.
    grouped_query = grouped_query * jnp.asarray(scale, dtype=grouped_query.dtype)
    scores = jnp.einsum("bqngd,bknd->bngqk", grouped_query, key, precision=None).astype(jnp.float32)

    allowed = jnp.asarray(allowed, dtype=jnp.bool_)
    if allowed.ndim == 3:
        allowed = allowed[:, None, :, :]
    expected = (batch, 1, query_length, key_length)
    if allowed.shape != expected:
        raise ValueError(f"allowed mask must have shape {expected}, got {allowed.shape}")
    grouped_allowed = allowed[:, :, None, :, :]
    minimum = jnp.finfo(scores.dtype).min
    probabilities = jax.nn.softmax(jnp.where(grouped_allowed, scores, minimum), axis=-1)
    # Avoid a uniform row for fully masked padding queries.
    probabilities = jnp.where(jnp.any(grouped_allowed, axis=-1, keepdims=True), probabilities, 0.0)
    probabilities = probabilities.astype(query.dtype)

    if dropout_rate and not deterministic:
        if dropout_rng is None:
            raise ValueError("dropout_rng is required when attention dropout is enabled")
        keep_probability = 1.0 - dropout_rate
        keep = jax.random.bernoulli(dropout_rng, keep_probability, probabilities.shape)
        probabilities = jnp.where(keep, probabilities / keep_probability, 0.0)

    output = jnp.einsum("bngqk,bknd->bqngd", probabilities, value, precision=None)
    return output.reshape(batch, query_length, query_heads, value.shape[-1])


def _splash_block_sizes(query_length: int, key_length: int, block_q: int, block_k: int):
    effective_q = validate_splash_block_size(query_length, block_q, "block_q")
    effective_k = validate_splash_block_size(key_length, block_k, "block_k")
    # Splash stores packed/causal mask metadata per outer Q/KV block, while
    # ``block_kv_compute`` controls the kernel working tile independently.
    # A compute tile no larger than 512 lets very long contexts use a larger
    # outer KV block to reduce SMEM metadata while staying within the kernel's
    # VMEM budget. Preserve established tiles up to 1024; above that, select
    # the largest supported sub-tile that exactly divides the outer block.
    effective_k_compute = effective_k
    if effective_k > 1024:
        effective_k_compute = next(tile for tile in (512, 256, 128) if effective_k % tile == 0)
    return BlockSizes(
        block_q=effective_q,
        block_kv_compute=effective_k_compute,
        block_kv=effective_k,
        block_q_dkv=effective_q,
        block_kv_dkv=effective_k,
        block_kv_dkv_compute=effective_k_compute,
        block_q_dq=effective_q,
        block_kv_dq=effective_k,
    )


def _active_mesh_axis(mesh: Mesh, names: tuple[str, ...]) -> str | tuple[str, ...] | None:
    active = tuple(name for name in names if name in mesh.axis_names and mesh.shape[name] > 1)
    if not active:
        return None
    return active[0] if len(active) == 1 else active


def splash_partition_specs(
    mesh: Mesh,
) -> tuple[PartitionSpec, PartitionSpec, PartitionSpec, PartitionSpec]:
    """Return Q/output, replicated-KV, query-segment, and KV-segment specs.

    Q and output retain the sequence shard. K/V tensors and their segment IDs
    are replicated over SP. Parameters remain replicated over SP.
    """

    batch_axis = _active_mesh_axis(mesh, ("dp", "fsdp"))
    tensor_axis = _active_mesh_axis(mesh, ("tp",))
    sequence_axis = _active_mesh_axis(mesh, ("sp",))
    query_spec = PartitionSpec(batch_axis, sequence_axis, tensor_axis, None)
    key_value_spec = PartitionSpec(batch_axis, None, None, None)
    query_segment_spec = PartitionSpec(batch_axis, sequence_axis)
    key_value_segment_spec = PartitionSpec(batch_axis, None)
    return query_spec, key_value_spec, query_segment_spec, key_value_segment_spec


def splash_attention(
    query: Array,
    key: Array,
    value: Array,
    segment_ids: Array,
    *,
    scale: float,
    block_q: int = 128,
    block_k: int = 128,
    mesh: Mesh | None = None,
) -> Array:
    """Run TPU Splash attention with packed segment IDs supplied to the kernel."""

    _validate_training_qkv(query, key, value)
    if query.shape[-1] % _SPLASH_LANES or value.shape[-1] % _SPLASH_LANES:
        raise ValueError(f"Splash Q/K and V head dimensions must be multiples of {_SPLASH_LANES}")

    segments = jnp.asarray(segment_ids, dtype=jnp.int32)
    if segments.shape != query.shape[:2]:
        raise ValueError(f"segment_ids must have shape {query.shape[:2]}, got {segments.shape}")

    def run_local(
        q: Array,
        k: Array,
        v: Array,
        local_query_segments: Array,
        local_key_value_segments: Array,
    ) -> Array:
        batch, sequence, query_heads, head_dim = q.shape
        key_value_heads = k.shape[2]
        groups = query_heads // key_value_heads
        # The single-device primitive consumes [query_heads, sequence, dim]
        # for Q and [sequence, dim] for each shared K/V head. The inner vmap
        # handles K/V head groups, and the outer vmap handles local batch rows.
        q = q.transpose(0, 2, 1, 3).reshape(batch, key_value_heads, groups, sequence, head_dim)
        k = k.transpose(0, 2, 1, 3)
        v = v.transpose(0, 2, 1, 3)
        head_mask = MultiHeadMask([CausalMask((sequence, sequence)) for _ in range(groups)])
        kernel = make_splash_attention_single_device(
            mask=head_mask,
            block_sizes=_splash_block_sizes(sequence, sequence, block_q, block_k),
        )
        over_kv_heads = jax.vmap(kernel, in_axes=(0, 0, 0, None))
        over_batch = jax.vmap(over_kv_heads, in_axes=(0, 0, 0, 0))
        packed_segments = SegmentIds(local_query_segments, local_key_value_segments)
        output = over_batch(q * jnp.asarray(scale, q.dtype), k, v, packed_segments)
        return output.reshape(batch, query_heads, sequence, v.shape[-1]).transpose(0, 2, 1, 3)

    if mesh is None:
        return run_local(query, key, value, segments, segments)

    (
        query_spec,
        key_value_spec,
        query_segment_spec,
        key_value_segment_spec,
    ) = splash_partition_specs(mesh)
    sequence_shards = int(mesh.shape.get("sp", 1))
    if query.shape[1] % sequence_shards:
        raise ValueError(f"Splash sequence length {query.shape[1]} must be divisible by SP={sequence_shards}")
    local_query_length = query.shape[1] // sequence_shards
    effective_q = validate_splash_block_size(query.shape[1], block_q, "block_q")
    if local_query_length % effective_q:
        raise ValueError(f"Splash block_q={effective_q} must divide the SP-local query length {local_query_length}")

    if sequence_shards == 1:
        distributed = shard_map(
            run_local,
            mesh=mesh,
            in_specs=(
                query_spec,
                key_value_spec,
                key_value_spec,
                query_segment_spec,
                key_value_segment_spec,
            ),
            out_specs=query_spec,
            check_vma=False,
        )
        return distributed(query, key, value, segments, segments)

    _, sequence, query_heads, _ = query.shape
    key_value_heads = key.shape[2]
    groups = query_heads // key_value_heads
    tensor_axis = _active_mesh_axis(mesh, ("tp",))
    sequence_axis = _active_mesh_axis(mesh, ("sp",))
    head_mask = MultiHeadMask([CausalMask((sequence, sequence)) for _ in range(groups)])
    kernel = make_splash_attention(
        mask=head_mask,
        block_sizes=_splash_block_sizes(sequence, sequence, block_q, block_k),
        head_shards=int(mesh.shape.get("tp", 1)),
        q_seq_shards=sequence_shards,
    )
    kernel_spec = kernel.manual_sharding_spec(NamedSharding(mesh, PartitionSpec(tensor_axis, sequence_axis)))

    def run_query_sharded(
        local_kernel,
        q: Array,
        k: Array,
        v: Array,
        local_query_segments: Array,
        local_key_value_segments: Array,
    ) -> Array:
        local_batch, local_sequence, local_query_heads, local_head_dim = q.shape
        local_key_value_heads = k.shape[2]
        local_groups = local_query_heads // local_key_value_heads
        q = q.transpose(0, 2, 1, 3).reshape(
            local_batch,
            local_key_value_heads,
            local_groups,
            local_sequence,
            local_head_dim,
        )
        k = k.transpose(0, 2, 1, 3)
        v = v.transpose(0, 2, 1, 3)
        over_kv_heads = jax.vmap(local_kernel, in_axes=(0, 0, 0, None))
        over_batch = jax.vmap(over_kv_heads, in_axes=(0, 0, 0, 0))
        packed_segments = SegmentIds(local_query_segments, local_key_value_segments)
        output = over_batch(
            q * jnp.asarray(scale, q.dtype),
            k,
            v,
            packed_segments,
        )
        return output.reshape(
            local_batch,
            local_query_heads,
            local_sequence,
            v.shape[-1],
        ).transpose(0, 2, 1, 3)

    distributed = shard_map(
        run_query_sharded,
        mesh=mesh,
        in_specs=(
            kernel_spec,
            query_spec,
            key_value_spec,
            key_value_spec,
            query_segment_spec,
            key_value_segment_spec,
        ),
        out_specs=query_spec,
        check_vma=False,
    )
    return distributed(kernel, query, key, value, segments, segments)


def _attention_precanonicalized(
    query: Array,
    key: Array,
    value: Array,
    attention_mask: Array,
    segment_ids: Array,
    *,
    implementation: AttentionImplementation = "efficient",
    scale: float | None = None,
    dropout_rate: float = 0.0,
    deterministic: bool = True,
    dropout_rng: Array | None = None,
    block_q: int = 128,
    block_k: int = 128,
    mesh: Mesh | None = None,
    attention_metadata: AttentionMetadata | None = None,
    attention_profile: AttentionKernelProfile | None = None,
    attention_options: AttentionOptions | None = None,
    dense_attention: bool | Array = False,
    attention_end: Array | None = None,
) -> Array:
    """Validate tensors and dispatch using canonical packed segment IDs."""

    if implementation not in ("splash", "vanilla", "efficient"):
        raise ValueError(f"unknown attention implementation: {implementation!r}")
    _validate_training_qkv(query, key, value)
    valid = jnp.asarray(attention_mask, dtype=jnp.bool_)
    if valid.shape != query.shape[:2]:
        raise ValueError(f"attention_mask must have shape {query.shape[:2]}, got {valid.shape}")
    segments = jnp.asarray(segment_ids, dtype=jnp.int32)
    if segments.shape != query.shape[:2]:
        raise ValueError(f"segment_ids must have shape {query.shape[:2]}, got {segments.shape}")
    scale = query.shape[-1] ** -0.5 if scale is None else scale

    if attention_end is not None:
        if jnp.shape(attention_end) != valid.shape:
            raise ValueError("attention_end must match attention_mask")
        if implementation == "splash" or (attention_options is not None and attention_options.method != "dense"):
            raise ValueError("hybrid attention requires the vanilla or efficient dense backend")

    if attention_options is not None and attention_options.method != "dense":
        if key.shape[2] != 1:
            raise ValueError("MoBA requires MQA (num_key_value_heads=1)")
        if dropout_rate and not deterministic:
            raise ValueError("MoBA requires zero attention dropout")
        from .approximate_attention import moba_attention

        def approximate(_):
            return moba_attention(query, key, value, segments, options=attention_options, scale=scale, mesh=mesh)

        if isinstance(dense_attention, bool) and not dense_attention:
            return approximate(None)

        def dense(_):
            return _attention_precanonicalized(
                query, key, value, valid, segments, implementation=implementation, scale=scale,
                dropout_rate=dropout_rate, deterministic=deterministic, dropout_rng=dropout_rng,
                block_q=block_q, block_k=block_k, mesh=mesh,
                attention_metadata=attention_metadata, attention_profile=attention_profile,
            )

        if isinstance(dense_attention, bool):
            return dense(None)
        return jax.lax.cond(dense_attention, dense, approximate, None)

    if implementation == "efficient":
        if dropout_rate and not deterministic:
            raise ValueError("Efficient attention requires dropout_rate=0")
        if mesh is None:
            raise ValueError("Efficient attention requires the training Mesh")
        profile = (
            resolve_attention_profile(
                query.shape, key.shape, mesh,
                **({"has_attention_end": True} if attention_end is not None else {}),
            )
            if attention_profile is None
            else attention_profile
        )
        metadata = (
            prepare_attention_metadata(
                segments, mesh=mesh, profile=profile,
                **({"attention_end": attention_end} if attention_end is not None else {}),
            )
            if attention_metadata is None
            else attention_metadata
        )
        return efficient_attention(
            query,
            key,
            value,
            metadata,
            mesh=mesh,
            profile=profile,
            scale=scale,
            causal=True,
        )

    if implementation == "splash":
        if dropout_rate and not deterministic:
            raise ValueError("Splash Attention requires dropout_rate=0")
        if mesh is None and jax.device_count() > 1:
            raise ValueError("distributed Splash Attention requires a Mesh")
        if mesh is not None and key.shape[2] != 1:
            raise ValueError(
                f"unsupported distributed Splash K/V head count: {key.shape[2]}"
            )
        output = splash_attention(
            query,
            key,
            value,
            segments,
            scale=scale,
            block_q=block_q,
            block_k=block_k,
            mesh=mesh,
        )
        return output * valid[:, :, None, None].astype(output.dtype)

    allowed = _make_causal_segment_mask_precanonicalized(valid, segments, attention_end)
    return vanilla_attention(
        query,
        key,
        value,
        allowed,
        scale=scale,
        dropout_rate=dropout_rate,
        deterministic=deterministic,
        dropout_rng=dropout_rng,
    )


def attention(
    query: Array,
    key: Array,
    value: Array,
    attention_mask: Array,
    segment_ids: Array | None = None,
    *,
    implementation: AttentionImplementation = "efficient",
    scale: float | None = None,
    dropout_rate: float = 0.0,
    deterministic: bool = True,
    dropout_rng: Array | None = None,
    block_q: int = 128,
    block_k: int = 128,
    mesh: Mesh | None = None,
    attention_profile: AttentionKernelProfile | None = None,
    attention_options: AttentionOptions | None = None,
    attention_end: Array | None = None,
) -> Array:
    """Canonicalize packed segments and dispatch to the selected backend."""

    valid = jnp.asarray(attention_mask, dtype=jnp.bool_)
    segments = normalize_segment_ids(valid, segment_ids)
    return _attention_precanonicalized(
        query,
        key,
        value,
        valid,
        segments,
        implementation=implementation,
        scale=scale,
        dropout_rate=dropout_rate,
        deterministic=deterministic,
        dropout_rng=dropout_rng,
        block_q=block_q,
        block_k=block_k,
        mesh=mesh,
        attention_profile=attention_profile,
        attention_options=attention_options,
        **({"attention_end": attention_end} if attention_end is not None else {}),
    )
