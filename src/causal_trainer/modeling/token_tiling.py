"""Shard-local token tiles for decoder projections.

The public arrays retain their global token order.  Only a token tile is
gathered over TP; the sequence loop runs *inside* ``shard_map`` so SP never
gathers the global sequence merely to reshape it into tiles.
"""

from __future__ import annotations

import operator
from collections.abc import Callable, Mapping
from typing import Any

import jax
import jax.numpy as jnp
from jax import lax, shard_map
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P

Array = jax.Array
Params = Mapping[str, Any]
_HIDDEN_SPEC = P(("dp", "fsdp"), "sp", "tp")
_COLUMN_SPEC = P("fsdp", "tp")
_ROW_SPEC = P("tp", "fsdp")


def _validate(hidden: Array, mesh: Mesh, chunk_size: int) -> int:
    if not isinstance(mesh, Mesh):
        raise TypeError("token tiles require a Mesh")
    if set(mesh.axis_names) != {"dp", "fsdp", "ep", "tp", "sp"}:
        raise ValueError("token tiles require dp/fsdp/ep/tp/sp mesh axes")
    if int(mesh.shape["fsdp"]) != 1 or int(mesh.shape["ep"]) != 1:
        raise ValueError("token tiles require fsdp=ep=1")
    if isinstance(chunk_size, bool):
        raise TypeError("chunk_size must be a static positive integer")
    chunk_size = operator.index(chunk_size)
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if hidden.ndim != 3 or any(dimension <= 0 for dimension in hidden.shape):
        raise ValueError("hidden states must have non-empty shape [batch, sequence, hidden]")
    if hidden.dtype != jnp.bfloat16:
        raise TypeError("token tiles require BF16 hidden states")
    for dimension, divisor, name in (
        (hidden.shape[0], int(mesh.shape["dp"]), "batch/DP"),
        (hidden.shape[1], int(mesh.shape["sp"]), "sequence/SP"),
        (hidden.shape[2], int(mesh.shape["tp"]), "hidden/TP"),
    ):
        if dimension % divisor:
            raise ValueError(f"token tiles require divisible {name} dimensions")
    return chunk_size


def _norm(tile: Array, scale: Array, eps: float) -> Array:
    work = tile.astype(jnp.float32)
    normalized = work * lax.rsqrt(jnp.mean(jnp.square(work), axis=-1, keepdims=True) + eps)
    return normalized.astype(tile.dtype) * scale.astype(tile.dtype)


def _gather_hidden(tile: Array) -> Array:
    return lax.all_gather(tile, "tp", axis=2, tiled=True)


def _hidden_slice(tile: Array, width: int) -> Array:
    return lax.dynamic_slice_in_dim(tile, lax.axis_index("tp") * width, width, axis=2)


def _scan_tiles(hidden: Array, project: Callable, chunk_size: int, *parameters):
    """Scan local sequence tiles, saving compact inputs rather than MLP slabs."""

    batch, sequence, width = hidden.shape
    tile_size = min(chunk_size, sequence)
    padding = (-sequence) % tile_size
    tiled = jnp.pad(hidden, ((0, 0), (0, padding), (0, 0)))
    tiled = tiled.reshape(batch, -1, tile_size, width).transpose(1, 0, 2, 3)
    # The transpose of scan accumulates gradients of broadcast parameters in
    # their input dtype. Promote only these small per-layer shards so hundreds
    # of token tiles do not repeatedly round parameter gradients to BF16.
    # Every tile casts weights back before compute; the outer cast returns
    # parameter cotangents in the original checkpoint dtype.
    parameters = jax.tree.map(lambda value: value.astype(jnp.float32), parameters)

    def compute(tile, *promoted_parameters):
        compute_parameters = jax.tree.map(lambda value: value.astype(hidden.dtype), promoted_parameters)
        return project(tile, *compute_parameters)

    rematerialized = jax.checkpoint(
        compute,
        policy=jax.checkpoint_policies.nothing_saveable,
        prevent_cse=False,
    )

    def body(carry, tile):
        return carry, rematerialized(tile, *parameters)

    _, outputs = lax.scan(body, None, tiled)

    def restore(value):
        # Projection outputs can have a different feature width from inputs.
        value = value.transpose(1, 0, 2, 3).reshape(batch, sequence + padding, value.shape[-1])
        return value[:, :sequence, :]

    return jax.tree.map(restore, outputs)


def tiled_norm_mlp(
    hidden_states: Array,
    norm_scale: Array,
    mlp_params: Params,
    *,
    mesh: Mesh,
    rms_norm_eps: float,
    chunk_size: int = 4096,
) -> Array:
    """Return the normalized MLP projection with TP-sharded hidden output."""

    chunk_size = _validate(hidden_states, mesh, chunk_size)
    if mlp_params["gate_proj"]["kernel"].shape[1] % int(mesh.shape["tp"]):
        raise ValueError("MLP intermediate size must be divisible by TP")

    def local(hidden, scale, gate_kernel, up_kernel, down_kernel):
        def project(tile, scale, gate_kernel, up_kernel, down_kernel):
            normed = _norm(_gather_hidden(tile), scale, rms_norm_eps)
            gate = jax.nn.silu(jnp.einsum("bsi,io->bso", normed, gate_kernel))
            up = jnp.einsum("bsi,io->bso", normed, up_kernel)
            partial = jnp.einsum("bsi,io->bso", gate * up, down_kernel, preferred_element_type=jnp.float32)
            return lax.psum_scatter(partial, "tp", scatter_dimension=2, tiled=True).astype(tile.dtype)

        return _scan_tiles(hidden, project, chunk_size, scale, gate_kernel, up_kernel, down_kernel)

    return shard_map(
        local,
        mesh=mesh,
        in_specs=(_HIDDEN_SPEC, P(), _COLUMN_SPEC, _COLUMN_SPEC, _ROW_SPEC),
        out_specs=_HIDDEN_SPEC,
        check_vma=False,
    )(
        hidden_states,
        norm_scale,
        mlp_params["gate_proj"]["kernel"],
        mlp_params["up_proj"]["kernel"],
        mlp_params["down_proj"]["kernel"],
    )


def tiled_norm_qkv(
    hidden_states: Array,
    norm_scale: Array,
    attention_params: Params,
    *,
    mesh: Mesh,
    rms_norm_eps: float,
    num_attention_heads: int,
    num_key_value_heads: int = 1,
    head_dim: int = 128,
    chunk_size: int = 4096,
) -> tuple[Array, Array, Array]:
    """Project SP-local Q/K/V; only Q heads are partitioned over TP.

    K/V remain sequence-sharded at this boundary.  Attention can gather the
    small shared MQA K/V sequence according to its own existing contract.
    """

    chunk_size = _validate(hidden_states, mesh, chunk_size)
    tp = int(mesh.shape["tp"])
    if num_key_value_heads != 1:
        raise ValueError("QKV tiles require one shared K/V head")
    if num_attention_heads % tp or head_dim % tp:
        raise ValueError("Q head count and K/V projection width must be divisible by TP")
    widths = (num_attention_heads * head_dim, num_key_value_heads * head_dim)
    if attention_params["q_proj"]["kernel"].shape != (hidden_states.shape[-1], widths[0]):
        raise ValueError("query projection shape does not match Q head dimensions")
    for name in ("k_proj", "v_proj"):
        if attention_params[name]["kernel"].shape != (hidden_states.shape[-1], widths[1]):
            raise ValueError("K/V projection shape does not match shared head dimensions")
    biases = tuple(
        attention_params[name].get("bias", jnp.zeros(width, dtype=hidden_states.dtype))
        for name, width in (("q_proj", widths[0]), ("k_proj", widths[1]), ("v_proj", widths[1]))
    )

    def local(hidden, scale, q_kernel, k_kernel, v_kernel, q_bias, k_bias, v_bias):
        # This gathers only the two small projection matrices, never hidden
        # states for a complete SP-local sequence. Autodiff retains the
        # ordinary TP-sharded parameter/optimizer contract.
        k_kernel = lax.all_gather(k_kernel, "tp", axis=1, tiled=True)
        v_kernel = lax.all_gather(v_kernel, "tp", axis=1, tiled=True)
        q_bias = lax.dynamic_slice_in_dim(
            q_bias, lax.axis_index("tp") * q_kernel.shape[1], q_kernel.shape[1], axis=0
        )

        def project(tile, scale, q_kernel, k_kernel, v_kernel, q_bias, k_bias, v_bias):
            normed = _norm(_gather_hidden(tile), scale, rms_norm_eps)
            return (
                jnp.einsum("bsi,io->bso", normed, q_kernel) + q_bias,
                jnp.einsum("bsi,io->bso", normed, k_kernel) + k_bias,
                jnp.einsum("bsi,io->bso", normed, v_kernel) + v_bias,
            )

        q, k, v = _scan_tiles(
            hidden, project, chunk_size, scale, q_kernel, k_kernel, v_kernel, q_bias, k_bias, v_bias
        )
        batch, sequence = hidden.shape[:2]
        return (
            q.reshape(batch, sequence, num_attention_heads // tp, head_dim),
            k.reshape(batch, sequence, num_key_value_heads, head_dim),
            v.reshape(batch, sequence, num_key_value_heads, head_dim),
        )

    return shard_map(
        local,
        mesh=mesh,
        in_specs=(_HIDDEN_SPEC, P(), _COLUMN_SPEC, _COLUMN_SPEC, _COLUMN_SPEC, P(), P(), P()),
        out_specs=(
            P(("dp", "fsdp"), "sp", "tp", None),
            P(("dp", "fsdp"), "sp", None, None),
            P(("dp", "fsdp"), "sp", None, None),
        ),
        check_vma=False,
    )(
        hidden_states,
        norm_scale,
        attention_params["q_proj"]["kernel"],
        attention_params["k_proj"]["kernel"],
        attention_params["v_proj"]["kernel"],
        *biases,
    )


def tiled_output_projection(
    attended: Array,
    o_proj_params: Params,
    *,
    mesh: Mesh,
    chunk_size: int = 4096,
) -> Array:
    """Project flattened TP-local attention heads with tile reduce-scatter."""

    chunk_size = _validate(attended, mesh, chunk_size)
    if "bias" in o_proj_params:
        raise ValueError("the tiled output projection is bias-free")

    def local(hidden, kernel):
        def project(tile, kernel):
            partial = jnp.einsum("bsi,io->bso", tile, kernel, preferred_element_type=jnp.float32)
            return lax.psum_scatter(partial, "tp", scatter_dimension=2, tiled=True).astype(tile.dtype)

        return _scan_tiles(hidden, project, chunk_size, kernel)

    return shard_map(
        local,
        mesh=mesh,
        in_specs=(_HIDDEN_SPEC, _ROW_SPEC),
        out_specs=_HIDDEN_SPEC,
        check_vma=False,
    )(attended, o_proj_params["kernel"])


def tiled_final_norm(
    hidden_states: Array,
    norm_scale: Array,
    *,
    mesh: Mesh,
    rms_norm_eps: float,
    chunk_size: int = 4096,
) -> Array:
    """Apply final RMSNorm without an SP-sequence-sized FP32 TP gather."""

    chunk_size = _validate(hidden_states, mesh, chunk_size)

    def local(hidden, scale):
        def project(tile, scale):
            return _hidden_slice(_norm(_gather_hidden(tile), scale, rms_norm_eps), tile.shape[-1])

        return _scan_tiles(hidden, project, chunk_size, scale)

    return shard_map(
        local,
        mesh=mesh,
        in_specs=(_HIDDEN_SPEC, P()),
        out_specs=_HIDDEN_SPEC,
        check_vma=False,
    )(hidden_states, norm_scale)


__all__ = ["tiled_final_norm", "tiled_norm_mlp", "tiled_norm_qkv", "tiled_output_projection"]
