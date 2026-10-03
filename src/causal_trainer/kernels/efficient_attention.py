"""Efficient training-only TPU attention kernel.

K/V use their shared-head representation. Query tiles own dQ work, K/V tiles
own dK/dV work, and the global custom VJP performs TP/SP dK/dV reduction.
"""

from __future__ import annotations

import dataclasses
import functools
import math
import operator
from collections.abc import Mapping, Sequence
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import lax, shard_map
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

Array = jax.Array

_NUM_LANES = 128
_NUM_SUBLANES = 8
_COMMON_PREFERRED_QUERY_ROWS = 2048
_LONG_PREFERRED_QUERY_ROWS = 1024
_SUPPORTED_HEAD_DIM = 128
_DOT_TRANSPOSE_RHS = (((1,), (1,)), ((), ()))


def _output_shape(
    shape: tuple[int, ...],
    dtype: jnp.dtype,
    like: Array,
) -> jax.ShapeDtypeStruct:
    """Preserve manual-axis variation for kernel calls inside ``shard_map``."""

    manual_axis_type = getattr(like, "manual_axis_type", None)
    if manual_axis_type is None:
        manual_axis_type = getattr(getattr(like, "aval", None), "manual_axis_type", None)
    if manual_axis_type is None:
        raise ValueError("kernel output inside shard_map is missing manual-axis metadata")
    return jax.ShapeDtypeStruct(shape, dtype, manual_axis_type=manual_axis_type)


@dataclasses.dataclass(frozen=True)
class AttentionKernelProfile:
    """Static lowering choices shared by forward, dQ, and dKV."""

    block_h: int
    block_q: int
    block_kv_outer: int
    block_kv_compute: int
    compact_statistics: bool = False


class AttentionMetadata(NamedTuple):
    """Per-batch packed-attention metadata reused by every decoder layer."""

    query_segment_ids: Array
    kv_segment_ids: Array
    q_block_segments: Array
    kv_block_segments: Array


def _shape_tuple(shape: Sequence[int], name: str) -> tuple[int, ...]:
    try:
        result = tuple(operator.index(dimension) for dimension in shape)
    except TypeError as error:
        raise TypeError(f"{name} must be a static integer shape") from error
    if any(dimension <= 0 for dimension in result):
        raise ValueError(f"{name} dimensions must be positive, got {result}")
    return result


def _mesh_sizes(mesh_shape: Mapping[str, int] | Mesh) -> Mapping[str, int]:
    sizes = mesh_shape.shape if isinstance(mesh_shape, Mesh) else mesh_shape
    if not isinstance(sizes, Mapping):
        raise TypeError("mesh_shape must be a Mesh or a mapping of named axis sizes")
    return sizes


def _mesh_size(mesh_shape: Mapping[str, int] | Mesh, axis: str) -> int:
    sizes = _mesh_sizes(mesh_shape)
    try:
        size = operator.index(sizes.get(axis, 1))
    except TypeError as error:
        raise TypeError(f"mesh axis {axis!r} must have a static integer size") from error
    if size <= 0:
        raise ValueError(f"mesh axis {axis!r} must have a positive size, got {size}")
    return size


def _validate_qkv_shapes(
    q_shape: Sequence[int],
    kv_shape: Sequence[int],
    mesh_shape: Mapping[str, int] | Mesh,
) -> tuple[tuple[int, ...], tuple[int, ...], int, int]:
    query_shape = _shape_tuple(q_shape, "q_shape")
    key_value_shape = _shape_tuple(kv_shape, "kv_shape")
    if len(query_shape) != 4 or len(key_value_shape) != 4:
        raise ValueError(
            "attention shapes must be [batch, sequence, heads, dim], got "
            f"q={query_shape}, kv={key_value_shape}"
        )
    if query_shape[:2] != key_value_shape[:2]:
        raise ValueError(
            "Q and KV batch/sequence dimensions must match, got "
            f"q={query_shape}, kv={key_value_shape}"
        )
    if query_shape[-1] != key_value_shape[-1]:
        raise ValueError(
            "Q and KV head dimensions must match, got "
            f"q={query_shape[-1]}, kv={key_value_shape[-1]}"
        )
    if query_shape[2] % key_value_shape[2]:
        raise ValueError(
            "query head count must be divisible by the KV head count, got "
            f"Hq={query_shape[2]}, Hkv={key_value_shape[2]}"
        )

    tp_size = _mesh_size(mesh_shape, "tp")
    sp_size = _mesh_size(mesh_shape, "sp")
    data_size = _mesh_size(mesh_shape, "dp") * _mesh_size(mesh_shape, "fsdp")
    if query_shape[0] % data_size:
        raise ValueError(
            f"global batch {query_shape[0]} must be divisible by dp*fsdp={data_size}"
        )
    if query_shape[2] % tp_size:
        raise ValueError(f"Hq={query_shape[2]} must be divisible by TP={tp_size}")
    if query_shape[1] % sp_size:
        raise ValueError(f"sequence={query_shape[1]} must be divisible by SP={sp_size}")
    return query_shape, key_value_shape, tp_size, sp_size


def _static_positive_int(value: int, name: str) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be a static positive integer")
    try:
        result = operator.index(value)
    except TypeError as error:
        raise TypeError(f"{name} must be a static positive integer") from error
    if result <= 0:
        raise ValueError(f"{name} must be positive, got {result}")
    return result


def validate_attention_profile(
    profile: AttentionKernelProfile,
    q_shape: Sequence[int],
    kv_shape: Sequence[int],
    mesh_shape: Mapping[str, int] | Mesh,
) -> None:
    """Validate mathematical divisibility and first-version TPU layout rules."""

    if not isinstance(profile, AttentionKernelProfile):
        raise TypeError(f"profile must be AttentionKernelProfile, got {type(profile).__name__}")
    if not isinstance(profile.compact_statistics, bool):
        raise TypeError("profile.compact_statistics must be a static bool")
    query_shape, _, tp_size, sp_size = _validate_qkv_shapes(q_shape, kv_shape, mesh_shape)
    block_h = _static_positive_int(profile.block_h, "block_h")
    block_q = _static_positive_int(profile.block_q, "block_q")
    block_kv_outer = _static_positive_int(profile.block_kv_outer, "block_kv_outer")
    block_kv_compute = _static_positive_int(profile.block_kv_compute, "block_kv_compute")
    local_heads = query_shape[2] // tp_size
    local_query = query_shape[1] // sp_size
    details = (
        f"q_shape={query_shape}, kv_shape={tuple(kv_shape)}, TP={tp_size}, SP={sp_size}, "
        f"local_heads={local_heads}, local_query={local_query}, profile={profile}"
    )

    # Full local-head coverage matches the optimized Mosaic layout.
    if block_h != local_heads:
        raise ValueError(f"the efficient kernel requires block_h=local_heads; {details}")
    if local_heads % block_h:
        raise ValueError(f"block_h must divide local_heads; {details}")
    if block_q % _NUM_LANES or local_query % block_q:
        raise ValueError(f"block_q must be a 128-aligned divisor of local_query; {details}")
    if block_kv_outer % _NUM_LANES or query_shape[1] % block_kv_outer:
        raise ValueError(f"block_kv_outer must be a 128-aligned divisor of sequence; {details}")
    if block_kv_compute % _NUM_LANES or query_shape[1] % block_kv_compute:
        raise ValueError(f"block_kv_compute must be a 128-aligned divisor of sequence; {details}")
    if block_kv_compute > block_kv_outer or block_kv_outer % block_kv_compute:
        raise ValueError(f"block_kv_compute must divide block_kv_outer; {details}")


def _nearest_query_tile(
    local_query: int,
    local_heads: int,
    preferred_query_rows: int,
) -> int:
    target = preferred_query_rows / local_heads
    candidates = tuple(
        block
        for block in range(_NUM_LANES, local_query + 1, _NUM_LANES)
        if local_query % block == 0
    )
    if not candidates:
        raise ValueError(
            "SP-local query length has no 128-aligned static divisor: "
            f"local_query={local_query}, local_heads={local_heads}"
        )
    return min(candidates, key=lambda block: (abs(block - target), block))


def _largest_aligned_divisor(length: int, preferred: int) -> int:
    candidates = tuple(
        block
        for block in range(_NUM_LANES, min(length, preferred) + 1, _NUM_LANES)
        if length % block == 0
    )
    if not candidates:
        raise ValueError(
            "length has no 128-aligned static divisor: "
            f"length={length}, preferred={preferred}"
        )
    return max(candidates)


def resolve_attention_profile(
    q_shape: Sequence[int],
    kv_shape: Sequence[int],
    mesh_shape: Mapping[str, int] | Mesh,
    *,
    compact_statistics: bool = False,
) -> AttentionKernelProfile:
    """Resolve a static kernel profile from global shapes and named mesh sizes."""

    query_shape, _, tp_size, sp_size = _validate_qkv_shapes(q_shape, kv_shape, mesh_shape)
    local_heads = query_shape[2] // tp_size
    local_query = query_shape[1] // sp_size
    sequence = query_shape[1]
    if sequence <= 16_384:
        preferred_query_rows = _COMMON_PREFERRED_QUERY_ROWS
        preferred_kv_outer, preferred_kv_compute = 512, 256
    elif sequence <= 262_144:
        preferred_query_rows = _LONG_PREFERRED_QUERY_ROWS
        preferred_kv_outer, preferred_kv_compute = 1024, 256
    else:
        preferred_query_rows = _LONG_PREFERRED_QUERY_ROWS
        preferred_kv_outer, preferred_kv_compute = 2048, 512
    block_q = _nearest_query_tile(local_query, local_heads, preferred_query_rows)
    block_kv_outer = _largest_aligned_divisor(sequence, preferred_kv_outer)
    block_kv_compute = _largest_aligned_divisor(block_kv_outer, preferred_kv_compute)
    profile = AttentionKernelProfile(
        block_h=local_heads,
        block_q=block_q,
        block_kv_outer=block_kv_outer,
        block_kv_compute=block_kv_compute,
        compact_statistics=compact_statistics,
    )
    validate_attention_profile(profile, q_shape, kv_shape, mesh_shape)
    return profile


def default_attention_scale(head_dim: int) -> float:
    """Return the default scale derived solely from the static head dimension."""

    dimension = _static_positive_int(head_dim, "head_dim")
    return dimension**-0.5


def _active_axis(mesh: Mesh, names: tuple[str, ...]) -> str | tuple[str, ...] | None:
    active = tuple(name for name in names if name in mesh.axis_names and int(mesh.shape[name]) > 1)
    if not active:
        return None
    return active[0] if len(active) == 1 else active


def attention_partition_specs(mesh: Mesh) -> dict[str, P]:
    """Return explicit query-sharded and replicated-KV partition specs."""

    batch_axis = _active_axis(mesh, ("dp", "fsdp"))
    sequence_axis = _active_axis(mesh, ("sp",))
    tensor_axis = _active_axis(mesh, ("tp",))
    return {
        "query": P(batch_axis, sequence_axis, tensor_axis, None),
        "stats": P(batch_axis, sequence_axis, tensor_axis),
        "kv": P(batch_axis, None, None, None),
        "query_segments": P(batch_axis, sequence_axis),
        "kv_segments": P(batch_axis, None),
        "query_summary": P(batch_axis, sequence_axis, None),
        "kv_summary": P(batch_axis, None, None),
    }


def _segment_block_summary(segment_ids: Array, block_size: int) -> Array:
    batch, sequence = segment_ids.shape
    blocks = segment_ids.reshape(batch, sequence // block_size, block_size)
    nonzero = blocks != 0
    maximum = jnp.max(jnp.where(nonzero, blocks, 0), axis=-1)
    minimum = jnp.min(
        jnp.where(nonzero, blocks, jnp.asarray(jnp.iinfo(jnp.int32).max, jnp.int32)),
        axis=-1,
    )
    minimum = jnp.where(maximum == 0, 0, minimum)
    return jnp.stack((minimum, maximum), axis=-1).astype(jnp.int32)


def prepare_attention_metadata(
    canonical_segment_ids: Array,
    *,
    mesh: Mesh,
    profile: AttentionKernelProfile,
) -> AttentionMetadata:
    """Build O(S/block) summaries once, immediately before the decoder loop."""

    if mesh is None:
        raise ValueError("attention metadata preparation requires the training mesh")
    if not isinstance(profile, AttentionKernelProfile):
        raise TypeError("profile must be an AttentionKernelProfile")
    segments = jnp.asarray(canonical_segment_ids, dtype=jnp.int32)
    if segments.ndim != 2:
        raise ValueError(f"canonical_segment_ids must be [batch, sequence], got {segments.shape}")
    if segments.shape[0] == 0 or segments.shape[1] == 0:
        raise ValueError("canonical_segment_ids must have non-empty batch and sequence dimensions")
    block_q = _static_positive_int(profile.block_q, "profile.block_q")
    block_kv_outer = _static_positive_int(profile.block_kv_outer, "profile.block_kv_outer")
    if segments.shape[1] % block_q or segments.shape[1] % block_kv_outer:
        raise ValueError(
            "metadata sequence must be divisible by profile blocks, got "
            f"shape={segments.shape}, profile={profile}"
        )

    q_summary = _segment_block_summary(segments, block_q)
    kv_summary = _segment_block_summary(segments, block_kv_outer)
    specs = attention_partition_specs(mesh)

    def constrain(value: Array, spec_name: str) -> Array:
        return lax.with_sharding_constraint(value, NamedSharding(mesh, specs[spec_name]))

    return AttentionMetadata(
        query_segment_ids=constrain(segments, "query_segments"),
        kv_segment_ids=constrain(segments, "kv_segments"),
        q_block_segments=constrain(q_summary, "query_summary"),
        kv_block_segments=constrain(kv_summary, "kv_summary"),
    )


def _summary_should_run(
    q_summary_ref,
    kv_summary_ref,
    offsets_ref,
    batch_index,
    q_block_index,
    kv_block_index,
    *,
    block_q: int,
    block_kv_outer: int,
    causal: bool,
):
    q_min = q_summary_ref[batch_index, q_block_index, 0]
    q_max = q_summary_ref[batch_index, q_block_index, 1]
    kv_min = kv_summary_ref[batch_index, kv_block_index, 0]
    kv_max = kv_summary_ref[batch_index, kv_block_index, 1]
    overlap = jnp.logical_and(
        jnp.logical_and(q_min != 0, kv_min != 0),
        jnp.maximum(q_min, kv_min) <= jnp.minimum(q_max, kv_max),
    )
    if not causal:
        return overlap
    q_global_end = offsets_ref[0] + (q_block_index + 1) * block_q - 1
    kv_global_start = kv_block_index * block_kv_outer
    return jnp.logical_and(overlap, kv_global_start <= q_global_end)


def _compute_tile_is_causally_reachable(
    offsets_ref,
    *,
    q_block_index,
    kv_block_index,
    compute_offset: int,
    block_q: int,
    block_kv_outer: int,
    causal: bool,
):
    if not causal:
        return jnp.asarray(True)
    q_global_end = offsets_ref[0] + (q_block_index + 1) * block_q - 1
    kv_global_start = kv_block_index * block_kv_outer + compute_offset
    return kv_global_start <= q_global_end


def _q_segment_rows(q_segment_ref, block_h: int) -> Array:
    q_lanes = q_segment_ref[0, :, :]
    return jnp.broadcast_to(
        q_lanes[:, None, :],
        (q_lanes.shape[0], block_h, _NUM_LANES),
    ).reshape((q_lanes.shape[0] * block_h, _NUM_LANES))


def _exact_mask(
    q_segment_ref,
    kv_segment_ref,
    offsets_ref,
    *,
    q_block_index,
    kv_block_index,
    compute_offset: int,
    block_h: int,
    block_q: int,
    block_kv_outer: int,
    block_kv_compute: int,
    causal: bool,
) -> Array:
    rows = block_h * block_q
    repeats = block_kv_compute // _NUM_LANES
    q_segments = jnp.tile(_q_segment_rows(q_segment_ref, block_h), (1, repeats))
    kv_segments = kv_segment_ref[0, :1, pl.ds(compute_offset, block_kv_compute)]
    valid = jnp.logical_and(q_segments != 0, q_segments == kv_segments)
    if causal:
        row_ids = lax.broadcasted_iota(jnp.int32, (rows, block_kv_compute), 0)
        q_global = offsets_ref[0] + q_block_index * block_q + row_ids // block_h
        column_ids = lax.broadcasted_iota(jnp.int32, (rows, block_kv_compute), 1)
        kv_global = kv_block_index * block_kv_outer + compute_offset + column_ids
        valid = jnp.logical_and(valid, kv_global <= q_global)
    return valid


def _broadcast_row_stat(stat: Array, width: int) -> Array:
    if width <= _NUM_LANES:
        return stat[:, :width]
    repeats, remainder = divmod(width, _NUM_LANES)
    if remainder:
        raise ValueError(f"row-stat width must be <=128 or a multiple of 128, got {width}")
    return jnp.tile(stat, (1, repeats))


def _load_row_stat(stat_ref, rows: int) -> Array:
    """Expand compact HBM statistics only after loading the current VMEM tile.

    Flattening batch, sequence, and heads makes the HBM window 128-aligned,
    including TP layouts with only one or two local query heads. Keeping the
    head count as the final HBM dimension would reintroduce TPU layout padding.
    """

    if len(stat_ref.shape) == 1:
        return lax.broadcast_in_dim(stat_ref[:], (rows, _NUM_LANES), (0,))
    return stat_ref[0, :, :, :].reshape((rows, _NUM_LANES))


def _statistics_kernel_input(stat: Array, *, compact: bool) -> Array:
    if compact:
        return stat.reshape((-1,))
    return lax.broadcast_in_dim(stat, (*stat.shape, _NUM_LANES), (0, 1, 2))


def _async_kv_copies(
    k_hbm_ref,
    v_hbm_ref,
    k_buffer_ref,
    v_buffer_ref,
    k_semaphore_ref,
    v_semaphore_ref,
    *,
    batch_index,
    kv_block_index,
    sequence: int,
    block_kv_outer: int,
    block_kv_compute: int,
):
    copies = []
    for compute_index in range(block_kv_outer // block_kv_compute):
        slot = compute_index % 2
        start = kv_block_index * block_kv_outer + compute_index * block_kv_compute
        # Flatten the shared K/V head dimension before Mosaic so DMA produces
        # the layout consumed by the TPU kernel.
        flat_start = batch_index * sequence + start
        k_copy = pltpu.make_async_copy(
            k_hbm_ref.at[pl.ds(flat_start, block_kv_compute), :],
            k_buffer_ref.at[slot],
            k_semaphore_ref.at[slot],
        )
        v_copy = pltpu.make_async_copy(
            v_hbm_ref.at[pl.ds(flat_start, block_kv_compute), :],
            v_buffer_ref.at[slot],
            v_semaphore_ref.at[slot],
        )
        copies.append((k_copy, v_copy))
    return copies


def _forward_compute_tile(
    q_ref,
    q_segment_ref,
    kv_segment_ref,
    offsets_ref,
    m_scratch_ref,
    l_scratch_ref,
    o_scratch_ref,
    k_buffer_ref,
    v_buffer_ref,
    *,
    slot: int,
    compute_offset: int,
    q_block_index,
    kv_block_index,
    block_h: int,
    block_q: int,
    block_kv_outer: int,
    block_kv_compute: int,
    causal: bool,
):
    rows = block_h * block_q
    head_dim = q_ref.shape[-1]
    query = q_ref[0, :, :, :].reshape((rows, head_dim))
    key = k_buffer_ref[slot, :, :]
    value = v_buffer_ref[slot, :, :]
    scores = lax.dot_general(
        query,
        key,
        _DOT_TRANSPOSE_RHS,
        preferred_element_type=jnp.float32,
    )
    valid = _exact_mask(
        q_segment_ref,
        kv_segment_ref,
        offsets_ref,
        q_block_index=q_block_index,
        kv_block_index=kv_block_index,
        compute_offset=compute_offset,
        block_h=block_h,
        block_q=block_q,
        block_kv_outer=block_kv_outer,
        block_kv_compute=block_kv_compute,
        causal=causal,
    )
    masked_scores = jnp.where(valid, scores, -jnp.inf)
    tile_max = jnp.tile(jnp.max(masked_scores, axis=1)[:, None], (1, _NUM_LANES))
    # Build the predicate at the 128-lane width consumed by the vector layout.
    row_has_lanes = tile_max != -jnp.inf
    m_previous = m_scratch_ref[:, :]
    l_previous = l_scratch_ref[:, :]
    m_next = jnp.where(row_has_lanes, jnp.maximum(m_previous, tile_max), m_previous)
    alpha = jnp.where(row_has_lanes, jnp.exp(m_previous - m_next), 1.0)
    expanded_max = jnp.tile(m_next, (1, block_kv_compute // _NUM_LANES))
    probabilities = jnp.where(valid, jnp.exp(scores - expanded_max), 0.0)
    tile_sum = jnp.tile(jnp.sum(probabilities, axis=1)[:, None], (1, _NUM_LANES))
    l_next = alpha * l_previous + tile_sum
    current = lax.dot(
        probabilities.astype(jnp.float32),
        value.astype(jnp.float32),
        preferred_element_type=jnp.float32,
    )
    o_scratch_ref[:, :] = (
        _broadcast_row_stat(alpha, head_dim) * o_scratch_ref[:, :] + current
    )
    m_scratch_ref[:, :] = m_next
    l_scratch_ref[:, :] = l_next


def _attention_forward_kernel(
    q_summary_ref,
    kv_summary_ref,
    offsets_ref,
    q_ref,
    k_hbm_ref,
    v_hbm_ref,
    q_segment_ref,
    kv_segment_ref,
    output_ref,
    lse_ref,
    m_scratch_ref,
    l_scratch_ref,
    o_scratch_ref,
    k_buffer_ref,
    v_buffer_ref,
    k_semaphore_ref,
    v_semaphore_ref,
    *,
    block_h: int,
    block_q: int,
    block_kv_outer: int,
    block_kv_compute: int,
    kv_blocks: int,
    sequence: int,
    causal: bool,
):
    batch_index = pl.program_id(0)
    q_block_index = pl.program_id(1)
    kv_block_index = pl.program_id(3)
    rows = block_h * block_q
    head_dim = q_ref.shape[-1]

    @pl.when(kv_block_index == 0)
    def initialize():
        m_scratch_ref[:, :] = jnp.full((rows, _NUM_LANES), -jnp.inf, jnp.float32)
        l_scratch_ref[:, :] = jnp.zeros((rows, _NUM_LANES), jnp.float32)
        o_scratch_ref[:, :] = jnp.zeros((rows, head_dim), jnp.float32)

    should_run = _summary_should_run(
        q_summary_ref,
        kv_summary_ref,
        offsets_ref,
        batch_index,
        q_block_index,
        kv_block_index,
        block_q=block_q,
        block_kv_outer=block_kv_outer,
        causal=causal,
    )

    @pl.when(should_run)
    def run():
        copies = _async_kv_copies(
            k_hbm_ref,
            v_hbm_ref,
            k_buffer_ref,
            v_buffer_ref,
            k_semaphore_ref,
            v_semaphore_ref,
            batch_index=batch_index,
            kv_block_index=kv_block_index,
            sequence=sequence,
            block_kv_outer=block_kv_outer,
            block_kv_compute=block_kv_compute,
        )
        copies[0][0].start()
        copies[0][1].start()
        for compute_index, (k_copy, v_copy) in enumerate(copies):
            k_copy.wait()
            v_copy.wait()
            if compute_index + 1 < len(copies):
                copies[compute_index + 1][0].start()
                copies[compute_index + 1][1].start()
            compute_offset = compute_index * block_kv_compute

            @pl.when(
                _compute_tile_is_causally_reachable(
                    offsets_ref,
                    q_block_index=q_block_index,
                    kv_block_index=kv_block_index,
                    compute_offset=compute_offset,
                    block_q=block_q,
                    block_kv_outer=block_kv_outer,
                    causal=causal,
                )
            )
            def compute(
                compute_index=compute_index,
                compute_offset=compute_offset,
            ):
                _forward_compute_tile(
                    q_ref,
                    q_segment_ref,
                    kv_segment_ref,
                    offsets_ref,
                    m_scratch_ref,
                    l_scratch_ref,
                    o_scratch_ref,
                    k_buffer_ref,
                    v_buffer_ref,
                    slot=compute_index % 2,
                    compute_offset=compute_offset,
                    q_block_index=q_block_index,
                    kv_block_index=kv_block_index,
                    block_h=block_h,
                    block_q=block_q,
                    block_kv_outer=block_kv_outer,
                    block_kv_compute=block_kv_compute,
                    causal=causal,
                )

    @pl.when(kv_block_index == kv_blocks - 1)
    def store():
        denominator = l_scratch_ref[:, :]
        valid_rows = denominator > 0.0
        safe_denominator = jnp.where(valid_rows, denominator, 1.0)
        normalized = jnp.where(
            _broadcast_row_stat(valid_rows, head_dim),
            o_scratch_ref[:, :] / _broadcast_row_stat(safe_denominator, head_dim),
            0.0,
        )
        lse = jnp.where(
            valid_rows,
            jnp.log(safe_denominator) + m_scratch_ref[:, :],
            0.0,
        )
        output_ref[0, :, :, :] = normalized.reshape(
            (block_q, block_h, head_dim)
        ).astype(output_ref.dtype)
        if len(lse_ref.shape) == 1:
            # All lanes contain the same row statistic. The reduction moves
            # it into a one-dimensional, densely stored TPU vector layout.
            lse_ref[:] = jnp.max(lse, axis=-1)
        else:
            lse_ref[0, :, :, :] = lse.reshape((block_q, block_h, _NUM_LANES))


def _forward_index_specs(
    local_q: Array,
    local_k: Array,
    profile: AttentionKernelProfile,
    *,
    causal: bool,
):
    batch, query_length, local_heads, head_dim = local_q.shape
    sequence = local_k.shape[1]
    q_blocks = query_length // profile.block_q
    kv_blocks = sequence // profile.block_kv_outer

    def q_index_map(batch_index, q_block_index, head_block_index, _kv_block_index, *scalar_refs):
        del scalar_refs
        return batch_index, q_block_index, head_block_index, 0

    def kv_segment_index_map(
        batch_index,
        q_block_index,
        _head_block_index,
        kv_block_index,
        q_summary_ref,
        kv_summary_ref,
        offsets_ref,
    ):
        should_run = _summary_should_run(
            q_summary_ref,
            kv_summary_ref,
            offsets_ref,
            batch_index,
            q_block_index,
            kv_block_index,
            block_q=profile.block_q,
            block_kv_outer=profile.block_kv_outer,
            causal=causal,
        )
        selected = lax.select(should_run, kv_block_index, 0)
        return batch_index, 0, selected

    q_spec = pl.BlockSpec(
        (1, profile.block_q, profile.block_h, head_dim),
        q_index_map,
    )
    q_segment_spec = pl.BlockSpec(
        (1, profile.block_q, _NUM_LANES),
        lambda batch_index, q_block_index, _head_block_index, _kv_block_index, *_: (
            batch_index,
            q_block_index,
            0,
        ),
    )
    kv_segment_spec = pl.BlockSpec(
        (1, _NUM_SUBLANES, profile.block_kv_outer),
        kv_segment_index_map,
    )
    if profile.compact_statistics:
        # block_h covers all local heads, hence one packed tile per Q block.
        # A flat array also avoids an 8-sublane alignment restriction on batch.
        lse_spec = pl.BlockSpec(
            (profile.block_q * profile.block_h,),
            lambda batch_index, q_block_index, _head_block_index, _kv_block_index, *_: (
                batch_index * q_blocks + q_block_index,
            ),
        )
    else:
        lse_spec = pl.BlockSpec(
            (1, profile.block_q, profile.block_h, _NUM_LANES),
            q_index_map,
        )
    grid = (
        batch,
        q_blocks,
        local_heads // profile.block_h,
        kv_blocks,
    )
    return grid, q_spec, q_segment_spec, kv_segment_spec, lse_spec


def _local_attention_forward(
    local_q: Array,
    local_k: Array,
    local_v: Array,
    local_q_segments: Array,
    local_kv_segments: Array,
    local_q_summary: Array,
    local_kv_summary: Array,
    offsets: Array,
    *,
    profile: AttentionKernelProfile,
    scale: float,
    causal: bool,
) -> tuple[Array, Array]:
    batch, query_length, local_heads, head_dim = local_q.shape
    sequence = local_k.shape[1]
    rows = profile.block_q * profile.block_h
    scale_bf16 = jnp.asarray(scale, dtype=local_q.dtype)
    scaled_q = (local_q * scale_bf16).astype(local_q.dtype)
    flat_k = local_k.reshape((batch * sequence, head_dim))
    flat_v = local_v.reshape((batch * sequence, head_dim))
    q_segment_lanes = lax.broadcast_in_dim(
        local_q_segments,
        (batch, query_length, _NUM_LANES),
        (0, 1),
    )
    kv_segment_lanes = lax.broadcast_in_dim(
        local_kv_segments,
        (batch, _NUM_SUBLANES, sequence),
        (0, 2),
    )
    grid, q_spec, q_segment_spec, kv_segment_spec, lse_spec = _forward_index_specs(
        local_q,
        local_k,
        profile,
        causal=causal,
    )
    output_shape = _output_shape(local_q.shape, local_q.dtype, local_q)
    lse_lane_shape = _output_shape(
        (batch * query_length * local_heads,)
        if profile.compact_statistics
        else (batch, query_length, local_heads, _NUM_LANES),
        jnp.float32,
        local_q,
    )
    output, lse_lanes = pl.pallas_call(
        functools.partial(
            _attention_forward_kernel,
            block_h=profile.block_h,
            block_q=profile.block_q,
            block_kv_outer=profile.block_kv_outer,
            block_kv_compute=profile.block_kv_compute,
            kv_blocks=sequence // profile.block_kv_outer,
            sequence=sequence,
            causal=causal,
        ),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=3,
            grid=grid,
            in_specs=(
                q_spec,
                pl.BlockSpec(memory_space=pl.ANY),
                pl.BlockSpec(memory_space=pl.ANY),
                q_segment_spec,
                kv_segment_spec,
            ),
            out_specs=(q_spec, lse_spec),
            scratch_shapes=(
                pltpu.VMEM((rows, _NUM_LANES), jnp.float32),
                pltpu.VMEM((rows, _NUM_LANES), jnp.float32),
                pltpu.VMEM((rows, head_dim), jnp.float32),
                pltpu.VMEM((2, profile.block_kv_compute, head_dim), local_k.dtype),
                pltpu.VMEM((2, profile.block_kv_compute, head_dim), local_v.dtype),
                pltpu.SemaphoreType.DMA((2,)),
                pltpu.SemaphoreType.DMA((2,)),
            ),
        ),
        out_shape=(output_shape, lse_lane_shape),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "parallel", "parallel", "arbitrary")
        ),
        name="causal_trainer_attention_fwd",
    )(
        local_q_summary,
        local_kv_summary,
        offsets,
        scaled_q,
        flat_k,
        flat_v,
        q_segment_lanes,
        kv_segment_lanes,
    )
    if profile.compact_statistics:
        return output, lse_lanes.reshape((batch, query_length, local_heads))
    # Compact the lane-expanded value at the kernel boundary to bound its lifetime.
    return output, lse_lanes[..., 0]


def _dq_compute_tile(
    q_ref,
    q_segment_ref,
    kv_segment_ref,
    lse_ref,
    do_ref,
    delta_ref,
    offsets_ref,
    dq_scratch_ref,
    k_buffer_ref,
    v_buffer_ref,
    *,
    slot: int,
    compute_offset: int,
    q_block_index,
    kv_block_index,
    block_h: int,
    block_q: int,
    block_kv_outer: int,
    block_kv_compute: int,
    causal: bool,
):
    rows = block_h * block_q
    head_dim = q_ref.shape[-1]
    query = q_ref[0, :, :, :].reshape((rows, head_dim))
    output_cotangent = do_ref[0, :, :, :].reshape((rows, head_dim)).astype(jnp.float32)
    logsumexp = _load_row_stat(lse_ref, rows)
    delta = _load_row_stat(delta_ref, rows)
    key = k_buffer_ref[slot, :, :]
    value = v_buffer_ref[slot, :, :]
    scores = lax.dot_general(
        query,
        key,
        _DOT_TRANSPOSE_RHS,
        preferred_element_type=jnp.float32,
    )
    valid = _exact_mask(
        q_segment_ref,
        kv_segment_ref,
        offsets_ref,
        q_block_index=q_block_index,
        kv_block_index=kv_block_index,
        compute_offset=compute_offset,
        block_h=block_h,
        block_q=block_q,
        block_kv_outer=block_kv_outer,
        block_kv_compute=block_kv_compute,
        causal=causal,
    )
    expanded_lse = jnp.tile(logsumexp, (1, block_kv_compute // _NUM_LANES))
    probability = jnp.where(valid, jnp.exp(scores - expanded_lse), 0.0)
    d_probability = lax.dot_general(
        output_cotangent,
        value.astype(jnp.float32),
        _DOT_TRANSPOSE_RHS,
        preferred_element_type=jnp.float32,
    )
    expanded_delta = jnp.tile(delta, (1, block_kv_compute // _NUM_LANES))
    d_score = probability * (d_probability - expanded_delta)
    dq_scratch_ref[:, :] += lax.dot(
        d_score.astype(jnp.float32),
        key.astype(jnp.float32),
        preferred_element_type=jnp.float32,
    )


def _attention_dq_kernel(
    q_summary_ref,
    kv_summary_ref,
    offsets_ref,
    q_ref,
    k_hbm_ref,
    v_hbm_ref,
    q_segment_ref,
    kv_segment_ref,
    lse_ref,
    do_ref,
    delta_ref,
    dq_ref,
    dq_scratch_ref,
    k_buffer_ref,
    v_buffer_ref,
    k_semaphore_ref,
    v_semaphore_ref,
    *,
    block_h: int,
    block_q: int,
    block_kv_outer: int,
    block_kv_compute: int,
    kv_blocks: int,
    sequence: int,
    scale: float,
    causal: bool,
):
    batch_index = pl.program_id(0)
    q_block_index = pl.program_id(1)
    kv_block_index = pl.program_id(3)
    rows = block_h * block_q
    head_dim = q_ref.shape[-1]

    @pl.when(kv_block_index == 0)
    def initialize():
        dq_scratch_ref[:, :] = jnp.zeros((rows, head_dim), jnp.float32)

    should_run = _summary_should_run(
        q_summary_ref,
        kv_summary_ref,
        offsets_ref,
        batch_index,
        q_block_index,
        kv_block_index,
        block_q=block_q,
        block_kv_outer=block_kv_outer,
        causal=causal,
    )

    @pl.when(should_run)
    def run():
        copies = _async_kv_copies(
            k_hbm_ref,
            v_hbm_ref,
            k_buffer_ref,
            v_buffer_ref,
            k_semaphore_ref,
            v_semaphore_ref,
            batch_index=batch_index,
            kv_block_index=kv_block_index,
            sequence=sequence,
            block_kv_outer=block_kv_outer,
            block_kv_compute=block_kv_compute,
        )
        copies[0][0].start()
        copies[0][1].start()
        for compute_index, (k_copy, v_copy) in enumerate(copies):
            k_copy.wait()
            v_copy.wait()
            if compute_index + 1 < len(copies):
                copies[compute_index + 1][0].start()
                copies[compute_index + 1][1].start()
            compute_offset = compute_index * block_kv_compute

            @pl.when(
                _compute_tile_is_causally_reachable(
                    offsets_ref,
                    q_block_index=q_block_index,
                    kv_block_index=kv_block_index,
                    compute_offset=compute_offset,
                    block_q=block_q,
                    block_kv_outer=block_kv_outer,
                    causal=causal,
                )
            )
            def compute(
                compute_index=compute_index,
                compute_offset=compute_offset,
            ):
                _dq_compute_tile(
                    q_ref,
                    q_segment_ref,
                    kv_segment_ref,
                    lse_ref,
                    do_ref,
                    delta_ref,
                    offsets_ref,
                    dq_scratch_ref,
                    k_buffer_ref,
                    v_buffer_ref,
                    slot=compute_index % 2,
                    compute_offset=compute_offset,
                    q_block_index=q_block_index,
                    kv_block_index=kv_block_index,
                    block_h=block_h,
                    block_q=block_q,
                    block_kv_outer=block_kv_outer,
                    block_kv_compute=block_kv_compute,
                    causal=causal,
                )

    @pl.when(kv_block_index == kv_blocks - 1)
    def store():
        quantized_scale = jnp.asarray(scale, jnp.bfloat16).astype(jnp.float32)
        dq_ref[0, :, :, :] = (dq_scratch_ref[:, :] * quantized_scale).reshape(
            (block_q, block_h, head_dim)
        ).astype(dq_ref.dtype)


def _local_attention_dq(
    local_q: Array,
    local_k: Array,
    local_v: Array,
    local_q_segments: Array,
    local_kv_segments: Array,
    local_q_summary: Array,
    local_kv_summary: Array,
    lse: Array,
    output_cotangent: Array,
    delta: Array,
    offsets: Array,
    *,
    profile: AttentionKernelProfile,
    scale: float,
    causal: bool,
) -> Array:
    batch, query_length, _local_heads, head_dim = local_q.shape
    sequence = local_k.shape[1]
    rows = profile.block_q * profile.block_h
    scale_bf16 = jnp.asarray(scale, dtype=local_q.dtype)
    scaled_q = (local_q * scale_bf16).astype(local_q.dtype)
    flat_k = local_k.reshape((batch * sequence, head_dim))
    flat_v = local_v.reshape((batch * sequence, head_dim))
    q_segment_lanes = lax.broadcast_in_dim(
        local_q_segments,
        (batch, query_length, _NUM_LANES),
        (0, 1),
    )
    kv_segment_lanes = lax.broadcast_in_dim(
        local_kv_segments,
        (batch, _NUM_SUBLANES, sequence),
        (0, 2),
    )
    lse_lanes = _statistics_kernel_input(lse, compact=profile.compact_statistics)
    delta_lanes = _statistics_kernel_input(delta, compact=profile.compact_statistics)
    grid, q_spec, q_segment_spec, kv_segment_spec, stat_spec = _forward_index_specs(
        local_q,
        local_k,
        profile,
        causal=causal,
    )
    return pl.pallas_call(
        functools.partial(
            _attention_dq_kernel,
            block_h=profile.block_h,
            block_q=profile.block_q,
            block_kv_outer=profile.block_kv_outer,
            block_kv_compute=profile.block_kv_compute,
            kv_blocks=sequence // profile.block_kv_outer,
            sequence=sequence,
            scale=scale,
            causal=causal,
        ),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=3,
            grid=grid,
            in_specs=(
                q_spec,
                pl.BlockSpec(memory_space=pl.ANY),
                pl.BlockSpec(memory_space=pl.ANY),
                q_segment_spec,
                kv_segment_spec,
                stat_spec,
                q_spec,
                stat_spec,
            ),
            out_specs=q_spec,
            scratch_shapes=(
                pltpu.VMEM((rows, head_dim), jnp.float32),
                pltpu.VMEM((2, profile.block_kv_compute, head_dim), local_k.dtype),
                pltpu.VMEM((2, profile.block_kv_compute, head_dim), local_v.dtype),
                pltpu.SemaphoreType.DMA((2,)),
                pltpu.SemaphoreType.DMA((2,)),
            ),
        ),
        out_shape=_output_shape(local_q.shape, local_q.dtype, local_q),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "parallel", "parallel", "arbitrary")
        ),
        name="causal_trainer_attention_dq",
    )(
        local_q_summary,
        local_kv_summary,
        offsets,
        scaled_q,
        flat_k,
        flat_v,
        q_segment_lanes,
        kv_segment_lanes,
        lse_lanes,
        output_cotangent,
        delta_lanes,
    )


def _dkv_compute_tile(
    q_ref,
    q_segment_ref,
    kv_segment_ref,
    lse_ref,
    do_ref,
    delta_ref,
    offsets_ref,
    dk_scratch_ref,
    dv_scratch_ref,
    k_ref,
    v_ref,
    *,
    compute_offset: int,
    q_block_index,
    kv_block_index,
    block_h: int,
    block_q: int,
    block_kv_outer: int,
    block_kv_compute: int,
    causal: bool,
):
    rows = block_h * block_q
    head_dim = q_ref.shape[-1]
    query_bf16 = q_ref[0, :, :, :].reshape((rows, head_dim))
    query = query_bf16.astype(jnp.float32)
    output_cotangent = do_ref[0, :, :, :].reshape((rows, head_dim)).astype(jnp.float32)
    logsumexp = _load_row_stat(lse_ref, rows)
    delta = _load_row_stat(delta_ref, rows)
    compute_slice = pl.ds(compute_offset, block_kv_compute)
    key = k_ref[compute_slice, :]
    value = v_ref[compute_slice, :]
    scores = lax.dot_general(
        query_bf16,
        key,
        _DOT_TRANSPOSE_RHS,
        preferred_element_type=jnp.float32,
    )
    valid = _exact_mask(
        q_segment_ref,
        kv_segment_ref,
        offsets_ref,
        q_block_index=q_block_index,
        kv_block_index=kv_block_index,
        compute_offset=compute_offset,
        block_h=block_h,
        block_q=block_q,
        block_kv_outer=block_kv_outer,
        block_kv_compute=block_kv_compute,
        causal=causal,
    )
    expanded_lse = jnp.tile(logsumexp, (1, block_kv_compute // _NUM_LANES))
    probability = jnp.where(valid, jnp.exp(scores - expanded_lse), 0.0)
    d_value = lax.dot(
        probability.T.astype(jnp.float32),
        output_cotangent,
        preferred_element_type=jnp.float32,
    )
    d_probability = lax.dot_general(
        output_cotangent,
        value.astype(jnp.float32),
        _DOT_TRANSPOSE_RHS,
        preferred_element_type=jnp.float32,
    )
    expanded_delta = jnp.tile(delta, (1, block_kv_compute // _NUM_LANES))
    d_score = probability * (d_probability - expanded_delta)
    d_key = lax.dot(
        d_score.T.astype(jnp.float32),
        query,
        preferred_element_type=jnp.float32,
    )
    target = pl.ds(compute_offset, block_kv_compute)
    dk_scratch_ref[target, :] += d_key
    dv_scratch_ref[target, :] += d_value


def _attention_dkv_kernel(
    q_summary_ref,
    kv_summary_ref,
    offsets_ref,
    q_ref,
    k_ref,
    v_ref,
    q_segment_ref,
    kv_segment_ref,
    lse_ref,
    do_ref,
    delta_ref,
    dk_ref,
    dv_ref,
    dk_scratch_ref,
    dv_scratch_ref,
    *,
    block_h: int,
    block_q: int,
    block_kv_outer: int,
    block_kv_compute: int,
    head_blocks: int,
    q_work_blocks: int,
    causal: bool,
):
    batch_index = pl.program_id(0)
    kv_block_index = pl.program_id(1)
    q_work_index = pl.program_id(2)
    q_block_index = q_work_index // head_blocks
    head_dim = q_ref.shape[-1]

    @pl.when(q_work_index == 0)
    def initialize():
        dk_scratch_ref[:, :] = jnp.zeros((block_kv_outer, head_dim), jnp.float32)
        dv_scratch_ref[:, :] = jnp.zeros((block_kv_outer, head_dim), jnp.float32)

    should_run = _summary_should_run(
        q_summary_ref,
        kv_summary_ref,
        offsets_ref,
        batch_index,
        q_block_index,
        kv_block_index,
        block_q=block_q,
        block_kv_outer=block_kv_outer,
        causal=causal,
    )

    @pl.when(should_run)
    def run():
        for compute_index in range(block_kv_outer // block_kv_compute):
            compute_offset = compute_index * block_kv_compute

            @pl.when(
                _compute_tile_is_causally_reachable(
                    offsets_ref,
                    q_block_index=q_block_index,
                    kv_block_index=kv_block_index,
                    compute_offset=compute_offset,
                    block_q=block_q,
                    block_kv_outer=block_kv_outer,
                    causal=causal,
                )
            )
            def compute(compute_offset=compute_offset):
                _dkv_compute_tile(
                    q_ref,
                    q_segment_ref,
                    kv_segment_ref,
                    lse_ref,
                    do_ref,
                    delta_ref,
                    offsets_ref,
                    dk_scratch_ref,
                    dv_scratch_ref,
                    k_ref,
                    v_ref,
                    compute_offset=compute_offset,
                    q_block_index=q_block_index,
                    kv_block_index=kv_block_index,
                    block_h=block_h,
                    block_q=block_q,
                    block_kv_outer=block_kv_outer,
                    block_kv_compute=block_kv_compute,
                    causal=causal,
                )

    @pl.when(q_work_index == q_work_blocks - 1)
    def store():
        dk_ref[:, :] = dk_scratch_ref[:, :]
        dv_ref[:, :] = dv_scratch_ref[:, :]


def _local_attention_dkv(
    local_q: Array,
    local_k: Array,
    local_v: Array,
    local_q_segments: Array,
    local_kv_segments: Array,
    local_q_summary: Array,
    local_kv_summary: Array,
    lse: Array,
    output_cotangent: Array,
    delta: Array,
    offsets: Array,
    *,
    profile: AttentionKernelProfile,
    scale: float,
    causal: bool,
) -> tuple[Array, Array]:
    batch, query_length, local_heads, head_dim = local_q.shape
    sequence = local_k.shape[1]
    head_blocks = local_heads // profile.block_h
    q_blocks = query_length // profile.block_q
    q_work_blocks = q_blocks * head_blocks
    scale_bf16 = jnp.asarray(scale, dtype=local_q.dtype)
    scaled_q = (local_q * scale_bf16).astype(local_q.dtype)
    flat_k = local_k.reshape((batch * sequence, head_dim))
    flat_v = local_v.reshape((batch * sequence, head_dim))
    q_segment_lanes = lax.broadcast_in_dim(
        local_q_segments,
        (batch, query_length, _NUM_LANES),
        (0, 1),
    )
    kv_segment_lanes = lax.broadcast_in_dim(
        local_kv_segments,
        (batch, _NUM_SUBLANES, sequence),
        (0, 2),
    )
    lse_lanes = _statistics_kernel_input(lse, compact=profile.compact_statistics)
    delta_lanes = _statistics_kernel_input(delta, compact=profile.compact_statistics)

    def q_index_map(batch_index, _kv_block_index, q_work_index, *_):
        return batch_index, q_work_index // head_blocks, q_work_index % head_blocks, 0

    def q_segment_index_map(batch_index, _kv_block_index, q_work_index, *_):
        return batch_index, q_work_index // head_blocks, 0

    def kv_segment_index_map(
        batch_index,
        kv_block_index,
        q_work_index,
        q_summary_ref,
        kv_summary_ref,
        offsets_ref,
    ):
        q_block_index = q_work_index // head_blocks
        should_run = _summary_should_run(
            q_summary_ref,
            kv_summary_ref,
            offsets_ref,
            batch_index,
            q_block_index,
            kv_block_index,
            block_q=profile.block_q,
            block_kv_outer=profile.block_kv_outer,
            causal=causal,
        )
        selected = lax.select(should_run, kv_block_index, 0)
        return batch_index, 0, selected

    def kv_index_map(batch_index, kv_block_index, _q_work_index, *scalar_refs):
        del scalar_refs
        return batch_index * (sequence // profile.block_kv_outer) + kv_block_index, 0

    q_spec = pl.BlockSpec(
        (1, profile.block_q, profile.block_h, head_dim),
        q_index_map,
    )
    q_segment_spec = pl.BlockSpec(
        (1, profile.block_q, _NUM_LANES),
        q_segment_index_map,
    )
    kv_segment_spec = pl.BlockSpec(
        (1, _NUM_SUBLANES, profile.block_kv_outer),
        kv_segment_index_map,
    )
    if profile.compact_statistics:
        stat_spec = pl.BlockSpec(
            (profile.block_q * profile.block_h,),
            lambda batch_index, _kv_block_index, q_work_index, *_: (batch_index * q_blocks + q_work_index,),
        )
    else:
        stat_spec = pl.BlockSpec(
            (1, profile.block_q, profile.block_h, _NUM_LANES),
            q_index_map,
        )
    dkv_spec = pl.BlockSpec(
        (profile.block_kv_outer, head_dim),
        kv_index_map,
    )
    output_shapes = (
        # These are partial dK/dV values and vary with each local Q shard even
        # though their physical shapes match replicated K/V.  The enclosing
        # shard_map psum removes that variation before returning them.
        _output_shape(flat_k.shape, jnp.float32, local_q),
        _output_shape(flat_v.shape, jnp.float32, local_q),
    )
    flat_dk, flat_dv = pl.pallas_call(
        functools.partial(
            _attention_dkv_kernel,
            block_h=profile.block_h,
            block_q=profile.block_q,
            block_kv_outer=profile.block_kv_outer,
            block_kv_compute=profile.block_kv_compute,
            head_blocks=head_blocks,
            q_work_blocks=q_work_blocks,
            causal=causal,
        ),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=3,
            grid=(batch, sequence // profile.block_kv_outer, q_work_blocks),
            in_specs=(
                q_spec,
                dkv_spec,
                dkv_spec,
                q_segment_spec,
                kv_segment_spec,
                stat_spec,
                q_spec,
                stat_spec,
            ),
            out_specs=(dkv_spec, dkv_spec),
            scratch_shapes=(
                pltpu.VMEM((profile.block_kv_outer, head_dim), jnp.float32),
                pltpu.VMEM((profile.block_kv_outer, head_dim), jnp.float32),
            ),
        ),
        out_shape=output_shapes,
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "parallel", "arbitrary")
        ),
        name="causal_trainer_attention_dkv",
    )(
        local_q_summary,
        local_kv_summary,
        offsets,
        scaled_q,
        flat_k,
        flat_v,
        q_segment_lanes,
        kv_segment_lanes,
        lse_lanes,
        output_cotangent,
        delta_lanes,
    )
    return flat_dk.reshape(local_k.shape), flat_dv.reshape(local_v.shape)


def _local_offsets(
    local_q: Array,
    *,
    global_query_heads: int,
    kv_heads: int,
    mesh: Mesh,
) -> Array:
    if "sp" in mesh.axis_names and int(mesh.shape["sp"]) > 1:
        q_offset = lax.axis_index("sp") * local_q.shape[1]
    else:
        q_offset = jnp.asarray(0, jnp.int32)
    if "tp" in mesh.axis_names and int(mesh.shape["tp"]) > 1:
        global_head_start = lax.axis_index("tp") * local_q.shape[2]
    else:
        global_head_start = jnp.asarray(0, jnp.int32)
    q_heads_per_kv = global_query_heads // kv_heads
    kv_head_index = global_head_start // q_heads_per_kv
    return jnp.stack(
        (jnp.asarray(q_offset, jnp.int32), jnp.asarray(kv_head_index, jnp.int32))
    )


def _distributed_forward(
    query: Array,
    key: Array,
    value: Array,
    metadata: AttentionMetadata,
    *,
    mesh: Mesh,
    profile: AttentionKernelProfile,
    scale: float,
    causal: bool,
) -> tuple[Array, Array]:
    specs = attention_partition_specs(mesh)
    global_query_heads = query.shape[2]
    kv_heads = key.shape[2]

    def local(
        local_q,
        local_k,
        local_v,
        local_q_segments,
        local_kv_segments,
        local_q_summary,
        local_kv_summary,
    ):
        offsets = _local_offsets(
            local_q,
            global_query_heads=global_query_heads,
            kv_heads=kv_heads,
            mesh=mesh,
        )
        return _local_attention_forward(
            local_q,
            local_k,
            local_v,
            local_q_segments,
            local_kv_segments,
            local_q_summary,
            local_kv_summary,
            offsets,
            profile=profile,
            scale=scale,
            causal=causal,
        )

    return shard_map(
        local,
        mesh=mesh,
        in_specs=(
            specs["query"],
            specs["kv"],
            specs["kv"],
            specs["query_segments"],
            specs["kv_segments"],
            specs["query_summary"],
            specs["kv_summary"],
        ),
        out_specs=(specs["query"], specs["stats"]),
        check_vma=True,
    )(
        query,
        key,
        value,
        metadata.query_segment_ids,
        metadata.kv_segment_ids,
        metadata.q_block_segments,
        metadata.kv_block_segments,
    )


def _distributed_backward(
    query: Array,
    key: Array,
    value: Array,
    metadata: AttentionMetadata,
    output: Array,
    lse: Array,
    output_cotangent: Array,
    *,
    mesh: Mesh,
    profile: AttentionKernelProfile,
    scale: float,
    causal: bool,
) -> tuple[Array, Array, Array]:
    specs = attention_partition_specs(mesh)
    global_query_heads = query.shape[2]
    kv_heads = key.shape[2]
    reduction_axes = tuple(
        axis for axis in ("tp", "sp") if axis in mesh.axis_names and int(mesh.shape[axis]) > 1
    )

    def local(
        local_q,
        local_k,
        local_v,
        local_q_segments,
        local_kv_segments,
        local_q_summary,
        local_kv_summary,
        local_output,
        local_lse,
        local_do,
    ):
        offsets = _local_offsets(
            local_q,
            global_query_heads=global_query_heads,
            kv_heads=kv_heads,
            mesh=mesh,
        )
        delta = jnp.sum(
            local_output.astype(jnp.float32) * local_do.astype(jnp.float32),
            axis=-1,
        )
        dq = _local_attention_dq(
            local_q,
            local_k,
            local_v,
            local_q_segments,
            local_kv_segments,
            local_q_summary,
            local_kv_summary,
            local_lse,
            local_do,
            delta,
            offsets,
            profile=profile,
            scale=scale,
            causal=causal,
        )
        partial_dk, partial_dv = _local_attention_dkv(
            local_q,
            local_k,
            local_v,
            local_q_segments,
            local_kv_segments,
            local_q_summary,
            local_kv_summary,
            local_lse,
            local_do,
            delta,
            offsets,
            profile=profile,
            scale=scale,
            causal=causal,
        )
        if reduction_axes:
            partial_dk = lax.psum(partial_dk, reduction_axes)
            partial_dv = lax.psum(partial_dv, reduction_axes)
        return dq, partial_dk.astype(local_k.dtype), partial_dv.astype(local_v.dtype)

    return shard_map(
        local,
        mesh=mesh,
        in_specs=(
            specs["query"],
            specs["kv"],
            specs["kv"],
            specs["query_segments"],
            specs["kv_segments"],
            specs["query_summary"],
            specs["kv_summary"],
            specs["query"],
            specs["stats"],
            specs["query"],
        ),
        out_specs=(specs["query"], specs["kv"], specs["kv"]),
        check_vma=True,
    )(
        query,
        key,
        value,
        metadata.query_segment_ids,
        metadata.kv_segment_ids,
        metadata.q_block_segments,
        metadata.kv_block_segments,
        output,
        lse,
        output_cotangent,
    )


@functools.partial(jax.custom_vjp, nondiff_argnums=(4, 5, 6, 7))
def _efficient_attention_impl(
    query: Array,
    key: Array,
    value: Array,
    metadata: AttentionMetadata,
    mesh: Mesh,
    profile: AttentionKernelProfile,
    scale: float,
    causal: bool,
) -> Array:
    output, _ = _distributed_forward(
        query,
        key,
        value,
        metadata,
        mesh=mesh,
        profile=profile,
        scale=scale,
        causal=causal,
    )
    return output


def _efficient_attention_impl_fwd(query, key, value, metadata, mesh, profile, scale, causal):
    output, lse = _distributed_forward(
        query,
        key,
        value,
        metadata,
        mesh=mesh,
        profile=profile,
        scale=scale,
        causal=causal,
    )
    return output, (query, key, value, metadata, output, lse)


def _efficient_attention_impl_bwd(mesh, profile, scale, causal, residual, output_cotangent):
    query, key, value, metadata, output, lse = residual
    dq, dk, dv = _distributed_backward(
        query,
        key,
        value,
        metadata,
        output,
        lse,
        output_cotangent,
        mesh=mesh,
        profile=profile,
        scale=scale,
        causal=causal,
    )
    return dq, dk, dv, None


_efficient_attention_impl.defvjp(_efficient_attention_impl_fwd, _efficient_attention_impl_bwd)


def _validate_metadata(
    metadata: AttentionMetadata,
    query_shape: tuple[int, ...],
    profile: AttentionKernelProfile,
) -> None:
    if not isinstance(metadata, AttentionMetadata):
        raise TypeError("metadata must be AttentionMetadata")
    batch, sequence = query_shape[:2]
    expected = {
        "query_segment_ids": (batch, sequence),
        "kv_segment_ids": (batch, sequence),
        "q_block_segments": (batch, sequence // profile.block_q, 2),
        "kv_block_segments": (batch, sequence // profile.block_kv_outer, 2),
    }
    for name, shape in expected.items():
        value = getattr(metadata, name)
        if value.shape != shape:
            raise ValueError(f"metadata.{name} must have shape {shape}, got {value.shape}")
        if value.dtype != jnp.int32:
            raise TypeError(f"metadata.{name} must have dtype int32, got {value.dtype}")


def efficient_attention(
    query: Array,
    key: Array,
    value: Array,
    metadata: AttentionMetadata,
    *,
    mesh: Mesh,
    profile: AttentionKernelProfile | None = None,
    scale: float | None = None,
    causal: bool = True,
) -> Array:
    """Apply the efficient TPU attention forward and its analytic custom VJP.

    This entry point accepts global arrays.  Q/output are sharded over SP and
    TP; shared K/V tensors and their packed metadata are replicated over both.
    Inputs are validated against the static profile, dtype, and mesh layout.
    """

    query = jnp.asarray(query)
    key = jnp.asarray(key)
    value = jnp.asarray(value)
    if mesh is None:
        raise ValueError("efficient_attention requires the training Mesh")
    if not isinstance(causal, bool):
        raise TypeError("causal must be a compile-time bool")
    if not causal:
        raise ValueError("the efficient backend implements causal attention only")
    if key.shape != value.shape:
        raise ValueError(f"key and value shapes differ: {key.shape} and {value.shape}")
    query_shape, key_shape, _, _ = _validate_qkv_shapes(query.shape, key.shape, mesh)
    if query_shape[-1] != _SUPPORTED_HEAD_DIM:
        raise ValueError(
            "the efficient attention kernel requires head_dim=128; "
            f"got head_dim={query_shape[-1]}"
        )
    if key_shape[2] != 1:
        raise ValueError(
            f"unsupported K/V head count for the efficient owner schedule: {key_shape[2]}"
        )
    if query.dtype != jnp.bfloat16 or key.dtype != jnp.bfloat16 or value.dtype != jnp.bfloat16:
        raise TypeError(
            "efficient_attention requires BF16 query/key/value, got "
            f"{query.dtype}, {key.dtype}, {value.dtype}"
        )
    if jax.default_backend() != "tpu":
        raise RuntimeError("efficient_attention is available only on TPU")

    resolved_profile = (
        resolve_attention_profile(query_shape, key_shape, mesh)
        if profile is None
        else profile
    )
    validate_attention_profile(resolved_profile, query_shape, key_shape, mesh)
    _validate_metadata(metadata, query_shape, resolved_profile)
    if scale is None:
        resolved_scale = default_attention_scale(query_shape[-1])
    else:
        if isinstance(scale, (jax.Array, jax.core.Tracer)):
            raise TypeError("scale must be a compile-time Python scalar")
        try:
            resolved_scale = float(scale)
        except (TypeError, ValueError) as error:
            raise TypeError("scale must be a compile-time Python scalar") from error
        if not math.isfinite(resolved_scale):
            raise ValueError(f"scale must be finite, got {resolved_scale}")
    return _efficient_attention_impl(
        query,
        key,
        value,
        metadata,
        mesh,
        resolved_profile,
        resolved_scale,
        causal,
    )


__all__ = [
    "AttentionKernelProfile",
    "AttentionMetadata",
    "attention_partition_specs",
    "default_attention_scale",
    "efficient_attention",
    "prepare_attention_metadata",
    "resolve_attention_profile",
    "validate_attention_profile",
]
