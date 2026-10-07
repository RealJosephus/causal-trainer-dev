"""TPU sparse FlashAttention with bit-packed routes and shared MQA storage.

The router supplies integer addresses, but the compute kernels never gather
individual K/V vectors. They stream contiguous, nonempty KV tiles through
double-buffered DMA and execute matrix products on the MXU. The fused backward
keeps dQ in VMEM and updates shared FP32 dK/dV tiles in place, without atomics.
Routing workspace is bounded by one query tile, including during backward.
"""

from __future__ import annotations

import functools
import math

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jax.sharding import PartitionSpec as P

from .indexed_attention import _pad

_NT = (((1,), (1,)), ((), ()))
_TN = (((0,), (0,)), ((), ()))


def _shape(shape, dtype, like):
    manual = getattr(getattr(like, "aval", like), "manual_axis_type", None)
    return jax.ShapeDtypeStruct(shape, dtype, **({"manual_axis_type": manual} if manual is not None else {}))


def pack_routes(indices, bank_length, key_tile):
    """Unique addresses [rows, selected] -> uint32 [bank_words, rows].

    The two small integer arrays are the DMA schedule: actual block count,
    then the sorted nonempty block IDs. No dense boolean Q-by-K mask exists.
    """
    rows = indices.shape[0]
    padded_length = math.ceil(bank_length / key_tile) * key_tile
    valid = (indices >= 0) & (indices < bank_length)
    safe = jnp.where(valid, indices, 0)
    bits = jnp.where(valid, jnp.left_shift(jnp.uint32(1), (safe % 32).astype(jnp.uint32)), jnp.uint32(0))
    words = jnp.zeros((padded_length // 32, rows), jnp.uint32)
    # Addresses are unique in a row, so addition is equivalent to bitwise OR,
    # including the uint32 sign bit. There is no floating-point scatter here.
    words = words.at[safe // 32, jnp.arange(rows)[:, None]].add(bits)
    return words, _tile_schedule(words, key_tile)


def _tile_schedule(words, key_tile):
    occupied = jnp.any(words.reshape(-1, key_tile // 32, words.shape[1]) != 0, axis=(1, 2))
    blocks = jnp.nonzero(occupied, size=occupied.size, fill_value=0)[0].astype(jnp.int32)
    schedule = jnp.concatenate((jnp.sum(occupied, dtype=jnp.int32)[None], blocks))
    return schedule


def _mask(words, key_tile):
    # Decode each word in the final [query, key] order.
    expanded = jnp.repeat(words.T, 32, axis=1)
    shifts = (jnp.arange(key_tile, dtype=jnp.uint32) % 32)[None, :]
    return ((expanded >> shifts) & jnp.uint32(1)) != 0


def _compact_row_stat(stat):
    # The lanes hold identical values. Transpose before selecting a row so
    # Mosaic uses its native transpose instead of a v4-unsupported sublane
    # gather when converting row-replicated registers to compact HBM storage.
    return stat.T[0, :]


def _expand_row_stat(stat, width):
    # Inverse of the compact store, entirely inside the current VMEM tile.
    lanes = jnp.broadcast_to(stat[None, :], (128, stat.shape[0])).T
    return jnp.tile(lanes, (1, width // 128))


def _copies(schedule, index, slot, k, v, mask, kb, vb, mb, ks, vs, ms, key_tile):
    block = schedule[index + 1]
    return (
        pltpu.make_async_copy(k.at[pl.ds(block * key_tile, key_tile), :], kb.at[slot], ks.at[slot]),
        pltpu.make_async_copy(v.at[pl.ds(block * key_tile, key_tile), :], vb.at[slot], vs.at[slot]),
        pltpu.make_async_copy(
            mask.at[pl.ds(block * (key_tile // 32), key_tile // 32), :], mb.at[slot], ms.at[slot],
        ),
    )


def _forward_kernel(schedule, q, k, v, mask, out, lse, m, den, acc, kb, vb, mb, ks, vs, ms, *, key_tile):
    rows, dim = q.shape
    m[...] = jnp.full((rows, 128), -1e30, jnp.float32)
    den[...] = jnp.zeros((rows, 128), jnp.float32)
    acc[...] = jnp.zeros((rows, dim), jnp.float32)

    @pl.when(schedule[0] > 0)
    def start():
        for copy in _copies(schedule, 0, 0, k, v, mask, kb, vb, mb, ks, vs, ms, key_tile):
            copy.start()

    def body(index, _):
        slot = index % 2
        for copy in _copies(schedule, index, slot, k, v, mask, kb, vb, mb, ks, vs, ms, key_tile):
            copy.wait()

        @pl.when(index + 1 < schedule[0])
        def prefetch():
            for copy in _copies(schedule, index + 1, 1 - slot, k, v, mask, kb, vb, mb, ks, vs, ms, key_tile):
                copy.start()

        scores = lax.dot_general(q[...], kb[slot], _NT, preferred_element_type=jnp.float32)
        allowed = _mask(mb[slot], key_tile)
        scores = jnp.where(allowed, scores, -1e30)
        next_m = jnp.maximum(m[...], jnp.broadcast_to(jnp.max(scores, axis=1)[:, None], (rows, 128)))
        correction = jnp.exp(m[...] - next_m)
        p = jnp.where(allowed, jnp.exp(scores - jnp.tile(next_m, (1, key_tile // 128))), 0)
        den[...] = den[...] * correction + jnp.broadcast_to(jnp.sum(p, axis=1)[:, None], (rows, 128))
        acc[...] = acc[...] * correction + lax.dot(
            p.astype(v.dtype), vb[slot], preferred_element_type=jnp.float32,
        )
        m[...] = next_m

    lax.fori_loop(0, schedule[0], body, None)
    safe_den = jnp.maximum(den[...], 1e-30)
    out[...] = acc[...] / safe_den
    lse[...] = _compact_row_stat(jnp.where(den[...] > 0, m[...] + jnp.log(safe_den), 0))


def _backward_kernel(
    schedule, q, k, v, mask, do, lse, delta, dk_in, dv_in,
    dq, dk, dv, dq_acc, kb, vb, mb, dkb, dvb, ks, vs, ms, dks, dvs, *, key_tile,
):
    del dk_in, dv_in  # Aliased with the output HBM buffers.
    rows, dim = q.shape
    dq_acc[...] = jnp.zeros((rows, dim), jnp.float32)
    log_normalizer = _expand_row_stat(lse[...], key_tile)
    row_delta = _expand_row_stat(delta[...], key_tile)

    @pl.when(schedule[0] > 0)
    def start():
        for copy in _copies(schedule, 0, 0, k, v, mask, kb, vb, mb, ks, vs, ms, key_tile):
            copy.start()

    def body(index, _):
        slot = index % 2
        block = schedule[index + 1]
        for copy in _copies(schedule, index, slot, k, v, mask, kb, vb, mb, ks, vs, ms, key_tile):
            copy.wait()

        @pl.when(index + 1 < schedule[0])
        def prefetch():
            for copy in _copies(schedule, index + 1, 1 - slot, k, v, mask, kb, vb, mb, ks, vs, ms, key_tile):
                copy.start()

        dk_view = dk.at[pl.ds(block * key_tile, key_tile), :]
        dv_view = dv.at[pl.ds(block * key_tile, key_tile), :]
        load_dk = pltpu.make_async_copy(dk_view, dkb, dks)
        load_dv = pltpu.make_async_copy(dv_view, dvb, dvs)
        load_dk.start()
        load_dv.start()
        scores = lax.dot_general(q[...], kb[slot], _NT, preferred_element_type=jnp.float32)
        allowed = _mask(mb[slot], key_tile)
        p = jnp.exp(jnp.where(allowed, scores - log_normalizer, -1e30))
        dp = lax.dot_general(do[...], vb[slot], _NT, preferred_element_type=jnp.float32)
        ds = p * (dp - row_delta)
        dq_acc[...] += lax.dot(ds.astype(q.dtype), kb[slot], preferred_element_type=jnp.float32)
        d_key = lax.dot_general(ds.astype(q.dtype), q[...], _TN, preferred_element_type=jnp.float32)
        d_value = lax.dot_general(p.astype(do.dtype), do[...], _TN, preferred_element_type=jnp.float32)
        load_dk.wait()
        load_dv.wait()
        dkb[...] += d_key
        dvb[...] += d_value
        store_dk = pltpu.make_async_copy(dkb, dk_view, dks)
        store_dv = pltpu.make_async_copy(dvb, dv_view, dvs)
        store_dk.start()
        store_dv.start()
        store_dk.wait()
        store_dv.wait()

    lax.fori_loop(0, schedule[0], body, None)
    dq[...] = dq_acc[...]


def _dma_scratch(rows, key_tile, dtype):
    return (
        pltpu.VMEM((2, key_tile, 128), dtype), pltpu.VMEM((2, key_tile, 128), dtype),
        pltpu.VMEM((2, key_tile // 32, rows), jnp.uint32),
    )


def flash_forward(q, k, v, words, schedule, *, key_tile):
    """One full-MXU query tile, any number of nonempty KV tiles."""
    rows, dim = q.shape
    if dim != 128 or rows % 128 or key_tile % 128:
        raise ValueError("TPU sparse FlashAttention requires head_dim=128 and 128-aligned tiles")
    q_spec = pl.BlockSpec(q.shape, lambda *_: (0, 0))
    stat_spec = pl.BlockSpec((rows,), lambda *_: (0,))
    hbm = pl.BlockSpec(memory_space=pl.ANY)
    return pl.pallas_call(
        functools.partial(_forward_kernel, key_tile=key_tile),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=1, grid=(1,), in_specs=(q_spec, hbm, hbm, hbm),
            out_specs=(q_spec, stat_spec), scratch_shapes=(
                pltpu.VMEM((rows, 128), jnp.float32), pltpu.VMEM((rows, 128), jnp.float32),
                pltpu.VMEM((rows, dim), jnp.float32), *_dma_scratch(rows, key_tile, q.dtype),
                pltpu.SemaphoreType.DMA((2,)), pltpu.SemaphoreType.DMA((2,)), pltpu.SemaphoreType.DMA((2,)),
            ),
        ),
        out_shape=(_shape(q.shape, jnp.float32, q), _shape((rows,), jnp.float32, q)),
        compiler_params=pltpu.CompilerParams(dimension_semantics=("arbitrary",)),
        name="causal_sparse_flash_fwd",
    )(schedule, q, k, v, words)


def flash_backward(q, k, v, words, schedule, do, lse, delta, dk, dv, *, key_tile):
    rows = q.shape[0]
    q_spec = pl.BlockSpec(q.shape, lambda *_: (0, 0))
    stat_spec = pl.BlockSpec((rows,), lambda *_: (0,))
    hbm = pl.BlockSpec(memory_space=pl.ANY)
    return pl.pallas_call(
        functools.partial(_backward_kernel, key_tile=key_tile),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=1, grid=(1,),
            in_specs=(q_spec, hbm, hbm, hbm, q_spec, stat_spec, stat_spec, hbm, hbm),
            out_specs=(q_spec, hbm, hbm), scratch_shapes=(
                pltpu.VMEM(q.shape, jnp.float32), *_dma_scratch(rows, key_tile, q.dtype),
                pltpu.VMEM((key_tile, 128), jnp.float32), pltpu.VMEM((key_tile, 128), jnp.float32),
                pltpu.SemaphoreType.DMA((2,)), pltpu.SemaphoreType.DMA((2,)), pltpu.SemaphoreType.DMA((2,)),
                pltpu.SemaphoreType.DMA, pltpu.SemaphoreType.DMA,
            ),
        ),
        out_shape=tuple(_shape(x.shape, jnp.float32, q) for x in (q, dk, dv)),
        input_output_aliases={8: 1, 9: 2},
        compiler_params=pltpu.CompilerParams(dimension_semantics=("arbitrary",)),
        name="causal_sparse_flash_bwd",
    )(schedule, q, k, v, words, do, lse, delta, dk, dv)


def _local_workspace(q, k, v, info, query_tile, key_tile):
    batch, length, heads, dim = q.shape
    if dim != 128:
        raise ValueError("TPU sparse attention currently requires head_dim=128")
    # A full MXU tile is mandatory; the reference implementation's tiny query
    # workspace setting must not silently turn TPU matmuls into vector work.
    alignment = 128 // math.gcd(heads, 128)
    query_tile = math.ceil(query_tile / alignment) * alignment
    key_tile = math.ceil(key_tile / 128) * 128
    padding = -length % query_tile
    q = _pad(q, padding)
    info = tuple(_pad(x, padding) for x in info)
    k, v = _pad(k, -k.shape[1] % key_tile), _pad(v, -v.shape[1] % key_tile)
    active = jnp.any(info[2].reshape(batch, -1, query_tile), axis=-1).reshape(-1)
    work = jnp.nonzero(active, size=active.size, fill_value=0)[0]
    return q, k, v, info, work, jnp.sum(active, dtype=jnp.int32), query_tile, key_tile


def _current_block_words(context, heads, bank_length, key_tile, block_size):
    positions, _, active, starts = (x[0] for x in context)
    first = starts + (positions - starts) // block_size * block_size
    length = min(bank_length, math.ceil((block_size + positions.size + key_tile - 1) / key_tile) * key_tile)
    begin = jnp.minimum(jnp.min(jnp.where(active, first, bank_length - 1)) // key_tile * key_tile,
                        bank_length - length)
    words = _interval_words(jnp.repeat(first, heads), jnp.repeat(positions + 1, heads),
                            jnp.repeat(active, heads), length, begin)
    return words, _tile_schedule(words, key_tile), begin, length


def _interval_words(low, high, active, length, begin=0):
    low, high = low[None, :], high[None, :]
    base = begin + jnp.arange(length // 32, dtype=jnp.int32)[:, None] * 32

    def prefix_bits(end):
        count = jnp.clip(end - base, 0, 32).astype(jnp.uint32)
        return jnp.where(count == 32, jnp.uint32(0xFFFFFFFF), (jnp.uint32(1) << count) - jnp.uint32(1))

    # Each row is a contiguous causal interval. Construct words directly,
    # avoiding a query_rows * block_size integer-address scatter.
    return jnp.where(active[None, :], prefix_bits(high) & ~prefix_bits(low), 0)


def _routed_window(k, v, query, context, metadata, route, key_tile, block_size):
    if block_size is not None:
        words, schedule, begin, length = _current_block_words(
            context, query.shape[2], k.shape[0], key_tile, block_size,
        )
        return (lax.dynamic_slice_in_dim(k, begin, length), lax.dynamic_slice_in_dim(v, begin, length),
                words, schedule, begin)
    indices = route(lax.stop_gradient(query), context, metadata)
    indices = jnp.where(context[2][:, :, None, None], indices, -1).reshape(query.shape[1] * query.shape[2], -1)
    words, schedule = pack_routes(indices, k.shape[0], key_tile)
    return k, v, words, schedule, jnp.int32(0)


def _local_forward(q, k, v, info, meta, *, route, scale, query_tile, key_tile, window_size=None):
    original_length = q.shape[1]
    q, k, v, info, work, count, query_tile, key_tile = _local_workspace(q, k, v, info, query_tile, key_tile)
    _, length, heads, dim = q.shape
    tiles = length // query_tile

    def body(i, state):
        b, tile = work[i] // tiles, work[i] % tiles
        qt = lax.dynamic_slice(q, (b, tile * query_tile, 0, 0), (1, query_tile, heads, dim))
        ctx = tuple(lax.dynamic_slice(x, (b, tile * query_tile), (1, query_tile)) for x in info)
        mt = tuple(lax.dynamic_slice_in_dim(x, b, 1, axis=0) for x in meta)
        kt, vt, words, schedule, _ = _routed_window(k[b], v[b], qt, ctx, mt, route, key_tile, window_size)
        scaled = (qt * jnp.asarray(scale, q.dtype)).reshape(-1, dim)
        out, lse = flash_forward(scaled, kt, vt, words, schedule, key_tile=key_tile)
        return (
            lax.dynamic_update_slice(state[0], out.reshape(1, query_tile, heads, dim), (b, tile * query_tile, 0, 0)),
            lax.dynamic_update_slice(state[1], lse.reshape(1, query_tile, heads), (b, tile * query_tile, 0)),
        )

    out, lse = lax.fori_loop(0, count, body, (jnp.zeros(q.shape, jnp.float32), jnp.zeros(q.shape[:-1], jnp.float32)))
    return out[:, :original_length], lse[:, :original_length]


def _local_backward(q, k, v, info, meta, out, lse, do, *, route, scale, query_tile, key_tile,
                    window_size=None):
    original_length, bank_length = q.shape[1], k.shape[1]
    q, k, v, info, work, count, query_tile, key_tile = _local_workspace(q, k, v, info, query_tile, key_tile)
    _, length, heads, dim = q.shape
    tiles = length // query_tile
    out, lse, do = (_pad(x, length - original_length) for x in (out, lse, do))
    delta = jnp.sum(out * do.astype(jnp.float32), axis=-1)
    quantized_scale = jnp.asarray(scale, q.dtype)

    def body(i, state):
        b, tile = work[i] // tiles, work[i] % tiles
        qt = lax.dynamic_slice(q, (b, tile * query_tile, 0, 0), (1, query_tile, heads, dim))
        ctx = tuple(lax.dynamic_slice(x, (b, tile * query_tile), (1, query_tile)) for x in info)
        mt = tuple(lax.dynamic_slice_in_dim(x, b, 1, axis=0) for x in meta)
        kt, vt, words, schedule, begin = _routed_window(
            k[b], v[b], qt, ctx, mt, route, key_tile, window_size,
        )
        dot = lax.dynamic_slice(do, (b, tile * query_tile, 0, 0), (1, query_tile, heads, dim)).reshape(-1, dim)
        stats = tuple(lax.dynamic_slice(x, (b, tile * query_tile, 0), (1, query_tile, heads)).reshape(-1)
                      for x in (lse, delta))
        dqt, dkt, dvt = flash_backward(
            (qt * quantized_scale).reshape(-1, dim), kt, vt, words, schedule, dot, *stats,
            lax.dynamic_slice_in_dim(state[1][b], begin, kt.shape[0], axis=0),
            lax.dynamic_slice_in_dim(state[2][b], begin, kt.shape[0], axis=0), key_tile=key_tile,
        )
        return (
            lax.dynamic_update_slice(state[0], (dqt * quantized_scale).reshape(1, query_tile, heads, dim),
                                     (b, tile * query_tile, 0, 0)),
            lax.dynamic_update_slice(state[1], dkt[None], (b, begin, 0)),
            lax.dynamic_update_slice(state[2], dvt[None], (b, begin, 0)),
        )

    grads = lax.fori_loop(0, count, body, tuple(jnp.zeros(x.shape, jnp.float32) for x in (q, k, v)))
    return grads[0][:, :original_length], grads[1][:, :bank_length], grads[2][:, :bank_length]


def sparse_flash_attention(query, key, value, query_info, metadata, *, route, metadata_specs,
                           scale, query_tile, key_tile, mesh, moba_config=None):
    """Distributed custom VJP; TP/SP reductions happen in FP32 before casting."""
    kwargs = dict(route=route, scale=scale, query_tile=query_tile, key_tile=key_tile)
    forward = functools.partial(_local_forward, **kwargs)
    backward = functools.partial(_local_backward, **kwargs)
    if moba_config is not None:
        from .grouped_moba import local_moba_backward, local_moba_forward
        kwargs.update(block_size=moba_config[0], top_k=moba_config[1])
        forward = functools.partial(local_moba_forward, **kwargs)
        backward = functools.partial(local_moba_backward, **kwargs)
    if mesh is not None:
        def axis(*names):
            names = tuple(n for n in names if n in mesh.axis_names and mesh.shape[n] > 1)
            return (names[0] if len(names) == 1 else names) or None
        batch, sequence, heads = axis("dp", "fsdp"), axis("sp"), axis("tp")
        q_spec, kv_spec = P(batch, sequence, heads, None), P(batch, None, None)
        stat_spec, info_spec = P(batch, sequence, heads), (P(batch, sequence),) * 4
        meta_spec = tuple(P(batch, None, heads if per_head else None) for per_head in metadata_specs)
        specs = (q_spec, kv_spec, kv_spec, info_spec, meta_spec)
        local_backward = backward

        def reduce_backward(*args):
            dq, dk, dv = local_backward(*args)
            reduction_axes = tuple(n for n in ("tp", "sp") if n in mesh.axis_names and mesh.shape[n] > 1)
            if reduction_axes:
                dk, dv = lax.psum((dk, dv), reduction_axes)
            return dq, dk, dv

        forward = jax.shard_map(forward, mesh=mesh, in_specs=specs, out_specs=(q_spec, stat_spec), check_vma=False)
        backward = jax.shard_map(
            reduce_backward, mesh=mesh, in_specs=(*specs, q_spec, stat_spec, q_spec),
            out_specs=(q_spec, kv_spec, kv_spec), check_vma=False,
        )

    @jax.custom_vjp
    def apply(q, k, v, info, meta):
        return forward(q, k, v, info, meta)[0].astype(q.dtype)

    def fwd(q, k, v, info, meta):
        out, lse = forward(q, k, v, info, meta)
        return out.astype(q.dtype), (q, k, v, info, meta, out, lse)

    def bwd(residual, do):
        q, k, v, *_ = residual
        grads = backward(*residual, do)
        return *(g.astype(x.dtype) for g, x in zip(grads, (q, k, v), strict=True)), None, None

    apply.defvjp(fwd, bwd)
    return apply(query, key, value, query_info, jax.tree.map(lax.stop_gradient, metadata))
