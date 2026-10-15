"""Stream sample-relative causal blocks without a per-query XLA loop.

The current block is a contiguous interval per query. Its bounds replace the
bit-packed route workspace. Forward query tiles are independent; backward
tiles run in order and accumulate shared K/V gradients without atomics.
"""

from __future__ import annotations

import functools

import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from .indexed_attention import _pad
from .sparse_flash_attention import (
    _NT,
    _TN,
    _compact_row_stat,
    _expand_row_stat,
    _local_workspace,
    _shape,
)


def _copies(first, index, slot, k, v, kb, vb, ks, vs, key_tile):
    window = pl.ds(first + index * key_tile, key_tile)
    return (pltpu.make_async_copy(k.at[window, :], kb.at[slot], ks.at[slot]),
            pltpu.make_async_copy(v.at[window, :], vb.at[slot], vs.at[slot]))


def _forward(schedule, q, low, high, k, v, out, lse, m, den, acc, kb, vb, ks, vs, *, key_tile):
    rows = q.shape[0]
    first, count = schedule[pl.program_id(0), 2 * pl.program_id(1)] * key_tile, schedule[
        pl.program_id(0), 2 * pl.program_id(1) + 1]
    lower, upper = _expand_row_stat(low[...], key_tile), _expand_row_stat(high[...], key_tile)
    m[...] = jnp.full((rows, 128), -1e30, jnp.float32)
    den[...] = jnp.zeros((rows, 128), jnp.float32)
    acc[...] = jnp.zeros((rows, 128), jnp.float32)

    @pl.when(count > 0)
    def start():
        for copy in _copies(first, 0, 0, k, v, kb, vb, ks, vs, key_tile):
            copy.start()

    def body(index, _):
        slot = index % 2
        for copy in _copies(first, index, slot, k, v, kb, vb, ks, vs, key_tile):
            copy.wait()

        @pl.when(index + 1 < count)
        def prefetch():
            for copy in _copies(first, index + 1, 1 - slot, k, v, kb, vb, ks, vs, key_tile):
                copy.start()

        positions = first + index * key_tile + lax.broadcasted_iota(jnp.int32, (rows, key_tile), 1)
        valid = (positions >= lower) & (positions < upper)
        scores = lax.dot_general(q[...], kb[slot], _NT, preferred_element_type=jnp.float32)
        scores = jnp.where(valid, scores, -1e30)
        next_m = jnp.maximum(m[...], jnp.broadcast_to(jnp.max(scores, axis=1)[:, None], (rows, 128)))
        correction = jnp.exp(m[...] - next_m)
        p = jnp.where(valid, jnp.exp(scores - jnp.tile(next_m, (1, key_tile // 128))), 0)
        den[...] = den[...] * correction + jnp.broadcast_to(jnp.sum(p, axis=1)[:, None], (rows, 128))
        acc[...] = acc[...] * correction + lax.dot(p.astype(v.dtype), vb[slot],
                                                   preferred_element_type=jnp.float32)
        m[...] = next_m

    lax.fori_loop(0, count, body, None)
    safe = jnp.maximum(den[...], 1e-30)
    out[...] = acc[...] / safe
    lse[...] = _compact_row_stat(jnp.where(den[...] > 0, m[...] + jnp.log(safe), 0))


def _backward(schedule, q, low, high, k, v, do, lse, delta, dk_in, dv_in,
              dq, dk, dv, acc, kb, vb, dkb, dvb, ks, vs, dks, dvs, *, key_tile):
    del dk_in, dv_in
    rows = q.shape[0]
    first, count = schedule[pl.program_id(0), 2 * pl.program_id(1)] * key_tile, schedule[
        pl.program_id(0), 2 * pl.program_id(1) + 1]
    lower, upper = _expand_row_stat(low[...], key_tile), _expand_row_stat(high[...], key_tile)
    normalizer, row_delta = _expand_row_stat(lse[...], key_tile), _expand_row_stat(delta[...], key_tile)
    acc[...] = jnp.zeros((rows, 128), jnp.float32)

    @pl.when(count > 0)
    def start():
        for copy in _copies(first, 0, 0, k, v, kb, vb, ks, vs, key_tile):
            copy.start()

    def body(index, _):
        slot = index % 2
        for copy in _copies(first, index, slot, k, v, kb, vb, ks, vs, key_tile):
            copy.wait()

        @pl.when(index + 1 < count)
        def prefetch():
            for copy in _copies(first, index + 1, 1 - slot, k, v, kb, vb, ks, vs, key_tile):
                copy.start()

        window = pl.ds(first + index * key_tile, key_tile)
        load_k = pltpu.make_async_copy(dk.at[window, :], dkb, dks)
        load_v = pltpu.make_async_copy(dv.at[window, :], dvb, dvs)
        load_k.start()
        load_v.start()
        positions = first + index * key_tile + lax.broadcasted_iota(jnp.int32, (rows, key_tile), 1)
        valid = (positions >= lower) & (positions < upper)
        scores = lax.dot_general(q[...], kb[slot], _NT, preferred_element_type=jnp.float32)
        p = jnp.exp(jnp.where(valid, scores - normalizer, -1e30))
        dp = lax.dot_general(do[...], vb[slot], _NT, preferred_element_type=jnp.float32)
        ds = p * (dp - row_delta)
        acc[...] += lax.dot(ds.astype(q.dtype), kb[slot], preferred_element_type=jnp.float32)
        d_key = lax.dot_general(ds.astype(q.dtype), q[...], _TN, preferred_element_type=jnp.float32)
        d_value = lax.dot_general(p.astype(do.dtype), do[...], _TN, preferred_element_type=jnp.float32)
        load_k.wait()
        load_v.wait()
        dkb[...] += d_key
        dvb[...] += d_value
        store_k = pltpu.make_async_copy(dkb, dk.at[window, :], dks)
        store_v = pltpu.make_async_copy(dvb, dv.at[window, :], dvs)
        store_k.start()
        store_v.start()
        store_k.wait()
        store_v.wait()

    lax.fori_loop(0, count, body, None)
    dq[...] = acc[...]


def _workspace(q, k, v, info, query_tile, key_tile, block_size, scale):
    q, k, v, info, _, _, query_tile, key_tile = _local_workspace(q, k, v, info, query_tile, key_tile)
    batch, length, heads, _ = q.shape
    positions, _, active, starts = info
    low = starts + (positions - starts) // block_size * block_size
    high = jnp.where(active, positions + 1, 0)
    low = jnp.where(active, low, 0)
    tile_active = active.reshape(batch, -1, query_tile)
    first = jnp.min(jnp.where(tile_active, low.reshape(batch, -1, query_tile), k.shape[1]), axis=-1)
    end = jnp.max(high.reshape(batch, -1, query_tile), axis=-1)
    first = jnp.where(jnp.any(tile_active, axis=-1), first // key_tile * key_tile, 0)
    count = (end + key_tile - 1) // key_tile - first // key_tile
    bank_offset = jnp.arange(batch, dtype=jnp.int32)[:, None] * k.shape[1]
    # Store tile IDs: multiplication inside the kernel lets Mosaic prove DMA
    # alignment even though the schedule values arrive through scalar memory.
    schedule = jnp.stack(((first + bank_offset) // key_tile, count), axis=-1).reshape(batch, -1)
    bounds = tuple(jnp.repeat(x + bank_offset, heads, axis=1).reshape(-1) for x in (low, high))
    scaled = (q * jnp.asarray(scale, q.dtype)).reshape(-1, 128)
    tiles, rows = length // query_tile, query_tile * heads
    q_spec = pl.BlockSpec((rows, 128), lambda b, t, *_: (b * tiles + t, 0))
    stat_spec = pl.BlockSpec((rows,), lambda b, t, *_: (b * tiles + t,))
    return q, k, v, schedule, bounds, scaled, (batch, tiles), rows, key_tile, q_spec, stat_spec


def current_block_forward(q, k, v, info, *, scale, query_tile, key_tile, block_size):
    original_length = q.shape[1]
    q, k, v, schedule, bounds, scaled, grid, rows, key_tile, q_spec, stat_spec = _workspace(
        q, k, v, info, query_tile, key_tile, block_size, scale)
    hbm = pl.BlockSpec(memory_space=pl.ANY)
    out, lse = pl.pallas_call(
        functools.partial(_forward, key_tile=key_tile),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=1, grid=grid, in_specs=(q_spec, stat_spec, stat_spec, hbm, hbm),
            out_specs=(q_spec, stat_spec), scratch_shapes=(
                *(pltpu.VMEM((rows, 128), jnp.float32) for _ in range(3)),
                pltpu.VMEM((2, key_tile, 128), q.dtype), pltpu.VMEM((2, key_tile, 128), v.dtype),
                pltpu.SemaphoreType.DMA((2,)), pltpu.SemaphoreType.DMA((2,)),
            ),
        ),
        out_shape=(_shape(scaled.shape, jnp.float32, q), _shape((scaled.shape[0],), jnp.float32, q)),
        compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel", "parallel")),
        name="moba_current_block_fwd",
    )(schedule, scaled, *bounds, k.reshape(-1, 128), v.reshape(-1, 128))
    return out.reshape(q.shape)[:, :original_length], lse.reshape(q.shape[:-1])[:, :original_length]


def current_block_backward(q, k, v, info, out, lse, do, *, scale, query_tile, key_tile, block_size):
    original_length, bank_length = q.shape[1], k.shape[1]
    q, k, v, schedule, bounds, scaled, grid, rows, key_tile, q_spec, stat_spec = _workspace(
        q, k, v, info, query_tile, key_tile, block_size, scale)
    out, lse, do = (_pad(x, q.shape[1] - original_length) for x in (out, lse, do))
    delta = jnp.sum(out * do.astype(jnp.float32), axis=-1).reshape(-1)
    k_flat, v_flat = k.reshape(-1, 128), v.reshape(-1, 128)
    zeros = tuple(jnp.zeros(x.shape, jnp.float32) for x in (k_flat, v_flat))
    hbm = pl.BlockSpec(memory_space=pl.ANY)
    dq, dk, dv = pl.pallas_call(
        functools.partial(_backward, key_tile=key_tile),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=1, grid=grid,
            in_specs=(q_spec, stat_spec, stat_spec, hbm, hbm, q_spec, stat_spec, stat_spec, hbm, hbm),
            out_specs=(q_spec, hbm, hbm), scratch_shapes=(
                pltpu.VMEM((rows, 128), jnp.float32),
                pltpu.VMEM((2, key_tile, 128), q.dtype), pltpu.VMEM((2, key_tile, 128), v.dtype),
                pltpu.VMEM((key_tile, 128), jnp.float32), pltpu.VMEM((key_tile, 128), jnp.float32),
                pltpu.SemaphoreType.DMA((2,)), pltpu.SemaphoreType.DMA((2,)),
                pltpu.SemaphoreType.DMA, pltpu.SemaphoreType.DMA,
            ),
        ),
        out_shape=tuple(_shape(x.shape, jnp.float32, q) for x in (scaled, k_flat, v_flat)),
        input_output_aliases={9: 1, 10: 2},
        # K/V gradients share aliased HBM refs across the whole call. Keep the
        # entire reduction grid ordered, including the batch axis: partitioning
        # that alias across TPU cores can replay gradient accumulation.
        compiler_params=pltpu.CompilerParams(dimension_semantics=("arbitrary", "arbitrary")),
        name="moba_current_block_bwd",
    )(schedule, scaled, *bounds, k_flat, v_flat, do.reshape(-1, 128), lse.reshape(-1), delta, *zeros)
    dq = (dq * jnp.asarray(scale, q.dtype)).reshape(q.shape)[:, :original_length]
    return dq, dk.reshape(k.shape)[:, :bank_length], dv.reshape(v.shape)[:, :bank_length]
