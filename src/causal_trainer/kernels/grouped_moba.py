"""Rank-streamed, KV-grouped MoBA kernels for MQA on TPU.

Each selected rank is grouped independently. Queries are permuted once per
rank; no top-k-expanded Q/O activation is retained. KV blocks remain shared,
and KV-owned backward programs reduce all routed queries before a single
FP32 gradient write. Per-query routing is unchanged by the execution grouping.
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
from .sparse_flash_attention import _NT, _TN, _compact_row_stat, _expand_row_stat, _shape


def group_queries(groups, num_groups, *, rows=128):
    """Bounded padded counting layout, retaining a reversible row permutation."""
    size = groups.size
    groups = jnp.where(groups >= 0, groups, num_groups).reshape(-1)
    order = jnp.argsort(groups, stable=True)
    sorted_groups = groups[order]
    # The sorted stream already contains the histogram. Binary-search its
    # boundaries instead of a highly colliding scatter-add into a few bins.
    offsets = jnp.searchsorted(sorted_groups, jnp.arange(num_groups + 2), side="left")
    counts = jnp.diff(offsets).astype(jnp.int32)
    padded_counts = (counts + rows - 1) // rows * rows
    padded_offsets = jnp.concatenate((jnp.zeros(1, jnp.int32), jnp.cumsum(padded_counts)))
    destinations = jnp.arange(size) - offsets[sorted_groups] + padded_offsets[sorted_groups]
    capacity = (math.ceil(size / rows) + num_groups + 1) * rows
    permutation = jnp.full((capacity,), -1, jnp.int32).at[destinations].set(
        order, indices_are_sorted=True, unique_indices=True,
    )
    inverse = jnp.empty((size,), jnp.int32).at[order].set(destinations, unique_indices=True)
    # Invalid assignments are deliberately not executed.
    return permutation, inverse, counts.at[-1].set(0), padded_offsets[:-1]


def _group_for_row(offsets, row):
    # Scalar binary search keeps SMEM O(number of KV blocks), independent of
    # query length. Prefetching a sequence-wide tile->group map can exhaust
    # TPUv4 SMEM at small TP/SP degrees.
    lower, upper = jnp.int32(0), jnp.int32(offsets.shape[0])
    for _ in range(math.ceil(math.log2(offsets.shape[0] + 1))):
        middle = (lower + upper) // 2
        past = offsets[jnp.minimum(middle, offsets.shape[0] - 1)] <= row
        lower = jnp.where(past, middle + 1, lower)
        upper = jnp.where(past, upper, middle)
    return jnp.minimum(lower - 1, offsets.shape[0] - 1)


def _kv_copies(k, v, kb, vb, ks, vs, start, index, slot, key_tile):
    return (
        pltpu.make_async_copy(k.at[pl.ds(start + index * key_tile, key_tile), :], kb.at[slot], ks.at[slot]),
        pltpu.make_async_copy(v.at[pl.ds(start + index * key_tile, key_tile), :], vb.at[slot], vs.at[slot]),
    )


def _block_window(start, block_size, key_tile):
    # Packed samples can begin at any token. DMA addresses must remain aligned
    # to physical tiles; the two boundary tiles are masked after loading.
    aligned = start // key_tile * key_tile
    count = block_size // key_tile + (start != aligned).astype(jnp.int32)
    return aligned, count


def _query_kernel(
    starts, counts, offsets, q, do, lse, delta, k, v, out, statistics,
    m, den, acc, kb, vb, ks, vs, *, block_size, key_tile, backward,
):
    rows = q.shape[0]
    group = _group_for_row(offsets, pl.program_id(0) * rows)
    row_positions = pl.program_id(0) * rows + lax.broadcasted_iota(jnp.int32, (rows, key_tile), 0)
    row_valid = row_positions < offsets[group] + counts[group]
    acc[...] = jnp.zeros((rows, 128), jnp.float32)
    m[...] = jnp.full((rows, 128), -1e30, jnp.float32)
    den[...] = jnp.zeros((rows, 128), jnp.float32)
    if backward:
        log_normalizer = _expand_row_stat(lse[...], key_tile)
        row_delta = _expand_row_stat(delta[...], key_tile)

    @pl.when(counts[group] > 0)
    def compute():
        first, tile_count = _block_window(starts[group], block_size, key_tile)
        for copy in _kv_copies(k, v, kb, vb, ks, vs, first, 0, 0, key_tile):
            copy.start()

        def body(index, _):
            slot = index % 2
            for copy in _kv_copies(k, v, kb, vb, ks, vs, first, index, slot, key_tile):
                copy.wait()

            @pl.when(index + 1 < tile_count)
            def prefetch():
                for copy in _kv_copies(k, v, kb, vb, ks, vs, first, index + 1, 1 - slot, key_tile):
                    copy.start()

            scores = lax.dot_general(q[...], kb[slot], _NT, preferred_element_type=jnp.float32)
            positions = first + index * key_tile + lax.broadcasted_iota(jnp.int32, (rows, key_tile), 1)
            key_valid = (positions >= starts[group]) & (positions < starts[group] + block_size)
            valid = row_valid & key_valid
            if backward:
                p = jnp.exp(jnp.where(valid, scores - log_normalizer, -1e30))
                dp = lax.dot_general(do[...], vb[slot], _NT, preferred_element_type=jnp.float32)
                ds = p * (dp - row_delta)
                acc[...] += lax.dot(ds.astype(q.dtype), kb[slot], preferred_element_type=jnp.float32)
            else:
                scores = jnp.where(valid, scores, -1e30)
                next_m = jnp.maximum(m[...], jnp.broadcast_to(jnp.max(scores, axis=1)[:, None], (rows, 128)))
                correction = jnp.exp(m[...] - next_m)
                p = jnp.where(valid, jnp.exp(scores - jnp.tile(next_m, (1, key_tile // 128))), 0)
                den[...] = den[...] * correction + jnp.broadcast_to(jnp.sum(p, axis=1)[:, None], (rows, 128))
                acc[...] = acc[...] * correction + lax.dot(p.astype(v.dtype), vb[slot],
                                                           preferred_element_type=jnp.float32)
                m[...] = next_m

        lax.fori_loop(0, tile_count, body, None)

    if backward:
        out[...] = acc[...]
        statistics[...] = jnp.zeros((rows,), jnp.float32)
    else:
        out[...] = acc[...] / jnp.maximum(den[...], 1e-30)
        statistics[...] = _compact_row_stat(
            jnp.where(den[...] > 0, m[...] + jnp.log(jnp.maximum(den[...], 1e-30)), -jnp.inf),
        )


def grouped_query(q, k, v, starts, layout, *, block_size, key_tile, query_rows=128, do=None, lse=None, delta=None):
    permutation, _, counts, offsets = layout
    q = q[jnp.maximum(permutation, 0)]
    backward = do is not None
    do = do[jnp.maximum(permutation, 0)] if backward else q
    lse = lse[jnp.maximum(permutation, 0)] if backward else jnp.zeros((q.shape[0],), jnp.float32)
    delta = delta[jnp.maximum(permutation, 0)] if backward else lse
    k, v = (_pad(x, -x.shape[0] % key_tile, axis=0) for x in (k, v))
    starts = jnp.concatenate((starts, jnp.zeros(1, jnp.int32)))
    query_spec = pl.BlockSpec((query_rows, 128), lambda i, *_: (i, 0))
    stat_spec = pl.BlockSpec((query_rows,), lambda i, *_: (i,))
    hbm = pl.BlockSpec(memory_space=pl.ANY)
    return pl.pallas_call(
        functools.partial(_query_kernel, block_size=block_size, key_tile=key_tile, backward=backward),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=3, grid=(q.shape[0] // query_rows,),
            in_specs=(query_spec, query_spec, stat_spec, stat_spec, hbm, hbm),
            out_specs=(query_spec, stat_spec), scratch_shapes=(
                pltpu.VMEM((query_rows, 128), jnp.float32), pltpu.VMEM((query_rows, 128), jnp.float32),
                pltpu.VMEM((query_rows, 128), jnp.float32),
                pltpu.VMEM((2, key_tile, 128), q.dtype), pltpu.VMEM((2, key_tile, 128), v.dtype),
                pltpu.SemaphoreType.DMA((2,)), pltpu.SemaphoreType.DMA((2,)),
            ),
        ),
        out_shape=(_shape(q.shape, jnp.float32, q), _shape((q.shape[0],), jnp.float32, q)),
        compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel",)),
        name="moba_grouped_dq" if backward else "moba_grouped_fwd",
    )(starts, counts, offsets, q, do, lse, delta, k, v)


def _query_copies(q, do, lse, delta, qb, dob, lb, db, qs, dos, ls, ds, tile, slot):
    rows = qb.shape[1]
    return (
        pltpu.make_async_copy(q.at[pl.ds(tile * rows, rows), :], qb.at[slot], qs.at[slot]),
        pltpu.make_async_copy(do.at[pl.ds(tile * rows, rows), :], dob.at[slot], dos.at[slot]),
        pltpu.make_async_copy(lse.at[pl.ds(tile * rows, rows)], lb.at[slot], ls.at[slot]),
        pltpu.make_async_copy(delta.at[pl.ds(tile * rows, rows)], db.at[slot], ds.at[slot]),
    )


def _kv_kernel(
    starts, counts, offsets, q, do, lse, delta, k, v, dk_in, dv_in, dk, dv,
    kb, vb, dkb, dvb, qb, dob, lb, db, ks, vs, dks, dvs, qs, dos, ls, ds, *, key_tile, block_size,
):
    del dk_in, dv_in
    rows = qb.shape[1]
    begin = pl.program_id(0) * key_tile
    key_slice = pl.ds(begin, key_tile)
    copies = (
        pltpu.make_async_copy(k.at[key_slice, :], kb, ks),
        pltpu.make_async_copy(v.at[key_slice, :], vb, vs),
        pltpu.make_async_copy(dk.at[key_slice, :], dkb, dks),
        pltpu.make_async_copy(dv.at[key_slice, :], dvb, dvs),
    )
    for copy in copies:
        copy.start()
    for copy in copies:
        copy.wait()
    last_group = _group_for_row(starts, begin + key_tile - 1)

    def group_body(neighbor, _):
        candidate = last_group - neighbor
        group = jnp.maximum(candidate, 0)

        @pl.when((candidate >= 0) & (counts[group] > 0) & (starts[group] + block_size > begin))
        def compute():
            first = offsets[group] // rows
            count = (counts[group] + rows - 1) // rows
            for copy in _query_copies(q, do, lse, delta, qb, dob, lb, db, qs, dos, ls, ds, first, 0):
                copy.start()

            def body(index, _):
                slot = index % 2
                for copy in _query_copies(q, do, lse, delta, qb, dob, lb, db, qs, dos, ls, ds, first + index, slot):
                    copy.wait()

                @pl.when(index + 1 < count)
                def prefetch():
                    for copy in _query_copies(q, do, lse, delta, qb, dob, lb, db, qs, dos, ls, ds,
                                              first + index + 1, 1 - slot):
                        copy.start()

                query_valid = index * rows + lax.broadcasted_iota(jnp.int32, (rows, key_tile), 0) < counts[group]
                positions = begin + lax.broadcasted_iota(jnp.int32, (rows, key_tile), 1)
                key_valid = (positions >= starts[group]) & (positions < starts[group] + block_size)
                valid = query_valid & key_valid
                logits = lax.dot_general(qb[slot], kb[...], _NT, preferred_element_type=jnp.float32)
                p = jnp.exp(jnp.where(valid, logits - _expand_row_stat(lb[slot], key_tile), -1e30))
                dp = lax.dot_general(dob[slot], vb[...], _NT, preferred_element_type=jnp.float32)
                dlogits = p * (dp - _expand_row_stat(db[slot], key_tile))
                dkb[...] += lax.dot_general(dlogits.astype(q.dtype), qb[slot], _TN, preferred_element_type=jnp.float32)
                dvb[...] += lax.dot_general(p.astype(do.dtype), dob[slot], _TN, preferred_element_type=jnp.float32)

            lax.fori_loop(0, count, body, None)

    # Disjoint logical blocks have length >= key_tile. At most two can
    # intersect a physical tile. Owning physical tiles eliminates boundary
    # write races and lets Mosaic schedule both cores in parallel.
    lax.fori_loop(0, 2, group_body, None)
    store_k = pltpu.make_async_copy(dkb, dk.at[key_slice, :], dks)
    store_v = pltpu.make_async_copy(dvb, dv.at[key_slice, :], dvs)
    store_k.start()
    store_v.start()
    store_k.wait()
    store_v.wait()


def grouped_kv(q, k, v, starts, layout, do, lse, delta, dk, dv, *, block_size, key_tile, query_rows=128):
    permutation, _, counts, offsets = layout
    index = jnp.maximum(permutation, 0)
    q, do, lse, delta = (x[index] for x in (q, do, lse, delta))
    original_length = k.shape[0]
    k, v, dk, dv = (_pad(x, -x.shape[0] % key_tile, axis=0) for x in (k, v, dk, dv))
    starts = jnp.where(starts >= 0, starts, k.shape[0] + block_size)
    hbm = pl.BlockSpec(memory_space=pl.ANY)
    result = pl.pallas_call(
        functools.partial(_kv_kernel, key_tile=key_tile, block_size=block_size),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=3, grid=(k.shape[0] // key_tile,), in_specs=(hbm,) * 8,
            out_specs=(hbm, hbm), scratch_shapes=(
                pltpu.VMEM((key_tile, 128), k.dtype), pltpu.VMEM((key_tile, 128), v.dtype),
                pltpu.VMEM((key_tile, 128), jnp.float32), pltpu.VMEM((key_tile, 128), jnp.float32),
                pltpu.VMEM((2, query_rows, 128), q.dtype), pltpu.VMEM((2, query_rows, 128), do.dtype),
                pltpu.VMEM((2, query_rows), jnp.float32), pltpu.VMEM((2, query_rows), jnp.float32),
                pltpu.SemaphoreType.DMA, pltpu.SemaphoreType.DMA,
                pltpu.SemaphoreType.DMA, pltpu.SemaphoreType.DMA,
                pltpu.SemaphoreType.DMA((2,)), pltpu.SemaphoreType.DMA((2,)),
                pltpu.SemaphoreType.DMA((2,)), pltpu.SemaphoreType.DMA((2,)),
            ),
        ),
        out_shape=(_shape(dk.shape, jnp.float32, q), _shape(dv.shape, jnp.float32, q)),
        input_output_aliases={9: 0, 10: 1},
        compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel",)),
        name="moba_grouped_dkv",
    )(starts, counts, offsets, q, do, lse, delta, k, v, dk, dv)
    return tuple(x[:original_length] for x in result)


def _pool_kernel(ends, key, mean, acc, buffer, semaphore, *, block_size, sequence, key_tile):
    batch, block = pl.program_id(0), pl.program_id(1)
    end = ends[batch, block]
    acc[...] = jnp.zeros((128,), jnp.float32)

    @pl.when(end >= 0)
    def compute():
        begin = batch * sequence + end + 1 - block_size
        first, tile_count = _block_window(begin, block_size, key_tile)

        def copy(index, slot):
            return pltpu.make_async_copy(
                key.at[pl.ds(first + index * key_tile, key_tile), :], buffer.at[slot], semaphore.at[slot],
            )

        copy(0, 0).start()

        def body(index, _):
            slot = index % 2
            copy(index, slot).wait()

            @pl.when(index + 1 < tile_count)
            def prefetch():
                copy(index + 1, 1 - slot).start()

            positions = first + index * key_tile + lax.broadcasted_iota(jnp.int32, (key_tile, 128), 0)
            valid = (positions >= begin) & (positions < begin + block_size)
            acc[...] += jnp.sum(jnp.where(valid, buffer[slot].astype(jnp.float32), 0), axis=0)

        lax.fori_loop(0, tile_count, body, None)

    mean[...] = acc[...] / block_size


def pool_blocks(key, ends, block_size):
    tile = math.gcd(block_size, 512)
    key = _pad(key, -key.shape[1] % tile)
    spec = pl.BlockSpec((128,), lambda b, c, *_: (b * ends.shape[1] + c,))
    result = pl.pallas_call(
        functools.partial(_pool_kernel, block_size=block_size, sequence=key.shape[1], key_tile=tile),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=1, grid=ends.shape, in_specs=(pl.BlockSpec(memory_space=pl.ANY),),
            out_specs=spec, scratch_shapes=(pltpu.VMEM((128,), jnp.float32),
                                           pltpu.VMEM((2, tile, 128), key.dtype), pltpu.SemaphoreType.DMA((2,))),
        ),
        out_shape=_shape((ends.size * 128,), jnp.float32, key),
        compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel", "parallel")),
        name="moba_block_pool",
    )(ends, key.reshape(-1, 128))
    return result.reshape(*ends.shape, 128)


def compact_blocks(key, segments, starts, block_size, mesh=None):
    """At most floor(N / block_size) complete sample-relative blocks."""
    positions = jnp.arange(key.shape[1])[None]
    complete = (segments > 0) & ((positions - starts + 1) % block_size == 0)
    capacity = key.shape[1] // block_size
    ends = jax.vmap(lambda mask: jnp.nonzero(mask, size=capacity, fill_value=-1)[0])(complete)
    safe = jnp.maximum(ends, 0)
    batch = jnp.arange(key.shape[0])[:, None]
    pool = functools.partial(pool_blocks, block_size=block_size)
    if mesh is not None:
        axes = tuple(n for n in ("dp", "fsdp") if n in mesh.axis_names and mesh.shape[n] > 1)
        batch_axis = (axes[0] if len(axes) == 1 else axes) or None
        pool = jax.shard_map(pool, mesh=mesh, in_specs=(P(batch_axis, None, None), P(batch_axis, None)),
                             out_specs=P(batch_axis, None, None), check_vma=False)
    means = pool(lax.stop_gradient(key), ends)
    block_segments = jnp.where(ends >= 0, segments[batch, safe], 0)
    return means, (ends - block_size + 1)[..., None], block_segments[..., None]


def block_routes(q, info, meta, *, top_k, block_size, query_tile=128):
    """Compact integer routing, never materializing N*H*block_count logits."""
    means, starts, segments = meta
    batch, length, heads, _ = q.shape
    ranks = min(top_k - 1, means.shape[1])
    if not ranks:
        return jnp.empty((0, batch, length, heads), jnp.int32)
    padded_q = _pad(q, -length % query_tile)
    padded_info = tuple(_pad(x, -length % query_tile) for x in info)
    result = jnp.full((*padded_q.shape[:3], ranks), -1, jnp.int32)

    def body(index, result):
        qt = lax.dynamic_slice_in_dim(padded_q, index * query_tile, query_tile, axis=1)
        pos, seg, active, origin = (lax.dynamic_slice_in_dim(x, index * query_tile, query_tile, axis=1)
                                     for x in padded_info)
        current = origin + (pos - origin) // block_size * block_size
        scores = jnp.einsum("bthd,bcd->bthc", qt.astype(jnp.float32), means)
        valid = (starts[:, None, None, :, 0] + block_size <= current[:, :, None, None])
        valid &= (segments[:, None, None, :, 0] == seg[:, :, None, None]) & active[:, :, None, None]
        scores = jnp.where(valid & (segments[:, None, None, :, 0] > 0), scores, -jnp.inf)
        best, ids = lax.top_k(scores, ranks)
        ids = jnp.where(jnp.isfinite(best), ids, -1)
        return lax.dynamic_update_slice_in_dim(result, ids, index * query_tile, axis=1)

    result = lax.fori_loop(0, padded_q.shape[1] // query_tile, body, result)
    return jnp.moveaxis(result[:, :length], -1, 0)


def local_moba_forward(q, k, v, info, meta, *, route, scale, query_tile, key_tile, block_size, top_k):
    from .sparse_flash_attention import _local_forward
    out, lse = _local_forward(q, k, v, info, (), route=route, scale=scale,
                              query_tile=query_tile, key_tile=key_tile, window_size=block_size)
    routes = block_routes(q, info, meta, top_k=min(top_k, math.ceil(k.shape[1] / block_size)), block_size=block_size)
    scaled = (q * jnp.asarray(scale, q.dtype)).reshape(q.shape[0], -1, 128)
    key_tile = math.gcd(block_size, max(128, key_tile // 128 * 128))
    query_rows = 512 if scaled.shape[1] >= 4096 else 128

    def body(index, state):
        rank, b = index // q.shape[0], index % q.shape[0]
        groups = routes[rank, b].reshape(-1)
        layout = group_queries(groups, meta[0].shape[1], rows=query_rows)
        part, part_lse = grouped_query(scaled[b], k[b], v[b], meta[1][b, :, 0], layout,
                                       block_size=block_size, key_tile=key_tile, query_rows=query_rows)
        part = part[layout[1]].reshape(out.shape[1:])
        part_lse = jnp.where(groups >= 0, part_lse[layout[1]], -jnp.inf).reshape(lse.shape[1:])
        old_out, old_lse = state[0][b], state[1][b]
        combined = jnp.logaddexp(old_lse, part_lse)
        mixed = jnp.exp(old_lse - combined)[..., None] * old_out
        mixed += jnp.exp(part_lse - combined)[..., None] * part
        return (lax.dynamic_update_slice_in_dim(state[0], mixed[None], b, axis=0),
                lax.dynamic_update_slice_in_dim(state[1], combined[None], b, axis=0))

    if routes.shape[0]:
        out, lse = lax.fori_loop(0, routes.shape[0] * q.shape[0], body, (out, lse))
    return out, lse


def local_moba_backward(q, k, v, info, meta, out, lse, do, *, route, scale,
                        query_tile, key_tile, block_size, top_k):
    from .sparse_flash_attention import _local_backward
    grads = _local_backward(q, k, v, info, (), out, lse, do, route=route, scale=scale,
                             query_tile=query_tile, key_tile=key_tile, window_size=block_size)
    routes = block_routes(q, info, meta, top_k=min(top_k, math.ceil(k.shape[1] / block_size)), block_size=block_size)
    quantized_scale = jnp.asarray(scale, q.dtype)
    scaled = (q * quantized_scale).reshape(q.shape[0], -1, 128)
    delta = jnp.sum(out * do.astype(jnp.float32), axis=-1).reshape(q.shape[0], -1)
    flat_do, flat_lse = do.reshape(q.shape[0], -1, 128), lse.reshape(q.shape[0], -1)
    key_tile = math.gcd(block_size, max(128, key_tile // 128 * 128))
    query_rows = 512 if scaled.shape[1] >= 4096 else 128

    def body(index, state):
        rank, b = index // q.shape[0], index % q.shape[0]
        groups = routes[rank, b].reshape(-1)
        layout = group_queries(groups, meta[0].shape[1], rows=query_rows)
        starts = meta[1][b, :, 0]
        dqt, _ = grouped_query(scaled[b], k[b], v[b], starts, layout, block_size=block_size,
                               key_tile=key_tile, query_rows=query_rows, do=flat_do[b], lse=flat_lse[b], delta=delta[b])
        dqt = jnp.where(groups[:, None] >= 0, dqt[layout[1]] * quantized_scale, 0).reshape(q.shape[1:])
        dkt, dvt = grouped_kv(scaled[b], k[b], v[b], starts, layout, flat_do[b], flat_lse[b], delta[b],
                             state[1][b], state[2][b], block_size=block_size, key_tile=key_tile, query_rows=query_rows)
        return (lax.dynamic_update_slice_in_dim(state[0], (state[0][b] + dqt)[None], b, axis=0),
                lax.dynamic_update_slice_in_dim(state[1], dkt[None], b, axis=0),
                lax.dynamic_update_slice_in_dim(state[2], dvt[None], b, axis=0))

    if routes.shape[0]:
        grads = lax.fori_loop(0, routes.shape[0] * q.shape[0], body, grads)
    return grads
