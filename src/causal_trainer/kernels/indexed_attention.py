"""Bounded-workspace indexed MQA attention with a recomputing analytic VJP.

Routes are integer addresses, never a dense sequence-by-sequence mask. Only a
query tile and a key tile are gathered at once. Backward recomputes the routes
and probabilities, scattering directly into shared FP32 K/V gradient buffers.
Neither autodiff scan tapes nor expanded per-query/per-head K/V are retained.
This explicitly selectable reference backend uses portable JAX/XLA. Automatic
TPU dispatch uses the specialized kernels in sparse_flash_attention instead.
"""

from __future__ import annotations

from collections.abc import Callable

import jax
import jax.numpy as jnp
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P

Array = jax.Array


def _pad(x: Array, amount: int, axis: int = 1, value=0) -> Array:
    widths = [(0, 0)] * x.ndim
    widths[axis] = (0, amount)
    return jnp.pad(x, widths, constant_values=value)


def _local_indexed_attention(q, k, v, info, metadata, *, route, scale, query_tile, key_tile):
    """Local Q/heads, shared complete KV; metadata and routes carry no gradients."""

    original_length = q.shape[1]
    padding = -original_length % query_tile
    q = _pad(q, padding)
    info = tuple(_pad(x, padding) for x in info)
    batch, length, heads, dim = q.shape

    def query_slice(x, index):
        return jax.lax.dynamic_slice_in_dim(x, index * query_tile, query_tile, axis=1)

    def routes(qt, context, meta):
        indices = route(jax.lax.stop_gradient(qt), context, meta)
        indices = jnp.where(context[2][:, :, None, None], indices, -1)
        return _pad(indices, -indices.shape[-1] % key_tile, axis=-1, value=-1)

    def gather_tile(indices, index, keys, values):
        batch_index = jnp.arange(batch)[:, None, None, None]
        ids = jax.lax.dynamic_slice_in_dim(indices, index * key_tile, key_tile, axis=-1)
        valid = (ids >= 0) & (ids < keys.shape[1])
        safe = jnp.clip(ids, 0, keys.shape[1] - 1)
        kt = jnp.where(valid[..., None], keys[batch_index, safe], 0).astype(jnp.float32)
        vt = jnp.where(valid[..., None], values[batch_index, safe], 0).astype(jnp.float32)
        return safe, valid, kt, vt

    def key_loop(indices, function, initial):
        def body(index, state):
            ids = jax.lax.dynamic_slice_in_dim(indices, index * key_tile, key_tile, axis=-1)
            return jax.lax.cond(
                jnp.any(ids >= 0), lambda carry: function(index, carry), lambda carry: carry, state,
            )
        return jax.lax.fori_loop(0, indices.shape[-1] // key_tile, body, initial)

    def forward_impl(queries, keys, values, context, meta):
        output = jnp.zeros_like(queries)
        lse = jnp.zeros(queries.shape[:-1], jnp.float32)

        def body(index, carry):
            def compute(state):
                result, statistics = state
                qt = query_slice(queries, index)
                ctx = tuple(query_slice(x, index) for x in context)
                ids = routes(qt, ctx, meta)
                scaled = (qt * jnp.asarray(scale, qt.dtype)).astype(jnp.float32)
                shape = (batch, query_tile, heads)
                initial = (
                    jnp.full(shape, -1e30, jnp.float32), jnp.zeros(shape, jnp.float32),
                    jnp.zeros((*shape, dim), jnp.float32),
                )

                def key_body(tile, running):
                    maximum, denominator, numerator = running
                    _, valid, kt, vt = gather_tile(ids, tile, keys, values)
                    logits = jnp.einsum("bthd,bthkd->bthk", scaled, kt)
                    logits = jnp.where(valid, logits, -1e30)
                    updated_maximum = jnp.maximum(maximum, jnp.max(logits, axis=-1))
                    correction = jnp.exp(maximum - updated_maximum)
                    weights = jnp.where(valid, jnp.exp(logits - updated_maximum[..., None]), 0)
                    return (
                        updated_maximum,
                        denominator * correction + jnp.sum(weights, axis=-1),
                        numerator * correction[..., None] + jnp.einsum("bthk,bthkd->bthd", weights, vt),
                    )

                maximum, denominator, numerator = key_loop(ids, key_body, initial)
                out = numerator / jnp.maximum(denominator[..., None], 1e-30)
                stats = jnp.where(denominator > 0, maximum + jnp.log(jnp.maximum(denominator, 1e-30)), 0)
                return (
                    jax.lax.dynamic_update_slice_in_dim(result, out.astype(qt.dtype), index * query_tile, axis=1),
                    jax.lax.dynamic_update_slice_in_dim(statistics, stats, index * query_tile, axis=1),
                )

            active = query_slice(context[2], index)
            return jax.lax.cond(jnp.any(active), compute, lambda state: state, carry)

        return jax.lax.fori_loop(0, length // query_tile, body, (output, lse))

    @jax.custom_vjp
    def apply(queries, keys, values, context, meta):
        return forward_impl(queries, keys, values, context, meta)[0]

    def forward(queries, keys, values, context, meta):
        output, lse = forward_impl(queries, keys, values, context, meta)
        # FP32 softmax derivatives use a recomputed output below, avoiding BF16
        # output rounding in the row-dot subtraction.
        return output, (queries, keys, values, context, meta, lse)

    def backward(residual, cotangent):
        queries, keys, values, context, meta, lse = residual
        initial = tuple(jnp.zeros(x.shape, jnp.float32) for x in (queries, keys, values))

        def body(index, carry):
            def compute(state):
                dq, dk, dv = state
                qt = query_slice(queries, index)
                ctx = tuple(query_slice(x, index) for x in context)
                ids = routes(qt, ctx, meta)
                scaled = (qt * jnp.asarray(scale, qt.dtype)).astype(jnp.float32)
                do = query_slice(cotangent, index).astype(jnp.float32)
                statistics = query_slice(lse, index)

                def probabilities(tile):
                    safe, valid, kt, vt = gather_tile(ids, tile, keys, values)
                    logits = jnp.einsum("bthd,bthkd->bthk", scaled, kt)
                    # Mask before exp: masked large logits must not overflow.
                    p = jnp.exp(jnp.where(valid, logits - statistics[..., None], -1e30))
                    return safe, kt, vt, p

                def row_dot(tile, accumulated):
                    _, _, vt, p = probabilities(tile)
                    dp = jnp.einsum("bthd,bthkd->bthk", do, vt)
                    return accumulated + jnp.sum(p * dp, axis=-1)

                delta = key_loop(
                    ids, row_dot,
                    jnp.zeros((batch, query_tile, heads), jnp.float32),
                )

                def key_body(tile, running):
                    dqt, dkeys, dvalues = running
                    batch_index = jnp.arange(batch)[:, None, None, None]
                    safe, kt, vt, p = probabilities(tile)
                    dp = jnp.einsum("bthd,bthkd->bthk", do, vt)
                    ds = p * (dp - delta[..., None])
                    dqt += jnp.einsum("bthk,bthkd->bthd", ds, kt)
                    dkeys = dkeys.at[batch_index, safe].add(ds[..., None] * scaled[..., None, :])
                    dvalues = dvalues.at[batch_index, safe].add(p[..., None] * do[..., None, :])
                    return dqt, dkeys, dvalues

                dqt, dk, dv = key_loop(
                    ids, key_body,
                    (jnp.zeros_like(qt, dtype=jnp.float32), dk, dv),
                )
                dq = jax.lax.dynamic_update_slice_in_dim(dq, dqt * scale, index * query_tile, axis=1)
                return dq, dk, dv

            return jax.lax.cond(
                jnp.any(query_slice(context[2], index)) & jnp.any(query_slice(cotangent, index) != 0),
                compute, lambda state: state, carry,
            )

        gradients = jax.lax.fori_loop(0, length // query_tile, body, initial)
        return *(g.astype(x.dtype) for g, x in zip(gradients, (queries, keys, values), strict=True)), None, None

    apply.defvjp(forward, backward)
    return apply(q, k, v, info, jax.tree.map(jax.lax.stop_gradient, metadata))[:, :original_length]


def indexed_attention(
    query: Array, key: Array, value: Array, query_info: tuple, metadata: tuple, *,
    route: Callable, metadata_specs: tuple, scale: float, query_tile: int, key_tile: int,
    mesh: Mesh | None,
) -> Array:
    """Shard Q/output over SP/TP, replicate a single KV bank, reduce its VJP.

    ``query_info`` is (physical positions, canonical segments, active, segment
    starts). ``route`` returns [batch, tile, heads, candidates] bank addresses,
    with -1 for unused entries. Repeated valid addresses count repeatedly;
    algorithms must deduplicate their selection.
    """

    def local(q, k, v, info, meta):
        return _local_indexed_attention(
            q, k, v, info, meta, route=route, scale=scale, query_tile=query_tile, key_tile=key_tile,
        )

    if mesh is None:
        return local(query, key, value, query_info, metadata)

    def axis(*names):
        active = tuple(name for name in names if name in mesh.axis_names and mesh.shape[name] > 1)
        return (active[0] if len(active) == 1 else active) or None

    batch, sequence, heads = axis("dp", "fsdp"), axis("sp"), axis("tp")
    q_spec = P(batch, sequence, heads, None)
    kv_spec = P(batch, None, None)
    info_spec = (P(batch, sequence),) * 4
    meta_spec = tuple(P(batch, None, heads if per_head else None) for per_head in metadata_specs)
    # Metadata leaves here all have rank 3; no sequence-wide Q/KV head expansion.
    mapped = jax.shard_map(
        local, mesh=mesh, in_specs=(q_spec, kv_spec, kv_spec, info_spec, meta_spec),
        out_specs=q_spec, check_vma=False,
    )
    return mapped(query, key, value, query_info, metadata)
