"""MQA MoBA with per-query/head top-k block selection."""

from __future__ import annotations

from dataclasses import replace

import jax
import jax.numpy as jnp
from jax.sharding import Mesh

from ..kernels.indexed_attention import indexed_attention
from .attention_options import AttentionOptions
from .rotary import _make_position_ids_precanonicalized


def _attention_operator(options):
    if options.backend == "pallas" or (options.backend == "auto" and jax.default_backend() == "tpu"):
        from ..kernels.sparse_flash_attention import sparse_flash_attention
        return sparse_flash_attention
    return indexed_attention


def _positions(segments):
    positions = jnp.broadcast_to(jnp.arange(segments.shape[1], dtype=jnp.int32), segments.shape)
    relative = _make_position_ids_precanonicalized(segments != 0, segments)
    return positions, positions - relative


def _moba_summaries(key, segments, starts, block_size):
    positions = jnp.arange(key.shape[1])[None, :]
    resets = ((positions - starts) % block_size == 0) | (segments == 0)

    def combine(a, b):
        reset_a, sum_a = a
        reset_b, sum_b = b
        return reset_a | reset_b, jnp.where(reset_b, sum_b, sum_a + sum_b)

    # Sum within sample-relative blocks, avoiding subtracting large cumulative
    # sums and preventing other samples from affecting even rounding of routing.
    _, sums = jax.lax.associative_scan(
        combine, (resets[..., None], key.astype(jnp.float32)), axis=1,
    )
    return jax.lax.stop_gradient(sums / block_size)


def moba_routes(query, context, metadata, *, block_size, top_k, sequence):
    """Return token addresses for distinct, segment-relative selected blocks."""

    positions, segments, active, starts = context
    del active
    (summaries,) = metadata
    candidates = (sequence + block_size - 1) // block_size
    current = (positions - starts) // block_size
    block = jnp.arange(candidates)
    ends = starts[..., None] + (block + 1) * block_size - 1
    means = summaries[jnp.arange(query.shape[0])[:, None, None], jnp.clip(ends, 0, sequence - 1)]
    scores = jnp.einsum("bthd,btcd->bthc", query.astype(jnp.float32), means)
    completed = (block[None, None, None, :] < current[:, :, None, None])
    scores = jnp.where(completed, scores, -jnp.inf)
    count = min(top_k - 1, max(candidates - 1, 0))
    if count:
        selected_scores, previous = jax.lax.top_k(scores, count)
        previous = jnp.where(jnp.isfinite(selected_scores), previous, -1)
    else:
        previous = jnp.empty((*query.shape[:3], 0), jnp.int32)
    current = jnp.broadcast_to(current[:, :, None, None], (*query.shape[:3], 1))
    selected = jnp.concatenate((current, previous), axis=-1)
    indices = starts[:, :, None, None, None] + selected[..., None] * block_size + jnp.arange(block_size)
    allowed = (
        (selected[..., None] >= 0) & (indices <= positions[:, :, None, None, None])
        & (indices < sequence) & (segments[:, :, None, None, None] > 0)
    )
    return jnp.where(allowed, indices, -1).reshape(*query.shape[:3], -1)


def moba_attention(query, key, value, segments, *, options: AttentionOptions, scale: float, mesh: Mesh | None):
    if (options.backend != "reference" and jax.default_backend() == "tpu" and mesh is not None
            and query.dtype == key.dtype == value.dtype == jnp.bfloat16 and query.shape[-1] == 128
            and query.shape[2] // int(mesh.shape.get("tp", 1)) <= 8
            and query.shape[1] % (128 * int(mesh.shape.get("sp", 1))) == 0
            and options.moba_top_k * options.moba_block_size >= query.shape[1]):
        # All sample histories fit the routing budget: the graph is exactly
        # dense, so use the existing fused Pallas kernel without routing work.
        from ..kernels.efficient_attention import (
            efficient_attention,
            prepare_attention_metadata,
            resolve_attention_profile,
        )
        profile = resolve_attention_profile(query.shape, key.shape, mesh)
        if profile.block_q * profile.block_h > 1024:
            profile = replace(profile, block_q=128)
        metadata = prepare_attention_metadata(segments, mesh=mesh, profile=profile)
        return efficient_attention(query, key, value, metadata, mesh=mesh, profile=profile, scale=scale)
    positions, starts = _positions(segments)
    key, value = key[:, :, 0], value[:, :, 0]
    block_size = min(options.moba_block_size, query.shape[1])
    operator = _attention_operator(options)
    if operator is not indexed_attention and block_size % 128 == 0:
        from ..kernels.grouped_moba import compact_blocks
        metadata = compact_blocks(key, segments, starts, block_size, mesh)

        def current_block(q, context, meta):
            del meta
            positions, _, active, starts = context
            begin = starts + (positions - starts) // block_size * block_size
            ids = begin[:, :, None] + jnp.arange(block_size)
            valid = (ids <= positions[:, :, None]) & active[:, :, None]
            return jnp.broadcast_to(jnp.where(valid, ids, -1)[:, :, None], (*q.shape[:3], block_size))

        return operator(
            query, key, value, (positions, segments, segments > 0, starts), metadata,
            route=current_block, metadata_specs=(False, False, False), scale=scale,
            query_tile=options.query_chunk_size, key_tile=options.key_chunk_size, mesh=mesh,
            moba_config=(block_size, options.moba_top_k),
        )
    summaries = _moba_summaries(key, segments, starts, block_size)

    def route(q, context, metadata):
        return moba_routes(
            q, context, metadata, block_size=block_size, top_k=options.moba_top_k, sequence=key.shape[1],
        )

    return _attention_operator(options)(
        query, key, value, (positions, segments, segments > 0, starts), (summaries,),
        route=route, metadata_specs=(False,), scale=scale,
        query_tile=options.query_chunk_size, key_tile=options.key_chunk_size, mesh=mesh,
    )
