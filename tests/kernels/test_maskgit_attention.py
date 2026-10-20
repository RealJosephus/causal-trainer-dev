"""Local CPU interpretation of the actual hybrid Efficient forward/dQ/dKV.

These gates check arithmetic and sequence sharding, not TPU compiler lowering
or hardware performance. No remote devices or model downloads are used.
"""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import Mesh

from causal_trainer.kernels import efficient_attention as kernel
from causal_trainer.modeling.attention import _make_causal_segment_mask_precanonicalized, vanilla_attention


def _fixture():
    segments = np.zeros((2, 512), np.int32)
    segments[0, 20:350] = 1
    segments[0, 350:500] = 2
    segments[1, :150] = 1
    segments[1, 150:] = 2
    ends = np.broadcast_to(np.arange(512, dtype=np.int32), segments.shape).copy()
    ends[0, 80:330] = 329  # Crosses Q/K tiles and the SP boundary.
    ends[0, 390:470] = 469
    return jnp.asarray(segments), jnp.asarray(ends)


@pytest.mark.parametrize("compact", [False, True])
@pytest.mark.parametrize("sp", [1, 2])
def test_cpu_interpreter_hybrid_forward_and_vjp_match_dense(monkeypatch, compact, sp):
    if jax.default_backend() != "cpu" or jax.local_device_count() < sp:
        pytest.skip("CPU interpretation; SP2 requires XLA_FLAGS=--xla_force_host_platform_device_count=2")
    devices = np.asarray(jax.local_devices()[:sp], dtype=object).reshape(1, 1, 1, 1, sp)
    mesh = Mesh(devices, ("dp", "fsdp", "ep", "tp", "sp"))
    profile = kernel.AttentionKernelProfile(2, 128, 256, 128, compact_statistics=compact)
    segments, ends = _fixture()
    metadata = kernel.prepare_attention_metadata(segments, mesh=mesh, profile=profile, attention_end=ends)
    allowed = _make_causal_segment_mask_precanonicalized(segments != 0, segments, ends)
    keys = jax.random.split(jax.random.PRNGKey(63), 4)
    q, k, v, cotangent = [
        (jax.random.normal(key, (2, 512, heads, 128)) * .3).astype(jnp.bfloat16)
        for key, heads in zip(keys, (2, 1, 1, 2), strict=True)
    ]
    monkeypatch.setattr(kernel.pl, "pallas_call", functools.partial(kernel.pl.pallas_call, interpret=True))
    # Same interpreter-only VMA workaround as test_tiled_training. TP remains 1.
    monkeypatch.setattr(
        kernel, "shard_map", lambda fn, **kwargs: jax.shard_map(fn, **(kwargs | {"check_vma": False})),
    )
    monkeypatch.setattr(jax, "default_backend", lambda: "tpu")

    def run(fn, query, key, value):
        output, backward = jax.vjp(fn, query, key, value)
        return output, *backward(cotangent)

    actual = jax.jit(functools.partial(run, lambda query, key, value: kernel.efficient_attention(
        query, key, value, metadata, mesh=mesh, profile=profile,
    )))(q, k, v)
    expected = jax.jit(functools.partial(run, lambda query, key, value: vanilla_attention(
        query, key, value, allowed, scale=128**-.5,
    )))(q, k, v)
    for label, result, reference in zip(("output", "dQ", "dK", "dV"), actual, expected, strict=True):
        result, reference = np.asarray(result, np.float32), np.asarray(reference, np.float32)
        assert np.isfinite(result).all(), label
        relative_error = np.linalg.norm(result - reference) / max(np.linalg.norm(reference), 1e-6)
        assert relative_error < .06, (label, relative_error)
        np.testing.assert_array_equal(result[0, :20], 0)


def test_hybrid_tile_summaries_allow_future_keys_but_keep_segment_pruning():
    mesh = Mesh(np.asarray(jax.devices()[:1], dtype=object).reshape(1, 1, 1, 1, 1),
                ("dp", "fsdp", "ep", "tp", "sp"))
    profile = kernel.AttentionKernelProfile(2, 128, 128, 128)
    segments, ends = _fixture()
    metadata = kernel.prepare_attention_metadata(segments, mesh=mesh, profile=profile, attention_end=ends)
    q_summary = kernel._query_summary_input(metadata.q_block_segments)
    kv_summary = metadata.kv_block_segments.reshape(2, -1)
    assert q_summary.shape == (2, 12)  # Flat SMEM, not [B, blocks, 128 padded lanes].
    offsets = jnp.array([0, 0], jnp.int32)
    assert kernel._summary_should_run(
        q_summary, kv_summary, offsets, 0, 0, 2,
        block_q=128, block_kv_outer=128, causal=True, has_attention_end=True,
    )
    assert not kernel._summary_should_run(
        q_summary, kv_summary, offsets, 0, 0, 3,
        block_q=128, block_kv_outer=128, causal=True, has_attention_end=True,
    )
    assert kernel._compute_tile_is_causally_reachable(
        offsets, q_block_index=0, kv_block_index=1, compute_offset=0,
        block_q=128, block_kv_outer=256, causal=True, has_attention_end=True,
        q_summary_ref=q_summary, batch_index=0,
    )


def test_hybrid_metadata_keeps_hbm_and_dma_windows_compact():
    segments, ends = _fixture()
    metadata = jnp.stack((segments, ends), axis=-1)
    packed = kernel._query_metadata_lanes(metadata, 128)
    assert packed.ndim == 1
    assert packed.nbytes == 2 * segments.nbytes
    tiles = np.asarray(packed).reshape(2, 4, 2, 128)
    np.testing.assert_array_equal(tiles[:, :, 0].reshape(2, 512), segments)
    np.testing.assert_array_equal(tiles[:, :, 1].reshape(2, 512), ends)
    # The established causal layout remains unchanged.
    assert kernel._query_metadata_lanes(segments, 128).shape == (2, 512, 128)


@pytest.mark.parametrize("local_heads", [1, 2, 4, 8])
def test_hybrid_profile_bounds_query_rows_and_preserves_causal_defaults(local_heads):
    q_shape, kv_shape = (1, 8192, 2 * local_heads, 128), (1, 8192, 1, 128)
    mesh = {"tp": 2, "sp": 2}
    causal = kernel.resolve_attention_profile(q_shape, kv_shape, mesh)
    hybrid = kernel.resolve_attention_profile(q_shape, kv_shape, mesh, has_attention_end=True)
    assert hybrid.block_q * local_heads <= 1024
    assert causal.block_q * local_heads <= 2048
    assert hybrid.block_kv_outer == causal.block_kv_outer
    assert hybrid.block_kv_compute == causal.block_kv_compute
    assert causal == kernel.resolve_attention_profile(q_shape, kv_shape, mesh, has_attention_end=False)
    with pytest.raises(TypeError, match="has_attention_end"):
        kernel.resolve_attention_profile(q_shape, kv_shape, mesh, has_attention_end=1)


def test_hybrid_tp2_sp2_traces_real_pallas_with_production_vma(monkeypatch):
    if jax.local_device_count() < 4:
        pytest.skip("requires four local devices for TP2/SP2 abstract tracing")
    mesh = Mesh(np.asarray(jax.devices()[:4], object).reshape(1, 1, 1, 2, 2),
                ("dp", "fsdp", "ep", "tp", "sp"))
    profile = kernel.AttentionKernelProfile(2, 128, 256, 128, compact_statistics=True)
    segments, ends = _fixture()
    metadata = kernel.prepare_attention_metadata(segments, mesh=mesh, profile=profile, attention_end=ends)
    q = jnp.zeros((2, 512, 4, 128), jnp.bfloat16)
    kv = jnp.zeros((2, 512, 1, 128), jnp.bfloat16)
    monkeypatch.setattr(jax, "default_backend", lambda: "tpu")

    def objective(q, k, v):
        return kernel.efficient_attention(q, k, v, metadata, mesh=mesh, profile=profile).astype(jnp.float32).sum()

    # Keep real Pallas primitives and check_vma=True. No CPU interpreter
    # workaround here: validate production forward/backward abstract contracts.
    value, gradients = jax.eval_shape(jax.value_and_grad(objective, argnums=(0, 1, 2)), q, kv, kv)
    assert value.shape == ()
    assert [gradient.shape for gradient in gradients] == [q.shape, kv.shape, kv.shape]
