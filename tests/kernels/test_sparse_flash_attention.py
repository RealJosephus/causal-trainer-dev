"""Execute the actual TPU tile programs through Pallas CPU interpretation."""

import dataclasses
import functools
from unittest.mock import patch

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import Mesh

from causal_trainer.kernels import sparse_flash_attention as sparse
from causal_trainer.modeling.approximate_attention import moba_attention
from causal_trainer.modeling.attention_options import AttentionOptions


@pytest.fixture
def interpret(monkeypatch):
    monkeypatch.setattr(sparse.pl, "pallas_call", functools.partial(sparse.pl.pallas_call, interpret=True))


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16])
def test_flash_tile_forward_and_accumulating_backward(interpret, dtype):
    rng = np.random.default_rng(701)
    q, k, v, do = (jnp.asarray(rng.normal(scale=0.2, size=shape), dtype) for shape in (
        (128, 128), (384, 128), (384, 128), (128, 128),
    ))
    # Empty rows, high bit 31, sparse holes, and entirely skipped middle tile.
    selected = np.stack([rng.choice(np.r_[0:128, 256:384], 97, replace=False) for _ in range(128)])
    selected[0] = -1
    ids = jnp.asarray(selected, jnp.int32)
    mask = np.zeros((128, 384), bool)
    for row in range(1, 128):
        mask[row, selected[row]] = True
    words, schedule = sparse.pack_routes(ids, 384, 128)
    np.testing.assert_array_equal(np.asarray(schedule[:3]), [2, 0, 2])

    def reference(q, k, v):
        logits = jnp.where(mask, q.astype(jnp.float32) @ k.astype(jnp.float32).T, -1e30)
        probs = jnp.where(mask, jax.nn.softmax(logits, axis=-1), 0)
        return probs @ v.astype(jnp.float32)

    expected, pullback = jax.vjp(reference, q, k, v)
    expected_grads = pullback(do.astype(jnp.float32))
    out, lse = sparse.flash_forward(q, k, v, words, schedule, key_tile=128)
    delta = jnp.sum(out * do.astype(jnp.float32), axis=-1)
    initial_dk = jnp.full(k.shape, 0.125, jnp.float32)
    initial_dv = jnp.full(v.shape, -0.125, jnp.float32)
    dq, dk, dv = sparse.flash_backward(
        q, k, v, words, schedule, do, lse, delta, initial_dk, initial_dv, key_tile=128,
    )
    tol = 0.02 if dtype == jnp.bfloat16 else 1e-5
    for actual, target in zip((out, dq, dk - initial_dk, dv - initial_dv), (expected, *expected_grads), strict=True):
        error = np.linalg.norm(np.asarray(actual, np.float32) - np.asarray(target, np.float32))
        assert error / max(np.linalg.norm(np.asarray(target, np.float32)), 1e-12) < tol
    np.testing.assert_array_equal(np.asarray(out[0]), 0)
    np.testing.assert_array_equal(np.asarray(dk[128:256]), 0.125)
    np.testing.assert_array_equal(np.asarray(dv[128:256]), -0.125)


@pytest.mark.parametrize("method", ["moba"])
def test_sparse_method_forward_and_vjp_matches_reference(interpret, method):
    rng = np.random.default_rng(992)
    q, k, v, do = (jnp.asarray(rng.normal(scale=0.2, size=shape), jnp.float32) for shape in (
        (1, 48, 2, 128), (1, 48, 1, 128), (1, 48, 1, 128), (1, 48, 2, 128),
    ))
    segments = jnp.asarray([[0] * 3 + [1] * 16 + [2] * 24 + [0] * 5], jnp.int32)
    options = AttentionOptions(method=method, backend="pallas", moba_block_size=8, moba_top_k=2,
    )
    function = moba_attention

    def run(options):
        fn = functools.partial(function, segments=segments, options=options, scale=0.125, mesh=None)
        result, backward = jax.vjp(jax.jit(fn), q, k, v)
        return result, *backward(do)

    reference = run(dataclasses.replace(options, backend="reference"))
    actual = run(options)
    for got, expected in zip(actual, reference, strict=True):
        np.testing.assert_allclose(got, expected, rtol=3e-4, atol=3e-6)


@pytest.mark.parametrize("top_k", [1, 2, 4])
def test_grouped_moba_unaligned_packed_blocks_and_global_softmax(interpret, top_k):
    rng = np.random.default_rng(996)
    q, k, v, do = (jnp.asarray(rng.normal(scale=0.3, size=shape), jnp.float32) for shape in (
        (1, 448, 2, 128), (1, 448, 1, 128), (1, 448, 1, 128), (1, 448, 2, 128),
    ))
    segments = jnp.asarray([[0] * 3 + [1] * 26 + [2] * 400 + [0] * 19], jnp.int32)
    options = AttentionOptions(method="moba", backend="pallas", moba_block_size=128, moba_top_k=top_k)

    def run(options):
        fn = functools.partial(moba_attention, segments=segments, options=options, scale=0.125, mesh=None)
        result, backward = jax.vjp(jax.jit(fn), q, k, v)
        return result, *backward(do)

    reference = run(dataclasses.replace(options, backend="reference"))
    actual = run(options)
    for got, expected in zip(actual, reference, strict=True):
        np.testing.assert_allclose(got, expected, rtol=5e-4, atol=4e-6)


@pytest.mark.parametrize("method", ["moba"])
def test_tpuv4_mosaic_frontend_export(method):
    """Offline TPU lowering, NOT device compilation or a performance test.

    The CPU host supplies an explicit v4 target description to Mosaic. This
    catches TPU-only primitive/ref restrictions missed by interpretation.
    Real executable compilation and memory allocation require the TPU gates.
    """
    from jax import export
    from jax._src.pallas.mosaic import tpu_info

    options = AttentionOptions(method=method, backend="pallas", moba_block_size=128, moba_top_k=3,
    )
    function = moba_attention
    length = 512 if method == "moba" else 64

    def objective(q, k, v):
        segments = jnp.ones(q.shape[:2], jnp.int32)
        out = function(q, k, v, segments, options=options, scale=0.125, mesh=None)
        return jnp.sum(out.astype(jnp.float32) ** 2)

    shapes = [jax.ShapeDtypeStruct((1, length, heads, 128), jnp.bfloat16) for heads in (2, 1, 1)]
    with patch.object(tpu_info, "get_device_kind", lambda: "TPU v4"), \
         patch.object(tpu_info, "get_num_device_cores", lambda: 2):
        lowered = export.export(jax.jit(jax.value_and_grad(objective, argnums=(0, 1, 2))), platforms=("tpu",))(*shapes)
    assert "tpu_custom_call" in lowered.mlir_module()


@pytest.mark.parametrize("method", ["moba"])
def test_tp_sp_interpreted_kernel_mqa_gradient_reduction(interpret, method):
    if jax.device_count() < 4:
        pytest.skip("requires JAX_NUM_CPU_DEVICES=4")
    mesh = Mesh(np.asarray(jax.devices()[:4]).reshape(1, 1, 1, 2, 2), ("dp", "fsdp", "ep", "tp", "sp"))
    rng = np.random.default_rng(881)
    q, k, v, do = (jnp.asarray(rng.normal(scale=0.1, size=shape), jnp.bfloat16) for shape in (
        (1, 512, 4, 128), (1, 512, 1, 128), (1, 512, 1, 128), (1, 512, 4, 128),
    ))
    segments = jnp.asarray([[0] * 7 + [1] * 31 + [2] * 454 + [0] * 20], jnp.int32)
    do = jnp.where((jnp.arange(512) >= 320)[None, :, None, None], do, 0)
    options = AttentionOptions(method=method, backend="pallas", moba_block_size=128, moba_top_k=3,
    )
    function = moba_attention

    def run(mesh, options):
        fn = functools.partial(function, segments=segments, options=options, scale=0.125, mesh=mesh)
        out, backward = jax.vjp(jax.jit(fn), q, k, v)
        return out, *backward(do)

    reference = run(None, dataclasses.replace(options, backend="reference"))
    with jax.set_mesh(mesh):
        actual = run(mesh, options)
    for got, expected in zip(actual, reference, strict=True):
        got, expected = np.asarray(got, np.float32), np.asarray(expected, np.float32)
        assert np.linalg.norm(got - expected) / max(np.linalg.norm(expected), 1e-12) < 0.025
    for grad in actual[2:]:
        assert np.any(np.asarray(grad, np.float32)[:, 38:256] != 0)


@pytest.mark.parametrize("method", ["moba"])
def test_million_token_pallas_shape_and_vjp(method):
    """No allocation: reject accidental sequence-squared graph construction."""
    length = 1_048_576
    options = AttentionOptions(method=method, backend="pallas")
    function = moba_attention
    shapes = [jax.ShapeDtypeStruct((1, length, heads, 128), jnp.bfloat16) for heads in (32, 1, 1)]

    def run(q, k, v):
        fn = functools.partial(function, segments=jnp.ones((1, length), jnp.int32),
                               options=options, scale=128**-0.5, mesh=None)
        out, backward = jax.vjp(fn, q, k, v)
        return out, *backward(jnp.ones_like(out))

    result = jax.eval_shape(run, *shapes)
    assert result[0].shape == shapes[0].shape
    assert [x.shape for x in result[1:]] == [x.shape for x in shapes]
