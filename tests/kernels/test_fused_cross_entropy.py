from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from causal_trainer.kernels.fused_cross_entropy import fused_linear_cross_entropy


@pytest.mark.parametrize("chunk_size", [0, 17])
def test_cut_local_chunk_option_is_forwarded_only_when_enabled(monkeypatch, chunk_size):
    import causal_trainer.kernels.cut_cross_entropy as cut_module

    def cut(hidden, head, targets, weights, **kwargs):
        if chunk_size:
            assert kwargs["local_chunk_size"] == chunk_size
        else:
            assert "local_chunk_size" not in kwargs
        assert kwargs["sparse_compaction"] is True
        assert hidden.dtype == head.dtype == jnp.bfloat16
        return jnp.sum(weights), jnp.sum(weights)

    monkeypatch.setattr(jax, "default_backend", lambda: "tpu")
    monkeypatch.setattr(cut_module, "pallas_cut_linear_cross_entropy", cut)
    fused_linear_cross_entropy(
        jnp.ones((1, 8, 128)), jnp.ones((128, 256)),
        jnp.zeros((1, 8), jnp.int32), jnp.ones((1, 8)),
        implementation="cut", mesh=object(), sparse_skip=True, cut_chunk_size=chunk_size,
    )


@pytest.mark.parametrize("chunk_size", [-1, True, 1.5])
def test_invalid_cut_local_chunk_size_is_rejected(chunk_size):
    with pytest.raises(ValueError, match="cut_chunk_size"):
        fused_linear_cross_entropy(
            jnp.ones((1, 1, 4)), jnp.ones((4, 8)),
            jnp.zeros((1, 1), jnp.int32), jnp.ones((1, 1)), cut_chunk_size=chunk_size,
        )


def _iter_jaxpr_equations(value: Any) -> Iterator[Any]:
    if hasattr(value, "jaxpr"):
        yield from _iter_jaxpr_equations(value.jaxpr)
        return
    if hasattr(value, "eqns"):
        for equation in value.eqns:
            yield equation
            for parameter in equation.params.values():
                yield from _iter_jaxpr_equations(parameter)
        return
    if isinstance(value, (tuple, list)):
        for item in value:
            yield from _iter_jaxpr_equations(item)


def _loss_and_aux(hidden, head, targets, weights, *, sparse_skip):
    nll_sum, weight_sum = fused_linear_cross_entropy(
        hidden,
        head,
        targets,
        weights,
        token_budget=4,
        compute_dtype=jnp.float32,
        implementation="xla",
        sparse_skip=sparse_skip,
    )
    return nll_sum / jnp.maximum(weight_sum, 1.0), (nll_sum, weight_sum)


def test_sparse_skip_with_trailing_inactive_chunks_matches_dense_path() -> None:
    batch, sequence, hidden_size, vocab = 2, 8, 4, 7
    hidden = (
        jax.random.normal(jax.random.PRNGKey(0), (batch, sequence, hidden_size)) * 0.125
    ).astype(jnp.float32)
    head = (
        jax.random.normal(jax.random.PRNGKey(1), (hidden_size, vocab)) * 0.125
    ).astype(jnp.float32)
    targets = jax.random.randint(
        jax.random.PRNGKey(2),
        (batch, sequence),
        0,
        vocab,
        dtype=jnp.int32,
    )
    weights = jnp.zeros((batch, sequence), dtype=jnp.float32)
    weights = weights.at[:, :2].set(jnp.asarray([[1.0, 0.25], [0.5, 1.0]], jnp.float32))

    def evaluate(sparse_skip):
        return jax.jit(
            jax.value_and_grad(
                lambda x, w: _loss_and_aux(
                    x,
                    w,
                    targets,
                    weights,
                    sparse_skip=sparse_skip,
                ),
                argnums=(0, 1),
                has_aux=True,
            )
        )(hidden, head)

    (expected_loss, expected_aux), expected_gradients = evaluate(False)
    (actual_loss, actual_aux), actual_gradients = evaluate(True)

    np.testing.assert_allclose(actual_loss, expected_loss, rtol=0.0, atol=1e-6)
    np.testing.assert_allclose(actual_aux[0], expected_aux[0], rtol=0.0, atol=1e-6)
    np.testing.assert_allclose(actual_aux[1], expected_aux[1], rtol=0.0, atol=1e-6)
    np.testing.assert_allclose(actual_gradients[0], expected_gradients[0], rtol=0.0, atol=1e-6)
    np.testing.assert_allclose(actual_gradients[1], expected_gradients[1], rtol=0.0, atol=1e-6)


def test_sparse_skip_cond_has_only_one_scalar_result() -> None:
    batch, sequence, hidden_size, vocab = 2, 8, 4, 7
    hidden = jax.ShapeDtypeStruct((batch, sequence, hidden_size), jnp.float32)
    head = jax.ShapeDtypeStruct((hidden_size, vocab), jnp.float32)
    targets = jax.ShapeDtypeStruct((batch, sequence), jnp.int32)
    weights = jax.ShapeDtypeStruct((batch, sequence), jnp.float32)

    closed_jaxpr = jax.make_jaxpr(
        lambda x, w, y, m: fused_linear_cross_entropy(
            x,
            w,
            y,
            m,
            token_budget=4,
            compute_dtype=jnp.float32,
            implementation="xla",
            sparse_skip=True,
        )
    )(hidden, head, targets, weights)
    conditionals = [
        equation
        for equation in _iter_jaxpr_equations(closed_jaxpr)
        if equation.primitive.name == "cond"
    ]

    assert len(conditionals) == 1
    branches = conditionals[0].params["branches"]
    assert all(len(branch.jaxpr.outvars) == 1 for branch in branches)


@pytest.mark.parametrize("sparse_skip", [False, True])
@pytest.mark.parametrize("train_head", [False, True])
def test_chunked_loss_vjp_does_not_save_a_head_copy_per_chunk(sparse_skip, train_head) -> None:
    # A conditional outside remat can turn the loop-invariant head into a
    # scan residual. At B32/S8192/DP8/TP4 this saved 256 copies of the head,
    # consuming 90.625 GiB per chip before other training allocations.
    batch, sequence, hidden_size, vocab = 2, 32, 16, 257
    hidden = jnp.zeros((batch, sequence, hidden_size), jnp.float32)
    head = jnp.zeros((hidden_size, vocab), jnp.float32)
    targets = jnp.zeros((batch, sequence), jnp.int32)
    weights = jnp.broadcast_to((jnp.arange(sequence) % 4 == 0).astype(jnp.float32), targets.shape)

    def loss(x, w):
        return _loss_and_aux(
            x, w if train_head else jax.lax.stop_gradient(w), targets, weights, sparse_skip=sparse_skip,
        )[0]

    _, pullback = jax.vjp(loss, hidden, head)
    residual_shapes = [value.shape for value in jax.tree.leaves(pullback) if hasattr(value, "shape")]
    chunks = batch * sequence // 4
    assert (chunks, hidden_size, vocab) not in residual_shapes


@pytest.mark.parametrize("pattern", ["inactive", "middle", "alternating", "dense"])
@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16])
def test_sparse_skip_padded_chunks_match_unchunked_loss_and_both_gradients(pattern, dtype) -> None:
    batch, sequence, hidden_size, vocab = 2, 9, 16, 19
    hidden = (jax.random.normal(jax.random.PRNGKey(20), (batch, sequence, hidden_size)) * 0.125).astype(dtype)
    head = (jax.random.normal(jax.random.PRNGKey(21), (hidden_size, vocab)) * 0.125).astype(dtype)
    targets = jax.random.randint(jax.random.PRNGKey(22), (batch, sequence), 0, vocab)
    positions = jnp.arange(sequence)
    active = {
        "inactive": positions < 0,
        "middle": (positions >= 2) & (positions < 4),
        "alternating": (positions // 2) % 2 == 0,
        "dense": positions >= 0,
    }[pattern]
    weights = jnp.asarray([[1.0], [0.25]], jnp.float32) * active[None, :]

    def reference(x, w):
        logits = jnp.einsum("bsh,hv->bsv", x, w).astype(jnp.float32)
        nll = jnp.sum(
            (jax.nn.logsumexp(logits, axis=-1)
             - jnp.take_along_axis(logits, targets[..., None], axis=-1)[..., 0]) * weights,
        )
        count = jnp.sum(weights)
        return nll / jnp.maximum(count, 1.0), (nll, count)

    def actual(x, w):
        nll, count = fused_linear_cross_entropy(
            x, w, targets, weights, token_budget=4, compute_dtype=dtype,
            implementation="xla", sparse_skip=True,
        )
        return nll / jnp.maximum(count, 1.0), (nll, count)

    expected = jax.jit(jax.value_and_grad(reference, argnums=(0, 1), has_aux=True))(hidden, head)
    result = jax.jit(jax.value_and_grad(actual, argnums=(0, 1), has_aux=True))(hidden, head)
    np.testing.assert_allclose(result[0][0], expected[0][0], rtol=0.0, atol=1e-6)
    # Chunked and unchunked FP32 sums differ in reduction order (one ULP at
    # this scale); the normalized loss and both gradients use absolute gates.
    np.testing.assert_allclose(result[0][1][0], expected[0][1][0], rtol=2e-7, atol=1e-6)
    np.testing.assert_array_equal(result[0][1][1], expected[0][1][1])
    for got, want in zip(result[1], expected[1], strict=True):
        np.testing.assert_allclose(
            np.asarray(got, np.float32), np.asarray(want, np.float32),
            rtol=0.0, atol=5e-3 if dtype == jnp.bfloat16 else 1e-6,
        )
    assert np.all(np.asarray(result[1][0])[:, ~np.asarray(active)] == 0)
