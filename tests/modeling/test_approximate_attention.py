"""Independent numerical, causality, packing, and distributed attention gates."""

from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

from causal_trainer.modeling.attention import (
    attention,
    make_causal_segment_mask,
    normalize_segment_ids,
    vanilla_attention,
)
from causal_trainer.modeling.attention_options import AttentionOptions


def _inputs(length=19, batch=2):
    qk, kk, vk = jax.random.split(jax.random.key(23), 3)
    return (
        jax.random.normal(qk, (batch, length, 4, 4)),
        jax.random.normal(kk, (batch, length, 1, 4)),
        jax.random.normal(vk, (batch, length, 1, 4)),
    )


def _options(method):
    return AttentionOptions(
        method=method, query_chunk_size=2, key_chunk_size=3, moba_block_size=3, moba_top_k=2,
    )


def _run(options, valid, segments=None, mesh=None):
    return lambda q, k, v: attention(
        q, k, v, valid, segments, implementation="vanilla", attention_options=options, mesh=mesh,
    )


def _moba_mask(q, k, valid, segments, block_size, top_k):
    """Slow host oracle following upstream moba_naive.py, with shared MQA KV."""
    q, k = np.asarray(q), np.asarray(k)
    ids = np.asarray(normalize_segment_ids(valid, segments))
    allowed = np.zeros((q.shape[0], q.shape[2], q.shape[1], q.shape[1]), bool)
    for batch in range(q.shape[0]):
        for segment in np.unique(ids[batch]):
            if segment == 0:
                continue
            tokens = np.flatnonzero(ids[batch] == segment)
            blocks = [tokens[start:start + block_size] for start in range(0, len(tokens), block_size)]
            means = np.stack([k[batch, block, 0].mean(axis=0) for block in blocks])
            for local, token in enumerate(tokens):
                current = local // block_size
                for head in range(q.shape[2]):
                    scores = means[:current] @ q[batch, token, head]
                    previous = np.argsort(-scores, kind="stable")[:top_k - 1]
                    chosen = np.concatenate([blocks[i] for i in [current, *previous]])
                    allowed[batch, head, token, chosen[chosen <= token]] = True
    return jnp.asarray(allowed)


def _dense_reference(q, k, v, allowed):
    scaled = q * jnp.asarray(q.shape[-1] ** -0.5, q.dtype)
    scores = jnp.einsum("bqhd,bkd->bhqk", scaled, k[:, :, 0]).astype(jnp.float32)
    probabilities = jax.nn.softmax(jnp.where(allowed, scores, -1e30), axis=-1)
    probabilities = jnp.where(jnp.any(allowed, axis=-1, keepdims=True), probabilities, 0)
    return jnp.einsum("bhqk,bkd->bqhd", probabilities, v[:, :, 0])


def _assert_value_and_grads(actual, reference, inputs, tol=3e-5):
    weight = jax.random.normal(jax.random.key(18), inputs[0].shape)
    def objective(fn):
        return lambda *xs: jnp.sum(fn(*xs) * weight)
    got = jax.jit(actual)(*inputs)
    expected = jax.jit(reference)(*inputs)
    np.testing.assert_allclose(got, expected, atol=tol, rtol=tol)
    gradients = jax.jit(jax.grad(objective(actual), argnums=(0, 1, 2)))(*inputs)
    reference_gradients = jax.jit(jax.grad(objective(reference), argnums=(0, 1, 2)))(*inputs)
    for got, expected in zip(gradients, reference_gradients, strict=True):
        np.testing.assert_allclose(got, expected, atol=tol, rtol=tol)


@pytest.mark.parametrize("top_k", [1, 2, 20])
def test_moba_matches_independent_block_reference_and_qkv_gradients(top_k):
    inputs = _inputs()
    valid = jnp.asarray([[1] * 18 + [0], [0] + [1] * 17 + [0]], bool)
    labels = jnp.asarray([[0] * 4 + [9] * 10 + [0] * 5, [0] * 19])
    options = replace(_options("moba"), moba_top_k=top_k)
    allowed = _moba_mask(*inputs[:2], valid, labels, options.moba_block_size, top_k)
    _assert_value_and_grads(
        _run(options, valid, labels), lambda q, k, v: _dense_reference(q, k, v, allowed), inputs,
    )


@pytest.mark.parametrize("method", ["moba"])
def test_dense_limits_and_padded_rows_match_dense_attention(method):
    inputs = _inputs(length=11)
    valid = jnp.asarray([[1] * 9 + [0, 0], [0] * 11], bool)
    labels = jnp.asarray([[0] * 3 + [2] * 4 + [0] * 4, [0] * 11])
    options = replace(_options(method), moba_top_k=100)
    mask = make_causal_segment_mask(valid, labels)
    _assert_value_and_grads(
        _run(options, valid, labels),
        lambda q, k, v: vanilla_attention(q, k, v, mask, scale=0.5), inputs,
    )


@pytest.mark.parametrize("method", ["moba"])
def test_future_qkv_and_padding_cannot_change_any_prefix(method):
    inputs = _inputs(length=17, batch=1)
    valid = jnp.ones((1, 17), bool)
    run = jax.jit(_run(_options(method), valid))
    baseline = run(*inputs)
    # Every cut, including cuts inside pools, routing blocks and query tiles.
    for cut in range(1, 17):
        changed = tuple(x.at[:, cut:].set((index + 1) * 500.0) for index, x in enumerate(inputs))
        np.testing.assert_array_equal(run(*changed)[:, :cut], baseline[:, :cut])
    changed_mask = valid.at[:, 13:].set(False)
    shortened = _run(_options(method), changed_mask)(*inputs)
    np.testing.assert_allclose(shortened[:, :13], baseline[:, :13], atol=1e-6)
    np.testing.assert_array_equal(shortened[:, 13:], 0)


@pytest.mark.parametrize("method", ["moba"])
def test_packing_isolation_and_no_gradient_through_other_samples_or_future(method):
    inputs = _inputs(length=23, batch=1)
    valid = jnp.asarray([[1] * 21 + [0, 0]], bool)
    labels = jnp.asarray([[0] * 5 + [8] * 12 + [0] * 6])
    run = jax.jit(_run(_options(method), valid, labels))
    baseline = run(*inputs)
    for begin, end in ((0, 5), (5, 17), (17, 23)):
        changed = tuple(x.at[:, begin:end].add(250) for x in inputs)
        result = run(*changed)
        np.testing.assert_array_equal(result[:, :begin], baseline[:, :begin])
        np.testing.assert_array_equal(result[:, end:], baseline[:, end:])
    grad = jax.jit(jax.grad(lambda q, k, v: run(q, k, v)[:, 12].sum(), argnums=(0, 1, 2)))(*inputs)
    for value in grad:
        np.testing.assert_array_equal(value[:, :5], 0)
        np.testing.assert_array_equal(value[:, 13:], 0)
        assert jnp.all(jnp.isfinite(value))


@pytest.mark.parametrize("method", ["moba"])
@pytest.mark.parametrize("layout", [(1, 1, 1, 2, 2), (2, 1, 1, 2, 1), (1, 2, 1, 1, 2)])
def test_mesh_output_and_shared_kv_gradients_match_unsharded(method, layout):
    if jax.device_count() < 4:
        if layout != (1, 1, 1, 2, 2):
            pytest.skip("requires four devices for DP/FSDP combinations")
        layout = (1, 1, 1, 1, 1)
    mesh = Mesh(np.asarray(jax.devices()[:np.prod(layout)]).reshape(layout), ("dp", "fsdp", "ep", "tp", "sp"))
    inputs = _inputs(length=24, batch=2)
    valid = jnp.asarray([[1] * 23 + [0]] * 2, bool)
    labels = jnp.asarray([[0] * 7 + [2] * 17] * 2)
    options = _options(method)
    _assert_value_and_grads(_run(options, valid, labels, mesh), _run(options, valid, labels), inputs, tol=6e-5)
    sharding = NamedSharding(mesh, P(("dp", "fsdp"), "sp", "tp", None))
    compiled = jax.jit(_run(options, valid, labels, mesh), out_shardings=sharding)
    assert compiled(*inputs).sharding.is_equivalent_to(sharding, ndim=4)


def test_methods_reject_non_mqa_and_stochastic_attention_dropout():
    q, k, v = _inputs(length=8, batch=1)
    for method in ("moba",):
        with pytest.raises(ValueError, match="MQA"):
            _run(_options(method), jnp.ones((1, 8), bool))(q, jnp.repeat(k, 2, axis=2), jnp.repeat(v, 2, axis=2))
        with pytest.raises(ValueError, match="dropout"):
            attention(q, k, v, jnp.ones((1, 8), bool), implementation="vanilla",
                      attention_options=_options(method), dropout_rate=0.1, deterministic=False)


@pytest.mark.parametrize("method", ["moba"])
def test_million_token_forward_and_vjp_trace_without_a_dense_backend(method):
    # Shape-only: does not allocate Q/K/V, claim device capacity, or time 1M.
    sequence = 1_048_576
    inputs = (
        jax.ShapeDtypeStruct((1, sequence, 32, 128), jnp.bfloat16),
        jax.ShapeDtypeStruct((1, sequence, 1, 128), jnp.bfloat16),
        jax.ShapeDtypeStruct((1, sequence, 1, 128), jnp.bfloat16),
    )
    def objective(q, k, v):
        out = attention(
            q, k, v, jnp.ones((1, sequence), jnp.bool_), implementation="efficient",
            attention_options=AttentionOptions(method=method),
        )
        return jnp.mean(out.astype(jnp.float32) ** 2)
    _, grads = jax.eval_shape(jax.value_and_grad(objective, argnums=(0, 1, 2)), *inputs)
    assert tuple(x.shape for x in grads) == tuple(x.shape for x in inputs)
