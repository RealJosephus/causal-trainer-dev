"""Tiled Cut CE arithmetic and collective contract, including CPU interpretation."""

from __future__ import annotations

import functools
import importlib
import inspect

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

cut = importlib.import_module("causal_trainer.kernels.cut_cross_entropy")


def _mesh(tp=1, sp=1, dp=1):
    count = tp * sp * dp
    if jax.device_count() < count:
        pytest.skip("requires four devices; CPU: XLA_FLAGS=--xla_force_host_platform_device_count=4")
    devices = np.asarray(jax.devices()[:count], dtype=object).reshape(dp, 1, 1, tp, sp)
    return Mesh(devices, ("dp", "fsdp", "ep", "tp", "sp"))


@pytest.fixture
def interpret(monkeypatch):
    monkeypatch.setattr(cut.pl, "pallas_call", functools.partial(cut.pl.pallas_call, interpret=True))
    monkeypatch.setattr(cut.jax, "default_backend", lambda: "tpu")


@pytest.fixture
def reference_kernels(monkeypatch):
    """Keep production VMA checking and collectives; replace only TPU arithmetic."""

    monkeypatch.setattr(cut.jax, "default_backend", lambda: "tpu")

    def stats(hidden, kernel, targets, weights, **kwargs):
        logits = jnp.dot(hidden, kernel, preferred_element_type=jnp.float32)
        maximum = jnp.max(logits, axis=-1)
        total = jnp.sum(jnp.exp(logits - maximum[:, None]), axis=-1)
        columns = jnp.arange(kernel.shape[1])
        target = jnp.sum(jnp.where(columns == targets[:, None], logits, 0), axis=-1)
        return jnp.where(weights != 0, maximum, -jnp.inf), jnp.where(weights != 0, total, 0), target

    def gradients(hidden, kernel, targets, weights, lse, dy):
        logits = jnp.dot(hidden, kernel, preferred_element_type=jnp.float32)
        probabilities = jnp.where(weights[:, None] != 0, jnp.exp(logits - lse[:, None]), 0)
        target = (jnp.arange(kernel.shape[1]) == targets[:, None]).astype(jnp.float32)
        return ((probabilities - target) * (weights * dy)[:, None]).astype(hidden.dtype)

    def hidden_bwd(hidden, kernel, targets, weights, lse, dy, **kwargs):
        return jnp.dot(gradients(hidden, kernel, targets, weights, lse, dy), kernel.T,
                       preferred_element_type=jnp.float32).astype(kwargs.get("output_dtype") or hidden.dtype)

    def head_bwd(hidden, kernel, targets, weights, lse, dy, **kwargs):
        return jnp.dot(hidden.T, gradients(hidden, kernel, targets, weights, lse, dy),
                       preferred_element_type=jnp.float32).astype(kwargs.get("output_dtype") or kernel.dtype)

    monkeypatch.setattr(cut, "_cut_stats_pallas", stats)
    monkeypatch.setattr(cut, "_cut_hidden_bwd_pallas", hidden_bwd)
    monkeypatch.setattr(cut, "_cut_head_bwd_pallas", head_bwd)


def _reference(hidden, kernel, targets, weights):
    logits = jnp.dot(hidden, kernel, preferred_element_type=jnp.float32)
    logp = jax.nn.log_softmax(logits, axis=-1)
    safe_targets = jnp.where(weights != 0, targets, 0)
    nll = -jnp.take_along_axis(logp, safe_targets[..., None], axis=-1)[..., 0]
    return jnp.sum(jnp.where(weights != 0, weights * nll, 0), dtype=jnp.float32), jnp.sum(weights)


def _data(batch, sequence, vocab=258):
    rng = np.random.default_rng(460)
    hidden = jnp.asarray(rng.normal(scale=0.06, size=(batch, sequence, 128)), jnp.bfloat16)
    kernel = jnp.asarray(rng.normal(scale=0.05, size=(128, vocab)), jnp.bfloat16)
    targets = rng.integers(vocab, size=(batch, sequence), dtype=np.int32)
    weights = rng.choice([0.0, 0.25, 1.0, 2.0], size=targets.shape).astype(np.float32)
    # Inputs are already shifted: these zero weights represent left padding,
    # packed record boundaries, and inactive supervision spans.
    weights[:, :17] = 0
    weights[:, 71:72] = 0
    weights[:, 121:159] = 0
    targets[weights == 0] = -100
    return hidden, kernel, jnp.asarray(targets), jnp.asarray(weights)


@pytest.mark.parametrize(
    ("tp", "sp", "dp", "batch", "sparse", "trainable"),
    [(1, 1, 1, 2, False, True), (2, 2, 1, 1, True, True), (2, 2, 1, 1, False, False), (2, 1, 2, 2, True, True)],
)
def test_chunked_cut_packed_tail_forward_and_gradients(reference_kernels, tp, sp, dp, batch, sparse, trainable):
    mesh = _mesh(tp, sp, dp)
    hidden, kernel, targets, weights = _data(batch, 386)
    if sp > 1:
        # One whole SP shard contributes nothing, but must join the head
        # gradient reduction and every vocabulary collective nevertheless.
        weights = weights.at[:, :193].set(0)
    specs = (P("dp" if dp > 1 else None, "sp" if sp > 1 else None, "tp" if tp > 1 else None),
             P(None, "tp" if tp > 1 else None))
    hidden = jax.device_put(hidden, NamedSharding(mesh, specs[0]))
    kernel = jax.device_put(kernel, NamedSharding(mesh, specs[1]))

    def actual(h, k):
        return cut.pallas_cut_linear_cross_entropy(
            h, k, targets, weights, mesh=mesh,
            sparse_compaction=sparse, lm_head_trainable=trainable, local_chunk_size=128,
        )

    def expected(h, k):
        return _reference(h, k if trainable else jax.lax.stop_gradient(k), targets, weights)

    with jax.set_mesh(mesh):
        result, backward = jax.vjp(actual, hidden, kernel)
        actual_grads = backward((jnp.asarray(0.37, jnp.float32), jnp.asarray(0.0, jnp.float32)))
        reference, backward_ref = jax.vjp(expected, hidden, kernel)
        reference_grads = backward_ref((jnp.asarray(0.37, jnp.float32), jnp.asarray(0.0, jnp.float32)))
    np.testing.assert_allclose(result[0], reference[0], rtol=2e-5, atol=2e-3)
    np.testing.assert_array_equal(result[1], reference[1])
    for actual_grad, reference_grad in zip(actual_grads, reference_grads, strict=True):
        actual_float = np.asarray(actual_grad, dtype=np.float32)
        reference_float = np.asarray(reference_grad, dtype=np.float32)
        relative_l2 = np.linalg.norm(actual_float - reference_float) / max(np.linalg.norm(reference_float), 1e-12)
        assert relative_l2 < 0.025
    assert not np.any(np.asarray(actual_grads[0], dtype=np.float32)[np.asarray(weights == 0)])
    if not trainable:
        assert not np.any(np.asarray(actual_grads[1], dtype=np.float32))


def test_chunked_cut_cpu_pallas_interpreter(interpret):
    # Actual TPU tile arithmetic can run on CPU. Multi-device HLO
    # interpretation in JAX 0.10 cannot discharge varying Pallas refs with
    # VMA enabled, so distributed arithmetic is tested above with VMA=True.
    test_chunked_cut_packed_tail_forward_and_gradients(None, 1, 1, 1, 2, True, True)


def test_chunked_cut_all_masked_rows_and_frozen_head(interpret, monkeypatch):
    mesh = _mesh()
    hidden, kernel, targets, weights = _data(1, 199)
    weights = jnp.zeros_like(weights)

    def unexpected_head(*args, **kwargs):
        raise AssertionError("frozen head must not construct a dHead kernel")

    monkeypatch.setattr(cut, "_cut_head_bwd_pallas", unexpected_head)
    with jax.set_mesh(mesh):
        result, backward = jax.vjp(lambda h, k: cut.pallas_cut_linear_cross_entropy(
            h, k, targets, weights, mesh=mesh, sparse_compaction=True,
            lm_head_trainable=False, local_chunk_size=128,
        ), hidden, kernel)
        grads = backward((jnp.asarray(1.0), jnp.asarray(0.0)))
    assert float(result[0]) == float(result[1]) == 0
    for gradient in grads:
        assert not np.any(np.asarray(gradient, dtype=np.float32))


def test_chunked_cut_is_opt_in_and_rejects_invalid_sizes():
    assert inspect.signature(cut.pallas_cut_linear_cross_entropy).parameters["local_chunk_size"].default == 0
    mesh = _mesh()
    hidden, kernel, targets, weights = _data(1, 2)
    for invalid in (True, 1.5, "128", -1):
        with pytest.raises((TypeError, ValueError), match="local_chunk_size"):
            cut.pallas_cut_linear_cross_entropy(
                hidden, kernel, targets, weights, mesh=mesh, local_chunk_size=invalid,
            )


def _equations(jaxpr, parents=()):
    if hasattr(jaxpr, "jaxpr"):
        yield from _equations(jaxpr.jaxpr, parents)
    elif hasattr(jaxpr, "eqns"):
        for equation in jaxpr.eqns:
            yield equation, parents
            for value in equation.params.values():
                yield from _equations(value, (*parents, equation.primitive.name))
    elif isinstance(jaxpr, (tuple, list)):
        for value in jaxpr:
            yield from _equations(value, parents)


def test_chunked_cut_real_pallas_vma_shapes_and_collective_placement(monkeypatch):
    """Trace production Pallas abstract eval with VMA enabled, without lowering."""

    monkeypatch.setattr(cut.jax, "default_backend", lambda: "tpu")
    mesh = _mesh(tp=2, sp=2)
    hidden, kernel, targets, weights = _data(1, 386)

    def loss(h, k):
        return cut.pallas_cut_linear_cross_entropy(
            h, k, targets, weights, mesh=mesh, sparse_compaction=True, local_chunk_size=128,
        )[0]

    differentiated = jax.value_and_grad(loss, argnums=(0, 1))
    with jax.set_mesh(mesh):
        shape = jax.eval_shape(differentiated, hidden, kernel)
        graph = jax.make_jaxpr(differentiated)(hidden, kernel)
    assert shape[1][0].shape == hidden.shape
    assert shape[1][1].shape == kernel.shape
    equations = list(_equations(graph))
    gathers = [(equation, parents) for equation, parents in equations if equation.primitive.name == "all_gather"]
    assert gathers
    for equation, parents in gathers:
        assert "scan" in parents
        assert equation.invars[0].aval.shape == (128, 64)
    head_reductions = [
        (equation, parents) for equation, parents in equations
        if equation.primitive.name.startswith("psum")
        and "sp" in equation.params.get("axes", ())
        and equation.invars[0].aval.shape == (128, 129)
    ]
    assert len(head_reductions) == 1
    assert "scan" not in head_reductions[0][1]
    assert head_reductions[0][0].invars[0].aval.dtype == jnp.float32
    pallas_outputs = [
        outvar.aval for equation, _parents in equations
        if equation.primitive.name == "pallas_call" for outvar in equation.outvars
    ]
    assert pallas_outputs
    for aval in pallas_outputs:
        assert aval.manual_axis_type.varying == frozenset(("tp", "sp"))


def test_chunked_cut_tpu_hardware_gate(monkeypatch):
    """Exercise real Pallas lowering without the CPU interpreter when available."""

    if jax.default_backend() != "tpu":
        pytest.skip("requires TPU")
    test_chunked_cut_packed_tail_forward_and_gradients(None, 2, 2, 1, 1, True, True)
