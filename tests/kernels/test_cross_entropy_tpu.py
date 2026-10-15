from __future__ import annotations

from unittest import mock

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

from causal_trainer.kernels.cross_entropy import pallas_fused_sparse_cross_entropy
from causal_trainer.kernels.cut_cross_entropy import pallas_cut_linear_cross_entropy
from causal_trainer.kernels.fused_cross_entropy import fused_linear_cross_entropy
from causal_trainer.training.losses import (
    _fused_sparse_cross_entropy,
    causal_lm_loss,
    chunked_causal_lm_loss,
)


def _has_four_tpu_devices() -> bool:
    try:
        return jax.default_backend() == "tpu" and jax.local_device_count() >= 4
    except RuntimeError:
        return False


pytestmark = pytest.mark.skipif(
    not _has_four_tpu_devices(),
    reason="requires at least four TPU devices with JAX 0.10",
)


def _mesh() -> Mesh:
    devices = np.asarray(jax.local_devices()[:4], dtype=object).reshape(1, 1, 1, 4, 1)
    return Mesh(devices, ("dp", "fsdp", "ep", "tp", "sp"))


def _dp_tp_mesh() -> Mesh:
    devices = np.asarray(jax.local_devices()[:4], dtype=object).reshape(2, 1, 1, 2, 1)
    return Mesh(devices, ("dp", "fsdp", "ep", "tp", "sp"))


def _sp_tp_mesh() -> Mesh:
    devices = np.asarray(jax.local_devices()[:4], dtype=object).reshape(1, 1, 1, 2, 2)
    return Mesh(devices, ("dp", "fsdp", "ep", "tp", "sp"))


def test_pallas_vocab_parallel_value_and_gradient_match_xla() -> None:
    # TP=4 gives a local vocabulary of 320, deliberately not divisible by the
    # TPU's 128-wide DMA minor tile. This exercises the fixed-width local pad
    # used by the production 185600/4=46400 vocabulary as well as value/VJP.
    batch, sequence, vocab = 1, 256, 1280
    logits = (jax.random.normal(jax.random.PRNGKey(0), (batch, sequence, vocab)) * 0.25).astype(
        jnp.bfloat16
    )
    targets = jax.random.randint(
        jax.random.PRNGKey(1),
        (batch, sequence),
        0,
        vocab,
        dtype=jnp.int32,
    )
    # Target 220 becomes local id -100 on TP shard 1.  It must remain a valid
    # row rather than colliding with the public ignore-index convention.
    targets = targets.at[0, 0].set(220)
    weights = jnp.where(
        jnp.arange(sequence)[None, :] < 96,
        jnp.asarray(0.5, jnp.float32),
        jnp.asarray(0.0, jnp.float32),
    )

    def reference(x):
        return jnp.sum(_fused_sparse_cross_entropy(x.astype(jnp.float32), targets, weights))

    mesh = _mesh()
    with jax.set_mesh(mesh):
        def actual(x):
            return jnp.sum(
                pallas_fused_sparse_cross_entropy(
                    x,
                    targets,
                    weights,
                    mesh=mesh,
                )
            )

        expected_value, expected_gradient = jax.value_and_grad(reference)(logits)
        actual_value, actual_gradient = jax.value_and_grad(actual)(logits)

    np.testing.assert_allclose(np.asarray(actual_value), np.asarray(expected_value), rtol=0.0, atol=8e-3)
    np.testing.assert_allclose(
        np.asarray(actual_gradient),
        np.asarray(expected_gradient),
        rtol=0.0,
        atol=3e-3,
    )


def test_chunked_fused_linear_loss_matches_xla_for_hidden_and_head_gradients() -> None:
    batch, sequence, hidden_size, vocab = 2, 9, 16, 1024
    hidden = (
        jax.random.normal(jax.random.PRNGKey(2), (batch, sequence, hidden_size)) * 0.125
    ).astype(jnp.bfloat16)
    head = (
        jax.random.normal(jax.random.PRNGKey(3), (hidden_size, vocab)) * 0.125
    ).astype(jnp.bfloat16)
    targets = jax.random.randint(
        jax.random.PRNGKey(4),
        (batch, sequence),
        0,
        vocab,
        dtype=jnp.int32,
    )
    weights = jnp.asarray(
        [
            [1.0, 1.0, 0.0, 0.5, 1.0, 1.0, 0.0, 1.0, 1.0],
            [1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.5, 0.0],
        ],
        dtype=jnp.float32,
    )

    def loss(hidden_states, lm_head, implementation, mesh=None):
        nll_sum, weight_sum = fused_linear_cross_entropy(
            hidden_states,
            lm_head,
            targets,
            weights,
            token_budget=4,
            compute_dtype=jnp.bfloat16,
            implementation=implementation,
            mesh=mesh,
            sparse_skip=True,
        )
        return nll_sum / jnp.maximum(weight_sum, 1.0)

    reference = jax.jit(jax.value_and_grad(lambda x, w: loss(x, w, "xla"), argnums=(0, 1)))
    expected_value, expected_gradients = reference(hidden, head)

    mesh = _mesh()
    with jax.set_mesh(mesh):
        actual = jax.jit(
            jax.value_and_grad(
                lambda x, w: loss(x, w, "pallas", mesh),
                argnums=(0, 1),
            )
        )
        actual_value, actual_gradients = actual(hidden, head)

    np.testing.assert_allclose(np.asarray(actual_value), np.asarray(expected_value), rtol=0.0, atol=2e-2)
    np.testing.assert_allclose(
        np.asarray(actual_gradients[0]),
        np.asarray(expected_gradients[0]),
        rtol=0.0,
        atol=5e-3,
    )
    np.testing.assert_allclose(
        np.asarray(actual_gradients[1]),
        np.asarray(expected_gradients[1]),
        rtol=0.0,
        atol=5e-3,
    )


@pytest.mark.parametrize("pattern", ["trailing", "inactive", "alternating"])
def test_pallas_sparse_skip_preserves_primal_loss_across_inactive_chunks(pattern: str) -> None:
    # The production token budget gives four sequence chunks here.  Only the
    # first is active, so three consecutive iterations take sparse_skip's false
    # branch.  This guards the TPU buffer-assignment failure that aliased the
    # accumulated NLL with the independent weight sum and returned loss 1.0.
    batch, sequence, hidden_size, vocab = 4, 128, 128, 1024
    hidden = (
        jax.random.normal(jax.random.PRNGKey(5), (batch, sequence, hidden_size)) * 0.05
    ).astype(jnp.bfloat16)
    head = (
        jax.random.normal(jax.random.PRNGKey(6), (hidden_size, vocab)) * 0.05
    ).astype(jnp.bfloat16)
    targets = jax.random.randint(
        jax.random.PRNGKey(7),
        (batch, sequence),
        0,
        vocab,
        dtype=jnp.int32,
    )
    positions = jnp.arange(sequence)[None, :]
    active = {
        "trailing": positions < 32,
        "inactive": positions < 0,
        "alternating": (positions // 32) % 2 == 0,
    }[pattern]
    weights = jnp.where(
        active,
        jnp.asarray(1.0, jnp.float32),
        jnp.asarray(0.0, jnp.float32),
    )
    weights = jnp.broadcast_to(weights, targets.shape)

    def loss_with_aux(x, implementation, mesh=None, *, sparse_skip):
        nll, count = fused_linear_cross_entropy(
            x,
            head,
            targets,
            weights,
            token_budget=128,
            compute_dtype=jnp.bfloat16,
            implementation=implementation,
            mesh=mesh,
            sparse_skip=sparse_skip,
        )
        return nll / jnp.maximum(count, 1.0), (nll, count)

    reference = jax.jit(
        jax.value_and_grad(
            lambda x: loss_with_aux(x, "xla", sparse_skip=False),
            has_aux=True,
        )
    )
    (expected_loss, expected_aux), expected_dhidden = reference(hidden)

    mesh = _mesh()
    with jax.set_mesh(mesh):
        actual = jax.jit(
            jax.value_and_grad(
                lambda x: loss_with_aux(x, "pallas", mesh, sparse_skip=True),
                has_aux=True,
            )
        )
        (actual_loss, actual_aux), actual_dhidden = actual(hidden)

    np.testing.assert_allclose(np.asarray(actual_loss), np.asarray(expected_loss), rtol=0.0, atol=2e-2)
    np.testing.assert_allclose(np.asarray(actual_aux[0]), np.asarray(expected_aux[0]), rtol=3e-3, atol=2.0)
    np.testing.assert_allclose(np.asarray(actual_aux[1]), np.asarray(expected_aux[1]), rtol=0.0, atol=0.0)
    np.testing.assert_allclose(
        np.asarray(actual_dhidden),
        np.asarray(expected_dhidden),
        rtol=0.0,
        atol=5e-3,
    )


def test_pallas_sparse_skip_production_head_and_chunk_count_compile() -> None:
    # One TP4 host reproduces the per-chip CE shape of B32/S8192/DP8/TP4.
    # Keep the real 4096x185600 head and 256 chunks: tiny heads conceal the
    # old conditional residual stack, which exhausted HBM during training.
    mesh = _mesh()
    hidden = jax.ShapeDtypeStruct(
        (4, 8192, 4096), jnp.bfloat16, sharding=NamedSharding(mesh, P(None, None, "tp")),
    )
    head = jax.ShapeDtypeStruct(
        (4096, 185600), jnp.bfloat16, sharding=NamedSharding(mesh, P(None, "tp")),
    )
    targets = jax.ShapeDtypeStruct((4, 8192), jnp.int32, sharding=NamedSharding(mesh, P()))
    weights = jax.ShapeDtypeStruct((4, 8192), jnp.float32, sharding=NamedSharding(mesh, P()))

    def loss(x, w, y, m):
        nll, count = fused_linear_cross_entropy(
            x, w, y, m, token_budget=128, implementation="pallas", mesh=mesh, sparse_skip=True,
        )
        return nll / jnp.maximum(count, 1.0)

    with jax.set_mesh(mesh):
        compiled = jax.jit(jax.value_and_grad(loss, argnums=(0, 1))).lower(
            hidden, head, targets, weights,
        ).compile()
    assert compiled.memory_analysis() is not None


def _cut_reference(hidden, head, targets, weights):
    logits = jax.lax.dot_general(
        hidden,
        head,
        dimension_numbers=(((2,), (0,)), ((), ())),
        preferred_element_type=jnp.float32,
    )
    per_token = _fused_sparse_cross_entropy(logits, targets, weights)
    return jnp.sum(per_token, dtype=jnp.float32), jnp.sum(weights, dtype=jnp.float32)


def test_cut_dp_tp_compaction_value_and_both_gradients_match_xla() -> None:
    batch, sequence, hidden_size, vocab = 2, 128, 128, 1024
    hidden = (
        jax.random.normal(jax.random.PRNGKey(10), (batch, sequence, hidden_size)) * 0.05
    ).astype(jnp.bfloat16)
    head = (
        jax.random.normal(jax.random.PRNGKey(11), (hidden_size, vocab)) * 0.05
    ).astype(jnp.bfloat16)
    targets = jax.random.randint(
        jax.random.PRNGKey(12),
        (batch, sequence),
        0,
        vocab,
        dtype=jnp.int32,
    )
    positions = jnp.arange(sequence)[None, :]
    # These zero rows stand in for padding, non-assistant turns, and packed
    # example boundaries.  Compaction must keep each hidden/target pair intact.
    active = (
        (jnp.arange(batch)[:, None] == 0)
        & (positions % 5 != 0)
        & (positions != 63)
        & (positions < 111)
    )
    active_weights = jnp.where(positions % 7 == 0, 0.25, 1.0).astype(jnp.float32)
    weights = jnp.where(active, active_weights, jnp.asarray(0.0, jnp.float32))
    # With DP=2 and one global row per shard, the second shard is completely
    # inactive. It must still enter the trainable-head DP psum schedule.
    # One complete DP shard is inactive.  It must still execute the fixed-shape
    # dHead path and participate in the DP psum rather than branch around it.
    weights = weights.at[1, :].set(0.0)

    def reference(x, w):
        nll, count = _cut_reference(x, w, targets, weights)
        return nll / jnp.maximum(count, 1.0)

    expected_value, expected_gradients = jax.value_and_grad(reference, argnums=(0, 1))(
        hidden,
        head,
    )
    mesh = _dp_tp_mesh()
    with jax.set_mesh(mesh):
        def actual(x, w):
            nll, count = pallas_cut_linear_cross_entropy(
                x,
                w,
                targets,
                weights,
                mesh=mesh,
                sparse_compaction=True,
                lm_head_trainable=True,
            )
            return nll / jnp.maximum(count, 1.0)

        actual_value, actual_gradients = jax.jit(
            jax.value_and_grad(actual, argnums=(0, 1))
        )(hidden, head)

    np.testing.assert_allclose(np.asarray(actual_value), np.asarray(expected_value), rtol=0.0, atol=2e-2)
    np.testing.assert_allclose(
        np.asarray(actual_gradients[0]),
        np.asarray(expected_gradients[0]),
        rtol=0.0,
        atol=6e-3,
    )
    np.testing.assert_allclose(
        np.asarray(actual_gradients[1]),
        np.asarray(expected_gradients[1]),
        rtol=0.0,
        atol=6e-3,
    )


def test_cut_sp_tp_value_and_both_gradients_match_xla() -> None:
    batch, sequence, hidden_size, vocab = 1, 256, 128, 1024
    hidden = (
        jax.random.normal(jax.random.PRNGKey(30), (batch, sequence, hidden_size)) * 0.05
    ).astype(jnp.bfloat16)
    head = (
        jax.random.normal(jax.random.PRNGKey(31), (hidden_size, vocab)) * 0.05
    ).astype(jnp.bfloat16)
    targets = jax.random.randint(
        jax.random.PRNGKey(32),
        (batch, sequence),
        0,
        vocab,
        dtype=jnp.int32,
    )
    positions = jnp.arange(sequence)[None, :]
    weights = jnp.where(
        (positions % 5 != 0) & (positions < 223),
        jnp.where(positions % 7 == 0, 0.25, 1.0),
        0.0,
    ).astype(jnp.float32)

    def reference(x, w):
        nll, count = _cut_reference(x, w, targets, weights)
        return nll / jnp.maximum(count, 1.0)

    expected_value, expected_gradients = jax.value_and_grad(reference, argnums=(0, 1))(
        hidden,
        head,
    )
    mesh = _sp_tp_mesh()
    with jax.set_mesh(mesh):
        def actual(x, w):
            nll, count = pallas_cut_linear_cross_entropy(
                x,
                w,
                targets,
                weights,
                mesh=mesh,
                sparse_compaction=True,
                lm_head_trainable=True,
            )
            return nll / jnp.maximum(count, 1.0)

        actual_value, actual_gradients = jax.jit(
            jax.value_and_grad(actual, argnums=(0, 1))
        )(hidden, head)

    np.testing.assert_allclose(np.asarray(actual_value), np.asarray(expected_value), rtol=0.0, atol=2e-2)
    np.testing.assert_allclose(
        np.asarray(actual_gradients[0]),
        np.asarray(expected_gradients[0]),
        rtol=0.0,
        atol=6e-3,
    )
    np.testing.assert_allclose(
        np.asarray(actual_gradients[1]),
        np.asarray(expected_gradients[1]),
        rtol=0.0,
        atol=6e-3,
    )


def test_sequence_parallel_shifted_cut_loss_matches_materialized_reference() -> None:
    batch, sequence, hidden_size, vocab = 1, 256, 128, 1024
    hidden = (
        jax.random.normal(jax.random.PRNGKey(33), (batch, sequence, hidden_size)) * 0.05
    ).astype(jnp.bfloat16)
    head = (
        jax.random.normal(jax.random.PRNGKey(34), (hidden_size, vocab)) * 0.05
    ).astype(jnp.bfloat16)
    input_ids = jax.random.randint(
        jax.random.PRNGKey(35),
        (batch, sequence),
        0,
        vocab,
        dtype=jnp.int32,
    )
    attention = jnp.ones_like(input_ids)
    segments = jnp.where(
        jnp.arange(sequence)[None, :] < 193,
        jnp.asarray(1, jnp.int32),
        jnp.asarray(2, jnp.int32),
    )
    loss_weights = jnp.where(
        jnp.arange(sequence)[None, :] % 7 == 0,
        jnp.asarray(0.25, jnp.float32),
        jnp.asarray(1.0, jnp.float32),
    )

    def reference(x, w):
        logits = jax.lax.dot_general(
            x,
            w,
            dimension_numbers=(((2,), (0,)), ((), ())),
            preferred_element_type=jnp.float32,
        )
        nll, count = causal_lm_loss(
            logits,
            input_ids,
            attention,
            segment_ids=segments,
            loss_mask=loss_weights,
        )
        return nll / jnp.maximum(count, 1.0)

    mesh = _sp_tp_mesh()

    def actual(x, w):
        nll, count = chunked_causal_lm_loss(
            x,
            w,
            input_ids,
            attention,
            segment_ids=segments,
            loss_weights=loss_weights,
            compute_dtype=jnp.bfloat16,
            loss_implementation="cut",
            mesh=mesh,
        )
        return nll / jnp.maximum(count, 1.0)

    expected_value, expected_gradients = jax.jit(
        jax.value_and_grad(reference, argnums=(0, 1))
    )(hidden, head)
    with jax.set_mesh(mesh):
        actual_value, actual_gradients = jax.jit(
            jax.value_and_grad(actual, argnums=(0, 1))
        )(hidden, head)

    np.testing.assert_allclose(np.asarray(actual_value), np.asarray(expected_value), rtol=0.0, atol=2e-2)
    np.testing.assert_allclose(
        np.asarray(actual_gradients[0]),
        np.asarray(expected_gradients[0]),
        rtol=0.0,
        atol=6e-3,
    )
    np.testing.assert_allclose(
        np.asarray(actual_gradients[1]),
        np.asarray(expected_gradients[1]),
        rtol=0.0,
        atol=6e-3,
    )
    assert np.any(np.asarray(actual_gradients[0][:, 127, :]) != 0.0)
    assert np.all(np.asarray(actual_gradients[0][:, 192, :]) == 0.0)
    assert np.all(np.asarray(actual_gradients[0][:, -1, :]) == 0.0)


def test_cut_frozen_head_does_not_construct_dhead_and_keeps_dhidden() -> None:
    batch, sequence, hidden_size, vocab = 2, 128, 128, 1024
    hidden = (
        jax.random.normal(jax.random.PRNGKey(20), (batch, sequence, hidden_size)) * 0.05
    ).astype(jnp.bfloat16)
    head = (
        jax.random.normal(jax.random.PRNGKey(21), (hidden_size, vocab)) * 0.05
    ).astype(jnp.bfloat16)
    targets = jax.random.randint(
        jax.random.PRNGKey(22),
        (batch, sequence),
        0,
        vocab,
        dtype=jnp.int32,
    )
    weights = jnp.where(
        (jnp.arange(sequence)[None, :] % 3 != 0) & (jnp.arange(sequence)[None, :] < 101),
        jnp.asarray(0.5, jnp.float32),
        jnp.asarray(0.0, jnp.float32),
    )
    weights = jnp.broadcast_to(weights, targets.shape)

    def reference(x):
        nll, count = _cut_reference(x, head, targets, weights)
        return nll / jnp.maximum(count, 1.0)

    expected_value, expected_dhidden = jax.value_and_grad(reference)(hidden)
    mesh = _dp_tp_mesh()
    with jax.set_mesh(mesh):
        def actual(x):
            # token_budget is intentionally smaller than the local batch: Cut
            # uses fixed VMEM tiles and must bypass logits-slab chunking.
            nll, count = fused_linear_cross_entropy(
                x,
                head,
                targets,
                weights,
                token_budget=1,
                compute_dtype=jnp.bfloat16,
                implementation="cut",
                mesh=mesh,
                sparse_skip=True,
                lm_head_trainable=False,
            )
            return nll / jnp.maximum(count, 1.0)

        # The frozen static branch must not merely rely on DCE.  Replacing the
        # Python dHead wrapper with a hard failure proves it is never traced.
        with mock.patch(
            "causal_trainer.kernels.cut_cross_entropy._cut_head_bwd_pallas",
            side_effect=AssertionError("frozen head traced dHead"),
        ):
            actual_value, actual_dhidden = jax.jit(jax.value_and_grad(actual))(hidden)

    np.testing.assert_allclose(np.asarray(actual_value), np.asarray(expected_value), rtol=0.0, atol=2e-2)
    np.testing.assert_allclose(
        np.asarray(actual_dhidden),
        np.asarray(expected_dhidden),
        rtol=0.0,
        atol=6e-3,
    )
