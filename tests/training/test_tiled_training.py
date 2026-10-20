"""Hardware and CPU-interpreter gates for explicit training memory controls.

This small correctness gate does not measure hardware capacity or throughput.
Both branches execute Efficient attention, Cut CE and a complete AdamW update.
"""

from __future__ import annotations

import functools
from itertools import pairwise

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import Mesh

from causal_trainer.distributed.runtime import (
    batch_sharding,
    named_shardings,
    parameter_partition_specs,
    path_to_string,
    replicated_sharding,
)
from causal_trainer.kernels import cut_cross_entropy as cut_module
from causal_trainer.kernels import efficient_attention as efficient_module
from causal_trainer.modeling.architecture import init_params
from causal_trainer.modeling.config import ModelConfig
from causal_trainer.training.optimizer import build_learning_rate_schedule, build_optimizer
from causal_trainer.training.steps import (
    infer_optimizer_state_shardings,
    make_train_step,
    optimizer_state_template,
)


def _has_four_local_tpus() -> bool:
    try:
        return jax.default_backend() == "tpu" and jax.local_device_count() >= 4
    except RuntimeError:
        return False


def _host_tree(tree):
    return jax.tree.map(lambda value: np.asarray(jax.device_get(value)).copy(), tree)


def _leaf_map(tree):
    return {
        path_to_string(path): np.asarray(value, dtype=np.float32)
        for path, value in jax.tree_util.tree_flatten_with_path(tree)[0]
    }


def _assert_relative_l2(actual, expected, *, limit, floor, label):
    assert np.isfinite(actual).all(), label
    assert np.isfinite(expected).all(), label
    denominator = max(float(np.linalg.norm(expected)), floor * np.sqrt(expected.size))
    error = float(np.linalg.norm(actual - expected)) / denominator
    assert error <= limit, f"{label}: relative L2 error {error:.6f} exceeds {limit}"


def _assert_full_update_matches_ordinary_scan(
    *, tp=2, sp=2, sequence=512, intermediate_size=1024, vocab_size=512, hybrid=False,
):
    devices = np.asarray(jax.local_devices()[:tp * sp], dtype=object).reshape(1, 1, 1, tp, sp)
    mesh = Mesh(devices, ("dp", "fsdp", "ep", "tp", "sp"))
    config = ModelConfig(
        hidden_size=512, intermediate_size=intermediate_size, num_attention_heads=4,
        num_key_value_heads=1, head_dim=128, num_hidden_layers=4,
        vocab_size=vocab_size, max_position_embeddings=sequence, pad_token_id=0, eos_token_id=1,
    )

    # Three records of different lengths. Record 2 crosses the SP boundary;
    # its supervised tail requires gradients through keys on the earlier SP
    # owner. The small records, head/tail padding and fractional weights share
    # the same row, as in ordinary packed mixed-length training.
    segments = np.zeros((1, sequence), dtype=np.int32)
    positions = np.zeros_like(segments)
    boundaries = [sequence * numerator // 32 for numerator in (1, 5, 25, 29)]
    for segment, (start, end) in enumerate(pairwise(boundaries), start=1):
        segments[:, start:end] = segment
        positions[:, start:end] = np.arange(end - start)
    valid = segments > 0
    tokens = np.random.default_rng(19).integers(2, config.vocab_size, size=(1, sequence), dtype=np.int32)
    tokens[~valid] = config.pad_token_id
    weights = np.zeros((1, sequence), dtype=np.float32)
    weights[:, sequence * 19 // 32:boundaries[2]] = 1.0
    weights[:, sequence * 27 // 32:boundaries[3]] = 0.25
    hole = sequence * 21 // 32
    weights[:, hole:hole + 3] = 0.0
    host_batch = {
        "input_ids": tokens, "attention_mask": valid, "position_ids": positions,
        "segment_ids": segments, "loss_weights": weights,
    }
    if hybrid:
        # A bidirectional island crossing SP shards, plus ordinary AR targets
        # in the final packed record. This gate exercises the device programs;
        # ChatML parsing and corruption are covered by the data-path tests.
        start, end = sequence * 8 // 32, boundaries[2]
        ends = np.arange(sequence, dtype=np.int32)[None, :].copy()
        ends[:, start:end] = end - 1
        targets = np.zeros_like(tokens)
        targets[:, :-1] = tokens[:, 1:]
        aligned = np.zeros_like(weights)
        aligned[:, :-1] = weights[:, 1:] * (segments[:, :-1] == segments[:, 1:])
        aligned[:, start - 1:end] = 0
        masked = np.arange(start, end, 2)
        targets[:, masked] = tokens[:, masked]
        aligned[:, masked] = weights[:, masked]
        tokens[:, masked] = vocab_size - 1
        host_batch.update(attention_end=ends, prediction_targets=targets, prediction_weights=aligned)
    initial_host = _host_tree(init_params(config, jax.random.PRNGKey(23), jnp.bfloat16))
    parameter_shardings = named_shardings(parameter_partition_specs(initial_host), mesh)
    scalar_sharding = replicated_sharding(mesh)
    input_sharding = batch_sharding(mesh)
    schedule = build_learning_rate_schedule("constant", 0.002, 0.002, total_steps=2)
    optimizer = build_optimizer(
        initial_host, schedule, weight_decay=0.0, max_grad_norm=0.0,
        # Stabilize Adam's first update near zero so BF16 cancellation is not
        # amplified into a sign-sized learning-rate jump; moments below still
        # check the actual gradient independently of the update denominator.
        beta1=0.9, beta2=0.999, epsilon=1e-3,
    )
    with jax.set_mesh(mesh):
        # A durable host snapshot owns the shared initial values. Each branch
        # receives fresh device arrays because make_train_step donates both
        # parameters and optimizer state; no donated buffer is reused.
        initial_device = jax.device_put(jax.tree.map(np.copy, initial_host), parameter_shardings)
        state_template = optimizer_state_template(optimizer, initial_device)
        state_shardings = infer_optimizer_state_shardings(state_template, initial_device, scalar_sharding)
        initial_state_host = _host_tree(optimizer.init(initial_device))
        initial_device = None

        def run(*, tiled):
            params = jax.device_put(jax.tree.map(np.copy, initial_host), parameter_shardings)
            state = jax.device_put(jax.tree.map(np.copy, initial_state_host), state_shardings)
            batch = jax.device_put(jax.tree.map(np.copy, host_batch), input_sharding)
            step = make_train_step(
                config, mesh, optimizer, schedule,
                attention_implementation="efficient", gradient_accumulation_steps=1,
                remat_policy="nothing_saveable", compute_dtype=jnp.bfloat16,
                block_q=128, block_k=128, loss_token_budget=128, loss_implementation="cut",
                mlp_chunk_size=128 if tiled else 0, scan_layers=True, sparse_loss_skip=True,
                param_shardings=parameter_shardings, optimizer_state_shardings=state_shardings,
                batch_named_sharding=input_sharding, replicated_named_sharding=scalar_sharding,
                **({
                    "projection_chunk_size": 128, "remat_group_size": 2, "cut_chunk_size": 128,
                } if tiled else {}),
            )
            arguments = (params, state, batch, jax.device_put(np.int32(0), scalar_sharding))
            # Trace and compile each execution choice independently, with
            # fresh device arrays for the donated parameters and Adam state.
            compiled = step.lower(*arguments).compile()
            return _host_tree(compiled(*arguments))

        expected_params, expected_state, expected_metrics = run(tiled=False)
        actual_params, actual_state, actual_metrics = run(tiled=True)

    for name in ("loss", "nll_sum", "grad_norm"):
        np.testing.assert_allclose(
            actual_metrics[name], expected_metrics[name], rtol=0.04, atol=2e-3,
            err_msg=f"training metric {name}",
        )
    for name in ("token_count", "learning_rate"):
        np.testing.assert_array_equal(actual_metrics[name], expected_metrics[name])
    assert actual_metrics["token_count"] > 0

    initial = _leaf_map(initial_host)
    expected = _leaf_map(expected_params)
    actual = _leaf_map(actual_params)
    assert initial.keys() == expected.keys() == actual.keys()
    changed = set()
    for path in initial:
        # Check every leaf, including norms and biases, then compare updates
        # separately so unchanged base weights cannot hide gradient errors.
        expected_delta, actual_delta = expected[path] - initial[path], actual[path] - initial[path]
        assert np.isfinite(actual_delta).all(), path
        assert np.isfinite(expected_delta).all(), path
        # Mixed relative/RMS-absolute tolerance handles BF16 cancellation in
        # tiny bias gradients without letting the unchanged weights dominate.
        error = float(np.linalg.norm(actual_delta - expected_delta))
        bound = 0.12 * float(np.linalg.norm(expected_delta)) + 2e-5 * np.sqrt(expected_delta.size)
        assert error <= bound, f"parameter update {path}: L2 error {error:.6g} exceeds {bound:.6g}"
        if np.any(actual_delta != 0):
            changed.add(path)
    assert "embed_tokens/embedding" in changed and "lm_head/kernel" in changed
    for layer in range(config.num_hidden_layers):
        assert any(path.startswith(f"layers/{layer}/") and path.endswith("/kernel") for path in changed)

    expected_moments, actual_moments = _leaf_map(expected_state), _leaf_map(actual_state)
    assert expected_moments.keys() == actual_moments.keys()
    seen_moments = {"mu": 0, "nu": 0}
    counters = 0
    for path, expected_value in expected_moments.items():
        actual_value = actual_moments[path]
        if "/mu/" in path or "/nu/" in path:
            moment = "mu" if "/mu/" in path else "nu"
            seen_moments[moment] += 1
            _assert_relative_l2(
                actual_value, expected_value,
                limit=0.10 if moment == "mu" else 0.16,
                floor=1e-7 if moment == "mu" else 1e-12,
                label=f"AdamW {path}",
            )
        else:
            np.testing.assert_array_equal(actual_value, expected_value, err_msg=path)
            if path.endswith("/count"):
                assert actual_value == 1
                counters += 1
    assert seen_moments == {"mu": len(initial), "nu": len(initial)}
    assert counters == 2  # Adam moments and learning-rate schedule each advanced.


@pytest.mark.skipif(
    not _has_four_local_tpus(),
    reason="requires four local TPU devices; this gate runs real Efficient and Cut kernels",
)
def test_tp2_sp2_explicit_tiling_packed_full_update_matches_ordinary_scan() -> None:
    _assert_full_update_matches_ordinary_scan()


@pytest.mark.parametrize("hybrid", [False, True])
def test_cpu_interpreter_tp1_sp4_explicit_tiling_packed_full_update_matches_ordinary_scan(monkeypatch, hybrid) -> None:
    """Check the complete arithmetic path; this does not validate TPU lowering."""

    if jax.default_backend() != "cpu" or jax.local_device_count() < 4:
        pytest.skip("requires CPU with XLA_FLAGS=--xla_force_host_platform_device_count=4")
    # Both kernel modules use the same Pallas module, so this single patch
    # interprets the actual attention and CE forward/backward tile programs.
    monkeypatch.setattr(
        efficient_module.pl, "pallas_call", functools.partial(efficient_module.pl.pallas_call, interpret=True),
    )
    # JAX 0.10's interpreter discharges scalar-prefetch refs and literals with
    # invariant VMA annotations, which conflicts with varying SP inputs.
    # Disabling this check changes shard_map transpose normalization, so TP=1
    # is deliberate: ordinary and chunked Cut then have identical replication
    # contracts. With TP>1, False defensively divides output cotangents by TP
    # but does not restore that factor for feature-sharded hidden inputs.
    # The independent TPU gate retains TP2/SP2 and production VMA;
    # this interpreter gate cannot establish TP2/SP2 hardware correctness.
    for module in (efficient_module, cut_module):
        monkeypatch.setattr(
            module, "shard_map", lambda fn, **kwargs: jax.shard_map(fn, **(kwargs | {"check_vma": False})),
        )
    monkeypatch.setattr(jax, "default_backend", lambda: "tpu")
    _assert_full_update_matches_ordinary_scan(
        tp=1, sp=4, sequence=512, intermediate_size=512, vocab_size=256, hybrid=hybrid,
    )
