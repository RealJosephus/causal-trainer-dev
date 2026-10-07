"""CPU reference gates for internal decoder tiling, including packed documents."""

from dataclasses import replace
from inspect import signature

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

from causal_trainer.modeling.architecture import (
    _grouped_layer_scan,
    decoder_forward,
    forward,
    init_params,
)
from causal_trainer.modeling.config import ModelConfig
from causal_trainer.training.losses import chunked_causal_lm_loss


def _mesh():
    tp, sp = (2, 2) if jax.device_count() >= 4 else (1, 1)
    return Mesh(np.asarray(jax.devices()[: tp * sp]).reshape(1, 1, 1, tp, sp), ("dp", "fsdp", "ep", "tp", "sp"))


def _config():
    return ModelConfig(
        vocab_size=32, hidden_size=16, intermediate_size=32, num_hidden_layers=4,
        num_attention_heads=4, num_key_value_heads=1, head_dim=4,
        partial_rotary_factor=0.5, max_position_embeddings=128,
        pad_token_id=0, eos_token_id=2, rope_theta=10_000.0,
    )


def _batch():
    # Different document lengths, reused labels, a boundary crossing the
    # second SP shard, and padding. Logical RoPE positions are independent of
    # physical causal positions, as in EndPrompt/prepared examples.
    ids = jnp.asarray([[1, 3, 5, 7, 9, 11, 4, 6, 8, 10, 12, 14, 16, 18, 0, 0]], jnp.int32)
    return {
        "input_ids": ids,
        "attention_mask": ids != 0,
        "segment_ids": jnp.asarray([[0, 0, 0, 2, 2, 2, 2, 2, 2, 2, 0, 0, 0, 0, -1, -1]], jnp.int32),
        "position_ids": jnp.asarray([[0, 1, 7, 0, 1, 2, 3, 50, 51, 52, 0, 1, 30, 31, 0, 0]], jnp.int32),
        "loss_weights": jnp.asarray([[0, 1, .5, 0, .25, 1, 1, 0, .5, 1, 0, 1, 1, .25, 0, 0]], jnp.float32),
    }


def _hidden(params, config, batch, mesh, enabled):
    return decoder_forward(
        params, config, batch["input_ids"], batch["attention_mask"],
        position_ids=batch["position_ids"], segment_ids=batch["segment_ids"],
        implementation="vanilla", mesh=mesh, compute_dtype=jnp.bfloat16,
        remat_policy="nothing_saveable", scan_layers=True,
        projection_chunk_size=3 if enabled else 0, mlp_chunk_size=3 if enabled else 0,
        remat_group_size=2 if enabled else 1,
    )


def test_packed_tiled_decoder_loss_and_all_parameter_gradients_match_reference():
    config, batch, mesh = _config(), _batch(), _mesh()
    params = init_params(config, jax.random.key(124), jnp.bfloat16)

    def objective(parameters, enabled):
        hidden = _hidden(parameters, config, batch, mesh, enabled)
        nll, weight = chunked_causal_lm_loss(
            hidden, parameters["lm_head"]["kernel"], batch["input_ids"], batch["attention_mask"],
            segment_ids=batch["segment_ids"], loss_weights=batch["loss_weights"],
            compute_dtype=jnp.bfloat16, loss_implementation="xla", mesh=mesh, token_budget=8,
        )
        return nll / weight, weight

    reference, reference_grads = jax.jit(jax.value_and_grad(lambda p: objective(p, False), has_aux=True))(params)
    actual, actual_grads = jax.jit(jax.value_and_grad(lambda p: objective(p, True), has_aux=True))(params)
    np.testing.assert_array_equal(actual[1], reference[1])
    np.testing.assert_allclose(actual[0], reference[0], atol=2e-3, rtol=2e-3)
    for path, ref in jax.tree_util.tree_flatten_with_path(reference_grads)[0]:
        got = actual_grads
        for key in path:
            got = got[key.idx] if isinstance(key, jax.tree_util.SequenceKey) else got[key.key]
        ref, got = np.asarray(ref, np.float32), np.asarray(got, np.float32)
        # TP reduction and BF16 tile accumulation may change rounding. Compare
        # each leaf's error energy, rather than hide small missing projections
        # behind a single full-model absolute tolerance.
        relative_error = np.linalg.norm(got - ref) / max(np.linalg.norm(ref), 1e-6)
        assert relative_error < 0.09, (jax.tree_util.keystr(path), relative_error)


def test_tiled_packing_prevents_cross_document_leakage_and_keeps_hidden_shards():
    config, batch, mesh = _config(), _batch(), _mesh()
    params = init_params(config, jax.random.key(125), jnp.bfloat16)
    output_sharding = NamedSharding(mesh, P("dp", "sp", "tp"))
    run = jax.jit(lambda b: _hidden(params, config, b, mesh, True), out_shardings=output_sharding)
    baseline = run(batch)
    changed = {**batch, "input_ids": batch["input_ids"].at[:, :3].set(jnp.asarray([[20, 21, 22]]))}
    modified = run(changed)
    np.testing.assert_array_equal(np.asarray(baseline[:, 3:], np.float32), np.asarray(modified[:, 3:], np.float32))
    assert baseline.sharding.is_equivalent_to(output_sharding, ndim=3)


def test_grouped_scan_preserves_layer_and_input_gradients():
    weights = jax.random.normal(jax.random.key(126), (12, 8, 8)) * .05
    inputs = jax.random.normal(jax.random.key(127), (2, 8))

    @jax.checkpoint
    def layer(x, weight):
        return x + jnp.tanh(x @ weight), None

    def ordinary(w, x):
        return jnp.sum(jax.lax.scan(layer, x, w)[0] ** 2)

    def grouped(w, x):
        return jnp.sum(_grouped_layer_scan(layer, x, w, 3) ** 2)
    expected = jax.jit(jax.value_and_grad(ordinary, argnums=(0, 1)))(weights, inputs)
    actual = jax.jit(jax.value_and_grad(grouped, argnums=(0, 1)))(weights, inputs)
    for got, ref in zip(jax.tree.leaves(actual), jax.tree.leaves(expected), strict=True):
        np.testing.assert_allclose(got, ref, atol=1e-5, rtol=1e-5)


def test_group_checkpoint_retains_only_group_boundaries(capsys):
    from jax.ad_checkpoint import print_saved_residuals

    @jax.checkpoint
    def layer(x, weight):
        return x + jnp.tanh(x @ weight), None

    weights, inputs = jnp.ones((12, 16, 16)), jnp.ones((11, 16))
    print_saved_residuals(lambda w, x: jax.lax.scan(layer, x, w)[0], weights, inputs)
    ordinary = capsys.readouterr().out
    print_saved_residuals(lambda w, x: _grouped_layer_scan(layer, x, w, 3), weights, inputs)
    grouped = capsys.readouterr().out
    # Inspect the actual AD residual set, not the number of source-code loops.
    assert "f32[12,11,16]" in ordinary
    assert "f32[12,11,16]" not in grouped
    assert "f32[4,11,16]" in grouped


@pytest.mark.parametrize("override,message", [
    ({"scan_layers": False}, "scan_layers"),
    ({"remat_policy": "none"}, "remat_policy"),
    ({"remat_group_size": 3}, "divide"),
    ({"compute_dtype": jnp.float32}, "bfloat16"),
    ({"projection_chunk_size": -1}, "projection_chunk_size"),
    ({"projection_chunk_size": True}, "projection_chunk_size"),
    ({"remat_group_size": 0}, "remat_group_size"),
    ({"remat_group_size": True}, "remat_group_size"),
])
def test_decoder_rejects_invalid_explicit_mechanism_options(override, message):
    config = _config()
    params = init_params(config, jax.random.key(129), jnp.bfloat16)
    kwargs = dict(
        implementation="vanilla", mesh=_mesh(), scan_layers=True,
        remat_policy="nothing_saveable", projection_chunk_size=3, remat_group_size=2,
    ) | override
    with pytest.raises((ValueError, TypeError), match=message):
        decoder_forward(params, config, _batch()["input_ids"], **kwargs)


def test_omitted_mechanism_options_match_explicit_disabled_defaults(monkeypatch):
    from causal_trainer.modeling import architecture

    config = replace(_config(), num_hidden_layers=1)
    params = init_params(config, jax.random.key(130), jnp.bfloat16)

    def fail_if_enabled(*args, **kwargs):
        raise AssertionError("default calls must not select extra tiling or checkpoint groups")

    monkeypatch.setattr(architecture, "_tiled_decoder_layer", fail_if_enabled)
    monkeypatch.setattr(architecture, "_grouped_layer_scan", fail_if_enabled)
    kwargs = dict(
        implementation="vanilla", compute_dtype=jnp.bfloat16, mesh=_mesh(),
        scan_layers=True, remat_policy="nothing_saveable",
    )
    expected = decoder_forward(params, config, _batch()["input_ids"], **kwargs)
    actual = decoder_forward(
        params, config, _batch()["input_ids"], **kwargs, projection_chunk_size=0, remat_group_size=1,
    )
    np.testing.assert_array_equal(np.asarray(actual, np.float32), np.asarray(expected, np.float32))
    assert set(signature(decoder_forward).parameters) == {
        "params", "config", "input_ids", "attention_mask", "position_ids", "segment_ids",
        "implementation", "mesh", "deterministic", "dropout_rng", "lora_params", "remat_policy",
        "compute_dtype", "block_q", "block_k", "mlp_chunk_size", "gather_kv_projection_weights", "scan_layers",
        "projection_chunk_size", "remat_group_size",
        "attention_options", "attention_step",
    }
    assert signature(decoder_forward).parameters["attention_options"].default is None
    assert signature(decoder_forward).parameters["attention_step"].default == 0
    for function in (decoder_forward, forward):
        assert signature(function).parameters["projection_chunk_size"].default == 0
        assert signature(function).parameters["remat_group_size"].default == 1


@pytest.mark.parametrize("mlp_chunk_size", [0, 2])
def test_projection_tiling_and_mlp_size_work_without_scan_or_layer_rematerialization(monkeypatch, mlp_chunk_size):
    from causal_trainer.modeling import architecture

    config, batch, mesh = replace(_config(), num_hidden_layers=1), _batch(), _mesh()
    params = init_params(config, jax.random.key(131), jnp.bfloat16)
    recorded_mlp_chunks = []
    original_mlp = architecture.tiled_norm_mlp

    def record_mlp(*args, **kwargs):
        recorded_mlp_chunks.append(kwargs["chunk_size"])
        return original_mlp(*args, **kwargs)

    monkeypatch.setattr(architecture, "tiled_norm_mlp", record_mlp)
    kwargs = dict(
        attention_mask=batch["attention_mask"], position_ids=batch["position_ids"], segment_ids=batch["segment_ids"],
        implementation="vanilla", mesh=mesh, scan_layers=False, remat_policy="none",
    )
    expected = jax.jit(lambda p: decoder_forward(p, config, batch["input_ids"], **kwargs))(params)
    actual = jax.jit(lambda p: decoder_forward(
        p, config, batch["input_ids"], **kwargs, projection_chunk_size=3, mlp_chunk_size=mlp_chunk_size,
    ))(params)
    np.testing.assert_allclose(np.asarray(actual, np.float32), np.asarray(expected, np.float32), atol=.03, rtol=.03)
    expected_mlp_chunk = mlp_chunk_size or batch["input_ids"].shape[1] // int(mesh.shape["sp"])
    assert recorded_mlp_chunks and set(recorded_mlp_chunks) == {expected_mlp_chunk}


@pytest.mark.parametrize("dtype", [jnp.bfloat16, jnp.float32])
def test_group_rematerialization_is_independent_of_projection_tiling(dtype):
    config, batch, mesh = _config(), _batch(), _mesh()
    params = init_params(config, jax.random.key(132), dtype)

    def objective(parameters, group_size):
        hidden = decoder_forward(
            parameters, config, batch["input_ids"], batch["attention_mask"],
            position_ids=batch["position_ids"], segment_ids=batch["segment_ids"],
            implementation="vanilla", mesh=mesh, compute_dtype=dtype,
            remat_policy="nothing_saveable", scan_layers=True, remat_group_size=group_size,
        )
        return jnp.sum(hidden.astype(jnp.float32) ** 2)

    expected = jax.jit(jax.value_and_grad(lambda p: objective(p, 1)))(params)
    actual = jax.jit(jax.value_and_grad(lambda p: objective(p, 2)))(params)
    for got, ref in zip(jax.tree.leaves(actual), jax.tree.leaves(expected), strict=True):
        np.testing.assert_array_equal(np.asarray(got, np.float32), np.asarray(ref, np.float32))


def test_projection_tiling_supports_one_sequence_partition():
    config, batch = replace(_config(), num_hidden_layers=1), _batch()
    tp = 4 if jax.device_count() >= 4 else 1
    mesh = Mesh(np.asarray(jax.devices()[:tp]).reshape(1, 1, 1, tp, 1), ("dp", "fsdp", "ep", "tp", "sp"))
    params = init_params(config, jax.random.key(134), jnp.bfloat16)
    kwargs = dict(implementation="vanilla", mesh=mesh, compute_dtype=jnp.bfloat16)
    expected = jax.jit(lambda p: decoder_forward(p, config, batch["input_ids"], **kwargs))(params)
    actual = jax.jit(lambda p: decoder_forward(
        p, config, batch["input_ids"], **kwargs, projection_chunk_size=3,
    ))(params)
    np.testing.assert_allclose(np.asarray(actual, np.float32), np.asarray(expected, np.float32), atol=.03, rtol=.03)


def test_forward_passes_both_mechanism_options_to_decoder():
    config, batch, mesh = _config(), _batch(), _mesh()
    params = init_params(config, jax.random.key(133), jnp.bfloat16)
    kwargs = dict(
        implementation="vanilla", mesh=mesh, compute_dtype=jnp.bfloat16, scan_layers=True,
        remat_policy="nothing_saveable", projection_chunk_size=3, mlp_chunk_size=2, remat_group_size=2,
    )
    hidden = decoder_forward(params, config, batch["input_ids"], **kwargs)
    logits = forward(params, config, batch["input_ids"], **kwargs)
    expected = jnp.einsum("bsh,hv->bsv", hidden, params["lm_head"]["kernel"])
    np.testing.assert_array_equal(np.asarray(logits, np.float32), np.asarray(expected, np.float32))
