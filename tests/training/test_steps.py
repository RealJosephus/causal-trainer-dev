import inspect
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from jax.sharding import Mesh

from causal_trainer.training import steps


def _run_step_with_memory_options(
    monkeypatch, *, options=None, mesh_values=None, config_values=None,
    batch_size=1, param_dtype=jnp.bfloat16, head_dtype=None, mesh_override=None,
):
    decoder_calls, loss_calls = [], []

    def decoder(params, _config, input_ids, **kwargs):
        decoder_calls.append(kwargs)
        return jnp.broadcast_to(params["body"], (*input_ids.shape, 1))

    def loss(hidden, head, *_args, **kwargs):
        loss_calls.append(kwargs)
        return jnp.sum(hidden * head[0, 0]), jnp.asarray(hidden.shape[0] * hidden.shape[1], jnp.float32)

    monkeypatch.setattr(steps, "decoder_forward", decoder)
    monkeypatch.setattr(steps, "chunked_causal_lm_loss", loss)
    config = SimpleNamespace(**({"num_hidden_layers": 4, "attention_dropout": 0.0} | (config_values or {})))
    mesh = SimpleNamespace(shape={"dp": 1, "fsdp": 1, "ep": 1, "tp": 1, "sp": 2} | (mesh_values or {}))
    if mesh_override is not None:
        mesh = mesh_override
    params = {
        "body": jnp.asarray([1.0], param_dtype),
        "lm_head": {"kernel": jnp.asarray([[2.0]], param_dtype if head_dtype is None else head_dtype)},
    }
    optimizer = optax.sgd(0.125)
    settings = {
        "attention_implementation": "efficient", "gradient_accumulation_steps": 1,
        "remat_policy": "nothing_saveable", "compute_dtype": jnp.bfloat16,
        "block_q": 128, "block_k": 128, "loss_token_budget": 128,
        "loss_implementation": "cut", "mlp_chunk_size": 0, "sparse_loss_skip": False,
        "param_shardings": None, "optimizer_state_shardings": None,
        "batch_named_sharding": None, "replicated_named_sharding": None, "scan_layers": True,
    } | (options or {})
    train_step = steps.make_train_step(config, mesh, optimizer, lambda _: 0.125, **settings)
    ids = jnp.broadcast_to(jnp.asarray([[1, 2, 3, 4]], jnp.int32), (batch_size, 4))
    batch = {
        "input_ids": ids, "attention_mask": jnp.ones_like(ids, jnp.bool_),
        "position_ids": jnp.broadcast_to(jnp.asarray([[0, 1, 0, 1]], jnp.int32), ids.shape),
        "segment_ids": jnp.broadcast_to(jnp.asarray([[1, 1, 2, 2]], jnp.int32), ids.shape),
        "loss_weights": jnp.ones_like(ids, jnp.float32),
    }
    updated, _, metrics = train_step(params, optimizer.init(params), batch, jnp.asarray(0, jnp.int32))
    assert jnp.allclose(updated["body"], 0.75)
    assert jnp.allclose(updated["lm_head"]["kernel"], 1.875)
    assert jnp.allclose(metrics["loss"], 2.0)
    return decoder_calls[0], loss_calls[0]


def test_full_step_defaults_do_not_automatically_enable_memory_controls(monkeypatch):
    # These existing settings previously triggered the abandoned auto policy.
    decoder, loss = _run_step_with_memory_options(monkeypatch)
    assert "projection_chunk_size" not in decoder
    assert "remat_group_size" not in decoder
    assert "cut_chunk_size" not in loss
    assert decoder["mlp_chunk_size"] == 0
    signature = inspect.signature(steps.make_train_step).parameters
    assert signature["projection_chunk_size"].default == 0
    assert signature["remat_group_size"].default == 1
    assert signature["cut_chunk_size"].default == 0
    assert not any(name.startswith("long_context") for name in signature)


@pytest.mark.parametrize("mlp_chunk_size", [0, 2])
def test_default_full_step_keeps_tp_only_mesh_compatible(monkeypatch, mlp_chunk_size):
    mesh = Mesh(np.asarray(jax.devices()[:1]), ("tp",))
    decoder, loss = _run_step_with_memory_options(
        monkeypatch, mesh_override=mesh, options={"mlp_chunk_size": mlp_chunk_size},
    )
    assert "projection_chunk_size" not in decoder
    assert "remat_group_size" not in decoder
    assert "cut_chunk_size" not in loss


@pytest.mark.parametrize("mode", ["projection", "group", "cut", "all", "projection_accumulation"])
def test_explicit_memory_controls_are_independent_and_preserve_updates(monkeypatch, mode):
    kwargs = {}
    options = {}
    if mode == "projection":
        options = {"projection_chunk_size": 3, "scan_layers": False, "remat_policy": "none",
                   "loss_implementation": "xla", "mlp_chunk_size": 2}
        kwargs["mesh_values"] = {"sp": 1}
    elif mode == "group":
        options = {"remat_group_size": 2, "compute_dtype": jnp.float32,
                   "attention_implementation": "vanilla", "loss_implementation": "xla"}
        kwargs.update(param_dtype=jnp.float32, batch_size=2, mesh_values={"sp": 1})
    elif mode == "cut":
        options = {"cut_chunk_size": 5, "scan_layers": False, "remat_policy": "none", "compute_dtype": jnp.float32}
        kwargs.update(param_dtype=jnp.float32, batch_size=2, mesh_values={"dp": 2, "sp": 1})
    elif mode == "projection_accumulation":
        options = {"projection_chunk_size": 3, "gradient_accumulation_steps": 2, "scan_layers": False}
        kwargs["batch_size"] = 2
    elif mode == "all":
        options = {"projection_chunk_size": 3, "remat_group_size": 2, "cut_chunk_size": 5, "mlp_chunk_size": 1}
    decoder, loss = _run_step_with_memory_options(monkeypatch, options=options, **kwargs)
    assert {key: decoder[key] for key in ("projection_chunk_size", "remat_group_size") if key in decoder} == {
        key: options[key] for key in ("projection_chunk_size", "remat_group_size") if key in options
    }
    assert ({"cut_chunk_size": loss["cut_chunk_size"]} if "cut_chunk_size" in loss else {}) == {
        key: options[key] for key in ("cut_chunk_size",) if key in options
    }
    assert decoder["mlp_chunk_size"] == options.get("mlp_chunk_size", 0)
    assert decoder["scan_layers"] is options.get("scan_layers", True)
    assert decoder["remat_policy"] == options.get("remat_policy", "nothing_saveable")


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"options": {"projection_chunk_size": -1}}, "projection_chunk_size"),
        ({"options": {"remat_group_size": 0}}, "remat_group_size"),
        ({"options": {"cut_chunk_size": -1}}, "cut_chunk_size"),
        ({"options": {"projection_chunk_size": 3, "compute_dtype": jnp.float32}}, "BF16"),
        ({"options": {"projection_chunk_size": 3}, "param_dtype": jnp.float32}, "BF16"),
        ({"options": {"projection_chunk_size": 3}, "head_dtype": jnp.float32}, "BF16"),
        ({"options": {"projection_chunk_size": 3}, "batch_size": 2}, "microbatch"),
        ({"options": {"projection_chunk_size": 3}, "mesh_values": {"dp": 2}}, "dp=fsdp=ep"),
        ({"options": {"projection_chunk_size": 3}, "config_values": {"attention_dropout": 0.1}}, "dropout"),
        ({"options": {"remat_group_size": 2, "scan_layers": False}}, "scan_layers"),
        ({"options": {"remat_group_size": 2, "remat_policy": "none"}}, "rematerialization"),
        ({"options": {"remat_group_size": 3}}, "divide"),
        ({"options": {"cut_chunk_size": 3, "loss_implementation": "xla"}}, "Cut loss"),
        ({"options": {"cut_chunk_size": 3}, "mesh_values": {"fsdp": 2}}, "fsdp=ep"),
    ],
)
def test_incompatible_explicit_memory_controls_raise_without_fallback(monkeypatch, kwargs, message):
    with pytest.raises(ValueError, match=message):
        _run_step_with_memory_options(monkeypatch, **kwargs)


def test_frozen_full_step_uses_frozen_head_contract_and_updates_only_trainables(
    monkeypatch,
) -> None:
    compose_calls: list[bool] = []

    def fake_compose(frozen_params, trainable_params):
        compose_calls.append(True)
        return {
            "body": trainable_params["body"],
            "lm_head": frozen_params["lm_head"],
        }

    def fake_decoder(params, _config, input_ids, **_kwargs):
        return jnp.broadcast_to(params["body"], (*input_ids.shape, 1))

    def fake_loss(
        hidden_states,
        lm_head_kernel,
        _input_ids,
        _attention_mask,
        *,
        lm_head_trainable,
        **_kwargs,
    ):
        assert lm_head_trainable is False
        nll_sum = jnp.sum(hidden_states) * lm_head_kernel[0, 0]
        return nll_sum, jnp.asarray(hidden_states.shape[1], jnp.float32)

    monkeypatch.setattr(steps, "compose_full_trainable_params", fake_compose)
    monkeypatch.setattr(steps, "decoder_forward", fake_decoder)
    monkeypatch.setattr(steps, "chunked_causal_lm_loss", fake_loss)

    trainable_params = {"body": jnp.asarray([1.0], jnp.float32)}
    frozen_params = {"lm_head": {"kernel": jnp.asarray([[2.0]], jnp.float32)}}
    optimizer = optax.sgd(1.0)
    optimizer_state = optimizer.init(trainable_params)
    train_step = steps.make_frozen_full_train_step(
        object(),
        None,
        optimizer,
        lambda _step: jnp.asarray(0.1, jnp.float32),
        attention_implementation="vanilla",
        gradient_accumulation_steps=1,
        remat_policy="none",
        compute_dtype=jnp.float32,
        block_q=1,
        block_k=1,
        loss_token_budget=1,
        loss_implementation="xla",
        mlp_chunk_size=0,
        sparse_loss_skip=False,
        frozen_param_shardings=None,
        trainable_param_shardings=None,
        optimizer_state_shardings=None,
        batch_named_sharding=None,
        replicated_named_sharding=None,
        lm_head_trainable=False,
        external_learning_rate=True,
    )
    batch = {
        "input_ids": jnp.asarray([[1, 2]], jnp.int32),
        "attention_mask": jnp.ones((1, 2), jnp.bool_),
        "position_ids": jnp.asarray([[0, 1]], jnp.int32),
        "segment_ids": jnp.zeros((1, 2), jnp.int32),
        "loss_weights": jnp.ones((1, 2), jnp.float32),
    }

    updated, _, metrics = train_step(
        frozen_params,
        trainable_params,
        optimizer_state,
        batch,
        jnp.asarray(0, jnp.int32),
    )

    assert compose_calls
    assert jnp.allclose(updated["body"], jnp.asarray([0.8], jnp.float32))
    assert jnp.allclose(metrics["loss"], 2.0)
    assert jnp.allclose(metrics["grad_norm"], 2.0)
    assert jnp.allclose(metrics["learning_rate"], 0.1)


def test_full_step_external_learning_rate_scales_updates_and_metric(
    monkeypatch,
) -> None:
    def fake_decoder(params, _config, input_ids, **_kwargs):
        return jnp.broadcast_to(params["body"], (*input_ids.shape, 1))

    def fake_loss(hidden_states, lm_head_kernel, *_args, **_kwargs):
        return (
            jnp.sum(hidden_states) * lm_head_kernel[0, 0],
            jnp.asarray(hidden_states.shape[1], jnp.float32),
        )

    monkeypatch.setattr(steps, "decoder_forward", fake_decoder)
    monkeypatch.setattr(steps, "chunked_causal_lm_loss", fake_loss)

    params = {
        "body": jnp.asarray([1.0], jnp.float32),
        "lm_head": {"kernel": jnp.asarray([[2.0]], jnp.float32)},
    }
    optimizer = optax.sgd(1.0)
    train_step = steps.make_train_step(
        object(),
        None,
        optimizer,
        lambda position: position.astype(jnp.float32) / 10.0,
        attention_implementation="vanilla",
        gradient_accumulation_steps=1,
        remat_policy="none",
        compute_dtype=jnp.float32,
        block_q=1,
        block_k=1,
        loss_token_budget=1,
        loss_implementation="xla",
        mlp_chunk_size=0,
        sparse_loss_skip=False,
        param_shardings=None,
        optimizer_state_shardings=None,
        batch_named_sharding=None,
        replicated_named_sharding=None,
        external_learning_rate=True,
    )
    batch = {
        "input_ids": jnp.asarray([[1, 2]], jnp.int32),
        "attention_mask": jnp.ones((1, 2), jnp.bool_),
        "position_ids": jnp.asarray([[0, 1]], jnp.int32),
        "segment_ids": jnp.zeros((1, 2), jnp.int32),
        "loss_weights": jnp.ones((1, 2), jnp.float32),
    }

    updated, _, metrics = train_step(
        params,
        optimizer.init(params),
        batch,
        jnp.asarray(3, jnp.int32),
    )

    assert jnp.allclose(updated["body"], jnp.asarray([0.4], jnp.float32))
    assert jnp.allclose(metrics["learning_rate"], 0.3)


def test_lora_step_external_learning_rate_scales_updates_and_metric(
    monkeypatch,
) -> None:
    def fake_decoder(_params, _config, input_ids, *, lora_params, **_kwargs):
        return jnp.broadcast_to(lora_params["adapter"], (*input_ids.shape, 1))

    def fake_loss(hidden_states, lm_head_kernel, *_args, **_kwargs):
        return (
            jnp.sum(hidden_states) * lm_head_kernel[0, 0],
            jnp.asarray(hidden_states.shape[1], jnp.float32),
        )

    monkeypatch.setattr(steps, "decoder_forward", fake_decoder)
    monkeypatch.setattr(steps, "chunked_causal_lm_loss", fake_loss)

    base_params = {"lm_head": {"kernel": jnp.asarray([[2.0]], jnp.float32)}}
    lora_params = {"adapter": jnp.asarray([1.0], jnp.float32)}
    optimizer = optax.sgd(1.0)
    train_step = steps.make_lora_train_step(
        object(),
        None,
        optimizer,
        lambda position: position.astype(jnp.float32) / 10.0,
        attention_implementation="vanilla",
        gradient_accumulation_steps=1,
        remat_policy="none",
        compute_dtype=jnp.float32,
        block_q=1,
        block_k=1,
        loss_token_budget=1,
        loss_implementation="xla",
        mlp_chunk_size=0,
        sparse_loss_skip=False,
        base_param_shardings=None,
        lora_param_shardings=None,
        optimizer_state_shardings=None,
        batch_named_sharding=None,
        replicated_named_sharding=None,
        external_learning_rate=True,
    )
    batch = {
        "input_ids": jnp.asarray([[1, 2]], jnp.int32),
        "attention_mask": jnp.ones((1, 2), jnp.bool_),
        "position_ids": jnp.asarray([[0, 1]], jnp.int32),
        "segment_ids": jnp.zeros((1, 2), jnp.int32),
        "loss_weights": jnp.ones((1, 2), jnp.float32),
    }

    updated, _, metrics = train_step(
        base_params,
        lora_params,
        optimizer.init(lora_params),
        batch,
        jnp.asarray(3, jnp.int32),
    )

    assert jnp.allclose(updated["adapter"], jnp.asarray([0.4], jnp.float32))
    assert jnp.allclose(metrics["learning_rate"], 0.3)
