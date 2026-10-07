"""Opt-in controls through real preprocessing, decoder, updates, and resume."""

from dataclasses import dataclass, replace
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from datasets import Dataset
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import WhitespaceSplit

from causal_trainer.data.packing import pack_examples
from causal_trainer.data.pipeline import packed_dataset_to_arrays
from causal_trainer.data.preprocessing import EndPromptSettings, preprocess_records
from causal_trainer.data.tokenizer import TrainingTokenizer
from causal_trainer.modeling.architecture import decoder_forward, init_params
from causal_trainer.modeling.attention_options import AttentionOptions
from causal_trainer.modeling.config import ModelConfig
from causal_trainer.modeling.freezing import split_full_trainable_params
from causal_trainer.modeling.lora import init_lora_params
from causal_trainer.training.arguments import parse_args
from causal_trainer.training.runner import _resume_signature_payload
from causal_trainer.training.steps import make_frozen_full_train_step, make_lora_train_step, make_train_step

BASE = ["--repo-id", "model", "--dataset-name", "data"]


def _config():
    return ModelConfig(
        vocab_size=16, hidden_size=8, intermediate_size=16, num_hidden_layers=2,
        num_attention_heads=2, num_key_value_heads=1, head_dim=4, pad_token_id=0, eos_token_id=2,
        max_position_embeddings=512, initializer_range=0.1,
    )


def _options(method):
    return AttentionOptions(
        method=method, query_chunk_size=2, key_chunk_size=4, moba_block_size=2, moba_top_k=2,
    )


def _batch(field):
    backend = Tokenizer(WordLevel({word: i for i, word in enumerate(
        ["pad", "unk", "eos", "user", "assistant", "question", "answer", "end", "text"]
    )}, unk_token="unk"))
    backend.pre_tokenizer = WhitespaceSplit()
    tokenizer = TrainingTokenizer(
        backend, special_tokens={"pad_token": "pad", "eos_token": "eos"},
        chat_template=(
            "{% for m in messages %}{{ m.role + ' ' }}{% if m.role == 'assistant' %}"
            "{% generation %}{{ m.content + ' eos ' }}{% endgeneration %}"
            "{% else %}{{ m.content + ' ' }}{% endif %}{% endfor %}"
        ),
    )
    message_mode = field == "messages"
    record = {"messages": [
        {"role": "user", "content": "question"},
        {"role": "assistant", "content": "answer", "loss": False},
        {"role": "user", "content": "question question"},
        {"role": "assistant", "content": "answer answer"},
    ]} if message_mode else {"text": "text question answer user assistant text question"}
    rows = preprocess_records(
        [record, record], tokenizer, dataset_text_field=field, max_sequence_length=16,
        last_assistant_only_loss=message_mode,
        endprompt=EndPromptSettings(128, 128, ("end",), prompt_loss_weight=0.25),
    )
    packed = pack_examples(rows, max_length=32, pad_token_id=0, padding_side="right")
    arrays = packed_dataset_to_arrays(
        Dataset.from_list(packed), max_sequence_length=32,
        assistant_only_loss=False, last_assistant_only_loss=message_mode,
    )
    # Two microbatches exercise accumulation while retaining packed boundaries,
    # assistant weights, masked earlier turns, padding and discontinuous RoPE.
    return {name: jnp.asarray(np.repeat(value, 2, axis=0)) for name, value in arrays.items()
            if name != "assistant_masks"}


def test_default_cli_and_resume_identity_leave_dense_path_unchanged():
    baseline = parse_args(BASE)
    assert baseline.attention_method == "dense"
    assert baseline.attn_mechanism == "efficient"
    assert baseline.attention_options is None
    @dataclass
    class Plan:
        total_steps: int = 10

    def payload(args):
        return _resume_signature_payload(
            args, _config(), Plan(), SimpleNamespace(_fingerprint="data"),
            SimpleNamespace(shape={"dp": 1, "fsdp": 1, "tp": 1, "sp": 1, "ep": 1}),
            preprocessing_process_count=1, prepared_real_rows=1, prepared_dummy_rows=0,
        )

    original = payload(baseline)
    assert "attention_options" not in original["arguments"]
    for method in ("moba",):
        configured = parse_args([*BASE, "--attention_method", method, "--attention_dense_after_step", "7"])
        current = payload(configured)
        assert current["arguments"]["attention_options"]["dense_after_step"] == 7
        assert current != original
        assert payload(replace(configured, attention_dense_after_step=8)) != current


@pytest.mark.parametrize("option,value", [
    ("moba_block_size", "0"), ("moba_top_k", "-1"), ("attention_query_chunk_size", "0"),
    ("attention_key_chunk_size", "0"), ("attention_dense_after_step", "-1"),
    ("attention_dense_suffix_layers", "-1"), ])
def test_invalid_options_fail_before_loading_model(option, value):
    with pytest.raises(ValueError):
        parse_args([*BASE, "--attention-method", "moba", "--" + option, value])


@pytest.mark.parametrize("method", ["moba"])
@pytest.mark.parametrize("field,mode", [("text", "full"), ("messages", "lora"), ("messages", "frozen")])
def test_real_preprocessing_and_optimizer_updates_include_dense_recovery(method, field, mode):
    config, batch = _config(), _batch(field)
    params = init_params(config, jax.random.key(17), jnp.float32)
    mesh = Mesh(np.asarray(jax.devices()[:1]).reshape(1, 1, 1, 1, 1), ("dp", "fsdp", "ep", "tp", "sp"))
    optimizer = optax.sgd(0.01)
    options = replace(_options(method), dense_after_step=1)
    shared = dict(
        attention_implementation="vanilla", gradient_accumulation_steps=2, remat_policy="nothing_saveable",
        compute_dtype=jnp.float32, block_q=128, block_k=128, loss_token_budget=8,
        loss_implementation="xla", mlp_chunk_size=3, sparse_loss_skip=field == "messages",
        optimizer_state_shardings=None, batch_named_sharding=None,
        replicated_named_sharding=NamedSharding(mesh, P()), scan_layers=True,
    )
    if mode == "full":
        trainable, fixed = params, None
        factory, extra = make_train_step, {"param_shardings": None}
    elif mode == "lora":
        trainable, fixed = init_lora_params(config, jax.random.key(9), 2, jnp.float32), params
        factory, extra = make_lora_train_step, {"base_param_shardings": None, "lora_param_shardings": None}
    else:
        fixed, trainable = split_full_trainable_params(params, ("embed", "lm_head", "norm"))
        factory, extra = make_frozen_full_train_step, {
            "frozen_param_shardings": None, "trainable_param_shardings": None, "lm_head_trainable": False,
        }

    def update(attention_options, optimizer_step):
        step = factory(config, mesh, optimizer, lambda _: 0.01, **shared, **extra, attention_options=attention_options)
        owned = jax.tree.map(lambda x: jnp.array(np.array(x), copy=True), trainable)
        args = (owned, optimizer.init(owned), batch, (jnp.asarray(99.5), jnp.asarray(optimizer_step)))
        return step(*args) if fixed is None else step(fixed, *args)

    approximate, _, metrics = update(options, 0)
    assert jnp.isfinite(metrics["loss"]) and metrics["token_count"] > 0
    assert metrics["grad_norm"] > 0
    assert any(not np.array_equal(a, b) for a, b in zip(
        jax.tree.leaves(approximate), jax.tree.leaves(trainable), strict=True,
    ))
    recovered, _, recovery_metrics = update(options, 1)
    dense, _, dense_metrics = update(None, 1)
    # The LR position (99.5) must not trigger recovery at optimizer step 0.
    assert not np.isclose(float(metrics["loss"]), float(dense_metrics["loss"]), atol=1e-8, rtol=0)
    np.testing.assert_allclose(recovery_metrics["loss"], dense_metrics["loss"], atol=1e-6)
    for got, expected in zip(jax.tree.leaves(recovered), jax.tree.leaves(dense), strict=True):
        np.testing.assert_allclose(got, expected, atol=1e-6)


@pytest.mark.parametrize("method", ["moba"])
def test_hybrid_layers_scanning_group_remat_and_rope_jumps(method):
    config = replace(_config(), num_hidden_layers=4)
    params = init_params(config, jax.random.key(4), jnp.float32)
    batch = _batch("text")
    options = replace(_options(method), dense_prefix_layers=1, dense_suffix_layers=1)

    def run(p, scan):
        return decoder_forward(
            p, config, batch["input_ids"], batch["attention_mask"], batch["position_ids"], batch["segment_ids"],
            implementation="vanilla", compute_dtype=jnp.float32, attention_options=options,
            remat_policy="nothing_saveable", scan_layers=scan, remat_group_size=2 if scan else 1,
        )
    ordinary = jax.jit(lambda p: run(p, False))(params)
    scanned = jax.jit(lambda p: run(p, True))(params)
    np.testing.assert_allclose(scanned, ordinary, atol=2e-5, rtol=2e-5)
    grads = jax.jit(jax.grad(lambda p: jnp.sum(run(p, True) ** 2)))(params)
    assert all(jnp.all(jnp.isfinite(x)) for x in jax.tree.leaves(grads))


@pytest.mark.parametrize("method", ["moba"])
def test_bfloat16_tiled_projections_keep_attention_method(method):
    config = _config()
    params = init_params(config, jax.random.key(6), jnp.bfloat16)
    batch = {name: value[:1] for name, value in _batch("text").items()}
    mesh = Mesh(np.asarray(jax.devices()[:1]).reshape(1, 1, 1, 1, 1), ("dp", "fsdp", "ep", "tp", "sp"))

    def run(p, tile):
        return decoder_forward(
            p, config, batch["input_ids"], batch["attention_mask"], batch["position_ids"], batch["segment_ids"],
            implementation="vanilla", mesh=mesh, compute_dtype=jnp.bfloat16, attention_options=_options(method),
            remat_policy="nothing_saveable", scan_layers=True, projection_chunk_size=tile,
            mlp_chunk_size=3, remat_group_size=2,
        )
    ordinary, tiled = (jax.jit(lambda p, tile=tile: run(p, tile))(params) for tile in (0, 3))
    np.testing.assert_allclose(np.asarray(tiled, np.float32), np.asarray(ordinary, np.float32), atol=0.04, rtol=0.04)
    grads = jax.jit(jax.grad(lambda p: run(p, 3).astype(jnp.float32).sum()))(params)
    assert all(jnp.all(jnp.isfinite(x)) for x in jax.tree.leaves(grads))
