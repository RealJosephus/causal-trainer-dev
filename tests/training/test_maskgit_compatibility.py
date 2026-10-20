"""Exercise MaskGIT through existing training and data controls, on CPU."""

from __future__ import annotations

import dataclasses
import functools
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from test_maskgit import batch_of, grid, record
from test_maskgit import tokenizer as tokenizer

from causal_trainer.data.batching import _collate
from causal_trainer.data.maskgit import MaskGITBatchTransform
from causal_trainer.data.packing import pack_examples, pad_examples
from causal_trainer.data.preprocessing import EndPromptSettings, preprocess_records
from causal_trainer.data.streaming import iter_streaming_batches
from causal_trainer.data.streaming_batching import (
    StreamingBatchPlan,
    iter_streaming_global_batches,
    streaming_global_batch_digest,
)
from causal_trainer.modeling.architecture import decoder_forward, init_params
from causal_trainer.modeling.config import ModelConfig
from causal_trainer.modeling.freezing import compose_full_trainable_params, split_full_trainable_params
from causal_trainer.modeling.lora import init_lora_params, split_lora_trainable_params
from causal_trainer.training.arguments import parse_args
from causal_trainer.training.losses import chunked_causal_lm_loss
from causal_trainer.training.runner import _distributed_arguments_digest, _resume_signature_payload
from causal_trainer.training.steps import make_frozen_full_train_step, make_lora_train_step, make_train_step


@pytest.mark.parametrize("mode", ["full", "full_chunked", "full_grouped", "frozen", "lora", "lora_head"])
def test_training_modes_match_direct_loss_and_adamw_update(tokenizer, mode):
    source = batch_of(tokenizer, [record(tokenizer), record(tokenizer, "text")], packing=False, length=128)
    source["loss_weights"][0] *= .25
    batch = jax.tree.map(jnp.asarray, MaskGITBatchTransform(tokenizer, probability=1, seed=9)(source, step=3))
    config = ModelConfig(
        hidden_size=16, intermediate_size=32, num_attention_heads=2, num_key_value_heads=1,
        head_dim=8, num_hidden_layers=2, vocab_size=len(tokenizer), max_position_embeddings=128,
        pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.get_vocab()["<|im_end|>"],
    )
    initial = init_params(config, jax.random.key(21), jnp.float32)
    frozen, trainable = {}, initial
    if mode == "frozen":
        frozen, trainable = split_full_trainable_params(initial, ("embed", "lm_head", "norm"))
    elif mode.startswith("lora"):
        frozen, trainable = split_lora_trainable_params(
            initial, init_lora_params(config, jax.random.key(22), 2, jnp.float32),
            train_embed_and_lm_head=mode == "lora_head",
        )

    def objective(parameters):
        adapters = None
        if mode == "frozen":
            combined = compose_full_trainable_params(frozen, parameters)
        elif mode.startswith("lora"):
            combined = dict(frozen)
            adapters = parameters
            if mode == "lora_head":
                combined.update({key: parameters[key] for key in ("embed_tokens", "lm_head")})
                adapters = parameters["adapters"]
        else:
            combined = parameters
        hidden = decoder_forward(
            combined, config, batch["input_ids"], attention_mask=batch["attention_mask"],
            position_ids=batch["position_ids"], segment_ids=batch["segment_ids"],
            attention_end=batch["attention_end"], implementation="vanilla", compute_dtype=jnp.float32,
            lora_params=adapters,
        )
        logp = jax.nn.log_softmax(hidden @ combined["lm_head"]["kernel"])
        nll = -jnp.take_along_axis(logp, batch["prediction_targets"][..., None], -1)[..., 0]
        return (nll * batch["prediction_weights"]).sum() / batch["prediction_weights"].sum()

    expected_loss, gradient = jax.value_and_grad(objective)(trainable)
    # Stabilize Adam near exact-zero bias gradients; compare moments as well
    # so tiny reduction-order differences cannot become sign-sized updates.
    optimizer = optax.chain(optax.clip_by_global_norm(.4), optax.adamw(1., weight_decay=.01, eps=1e-3))
    updates, expected_state = optimizer.update(gradient, optimizer.init(trainable), trainable)
    expected_params = optax.apply_updates(trainable, jax.tree.map(lambda value: value * .01, updates))
    # Retain host copies before the train step donates trainables/state.
    expected = jax.tree.map(lambda value: np.asarray(value).copy(), (expected_params, expected_state))
    frozen_before = jax.tree.map(lambda value: np.asarray(value).copy(), frozen)
    options = dict(
        attention_implementation="vanilla", gradient_accumulation_steps=2, compute_dtype=jnp.float32,
        remat_policy="nothing_saveable", block_q=128, block_k=128, loss_token_budget=13,
        loss_implementation="xla", mlp_chunk_size=3 if mode == "full_chunked" else 0,
        sparse_loss_skip=True, optimizer_state_shardings=None, batch_named_sharding=None,
        replicated_named_sharding=None, scan_layers=mode != "full_chunked", external_learning_rate=True,
    )
    def schedule(progress):
        return progress.astype(jnp.float32) * .005
    if mode.startswith("full"):
        step = make_train_step(
            config, None, optimizer, schedule, **options, param_shardings=None,
            remat_group_size=2 if mode == "full_grouped" else 1,
        )
        prefix = (trainable,)
    elif mode == "frozen":
        step = make_frozen_full_train_step(
            config, None, optimizer, schedule, **options, frozen_param_shardings=None,
            trainable_param_shardings=None, lm_head_trainable=False,
        )
        prefix = (frozen, trainable)
    else:
        step = make_lora_train_step(
            config, None, optimizer, schedule, **options, base_param_shardings=None,
            lora_param_shardings=None, train_embed_and_lm_head=mode == "lora_head",
        )
        prefix = (frozen, trainable)
    actual_params, actual_state, metrics = step(
        *prefix, optimizer.init(trainable), batch, (jnp.asarray(2), jnp.asarray(37)),
    )
    np.testing.assert_allclose(metrics["loss"], expected_loss, rtol=2e-5)
    np.testing.assert_allclose(metrics["token_count"], batch["prediction_weights"].sum())
    actual_leaves = jax.tree.leaves((actual_params, actual_state))
    for actual, reference in zip(actual_leaves, jax.tree.leaves(expected), strict=True):
        np.testing.assert_allclose(actual, reference, rtol=4e-5, atol=3e-6)
    for actual, reference in zip(jax.tree.leaves(frozen), jax.tree.leaves(frozen_before), strict=True):
        np.testing.assert_array_equal(actual, reference)


@pytest.mark.parametrize("loss_mode", ["all", "assistant", "last"])
@pytest.mark.parametrize("endprompt", [False, True])
def test_preprocessing_supervision_endprompt_and_packing(tokenizer, loss_mode, endprompt):
    messages = [
        {"role": "user", "content": grid()},
        {"role": "assistant", "content": grid(), "loss": False},
        {"role": "user", "content": "next"},
        {"role": "assistant", "content": grid()},
        {"role": "user", "content": "last"},
        {"role": "assistant", "content": grid(2, 2)},
    ]
    settings = EndPromptSettings(400, 500, ("END",), prompt_loss_weight=.2, context_loss_weight=.3)
    records = preprocess_records(
        [{"messages": messages}], tokenizer, dataset_text_field="messages", max_sequence_length=256,
        assistant_only_loss=loss_mode == "assistant", last_assistant_only_loss=loss_mode == "last",
        endprompt=settings if endprompt else None,
    )
    pack = pack_examples if endprompt else pad_examples
    source = _collate(pack(records, max_length=256, pad_token_id=tokenizer.pad_token_id), np.array([0]),
                      loss_mode != "all")
    transform = MaskGITBatchTransform(tokenizer, probability=1, seed=5)
    mixed = transform(source, step=0)
    blocks = list(transform._blocks(source["input_ids"][0], source["attention_mask"][0], source["segment_ids"][0]))
    assert len(blocks) == 3
    selected = [bool(np.any(mixed["input_ids"][0, ids] == transform.mask_id)) for _, _, ids in blocks]
    expected = {"all": [True, True, True], "assistant": [False, True, True], "last": [False, False, True]}
    assert selected == expected[loss_mode]
    user_end = np.flatnonzero(source["input_ids"][0] == transform.end_id)[0]
    np.testing.assert_array_equal(mixed["input_ids"][0, :user_end], source["input_ids"][0, :user_end])
    masked = mixed["input_ids"] == transform.mask_id
    np.testing.assert_array_equal(mixed["prediction_weights"][masked], source["loss_weights"][masked])
    np.testing.assert_array_equal(mixed["position_ids"], source["position_ids"])
    if endprompt:
        terminal = np.flatnonzero(source["attention_mask"][0])[-3:]
        assert source["position_ids"][0, terminal[0]] > 256
        np.testing.assert_allclose(mixed["prediction_weights"][0, terminal[:-1]], .2)
        np.testing.assert_array_equal(mixed["attention_end"][0, terminal], terminal)


def test_trimmed_or_truncated_message_is_explicitly_ar(tokenizer):
    tokenizer.chat_template = tokenizer.chat_template.replace(
        "message.content + '<|im_end|>'", "message.content",
    ).replace("{% endgeneration %}", "{% endgeneration %}{{ '<|im_end|>' }}")
    records = preprocess_records(
        [{"messages": [{"role": "assistant", "content": grid()}]}], tokenizer,
        dataset_text_field="messages", max_sequence_length=128, assistant_only_loss=True,
    )
    source = batch_of(tokenizer, records, length=128)
    transform = MaskGITBatchTransform(tokenizer, probability=1, seed=5)
    assert transform.end_id not in source["input_ids"]
    mixed = transform(source, step=0)
    np.testing.assert_array_equal(mixed["input_ids"], source["input_ids"])
    assert not np.any(mixed["input_ids"] == transform.mask_id)


@pytest.mark.parametrize("flags", [
    [], ["--assistant_only_loss"], ["--last_assistant_only_loss"],
    ["--packing"], ["--dataset_streaming", "--prefetch_batches", "0", "--async_dispatch_steps", "3"],
    ["--dataset_format", "prepared"], ["--endprompt_enable"],
    ["--lora", "--save_optimizer_state", "False"],
    ["--lora", "--lora_train_embed_and_lm_head", "--lora_save_adapter_only"],
    ["--frozen_parameters", "lm_head|embed|norm"],
    ["--gradient_accumulation_steps", "3", "--scan_layers", "--remat_group_size", "2"],
    ["--projection_chunk_size", "128", "--total_batch_size", "1", "--cut_chunk_size", "128"],
    ["--attn_mechanism", "vanilla", "--dtype", "float32", "--param_dtype", "float32"],
])
def test_enabling_maskgit_preserves_other_option_values(flags):
    base = ["--repo_id", "model", "--dataset_name", "data", "--dataset_text_field", "messages", *flags]
    original = parse_args(base)
    mixed = parse_args([*base, "--maskgit_probability", ".5"])
    assert dataclasses.replace(mixed, maskgit_probability=0) == original


def test_resume_signature_preserves_old_default_and_tracks_maskgit():
    args = parse_args(["--repo_id", "model", "--dataset_name", "data", "--dataset_text_field", "messages"])

    @dataclasses.dataclass
    class Plan:
        total_steps: int = 1

    def signature(arguments):
        return _resume_signature_payload(
            arguments, SimpleNamespace(to_dict=lambda: {}), Plan(), SimpleNamespace(_fingerprint="fixture"),
            SimpleNamespace(shape={"dp": 1, "fsdp": 1, "ep": 1, "tp": 1, "sp": 1}),
            preprocessing_process_count=1, prepared_real_rows=1, prepared_dummy_rows=0,
        )

    baseline = signature(args)
    assert "maskgit_probability" not in baseline["arguments"]
    mixed = dataclasses.replace(args, maskgit_probability=.5)
    payload = signature(mixed)
    assert payload["arguments"].pop("maskgit_probability") == .5
    assert payload == baseline
    assert _distributed_arguments_digest(args) != _distributed_arguments_digest(mixed)


@pytest.mark.parametrize("packing", [False, True])
def test_real_streaming_resume_digest_and_dummy_rows(tokenizer, packing):
    source = [{"messages": [{"role": "assistant", "content": grid()}]}] * 3
    plan = StreamingBatchPlan.create(3, 2, 1, 2, -1)
    transform = MaskGITBatchTransform(tokenizer, probability=.5, seed=19)

    def batches(start=0, **resume):
        windows = iter_streaming_batches(
            lambda _epoch: iter(source), tokenizer, num_epochs=2, dataset_text_field="messages",
            max_sequence_length=64, pad_token_id=tokenizer.pad_token_id, assistant_only_loss=True,
            packing=packing, packing_batch_size=2,
        )
        return transform.batches(iter_streaming_global_batches(
            windows, plan, assistant_only_loss=True, start_step=start, **resume,
        ), start_step=start)

    uninterrupted = list(batches())
    assert len(uninterrupted) >= 2
    last = uninterrupted[-1]
    replayed = list(batches(
        len(uninterrupted) - 1, expected_source_examples_before=last.source_examples_before,
        expected_stream_digest_before=last.stream_digest_before,
    ))
    assert len(replayed) == 1
    assert streaming_global_batch_digest(replayed[0]) == streaming_global_batch_digest(last)
    for batch in uninterrupted:
        dummy = ~batch.data["attention_mask"].any(axis=1)
        assert not np.any(batch.data["prediction_weights"][dummy])
        assert not np.any(batch.data["input_ids"][dummy] == transform.mask_id)
    changed = last.with_data({**last.data, "attention_end": last.data["attention_end"] + 1})
    assert streaming_global_batch_digest(changed) != streaming_global_batch_digest(last)


@pytest.mark.parametrize("backend,chunk,trainable", [
    ("xla", 0, True), ("pallas", 0, True), ("cut", 0, True), ("cut", 128, True), ("cut", 128, False),
])
def test_aligned_targets_across_ce_backends_and_frozen_cut(monkeypatch, backend, chunk, trainable):
    from jax.sharding import Mesh

    from causal_trainer.kernels import cross_entropy, cut_cross_entropy

    if jax.default_backend() != "cpu":
        pytest.skip("local CPU arithmetic gate")
    mesh = Mesh(np.asarray(jax.devices()[:1], object).reshape(1, 1, 1, 1, 1),
                ("dp", "fsdp", "ep", "tp", "sp"))
    monkeypatch.setattr(cut_cross_entropy.pl, "pallas_call",
                        functools.partial(cut_cross_entropy.pl.pallas_call, interpret=True))
    monkeypatch.setattr(jax, "default_backend", lambda: "tpu")
    hidden = (jax.random.normal(jax.random.key(81), (2, 130, 128)) * .03).astype(jnp.bfloat16)
    head = (jax.random.normal(jax.random.key(82), (128, 258)) * .03).astype(jnp.bfloat16)
    targets = jnp.arange(260).reshape(2, 130) % 258
    weights = jnp.zeros((2, 130)).at[0, 65::2].set(.3).at[0, -1].set(1.)

    def actual(h, k):
        return chunked_causal_lm_loss(
            h, k, jnp.zeros_like(targets), jnp.ones_like(targets),
            prediction_targets=targets, prediction_weights=weights,
            token_budget=128, compute_dtype=jnp.bfloat16, loss_implementation=backend,
            mesh=mesh, sparse_skip=True, lm_head_trainable=trainable, cut_chunk_size=chunk,
        )

    def reference(h, k):
        logits = jnp.einsum("bsh,hv->bsv", h, k, preferred_element_type=jnp.float32)
        nll = -jnp.take_along_axis(jax.nn.log_softmax(logits), targets[..., None], -1)[..., 0]
        return (nll * weights).sum(), weights.sum()

    argnums = (0, 1) if trainable else (0,)
    with jax.set_mesh(mesh):
        # First check production VMA without interpreter lowering. JAX 0.10's
        # CPU interpreter then needs invariant literal/ref VMA relaxed; TP=1.
        jax.eval_shape(jax.value_and_grad(actual, argnums=argnums, has_aux=True), hidden, head)
        if backend == "pallas":
            monkeypatch.setattr(
                cross_entropy, "shard_map", lambda fn, **kw: jax.shard_map(fn, **(kw | {"check_vma": False})),
            )
        result, gradient = jax.value_and_grad(actual, argnums=argnums, has_aux=True)(hidden, head)
        expected, reference_gradient = jax.value_and_grad(reference, argnums=argnums, has_aux=True)(hidden, head)
    np.testing.assert_allclose(result[0], expected[0], rtol=3e-3)
    np.testing.assert_allclose(result[1], expected[1], rtol=1e-6)
    for value, ref in zip(gradient, reference_gradient, strict=True):
        value, ref = np.asarray(value, np.float32), np.asarray(ref, np.float32)
        assert np.linalg.norm(value - ref) / max(np.linalg.norm(ref), 1e-12) < .03
    assert np.any(np.asarray(gradient[0][0, -1]))  # Same-position target at final sequence token.
    np.testing.assert_array_equal(gradient[0][1], 0)  # Entire zero-loss row.
