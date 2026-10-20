from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from tokenizers import Tokenizer, decoders, pre_tokenizers
from tokenizers.models import BPE

from causal_trainer.data.batching import _collate
from causal_trainer.data.maskgit import MaskGITBatchTransform
from causal_trainer.data.packing import pack_examples, pad_examples
from causal_trainer.data.tokenizer import TrainingTokenizer
from causal_trainer.modeling.architecture import decoder_forward, init_params
from causal_trainer.modeling.attention import _make_causal_segment_mask_precanonicalized, attention
from causal_trainer.modeling.config import ModelConfig
from causal_trainer.training.arguments import parse_args
from causal_trainer.training.losses import _shifted_targets_and_weights, chunked_causal_lm_loss
from causal_trainer.training.steps import make_train_step


@pytest.fixture
def tokenizer():
    alphabet = sorted(pre_tokenizers.ByteLevel.alphabet())
    backend = Tokenizer(BPE({token: i for i, token in enumerate(alphabet)}, []))
    backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False)
    backend.decoder = decoders.ByteLevel()
    backend.add_special_tokens(["[MASK]", "<|im_start|>", "<|im_end|>", "[PAD]", "<|image_16383|>", *[
        f"<|image_{i}|>" for i in range(32)
    ]])
    return TrainingTokenizer(
        backend, special_tokens={"pad_token": "[PAD]"},
        chat_template=(
            "{% for message in messages %}{{ '<|im_start|>' + message.role + '\n' }}"
            "{% if message.role == 'assistant' %}{% generation %}{{ message.content + '<|im_end|>' }}"
            "{% endgeneration %}{% else %}{{ message.content + '<|im_end|>' }}{% endif %}{% endfor %}"
        ),
    )


def grid(width=3, height=2):
    return "\n".join("".join(f"<|image_{y * width + x}|>" for x in range(width)) for y in range(height))


def record(tokenizer, response=None, prompt="Width: 999 tokens\nHeight: 999 tokens", role="assistant"):
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}, {"role": role, "content": grid() if response is None else response}],
        return_dict=True, return_assistant_tokens_mask=True,
    )


def batch_of(tokenizer, records, *, packing=True, length=256):
    fn = pack_examples if packing else pad_examples
    packed = fn(records, max_length=length, pad_token_id=tokenizer.pad_token_id)
    return _collate(packed, np.arange(len(packed)), True)


def test_probability_zero_is_identity_and_bad_options_are_rejected(tokenizer):
    batch = batch_of(tokenizer, [record(tokenizer)])
    result = MaskGITBatchTransform(tokenizer, probability=0, seed=3)(batch, step=9)
    assert result.keys() == batch.keys()
    for key in batch:
        np.testing.assert_array_equal(batch[key], result[key])
    base = ["--repo_id", "model", "--dataset_name", "data", "--dataset_text_field", "messages"]
    assert parse_args(base).maskgit_probability == 0
    assert parse_args([*base, "--maskgit_probability", "0.5"]).maskgit_probability == 0.5
    for value in ("nan", "inf", "-0.1", "1.1"):
        with pytest.raises(ValueError, match="maskgit_probability"):
            parse_args([*base, "--maskgit_probability", value])
    for flags in (["--attn_mechanism", "splash"], ["--attention_method", "moba"]):
        with pytest.raises(ValueError, match="MaskGIT"):
            parse_args([*base, "--maskgit_probability", "1", *flags])


def test_vq_code_numbers_use_tokenizer_ids_including_16383(tokenizer):
    # VQ indices are names, not model vocabulary IDs or a contiguous ID range.
    vocabulary = tokenizer.get_vocab()
    low, high = vocabulary["<|image_0|>"], vocabulary["<|image_16383|>"]
    assert low != 0 and high != 16383
    batch = batch_of(tokenizer, [record(tokenizer, "<|image_0|><|image_16383|>")])
    transform = MaskGITBatchTransform(tokenizer, probability=1, seed=3)
    assert transform.is_image[low] and transform.is_image[high]
    mixed = transform(batch, step=0)
    masked = mixed["input_ids"] == transform.mask_id
    assert masked.any()
    np.testing.assert_array_equal(mixed["prediction_targets"][masked], batch["input_ids"][masked])
    assert set(mixed["prediction_targets"][masked]) <= {low, high}


@pytest.mark.parametrize("packing", [False, True])
def test_actual_shape_newlines_and_mixed_text_are_preserved(tokenizer, packing):
    batch = batch_of(tokenizer, [record(tokenizer), record(tokenizer, "plain answer")], packing=packing)
    saved = {key: value.copy() for key, value in batch.items()}
    transform = MaskGITBatchTransform(tokenizer, probability=1, seed=1)
    mixed = transform(batch, step=0)
    blocks = [(r, *block) for r in range(len(batch["input_ids"])) for block in transform._blocks(
        batch["input_ids"][r], batch["attention_mask"][r], batch["segment_ids"][r],
    )]
    assert len(blocks) == 1
    row, start, end, images = blocks[0]
    assert len(images) == 6  # Prompt claims 999x999: it must not determine shape.
    masked = np.flatnonzero(mixed["input_ids"][row] == transform.mask_id)
    assert set(masked) <= set(images) and len(masked) > 0
    np.testing.assert_array_equal(mixed["prediction_targets"][row, masked], batch["input_ids"][row, masked])
    assert np.all(mixed["prediction_weights"][row, masked] == 1)
    newline_positions = np.setdiff1d(np.arange(start, end), images)
    assert len(newline_positions) > 0
    np.testing.assert_array_equal(
        mixed["input_ids"][row, newline_positions], batch["input_ids"][row, newline_positions],
    )
    assert np.all(mixed["prediction_weights"][row, newline_positions] == 0)
    assert mixed["prediction_weights"][row, start - 1] == 0
    assert np.all(mixed["attention_end"][row, start:end] == end - 1)
    allowed = np.asarray(_make_causal_segment_mask_precanonicalized(
        mixed["attention_mask"], mixed["segment_ids"], mixed["attention_end"],
    ))[:, 0]
    assert allowed[row, start, end - 1]
    assert not allowed[row, start - 1, start]
    assert not allowed[row, start, end]
    for r, mask in enumerate(allowed):
        segments = mixed["segment_ids"][r]
        assert not mask[segments[:, None] != segments[None, :]].any()
    for key in batch:
        np.testing.assert_array_equal(batch[key], saved[key])


def test_per_block_selection_reproducibility_resume_and_multiple_grids(tokenizer):
    batch = batch_of(tokenizer, [record(tokenizer, grid() + "\n\nnext:\n" + grid(2, 2))])
    transform = MaskGITBatchTransform(tokenizer, probability=0.5, seed=34)
    blocks = list(transform._blocks(batch["input_ids"][0], batch["attention_mask"][0], batch["segment_ids"][0]))
    assert [len(block[2]) for block in blocks] == [6, 4]
    selected = []
    for step in range(100):
        mixed = transform(batch, step=step)
        repeated = transform(batch, step=step)
        for key in mixed:
            np.testing.assert_array_equal(mixed[key], repeated[key])
        selected.append(tuple(mixed["attention_end"][0, start] == end - 1 for start, end, _ in blocks))
    assert set(selected) == {(False, False), (False, True), (True, False), (True, True)}
    assert 65 < np.sum(selected) < 135
    resumed = next(transform.batches([batch], start_step=37))
    np.testing.assert_array_equal(resumed["input_ids"], transform(batch, step=37)["input_ids"])


def test_unselected_blocks_keep_original_ar_targets_and_attention(tokenizer):
    batch = batch_of(tokenizer, [record(tokenizer), record(tokenizer, "ordinary text")])
    transform = MaskGITBatchTransform(tokenizer, probability=.001, seed=2)
    mixed = transform(batch, step=0)
    np.testing.assert_array_equal(mixed["input_ids"], batch["input_ids"])
    targets, weights = _shifted_targets_and_weights(
        batch["input_ids"], batch["attention_mask"], batch["segment_ids"], loss_mask=batch["loss_weights"],
    )
    np.testing.assert_array_equal(mixed["prediction_weights"][:, :-1], weights)
    supervised = np.asarray(weights) > 0
    np.testing.assert_array_equal(mixed["prediction_targets"][:, :-1][supervised], np.asarray(targets)[supervised])
    actual = _make_causal_segment_mask_precanonicalized(
        batch["attention_mask"], batch["segment_ids"], mixed["attention_end"],
    )
    expected = _make_causal_segment_mask_precanonicalized(batch["attention_mask"], batch["segment_ids"])
    np.testing.assert_array_equal(actual, expected)


def test_crlf_and_disabled_turn_preserve_context(tokenizer):
    first = record(tokenizer, grid().replace("\n", "\r\n"))
    disabled = record(tokenizer)
    disabled["assistant_masks"] = [0] * len(disabled["input_ids"])
    batch = batch_of(tokenizer, [first, disabled])
    transform = MaskGITBatchTransform(tokenizer, probability=1, seed=9)
    mixed = transform(batch, step=0)
    masked = mixed["input_ids"] == transform.mask_id
    assert masked.any()
    assert not masked[batch["loss_weights"] == 0].any()
    images = transform.is_image[batch["input_ids"]]
    np.testing.assert_array_equal(mixed["input_ids"][~images], batch["input_ids"][~images])


def test_disabled_input_images_and_truncation_are_not_corrupted(tokenizer):
    transform = MaskGITBatchTransform(tokenizer, probability=1, seed=0)
    for entry in (record(tokenizer, role="user"), record(tokenizer)):
        batch = batch_of(tokenizer, [entry])
        batch["loss_weights"][:] = 0
        mixed = transform(batch, step=0)
        np.testing.assert_array_equal(mixed["input_ids"], batch["input_ids"])
        assert not mixed["prediction_weights"].any()
    truncated = record(tokenizer)
    truncated = {key: value[:-2] for key, value in truncated.items()}
    batch = batch_of(tokenizer, [truncated])
    np.testing.assert_array_equal(transform(batch, step=0)["input_ids"], batch["input_ids"])


def test_non_rectangular_grid_is_rejected(tokenizer):
    batch = batch_of(tokenizer, [record(tokenizer, grid() + "<|image_9|>")])
    with pytest.raises(ValueError, match="unequal row widths"):
        MaskGITBatchTransform(tokenizer, probability=1, seed=0)(batch, step=0)


def test_hybrid_attention_bidirectional_dependencies_and_no_cross_sample_gradients():
    keys = jax.random.split(jax.random.PRNGKey(3), 3)
    q, k, v = (jax.random.normal(key, (1, 8, 1, 4)) for key in keys)
    segments = jnp.array([[1, 1, 1, 1, 1, 2, 2, 2]])
    ends = jnp.array([[0, 1, 4, 4, 4, 5, 6, 7]])

    def run(values):
        return attention(q, k, values, jnp.ones((1, 8)), segments, implementation="vanilla", attention_end=ends)

    gradient = jax.grad(lambda values: jnp.sum(run(values)[:, 2]))(v)
    assert float(jnp.linalg.norm(gradient[:, 4])) > 0  # Right neighbor in same image.
    np.testing.assert_array_equal(gradient[:, 5:], 0)
    prefix_gradient = jax.grad(lambda values: jnp.sum(run(values)[:, 1]))(v)
    np.testing.assert_array_equal(prefix_gradient[:, 2:], 0)


def test_aligned_fused_loss_matches_direct_cross_entropy_and_gradients():
    h = jax.random.normal(jax.random.PRNGKey(4), (2, 7, 4))
    head = jax.random.normal(jax.random.PRNGKey(5), (4, 13))
    targets = jnp.arange(14).reshape(2, 7) % 13
    weights = jnp.array([[0, 1, .3, 0, 1, 1, 0], [1, 0, 0, .5, 1, 0, 1]])

    def fused(hidden, kernel):
        return chunked_causal_lm_loss(
            hidden, kernel, jnp.zeros((2, 7), jnp.int32), jnp.ones((2, 7)),
            prediction_targets=targets, prediction_weights=weights, token_budget=3, compute_dtype=jnp.float32,
        )[0]

    def reference(hidden, kernel):
        logp = jax.nn.log_softmax(hidden @ kernel)
        return -(jnp.take_along_axis(logp, targets[..., None], -1)[..., 0] * weights).sum()

    actual, grads = jax.value_and_grad(fused, argnums=(0, 1))(h, head)
    expected, expected_grads = jax.value_and_grad(reference, argnums=(0, 1))(h, head)
    np.testing.assert_allclose(actual, expected, rtol=1e-6)
    for a, b in zip(grads, expected_grads, strict=True):
        np.testing.assert_allclose(a, b, atol=2e-6, rtol=2e-5)


@pytest.mark.parametrize("scan_layers", [False, True])
def test_mixed_training_update_matches_reference_with_accumulation(tokenizer, scan_layers):
    source = batch_of(tokenizer, [record(tokenizer), record(tokenizer, "plain answer")], packing=False, length=128)
    batch = jax.tree.map(jnp.asarray, MaskGITBatchTransform(tokenizer, probability=1, seed=2)(source, step=0))
    config = ModelConfig(
        hidden_size=16, intermediate_size=32, num_attention_heads=2, num_key_value_heads=1,
        head_dim=8, num_hidden_layers=2, vocab_size=len(tokenizer), max_position_embeddings=128,
        pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.get_vocab()["<|im_end|>"],
    )
    params = init_params(config, jax.random.PRNGKey(8), jnp.float32)

    def objective(parameters):
        hidden = decoder_forward(
            parameters, config, batch["input_ids"], attention_mask=batch["attention_mask"],
            position_ids=batch["position_ids"], segment_ids=batch["segment_ids"],
            attention_end=batch["attention_end"], implementation="vanilla", compute_dtype=jnp.float32,
        )
        logits = hidden @ parameters["lm_head"]["kernel"]
        nll = -jnp.take_along_axis(jax.nn.log_softmax(logits), batch["prediction_targets"][..., None], -1)[..., 0]
        return (nll * batch["prediction_weights"]).sum() / batch["prediction_weights"].sum()

    loss, gradient = jax.value_and_grad(objective)(params)
    optimizer = optax.sgd(0.01)
    step = make_train_step(
        config, None, optimizer, lambda _: jnp.asarray(.01), attention_implementation="vanilla",
        gradient_accumulation_steps=2, remat_policy="nothing_saveable", compute_dtype=jnp.float32,
        block_q=128, block_k=128, loss_token_budget=32, loss_implementation="xla", mlp_chunk_size=0,
        sparse_loss_skip=True, param_shardings=None, optimizer_state_shardings=None,
        batch_named_sharding=None, replicated_named_sharding=None, scan_layers=scan_layers,
    )
    original = jax.tree.map(lambda value: np.asarray(value).copy(), params)
    updated, _, metrics = step(params, optimizer.init(params), batch, jnp.asarray(0))
    np.testing.assert_allclose(metrics["loss"], loss, rtol=2e-5)
    leaves = (jax.tree.leaves(tree) for tree in (original, gradient, updated))
    for before, grad, after in zip(*leaves, strict=True):
        np.testing.assert_allclose(after, before - .01 * grad, rtol=2e-5, atol=2e-6)


def test_streaming_envelope_state_is_preserved(tokenizer):
    @dataclasses.dataclass(frozen=True)
    class Envelope:
        data: dict
        cursor: int

        def with_data(self, data):
            return dataclasses.replace(self, data=data)

    source = Envelope(batch_of(tokenizer, [record(tokenizer)]), 42)
    result = next(MaskGITBatchTransform(tokenizer, probability=1, seed=0).batches([source], start_step=3))
    assert result.cursor == 42
    assert "attention_end" in result.data
