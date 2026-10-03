from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np

from causal_trainer.data.packing import pack_examples
from causal_trainer.training.losses import causal_lm_loss, chunked_causal_lm_loss, shifted_loss_mask


def test_cut_chunk_preserves_globally_shifted_packed_targets(monkeypatch):
    import causal_trainer.training.losses as loss_module

    class MeshShape:
        def __init__(self):
            self.shape = {"sp": 2}

    ids = jnp.asarray([[1, 2, 3, 4, 5, 6, 0, 0]], jnp.int32)
    segments = jnp.asarray([[1, 1, 2, 2, 2, 2, 0, 0]], jnp.int32)
    weights = jnp.asarray([[0, 1, .5, .25, .5, 1, 0, 0]], jnp.float32)

    def check_loss(hidden, _head, targets, shifted_weights, **kwargs):
        assert kwargs["cut_chunk_size"] == 3
        assert hidden.shape == (1, 8, 4)
        # Token 3 -> 4 crosses the SP boundary inside one document and keeps
        # supervision. Token 1 -> 2 crosses documents and is excluded.
        np.testing.assert_array_equal(targets[:, :6], [[2, 0, 4, 5, 6, 0]])
        np.testing.assert_array_equal(shifted_weights, [[1, 0, .25, .5, 1, 0, 0, 0]])
        return jnp.asarray(3.0), jnp.sum(shifted_weights)

    monkeypatch.setattr(loss_module, "fused_linear_cross_entropy", check_loss)
    result = chunked_causal_lm_loss(
        jnp.zeros((1, 8, 4)), jnp.zeros((4, 8)), ids, ids != 0,
        segment_ids=segments, loss_weights=weights, loss_implementation="cut",
        mesh=MeshShape(), cut_chunk_size=3,
    )
    np.testing.assert_allclose(result[1], 2.75)


def test_shifted_mask_excludes_padding_and_packed_boundaries() -> None:
    ids = jnp.asarray([[0, 10, 11, 20, 21]], jnp.int32)
    attention = jnp.asarray([[0, 1, 1, 1, 1]], jnp.int32)
    segments = jnp.asarray([[0, 1, 1, 2, 2]], jnp.int32)
    targets, valid = shifted_loss_mask(ids, attention, segments)
    assert targets.tolist() == [[0, 11, 0, 21]]
    assert valid.tolist() == [[False, True, False, True]]


def test_shifted_mask_canonicalizes_valid_zero_and_reused_segment_labels() -> None:
    ids = jnp.asarray([[10, 11, 20, 21, 30, 31]], jnp.int32)
    attention = jnp.ones_like(ids)
    # Zero is a valid label here, and its second contiguous run is a distinct
    # packed sample. Loss boundaries must match attention/RoPE canonicalization.
    segments = jnp.asarray([[0, 0, 7, 7, 0, 0]], jnp.int32)

    targets, valid = shifted_loss_mask(ids, attention, segments)

    assert targets.tolist() == [[11, 0, 21, 0, 31]]
    assert valid.tolist() == [[True, False, True, False, True]]


def test_assistant_mask_applies_at_target_position() -> None:
    ids = jnp.asarray([[1, 2, 3, 4]], jnp.int32)
    attention = jnp.ones_like(ids)
    assistant = jnp.asarray([[0, 0, 1, 1]], jnp.int32)
    _, valid = shifted_loss_mask(ids, attention, assistant_mask=assistant)
    assert valid.tolist() == [[False, True, True]]


def test_assistant_target_requires_a_valid_predecessor() -> None:
    ids = jnp.asarray([[10, 11, 12]], jnp.int32)
    attention = jnp.asarray([[0, 1, 1]], jnp.int32)
    assistant = jnp.asarray([[0, 1, 0]], jnp.int32)

    _, valid = shifted_loss_mask(ids, attention, assistant_mask=assistant)

    assert valid.tolist() == [[False, False]]


def test_packed_assistant_mask_never_enables_cross_segment_target() -> None:
    packed = pack_examples(
        [
            {
                "input_ids": [10, 11],
                "attention_mask": [1, 1],
                "assistant_masks": [0, 1],
            },
            {
                "input_ids": [20, 21],
                "attention_mask": [1, 1],
                # The first token is intentionally marked trainable: it must
                # still not become the target of the previous packed segment.
                "assistant_masks": [1, 1],
            },
        ],
        max_length=4,
        pad_token_id=0,
    )[0]

    targets, valid = shifted_loss_mask(
        jnp.asarray([packed["input_ids"]]),
        jnp.asarray([packed["attention_mask"]]),
        jnp.asarray([packed["segment_ids"]]),
        assistant_mask=jnp.asarray([packed["assistant_masks"]]),
    )

    assert packed["assistant_masks"] == [0, 1, 1, 1]
    assert targets.tolist() == [[11, 0, 21]]
    assert valid.tolist() == [[True, False, True]]

    # The production collation path converts assistant_masks to float32
    # loss_weights before calling the chunked loss.
    hidden = jnp.zeros((1, 4, 2), jnp.float32)
    kernel = jnp.zeros((2, 32), jnp.float32)
    nll, weight_sum = chunked_causal_lm_loss(
        hidden,
        kernel,
        jnp.asarray([packed["input_ids"]]),
        jnp.asarray([packed["attention_mask"]]),
        segment_ids=jnp.asarray([packed["segment_ids"]]),
        loss_weights=jnp.asarray([packed["assistant_masks"]], jnp.float32),
        token_budget=2,
        compute_dtype=jnp.float32,
    )
    assert jnp.allclose(nll, 2.0 * jnp.log(32.0))
    assert jnp.allclose(weight_sum, 2.0)


def test_loss_returns_sum_and_count() -> None:
    logits = jnp.zeros((1, 3, 5), jnp.float32)
    ids = jnp.asarray([[1, 2, 3]], jnp.int32)
    nll, count = causal_lm_loss(logits, ids, jnp.ones_like(ids))
    assert jnp.allclose(nll, 2 * jnp.log(5.0))
    assert count == 2


def test_fractional_loss_weights_apply_at_shifted_target_and_normalize_by_weight_sum() -> None:
    logits = jnp.zeros((1, 4, 5), jnp.float32)
    ids = jnp.asarray([[1, 2, 3, 4]], jnp.int32)
    weights = jnp.asarray([[1.0, 1.0, 0.25, 0.5]], jnp.float32)

    nll, weight_sum = causal_lm_loss(
        logits,
        ids,
        jnp.ones_like(ids),
        loss_mask=weights,
    )

    assert jnp.allclose(nll, 1.75 * jnp.log(5.0))
    assert jnp.allclose(weight_sum, 1.75)


def test_packed_weighted_loss_still_rejects_cross_segment_edge() -> None:
    logits = jnp.zeros((1, 4, 5), jnp.float32)
    ids = jnp.asarray([[1, 2, 3, 4]], jnp.int32)
    segments = jnp.asarray([[1, 1, 2, 2]], jnp.int32)
    weights = jnp.asarray([[1.0, 0.5, 1.0, 0.25]], jnp.float32)

    nll, weight_sum = causal_lm_loss(
        logits,
        ids,
        jnp.ones_like(ids),
        segment_ids=segments,
        loss_mask=weights,
    )

    assert jnp.allclose(nll, 0.75 * jnp.log(5.0))
    assert jnp.allclose(weight_sum, 0.75)


def test_endprompt_weights_survive_real_packing_and_boundary_masking() -> None:
    packed = pack_examples(
        [
            {
                "input_ids": [10, 11, 90],
                "attention_mask": [1, 1, 1],
                "position_ids": [0, 1, 15],
                "loss_weights": [1.0, 1.0, 0.1],
            },
            {
                "input_ids": [20, 91],
                "attention_mask": [1, 1],
                "position_ids": [0, 31],
                "loss_weights": [1.0, 0.25],
            },
        ],
        max_length=5,
        pad_token_id=0,
    )[0]
    logits = jnp.zeros((1, 5, 100), jnp.float32)

    nll, weight_sum = causal_lm_loss(
        logits,
        jnp.asarray([packed["input_ids"]]),
        jnp.asarray([packed["attention_mask"]]),
        segment_ids=jnp.asarray([packed["segment_ids"]]),
        loss_mask=jnp.asarray([packed["loss_weights"]]),
    )

    # 90 -> 20 crosses the packed boundary, so target weight 1.0 at token 20
    # is excluded. The remaining targets carry weights 1.0, 0.1, and 0.25.
    assert jnp.allclose(weight_sum, 1.35)
    assert jnp.allclose(nll, 1.35 * jnp.log(100.0))


def test_packed_endprompt_assistant_weights_zero_the_shifted_cross_segment_edge() -> None:
    packed = pack_examples(
        [
            {
                "input_ids": [10, 11, 90],
                "attention_mask": [1, 1, 1],
                "assistant_masks": [0, 1, 0],
                "position_ids": [0, 1, 15],
                # These are the final EndPrompt + assistant-role weights, not
                # a mask that the loss should combine a second time.
                "loss_weights": [0.0, 0.5, 0.1],
            },
            {
                "input_ids": [20, 21, 91],
                "attention_mask": [1, 1, 1],
                # Keep the second segment's first target positive so only the
                # packed-segment check can suppress the 90 -> 20 edge.
                "assistant_masks": [1, 1, 0],
                "position_ids": [0, 1, 31],
                "loss_weights": [0.75, 0.5, 0.25],
            },
        ],
        max_length=6,
        pad_token_id=0,
    )[0]

    targets, valid = shifted_loss_mask(
        jnp.asarray([packed["input_ids"]]),
        jnp.asarray([packed["attention_mask"]]),
        segment_ids=jnp.asarray([packed["segment_ids"]]),
        loss_mask=jnp.asarray([packed["loss_weights"]]),
    )

    assert packed["loss_weights"][3] == 0.75
    assert targets.tolist() == [[11, 90, 0, 21, 91]]
    assert valid.tolist() == [[True, True, False, True, True]]

    logits = jnp.zeros((1, 6, 100), jnp.float32)
    nll, weight_sum = causal_lm_loss(
        logits,
        jnp.asarray([packed["input_ids"]]),
        jnp.asarray([packed["attention_mask"]]),
        segment_ids=jnp.asarray([packed["segment_ids"]]),
        loss_mask=jnp.asarray([packed["loss_weights"]]),
    )
    assert jnp.allclose(weight_sum, 1.35)
    assert jnp.allclose(nll, 1.35 * jnp.log(100.0))


def test_chunked_hidden_loss_matches_materialized_logits() -> None:
    hidden = jnp.arange(2 * 5 * 4, dtype=jnp.float32).reshape(2, 5, 4) / 20
    kernel = jnp.arange(4 * 7, dtype=jnp.float32).reshape(4, 7) / 10
    ids = jnp.asarray([[1, 2, 3, 4, 5], [2, 3, 4, 5, 6]], jnp.int32)
    attention = jnp.ones_like(ids)
    weights = jnp.asarray(
        [[1.0, 0.1, 1.0, 0.25, 1.0], [1.0, 0.5, 0.0, 1.0, 0.75]],
        jnp.float32,
    )
    logits = jnp.einsum("bsh,hv->bsv", hidden, kernel)
    expected = causal_lm_loss(logits, ids, attention, loss_mask=weights)
    actual = chunked_causal_lm_loss(
        hidden,
        kernel,
        ids,
        attention,
        loss_weights=weights,
        token_budget=2,
        compute_dtype=jnp.float32,
    )
    assert jnp.allclose(actual[0], expected[0])
    assert actual[1] == expected[1]


def test_chunked_fused_loss_gradients_match_materialized_reference() -> None:
    hidden = jnp.arange(1 * 5 * 4, dtype=jnp.float32).reshape(1, 5, 4) / 20
    kernel = jnp.arange(4 * 7, dtype=jnp.float32).reshape(4, 7) / 10
    ids = jnp.asarray([[1, 2, 3, 4, 5]], jnp.int32)
    attention = jnp.ones_like(ids)
    segments = jnp.asarray([[1, 1, 2, 2, 2]], jnp.int32)
    weights = jnp.asarray([[1.0, 0.5, 1.0, 0.25, 0.75]], jnp.float32)

    def materialized_loss(states, head):
        logits = jnp.einsum("bsh,hv->bsv", states, head)
        return causal_lm_loss(
            logits,
            ids,
            attention,
            segment_ids=segments,
            loss_mask=weights,
        )[0]

    def fused_loss(states, head):
        return chunked_causal_lm_loss(
            states,
            head,
            ids,
            attention,
            segment_ids=segments,
            loss_weights=weights,
            token_budget=2,
            compute_dtype=jnp.float32,
        )[0]

    expected_hidden_grad, expected_kernel_grad = jax.grad(materialized_loss, argnums=(0, 1))(
        hidden,
        kernel,
    )
    actual_hidden_grad, actual_kernel_grad = jax.grad(fused_loss, argnums=(0, 1))(
        hidden,
        kernel,
    )

    assert jnp.allclose(actual_hidden_grad, expected_hidden_grad)
    assert jnp.allclose(actual_kernel_grad, expected_kernel_grad)


def test_sequence_parallel_shift_padding_preserves_loss_and_gradients() -> None:
    hidden = jnp.arange(1 * 8 * 4, dtype=jnp.float32).reshape(1, 8, 4) / 20
    kernel = jnp.arange(4 * 11, dtype=jnp.float32).reshape(4, 11) / 10
    ids = jnp.asarray([[1, 2, 3, 4, 5, 6, 7, 8]], jnp.int32)
    attention = jnp.ones_like(ids)
    # Position 3 -> 4 crosses the SP shard boundary but remains a valid LM edge.
    segments = jnp.asarray([[1, 1, 1, 1, 1, 2, 2, 2]], jnp.int32)
    weights = jnp.asarray([[1.0, 0.5, 1.0, 0.25, 1.0, 0.75, 1.0, 0.5]], jnp.float32)
    mesh = SimpleNamespace(shape={"dp": 1, "fsdp": 1, "sp": 2})

    def reference(states, head):
        logits = jnp.einsum("bsh,hv->bsv", states, head)
        return causal_lm_loss(
            logits,
            ids,
            attention,
            segment_ids=segments,
            loss_mask=weights,
        )[0]

    def sequence_parallel(states, head):
        return chunked_causal_lm_loss(
            states,
            head,
            ids,
            attention,
            segment_ids=segments,
            loss_weights=weights,
            token_budget=2,
            compute_dtype=jnp.float32,
            loss_implementation="xla",
            mesh=mesh,
        )[0]

    expected_value, expected_gradients = jax.value_and_grad(reference, argnums=(0, 1))(
        hidden,
        kernel,
    )
    actual_value, actual_gradients = jax.value_and_grad(
        sequence_parallel,
        argnums=(0, 1),
    )(hidden, kernel)

    assert jnp.allclose(actual_value, expected_value)
    assert jnp.allclose(actual_gradients[0], expected_gradients[0])
    assert jnp.allclose(actual_gradients[1], expected_gradients[1])
    assert jnp.any(actual_gradients[0][:, 3, :] != 0.0)
    assert jnp.all(actual_gradients[0][:, -1, :] == 0.0)


def test_chunked_fused_loss_handles_a_single_token_sequence() -> None:
    hidden = jnp.ones((1, 1, 4), jnp.float32)
    kernel = jnp.ones((4, 7), jnp.float32)
    ids = jnp.asarray([[1]], jnp.int32)

    nll, weight_sum = chunked_causal_lm_loss(
        hidden,
        kernel,
        ids,
        jnp.ones_like(ids),
        token_budget=2,
        compute_dtype=jnp.float32,
    )

    assert nll == 0.0
    assert weight_sum == 0.0
