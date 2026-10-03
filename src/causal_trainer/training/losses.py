"""Shifted causal language-model loss for padded and packed batches."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from ..kernels.fused_cross_entropy import (
    fused_linear_cross_entropy,
    xla_fused_sparse_cross_entropy,
)
from ..modeling.rotary import canonicalize_segment_ids

Array = jax.Array


# Kept as a private compatibility alias for the small reference tests.
_fused_sparse_cross_entropy = xla_fused_sparse_cross_entropy


def _shifted_targets_and_weights(
    input_ids: Array,
    attention_mask: Array | None,
    segment_ids: Array | None = None,
    assistant_mask: Array | None = None,
    loss_mask: Array | None = None,
    labels: Array | None = None,
    *,
    ignore_index: int = -100,
) -> tuple[Array, Array]:
    """Return safe shifted targets and target-position loss weights.

    A target is excluded when either side of the prediction edge is padding or
    when the edge crosses a packed-segment boundary. Assistant/loss masks are
    interpreted at the target-token position, matching the shifted labels.
    Unlike an assistant mask, ``loss_mask`` may contain fractional weights.
    """

    tokens = jnp.asarray(input_ids, dtype=jnp.int32)
    if tokens.ndim != 2:
        raise ValueError(f"input_ids must have shape [batch, sequence], got {tokens.shape}")
    if labels is None:
        labels = tokens
    else:
        labels = jnp.asarray(labels, dtype=jnp.int32)
        if labels.shape != tokens.shape:
            raise ValueError(f"labels must have shape {tokens.shape}, got {labels.shape}")

    if attention_mask is None:
        valid_tokens = jnp.ones(tokens.shape, dtype=jnp.bool_)
    else:
        valid_tokens = jnp.asarray(attention_mask) == 1
        if valid_tokens.shape != tokens.shape:
            raise ValueError(f"attention_mask must have shape {tokens.shape}, got {valid_tokens.shape}")

    targets = labels[:, 1:]
    valid = valid_tokens[:, :-1] & valid_tokens[:, 1:] & (targets != ignore_index)

    if segment_ids is not None:
        segments = jnp.asarray(segment_ids, dtype=jnp.int32)
        if segments.shape != tokens.shape:
            raise ValueError(f"segment_ids must have shape {tokens.shape}, got {segments.shape}")
        # Keep loss boundaries identical to RoPE and attention boundaries. A
        # valid run labelled zero is not padding, and a reused label after a
        # different run must not reconnect the two samples.
        segments = canonicalize_segment_ids(valid_tokens, segments)
        valid &= (segments[:, :-1] == segments[:, 1:]) & (segments[:, 1:] != 0)

    weights = valid.astype(jnp.float32)

    if assistant_mask is not None:
        assistant = jnp.asarray(assistant_mask) != 0
        if assistant.shape != tokens.shape:
            raise ValueError(f"assistant_mask must have shape {tokens.shape}, got {assistant.shape}")
        weights *= assistant[:, 1:].astype(jnp.float32)

    if loss_mask is not None:
        extra = jnp.asarray(loss_mask, dtype=jnp.float32)
        if extra.shape != tokens.shape:
            raise ValueError(f"loss_mask must have shape {tokens.shape}, got {extra.shape}")
        weights *= extra[:, 1:]

    # Masked labels may contain -100 and must not reach take_along_axis.
    safe_targets = jnp.where(weights > 0.0, targets, 0)
    return safe_targets, weights


def shifted_loss_mask(
    input_ids: Array,
    attention_mask: Array | None,
    segment_ids: Array | None = None,
    assistant_mask: Array | None = None,
    loss_mask: Array | None = None,
    labels: Array | None = None,
    *,
    ignore_index: int = -100,
) -> tuple[Array, Array]:
    """Return safe shifted targets and their boolean compatibility mask."""

    targets, weights = _shifted_targets_and_weights(
        input_ids,
        attention_mask,
        segment_ids,
        assistant_mask,
        loss_mask,
        labels,
        ignore_index=ignore_index,
    )
    return targets, weights > 0.0


def causal_lm_loss(
    logits: Array,
    input_ids: Array,
    attention_mask: Array | None,
    segment_ids: Array | None = None,
    assistant_mask: Array | None = None,
    labels: Array | None = None,
    loss_mask: Array | None = None,
    *,
    ignore_index: int = -100,
) -> tuple[Array, Array]:
    """Return ``(weighted_nll_sum, effective_loss_weight_sum)``.

    The caller performs any cross-device ``psum`` before division, so uneven
    effective weights across data-parallel shards are normalized correctly.
    """

    logits = jnp.asarray(logits)
    tokens = jnp.asarray(input_ids)
    if logits.ndim != 3:
        raise ValueError(f"logits must have shape [batch, sequence, vocab], got {logits.shape}")
    if logits.shape[:2] != tokens.shape:
        raise ValueError(f"logits/input_ids shapes are incompatible: {logits.shape} and {tokens.shape}")

    targets, weights = _shifted_targets_and_weights(
        tokens,
        attention_mask,
        segment_ids,
        assistant_mask,
        loss_mask,
        labels,
        ignore_index=ignore_index,
    )
    shifted_logits = logits[:, :-1, :].astype(jnp.float32)
    per_token_nll = _fused_sparse_cross_entropy(shifted_logits, targets, weights)
    nll_sum = jnp.sum(per_token_nll, dtype=jnp.float32)
    token_count = jnp.sum(weights, dtype=jnp.float32)
    return nll_sum, token_count


def chunked_causal_lm_loss(
    hidden_states: Array,
    lm_head_kernel: Array,
    input_ids: Array,
    attention_mask: Array | None,
    segment_ids: Array | None = None,
    assistant_mask: Array | None = None,
    loss_weights: Array | None = None,
    *,
    token_budget: int = 128,
    compute_dtype: jnp.dtype = jnp.bfloat16,
    loss_implementation: str = "xla",
    mesh=None,
    sparse_skip: bool = False,
    lm_head_trainable: bool = True,
    cut_chunk_size: int = 0,
) -> tuple[Array, Array]:
    """Compute shifted LM loss without materializing full-sequence logits.

    ``token_budget`` bounds local projected token rows independently of the
    local data-parallel batch size. The XLA backend is a reference path, the
    chunked Pallas backend aliases its final logits buffer with ``dlogits``,
    and Cut CE never writes logits or ``dlogits`` to HBM. ``lm_head_trainable``
    is static so a frozen LoRA head does not construct a weight-gradient
    kernel. The second result is the effective loss weight sum (equal to a
    token count for ordinary 0/1 masks).
    """

    hidden_states = jnp.asarray(hidden_states)
    if hidden_states.ndim != 3:
        raise ValueError("hidden_states must have shape [batch, sequence, hidden]")
    if lm_head_kernel.ndim != 2 or hidden_states.shape[-1] != lm_head_kernel.shape[0]:
        raise ValueError("lm_head_kernel must have shape [hidden, vocab]")
    targets, weights = _shifted_targets_and_weights(
        input_ids,
        attention_mask,
        segment_ids=segment_ids,
        assistant_mask=assistant_mask,
        loss_mask=loss_weights,
    )
    loss_hidden_states = hidden_states[:, :-1, :]
    if mesh is not None and int(mesh.shape.get("sp", 1)) > 1:
        # Keep the global sequence extent SP-divisible. The shifted target at
        # each shard boundary is still formed from the following global token;
        # the final source row is appended with zero weight and contributes no
        # loss or gradient. This avoids resharding an S-1 logits tensor.
        loss_hidden_states = hidden_states
        targets = jnp.pad(targets, ((0, 0), (0, 1)), constant_values=0)
        weights = jnp.pad(weights, ((0, 0), (0, 1)), constant_values=0.0)
    return fused_linear_cross_entropy(
        loss_hidden_states,
        lm_head_kernel,
        targets,
        weights,
        token_budget=token_budget,
        compute_dtype=compute_dtype,
        implementation=loss_implementation,
        mesh=mesh,
        sparse_skip=sparse_skip,
        lm_head_trainable=lm_head_trainable,
        **({"cut_chunk_size": cut_chunk_size} if cut_chunk_size else {}),
    )


def mean_loss(nll_sum: Array, token_count: Array) -> Array:
    """Safely normalize an already globally reduced loss sum."""

    return nll_sum / jnp.maximum(token_count, jnp.asarray(1.0, dtype=jnp.float32))
