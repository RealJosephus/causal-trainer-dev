"""Memory-bounded linear cross-entropy for the training-only LM head.

The XLA and logits-level Pallas implementations evaluate the projection in a
per-device token-row budget and rematerialize each projection/loss body during
backward.  The Cut implementation instead fuses projection and CE at the tile
level and saves LSE, so it must not be enclosed in the logits rematerialization
policy.

The XLA primitive below is retained as an explicit correctness/reference
backend.  ``cross_entropy.py`` provides a logits-level TPU Pallas kernel, while
``cut_cross_entropy.py`` removes the HBM logits/dlogits slab altogether.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

Array = jax.Array


def _per_device_sequence_chunk_size(
    *,
    batch_size: int,
    sequence_length: int,
    token_budget: int,
    mesh,
) -> int:
    """Translate a local token-row budget to a global sequence slice.

    The returned slice keeps the global ``[batch, sequence_chunk, ...]``
    structure.  It is never a flatten across data shards: DP/FSDP divide the
    batch and SP divides the sequence only after the global slice is formed.
    """

    if token_budget <= 0:
        raise ValueError("token_budget must be positive")
    if batch_size <= 0 or sequence_length <= 0:
        raise ValueError("batch_size and sequence_length must be positive")
    if mesh is None:
        data_shards = 1
        sequence_shards = 1
    else:
        data_shards = int(mesh.shape.get("dp", 1)) * int(mesh.shape.get("fsdp", 1))
        sequence_shards = int(mesh.shape.get("sp", 1))
    if batch_size % data_shards:
        raise ValueError("global batch size must be divisible by dp*fsdp")
    if sequence_length % sequence_shards:
        raise ValueError("global sequence length must be divisible by sp")

    local_batch = batch_size // data_shards
    if token_budget < local_batch:
        raise ValueError(
            "token_budget is smaller than one sequence position for the local batch "
            f"({token_budget} < {local_batch})"
        )
    local_sequence_budget = token_budget // local_batch
    global_sequence_budget = local_sequence_budget * sequence_shards
    # Both operands are SP-divisible, so dynamic slices preserve the existing
    # global sequence sharding without a cross-shard flatten or reshard.
    return min(sequence_length, global_sequence_budget)


@jax.custom_vjp
def xla_fused_sparse_cross_entropy(
    logits: Array,
    targets: Array,
    weights: Array,
) -> Array:
    """Return weighted per-row sparse CE with an analytic logits VJP."""

    compute_logits = logits.astype(jnp.float32)
    log_normalizer = jax.nn.logsumexp(compute_logits, axis=-1)
    target_logits = jnp.take_along_axis(compute_logits, targets[..., None], axis=-1)[..., 0]
    return ((log_normalizer - target_logits) * weights).astype(jnp.float32)


def _xla_fused_sparse_cross_entropy_fwd(
    logits: Array,
    targets: Array,
    weights: Array,
) -> tuple[Array, tuple[Array, Array, Array, Array]]:
    compute_logits = logits.astype(jnp.float32)
    log_normalizer = jax.nn.logsumexp(compute_logits, axis=-1)
    target_logits = jnp.take_along_axis(compute_logits, targets[..., None], axis=-1)[..., 0]
    losses = ((log_normalizer - target_logits) * weights).astype(jnp.float32)
    return losses, (logits, log_normalizer, targets, weights)


def _xla_fused_sparse_cross_entropy_bwd(
    residual: tuple[Array, Array, Array, Array],
    output_cotangent: Array,
) -> tuple[Array, None, None]:
    logits, log_normalizer, targets, weights = residual
    probabilities = jnp.exp(logits.astype(jnp.float32) - log_normalizer[..., None])
    target_distribution = jax.nn.one_hot(
        targets,
        logits.shape[-1],
        dtype=probabilities.dtype,
    )
    factor = (weights.astype(probabilities.dtype) * output_cotangent.astype(probabilities.dtype))[..., None]
    logits_cotangent = (probabilities - target_distribution) * factor
    return logits_cotangent.astype(logits.dtype), None, None


xla_fused_sparse_cross_entropy.defvjp(
    _xla_fused_sparse_cross_entropy_fwd,
    _xla_fused_sparse_cross_entropy_bwd,
)


def fused_linear_cross_entropy(
    hidden_states: Array,
    lm_head_kernel: Array,
    targets: Array,
    weights: Array,
    *,
    token_budget: int = 128,
    compute_dtype: jnp.dtype = jnp.bfloat16,
    implementation: str = "pallas",
    mesh=None,
    sparse_skip: bool = False,
    lm_head_trainable: bool = True,
    cut_chunk_size: int = 0,
) -> tuple[Array, Array]:
    """Return ``(nll_sum, weight_sum)`` for the selected exact CE backend.

    ``targets`` and ``weights`` are already shifted and must match
    ``hidden_states.shape[:-1]``.  ``lm_head_trainable`` is a static contract:
    the Cut frozen-head path does not trace or execute a dHead kernel.  The
    per-device ``token_budget`` bounds XLA/Pallas logits slabs; Cut ignores it
    because its fixed VMEM tiles never materialize an HBM logits slab.
    ``cut_chunk_size`` explicitly bounds the SP-local hidden rows gathered
    across TP for tiled Cut execution; zero keeps ordinary Cut.
    """

    if token_budget <= 0:
        raise ValueError("token_budget must be positive")
    if implementation not in {"xla", "pallas", "cut"}:
        raise ValueError(f"unknown loss implementation: {implementation!r}")
    if isinstance(cut_chunk_size, bool) or not isinstance(cut_chunk_size, int) or cut_chunk_size < 0:
        raise ValueError("cut_chunk_size must be a non-negative static integer")
    if cut_chunk_size and implementation != "cut":
        raise ValueError("cut_chunk_size requires the Cut loss backend")

    hidden_states = jnp.asarray(hidden_states)
    lm_head_kernel = jnp.asarray(lm_head_kernel)
    targets = jnp.asarray(targets, dtype=jnp.int32)
    weights = jnp.asarray(weights, dtype=jnp.float32)
    if hidden_states.ndim != 3:
        raise ValueError("hidden_states must have shape [batch, sequence, hidden]")
    if lm_head_kernel.ndim != 2 or hidden_states.shape[-1] != lm_head_kernel.shape[0]:
        raise ValueError("lm_head_kernel must have shape [hidden, vocab]")
    if targets.shape != hidden_states.shape[:-1] or weights.shape != targets.shape:
        raise ValueError("targets and weights must match hidden_states.shape[:-1]")

    if implementation in {"pallas", "cut"}:
        if mesh is None:
            raise ValueError(f"{implementation} cross-entropy requires the training mesh")
        if jax.default_backend() != "tpu":
            raise RuntimeError(f"{implementation} cross-entropy is available only on TPU")
    if implementation == "pallas":
        from .cross_entropy import pallas_fused_sparse_cross_entropy
        pallas_cut_linear_cross_entropy = None
    elif implementation == "cut":
        from .cut_cross_entropy import pallas_cut_linear_cross_entropy
        pallas_fused_sparse_cross_entropy = None
    else:
        pallas_fused_sparse_cross_entropy = None
        pallas_cut_linear_cross_entropy = None

    sequence = hidden_states.shape[1]
    if sequence == 0:
        # Keep the degenerate result differentiable with respect to both inputs.
        zero = jnp.asarray(0.0, jnp.float32) * (
            jnp.sum(hidden_states.astype(jnp.float32))
            + jnp.sum(lm_head_kernel.astype(jnp.float32))
        )
        return zero, jnp.asarray(0.0, jnp.float32)

    kernel = lm_head_kernel.astype(compute_dtype)
    if implementation == "cut":
        # Cut is already bounded by its fixed VMEM row/vocabulary tiles and
        # never creates an HBM projection slab.  Invoke it once so the custom
        # VJP saves exactly one LSE vector and no loop-stacked head residuals;
        # token_budget controls only logits-materializing backends.
        if pallas_cut_linear_cross_entropy is None:
            raise AssertionError("Cut CE import was not initialized")
        return pallas_cut_linear_cross_entropy(
            hidden_states.astype(compute_dtype),
            kernel,
            targets,
            weights,
            mesh=mesh,
            sparse_compaction=sparse_skip,
            lm_head_trainable=lm_head_trainable,
            **({"local_chunk_size": cut_chunk_size} if cut_chunk_size else {}),
        )

    effective_chunk_size = _per_device_sequence_chunk_size(
        batch_size=hidden_states.shape[0],
        sequence_length=sequence,
        token_budget=token_budget,
        mesh=mesh,
    )

    pad = (-sequence) % effective_chunk_size
    hidden_states = jnp.pad(hidden_states, ((0, 0), (0, pad), (0, 0)))
    targets = jnp.pad(targets, ((0, 0), (0, pad)))
    weights = jnp.pad(weights, ((0, 0), (0, pad)), constant_values=0.0)
    chunks = (sequence + pad) // effective_chunk_size

    def compute_chunk_nll(
        hidden_chunk: Array,
        target_chunk: Array,
        weight_chunk: Array,
    ) -> Array:
        logits = jnp.einsum(
            "bch,hv->bcv",
            hidden_chunk.astype(compute_dtype),
            kernel,
            precision=None,
        )
        if implementation == "pallas":
            if pallas_fused_sparse_cross_entropy is None:
                raise AssertionError("Pallas CE import was not initialized")
            per_token_nll = pallas_fused_sparse_cross_entropy(
                logits,
                target_chunk,
                weight_chunk,
                mesh=mesh,
            )
        else:
            per_token_nll = xla_fused_sparse_cross_entropy(logits, target_chunk, weight_chunk)
        return jnp.sum(per_token_nll, dtype=jnp.float32)

    def chunk_nll(
        hidden_chunk: Array,
        target_chunk: Array,
        weight_chunk: Array,
    ) -> Array:
        if sparse_skip:
            # This reduction is over the global SPMD value, so every mesh shard
            # takes the same branch before the nested Pallas shard_map enters TP
            # collectives.  Fully inactive assistant/padding chunks do no LM-head
            # projection in either forward or backward.
            return jax.lax.cond(
                jnp.any(weight_chunk != 0.0),
                lambda args: compute_chunk_nll(*args),
                lambda _: jnp.asarray(0.0, jnp.float32),
                (hidden_chunk, target_chunk, weight_chunk),
            )
        return compute_chunk_nll(hidden_chunk, target_chunk, weight_chunk)

    # Rematerialize the conditional as well as its projection. Otherwise AD
    # saves the head as a branch residual at every scan iteration, turning a
    # shared [hidden, vocab] parameter into [chunks, hidden, vocab] HBM storage
    # even when the head is frozen. Saving the inputs of the whole conditional
    # keeps the head loop-invariant and also skips inactive backward work.
    evaluated_chunk_nll = jax.checkpoint(
        chunk_nll,
        policy=jax.checkpoint_policies.nothing_saveable,
        prevent_cse=False,
    )

    def accumulate(chunk_index: int, carry: tuple[Array, Array]) -> tuple[Array, Array]:
        start = chunk_index * effective_chunk_size
        hidden_chunk = jax.lax.dynamic_slice_in_dim(
            hidden_states,
            start,
            effective_chunk_size,
            axis=1,
        )
        target_chunk = jax.lax.dynamic_slice_in_dim(
            targets,
            start,
            effective_chunk_size,
            axis=1,
        )
        weight_chunk = jax.lax.dynamic_slice_in_dim(
            weights,
            start,
            effective_chunk_size,
            axis=1,
        )
        nll = evaluated_chunk_nll(
            hidden_chunk,
            target_chunk,
            weight_chunk,
        )
        # Keep the weight sum outside the rematerialized conditional: two
        # scalar results can alias in TPU buffer assignment on inactive tiles.
        chunk_weight = jnp.sum(weight_chunk, dtype=jnp.float32)
        return carry[0] + nll, carry[1] + chunk_weight

    return jax.lax.fori_loop(
        0,
        chunks,
        accumulate,
        (jnp.asarray(0.0, jnp.float32), jnp.asarray(0.0, jnp.float32)),
    )


__all__ = [
    "fused_linear_cross_entropy",
    "xla_fused_sparse_cross_entropy",
]
