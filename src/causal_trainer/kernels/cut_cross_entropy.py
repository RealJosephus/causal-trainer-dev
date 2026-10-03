"""Exact TPU cut linear cross-entropy.

The LM-head projection and sparse cross-entropy are fused at the tile level:
logits exist only as a VMEM ``[token_tile, vocab_tile]`` value.  Forward saves
one FP32 log-normalizer per active row.  Backward independently recomputes
tiles for ``dHidden`` and, when requested by the static
``lm_head_trainable`` flag, ``dHead``.  Neither logits nor dlogits are written
to HBM.

The training contract covers exact integer-label cross-entropy, arbitrary
non-negative row weights, BF16/FP32 floating inputs, and an optional
fixed-shape active-row compaction.
"""

from __future__ import annotations

import functools
import operator

import jax
import jax.numpy as jnp
from jax import shard_map
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P

from .cross_entropy import _pallas_out_shape

Array = jax.Array

# BF16 MXU tiles have a 128-wide contracting/non-contracting minor dimension.
# A smaller row tile bounds the online-softmax and LSE working set while still
# giving the matrix unit enough rows to amortize the projection.
_CUT_BLOCK_M = 128
_CUT_BLOCK_V = 128
_MXU_MINOR = 128


def _partial_output_shape(shape, dtype, hidden, kernel):
    """Partial CE results vary over both token owners and vocabulary owners.

    The abstract mesh can lose its manual-axis annotation inside a scan body;
    use input avals instead. Explicitly include kernel axes even when a TP
    all-gather proves the hidden input invariant over the vocabulary axis.
    """

    varying = set()
    found = False
    for value in (hidden, kernel):
        axis_type = getattr(value, "manual_axis_type", None)
        if axis_type is None:
            axis_type = getattr(getattr(value, "aval", None), "manual_axis_type", None)
        if axis_type is not None:
            varying.update(axis_type.varying)
            found = True
    if not found:
        return _pallas_out_shape(shape, dtype)
    return jax.ShapeDtypeStruct(
        shape, dtype, manual_axis_type=jax.sharding.ManualAxisType(varying=frozenset(varying)),
    )


def _pad_rows_2d(x: Array, padded_rows: int) -> Array:
    pad = padded_rows - x.shape[0]
    return x if pad == 0 else jnp.pad(x, ((0, pad), (0, 0)))


def _pad_rows_1d(x: Array, padded_rows: int, value: int | float = 0) -> Array:
    pad = padded_rows - x.shape[0]
    return x if pad == 0 else jnp.pad(x, ((0, pad),), constant_values=value)


def _repeat_row_metadata(x: Array) -> Array:
    """Store row scalars in the TPU's native 128-lane layout."""

    return jnp.broadcast_to(x[:, None], (x.shape[0], _MXU_MINOR))


def _pad_vocab(kernel: Array, padded_vocab: int) -> Array:
    pad = padded_vocab - kernel.shape[1]
    return kernel if pad == 0 else jnp.pad(kernel, ((0, 0), (0, pad)))


def _cut_stats_kernel(
    hidden_ref,
    kernel_ref,
    targets_ref,
    weights_ref,
    max_out_ref,
    sum_exp_out_ref,
    target_logit_out_ref,
    max_acc_ref,
    sum_exp_acc_ref,
    target_logit_acc_ref,
    *,
    vocab_size: int,
    block_v: int,
) -> None:
    """Online softmax statistics for one row block across vocab blocks."""

    vocab_block = pl.program_id(1)
    columns = vocab_block * block_v + jax.lax.broadcasted_iota(
        jnp.int32,
        (hidden_ref.shape[0], block_v),
        1,
    )
    column_valid = columns < vocab_size
    weights = weights_ref[...].astype(jnp.float32)
    row_valid = weights != 0.0

    @pl.when(vocab_block == 0)
    def _initialize() -> None:
        max_acc_ref[...] = jnp.full_like(max_acc_ref[...], -jnp.inf)
        sum_exp_acc_ref[...] = jnp.zeros_like(sum_exp_acc_ref[...])
        target_logit_acc_ref[...] = jnp.zeros_like(target_logit_acc_ref[...])

    @pl.when(jnp.any(row_valid))
    def _accumulate() -> None:
        logits = jnp.dot(
            hidden_ref[...],
            kernel_ref[...],
            preferred_element_type=jnp.float32,
        )
        masked_logits = jnp.where(column_valid, logits, -jnp.inf)
        old_max = max_acc_ref[...]
        tile_max = jnp.tile(jnp.max(masked_logits, axis=1)[:, None], (1, _MXU_MINOR))
        new_max = jnp.maximum(old_max, tile_max)
        old_scale = jnp.exp(old_max - new_max)
        expanded_new_max = jnp.tile(new_max, (1, block_v // _MXU_MINOR))
        tile_sum_exp = jnp.sum(jnp.exp(masked_logits - expanded_new_max), axis=1)
        tile_sum_exp = jnp.tile(tile_sum_exp[:, None], (1, _MXU_MINOR))
        max_acc_ref[...] = new_max
        sum_exp_acc_ref[...] = sum_exp_acc_ref[...] * old_scale + tile_sum_exp

        targets = targets_ref[...].astype(jnp.int32)
        is_target = column_valid & (columns == targets)
        tile_target_logit = jnp.sum(
            jnp.where(is_target, logits, 0.0),
            axis=1,
        )
        target_logit_acc_ref[...] = target_logit_acc_ref[...] + jnp.tile(
            tile_target_logit[:, None],
            (1, _MXU_MINOR),
        )

    # The vocab grid dimension is sequential ("arbitrary"), so each store
    # replaces the same output block and the final iteration leaves complete
    # statistics in HBM.
    max_out_ref[...] = jnp.where(row_valid, max_acc_ref[...], -jnp.inf)
    sum_exp_out_ref[...] = jnp.where(row_valid, sum_exp_acc_ref[...], 0.0)
    target_logit_out_ref[...] = jnp.where(
        row_valid,
        target_logit_acc_ref[...],
        0.0,
    )


def _cut_stats_pallas(
    hidden: Array,
    kernel: Array,
    local_targets: Array,
    weights: Array,
    *,
    block_m: int,
    block_v: int,
    output_like: Array | None = None,
) -> tuple[Array, Array, Array]:
    if block_v != _MXU_MINOR:
        raise ValueError("Cut cross-entropy vocab tile must be exactly 128")
    rows, hidden_size = hidden.shape
    vocab_size = kernel.shape[1]
    padded_rows = pl.cdiv(rows, block_m) * block_m
    padded_vocab = pl.cdiv(vocab_size, block_v) * block_v
    hidden_pad = _pad_rows_2d(hidden, padded_rows)
    targets_pad = _repeat_row_metadata(
        _pad_rows_1d(local_targets, padded_rows).astype(jnp.int32)
    )
    weights_pad = _repeat_row_metadata(
        _pad_rows_1d(weights, padded_rows).astype(jnp.float32)
    )
    kernel_pad = _pad_vocab(kernel, padded_vocab)

    outputs = pl.pallas_call(
        functools.partial(
            _cut_stats_kernel,
            vocab_size=vocab_size,
            block_v=block_v,
        ),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            in_specs=[
                pl.BlockSpec((block_m, hidden_size), lambda row, _vocab: (row, 0)),
                pl.BlockSpec((hidden_size, block_v), lambda _row, vocab: (0, vocab)),
                pl.BlockSpec((block_m, _MXU_MINOR), lambda row, _vocab: (row, 0)),
                pl.BlockSpec((block_m, _MXU_MINOR), lambda row, _vocab: (row, 0)),
            ],
            out_specs=[
                pl.BlockSpec((block_m, _MXU_MINOR), lambda row, _vocab: (row, 0)),
                pl.BlockSpec((block_m, _MXU_MINOR), lambda row, _vocab: (row, 0)),
                pl.BlockSpec((block_m, _MXU_MINOR), lambda row, _vocab: (row, 0)),
            ],
            scratch_shapes=[
                pltpu.VMEM((block_m, _MXU_MINOR), jnp.float32),
                pltpu.VMEM((block_m, _MXU_MINOR), jnp.float32),
                pltpu.VMEM((block_m, _MXU_MINOR), jnp.float32),
            ],
            grid=(padded_rows // block_m, padded_vocab // block_v),
        ),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "arbitrary"),
        ),
        out_shape=[
            (
                _pallas_out_shape((padded_rows, _MXU_MINOR), jnp.float32)
                if output_like is None
                else _partial_output_shape((padded_rows, _MXU_MINOR), jnp.float32, output_like, kernel)
            )
            for _ in range(3)
        ],
    )(hidden_pad, kernel_pad, targets_pad, weights_pad)
    return tuple(output[:rows, 0] for output in outputs)


def _cut_hidden_bwd_kernel(
    hidden_ref,
    kernel_ref,
    local_targets_ref,
    weights_ref,
    lse_ref,
    dy_ref,
    dhidden_out_ref,
    dhidden_acc_ref,
    *,
    vocab_size: int,
    block_v: int,
) -> None:
    """Accumulate one vocab tile's contribution to dHidden."""

    vocab_block = pl.program_id(1)
    columns = vocab_block * block_v + jax.lax.broadcasted_iota(
        jnp.int32,
        (hidden_ref.shape[0], block_v),
        1,
    )
    column_valid = columns < vocab_size
    weights = weights_ref[...].astype(jnp.float32)
    row_valid = weights != 0.0

    @pl.when(vocab_block == 0)
    def _initialize() -> None:
        dhidden_acc_ref[...] = jnp.zeros_like(dhidden_acc_ref[...])

    @pl.when(jnp.any(row_valid))
    def _accumulate() -> None:
        hidden = hidden_ref[...]
        kernel = kernel_ref[...]
        logits = jnp.dot(hidden, kernel, preferred_element_type=jnp.float32)
        centered = logits - lse_ref[...].astype(jnp.float32)
        probabilities = jnp.exp(
            jnp.where(row_valid & column_valid, centered, -jnp.inf)
        )
        targets = local_targets_ref[...].astype(jnp.int32)
        target = (
            row_valid
            & column_valid
            & (columns == targets)
        ).astype(jnp.float32)
        factor = weights * dy_ref[...].astype(jnp.float32)
        dlogits_tile = factor * (probabilities - target)
        contribution = jnp.dot(
            dlogits_tile.astype(kernel.dtype),
            jnp.swapaxes(kernel, 0, 1),
            preferred_element_type=jnp.float32,
        )
        dhidden_acc_ref[...] = dhidden_acc_ref[...] + contribution

    dhidden_out_ref[...] = dhidden_acc_ref[...].astype(dhidden_out_ref.dtype)


def _cut_hidden_bwd_pallas(
    hidden: Array,
    kernel: Array,
    local_targets: Array,
    weights: Array,
    lse: Array,
    dy: Array,
    *,
    block_m: int,
    block_v: int,
    output_dtype: jnp.dtype | None = None,
) -> Array:
    if block_v != _MXU_MINOR:
        raise ValueError("Cut cross-entropy vocab tile must be exactly 128")
    rows, hidden_size = hidden.shape
    vocab_size = kernel.shape[1]
    padded_rows = pl.cdiv(rows, block_m) * block_m
    padded_vocab = pl.cdiv(vocab_size, block_v) * block_v
    hidden_pad = _pad_rows_2d(hidden, padded_rows)
    kernel_pad = _pad_vocab(kernel, padded_vocab)
    targets_pad = _repeat_row_metadata(
        _pad_rows_1d(local_targets, padded_rows).astype(jnp.int32)
    )
    weights_pad = _repeat_row_metadata(
        _pad_rows_1d(weights, padded_rows).astype(jnp.float32)
    )
    lse_pad = _repeat_row_metadata(
        _pad_rows_1d(lse, padded_rows).astype(jnp.float32)
    )
    dy_pad = _repeat_row_metadata(
        _pad_rows_1d(dy, padded_rows).astype(jnp.float32)
    )

    output = pl.pallas_call(
        functools.partial(
            _cut_hidden_bwd_kernel,
            vocab_size=vocab_size,
            block_v=block_v,
        ),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            in_specs=[
                pl.BlockSpec((block_m, hidden_size), lambda row, _vocab: (row, 0)),
                pl.BlockSpec((hidden_size, block_v), lambda _row, vocab: (0, vocab)),
                pl.BlockSpec((block_m, _MXU_MINOR), lambda row, _vocab: (row, 0)),
                pl.BlockSpec((block_m, _MXU_MINOR), lambda row, _vocab: (row, 0)),
                pl.BlockSpec((block_m, _MXU_MINOR), lambda row, _vocab: (row, 0)),
                pl.BlockSpec((block_m, _MXU_MINOR), lambda row, _vocab: (row, 0)),
            ],
            out_specs=pl.BlockSpec(
                (block_m, hidden_size),
                lambda row, _vocab: (row, 0),
            ),
            scratch_shapes=[pltpu.VMEM((block_m, hidden_size), jnp.float32)],
            grid=(padded_rows // block_m, padded_vocab // block_v),
        ),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "arbitrary"),
        ),
        out_shape=(
            _pallas_out_shape(hidden_pad.shape, hidden.dtype, hidden)
            if output_dtype is None
            else _partial_output_shape(hidden_pad.shape, output_dtype, hidden, kernel)
        ),
    )(hidden_pad, kernel_pad, targets_pad, weights_pad, lse_pad, dy_pad)
    return output[:rows, :]


def _cut_head_bwd_kernel(
    hidden_ref,
    kernel_ref,
    local_targets_ref,
    weights_ref,
    lse_ref,
    dy_ref,
    dkernel_out_ref,
    dkernel_acc_ref,
    *,
    vocab_size: int,
    block_v: int,
) -> None:
    """Sequentially reduce token blocks into one conflict-free dHead tile."""

    row_block = pl.program_id(1)
    vocab_block = pl.program_id(0)
    columns = vocab_block * block_v + jax.lax.broadcasted_iota(
        jnp.int32,
        (hidden_ref.shape[0], block_v),
        1,
    )
    column_valid = columns < vocab_size
    weights = weights_ref[...].astype(jnp.float32)
    row_valid = weights != 0.0

    @pl.when(row_block == 0)
    def _initialize() -> None:
        dkernel_acc_ref[...] = jnp.zeros_like(dkernel_acc_ref[...])

    @pl.when(jnp.any(row_valid))
    def _accumulate() -> None:
        hidden = hidden_ref[...]
        kernel = kernel_ref[...]
        logits = jnp.dot(hidden, kernel, preferred_element_type=jnp.float32)
        centered = logits - lse_ref[...].astype(jnp.float32)
        probabilities = jnp.exp(
            jnp.where(row_valid & column_valid, centered, -jnp.inf)
        )
        targets = local_targets_ref[...].astype(jnp.int32)
        target = (
            row_valid
            & column_valid
            & (columns == targets)
        ).astype(jnp.float32)
        factor = weights * dy_ref[...].astype(jnp.float32)
        dlogits_tile = factor * (probabilities - target)
        contribution = jnp.dot(
            jnp.swapaxes(hidden, 0, 1),
            dlogits_tile.astype(hidden.dtype),
            preferred_element_type=jnp.float32,
        )
        dkernel_acc_ref[...] = dkernel_acc_ref[...] + contribution

    dkernel_out_ref[...] = dkernel_acc_ref[...].astype(dkernel_out_ref.dtype)


def _cut_head_bwd_pallas(
    hidden: Array,
    kernel: Array,
    local_targets: Array,
    weights: Array,
    lse: Array,
    dy: Array,
    *,
    block_m: int,
    block_v: int,
    output_dtype: jnp.dtype | None = None,
) -> Array:
    if block_v != _MXU_MINOR:
        raise ValueError("Cut cross-entropy vocab tile must be exactly 128")
    rows, hidden_size = hidden.shape
    vocab_size = kernel.shape[1]
    padded_rows = pl.cdiv(rows, block_m) * block_m
    padded_vocab = pl.cdiv(vocab_size, block_v) * block_v
    hidden_pad = _pad_rows_2d(hidden, padded_rows)
    kernel_pad = _pad_vocab(kernel, padded_vocab)
    targets_pad = _repeat_row_metadata(
        _pad_rows_1d(local_targets, padded_rows).astype(jnp.int32)
    )
    weights_pad = _repeat_row_metadata(
        _pad_rows_1d(weights, padded_rows).astype(jnp.float32)
    )
    lse_pad = _repeat_row_metadata(
        _pad_rows_1d(lse, padded_rows).astype(jnp.float32)
    )
    dy_pad = _repeat_row_metadata(
        _pad_rows_1d(dy, padded_rows).astype(jnp.float32)
    )

    output = pl.pallas_call(
        functools.partial(
            _cut_head_bwd_kernel,
            vocab_size=vocab_size,
            block_v=block_v,
        ),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            in_specs=[
                pl.BlockSpec((block_m, hidden_size), lambda _vocab, row: (row, 0)),
                pl.BlockSpec((hidden_size, block_v), lambda vocab, _row: (0, vocab)),
                pl.BlockSpec((block_m, _MXU_MINOR), lambda _vocab, row: (row, 0)),
                pl.BlockSpec((block_m, _MXU_MINOR), lambda _vocab, row: (row, 0)),
                pl.BlockSpec((block_m, _MXU_MINOR), lambda _vocab, row: (row, 0)),
                pl.BlockSpec((block_m, _MXU_MINOR), lambda _vocab, row: (row, 0)),
            ],
            out_specs=pl.BlockSpec(
                (hidden_size, block_v),
                lambda vocab, _row: (0, vocab),
            ),
            scratch_shapes=[pltpu.VMEM((hidden_size, block_v), jnp.float32)],
            grid=(padded_vocab // block_v, padded_rows // block_m),
        ),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "arbitrary"),
        ),
        out_shape=(
            _pallas_out_shape(kernel_pad.shape, kernel.dtype, kernel)
            if output_dtype is None
            else _partial_output_shape(kernel_pad.shape, output_dtype, hidden, kernel)
        ),
    )(hidden_pad, kernel_pad, targets_pad, weights_pad, lse_pad, dy_pad)
    return output[:, :vocab_size]


def _loss_and_lse(
    hidden: Array,
    kernel: Array,
    targets: Array,
    weights: Array,
    *,
    block_m: int,
    block_v: int,
    vocab_axis: str | None,
    statistics_like: Array | None = None,
) -> tuple[Array, Array, Array, Array]:
    local_vocab = kernel.shape[1]
    if vocab_axis is None:
        local_targets = targets.astype(jnp.int32)
    else:
        vocab_start = jax.lax.axis_index(vocab_axis) * local_vocab
        local_targets = targets.astype(jnp.int32) - vocab_start

    stats_options = {"block_m": block_m, "block_v": block_v}
    if statistics_like is not None:
        stats_options["output_like"] = statistics_like
    local_max, local_sum_exp, local_target_logit = _cut_stats_pallas(
        hidden, kernel, local_targets, weights, **stats_options,
    )
    if vocab_axis is None:
        global_max = local_max
        global_sum_exp = local_sum_exp
        target_logit = local_target_logit
    else:
        global_max = jax.lax.pmax(local_max, vocab_axis)
        finite = jnp.isfinite(local_max) & jnp.isfinite(global_max)
        scaled_sum_exp = jnp.where(
            finite,
            local_sum_exp * jnp.exp(local_max - global_max),
            0.0,
        )
        global_sum_exp = jax.lax.psum(scaled_sum_exp, vocab_axis)
        target_logit = jax.lax.psum(local_target_logit, vocab_axis)

    valid = weights != 0.0
    lse = jnp.where(valid, jnp.log(global_sum_exp) + global_max, 0.0)
    losses = jnp.where(valid, weights * (lse - target_logit), 0.0)
    return (
        jnp.sum(losses, dtype=jnp.float32),
        jnp.sum(weights, dtype=jnp.float32),
        lse.astype(jnp.float32),
        local_targets,
    )


@functools.partial(jax.custom_vjp, nondiff_argnums=(4, 5, 6, 7, 8, 9))
def _cut_ce(
    hidden: Array,
    kernel: Array,
    targets: Array,
    weights: Array,
    lm_head_trainable: bool,
    sparse_compaction: bool,
    block_m: int,
    block_v: int,
    vocab_axis: str | None,
    data_axes: tuple[str, ...],
) -> tuple[Array, Array]:
    del data_axes
    if sparse_compaction:
        hidden, targets, weights, _, _ = _compact_active_rows(
            hidden,
            targets,
            weights,
        )
    nll_sum, weight_sum, _, _ = _loss_and_lse(
        hidden,
        kernel,
        targets,
        weights,
        block_m=block_m,
        block_v=block_v,
        vocab_axis=vocab_axis,
    )
    return nll_sum, weight_sum


def _cut_ce_fwd(
    hidden,
    kernel,
    targets,
    weights,
    lm_head_trainable,
    sparse_compaction,
    block_m,
    block_v,
    vocab_axis,
    data_axes,
):
    del lm_head_trainable, data_axes
    work_hidden = hidden
    work_targets = targets
    work_weights = weights
    if sparse_compaction:
        work_hidden, work_targets, work_weights, _, _ = _compact_active_rows(
            hidden,
            targets,
            weights,
        )
    nll_sum, weight_sum, lse, _local_targets = _loss_and_lse(
        work_hidden,
        kernel,
        work_targets,
        work_weights,
        block_m=block_m,
        block_v=block_v,
        vocab_axis=vocab_axis,
    )
    # This is the complete activation residual: notably, there is no logits
    # tensor.  Kernel is required even for a frozen head because dHidden uses
    # it, while LSE is the sole saved softmax statistic.
    # Save the original rows rather than the gathered hidden buffer.  The
    # fixed-shape gather is cheap to reproduce in backward and otherwise would
    # become another sequence-sized activation residual.
    residual = (hidden, kernel, targets, weights, lse)
    return (nll_sum, weight_sum), residual


def _cut_ce_bwd(
    lm_head_trainable,
    sparse_compaction,
    block_m,
    block_v,
    vocab_axis,
    data_axes,
    residual,
    output_cotangents,
):
    hidden, kernel, targets, weights, lse = residual
    nll_cotangent, _weight_cotangent = output_cotangents
    work_hidden = hidden
    work_targets = targets
    work_weights = weights
    compact_indices = None
    compact_slots = None
    if sparse_compaction:
        work_hidden, work_targets, work_weights, compact_indices, compact_slots = (
            _compact_active_rows(hidden, targets, weights)
        )
    if vocab_axis is None:
        local_targets = work_targets.astype(jnp.int32)
    else:
        vocab_start = jax.lax.axis_index(vocab_axis) * kernel.shape[1]
        local_targets = work_targets.astype(jnp.int32) - vocab_start
    dy = jnp.broadcast_to(nll_cotangent.astype(jnp.float32), work_weights.shape)
    work_dhidden = _cut_hidden_bwd_pallas(
        work_hidden,
        kernel,
        local_targets,
        work_weights,
        lse,
        dy,
        block_m=block_m,
        block_v=block_v,
    )
    if vocab_axis is not None:
        # Hidden is replicated across vocabulary shards.  Make its cotangent
        # invariant before returning from the local custom VJP.
        work_dhidden = jax.lax.psum(work_dhidden, vocab_axis)
    if sparse_compaction:
        if compact_indices is None or compact_slots is None:
            raise AssertionError("compaction metadata was not initialized")
        updates = jnp.where(compact_slots[:, None], work_dhidden, 0.0)
        dhidden = jnp.zeros_like(hidden).at[compact_indices].add(updates)
    else:
        dhidden = work_dhidden
    if lm_head_trainable:
        # A separate sequential token reduction avoids atomics and write
        # conflicts.  Keeping this as a Python branch on a non-differentiable
        # static argument guarantees frozen-head LoRA traces never construct
        # the dHead kernel at all.
        dkernel = _cut_head_bwd_pallas(
            work_hidden,
            kernel,
            local_targets,
            work_weights,
            lse,
            dy,
            block_m=block_m,
            block_v=block_v,
        )
        if data_axes:
            # The vocabulary-local head slab is replicated over data shards;
            # every shard executes this collective, including shards whose
            # fixed-shape compacted rows all have zero weight.
            dkernel = jax.lax.psum(dkernel, data_axes)
    else:
        dkernel = None
    return dhidden, dkernel, None, None


_cut_ce.defvjp(_cut_ce_fwd, _cut_ce_bwd)


def _compact_active_rows(
    hidden: Array,
    targets: Array,
    weights: Array,
) -> tuple[Array, Array, Array, Array, Array]:
    """Stable fixed-shape compaction local to one data-parallel shard.

    Targets and weights have already been causally shifted by the caller.  A
    packed-sequence boundary is therefore just another zero-weight row; all
    three arrays are gathered with the same index, so compaction cannot pair a
    hidden state with a target from another packed example.
    """

    rows = weights.shape[0]
    active = weights != 0.0
    indices = jnp.nonzero(active, size=rows, fill_value=0)[0]
    active_count = jnp.sum(active, dtype=jnp.int32)
    slot_valid = jnp.arange(rows, dtype=jnp.int32) < active_count
    compact_hidden = hidden[indices]
    compact_targets = jnp.where(slot_valid, targets[indices], 0)
    compact_weights = jnp.where(slot_valid, weights[indices], 0.0)
    return compact_hidden, compact_targets, compact_weights, indices, slot_valid


def _chunked_rows(hidden, targets, weights, chunk_size):
    """Pad only feature-sharded local rows, before any TP all-gather."""

    rows, local_hidden_size = hidden.shape
    padded_rows = pl.cdiv(rows, chunk_size) * chunk_size
    return (
        _pad_rows_2d(hidden, padded_rows).reshape((-1, chunk_size, local_hidden_size)),
        _pad_rows_1d(targets, padded_rows).reshape((-1, chunk_size)),
        _pad_rows_1d(weights, padded_rows).reshape((-1, chunk_size)),
    )


def _vary_over(value: Array, axes: tuple[str, ...]) -> Array:
    return jax.lax.pcast(value, axes, to="varying") if axes else value


def _gather_hidden_chunk(hidden: Array, vocab_axis: str | None) -> Array:
    if vocab_axis is None:
        return hidden
    return jax.lax.all_gather(hidden, vocab_axis, axis=1, tiled=True)


def _chunked_cut_forward(
    hidden, kernel, targets, weights, sparse_compaction, chunk_size, vocab_axis, data_axes,
):
    chunks = _chunked_rows(hidden, targets, weights, chunk_size)

    def step(totals, inputs):
        hidden_chunk, target_chunk, weight_chunk = inputs
        work_hidden = _gather_hidden_chunk(hidden_chunk, vocab_axis)
        if sparse_compaction:
            work_hidden, target_chunk, weight_chunk, _, _ = _compact_active_rows(
                work_hidden, target_chunk, weight_chunk
            )
        nll, weight, lse, _ = _loss_and_lse(
            work_hidden, kernel, target_chunk, weight_chunk,
            block_m=_CUT_BLOCK_M, block_v=_CUT_BLOCK_V, vocab_axis=vocab_axis, statistics_like=work_hidden,
        )
        return (totals[0] + nll, totals[1] + weight), lse

    zero = _vary_over(jnp.asarray(0.0, jnp.float32), data_axes)
    return jax.lax.scan(step, (zero, zero), chunks)


@functools.partial(jax.custom_vjp, nondiff_argnums=(4, 5, 6, 7, 8))
def _chunked_cut_ce(
    hidden, kernel, targets, weights, lm_head_trainable, sparse_compaction, chunk_size, vocab_axis, data_axes,
):
    del lm_head_trainable
    totals, _ = _chunked_cut_forward(
        hidden, kernel, targets, weights, sparse_compaction, chunk_size, vocab_axis, data_axes,
    )
    return totals


def _chunked_cut_ce_fwd(
    hidden, kernel, targets, weights, lm_head_trainable, sparse_compaction, chunk_size, vocab_axis, data_axes,
):
    del lm_head_trainable
    totals, lse_chunks = _chunked_cut_forward(
        hidden, kernel, targets, weights, sparse_compaction, chunk_size, vocab_axis, data_axes,
    )
    # The whole loop owns its VJP: automatic scan transposition cannot retain
    # every gathered/compacted [chunk, full_hidden] input. Only feature-sharded
    # original hidden and one FP32 LSE per (possibly padded) row are saved.
    return totals, (hidden, kernel, targets, weights, lse_chunks)


def _chunked_cut_ce_bwd(
    lm_head_trainable, sparse_compaction, chunk_size, vocab_axis, data_axes, residual, cotangents,
):
    hidden, kernel, targets, weights, lse_chunks = residual
    nll_cotangent, _weight_cotangent = cotangents
    chunks = (*_chunked_rows(hidden, targets, weights, chunk_size), lse_chunks)
    varying_axes = (*data_axes, *((vocab_axis,) if vocab_axis is not None else ()))
    if lm_head_trainable:
        head_accumulator = _vary_over(jnp.zeros(kernel.shape, jnp.float32), varying_axes)
    else:
        head_accumulator = None

    def step(head_accumulator, inputs):
        hidden_chunk, target_chunk, weight_chunk, lse = inputs
        work_hidden = _gather_hidden_chunk(hidden_chunk, vocab_axis)
        compact_indices = compact_slots = None
        if sparse_compaction:
            work_hidden, target_chunk, weight_chunk, compact_indices, compact_slots = _compact_active_rows(
                work_hidden, target_chunk, weight_chunk
            )
        if vocab_axis is None:
            local_targets = target_chunk.astype(jnp.int32)
        else:
            vocab_start = jax.lax.axis_index(vocab_axis) * kernel.shape[1]
            local_targets = target_chunk.astype(jnp.int32) - vocab_start
        dy = jnp.broadcast_to(nll_cotangent.astype(jnp.float32), weight_chunk.shape)
        work_dhidden = _cut_hidden_bwd_pallas(
            work_hidden, kernel, local_targets, weight_chunk, lse, dy,
            block_m=_CUT_BLOCK_M, block_v=_CUT_BLOCK_V, output_dtype=jnp.float32,
        )
        if sparse_compaction:
            if compact_indices is None or compact_slots is None:
                raise AssertionError("compaction metadata was not initialized")
            updates = jnp.where(compact_slots[:, None], work_dhidden, 0.0)
            work_dhidden = jnp.zeros(work_hidden.shape, work_dhidden.dtype).at[compact_indices].add(updates)
        if vocab_axis is not None:
            # Sum vocabulary contributions and return only this TP rank's
            # feature shard; no full-sequence hidden gradient is gathered.
            work_dhidden = jax.lax.psum_scatter(
                work_dhidden, vocab_axis, scatter_dimension=1, tiled=True,
            )
        if lm_head_trainable:
            dhead = _cut_head_bwd_pallas(
                work_hidden, kernel, local_targets, weight_chunk, lse, dy,
                block_m=_CUT_BLOCK_M, block_v=_CUT_BLOCK_V, output_dtype=jnp.float32,
            )
            head_accumulator = head_accumulator + dhead
        return head_accumulator, work_dhidden.astype(hidden.dtype)

    dhead, dhidden_chunks = jax.lax.scan(step, head_accumulator, chunks)
    if lm_head_trainable:
        # Deliberately outside scan: one data/SP reduction per whole loss,
        # with FP32 accumulation across all chunks before the final BF16 cast.
        if data_axes:
            dhead = jax.lax.psum(dhead, data_axes)
        dhead = dhead.astype(kernel.dtype)
    dhidden = dhidden_chunks.reshape((-1, hidden.shape[-1]))[:hidden.shape[0]]
    return dhidden, dhead, None, None


_chunked_cut_ce.defvjp(_chunked_cut_ce_fwd, _chunked_cut_ce_bwd)


def pallas_cut_linear_cross_entropy(
    hidden_states: Array,
    lm_head_kernel: Array,
    targets: Array,
    weights: Array,
    *,
    mesh: Mesh,
    sparse_compaction: bool = False,
    lm_head_trainable: bool = True,
    local_chunk_size: int = 0,
) -> tuple[Array, Array]:
    """Return exact globally reduced ``(nll_sum, weight_sum)`` without logits.

    Data-parallel rows are flattened only *inside* each ``shard_map`` shard.
    Thus fixed-shape compaction and Pallas row grids never move examples across
    DP shards.  Vocabulary shards compute local online-softmax statistics and
    combine them with ``pmax``/``psum`` before the custom VJP saves LSE.

    A positive ``local_chunk_size`` opts into feature-sharded hidden inputs and
    gathers only one shard-local token chunk across TP at a time. Compaction is
    limited to that chunk; dHead is accumulated in FP32 and reduced over DP/SP
    once after all chunks. Zero preserves the original full-local-row path.

    The current kernel supports arbitrary ``dp``, ``sp``, and ``tp`` sizes.
    Sequence shards own disjoint token rows and reduce only the scalar loss and
    trainable-head gradient over ``sp``. ``fsdp`` and ``ep`` must remain one
    because sharding the head's contracting dimension requires a different
    collective matmul schedule.
    """

    if isinstance(local_chunk_size, bool):
        raise TypeError("local_chunk_size must be a static non-negative integer")
    try:
        local_chunk_size = operator.index(local_chunk_size)
    except TypeError as error:
        raise TypeError("local_chunk_size must be a static non-negative integer") from error
    if local_chunk_size < 0:
        raise ValueError("local_chunk_size must be non-negative")
    hidden_states = jnp.asarray(hidden_states)
    lm_head_kernel = jnp.asarray(lm_head_kernel)
    targets = jnp.asarray(targets, dtype=jnp.int32)
    weights = jnp.asarray(weights, dtype=jnp.float32)
    if hidden_states.ndim != 3:
        raise ValueError("Cut cross-entropy expects hidden_states [batch, sequence, hidden]")
    if lm_head_kernel.ndim != 2 or lm_head_kernel.shape[0] != hidden_states.shape[-1]:
        raise ValueError("lm_head_kernel must have shape [hidden, vocab]")
    if targets.shape != hidden_states.shape[:-1] or weights.shape != targets.shape:
        raise ValueError("targets and weights must match hidden_states.shape[:-1]")
    if hidden_states.shape[0] == 0 or hidden_states.shape[1] == 0:
        raise ValueError("Cut cross-entropy requires at least one token row")
    if hidden_states.shape[-1] == 0 or hidden_states.shape[-1] % _MXU_MINOR:
        raise ValueError("Cut cross-entropy hidden size must be divisible by 128 on TPU")
    if lm_head_kernel.shape[1] == 0:
        raise ValueError("Cut cross-entropy requires a non-empty vocabulary")
    if hidden_states.dtype != lm_head_kernel.dtype:
        raise TypeError("hidden_states and lm_head_kernel must use the same compute dtype")
    if not jnp.issubdtype(hidden_states.dtype, jnp.floating):
        raise TypeError("Cut cross-entropy inputs must have a floating dtype")
    if jax.default_backend() != "tpu":
        raise RuntimeError("Cut cross-entropy is available only on TPU")

    unsupported = tuple(
        axis
        for axis in ("fsdp", "ep")
        if axis in mesh.axis_names and int(mesh.shape[axis]) != 1
    )
    if unsupported:
        raise ValueError(
            "Cut cross-entropy currently requires fsdp=ep=1; active unsupported axes: "
            + ", ".join(unsupported)
        )
    dp_size = int(mesh.shape.get("dp", 1))
    sp_size = int(mesh.shape.get("sp", 1))
    tp_size = int(mesh.shape.get("tp", 1))
    if hidden_states.shape[0] % dp_size:
        raise ValueError("global batch size must be divisible by the data-parallel mesh size")
    if lm_head_kernel.shape[1] % tp_size:
        raise ValueError("vocabulary size must be divisible by the tensor-parallel mesh size")
    if hidden_states.shape[1] % sp_size:
        raise ValueError("global sequence length must be divisible by the sequence-parallel mesh size")
    if local_chunk_size and hidden_states.shape[-1] % tp_size:
        raise ValueError("chunked Cut hidden size must be divisible by the tensor-parallel mesh size")

    batch_axis = "dp" if "dp" in mesh.axis_names and dp_size > 1 else None
    sequence_axis = "sp" if "sp" in mesh.axis_names and sp_size > 1 else None
    vocab_axis = "tp" if "tp" in mesh.axis_names and tp_size > 1 else None
    hidden_spec = P(batch_axis, sequence_axis, vocab_axis if local_chunk_size else None)
    rows_spec = P(batch_axis, sequence_axis)
    kernel_spec = P(None, vocab_axis)
    data_axes = tuple(axis for axis in (batch_axis, sequence_axis) if axis is not None)

    def local(
        local_hidden: Array,
        local_kernel: Array,
        local_targets: Array,
        local_weights: Array,
    ) -> tuple[Array, Array]:
        hidden_size = local_hidden.shape[-1]
        flat_hidden = local_hidden.reshape((-1, hidden_size))
        flat_targets = local_targets.reshape(-1)
        flat_weights = local_weights.reshape(-1)
        if local_chunk_size:
            nll_sum, weight_sum = _chunked_cut_ce(
                flat_hidden, local_kernel, flat_targets, flat_weights,
                lm_head_trainable, sparse_compaction, local_chunk_size, vocab_axis, data_axes,
            )
        else:
            nll_sum, weight_sum = _cut_ce(
                flat_hidden,
                local_kernel,
                flat_targets,
                flat_weights,
                lm_head_trainable,
                sparse_compaction,
                _CUT_BLOCK_M,
                _CUT_BLOCK_V,
                vocab_axis,
                data_axes,
            )
        if data_axes:
            nll_sum = jax.lax.psum(nll_sum, data_axes)
            weight_sum = jax.lax.psum(weight_sum, data_axes)
        return nll_sum, weight_sum

    return shard_map(
        local,
        mesh=mesh,
        in_specs=(hidden_spec, kernel_spec, rows_spec, rows_spec),
        out_specs=(P(), P()),
        check_vma=True,
    )(hidden_states, lm_head_kernel, targets, weights)


__all__ = ["pallas_cut_linear_cross_entropy"]
