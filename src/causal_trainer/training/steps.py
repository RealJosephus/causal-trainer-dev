from __future__ import annotations

from collections.abc import Callable

import jax
import jax.numpy as jnp
import optax

from ..distributed.runtime import path_to_string
from ..modeling.architecture import decoder_forward
from ..modeling.attention_options import AttentionOptions
from ..modeling.config import ModelConfig
from ..modeling.freezing import compose_full_trainable_params
from .losses import chunked_causal_lm_loss


def _step_positions(step):
    """Separate streaming LR progress from the absolute optimizer step.

    Existing callers pass one scalar. Recovery-enabled streaming runs pass
    (schedule progress, optimizer step), preserving the original LR schedule.
    """
    return step if isinstance(step, tuple) else (step, step)


def make_train_step(
    config: ModelConfig,
    mesh,
    optimizer: optax.GradientTransformation,
    learning_rate_schedule: Callable[[jax.Array], jax.Array],
    *,
    attention_implementation: str,
    gradient_accumulation_steps: int,
    remat_policy: str,
    compute_dtype,
    block_q: int,
    block_k: int,
    loss_token_budget: int,
    loss_implementation: str,
    mlp_chunk_size: int,
    sparse_loss_skip: bool,
    param_shardings,
    optimizer_state_shardings,
    batch_named_sharding,
    replicated_named_sharding,
    gather_kv_projection_weights: bool = False,
    scan_layers: bool = False,
    external_learning_rate: bool = False,
    projection_chunk_size: int = 0,
    remat_group_size: int = 1,
    cut_chunk_size: int = 0,
    attention_options: AttentionOptions | None = None,
):
    if gradient_accumulation_steps <= 0:
        raise ValueError("gradient_accumulation_steps must be positive")
    for name, value, minimum in (
        ("projection_chunk_size", projection_chunk_size, 0),
        ("remat_group_size", remat_group_size, 1),
        ("cut_chunk_size", cut_chunk_size, 0),
    ):
        if type(value) is not int or value < minimum:
            raise ValueError(f"{name} must be a static integer >= {minimum}")
    if projection_chunk_size:
        if jnp.dtype(compute_dtype) != jnp.dtype(jnp.bfloat16):
            raise ValueError("projection_chunk_size requires BF16 compute and parameters")
    if projection_chunk_size:
        if mesh is None or any(int(mesh.shape[axis]) != 1 for axis in ("dp", "fsdp", "ep")):
            raise ValueError("projection_chunk_size requires dp=fsdp=ep=1")
        if attention_implementation != "efficient" or config.attention_dropout != 0.0:
            raise ValueError("projection_chunk_size requires efficient attention and zero attention dropout")
    if remat_group_size > 1:
        if not scan_layers or remat_policy != "nothing_saveable":
            raise ValueError("remat_group_size>1 requires scan_layers=True and nothing_saveable rematerialization")
        if config.num_hidden_layers % remat_group_size:
            raise ValueError("remat_group_size must divide num_hidden_layers")
    if cut_chunk_size:
        if loss_implementation != "cut":
            raise ValueError("cut_chunk_size requires the Cut loss backend")
        if mesh is None or any(int(mesh.shape.get(axis, 1)) != 1 for axis in ("fsdp", "ep")):
            raise ValueError("cut_chunk_size requires fsdp=ep=1")
    decoder_options = {}
    if projection_chunk_size:
        decoder_options["projection_chunk_size"] = projection_chunk_size
    if remat_group_size > 1:
        decoder_options["remat_group_size"] = remat_group_size

    def one_microbatch(params, batch, step):
        if projection_chunk_size:
            if any(value.dtype != jnp.dtype(jnp.bfloat16) for value in jax.tree.leaves(params)):
                raise ValueError("projection_chunk_size requires BF16 parameters")
        if projection_chunk_size and batch["input_ids"].shape[0] != 1:
            raise ValueError("projection_chunk_size requires global microbatch size 1")
        hidden_states = decoder_forward(
            params,
            config,
            batch["input_ids"],
            attention_mask=batch["attention_mask"],
            position_ids=batch["position_ids"],
            **({"attention_end": batch["attention_end"]} if "attention_end" in batch else {}),
            segment_ids=batch["segment_ids"],
            implementation=attention_implementation,
            mesh=mesh,
            deterministic=True,
            remat_policy=remat_policy,
            compute_dtype=compute_dtype,
            block_q=block_q,
            block_k=block_k,
            mlp_chunk_size=mlp_chunk_size,
            gather_kv_projection_weights=gather_kv_projection_weights,
            scan_layers=scan_layers,
            **decoder_options,
            **({"attention_options": attention_options, "attention_step": step} if attention_options else {}),
        )
        return chunked_causal_lm_loss(
            hidden_states,
            params["lm_head"]["kernel"],
            batch["input_ids"],
            batch["attention_mask"],
            segment_ids=batch["segment_ids"],
            loss_weights=batch["loss_weights"],
            **({"prediction_targets": batch["prediction_targets"],
                "prediction_weights": batch["prediction_weights"]} if "prediction_targets" in batch else {}),
            token_budget=loss_token_budget,
            compute_dtype=compute_dtype,
            loss_implementation=loss_implementation,
            mesh=mesh,
            sparse_skip=sparse_loss_skip,
            lm_head_trainable=True,
            **({"cut_chunk_size": cut_chunk_size} if cut_chunk_size else {}),
        )

    def train_step(params, optimizer_state, batch, step):
        step, attention_step = _step_positions(step)
        full_batch = batch["input_ids"].shape[0]
        if full_batch % gradient_accumulation_steps:
            raise ValueError("effective global batch is not divisible by gradient_accumulation_steps")
        micro_batch_size = full_batch // gradient_accumulation_steps
        if gradient_accumulation_steps == 1:
            (nll_and_tokens, gradient_sums) = jax.value_and_grad(one_microbatch, has_aux=True)(
                params, batch, attention_step,
            )
            nll_sum, token_count = nll_and_tokens
        else:
            # Accumulate in float32 only when it is requested. The common
            # accumulation=1 path avoids an additional full-model buffer.
            float_grads = jax.tree.map(lambda value: jnp.zeros(value.shape, jnp.float32), params)

            def accumulate(carry, micro_index):
                accumulated_grads, accumulated_nll, accumulated_tokens = carry
                start = micro_index * micro_batch_size
                micro = jax.tree.map(
                    lambda value: jax.lax.dynamic_slice_in_dim(value, start, micro_batch_size, axis=0),
                    batch,
                )
                (nll_and_tokens, grads) = jax.value_and_grad(one_microbatch, has_aux=True)(
                    params, micro, attention_step,
                )
                nll_sum, token_count = nll_and_tokens
                grads = jax.tree.map(lambda grad: grad.astype(jnp.float32), grads)
                accumulated_grads = jax.tree.map(jnp.add, accumulated_grads, grads)
                return (
                    accumulated_grads,
                    accumulated_nll + nll_sum,
                    accumulated_tokens + token_count,
                ), None

            (gradient_sums, nll_sum, token_count), _ = jax.lax.scan(
                accumulate,
                (float_grads, jnp.asarray(0.0, jnp.float32), jnp.asarray(0.0, jnp.float32)),
                jnp.arange(gradient_accumulation_steps, dtype=jnp.int32),
            )
        denominator = jnp.maximum(token_count, 1.0)
        grads = jax.tree.map(
            lambda gradient, parameter: (gradient / denominator).astype(parameter.dtype),
            gradient_sums,
            params,
        )
        grad_norm = optax.global_norm(grads)
        learning_rate = learning_rate_schedule(step)
        updates, optimizer_state = optimizer.update(grads, optimizer_state, params)
        if external_learning_rate:
            updates = jax.tree.map(lambda update: update * learning_rate, updates)
        updates = jax.tree.map(lambda update, parameter: update.astype(parameter.dtype), updates, params)
        updated = optax.apply_updates(params, updates)
        # Full training intentionally keeps checkpoint parameters in param_dtype.
        params = jax.tree.map(lambda new, old: new.astype(old.dtype), updated, params)
        metrics = {
            "loss": nll_sum / denominator,
            "nll_sum": nll_sum,
            "token_count": token_count,
            "grad_norm": grad_norm,
            "learning_rate": learning_rate,
        }
        return params, optimizer_state, metrics

    metric_shardings = {
        key: replicated_named_sharding
        for key in ("loss", "nll_sum", "token_count", "grad_norm", "learning_rate")
    }
    return jax.jit(
        train_step,
        in_shardings=(
            param_shardings,
            optimizer_state_shardings,
            batch_named_sharding,
            replicated_named_sharding,
        ),
        out_shardings=(param_shardings, optimizer_state_shardings, metric_shardings),
        donate_argnums=(0, 1),
    )


def make_frozen_full_train_step(
    config: ModelConfig,
    mesh,
    optimizer: optax.GradientTransformation,
    learning_rate_schedule: Callable[[jax.Array], jax.Array],
    *,
    attention_implementation: str,
    gradient_accumulation_steps: int,
    remat_policy: str,
    compute_dtype,
    block_q: int,
    block_k: int,
    loss_token_budget: int,
    loss_implementation: str,
    mlp_chunk_size: int,
    sparse_loss_skip: bool,
    frozen_param_shardings,
    trainable_param_shardings,
    optimizer_state_shardings,
    batch_named_sharding,
    replicated_named_sharding,
    lm_head_trainable: bool,
    gather_kv_projection_weights: bool = False,
    scan_layers: bool = False,
    external_learning_rate: bool = False,
    attention_options: AttentionOptions | None = None,
):
    """Build a full-rank step that excludes frozen leaves from autodiff and AdamW."""

    if gradient_accumulation_steps <= 0:
        raise ValueError("gradient_accumulation_steps must be positive")

    def one_microbatch(frozen_params, trainable_params, batch, step):
        forward_params = compose_full_trainable_params(frozen_params, trainable_params)
        hidden_states = decoder_forward(
            forward_params,
            config,
            batch["input_ids"],
            attention_mask=batch["attention_mask"],
            position_ids=batch["position_ids"],
            **({"attention_end": batch["attention_end"]} if "attention_end" in batch else {}),
            segment_ids=batch["segment_ids"],
            implementation=attention_implementation,
            mesh=mesh,
            deterministic=True,
            remat_policy=remat_policy,
            compute_dtype=compute_dtype,
            block_q=block_q,
            block_k=block_k,
            mlp_chunk_size=mlp_chunk_size,
            gather_kv_projection_weights=gather_kv_projection_weights,
            scan_layers=scan_layers,
            **({"attention_options": attention_options, "attention_step": step} if attention_options else {}),
        )
        return chunked_causal_lm_loss(
            hidden_states,
            forward_params["lm_head"]["kernel"],
            batch["input_ids"],
            batch["attention_mask"],
            segment_ids=batch["segment_ids"],
            loss_weights=batch["loss_weights"],
            **({"prediction_targets": batch["prediction_targets"],
                "prediction_weights": batch["prediction_weights"]} if "prediction_targets" in batch else {}),
            token_budget=loss_token_budget,
            compute_dtype=compute_dtype,
            loss_implementation=loss_implementation,
            mesh=mesh,
            sparse_skip=sparse_loss_skip,
            lm_head_trainable=lm_head_trainable,
        )

    def train_step(frozen_params, trainable_params, optimizer_state, batch, step):
        step, attention_step = _step_positions(step)
        full_batch = batch["input_ids"].shape[0]
        if full_batch % gradient_accumulation_steps:
            raise ValueError("effective global batch is not divisible by gradient_accumulation_steps")
        micro_batch_size = full_batch // gradient_accumulation_steps
        if gradient_accumulation_steps == 1:
            (nll_and_tokens, gradient_sums) = jax.value_and_grad(
                one_microbatch,
                argnums=1,
                has_aux=True,
            )(frozen_params, trainable_params, batch, attention_step)
            nll_sum, token_count = nll_and_tokens
        else:
            float_grads = jax.tree.map(
                lambda value: jnp.zeros(value.shape, jnp.float32),
                trainable_params,
            )

            def accumulate(carry, micro_index):
                accumulated_grads, accumulated_nll, accumulated_tokens = carry
                start = micro_index * micro_batch_size
                micro = jax.tree.map(
                    lambda value: jax.lax.dynamic_slice_in_dim(value, start, micro_batch_size, axis=0),
                    batch,
                )
                (nll_and_tokens, grads) = jax.value_and_grad(
                    one_microbatch,
                    argnums=1,
                    has_aux=True,
                )(frozen_params, trainable_params, micro, attention_step)
                micro_nll, micro_tokens = nll_and_tokens
                grads = jax.tree.map(lambda grad: grad.astype(jnp.float32), grads)
                accumulated_grads = jax.tree.map(jnp.add, accumulated_grads, grads)
                return (
                    accumulated_grads,
                    accumulated_nll + micro_nll,
                    accumulated_tokens + micro_tokens,
                ), None

            (gradient_sums, nll_sum, token_count), _ = jax.lax.scan(
                accumulate,
                (float_grads, jnp.asarray(0.0, jnp.float32), jnp.asarray(0.0, jnp.float32)),
                jnp.arange(gradient_accumulation_steps, dtype=jnp.int32),
            )
        denominator = jnp.maximum(token_count, 1.0)
        grads = jax.tree.map(
            lambda gradient, parameter: (gradient / denominator).astype(parameter.dtype),
            gradient_sums,
            trainable_params,
        )
        grad_norm = optax.global_norm(grads)
        learning_rate = learning_rate_schedule(step)
        updates, optimizer_state = optimizer.update(grads, optimizer_state, trainable_params)
        if external_learning_rate:
            updates = jax.tree.map(lambda update: update * learning_rate, updates)
        updates = jax.tree.map(
            lambda update, parameter: update.astype(parameter.dtype),
            updates,
            trainable_params,
        )
        updated = optax.apply_updates(trainable_params, updates)
        trainable_params = jax.tree.map(
            lambda new, old: new.astype(old.dtype),
            updated,
            trainable_params,
        )
        metrics = {
            "loss": nll_sum / denominator,
            "nll_sum": nll_sum,
            "token_count": token_count,
            "grad_norm": grad_norm,
            "learning_rate": learning_rate,
        }
        return trainable_params, optimizer_state, metrics

    metric_shardings = {
        key: replicated_named_sharding
        for key in ("loss", "nll_sum", "token_count", "grad_norm", "learning_rate")
    }
    return jax.jit(
        train_step,
        in_shardings=(
            frozen_param_shardings,
            trainable_param_shardings,
            optimizer_state_shardings,
            batch_named_sharding,
            replicated_named_sharding,
        ),
        out_shardings=(trainable_param_shardings, optimizer_state_shardings, metric_shardings),
        donate_argnums=(1, 2),
    )


def make_lora_train_step(
    config: ModelConfig,
    mesh,
    optimizer: optax.GradientTransformation,
    learning_rate_schedule: Callable[[jax.Array], jax.Array],
    *,
    attention_implementation: str,
    gradient_accumulation_steps: int,
    remat_policy: str,
    compute_dtype,
    block_q: int,
    block_k: int,
    loss_token_budget: int,
    loss_implementation: str,
    mlp_chunk_size: int,
    sparse_loss_skip: bool,
    base_param_shardings,
    lora_param_shardings,
    optimizer_state_shardings,
    batch_named_sharding,
    replicated_named_sharding,
    train_embed_and_lm_head: bool = False,
    gather_kv_projection_weights: bool = False,
    scan_layers: bool = False,
    external_learning_rate: bool = False,
    attention_options: AttentionOptions | None = None,
):
    """Build a step over adapters and, optionally, embedding/head leaves."""

    if gradient_accumulation_steps <= 0:
        raise ValueError("gradient_accumulation_steps must be positive")

    def one_microbatch(base_params, lora_params, batch, step):
        if train_embed_and_lm_head:
            forward_params = dict(base_params)
            forward_params["embed_tokens"] = lora_params["embed_tokens"]
            forward_params["lm_head"] = lora_params["lm_head"]
            adapters = lora_params["adapters"]
            lm_head_kernel = lora_params["lm_head"]["kernel"]
        else:
            forward_params = base_params
            adapters = lora_params
            lm_head_kernel = base_params["lm_head"]["kernel"]
        hidden_states = decoder_forward(
            forward_params,
            config,
            batch["input_ids"],
            attention_mask=batch["attention_mask"],
            position_ids=batch["position_ids"],
            **({"attention_end": batch["attention_end"]} if "attention_end" in batch else {}),
            segment_ids=batch["segment_ids"],
            implementation=attention_implementation,
            mesh=mesh,
            deterministic=True,
            lora_params=adapters,
            remat_policy=remat_policy,
            compute_dtype=compute_dtype,
            block_q=block_q,
            block_k=block_k,
            mlp_chunk_size=mlp_chunk_size,
            gather_kv_projection_weights=gather_kv_projection_weights,
            scan_layers=scan_layers,
            **({"attention_options": attention_options, "attention_step": step} if attention_options else {}),
        )
        return chunked_causal_lm_loss(
            hidden_states,
            lm_head_kernel,
            batch["input_ids"],
            batch["attention_mask"],
            segment_ids=batch["segment_ids"],
            loss_weights=batch["loss_weights"],
            **({"prediction_targets": batch["prediction_targets"],
                "prediction_weights": batch["prediction_weights"]} if "prediction_targets" in batch else {}),
            token_budget=loss_token_budget,
            compute_dtype=compute_dtype,
            loss_implementation=loss_implementation,
            mesh=mesh,
            sparse_skip=sparse_loss_skip,
            lm_head_trainable=train_embed_and_lm_head,
        )

    def train_step(base_params, lora_params, optimizer_state, batch, step):
        step, attention_step = _step_positions(step)
        full_batch = batch["input_ids"].shape[0]
        if full_batch % gradient_accumulation_steps:
            raise ValueError("effective global batch is not divisible by gradient_accumulation_steps")
        micro_batch_size = full_batch // gradient_accumulation_steps
        if gradient_accumulation_steps == 1:
            (nll_and_tokens, gradient_sums) = jax.value_and_grad(
                one_microbatch,
                argnums=1,
                has_aux=True,
            )(base_params, lora_params, batch, attention_step)
            nll_sum, token_count = nll_and_tokens
        else:
            float_grads = jax.tree.map(
                lambda value: jnp.zeros(value.shape, jnp.float32),
                lora_params,
            )

            def accumulate(carry, micro_index):
                accumulated_grads, accumulated_nll, accumulated_tokens = carry
                start = micro_index * micro_batch_size
                micro = jax.tree.map(
                    lambda value: jax.lax.dynamic_slice_in_dim(value, start, micro_batch_size, axis=0),
                    batch,
                )
                (nll_and_tokens, grads) = jax.value_and_grad(
                    one_microbatch,
                    argnums=1,
                    has_aux=True,
                )(base_params, lora_params, micro, attention_step)
                micro_nll, micro_tokens = nll_and_tokens
                grads = jax.tree.map(lambda grad: grad.astype(jnp.float32), grads)
                accumulated_grads = jax.tree.map(jnp.add, accumulated_grads, grads)
                return (
                    accumulated_grads,
                    accumulated_nll + micro_nll,
                    accumulated_tokens + micro_tokens,
                ), None

            (gradient_sums, nll_sum, token_count), _ = jax.lax.scan(
                accumulate,
                (float_grads, jnp.asarray(0.0, jnp.float32), jnp.asarray(0.0, jnp.float32)),
                jnp.arange(gradient_accumulation_steps, dtype=jnp.int32),
            )

        denominator = jnp.maximum(token_count, 1.0)
        grads = jax.tree.map(
            lambda gradient, parameter: (gradient / denominator).astype(parameter.dtype),
            gradient_sums,
            lora_params,
        )
        grad_norm = optax.global_norm(grads)
        learning_rate = learning_rate_schedule(step)
        updates, optimizer_state = optimizer.update(grads, optimizer_state, lora_params)
        if external_learning_rate:
            updates = jax.tree.map(lambda update: update * learning_rate, updates)
        updates = jax.tree.map(
            lambda update, parameter: update.astype(parameter.dtype),
            updates,
            lora_params,
        )
        updated = optax.apply_updates(lora_params, updates)
        lora_params = jax.tree.map(
            lambda new, old: new.astype(old.dtype),
            updated,
            lora_params,
        )
        metrics = {
            "loss": nll_sum / denominator,
            "nll_sum": nll_sum,
            "token_count": token_count,
            "grad_norm": grad_norm,
            "learning_rate": learning_rate,
        }
        return lora_params, optimizer_state, metrics

    metric_shardings = {
        key: replicated_named_sharding
        for key in ("loss", "nll_sum", "token_count", "grad_norm", "learning_rate")
    }
    return jax.jit(
        train_step,
        in_shardings=(
            base_param_shardings,
            lora_param_shardings,
            optimizer_state_shardings,
            batch_named_sharding,
            replicated_named_sharding,
        ),
        out_shardings=(lora_param_shardings, optimizer_state_shardings, metric_shardings),
        donate_argnums=(1, 2),
    )


def optimizer_state_template(optimizer: optax.GradientTransformation, params):
    """Trace optimizer initialization without allocating its moment buffers."""

    return jax.eval_shape(optimizer.init, params)


def infer_optimizer_state_shardings(template, params, replicated_sharding):
    """Match moment-buffer leaves to their parameter shardings by tree suffix."""

    parameter_records = []
    for key_path, parameter in jax.tree_util.tree_flatten_with_path(params)[0]:
        parameter_records.append(
            (
                path_to_string(key_path),
                tuple(parameter.shape),
                parameter.sharding,
            )
        )

    def infer(key_path, leaf):
        optimizer_path = path_to_string(key_path)
        shape = tuple(leaf.shape)
        matches = [
            sharding
            for parameter_path, parameter_shape, sharding in parameter_records
            if shape == parameter_shape
            and (optimizer_path == parameter_path or optimizer_path.endswith(f"/{parameter_path}"))
        ]
        if len(matches) == 1:
            return matches[0]
        if leaf.ndim == 0:
            return replicated_sharding
        raise ValueError(f"cannot infer optimizer-state sharding for {optimizer_path!r} with shape {shape}")

    return jax.tree_util.tree_map_with_path(infer, template)


def initialize_optimizer(optimizer: optax.GradientTransformation, params, shardings):
    return jax.jit(optimizer.init, out_shardings=shardings)(params)


__all__ = [
    "infer_optimizer_state_shardings",
    "initialize_optimizer",
    "make_frozen_full_train_step",
    "make_lora_train_step",
    "make_train_step",
    "optimizer_state_template",
]
