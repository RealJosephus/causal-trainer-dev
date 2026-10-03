from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import time
from collections import deque
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

from tqdm.auto import tqdm

from ..checkpointing.bundles import ResumeCheckpoint
from .arguments import TrainingArguments, parse_args

_PROCESSOR_ALLOW_PATTERNS = (
    "*.json",
    "*.jinja",
    "chat_templates/*.jinja",
    "*.model",
    "*.tiktoken",
    "*.txt",
)


def _resolve_processor_source(
    repo_id_or_path: str | Path,
    *,
    revision: str | None,
    token: str | None,
) -> Path:
    """Resolve tokenizer assets without downloading another model's weights."""

    candidate = Path(repo_id_or_path).expanduser()
    if candidate.exists():
        if not candidate.is_dir():
            raise ValueError(f"processor source must be a directory, got {candidate}")
        return candidate.resolve()

    from huggingface_hub import snapshot_download

    return Path(
        snapshot_download(
            repo_id=str(repo_id_or_path),
            revision=revision,
            token=token,
            allow_patterns=list(_PROCESSOR_ALLOW_PATTERNS),
        )
    )


def _jsonable_arguments(args: TrainingArguments) -> dict[str, Any]:
    values = dataclasses.asdict(args)
    values.pop("token", None)
    return {key: str(value) if isinstance(value, Path) else value for key, value in values.items()}


def _distributed_arguments_digest(args: TrainingArguments) -> bytes:
    values = _jsonable_arguments(args)
    # These two launch values are intentionally process-local.  Every other
    # argument must match before any mode-dependent collective is entered.
    values.pop("process_id", None)
    values.pop("local_device_ids", None)
    encoded = json.dumps(values, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).digest()


def _prepared_tokenizer_digest(
    source: str | Path,
    filenames: Any,
) -> str:
    """Hash the exact tokenizer files recorded by the offline command."""

    if not isinstance(filenames, list) or not filenames:
        raise ValueError("prepared tokenizer metadata must list its files")
    root = Path(source).resolve()
    digest = hashlib.sha256()
    for filename in filenames:
        if not isinstance(filename, str) or not filename:
            raise ValueError("prepared tokenizer filenames must be non-empty strings")
        relative = PurePosixPath(filename)
        if (
            "\\" in filename
            or relative.is_absolute()
            or PureWindowsPath(filename).drive
            or str(relative) != filename
            or any(part in {"", ".", ".."} for part in relative.parts)
        ):
            raise ValueError(
                f"prepared tokenizer filename escapes its source: {filename!r}"
            )
        # Keep the containment check lexical. Hugging Face snapshot files are
        # normally symlinks whose resolved targets live in the sibling blobs
        # directory, and following those links is both expected and safe here.
        path = root.joinpath(*relative.parts)
        if not path.is_file():
            raise FileNotFoundError(
                f"prepared tokenizer file is missing from the training processor: {path}"
            )
        encoded = filename.encode("utf-8")
        digest.update(len(encoded).to_bytes(4, "little", signed=False))
        digest.update(encoded)
        with path.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
    return digest.hexdigest()


def _explicit_memory_options(args: TrainingArguments) -> dict[str, int]:
    return {
        name: getattr(args, name, default)
        for name, default in (("projection_chunk_size", 0), ("remat_group_size", 1), ("cut_chunk_size", 0))
        if getattr(args, name, default) != default
    }


def _validate_memory_options(args: TrainingArguments, config, mesh) -> None:
    options = _explicit_memory_options(args)
    if not options:
        return
    if args.lora or getattr(args, "frozen_parameters", None):
        raise ValueError("explicit memory options require full-parameter training")
    if options.get("projection_chunk_size", 0):
        if args.dtype != "bfloat16" or args.param_dtype != "bfloat16":
            raise ValueError("projection_chunk_size requires BF16 compute and parameters")
    if options.get("projection_chunk_size", 0):
        if any(int(mesh.shape[axis]) != 1 for axis in ("dp", "fsdp", "ep")):
            raise ValueError("projection_chunk_size requires dp=fsdp=ep=1")
        if args.attn_mechanism != "efficient" or args.total_batch_size != 1 or config.attention_dropout != 0.0:
            raise ValueError(
                "projection_chunk_size requires efficient attention, batch size 1 and zero attention dropout"
            )
    if options.get("remat_group_size", 1) > 1:
        if not args.scan_layers or args.gradient_checkpointing != "nothing_saveable":
            raise ValueError("remat_group_size>1 requires scan_layers=True and nothing_saveable rematerialization")
        if config.num_hidden_layers % options["remat_group_size"]:
            raise ValueError("remat_group_size must divide num_hidden_layers")
    if options.get("cut_chunk_size", 0):
        if any(int(mesh.shape[axis]) != 1 for axis in ("fsdp", "ep")):
            raise ValueError("cut_chunk_size requires fsdp=ep=1")
        if args.loss_implementation not in {"auto", "cut"}:
            raise ValueError("cut_chunk_size requires the Cut loss backend")


def _validate_supported_config(
    raw_config: dict[str, Any],
    config,
    args: TrainingArguments,
    mesh,
) -> None:
    # Concrete configuration fields define the architecture compatibility contract.
    required_fields = {
        "attention_bias",
        "attention_dropout",
        "eos_token_id",
        "head_dim",
        "hidden_act",
        "hidden_size",
        "initializer_range",
        "intermediate_size",
        "max_position_embeddings",
        "num_attention_heads",
        "num_hidden_layers",
        "num_key_value_heads",
        "pad_token_id",
        "rms_norm_eps",
        "tie_word_embeddings",
        "use_cache",
        "vocab_size",
    }
    missing_fields = sorted(required_fields - raw_config.keys())
    if missing_fields:
        raise ValueError(
            "checkpoint config is missing required architecture fields: "
            + ", ".join(missing_fields)
        )
    if "torch_dtype" not in raw_config and "dtype" not in raw_config:
        raise ValueError("checkpoint config is missing required architecture field: dtype/torch_dtype")
    has_canonical_rope = isinstance(raw_config.get("rope_parameters"), dict)
    has_source_rope = all(
        key in raw_config
        for key in ("partial_rotary_factor", "rope_scaling", "rope_theta")
    )
    if not has_canonical_rope and not has_source_rope:
        raise ValueError(
            "checkpoint config must define RoPE through rope_parameters or "
            "partial_rotary_factor/rope_scaling/rope_theta"
        )
    requirements = {
        "attention_bias=True": config.attention_bias is True,
        "attention_dropout=0": config.attention_dropout == 0.0,
        "eos_token_id=151650": config.eos_token_id == 151_650,
        "head_dim=128": config.head_dim == 128,
        "hidden_act=silu": config.hidden_act == "silu",
        "hidden_size=4096": config.hidden_size == 4096,
        "initializer_range=0.02": config.initializer_range == 0.02,
        "intermediate_size=13568": config.intermediate_size == 13_568,
        "num_attention_heads=32": config.num_attention_heads == 32,
        "num_hidden_layers=40": config.num_hidden_layers == 40,
        "num_key_value_heads=1": config.num_key_value_heads == 1,
        "pad_token_id=151329": config.pad_token_id == 151_329,
        "partial_rotary_factor=0.5": config.partial_rotary_factor == 0.5,
        "rms_norm_eps=1e-6": config.rms_norm_eps == 1e-6,
        "rope_scaling=None": config.rope_scaling is None,
        "rope_theta=1e9": config.rope_theta == 1_000_000_000.0,
        "sliding_window=None": config.sliding_window is None,
        "tie_word_embeddings=False": config.tie_word_embeddings is False,
        "torch_dtype=bfloat16": config.torch_dtype == "bfloat16",
        "use_cache=True": config.use_cache is True,
        "use_sliding_window=False": config.use_sliding_window is False,
        "vocab_size=185600": config.vocab_size == 185_600,
    }
    missing = [name for name, valid in requirements.items() if not valid]
    if missing:
        raise ValueError("checkpoint does not use the supported architecture: " + ", ".join(missing))
    if args.max_sequence_length > config.max_position_embeddings:
        raise ValueError("max_sequence_length exceeds the checkpoint's max_position_embeddings")
    if (
        args.dataset_format == "raw"
        and args.endprompt_enable
        and args.effective_endprompt_logical_length > config.max_position_embeddings
    ):
        raise ValueError("EndPrompt logical length exceeds the checkpoint's max_position_embeddings")

    _validate_memory_options(args, config, mesh)

    if mesh.shape["ep"] != 1:
        raise ValueError("this fixed architecture requires ep=1")
    tensor_parallel = mesh.shape["tp"]
    divisible = {
        "hidden_size": config.hidden_size,
        "intermediate_size": config.intermediate_size,
        "query width": config.query_width,
        "key/value width": config.key_value_width,
        "vocab_size": config.vocab_size,
        "num_attention_heads": config.num_attention_heads,
    }
    conflicts = [f"{name}={value}" for name, value in divisible.items() if value % tensor_parallel]
    if conflicts:
        raise ValueError(f"TP={tensor_parallel} does not divide: " + ", ".join(conflicts))
    fsdp = mesh.shape["fsdp"]
    fsdp_divisible = {
        "hidden_size": config.hidden_size,
        # The embedding table is sharded over its vocabulary/input axis.
        "vocab_size": config.vocab_size,
    }
    fsdp_conflicts = [
        f"{name}={value}"
        for name, value in fsdp_divisible.items()
        if value % fsdp
    ]
    if fsdp_conflicts:
        raise ValueError(f"FSDP={fsdp} does not divide: " + ", ".join(fsdp_conflicts))
    if args.lora and args.lora_rank % fsdp:
        raise ValueError(
            f"lora_rank={args.lora_rank} must be divisible by FSDP={fsdp} "
            "because FSDP shards the LoRA rank axis"
        )
    sequence_parallel = int(mesh.shape["sp"])
    if args.max_sequence_length % sequence_parallel:
        raise ValueError(
            f"max_sequence_length={args.max_sequence_length} must be divisible by "
            f"SP={sequence_parallel}"
        )
    if sequence_parallel > 1 and args.mlp_chunk_size > 0 and not getattr(args, "projection_chunk_size", 0):
        raise ValueError(
            "mlp_chunk_size must be 0 when SP>1: the current global-sequence "
            "scan would all-gather and redundantly recompute MLP tiles"
        )
    data_partitions = mesh.shape["dp"] * mesh.shape["fsdp"]
    if args.total_batch_size % data_partitions:
        raise ValueError(
            f"total_batch_size={args.total_batch_size} must be divisible by dp*fsdp={data_partitions}"
        )
    if args.attn_mechanism == "splash":
        from ..modeling.config import validate_splash_block_size

        if args.dtype != "bfloat16":
            raise ValueError("Splash Attention requires --dtype bfloat16 on TPU")
        if config.head_dim % 128:
            raise ValueError("Splash Attention requires head_dim divisible by 128")
        effective_block_q = validate_splash_block_size(
            args.max_sequence_length,
            args.block_size_q,
            "block_size_q",
        )
        validate_splash_block_size(args.max_sequence_length, args.block_size_k, "block_size_k")
        local_query_length = args.max_sequence_length // sequence_parallel
        if local_query_length % effective_block_q:
            raise ValueError(
                f"block_size_q={effective_block_q} must divide the SP-local query length "
                f"{local_query_length}"
            )
    elif args.attn_mechanism == "efficient":
        from ..kernels.efficient_attention import resolve_attention_profile

        if args.dtype != "bfloat16":
            raise ValueError("Efficient attention requires --dtype bfloat16 on TPU")
        if config.attention_dropout != 0.0:
            raise ValueError("Efficient attention requires attention_dropout=0")
        if config.num_key_value_heads != 1:
            raise ValueError(
                "unsupported num_key_value_heads for Efficient attention: "
                f"{config.num_key_value_heads}"
            )
        resolve_attention_profile(
            (
                args.total_batch_size,
                args.max_sequence_length,
                config.num_attention_heads,
                config.head_dim,
            ),
            (
                args.total_batch_size,
                args.max_sequence_length,
                config.num_key_value_heads,
                config.head_dim,
            ),
            mesh,
        )


def _resolve_loss_implementation(
    args: TrainingArguments,
    backend: str,
    *,
    cut_mesh_supported: bool = True,
) -> str:
    if backend != "tpu":
        raise ValueError("this trainer requires a TPU JAX backend")
    if args.loss_implementation == "auto":
        implementation = "cut" if cut_mesh_supported else "pallas"
    else:
        implementation = args.loss_implementation
    if getattr(args, "cut_chunk_size", 0) > 0 and (implementation != "cut" or not cut_mesh_supported):
        raise ValueError("cut_chunk_size requires the Cut loss backend and a supported mesh")
    return implementation


_STREAMING_SCHEDULE_METADATA_PERCENT = 105
_STREAMING_SCHEDULE_POLICY = "metadata-105pct-horizon-cap-at-metadata-v1"


def _buffered_streaming_schedule_total(source_examples: int) -> int:
    """Add a fixed 5% tail to a metadata-derived scheduler horizon."""

    if type(source_examples) is not int or source_examples < 0:
        raise ValueError("source_examples must be a non-negative integer")
    return max(
        1,
        (
            source_examples * _STREAMING_SCHEDULE_METADATA_PERCENT
            + 99
        )
        // 100,
    )


def _streaming_schedule_position(
    batch: Any,
    source_examples_per_epoch: int | None,
) -> int:
    """Map real epoch progress into a metadata-estimated schedule horizon.

    Each real epoch advances through at most the unbuffered metadata estimate.
    Extra rows therefore keep the learning rate at the metadata position. The
    complete schedule has a separate fixed 5% tail, so epoch-driven streaming
    never reaches the configured terminal learning rate from source progress.
    """

    if type(source_examples_per_epoch) is not int or source_examples_per_epoch < 0:
        raise ValueError("source_examples_per_epoch must be a non-negative integer")
    epoch = int(batch.epoch)
    epoch_source_before = int(batch.epoch_source_examples_before)
    if epoch < 0 or epoch_source_before < 0:
        raise ValueError("streaming batch epoch progress cannot be negative")
    return (
        epoch * source_examples_per_epoch
        + min(epoch_source_before, source_examples_per_epoch)
    )


def _project_streaming_progress_total(
    source_examples_seen: int,
    completed_epochs: int,
    total_epochs: int,
) -> int:
    """Project a source-row total from completed real epoch boundaries."""

    if type(source_examples_seen) is not int or source_examples_seen < 0:
        raise ValueError("source_examples_seen must be a non-negative integer")
    if type(completed_epochs) is not int or completed_epochs <= 0:
        raise ValueError("completed_epochs must be a positive integer")
    if type(total_epochs) is not int or total_epochs < completed_epochs:
        raise ValueError("total_epochs must be at least completed_epochs")
    return max(
        source_examples_seen,
        (
            source_examples_seen * total_epochs
            + completed_epochs
            - 1
        )
        // completed_epochs,
    )


def _prepared_streaming_limits(
    rows_per_epoch: int,
    effective_batch_size: int,
    epochs: float,
    max_steps: int,
) -> tuple[int, int | None]:
    """Return the exact optimizer cap and source epochs for prepared streaming."""

    if rows_per_epoch <= 0 or effective_batch_size <= 0:
        raise ValueError("prepared streaming rows and batch size must be positive")
    steps_per_epoch = math.ceil(rows_per_epoch / effective_batch_size)
    if max_steps > 0:
        return max_steps, None
    if epochs <= 0:
        raise ValueError("prepared streaming epochs must be positive without max_steps")
    total_steps = max(1, math.floor(steps_per_epoch * epochs))
    source_epochs = max(1, math.ceil(total_steps / steps_per_epoch))
    return total_steps, source_epochs


def _block_metric_tree(jax_module: Any, metrics: Any) -> Any:
    """Synchronize a device metric pytree without copying it to the host."""

    return jax_module.tree.map(lambda value: value.block_until_ready(), metrics)


def _enforce_metric_window(
    jax_module: Any,
    pending: deque[tuple[int, Any]],
    max_pending: int,
) -> tuple[int, ...]:
    if max_pending <= 0:
        raise ValueError("max_pending must be positive")
    completed: list[int] = []
    while len(pending) > max_pending:
        step, metrics = pending.popleft()
        _block_metric_tree(jax_module, metrics)
        completed.append(step)
    return tuple(completed)


def _synchronize_metric_window(
    jax_module: Any,
    pending: deque[tuple[int, Any]],
) -> tuple[int, Any] | None:
    """Wait for the newest queued step, then release every older metric tree."""

    if not pending:
        return None
    latest_step, latest_metrics = pending[-1]
    _block_metric_tree(jax_module, latest_metrics)
    pending.clear()
    return latest_step, latest_metrics


def _write_run_config(args: TrainingArguments) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "training_args.json").open("w", encoding="utf-8") as handle:
        json.dump(_jsonable_arguments(args), handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def _resume_signature_payload(
    args: TrainingArguments,
    config,
    plan,
    dataset: Any,
    mesh: Any,
    *,
    preprocessing_process_count: int,
    prepared_real_rows: int,
    prepared_dummy_rows: int,
    prepared_revision: str | None = None,
) -> dict[str, Any]:
    argument_names = (
        "repo_id",
        "dataset_name",
        "dataset_format",
        "dataset_config_name",
        "dataset_revision",
        "dataset_split",
        "dataset_streaming",
        "dataset_text_field",
        "processor_repo_id",
        "processor_revision",
        "revision",
        "max_sequence_length",
        "packing",
        "packing_batch_size",
        "streaming_shuffle_buffer",
        "assistant_only_loss",
        "last_assistant_only_loss",
        "endprompt_enable",
        "endprompt_logical_length",
        "endprompt_logical_length_min",
        "endprompt_prompts",
        "endprompt_prompt_loss_weight",
        "endprompt_context_loss_weight",
        "preprocessing_mode",
        "attn_mechanism",
        "block_size_q",
        "block_size_k",
        "loss_token_budget",
        "loss_implementation",
        "mlp_chunk_size",
        "scan_layers",
        "total_batch_size",
        "gradient_accumulation_steps",
        "learning_rate",
        "learning_rate_end",
        "scheduler",
        "warmup_steps",
        "weight_decay",
        "max_grad_norm",
        "adam_beta1",
        "adam_beta2",
        "adam_epsilon",
        "num_train_epochs",
        "max_steps",
        "shuffle",
        "seed",
        "gradient_checkpointing",
        "lora",
        "lora_rank",
        "lora_train_embed_and_lm_head",
        "lora_save_adapter_only",
        "frozen_parameters",
        "param_dtype",
        "dtype",
        "sharding_axis",
        "sharding_dcn_axis",
    )
    arguments = {name: getattr(args, name) for name in argument_names}
    arguments.update(_explicit_memory_options(args))
    payload = {
        "arguments": arguments,
        "dataset_fingerprint": getattr(dataset, "_fingerprint", None),
        "preprocessing_process_count": preprocessing_process_count,
        "prepared_real_rows": prepared_real_rows,
        "prepared_dummy_rows": prepared_dummy_rows,
        "mesh_shape": {name: int(size) for name, size in mesh.shape.items()},
        "model_config": config.to_dict(),
        "plan": dataclasses.asdict(plan),
        "streaming_scheduler_policy": (
            _STREAMING_SCHEDULE_POLICY
            if args.uses_raw_streaming and args.max_steps <= 0
            else None
        ),
    }
    if args.dataset_format == "prepared":
        payload["prepared_revision"] = prepared_revision
    return payload


def _resume_action(
    checkpoint: ResumeCheckpoint | None,
    *,
    output_dir: Path,
    total_steps: int,
) -> str:
    """Classify a discovered checkpoint without silently resetting training state."""

    if checkpoint is None:
        return "start"
    if checkpoint.training_complete:
        if checkpoint.path != output_dir:
            raise ValueError("a completed streaming checkpoint must be stored at output_dir")
        return "complete"
    if checkpoint.total_steps != total_steps:
        raise ValueError(
            f"checkpoint total_steps={checkpoint.total_steps} does not match current plan {total_steps}"
        )
    if checkpoint.global_step < total_steps and not checkpoint.has_optimizer_state:
        raise ValueError(
            "the latest checkpoint does not contain optimizer state, so exact continuation is impossible; "
            "optimizer state must have been saved before the interruption with "
            "--save_optimizer_state True, and enabling it only now cannot repair this checkpoint"
        )
    if checkpoint.global_step == total_steps:
        return "complete" if checkpoint.path == output_dir else "finalize"
    return "continue"


def _resume_identity_values(
    checkpoint: ResumeCheckpoint | None,
    *,
    output_dir: Path,
    training_signature: str,
) -> tuple[int, ...]:
    """Build a fixed-width identity that every worker can compare collectively."""

    payload = {
        "checkpoint_id": checkpoint.checkpoint_id if checkpoint is not None else None,
        "exists": checkpoint is not None,
        "global_step": checkpoint.global_step if checkpoint is not None else 0,
        "has_optimizer_state": (
            checkpoint.has_optimizer_state if checkpoint is not None else False
        ),
        "is_output_root": checkpoint is not None and checkpoint.path == output_dir,
        "manifest_digest": (
            checkpoint.manifest_digest if checkpoint is not None else None
        ),
        "source_examples_seen": (
            checkpoint.source_examples_seen if checkpoint is not None else 0
        ),
        "streaming_data_digest": (
            checkpoint.streaming_data_digest if checkpoint is not None else None
        ),
        "total_steps": checkpoint.total_steps if checkpoint is not None else 0,
        "training_complete": (
            checkpoint.training_complete if checkpoint is not None else False
        ),
        "training_signature": training_signature,
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return tuple(hashlib.sha256(encoded).digest())


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    # Importing JAX is safe; device discovery is not. Initialization is the
    # first operation below that can establish a backend client.
    from ..distributed.runtime import DistributedOptions, initialize_distributed, parse_local_device_ids

    initialize_distributed(
        DistributedOptions(
            enabled=args.distributed,
            coordinator_address=args.coordinator_address,
            num_processes=args.num_processes,
            process_id=args.process_id,
            local_device_ids=parse_local_device_ids(args.local_device_ids),
        )
    )

    import jax
    import jax.numpy as jnp
    import numpy as np

    from ..checkpointing.bundles import (
        find_latest_checkpoint,
        prune_periodic_exports,
        save_adapter_checkpoint_bundle,
        save_checkpoint_bundle,
        training_signature,
    )
    from ..checkpointing.huggingface import (
        _raise_if_any_process_error,
        _raise_if_process_zero_error,
        load_hf_config,
        load_optimizer_checkpoint,
        load_sharded_parameters,
        parameter_hf_layout,
        resolve_hf_source,
    )
    from ..data.batching import BatchPlan, iter_global_batches, prefetch_map
    from ..data.pipeline import (
        load_training_split,
        packed_dataset_to_arrays,
        pad_packed_arrays_to_batch_multiple,
        prepare_training_dataset,
        select_process_shard,
        source_dataset_identity,
    )
    from ..data.prepared import (
        iter_prepared_streaming_batches,
        load_prepared_dataset,
        load_prepared_manifest,
        resolve_prepared_revision,
    )
    from ..data.preprocessing import EndPromptSettings
    from ..data.streaming import iter_streaming_batches, streaming_dataset_metadata
    from ..data.streaming_batching import (
        StreamingBatchPlan,
        iter_streaming_global_batches,
        streaming_global_batch_digest,
    )
    from ..data.tokenizer import TrainingTokenizer
    from ..distributed.runtime import (
        batch_sharding,
        create_mesh,
        host_global_to_array,
        merge_packed_host_arrays,
        named_shardings,
        parameter_partition_specs,
        replicated_sharding,
        sync_processes,
    )
    from ..hf_retry import retry_hf_call
    from ..modeling.architecture import parameter_shapes
    from ..modeling.config import ModelConfig
    from ..modeling.freezing import (
        compose_full_trainable_params,
        split_full_trainable_params,
    )
    from ..modeling.lora import (
        compose_lora_export_params,
        init_lora_params,
        lora_adapter_params,
        lora_export_plan,
        lora_parameter_shapes,
        lora_parameter_to_peft_mapping,
        lora_partition_specs,
        make_lora_export_transform,
        peft_adapter_config,
        split_lora_trainable_params,
    )
    from .logging import ExperimentLogger, configure_logging, memory_metrics
    from .optimizer import build_learning_rate_schedule, build_optimizer
    from .steps import (
        infer_optimizer_state_shardings,
        initialize_optimizer,
        make_frozen_full_train_step,
        make_lora_train_step,
        make_train_step,
        optimizer_state_template,
    )

    if jax.process_count() > 1:
        argument_error = None
        try:
            from jax.experimental import multihost_utils

            multihost_utils.assert_equal(
                np.frombuffer(_distributed_arguments_digest(args), dtype=np.uint8),
                fail_message="JAX processes received different training arguments",
            )
        except Exception as exc:
            argument_error = exc
        _raise_if_any_process_error("checking distributed training arguments", argument_error)

    is_primary = jax.process_index() == 0
    logger = configure_logging(is_primary)
    mesh = create_mesh(args.sharding_axis, dcn_axis_dims=args.sharding_dcn_axis)
    raw_streaming = args.uses_raw_streaming
    dataset_uses_assistant_loss_mask = args.uses_assistant_loss_mask
    logger.info("JAX backend=%s global_devices=%d mesh=%s", jax.default_backend(), jax.device_count(), mesh.shape)
    full_frozen_components = () if args.lora else args.frozen_parameter_components
    trainable_lm_head = (
        args.lora_train_embed_and_lm_head
        if args.lora
        else "lm_head" not in full_frozen_components
    )
    cut_mesh_supported = all(
        int(mesh.shape.get(axis, 1)) == 1 for axis in ("fsdp", "ep")
    )
    loss_implementation = _resolve_loss_implementation(
        args,
        jax.default_backend(),
        cut_mesh_supported=cut_mesh_supported,
    )
    if loss_implementation == "cut" and any(
        int(mesh.shape.get(axis, 1)) != 1 for axis in ("fsdp", "ep")
    ):
        raise ValueError(
            "Cut cross-entropy currently requires fsdp=ep=1; use "
            "--loss_implementation pallas for another mesh layout"
        )
    logger.info(
        "cross-entropy implementation=%s trainable_lm_head=%s",
        loss_implementation,
        trainable_lm_head,
    )

    model_source = None
    processor_source = None
    raw_config = None
    config = None
    model_source_error = None
    try:
        model_source = retry_hf_call(
            lambda: resolve_hf_source(
                args.repo_id,
                revision=args.revision,
                token=args.token,
            ),
            initial_delay=args.hf_retry_initial_delay,
            max_delay=args.hf_retry_max_delay,
            operation="resolving the model repository",
        )
        processor_source = (
            model_source
            if args.processor_repo_id is None
            else retry_hf_call(
                lambda: _resolve_processor_source(
                    args.processor_repo_id,
                    revision=args.effective_processor_revision,
                    token=args.token,
                ),
                initial_delay=args.hf_retry_initial_delay,
                max_delay=args.hf_retry_max_delay,
                operation="resolving the tokenizer repository",
            )
        )
        raw_config = load_hf_config(model_source)
        config = ModelConfig.from_dict(raw_config)
        _validate_supported_config(raw_config, config, args, mesh)
    except Exception as exc:
        model_source_error = exc
    _raise_if_any_process_error(
        "resolving and validating model/processor sources",
        model_source_error,
    )
    if model_source is None or processor_source is None or raw_config is None or config is None:
        raise RuntimeError("model/processor source setup failed without a reported error")
    param_dtype = jnp.bfloat16 if args.param_dtype == "bfloat16" else jnp.float32
    compute_dtype = jnp.bfloat16 if args.dtype == "bfloat16" else jnp.float32
    template = parameter_shapes(config, param_dtype)
    lora_template = (
        lora_parameter_shapes(config, args.lora_rank, param_dtype)
        if args.lora
        else None
    )
    expected_model_layout = parameter_hf_layout(template)
    adapter_export_template = None
    expected_adapter_layout = None
    adapter_config = None
    if args.lora:
        if lora_template is None:
            raise RuntimeError("LoRA parameter template was not created")
        if args.lora_train_embed_and_lm_head:
            _, adapter_export_template = split_lora_trainable_params(
                template,
                lora_template,
                train_embed_and_lm_head=True,
            )
        else:
            adapter_export_template = lora_template
        expected_adapter_layout = parameter_hf_layout(
            adapter_export_template,
            mapping_fn=lora_parameter_to_peft_mapping,
        )
        adapter_config = peft_adapter_config(
            args.repo_id,
            args.lora_rank,
            train_embed_and_lm_head=args.lora_train_embed_and_lm_head,
            revision=args.revision,
        )

    tokenizer = None
    if args.dataset_format == "raw":
        tokenizer = TrainingTokenizer.from_directory(processor_source)
        tokenizer.validate_model_vocabulary(config.vocab_size)
        # A tokenizer may intentionally use a padding token distinct from the
        # model config's historical pad ID. Preserve it when defined; the model's
        # ID is only a fallback, matching the established training behavior.
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = config.pad_token_id

    endprompt = None
    if args.endprompt_enable and args.dataset_format == "raw":
        endprompt = EndPromptSettings(
            logical_length_min=args.effective_endprompt_logical_length_min,
            logical_length_max=args.effective_endprompt_logical_length,
            prompts=args.endprompt_prompt_texts,
            prompt_loss_weight=args.endprompt_prompt_loss_weight,
            context_loss_weight=args.endprompt_context_loss_weight,
        )

    prepared_manifest = None
    prepared_revision = None
    source_dataset = None
    source_load_error = None
    try:
        if args.dataset_format == "prepared":
            prepared_revision = retry_hf_call(
                lambda: resolve_prepared_revision(
                    args.dataset_name,
                    token=args.token,
                    revision=args.dataset_revision,
                ),
                initial_delay=args.hf_retry_initial_delay,
                max_delay=args.hf_retry_max_delay,
                operation="resolving the prepared dataset revision",
            )
            prepared_manifest = retry_hf_call(
                lambda: load_prepared_manifest(
                    args.dataset_name,
                    token=args.token,
                    revision=prepared_revision,
                ),
                initial_delay=args.hf_retry_initial_delay,
                max_delay=args.hf_retry_max_delay,
                operation="loading the prepared dataset manifest",
            )
            if prepared_manifest.split != args.dataset_split:
                raise ValueError(
                    f"prepared dataset split is {prepared_manifest.split!r}, "
                    f"but --dataset_split is {args.dataset_split!r}"
                )
            if prepared_manifest.max_sequence_length != args.max_sequence_length:
                raise ValueError(
                    "prepared max_sequence_length="
                    f"{prepared_manifest.max_sequence_length} does not match training "
                    f"max_sequence_length={args.max_sequence_length}"
                )
            tokenizer_metadata = prepared_manifest.tokenizer
            expected_tokenizer_digest = tokenizer_metadata.get("sha256")
            if not isinstance(expected_tokenizer_digest, str):
                raise ValueError(
                    "prepared tokenizer metadata is missing its sha256 digest"
                )
            actual_tokenizer_digest = _prepared_tokenizer_digest(
                processor_source,
                tokenizer_metadata.get("files"),
            )
            if actual_tokenizer_digest != expected_tokenizer_digest:
                raise ValueError(
                    "the training processor tokenizer files do not match the "
                    "tokenizer used by causal-prepare"
                )
            tokenizer_vocab_size = tokenizer_metadata.get("vocab_size")
            if (
                type(tokenizer_vocab_size) is not int
                or tokenizer_vocab_size <= 0
                or tokenizer_vocab_size > config.vocab_size
            ):
                raise ValueError(
                    "prepared tokenizer vocab_size must be positive and no greater "
                    f"than model vocab_size={config.vocab_size}"
                )
            if prepared_manifest.max_token_id >= config.vocab_size:
                raise ValueError(
                    f"prepared token id {prepared_manifest.max_token_id} is outside "
                    f"model vocab_size={config.vocab_size}"
                )
            if prepared_manifest.max_position_id >= config.max_position_embeddings:
                raise ValueError(
                    f"prepared position id {prepared_manifest.max_position_id} is outside "
                    f"model max_position_embeddings={config.max_position_embeddings}"
                )
            dataset_uses_assistant_loss_mask = (
                prepared_manifest.uses_assistant_loss_mask
            )
            source_dataset = retry_hf_call(
                lambda: load_prepared_dataset(
                    args.dataset_name,
                    prepared_manifest,
                    token=args.token,
                    revision=prepared_revision,
                    streaming=args.dataset_streaming,
                    process_index=(
                        None if args.dataset_streaming else jax.process_index()
                    ),
                    process_count=(
                        None if args.dataset_streaming else jax.process_count()
                    ),
                ),
                initial_delay=args.hf_retry_initial_delay,
                max_delay=args.hf_retry_max_delay,
                operation="loading the assigned prepared dataset files",
            )
        else:
            source_dataset = retry_hf_call(
                lambda: load_training_split(
                    args.dataset_name,
                    args.dataset_split,
                    token=args.token,
                    config_name=args.dataset_config_name,
                    revision=args.dataset_revision,
                    streaming=args.dataset_streaming,
                ),
                initial_delay=args.hf_retry_initial_delay,
                max_delay=args.hf_retry_max_delay,
                operation="loading the training dataset",
            )
    except Exception as exc:
        source_load_error = exc
    _raise_if_any_process_error("loading the training dataset", source_load_error)
    if source_dataset is None:
        raise RuntimeError("training dataset loading failed without a reported error")

    streaming_metadata = None
    metadata_error = None
    if args.dataset_streaming and prepared_manifest is None:
        try:
            streaming_metadata = retry_hf_call(
                lambda: streaming_dataset_metadata(
                    source_dataset,
                    args.dataset_split,
                ),
                initial_delay=args.hf_retry_initial_delay,
                max_delay=args.hf_retry_max_delay,
                operation="reading streaming dataset metadata",
            )
        except Exception as exc:
            metadata_error = exc
    _raise_if_any_process_error("reading streaming dataset metadata", metadata_error)
    source_examples_per_epoch = (
        prepared_manifest.total_rows
        if args.dataset_streaming and prepared_manifest is not None
        else (
            streaming_metadata.num_examples
            if streaming_metadata is not None
            else None
        )
    )

    if jax.process_count() > 1:
        source_identity_error = None
        try:
            from jax.experimental import multihost_utils

            if prepared_manifest is not None:
                identity = np.frombuffer(
                    hashlib.sha256(
                        json.dumps(
                            {
                                "manifest": prepared_manifest.global_digest,
                                "revision": prepared_revision,
                            },
                            separators=(",", ":"),
                            sort_keys=True,
                        ).encode("utf-8")
                    ).digest(),
                    dtype=np.uint8,
                ).copy()
            elif args.dataset_streaming:
                features = getattr(source_dataset, "features", None)
                if hasattr(features, "to_dict"):
                    features = features.to_dict()
                identity_payload = {
                    "config": args.dataset_config_name,
                    "features": features,
                    "fingerprint": getattr(source_dataset, "_fingerprint", None),
                    "num_examples": source_examples_per_epoch,
                    "num_shards": getattr(source_dataset, "num_shards", None),
                    "revision": args.dataset_revision,
                    "split": args.dataset_split,
                }
                identity = np.frombuffer(
                    hashlib.sha256(
                        json.dumps(
                            identity_payload,
                            ensure_ascii=False,
                            sort_keys=True,
                            default=str,
                        ).encode("utf-8")
                    ).digest(),
                    dtype=np.uint8,
                ).copy()
            else:
                identity = source_dataset_identity(source_dataset)
            multihost_utils.assert_equal(
                identity,
                fail_message="JAX processes loaded different source datasets",
            )
        except Exception as exc:
            source_identity_error = exc
        _raise_if_any_process_error("checking the source dataset identity", source_identity_error)

    prepared_stream_num_epochs = None
    if args.dataset_streaming:
        dataset = source_dataset
        prepared_real_rows = 0
        prepared_dummy_rows = 0
        local_prepared_rows = 0
        if prepared_manifest is not None:
            prepared_total_steps, prepared_stream_num_epochs = (
                _prepared_streaming_limits(
                    prepared_manifest.total_rows,
                    args.effective_batch_size,
                    args.num_train_epochs,
                    args.max_steps,
                )
            )
            # A prepared manifest has an exact row count. Reuse the streaming
            # envelope/checkpoint machinery with an exact optimizer-step cap,
            # rather than the estimate-driven raw streaming horizon.
            plan = StreamingBatchPlan.create(
                prepared_manifest.total_rows,
                args.total_batch_size,
                args.gradient_accumulation_steps,
                prepared_stream_num_epochs or 1,
                prepared_total_steps,
            )
            if args.shuffle:
                prepared_fields = 6 if prepared_manifest.uses_assistant_loss_mask else 5
                approximate_shuffle_bytes = (
                    args.streaming_shuffle_buffer
                    * prepared_manifest.max_sequence_length
                    * prepared_fields
                    * 4
                )
                if approximate_shuffle_bytes >= 512 * 1024**2:
                    logger.warning(
                        "prepared streaming shuffle may retain about %.1f GiB of "
                        "decoded rows; use --no-shuffle or lower "
                        "--streaming_shuffle_buffer",
                        approximate_shuffle_bytes / 1024**3,
                    )
        else:
            plan = StreamingBatchPlan.create(
                source_examples_per_epoch,
                args.total_batch_size,
                args.gradient_accumulation_steps,
                int(args.num_train_epochs),
                args.max_steps,
            )
        if jax.process_count() > 1:
            logger.info(
                "streaming input is deterministically reproduced on every JAX process; "
                "each global source record is still counted and trained exactly once"
            )
        logger.info(
            "streaming_source_examples_per_epoch_%s=%s "
            "steps_per_epoch_%s=%s effective_global_batch=%d "
            "shuffle_buffer=%d mode=%s",
            "exact" if prepared_manifest is not None else "estimate",
            (
                source_examples_per_epoch
                if source_examples_per_epoch is not None
                else "unknown"
            ),
            "exact" if prepared_manifest is not None else "estimate",
            plan.steps_per_epoch if plan.steps_per_epoch is not None else "unknown",
            plan.effective_batch_size,
            args.streaming_shuffle_buffer,
            (
                "lora+embed+head"
                if args.lora and args.lora_train_embed_and_lm_head
                else (
                    "lora"
                    if args.lora
                    else (
                        "full-frozen-" + "+".join(full_frozen_components)
                        if full_frozen_components
                        else "full"
                    )
                )
            ),
        )
    else:
        local_columns = None
        local_preparation_error = None
        if prepared_manifest is not None:
            try:
                local_columns = packed_dataset_to_arrays(
                    source_dataset,
                    max_sequence_length=args.max_sequence_length,
                    assistant_only_loss=(
                        prepared_manifest.uses_assistant_loss_mask
                    ),
                )
            except Exception as exc:
                local_preparation_error = exc
            _raise_if_any_process_error(
                "reading canonical prepared dataset columns",
                local_preparation_error,
            )
            del source_dataset
        else:
            source_shard = None
            source_start = 0
            shard_error = None
            try:
                if args.preprocessing_mode == "shard_then_merge":
                    source_shard, source_start = select_process_shard(
                        source_dataset,
                        process_index=jax.process_index(),
                        process_count=jax.process_count(),
                    )
                else:
                    source_shard = source_dataset
            except Exception as exc:
                shard_error = exc
            _raise_if_any_process_error(
                "selecting the local preprocessing shard",
                shard_error,
            )
            if source_shard is None:
                raise RuntimeError(
                    "source dataset sharding failed without a reported error"
                )
            del source_dataset

            try:
                if tokenizer is None:
                    raise RuntimeError("raw preprocessing requires a tokenizer")
                prepared_shard = prepare_training_dataset(
                    source_shard,
                    tokenizer,
                    dataset_text_field=args.dataset_text_field,
                    max_sequence_length=args.max_sequence_length,
                    assistant_only_loss=args.assistant_only_loss,
                    last_assistant_only_loss=args.last_assistant_only_loss,
                    endprompt=endprompt,
                    packing=args.packing,
                    packing_batch_size=args.packing_batch_size,
                    preprocessing_num_workers=args.preprocessing_num_workers,
                    example_index_offset=source_start,
                    allow_empty=args.preprocessing_mode == "shard_then_merge",
                )
                local_columns = packed_dataset_to_arrays(
                    prepared_shard,
                    max_sequence_length=args.max_sequence_length,
                    assistant_only_loss=args.assistant_only_loss,
                    last_assistant_only_loss=args.last_assistant_only_loss,
                )
                del prepared_shard
            except Exception as exc:
                local_preparation_error = exc
            _raise_if_any_process_error(
                "tokenizing and packing the local dataset",
                local_preparation_error,
            )
            del source_shard
        if local_columns is None:
            raise RuntimeError("local dataset preparation failed without a reported error")

        local_prepared_rows = next(iter(local_columns.values())).shape[0]
        merged_columns = None
        merge_error = None
        try:
            merged_columns = (
                merge_packed_host_arrays(local_columns)
                if (
                    prepared_manifest is not None
                    or args.preprocessing_mode == "shard_then_merge"
                )
                else local_columns
            )
        except Exception as exc:
            merge_error = exc
        _raise_if_any_process_error("merging prepared dataset shards", merge_error)
        if merged_columns is None:
            raise RuntimeError("prepared dataset merge failed without a reported error")
        if merged_columns is not local_columns:
            del local_columns

        prepared_real_rows = next(iter(merged_columns.values())).shape[0]
        dataset = None
        prepared_dummy_rows = 0
        finalization_error = None
        try:
            dataset, prepared_dummy_rows = pad_packed_arrays_to_batch_multiple(
                merged_columns,
                batch_multiple=args.effective_batch_size,
            )
        except Exception as exc:
            finalization_error = exc
        _raise_if_any_process_error("finalizing the merged prepared dataset", finalization_error)
        if dataset is None:
            raise RuntimeError("prepared dataset finalization failed without a reported error")
        del merged_columns

        if jax.process_count() > 1:
            prepared_identity_error = None
            try:
                from jax.experimental import multihost_utils

                multihost_utils.assert_equal(
                    source_dataset_identity(dataset),
                    fail_message="JAX processes reconstructed different prepared datasets",
                )
            except Exception as exc:
                prepared_identity_error = exc
            _raise_if_any_process_error("checking the prepared dataset identity", prepared_identity_error)

        plan = BatchPlan.create(
            len(dataset),
            args.total_batch_size,
            args.gradient_accumulation_steps,
            args.num_train_epochs,
            args.max_steps,
        )
        logger.info(
            "prepared_real_rows=%d dummy_rows=%d local_process_rows=%d steps_per_epoch=%d "
            "total_steps=%d effective_global_batch=%d preprocessing=%s mode=%s",
            prepared_real_rows,
            prepared_dummy_rows,
            local_prepared_rows,
            plan.steps_per_epoch,
            plan.total_steps,
            plan.effective_batch_size,
            (
                "prepared_files"
                if prepared_manifest is not None
                else args.preprocessing_mode
            ),
            (
                "lora+embed+head"
                if args.lora and args.lora_train_embed_and_lm_head
                else (
                    "lora"
                    if args.lora
                    else (
                        "full-frozen-" + "+".join(full_frozen_components)
                        if full_frozen_components
                        else "full"
                    )
                )
            ),
        )
    if args.lora and args.frozen_parameters:
        if args.lora_train_embed_and_lm_head:
            logger.info(
                "frozen_parameters=%r is accepted for compatibility; "
                "embedding and lm_head remain explicitly trainable",
                args.frozen_parameters,
            )
        else:
            logger.info(
                "frozen_parameters=%r is redundant in LoRA mode because every base-model parameter is frozen",
                args.frozen_parameters,
            )
    resume_signature = training_signature(
        _resume_signature_payload(
            args,
            config,
            plan,
            prepared_manifest if prepared_manifest is not None else dataset,
            mesh,
            preprocessing_process_count=(
                jax.process_count()
                if (
                    args.dataset_format == "raw"
                    and not args.dataset_streaming
                    and args.preprocessing_mode == "shard_then_merge"
                )
                else 1
            ),
            prepared_real_rows=prepared_real_rows,
            prepared_dummy_rows=prepared_dummy_rows,
            prepared_revision=prepared_revision,
        )
    )
    resume_checkpoint = None
    discovery_error = None
    discovery_started_at = time.monotonic()
    logger.info("checking %s for a resumable checkpoint", args.output_dir)
    try:
        resume_checkpoint = find_latest_checkpoint(
            args.output_dir,
            resume_signature,
            expected_model_layout=(
                expected_adapter_layout if args.lora_save_adapter_only else expected_model_layout
            ),
            expected_artifact_kind=(
                "peft-adapter" if args.lora_save_adapter_only else "merged"
            ),
            # The primary process hashes the full payload; every worker validates
            # sizes, headers, and local slices.
            verify_artifact_digests=is_primary,
        )
    except Exception as exc:
        discovery_error = exc
    _raise_if_any_process_error("discovering a resume checkpoint", discovery_error)
    logger.info(
        "checkpoint discovery complete in %.1fs (%s)",
        time.monotonic() - discovery_started_at,
        "checkpoint found" if resume_checkpoint is not None else "starting fresh",
    )
    start_step = resume_checkpoint.global_step if resume_checkpoint is not None else 0
    if jax.process_count() > 1:
        from jax.experimental import multihost_utils

        resume_identity = None
        resume_identity_error = None
        try:
            resume_identity = np.asarray(
                _resume_identity_values(
                    resume_checkpoint,
                    output_dir=args.output_dir,
                    training_signature=resume_signature,
                ),
                dtype=np.uint8,
            )
        except Exception as exc:
            resume_identity_error = exc
        _raise_if_any_process_error(
            "constructing the resume checkpoint identity",
            resume_identity_error,
        )
        if resume_identity is None:
            raise RuntimeError(
                "resume checkpoint identity construction failed without a reported error"
            )
        multihost_utils.assert_equal(
            resume_identity,
            fail_message=(
                "workers discovered different resume checkpoints; synchronize the complete "
                "output_dir checkpoint contents across workers before resuming"
            ),
        )
    if resume_checkpoint is not None:
        if (
            args.lora
            and not args.lora_save_adapter_only
            and resume_checkpoint.global_step < plan.total_steps
            and not resume_checkpoint.training_complete
        ):
            raise ValueError(
                "partial LoRA continuation requires adapter-only checkpoints; "
                "use --lora_save_adapter_only True for resumable LoRA training"
            )
        resume_action = _resume_action(
            resume_checkpoint,
            output_dir=args.output_dir,
            total_steps=plan.total_steps,
        )
        logger.info(
            "found completed checkpoint at %s (step=%d/%d, optimizer=%s)",
            resume_checkpoint.path,
            resume_checkpoint.global_step,
            resume_checkpoint.total_steps,
            resume_checkpoint.has_optimizer_state,
        )
        if (
            resume_checkpoint.global_step < plan.total_steps
            and resume_checkpoint.has_optimizer_state
            and not resume_checkpoint.training_complete
            and not args.save_optimizer_state
        ):
            logger.warning(
                "optimizer saving is disabled for this continuation; future checkpoints will not be resumable"
            )
        if resume_action == "complete":
            logger.info("training is already complete; no updates are required")
            sync_processes("training-already-complete")
            return

    specs = parameter_partition_specs(template)
    with jax.set_mesh(mesh):
        restoring_adapter = resume_checkpoint is not None and args.lora_save_adapter_only
        parameter_source = model_source if restoring_adapter else (
            resume_checkpoint.path if resume_checkpoint is not None else model_source
        )
        params = None
        parameter_load_error = None
        parameter_load_started_at = time.monotonic()
        logger.info("loading and sharding model parameters from %s", parameter_source)
        try:
            params = load_sharded_parameters(
                parameter_source,
                template,
                mesh,
                specs=specs,
                expected_checkpoint_id=(
                    resume_checkpoint.checkpoint_id
                    if resume_checkpoint is not None and not restoring_adapter
                    else None
                ),
            )
            params = jax.tree.map(lambda value: value.block_until_ready(), params)
        except Exception as exc:
            parameter_load_error = exc
        _raise_if_any_process_error("loading model parameters", parameter_load_error)
        if params is None:
            raise RuntimeError("model parameter loading failed without a reported error")
        logger.info(
            "model parameters ready in %.1fs",
            time.monotonic() - parameter_load_started_at,
        )
        base_param_shardings = jax.tree.map(lambda value: value.sharding, params)
        lora_params = None
        full_trainable_params = None
        if args.lora:
            if lora_template is None:
                raise RuntimeError("LoRA parameter template was not created")
            lora_specs = lora_partition_specs(lora_template)
            lora_param_shardings = named_shardings(lora_specs, mesh)
            if args.lora_train_embed_and_lm_head:
                _, adapter_export_specs = split_lora_trainable_params(
                    specs,
                    lora_specs,
                    train_embed_and_lm_head=True,
                )
            else:
                adapter_export_specs = lora_specs
            base_param_shardings, trainable_param_shardings = split_lora_trainable_params(
                base_param_shardings,
                lora_param_shardings,
                train_embed_and_lm_head=args.lora_train_embed_and_lm_head,
            )
            if restoring_adapter:
                if resume_checkpoint is None or adapter_export_template is None:
                    raise RuntimeError("adapter resume metadata is unavailable")
                lora_load_error = None
                try:
                    lora_params = load_sharded_parameters(
                        resume_checkpoint.path,
                        adapter_export_template,
                        mesh,
                        specs=adapter_export_specs,
                        mapping_fn=lora_parameter_to_peft_mapping,
                        expected_checkpoint_id=resume_checkpoint.checkpoint_id,
                    )
                    lora_params = jax.tree.map(
                        lambda value: value.block_until_ready(),
                        lora_params,
                    )
                except Exception as exc:
                    lora_load_error = exc
                _raise_if_any_process_error("loading LoRA trainables", lora_load_error)
                if lora_params is None:
                    raise RuntimeError("LoRA parameter loading failed without a reported error")
                params, _ = split_lora_trainable_params(
                    params,
                    lora_adapter_params(lora_params),
                    train_embed_and_lm_head=args.lora_train_embed_and_lm_head,
                )
            else:

                def initialize_lora(rng):
                    return init_lora_params(config, rng, args.lora_rank, param_dtype)

                adapter_params = jax.jit(
                    initialize_lora,
                    out_shardings=lora_param_shardings,
                )(jax.random.PRNGKey(args.seed))
                adapter_params = jax.tree.map(
                    lambda value: value.block_until_ready(),
                    adapter_params,
                )
                params, lora_params = split_lora_trainable_params(
                    params,
                    adapter_params,
                    train_embed_and_lm_head=args.lora_train_embed_and_lm_head,
                )
            trainable_params = lora_params
        elif full_frozen_components:
            params, full_trainable_params = split_full_trainable_params(
                params,
                full_frozen_components,
            )
            base_param_shardings, trainable_param_shardings = split_full_trainable_params(
                base_param_shardings,
                full_frozen_components,
            )
            trainable_params = full_trainable_params
        else:
            trainable_params = params
            trainable_param_shardings = base_param_shardings

        source_progress_schedule = (
            raw_streaming
            and args.max_steps <= 0
            and plan.total_source_examples is not None
        )
        streaming_schedule_without_total = (
            raw_streaming
            and args.max_steps <= 0
            and plan.total_source_examples is None
        )
        if source_progress_schedule:
            if plan.total_source_examples is None:
                raise RuntimeError("streaming source schedule estimate is unavailable")
            schedule_total_units = _buffered_streaming_schedule_total(
                plan.total_source_examples
            )
            logger.info(
                "streaming scheduler metadata estimate=%d buffered_horizon=%d "
                "buffer=5%%; source progress is capped at the unbuffered metadata "
                "position within each real epoch",
                plan.total_source_examples,
                schedule_total_units,
            )
        else:
            schedule_total_units = plan.total_steps
        schedule_warmup_units = (
            min(
                args.warmup_steps * plan.effective_batch_size,
                max(schedule_total_units - 1, 0),
            )
            if source_progress_schedule
            else args.warmup_steps
        )
        schedule_name = (
            "constant"
            if streaming_schedule_without_total
            else args.scheduler
        )
        if streaming_schedule_without_total and args.scheduler != "constant":
            logger.warning(
                "streaming split metadata has no row-count estimate; "
                "%s decay cannot infer a horizon, so epoch-driven training "
                "will use a constant learning rate after any configured warmup",
                args.scheduler,
            )
        schedule = build_learning_rate_schedule(
            schedule_name,
            args.learning_rate,
            args.learning_rate_end,
            schedule_total_units,
            schedule_warmup_units,
        )
        optimizer = build_optimizer(
            trainable_params,
            schedule,
            weight_decay=args.weight_decay,
            max_grad_norm=args.max_grad_norm,
            beta1=args.adam_beta1,
            beta2=args.adam_beta2,
            epsilon=args.adam_epsilon,
            external_learning_rate=source_progress_schedule,
        )
        optimizer_template = optimizer_state_template(optimizer, trainable_params)
        input_sharding = batch_sharding(mesh)
        scalar_sharding = replicated_sharding(mesh)
        optimizer_state_shardings = infer_optimizer_state_shardings(
            optimizer_template,
            trainable_params,
            scalar_sharding,
        )
        should_restore_optimizer = (
            resume_checkpoint is not None
            and resume_checkpoint.has_optimizer_state
            and (start_step < plan.total_steps or args.save_optimizer_state)
        )
        optimizer_action = (
            "restoring optimizer state"
            if should_restore_optimizer
            else ("initializing AdamW state" if start_step < plan.total_steps else None)
        )
        optimizer_started_at = time.monotonic()
        if optimizer_action is not None:
            logger.info("%s", optimizer_action)
        if should_restore_optimizer:
            optimizer_state = None
            optimizer_load_error = None
            try:
                optimizer_state = load_optimizer_checkpoint(
                    resume_checkpoint.path,
                    optimizer_template,
                    optimizer_state_shardings,
                    expected_step=start_step,
                    expected_checkpoint_id=resume_checkpoint.checkpoint_id,
                )
                optimizer_state = jax.tree.map(
                    lambda value: value.block_until_ready(),
                    optimizer_state,
                )
            except Exception as exc:
                optimizer_load_error = exc
            _raise_if_any_process_error("loading optimizer state", optimizer_load_error)
            if optimizer_state is None:
                raise RuntimeError("optimizer loading failed without a reported error")
        elif start_step == plan.total_steps:
            optimizer_state = None
        else:
            optimizer_state = initialize_optimizer(
                optimizer,
                trainable_params,
                optimizer_state_shardings,
            )
        if optimizer_action is not None:
            logger.info(
                "%s complete in %.1fs",
                optimizer_action,
                time.monotonic() - optimizer_started_at,
            )
        common_step_options = {
            "attention_implementation": args.attn_mechanism,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "remat_policy": (
                "none" if args.gradient_checkpointing == "disabled" else args.gradient_checkpointing
            ),
            "compute_dtype": compute_dtype,
            "block_q": args.block_size_q,
            "block_k": args.block_size_k,
            "loss_token_budget": args.loss_token_budget,
            "loss_implementation": loss_implementation,
            "mlp_chunk_size": args.mlp_chunk_size,
            "scan_layers": args.scan_layers,
            "sparse_loss_skip": dataset_uses_assistant_loss_mask,
            "optimizer_state_shardings": optimizer_state_shardings,
            "batch_named_sharding": input_sharding,
            "replicated_named_sharding": scalar_sharding,
            "external_learning_rate": source_progress_schedule,
        }
        common_step_options.update(_explicit_memory_options(args))
        if args.lora:
            train_step = make_lora_train_step(
                config,
                mesh,
                optimizer,
                schedule,
                base_param_shardings=base_param_shardings,
                lora_param_shardings=trainable_param_shardings,
                train_embed_and_lm_head=args.lora_train_embed_and_lm_head,
                **common_step_options,
            )
        elif full_frozen_components:
            train_step = make_frozen_full_train_step(
                config,
                mesh,
                optimizer,
                schedule,
                frozen_param_shardings=base_param_shardings,
                trainable_param_shardings=trainable_param_shardings,
                lm_head_trainable=trainable_lm_head,
                **common_step_options,
            )
        else:
            train_step = make_train_step(
                config,
                mesh,
                optimizer,
                schedule,
                param_shardings=trainable_param_shardings,
                **common_step_options,
            )

    config_write_error = None
    if is_primary:
        try:
            _write_run_config(args)
        except Exception as exc:
            config_write_error = exc
    _raise_if_process_zero_error("writing training arguments", config_write_error)
    sync_processes("training-output-ready")
    experiment = None
    experiment_error = None
    try:
        experiment = ExperimentLogger(
            args.use_wandb and is_primary,
            args.wandb_project,
            args.wandb_run_name,
            _jsonable_arguments(args),
        )
    except Exception as exc:
        experiment_error = exc
    _raise_if_process_zero_error("initializing experiment logging", experiment_error)
    if experiment is None:
        raise RuntimeError("experiment logging initialization failed without a reported error")
    if args.weight_distribution_log_steps:
        logger.warning(
            "weight_distribution_log_steps is accepted for CLI compatibility but histogram logging is omitted"
        )
    if resume_checkpoint is not None and not resume_checkpoint.has_optimizer_state:
        logger.info("finalizing completed checkpoint without optimizer state at step %d", start_step)

    def save_current_checkpoint(
        destination: Path,
        step: int,
        *,
        source_examples_seen: int = 0,
        training_complete: bool = False,
        total_steps: int | None = None,
        streaming_data_digest: str | None = None,
    ) -> None:
        checkpoint_total_steps = plan.total_steps if total_steps is None else total_steps
        if args.lora_save_adapter_only:
            if lora_params is None or adapter_config is None or expected_adapter_layout is None:
                raise RuntimeError("adapter-only export requires initialized LoRA parameters and metadata")
            save_adapter_checkpoint_bundle(
                lora_params,
                destination,
                optimizer_state=optimizer_state,
                step=step,
                save_optimizer_state=args.save_optimizer_state and optimizer_state is not None,
                tokenizer_source=processor_source,
                adapter_config=adapter_config,
                mapping_fn=lora_parameter_to_peft_mapping,
                training_signature=resume_signature,
                total_steps=checkpoint_total_steps,
                expected_adapter_layout=expected_adapter_layout,
                source_examples_seen=source_examples_seen,
                training_complete=training_complete,
                streaming_data_digest=streaming_data_digest,
            )
            return

        if lora_params is not None:
            checkpoint_params = compose_lora_export_params(params, lora_params)
        elif full_trainable_params is not None:
            checkpoint_params = compose_full_trainable_params(params, full_trainable_params)
        else:
            checkpoint_params = params
        save_checkpoint_bundle(
            checkpoint_params,
            optimizer_state,
            destination,
            step=step,
            save_optimizer_state=args.save_optimizer_state and optimizer_state is not None,
            model_source=model_source,
            tokenizer_source=processor_source,
            config=config,
            training_signature=resume_signature,
            total_steps=checkpoint_total_steps,
            expected_model_layout=expected_model_layout,
            leaf_transform=(
                make_lora_export_transform(lora_adapter_params(lora_params))
                if lora_params is not None
                else None
            ),
            transform_plan=(
                lora_export_plan(
                    args.lora_rank,
                    train_embed_and_lm_head=args.lora_train_embed_and_lm_head,
                )
                if args.lora
                else None
            ),
            source_examples_seen=source_examples_seen,
            training_complete=training_complete,
            streaming_data_digest=streaming_data_digest,
        )

    if args.dataset_streaming:

        def streaming_source_factory(epoch: int):
            stream = dataset
            if args.shuffle:
                shuffle_method = getattr(stream, "shuffle", None)
                if not callable(shuffle_method):
                    raise TypeError("streaming dataset does not provide shuffle()")
                stream = shuffle_method(
                    seed=args.seed,
                    buffer_size=args.streaming_shuffle_buffer,
                )
            set_epoch = getattr(stream, "set_epoch", None)
            if callable(set_epoch):
                set_epoch(epoch)
            return stream

        if prepared_manifest is not None:
            packing_windows = iter_prepared_streaming_batches(
                streaming_source_factory,
                prepared_stream_num_epochs,
                prepared_manifest.total_rows,
                prepared_manifest.max_sequence_length,
                retry_initial_delay=args.hf_retry_initial_delay,
                retry_max_delay=args.hf_retry_max_delay,
                uses_assistant_loss_mask=(
                    prepared_manifest.uses_assistant_loss_mask
                ),
                max_token_id=prepared_manifest.max_token_id,
                max_position_id=prepared_manifest.max_position_id,
            )
        else:
            if tokenizer is None:
                raise RuntimeError("raw streaming requires a tokenizer")
            pad_token_id = tokenizer.pad_token_id
            if pad_token_id is None:
                pad_token_id = tokenizer.eos_token_id
            if pad_token_id is None:
                raise ValueError(
                    "the tokenizer must define pad_token_id or eos_token_id"
                )
            packing_windows = iter_streaming_batches(
                streaming_source_factory,
                tokenizer,
                num_epochs=plan.epochs,
                dataset_text_field=args.dataset_text_field,
                max_sequence_length=args.max_sequence_length,
                pad_token_id=int(pad_token_id),
                assistant_only_loss=args.assistant_only_loss,
                last_assistant_only_loss=args.last_assistant_only_loss,
                endprompt=endprompt,
                packing=args.packing,
                packing_batch_size=args.packing_batch_size,
                retry_initial_delay=args.hf_retry_initial_delay,
                retry_max_delay=args.hf_retry_max_delay,
            )
        host_batches = iter_streaming_global_batches(
            packing_windows,
            plan,
            assistant_only_loss=dataset_uses_assistant_loss_mask,
            start_step=start_step,
            expected_source_examples_before=(
                resume_checkpoint.source_examples_seen
                if resume_checkpoint is not None
                else None
            ),
            expected_stream_digest_before=(
                resume_checkpoint.streaming_data_digest
                if resume_checkpoint is not None
                else None
            ),
        )

        def place_streaming_batch(envelope, *, synchronize: bool):
            placed_data = jax.tree.map(
                lambda value: host_global_to_array(value, input_sharding),
                envelope.data,
            )
            if synchronize:
                # Surface process-local placement/OOM failures before any host
                # is allowed to advance to the next batch-control collective.
                placed_data = jax.tree.map(
                    lambda value: value.block_until_ready(),
                    placed_data,
                )
            return envelope.with_data(placed_data)

        if jax.process_count() > 1:
            from jax.experimental import multihost_utils

            def checked_streaming_batches(values):
                iterator = iter(values)
                batch_index = start_step
                while True:
                    local_batch = None
                    local_error = None
                    status = 0
                    try:
                        local_batch = next(iterator)
                        batch_digest = streaming_global_batch_digest(local_batch)
                        local_batch = place_streaming_batch(
                            local_batch,
                            synchronize=True,
                        )
                    except StopIteration:
                        status = 1
                        batch_digest = bytes(32)
                    except Exception as exc:
                        status = 2
                        local_error = exc
                        batch_digest = hashlib.sha256(
                            f"{type(exc).__name__}: {exc}".encode(
                                "utf-8",
                                errors="replace",
                            )
                        ).digest()

                    control = np.empty(33, dtype=np.uint8)
                    control[0] = status
                    control[1:] = np.frombuffer(batch_digest, dtype=np.uint8)
                    gathered = np.asarray(
                        multihost_utils.process_allgather(
                            control,
                            tiled=False,
                        )
                    ).reshape(-1, control.size)
                    statuses = set(int(value) for value in gathered[:, 0])
                    if 2 in statuses:
                        propagated = RuntimeError(
                            "constructing streaming global batch "
                            f"{batch_index} failed on one or more JAX processes"
                        )
                        if local_error is not None:
                            raise propagated from local_error
                        raise propagated
                    if statuses == {1}:
                        return
                    if statuses != {0}:
                        raise RuntimeError(
                            "JAX processes exhausted the streaming dataset at "
                            f"different points near global batch {batch_index}"
                        )
                    if not np.all(gathered[:, 1:] == gathered[0, 1:]):
                        raise RuntimeError(
                            "JAX processes constructed different streaming global "
                            f"batches at index {batch_index}; pin dataset_revision "
                            "and verify deterministic preprocessing"
                        )
                    if local_batch is None:
                        raise RuntimeError(
                            "streaming batch construction completed without a batch"
                        )
                    yield local_batch
                    batch_index += 1

            host_batches = checked_streaming_batches(host_batches)
            batches = prefetch_map(
                host_batches,
                lambda envelope: envelope,
                prefetch_batches=args.prefetch_batches,
            )
        else:
            batches = prefetch_map(
                host_batches,
                lambda envelope: place_streaming_batch(
                    envelope,
                    synchronize=False,
                ),
                prefetch_batches=args.prefetch_batches,
            )
    else:
        host_batches = iter_global_batches(
            dataset,
            plan,
            assistant_only_loss=dataset_uses_assistant_loss_mask,
            shuffle=args.shuffle,
            seed=args.seed,
            start_step=start_step,
        )
        batches = prefetch_map(
            host_batches,
            lambda host_batch: jax.tree.map(
                lambda value: host_global_to_array(value, input_sharding),
                host_batch,
            ),
            prefetch_batches=args.prefetch_batches,
        )
    start_time = time.monotonic()
    completed_steps = start_step
    committed_source_examples = (
        resume_checkpoint.source_examples_seen
        if args.dataset_streaming and resume_checkpoint is not None
        else 0
    )
    committed_stream_digest = (
        resume_checkpoint.streaming_data_digest
        if args.dataset_streaming and resume_checkpoint is not None
        else None
    )
    starting_source_examples = committed_source_examples
    current_epoch = 0
    committed_epoch_complete = False
    last_logged_step = start_step
    pending_metrics: deque[tuple[int, Any]] = deque()
    pending_commits: deque[
        tuple[int, int, str | None, int, int, bool]
    ] = deque()
    global_batch = None
    device_metrics = None
    progress_bar = None
    progress_uses_source_examples = raw_streaming and args.max_steps <= 0
    progress_total_estimate = (
        plan.total_source_examples
        if progress_uses_source_examples
        else plan.total_steps
    )

    def commit_through(target_step: int) -> None:
        nonlocal committed_source_examples
        nonlocal committed_stream_digest
        nonlocal completed_steps
        nonlocal current_epoch
        nonlocal committed_epoch_complete
        nonlocal progress_total_estimate

        while pending_commits and pending_commits[0][0] <= target_step:
            (
                commit_step,
                source_examples_seen,
                stream_digest,
                epoch,
                source_increment,
                is_epoch_end,
            ) = pending_commits.popleft()
            completed_steps = commit_step
            if args.dataset_streaming:
                committed_source_examples = source_examples_seen
                committed_stream_digest = stream_digest
                current_epoch = epoch
                committed_epoch_complete = is_epoch_end
                if (
                    progress_uses_source_examples
                    and is_epoch_end
                    and plan.epochs is not None
                ):
                    completed_epochs = epoch + 1
                    projected_total = _project_streaming_progress_total(
                        source_examples_seen,
                        completed_epochs,
                        plan.epochs,
                    )
                    if projected_total != progress_total_estimate:
                        logger.info(
                            "updating streaming source progress estimate from %s to %d "
                            "after real EOF for epoch %d",
                            (
                                progress_total_estimate
                                if progress_total_estimate is not None
                                else "unknown"
                            ),
                            projected_total,
                            completed_epochs,
                        )
                        progress_total_estimate = projected_total
                        if progress_bar is not None:
                            progress_bar.total = projected_total
                            progress_bar.refresh()
            if progress_bar is not None:
                progress_bar.update(
                    source_increment if progress_uses_source_examples else 1
                )
        if completed_steps < target_step:
            raise RuntimeError(
                "the asynchronous progress queue lost committed optimizer step "
                f"{target_step}"
            )

    def log_device_metrics(metric_step: int, synchronized_metrics: Any) -> dict[str, float]:
        metrics = {
            key: float(np.asarray(jax.device_get(value)))
            for key, value in synchronized_metrics.items()
        }
        elapsed = max(time.monotonic() - start_time, 1e-6)
        metrics["steps_per_second"] = (metric_step - start_step) / elapsed
        if args.dataset_streaming:
            metrics["source_examples"] = float(committed_source_examples)
            metrics["examples_per_second"] = (
                committed_source_examples - starting_source_examples
            ) / elapsed
            metrics["epoch"] = float(
                current_epoch + int(committed_epoch_complete)
            )
        primary_logging_error = None
        if is_primary:
            try:
                if args.track_memory:
                    metrics.update(memory_metrics(jax))
                experiment.log(metrics, metric_step)
            except Exception as exc:
                primary_logging_error = exc
        if progress_uses_source_examples and progress_total_estimate is not None:
            source_total = progress_total_estimate
            logger.info(
                "step=%d source_examples=%d/%d epoch=%d loss=%.6f "
                "effective_weight=%.3f grad_norm=%.4f lr=%.3e",
                metric_step,
                committed_source_examples,
                source_total,
                current_epoch + 1,
                metrics["loss"],
                metrics["token_count"],
                metrics["grad_norm"],
                metrics["learning_rate"],
            )
        elif args.dataset_streaming:
            logger.info(
                "step=%d/%s source_examples=%d epoch=%d loss=%.6f "
                "effective_weight=%.3f grad_norm=%.4f lr=%.3e",
                metric_step,
                (
                    plan.total_steps
                    if args.max_steps > 0 or prepared_manifest is not None
                    else "?"
                ),
                committed_source_examples,
                current_epoch + 1,
                metrics["loss"],
                metrics["token_count"],
                metrics["grad_norm"],
                metrics["learning_rate"],
            )
        else:
            logger.info(
                "step=%d/%d loss=%.6f effective_weight=%.3f grad_norm=%.4f lr=%.3e",
                metric_step,
                plan.total_steps,
                metrics["loss"],
                metrics["token_count"],
                metrics["grad_norm"],
                metrics["learning_rate"],
            )
        if args.use_wandb or args.track_memory:
            _raise_if_process_zero_error(
                f"logging training metrics at step {metric_step}",
                primary_logging_error,
            )
        return metrics

    first_update_started_at = time.monotonic()
    if start_step < plan.total_steps:
        logger.info(
            "compiling and executing the first training step; a cold TPU compile can take several minutes"
        )
    try:
        for zero_based_step, training_batch in enumerate(batches, start=start_step):
            step = zero_based_step + 1
            if args.dataset_streaming:
                global_batch = training_batch.data
                schedule_position = (
                    _streaming_schedule_position(
                        training_batch,
                        plan.source_examples_per_epoch,
                    )
                    if source_progress_schedule
                    else zero_based_step
                )
            else:
                global_batch = training_batch
                schedule_position = zero_based_step
            with jax.set_mesh(mesh):
                if args.lora:
                    if lora_params is None:
                        raise RuntimeError("LoRA parameters are unavailable")
                    lora_params, optimizer_state, device_metrics = train_step(
                        params,
                        lora_params,
                        optimizer_state,
                        global_batch,
                        jnp.asarray(
                            schedule_position,
                            jnp.float32 if source_progress_schedule else jnp.int32,
                        ),
                    )
                elif full_frozen_components:
                    if full_trainable_params is None:
                        raise RuntimeError("full-parameter trainables are unavailable")
                    full_trainable_params, optimizer_state, device_metrics = train_step(
                        params,
                        full_trainable_params,
                        optimizer_state,
                        global_batch,
                        jnp.asarray(
                            schedule_position,
                            jnp.float32 if source_progress_schedule else jnp.int32,
                        ),
                    )
                else:
                    params, optimizer_state, device_metrics = train_step(
                        params,
                        optimizer_state,
                        global_batch,
                        jnp.asarray(
                            schedule_position,
                            jnp.float32 if source_progress_schedule else jnp.int32,
                        ),
                    )
            pending_metrics.append((step, device_metrics))
            if args.dataset_streaming:
                pending_commits.append(
                    (
                        step,
                        training_batch.source_examples_seen,
                        training_batch.stream_digest,
                        training_batch.epoch,
                        training_batch.source_examples,
                        training_batch.is_epoch_end,
                    )
                )
            else:
                pending_commits.append((step, 0, None, 0, 0, False))
            completed_window = _enforce_metric_window(
                jax,
                pending_metrics,
                args.async_dispatch_steps,
            )
            if completed_window:
                commit_through(completed_window[-1])

            if step % args.logging_steps == 0 or step == start_step + 1:
                synchronized = _synchronize_metric_window(jax, pending_metrics)
                if synchronized is None or synchronized[0] != step:
                    raise RuntimeError("the metric synchronization window lost the current training step")
                commit_through(step)
                metrics = log_device_metrics(step, synchronized[1])
                last_logged_step = step
                if step == start_step + 1:
                    logger.info(
                        "first training step complete in %.1fs (includes tracing, compilation, and execution)",
                        time.monotonic() - first_update_started_at,
                    )
                    if is_primary:
                        # Start timing only after the cold compile so tqdm's rate and ETA
                        # describe steady training instead of compilation latency.
                        progress_total = (
                            progress_total_estimate
                            if progress_uses_source_examples
                            else plan.total_steps
                        )
                        progress_initial = (
                            committed_source_examples
                            if progress_uses_source_examples
                            else step
                        )
                        progress_bar = tqdm(
                            total=progress_total,
                            initial=progress_initial,
                            desc="Training",
                            unit=(
                                "example"
                                if progress_uses_source_examples
                                else "step"
                            ),
                            dynamic_ncols=True,
                            mininterval=1.0,
                        )
                if progress_bar is not None:
                    progress_bar.set_postfix(
                        loss=f"{metrics['loss']:.4f}",
                        lr=f"{metrics['learning_rate']:.2e}",
                        grad=f"{metrics['grad_norm']:.3f}",
                        refresh=True,
                    )

            if args.save_steps > 0 and step % args.save_steps == 0 and step < plan.total_steps:
                checkpoint_metrics = _synchronize_metric_window(jax, pending_metrics)
                if checkpoint_metrics is not None:
                    commit_through(checkpoint_metrics[0])
                if completed_steps != step:
                    raise RuntimeError(
                        f"checkpoint step {step} has not completed on the device"
                    )
                # The current batch is no longer needed. Release its device arrays
                # before checkpoint collectives request their bounded transfer
                # buffers; the prefetch queue remains independently bounded.
                global_batch = None
                training_batch = None
                device_metrics = None
                checkpoint_dir = args.output_dir / f"checkpoint-{step}"
                save_current_checkpoint(
                    checkpoint_dir,
                    step,
                    source_examples_seen=committed_source_examples,
                    streaming_data_digest=committed_stream_digest,
                )
                prune_error = None
                if is_primary:
                    try:
                        prune_periodic_exports(args.output_dir, args.save_total_limit)
                    except Exception as exc:
                        prune_error = exc
                _raise_if_process_zero_error("pruning periodic checkpoints", prune_error)
                sync_processes(f"periodic-prune-{step}")

        final_metrics = _synchronize_metric_window(jax, pending_metrics)
        if final_metrics is not None:
            commit_through(final_metrics[0])
            if final_metrics[0] != last_logged_step:
                metrics = log_device_metrics(final_metrics[0], final_metrics[1])
                last_logged_step = final_metrics[0]
                if progress_bar is not None:
                    progress_bar.set_postfix(
                        loss=f"{metrics['loss']:.4f}",
                        lr=f"{metrics['learning_rate']:.2e}",
                        grad=f"{metrics['grad_norm']:.3f}",
                        refresh=True,
                    )
        if pending_commits:
            raise RuntimeError("the asynchronous progress queue was not fully committed")
    finally:
        if progress_bar is not None:
            progress_bar.close()
    global_batch = None
    device_metrics = None
    if (
        raw_streaming
        and plan.total_source_examples is not None
        and args.max_steps <= 0
        and committed_source_examples != plan.total_source_examples
    ):
        logger.warning(
            "streaming split metadata estimated %d source examples across all epochs, "
            "but real EOF boundaries committed %d; metadata is informational and "
            "training completion follows the real stream",
            plan.total_source_examples,
            committed_source_examples,
        )
    logger.info(
        "training complete; exporting final %s checkpoint to %s",
        "PEFT adapter" if args.lora_save_adapter_only else "Hugging Face model",
        args.output_dir,
    )
    save_current_checkpoint(
        args.output_dir,
        completed_steps,
        source_examples_seen=committed_source_examples,
        training_complete=args.dataset_streaming,
        streaming_data_digest=committed_stream_digest,
        total_steps=(
            max(completed_steps, 1) if args.dataset_streaming else plan.total_steps
        ),
    )
    finish_error = None
    if is_primary:
        try:
            experiment.finish()
        except Exception as exc:
            finish_error = exc
    _raise_if_process_zero_error("finishing experiment logging", finish_error)
    sync_processes("training-complete")


if __name__ == "__main__":
    main()


__all__ = ["main"]
