from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from pathlib import Path

from ..modeling.attention_options import AttentionOptions

_FROZEN_PARAMETER_COMPONENTS = ("lm_head", "embed", "norm")


def parse_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"expected a boolean value, got {value!r}")


def parse_frozen_parameter_components(value: str | None) -> tuple[str, ...]:
    if value is None:
        return ()
    requested = {component.strip() for component in value.replace(",", "|").split("|")}
    requested.discard("")
    if not requested:
        raise ValueError("frozen_parameters must name at least one component")
    unsupported = requested - set(_FROZEN_PARAMETER_COMPONENTS)
    if unsupported:
        raise ValueError(
            "unsupported frozen_parameters components: " + ", ".join(sorted(unsupported))
        )
    return tuple(component for component in _FROZEN_PARAMETER_COMPONENTS if component in requested)


def _flags(name: str) -> tuple[str, ...]:
    dashed = f"--{name.replace('_', '-')}"
    underscored = f"--{name}"
    return (dashed,) if dashed == underscored else (dashed, underscored)


def _add(parser: argparse.ArgumentParser, name: str, **kwargs) -> None:
    parser.add_argument(*_flags(name), dest=name, **kwargs)


def _add_bool(parser: argparse.ArgumentParser, name: str, default: bool, help: str) -> None:
    _add(parser, name, nargs="?", const=True, default=default, type=parse_bool, help=help)
    parser.add_argument(f"--no-{name.replace('_', '-')}", dest=name, action="store_false")


@dataclass(frozen=True)
class TrainingArguments:
    repo_id: str
    dataset_name: str
    dataset_format: str
    dataset_split: str
    dataset_text_field: str
    dataset_streaming: bool
    dataset_config_name: str | None
    dataset_revision: str | None
    streaming_shuffle_buffer: int
    hf_retry_initial_delay: float
    hf_retry_max_delay: float
    processor_repo_id: str | None
    processor_revision: str | None
    output_dir: Path
    revision: str | None
    token: str | None
    max_sequence_length: int
    packing: bool
    packing_batch_size: int
    assistant_only_loss: bool
    last_assistant_only_loss: bool
    endprompt_enable: bool
    endprompt_logical_length: int | None
    endprompt_logical_length_min: int | None
    endprompt_prompts: str
    endprompt_prompt_loss_weight: float
    endprompt_context_loss_weight: float
    preprocessing_num_workers: int | None
    preprocessing_mode: str
    attn_mechanism: str
    block_size_q: int
    block_size_k: int
    loss_token_budget: int
    loss_implementation: str
    mlp_chunk_size: int
    scan_layers: bool
    async_dispatch_steps: int
    prefetch_batches: int
    total_batch_size: int
    gradient_accumulation_steps: int
    learning_rate: float
    learning_rate_end: float
    scheduler: str
    warmup_steps: int
    weight_decay: float
    max_grad_norm: float
    adam_beta1: float
    adam_beta2: float
    adam_epsilon: float
    num_train_epochs: float
    max_steps: int
    shuffle: bool
    seed: int
    gradient_checkpointing: str
    lora: bool
    lora_rank: int
    lora_train_embed_and_lm_head: bool
    lora_save_adapter_only: bool
    frozen_parameters: str | None
    param_dtype: str
    dtype: str
    sharding_axis: str
    sharding_dcn_axis: str | None
    coordinator_address: str | None
    num_processes: int | None
    process_id: int | None
    local_device_ids: str | None
    distributed: bool
    save_steps: int
    save_total_limit: int
    save_optimizer_state: bool
    use_wandb: bool
    wandb_project: str
    wandb_run_name: str | None
    logging_steps: int
    track_memory: bool
    weight_distribution_log_steps: int
    projection_chunk_size: int = 0
    remat_group_size: int = 1
    cut_chunk_size: int = 0
    attention_method: str = "dense"
    attention_sparse_backend: str = "auto"
    attention_query_chunk_size: int = 64
    attention_key_chunk_size: int = 512
    moba_block_size: int = 4096
    moba_top_k: int = 12
    attention_dense_prefix_layers: int = 0
    attention_dense_suffix_layers: int = 0
    attention_dense_after_step: int | None = None

    @property
    def attention_options(self) -> AttentionOptions | None:
        options = AttentionOptions(
            method=self.attention_method, backend=self.attention_sparse_backend,
            query_chunk_size=self.attention_query_chunk_size,
            key_chunk_size=self.attention_key_chunk_size,
            moba_block_size=self.moba_block_size, moba_top_k=self.moba_top_k,
            dense_prefix_layers=self.attention_dense_prefix_layers,
            dense_suffix_layers=self.attention_dense_suffix_layers, dense_after_step=self.attention_dense_after_step,
        )
        return None if options.method == "dense" else options

    @property
    def effective_batch_size(self) -> int:
        return self.total_batch_size * self.gradient_accumulation_steps

    @property
    def uses_assistant_loss_mask(self) -> bool:
        return self.assistant_only_loss or self.last_assistant_only_loss

    @property
    def uses_raw_streaming(self) -> bool:
        """Whether records still need online tokenization while streaming."""

        return self.dataset_format == "raw" and self.dataset_streaming

    @property
    def effective_processor_revision(self) -> str | None:
        """Revision used for the effective processor source."""

        return self.revision if self.processor_repo_id is None else self.processor_revision

    @property
    def effective_endprompt_logical_length(self) -> int:
        return (
            self.max_sequence_length
            if self.endprompt_logical_length is None
            else self.endprompt_logical_length
        )

    @property
    def effective_endprompt_logical_length_min(self) -> int:
        return (
            self.effective_endprompt_logical_length
            if self.endprompt_logical_length_min is None
            else self.endprompt_logical_length_min
        )

    @property
    def endprompt_prompt_texts(self) -> tuple[str, ...]:
        return tuple(prompt for prompt in self.endprompt_prompts.split("||") if prompt)

    @property
    def frozen_parameter_components(self) -> tuple[str, ...]:
        return parse_frozen_parameter_components(self.frozen_parameters)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="causal-train",
        description="Distributed full-parameter or LoRA training for the supported decoder architecture.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    _add(parser, "repo_id", required=True, help="Local checkpoint directory or Hugging Face repository ID.")
    _add(parser, "dataset_name", required=True, help="Local dataset file/directory or Hugging Face dataset ID.")
    _add(
        parser,
        "dataset_format",
        choices=("raw", "prepared"),
        default="raw",
        help=(
            "Use 'raw' to tokenize/pack during training, or 'prepared' for a "
            "causal-prepare artifact whose masks and loss weights are already final."
        ),
    )
    _add(parser, "dataset_split", default="train")
    _add(parser, "dataset_text_field", default="text", help="Use 'messages' for conversational records.")
    _add_bool(
        parser,
        "dataset_streaming",
        False,
        "Stream dataset records without materializing the complete split in memory or on local disk.",
    )
    _add(
        parser,
        "dataset_config_name",
        default=None,
        help="Optional Hugging Face dataset configuration/subset name.",
    )
    _add(
        parser,
        "dataset_revision",
        default=None,
        help="Optional Hugging Face dataset revision, independent of the model revision.",
    )
    _add(
        parser,
        "streaming_shuffle_buffer",
        type=int,
        default=10_000,
        help="In-memory shuffle buffer used for streaming datasets.",
    )
    _add(
        parser,
        "hf_retry_initial_delay",
        type=float,
        default=1.0,
        help="Initial delay in seconds before retrying a transient Hugging Face read failure.",
    )
    _add(
        parser,
        "hf_retry_max_delay",
        type=float,
        default=60.0,
        help="Maximum delay in seconds between retries for transient Hugging Face read failures.",
    )
    _add(parser, "processor_repo_id", default=None)
    _add(
        parser,
        "processor_revision",
        default=None,
        help=(
            "Revision for a separately provided processor repository. When no "
            "processor_repo_id is set, the already resolved model source is reused."
        ),
    )
    _add(parser, "output_dir", type=Path, default=Path("causal-trainer-output"))
    _add(parser, "revision", default=None)
    _add(parser, "token", default=None, help="HF token; prefer the HF_TOKEN environment variable.")

    _add(parser, "max_sequence_length", type=int, default=4096)
    _add_bool(parser, "packing", False, "Pack multiple records into fixed-length rows.")
    _add(
        parser,
        "packing_batch_size",
        type=int,
        default=1000,
        help="Per-JAX-process packing window; its final short window is retained.",
    )
    _add_bool(parser, "assistant_only_loss", False, "Supervise only chat-template generation spans.")
    _add_bool(
        parser,
        "last_assistant_only_loss",
        False,
        "Supervise only the final chat-template generation span in each messages record.",
    )
    _add_bool(
        parser,
        "endprompt_enable",
        False,
        "Append a terminal anchor prompt to raw-text or messages records; disabled by default.",
    )
    _add(
        parser,
        "endprompt_logical_length",
        type=int,
        default=None,
        help="Maximum logical context length used for terminal-anchor position IDs.",
    )
    _add(
        parser,
        "endprompt_logical_length_min",
        type=int,
        default=None,
        help="Minimum logical context length; rows sample deterministically between min and max.",
    )
    _add(
        parser,
        "endprompt_prompts",
        default="This is the end of text, please pay attention here",
        help="Terminal prompt text; separate multiple prompts with '||'.",
    )
    _add(parser, "endprompt_prompt_loss_weight", type=float, default=0.1)
    _add(parser, "endprompt_context_loss_weight", type=float, default=1.0)
    _add(
        parser,
        "preprocessing_num_workers",
        type=int,
        default=None,
        help=(
            "Tokenizer worker processes used independently inside each JAX process; "
            "not supported by the single-process streaming pipeline."
        ),
    )
    _add(
        parser,
        "preprocessing_mode",
        choices=("shard_then_merge", "replicated"),
        default="shard_then_merge",
        help=(
            "Shard source rows across JAX processes, preprocess/pack locally, and merge the prepared "
            "arrays over the network before training; 'replicated' preprocesses all rows everywhere. "
            "Streaming uses a deterministic replicated stream on every process and ignores this setting."
        ),
    )

    _add(
        parser,
        "attn_mechanism",
        choices=("splash", "vanilla", "efficient"),
        default="efficient",
        help=(
            "Attention backend. Defaults to the efficient training kernel; "
            "'splash' selects the official JAX implementation and 'vanilla' provides reference checks."
        ),
    )
    _add(parser, "attention_method", choices=("dense", "moba"), default="dense",
         help="Opt-in attention approximation, independent of the dense attention backend.")
    _add(parser, "attention_sparse_backend", choices=("auto", "pallas", "reference"), default="auto",
         help="Sparse kernel: auto uses Pallas on TPU and the numerical reference on CPU; no TPU fallback.")
    for name, default, help_text in (
        ("attention_query_chunk_size", 64, "Sparse attention query workspace tile."),
        ("attention_key_chunk_size", 512, "Sparse attention key workspace tile."),
        ("moba_block_size", 4096, "Sample-relative MoBA routing block size."),
        ("moba_top_k", 12, "MoBA block budget per query/head, including its current block."),
        ("attention_dense_prefix_layers", 0, "Leading layers kept on the original dense backend."),
        ("attention_dense_suffix_layers", 0, "Trailing layers kept on the original dense backend."),
    ):
        _add(parser, name, type=int, default=default, help=help_text)
    _add(parser, "attention_dense_after_step", type=int, default=None,
         help="Use dense attention starting at this zero-based optimizer step (including after resume).")
    _add(
        parser,
        "block_size_q",
        type=int,
        default=None,
        help=(
            "Splash Attention query block size. If omitted, use 128 below 131072 tokens, "
            "256 below 262144, 512 below 524288, and 1024 otherwise."
        ),
    )
    _add(
        parser,
        "block_size_k",
        type=int,
        default=None,
        help=(
            "Splash Attention key block size. If omitted, use 128 below 131072 tokens, "
            "256 below 262144, 512 below 524288, and 1024 otherwise."
        ),
    )
    _add(
        parser,
        "loss_token_budget",
        type=int,
        default=128,
        help=(
            "Maximum local token rows in an XLA/Pallas projected-logits slab; "
            "projection-fused Cut CE is independently tiled and ignores this budget."
        ),
    )
    _add(
        parser,
        "loss_implementation",
        choices=("auto", "xla", "pallas", "cut"),
        default="auto",
        help=(
            "Cross-entropy backend. 'auto' selects between direct Cut CE and chunked TPU Pallas "
            "according to mesh support; 'xla' provides the analytic-VJP reference implementation."
        ),
    )
    _add(
        parser,
        "mlp_chunk_size",
        type=int,
        default=None,
        help=(
            "Static per-sequence tile size for tiled MLP execution. If omitted, full training "
            "uses 1024 only for contexts from 4096 through 131071 tokens and disables tiling "
            "outside that range; LoRA and sequence-parallel training disable it automatically. "
            "Each device processes local_batch_size * mlp_chunk_size token rows per tile; "
            "explicit 0 disables tiling. A positive projection_chunk_size permits explicit "
            "MLP tiles within each SP shard."
        ),
    )
    _add(
        parser,
        "scan_layers",
        nargs="?",
        const=True,
        default=None,
        type=parse_bool,
        help=(
            "Represent the decoder layer loop with lax.scan. If omitted, enable it only for "
            "full training at contexts of at least 524288 tokens with an untiled MLP and "
            "nothing-saveable rematerialization; LoRA remains opt-in."
        ),
    )
    parser.add_argument(
        "--no-scan-layers",
        dest="scan_layers",
        action="store_false",
        default=None,
    )
    _add(parser, "projection_chunk_size", type=int, default=0,
         help="SP-local token rows per projection tile; 0 preserves ordinary projections.")
    _add(parser, "remat_group_size", type=int, default=1,
         help="Layers per rematerialized scan group; 1 preserves the ordinary layer loop.")
    _add(parser, "cut_chunk_size", type=int, default=0,
         help="Local token rows per Cut loss chunk; 0 preserves ordinary Cut execution.")
    _add(
        parser,
        "async_dispatch_steps",
        type=int,
        default=2,
        help="Maximum number of completed-step metric pytrees retained without host synchronization.",
    )
    _add(
        parser,
        "prefetch_batches",
        type=int,
        default=1,
        help="Number of converted global JAX batches prepared ahead of the batch being trained.",
    )
    _add(parser, "total_batch_size", type=int, default=32, help="Global micro-batch size.")
    _add(parser, "gradient_accumulation_steps", type=int, default=1)
    _add(parser, "learning_rate", type=float, default=2e-5)
    _add(parser, "learning_rate_end", type=float, default=0.0)
    _add(parser, "scheduler", choices=("cosine", "linear", "constant"), default="cosine")
    _add(parser, "warmup_steps", type=int, default=0)
    _add(parser, "weight_decay", type=float, default=0.0)
    _add(parser, "max_grad_norm", type=float, default=1.0)
    _add(parser, "adam_beta1", type=float, default=0.9)
    _add(parser, "adam_beta2", type=float, default=0.999)
    _add(parser, "adam_epsilon", type=float, default=1e-8)
    _add(parser, "num_train_epochs", type=float, default=1.0)
    _add(parser, "max_steps", type=int, default=-1)
    _add_bool(parser, "shuffle", True, "Shuffle records deterministically each epoch.")
    _add(parser, "seed", type=int, default=42)
    _add(
        parser,
        "gradient_checkpointing",
        choices=("nothing_saveable", "disabled"),
        default="nothing_saveable",
    )
    _add_bool(
        parser,
        "lora",
        False,
        "Train rank-decomposition adapters on the seven decoder projections; exports are merged by default.",
    )
    _add(parser, "lora_rank", type=int, default=256)
    _add_bool(
        parser,
        "lora_train_embed_and_lm_head",
        False,
        "Alongside LoRA adapters, update the token embedding and language-model head.",
    )
    _add_bool(
        parser,
        "lora_save_adapter_only",
        False,
        "Save a PEFT-compatible, resumable adapter checkpoint instead of merged model weights.",
    )
    _add(
        parser,
        "frozen_parameters",
        default=None,
        help=(
            "Pipe- or comma-separated components to freeze during full-rank training. Supported components "
            "are lm_head, embed, and norm. In LoRA mode this is accepted for command compatibility because "
            "the base model is already frozen."
        ),
    )
    _add(parser, "param_dtype", choices=("bfloat16", "float32"), default="bfloat16")
    _add(parser, "dtype", choices=("bfloat16", "float32"), default="bfloat16")
    _add(parser, "sharding_axis", default="-1,1,1,4,1", help="dp,fsdp,ep,tp,sp mesh dimensions.")
    _add(parser, "sharding_dcn_axis", default=None, help="Optional explicit DCN mesh dimensions.")

    _add(parser, "coordinator_address", default=None)
    _add(parser, "num_processes", type=int, default=None)
    _add(parser, "process_id", type=int, default=None)
    _add(parser, "local_device_ids", default=None, help="Comma-separated local device IDs.")
    _add_bool(parser, "distributed", True, "Auto-initialize JAX distributed on a multi-host environment.")

    _add(parser, "save_steps", type=int, default=0, help="Periodic HF export interval; 0 disables it.")
    _add(parser, "save_total_limit", type=int, default=1, help="Periodic exports to retain; 0 keeps all.")
    _add_bool(
        parser,
        "save_optimizer_state",
        False,
        "Save optimizer state alongside every periodic and final training checkpoint.",
    )
    _add_bool(parser, "use_wandb", False, "Log scalar metrics to Weights & Biases on process zero.")
    _add(parser, "wandb_project", default="causal-trainer")
    _add(parser, "wandb_run_name", default=None)
    _add(
        parser,
        "logging_steps",
        type=int,
        default=10,
        help="Host-log interval; the first and final update are always logged.",
    )
    _add_bool(parser, "track_memory", False, "Log JAX device memory statistics when available.")
    _add(parser, "weight_distribution_log_steps", type=int, default=0)
    return parser


def parse_args(argv: list[str] | None = None) -> TrainingArguments:
    # argparse treats a comma-separated mesh beginning with ``-1`` as another
    # option instead of as a value. Preserve compatibility with the existing
    # quoted ``--sharding_axis "-1, 1, ..."`` command style by joining it to
    # the option before normal parsing.
    raw = list(sys.argv[1:] if argv is None else argv)
    for index in range(len(raw) - 1):
        if raw[index] in {"--sharding-axis", "--sharding_axis", "--sharding-dcn-axis", "--sharding_dcn_axis"}:
            if raw[index + 1].lstrip().startswith("-1") and "," in raw[index + 1]:
                raw[index] = f"{raw[index]}={raw[index + 1]}"
                raw[index + 1] = ""
    namespace = build_parser().parse_args([value for value in raw if value])
    if namespace.max_sequence_length < 131072:
        automatic_attention_block_size = 128
    elif namespace.max_sequence_length < 262144:
        automatic_attention_block_size = 256
    elif namespace.max_sequence_length < 524288:
        automatic_attention_block_size = 512
    else:
        automatic_attention_block_size = 1024
    if namespace.block_size_q is None:
        namespace.block_size_q = automatic_attention_block_size
    if namespace.block_size_k is None:
        namespace.block_size_k = automatic_attention_block_size
    if namespace.mlp_chunk_size is None:
        # Preserve the existing defaults, including untiled MLP execution
        # under SP. Explicit projection_chunk_size enables shard-local MLP
        # tiles when mlp_chunk_size is supplied; it does not change this default.
        try:
            requested_sp = int(namespace.sharding_axis.split(",")[-1].strip())
        except (AttributeError, ValueError):
            requested_sp = -1
        namespace.mlp_chunk_size = (
            1024
            if (
                not namespace.lora
                and requested_sp == 1
                and 4096 <= namespace.max_sequence_length < 131072
            )
            else 0
        )
    if namespace.scan_layers is None:
        # Full-model TPU measurements show that scanning the 40-layer loop is
        # shape-dependent: it raises compiled HBM at S8192 but enables the
        # verified S524288 frontier. Keep LoRA explicit until equivalent
        # capacity and performance measurements are available.
        namespace.scan_layers = (
            not namespace.lora
            and namespace.max_sequence_length >= 524288
            and namespace.mlp_chunk_size == 0
            and namespace.gradient_checkpointing == "nothing_saveable"
        )
    if namespace.frozen_parameters is not None:
        namespace.frozen_parameters = "|".join(
            parse_frozen_parameter_components(namespace.frozen_parameters)
        )
    args = TrainingArguments(**vars(namespace))
    _ = args.attention_options  # Validate algorithm options before model/data loading.
    if args.projection_chunk_size < 0 or args.cut_chunk_size < 0 or args.remat_group_size < 1:
        raise ValueError(
            "projection_chunk_size and cut_chunk_size must be nonnegative; remat_group_size must be positive"
        )
    explicit_memory = args.projection_chunk_size > 0 or args.cut_chunk_size > 0 or args.remat_group_size > 1
    if explicit_memory and (args.lora or args.frozen_parameters):
        raise ValueError("projection_chunk_size, remat_group_size and cut_chunk_size require full-parameter training")
    if args.projection_chunk_size > 0:
        if args.dtype != "bfloat16" or args.param_dtype != "bfloat16":
            raise ValueError("projection_chunk_size requires BF16 compute and parameters")
    if args.projection_chunk_size > 0:
        if args.attn_mechanism != "efficient" or args.total_batch_size != 1:
            raise ValueError("projection_chunk_size requires efficient attention and total_batch_size=1")
    if args.remat_group_size > 1 and (not args.scan_layers or args.gradient_checkpointing != "nothing_saveable"):
        raise ValueError("remat_group_size>1 requires scan_layers=True and gradient_checkpointing=nothing_saveable")
    if args.cut_chunk_size > 0 and args.loss_implementation not in {"auto", "cut"}:
        raise ValueError("cut_chunk_size requires the Cut loss backend")
    if args.max_sequence_length <= 1:
        raise ValueError("max_sequence_length must be greater than one")
    if args.num_train_epochs <= 0 and args.max_steps <= 0:
        raise ValueError("num_train_epochs must be positive unless max_steps is set")
    if (
        args.uses_raw_streaming
        and args.max_steps <= 0
        and not args.num_train_epochs.is_integer()
    ):
        raise ValueError(
            "streaming datasets require an integer num_train_epochs unless max_steps is set"
        )
    if args.total_batch_size <= 0 or args.gradient_accumulation_steps <= 0:
        raise ValueError("batch size and gradient accumulation must be positive")
    if args.packing_batch_size <= 0 or args.logging_steps <= 0:
        raise ValueError("packing_batch_size and logging_steps must be positive")
    if args.streaming_shuffle_buffer <= 0:
        raise ValueError("streaming_shuffle_buffer must be positive")
    if (
        not math.isfinite(args.hf_retry_initial_delay)
        or not math.isfinite(args.hf_retry_max_delay)
        or args.hf_retry_initial_delay <= 0
        or args.hf_retry_max_delay <= 0
    ):
        raise ValueError("Hugging Face retry delays must be positive")
    if args.hf_retry_initial_delay > args.hf_retry_max_delay:
        raise ValueError("hf_retry_initial_delay cannot exceed hf_retry_max_delay")
    if args.uses_raw_streaming and args.preprocessing_num_workers is not None:
        raise ValueError(
            "preprocessing_num_workers is not supported with dataset_streaming"
        )
    if args.preprocessing_num_workers is not None and args.preprocessing_num_workers <= 0:
        raise ValueError("preprocessing_num_workers must be positive when provided")
    if args.warmup_steps < 0 or args.weight_distribution_log_steps < 0 or args.seed < 0:
        raise ValueError("warmup_steps, weight_distribution_log_steps, and seed cannot be negative")
    if args.block_size_q <= 0 or args.block_size_k <= 0 or args.loss_token_budget <= 0:
        raise ValueError("attention block sizes and loss_token_budget must be positive")
    if args.mlp_chunk_size < 0:
        raise ValueError("mlp_chunk_size cannot be negative")
    if args.async_dispatch_steps <= 0 or args.prefetch_batches < 0:
        raise ValueError("async_dispatch_steps must be positive and prefetch_batches cannot be negative")
    if args.lora_rank <= 0:
        raise ValueError("lora_rank must be positive")
    if args.lora_train_embed_and_lm_head and not args.lora:
        raise ValueError("lora_train_embed_and_lm_head requires --lora True")
    if args.lora_save_adapter_only and not args.lora:
        raise ValueError("lora_save_adapter_only requires --lora True")
    if args.save_steps < 0 or args.save_total_limit < 0:
        raise ValueError("checkpoint intervals and limits must be non-negative")
    if args.learning_rate <= 0 or args.learning_rate_end < 0:
        raise ValueError("learning rates must be non-negative and the initial rate must be positive")
    if args.learning_rate_end > args.learning_rate and args.scheduler != "constant":
        raise ValueError("learning_rate_end cannot exceed learning_rate for a decay schedule")
    if args.assistant_only_loss and args.last_assistant_only_loss:
        raise ValueError("assistant_only_loss and last_assistant_only_loss are mutually exclusive")
    if (
        args.dataset_format == "raw"
        and args.uses_assistant_loss_mask
        and args.dataset_text_field != "messages"
    ):
        option = (
            "last_assistant_only_loss"
            if args.last_assistant_only_loss
            else "assistant_only_loss"
        )
        raise ValueError(f"{option} requires --dataset_text_field messages")
    if args.endprompt_enable:
        logical_max = args.effective_endprompt_logical_length
        logical_min = args.effective_endprompt_logical_length_min
        if logical_max < args.max_sequence_length or logical_min < args.max_sequence_length:
            raise ValueError("EndPrompt logical lengths must be at least max_sequence_length")
        if logical_min > logical_max:
            raise ValueError("endprompt_logical_length_min cannot exceed endprompt_logical_length")
        if not args.endprompt_prompt_texts:
            raise ValueError("endprompt_prompts must contain at least one non-empty prompt")
        weights = (args.endprompt_context_loss_weight, args.endprompt_prompt_loss_weight)
        if any(not math.isfinite(weight) or weight < 0.0 for weight in weights):
            raise ValueError("EndPrompt loss weights must be finite and non-negative")
        if not any(weight > 0.0 for weight in weights):
            raise ValueError("at least one EndPrompt loss weight must be positive")
    if args.lora and args.save_optimizer_state and not args.lora_save_adapter_only:
        raise ValueError(
            "LoRA optimizer state can be saved only with "
            "--lora_save_adapter_only True"
        )
    return args


__all__ = [
    "TrainingArguments",
    "build_parser",
    "parse_args",
    "parse_bool",
    "parse_frozen_parameter_components",
]
