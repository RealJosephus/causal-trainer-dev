import pytest

from causal_trainer.training.arguments import build_parser, parse_args
from causal_trainer.training.runner import _resolve_loss_implementation


def test_cli_accepts_reference_boolean_style() -> None:
    args = parse_args(
        [
            "--repo_id",
            "local-model",
            "--dataset_name",
            "local-data",
            "--packing",
            "True",
            "--assistant_only_loss",
            "False",
            "--sharding_axis",
            "-1,1,1,4,1",
        ]
    )
    assert args.packing is True
    assert args.assistant_only_loss is False
    assert args.last_assistant_only_loss is False
    assert args.uses_assistant_loss_mask is False
    assert args.sharding_axis == "-1,1,1,4,1"
    assert args.attn_mechanism == "efficient"
    assert args.loss_implementation == "auto"
    assert args.loss_token_budget == 128
    assert args.mlp_chunk_size == 1024
    assert args.scan_layers is False
    assert args.async_dispatch_steps == 2
    assert args.prefetch_batches == 1
    assert args.logging_steps == 10


def test_cli_does_not_expose_remote_tokenizer_code_execution() -> None:
    with pytest.raises(SystemExit):
        parse_args(
            [
                "--repo-id",
                "local-model",
                "--dataset-name",
                "local-data",
                "--trust-remote-code",
                "True",
            ]
        )


@pytest.mark.parametrize(
    "training_mode_args",
    [
        [],
        ["--lora", "True"],
        ["--lora", "True", "--lora-train-embed-and-lm-head", "True"],
    ],
    ids=["full", "lora", "lora_with_embed_and_head"],
)
def test_loss_implementation_auto_uses_cut_for_a_compatible_mesh(
    training_mode_args: list[str],
) -> None:
    args = parse_args(
        ["--repo-id", "local-model", "--dataset-name", "local-data", *training_mode_args]
    )
    assert _resolve_loss_implementation(args, "tpu") == "cut"


@pytest.mark.parametrize(
    "training_mode_args",
    [
        [],
        ["--lora", "True"],
        ["--lora", "True", "--lora-train-embed-and-lm-head", "True"],
    ],
    ids=["full", "lora", "lora_with_embed_and_head"],
)
def test_loss_implementation_auto_uses_pallas_for_an_incompatible_mesh(
    training_mode_args: list[str],
) -> None:
    args = parse_args(
        ["--repo-id", "local-model", "--dataset-name", "local-data", *training_mode_args]
    )
    assert (
        _resolve_loss_implementation(args, "tpu", cut_mesh_supported=False) == "pallas"
    )


@pytest.mark.parametrize("implementation", ["cut", "pallas", "xla"])
def test_explicit_loss_selection_is_unchanged_by_mesh_compatibility(
    implementation: str,
) -> None:
    args = parse_args(
        [
            "--repo-id",
            "local-model",
            "--dataset-name",
            "local-data",
            "--loss-implementation",
            implementation,
        ]
    )
    assert (
        _resolve_loss_implementation(args, "tpu", cut_mesh_supported=False)
        == implementation
    )


def test_training_rejects_non_tpu_backend() -> None:
    args = parse_args(
        [
            "--repo-id",
            "local-model",
            "--dataset-name",
            "local-data",
            "--loss-implementation",
            "pallas",
        ]
    )
    with pytest.raises(ValueError, match="requires a TPU"):
        _resolve_loss_implementation(args, "cpu")


def test_cli_accepts_bare_boolean_and_hyphenated_names() -> None:
    args = parse_args(
        [
            "--repo-id",
            "local-model",
            "--dataset-name",
            "local-data",
            "--packing",
            "--no-shuffle",
        ]
    )
    assert args.packing is True
    assert args.shuffle is False


def test_save_optimizer_state_defaults_to_false() -> None:
    args = parse_args(["--repo-id", "local-model", "--dataset-name", "local-data"])
    assert args.save_optimizer_state is False
    assert args.dataset_format == "raw"
    assert args.dataset_streaming is False
    assert args.uses_raw_streaming is False
    assert args.dataset_config_name is None
    assert args.dataset_revision is None
    assert args.processor_repo_id is None
    assert args.processor_revision is None
    assert args.effective_processor_revision is None
    assert args.streaming_shuffle_buffer == 10_000
    assert args.hf_retry_initial_delay == 1.0
    assert args.hf_retry_max_delay == 60.0
    assert args.learning_rate_end == 0.0
    assert args.endprompt_enable is False
    assert args.frozen_parameters is None
    assert args.frozen_parameter_components == ()
    assert args.lora_train_embed_and_lm_head is False
    assert args.preprocessing_mode == "shard_then_merge"
    assert args.sharding_axis == "-1,1,1,4,1"


def test_processor_revision_is_independent_for_a_separate_repository() -> None:
    separate = parse_args(
        [
            "--repo-id",
            "org/model",
            "--revision",
            "model-commit",
            "--processor-repo-id",
            "org/processor",
            "--dataset-name",
            "local-data",
        ]
    )
    pinned = parse_args(
        [
            "--repo-id",
            "org/model",
            "--revision",
            "model-commit",
            "--processor-repo-id",
            "org/processor",
            "--processor-revision",
            "processor-commit",
            "--dataset-name",
            "local-data",
        ]
    )
    reused_model = parse_args(
        [
            "--repo-id",
            "org/model",
            "--revision",
            "model-commit",
            "--dataset-name",
            "local-data",
        ]
    )

    assert separate.processor_revision is None
    assert separate.effective_processor_revision is None
    assert pinned.effective_processor_revision == "processor-commit"
    assert reused_model.effective_processor_revision == "model-commit"


def test_cli_accepts_streaming_dataset_options() -> None:
    args = parse_args(
        [
            "--repo-id",
            "local-model",
            "--dataset-name",
            "org/large-dataset",
            "--dataset-streaming",
            "--dataset_config_name",
            "clean",
            "--dataset-revision",
            "dataset-commit",
            "--streaming-shuffle-buffer",
            "2048",
            "--hf-retry-initial-delay",
            "0.5",
            "--hf_retry_max_delay",
            "30",
            "--preprocessing-mode",
            "replicated",
            "--num-train-epochs",
            "2",
        ]
    )

    assert args.dataset_streaming is True
    assert args.dataset_format == "raw"
    assert args.uses_raw_streaming is True
    assert args.dataset_config_name == "clean"
    assert args.dataset_revision == "dataset-commit"
    assert args.streaming_shuffle_buffer == 2048
    assert args.hf_retry_initial_delay == 0.5
    assert args.hf_retry_max_delay == 30.0
    assert args.preprocessing_mode == "replicated"


def test_prepared_streaming_is_not_raw_streaming() -> None:
    args = parse_args(
        [
            "--repo-id",
            "local-model",
            "--dataset-name",
            "org/prepared-dataset",
            "--dataset-format",
            "prepared",
            "--dataset-streaming",
            "--num-train-epochs",
            "2",
        ]
    )

    assert args.dataset_format == "prepared"
    assert args.dataset_streaming is True
    assert args.uses_raw_streaming is False


def test_dataset_format_rejects_unknown_values() -> None:
    with pytest.raises(SystemExit):
        parse_args(
            [
                "--repo-id",
                "local-model",
                "--dataset-name",
                "local-data",
                "--dataset-format",
                "auto",
            ]
        )


def test_streaming_dataset_rejects_preprocessing_workers() -> None:
    with pytest.raises(
        ValueError,
        match="preprocessing_num_workers is not supported with dataset_streaming",
    ):
        parse_args(
            [
                "--repo-id",
                "local-model",
                "--dataset-name",
                "org/large-dataset",
                "--dataset-streaming",
                "--preprocessing-num-workers",
                "2",
            ]
        )


@pytest.mark.parametrize("epochs", ["0", "-1", "1.5"])
def test_streaming_epoch_budget_requires_a_positive_integer(epochs: str) -> None:
    with pytest.raises(ValueError, match="num_train_epochs"):
        parse_args(
            [
                "--repo-id",
                "local-model",
                "--dataset-name",
                "org/large-dataset",
                "--dataset-streaming",
                "--num-train-epochs",
                epochs,
            ]
        )


def test_streaming_fractional_epochs_are_allowed_with_max_steps() -> None:
    args = parse_args(
        [
            "--repo-id",
            "local-model",
            "--dataset-name",
            "org/large-dataset",
            "--dataset-streaming",
            "--num-train-epochs",
            "0.5",
            "--max-steps",
            "7",
        ]
    )
    assert args.num_train_epochs == 0.5
    assert args.max_steps == 7


def test_prepared_streaming_allows_fractional_epochs_without_max_steps() -> None:
    args = parse_args(
        [
            "--repo-id",
            "local-model",
            "--dataset-name",
            "org/prepared-dataset",
            "--dataset-format",
            "prepared",
            "--dataset-streaming",
            "--num-train-epochs",
            "0.5",
        ]
    )

    assert args.num_train_epochs == 0.5
    assert args.uses_raw_streaming is False


@pytest.mark.parametrize(
    ("options", "match"),
    [
        (["--streaming-shuffle-buffer", "0"], "streaming_shuffle_buffer must be positive"),
        (["--hf-retry-initial-delay", "0"], "retry delays must be positive"),
        (["--hf-retry-max-delay", "0"], "retry delays must be positive"),
        (["--hf-retry-max-delay", "nan"], "retry delays must be positive"),
        (
            ["--hf-retry-initial-delay", "61", "--hf-retry-max-delay", "60"],
            "hf_retry_initial_delay cannot exceed hf_retry_max_delay",
        ),
    ],
)
def test_streaming_buffer_and_retry_delays_are_validated(
    options: list[str],
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        parse_args(
            [
                "--repo-id",
                "local-model",
                "--dataset-name",
                "org/large-dataset",
                *options,
            ]
        )


@pytest.mark.parametrize(
    "max_sequence_length",
    [2, 512, 2048, 4095, 131072, 262144, 524288],
)
def test_full_training_automatically_disables_mlp_tiling_outside_target_range(
    max_sequence_length: int,
) -> None:
    args = parse_args(
        [
            "--repo-id",
            "local-model",
            "--dataset-name",
            "local-data",
            "--max-sequence-length",
            str(max_sequence_length),
        ]
    )
    assert args.mlp_chunk_size == 0


@pytest.mark.parametrize("max_sequence_length", [4096, 8192, 131071])
def test_full_training_automatically_enables_mlp_tiling_in_target_range(
    max_sequence_length: int,
) -> None:
    args = parse_args(
        [
            "--repo-id",
            "local-model",
            "--dataset-name",
            "local-data",
            "--max-sequence-length",
            str(max_sequence_length),
        ]
    )
    assert args.mlp_chunk_size == 1024


def test_sequence_parallel_training_automatically_disables_mlp_tiling() -> None:
    args = parse_args(
        [
            "--repo-id",
            "local-model",
            "--dataset-name",
            "local-data",
            "--max-sequence-length",
            "8192",
            "--sharding-axis",
            "4,1,1,4,2",
        ]
    )
    assert args.mlp_chunk_size == 0


@pytest.mark.parametrize("train_embed_and_head", [False, True])
def test_lora_automatically_disables_mlp_tiling(
    train_embed_and_head: bool,
) -> None:
    command = [
        "--repo-id",
        "local-model",
        "--dataset-name",
        "local-data",
        "--max-sequence-length",
        "131072",
        "--lora",
        "True",
    ]
    if train_embed_and_head:
        command.extend(["--lora-train-embed-and-lm-head", "True"])
    args = parse_args(command)
    assert args.mlp_chunk_size == 0


@pytest.mark.parametrize(
    ("training_mode_args", "max_sequence_length", "mlp_chunk_size"),
    [
        ([], 512, 1024),
        ([], 131072, 0),
        (["--lora", "True"], 131072, 2048),
        (["--sharding-axis", "4,1,1,4,2"], 8192, 2048),
    ],
    ids=["full_short", "full_long", "lora_long", "sequence_parallel"],
)
def test_explicit_mlp_chunk_size_overrides_automatic_selection(
    training_mode_args: list[str],
    max_sequence_length: int,
    mlp_chunk_size: int,
) -> None:
    args = parse_args(
        [
            "--repo-id",
            "local-model",
            "--dataset-name",
            "local-data",
            "--max-sequence-length",
            str(max_sequence_length),
            *training_mode_args,
            "--mlp-chunk-size",
            str(mlp_chunk_size),
        ]
    )
    assert args.mlp_chunk_size == mlp_chunk_size


@pytest.mark.parametrize(
    ("training_mode_args", "expected"),
    [
        ([], True),
        (["--lora", "True"], False),
        (
            ["--lora", "True", "--lora-train-embed-and-lm-head", "True"],
            False,
        ),
    ],
    ids=["full", "lora", "lora_with_embed_and_head"],
)
def test_layer_scan_automatic_selection_is_limited_to_verified_full_long_context(
    training_mode_args: list[str],
    expected: bool,
) -> None:
    args = parse_args(
        [
            "--repo-id",
            "local-model",
            "--dataset-name",
            "local-data",
            "--max-sequence-length",
            "524288",
            *training_mode_args,
        ]
    )
    assert args.scan_layers is expected

    below_threshold = parse_args(
        [
            "--repo-id",
            "local-model",
            "--dataset-name",
            "local-data",
            "--max-sequence-length",
            "524287",
            *training_mode_args,
        ]
    )
    assert below_threshold.scan_layers is False


def test_explicit_layer_scan_overrides_automatic_selection() -> None:
    enabled = parse_args(
        [
            "--repo-id",
            "local-model",
            "--dataset-name",
            "local-data",
            "--max-sequence-length",
            "8192",
            "--scan-layers",
        ]
    )
    disabled = parse_args(
        [
            "--repo-id",
            "local-model",
            "--dataset-name",
            "local-data",
            "--max-sequence-length",
            "524288",
            "--no-scan-layers",
        ]
    )
    lora_enabled = parse_args(
        [
            "--repo-id",
            "local-model",
            "--dataset-name",
            "local-data",
            "--max-sequence-length",
            "8192",
            "--lora",
            "True",
            "--scan-layers",
        ]
    )
    assert enabled.scan_layers is True
    assert disabled.scan_layers is False
    assert lora_enabled.scan_layers is True


@pytest.mark.parametrize(
    "extra_args",
    [
        ["--mlp-chunk-size", "1024"],
        ["--gradient-checkpointing", "disabled"],
    ],
    ids=["nested_mlp_scan", "checkpointing_disabled"],
)
def test_layer_scan_auto_requires_the_verified_remat_and_untiled_mlp(
    extra_args: list[str],
) -> None:
    args = parse_args(
        [
            "--repo-id",
            "local-model",
            "--dataset-name",
            "local-data",
            "--max-sequence-length",
            "524288",
            *extra_args,
        ]
    )
    assert args.scan_layers is False


def test_automatic_attention_block_sizes_follow_context_threshold() -> None:
    expected_by_context = {
        131071: 128,
        131072: 256,
        262143: 256,
        262144: 512,
        524287: 512,
        524288: 1024,
    }
    for max_sequence_length, expected in expected_by_context.items():
        args = parse_args(
            [
                "--repo-id",
                "local-model",
                "--dataset-name",
                "local-data",
                "--max-sequence-length",
                str(max_sequence_length),
            ]
        )
        assert (args.block_size_q, args.block_size_k) == (expected, expected)


def test_explicit_attention_block_sizes_override_automatic_selection_independently(
) -> None:
    args = parse_args(
        [
            "--repo-id",
            "local-model",
            "--dataset-name",
            "local-data",
            "--max-sequence-length",
            "131072",
            "--block-size-q",
            "64",
            "--block-size-k",
            "512",
        ]
    )
    assert args.block_size_q == 64
    assert args.block_size_k == 512

    q_only = parse_args(
        [
            "--repo-id",
            "local-model",
            "--dataset-name",
            "local-data",
            "--max-sequence-length",
            "131072",
            "--block-size-q",
            "64",
        ]
    )
    assert (q_only.block_size_q, q_only.block_size_k) == (64, 256)

    k_only = parse_args(
        [
            "--repo-id",
            "local-model",
            "--dataset-name",
            "local-data",
            "--max-sequence-length",
            "131072",
            "--block-size-k",
            "512",
        ]
    )
    assert (k_only.block_size_q, k_only.block_size_k) == (256, 512)


@pytest.mark.parametrize(
    "option",
    ["--mlp-chunk-size", "--block-size-q", "--block-size-k"],
)
def test_automatic_tuning_arguments_still_reject_negative_explicit_values(
    option: str,
) -> None:
    with pytest.raises(ValueError):
        parse_args(
            [
                "--repo-id",
                "local-model",
                "--dataset-name",
                "local-data",
                option,
                "-1",
            ]
        )


def test_automatic_tuning_help_describes_context_and_mode_selection() -> None:
    help_text = " ".join(build_parser().format_help().split())
    assert "use 128 below 131072 tokens, 256 below 262144" in help_text
    assert "512 below 524288, and 1024 otherwise" in help_text
    assert "uses 1024 only for contexts from 4096 through 131071 tokens" in help_text
    assert "LoRA and sequence-parallel training disable it automatically" in help_text
    assert "full training at contexts of at least 524288 tokens" in help_text


def test_replicated_preprocessing_remains_available() -> None:
    args = parse_args(
        [
            "--repo-id",
            "local-model",
            "--dataset-name",
            "local-data",
            "--preprocessing-mode",
            "replicated",
        ]
    )
    assert args.preprocessing_mode == "replicated"


@pytest.mark.parametrize(
    ("option", "value"),
    [("--save-optimizer-state", None), ("--save_optimizer_state", "True")],
)
def test_cli_accepts_save_optimizer_state(option: str, value: str | None) -> None:
    command = ["--repo-id", "local-model", "--dataset-name", "local-data", option]
    if value is not None:
        command.append(value)
    args = parse_args(command)
    assert args.save_optimizer_state is True


def test_cli_accepts_lora_reference_options() -> None:
    args = parse_args(
        [
            "--repo-id",
            "local-model",
            "--dataset-name",
            "local-data",
            "--lora",
            "True",
            "--lora-rank",
            "256",
            "--frozen_parameters",
            "lm_head|embed|norm",
        ]
    )
    assert args.lora is True
    assert args.lora_rank == 256
    assert args.frozen_parameters == "lm_head|embed|norm"


def test_cli_enables_embedding_and_head_training_for_lora() -> None:
    args = parse_args(
        [
            "--repo-id",
            "local-model",
            "--dataset-name",
            "local-data",
            "--lora",
            "True",
            "--lora-train-embed-and-lm-head",
            "True",
        ]
    )
    assert args.lora_train_embed_and_lm_head is True


@pytest.mark.parametrize(
    ("value", "expected_components", "canonical"),
    [
        ("lm_head", ("lm_head",), "lm_head"),
        ("embed,norm", ("embed", "norm"), "embed|norm"),
        ("norm|lm_head|embed", ("lm_head", "embed", "norm"), "lm_head|embed|norm"),
    ],
)
def test_cli_selects_full_parameter_components_to_freeze(
    value: str,
    expected_components: tuple[str, ...],
    canonical: str,
) -> None:
    args = parse_args(
        [
            "--repo-id",
            "local-model",
            "--dataset-name",
            "local-data",
            "--frozen-parameters",
            value,
        ]
    )

    assert args.frozen_parameter_components == expected_components
    assert args.frozen_parameters == canonical


@pytest.mark.parametrize("value", ["attention", "|"])
def test_cli_rejects_unsupported_or_empty_frozen_components(value: str) -> None:
    with pytest.raises(ValueError, match="frozen_parameters"):
        parse_args(
            [
                "--repo-id",
                "local-model",
                "--dataset-name",
                "local-data",
                "--frozen-parameters",
                value,
            ]
        )


def test_embedding_and_head_training_requires_lora() -> None:
    with pytest.raises(ValueError, match="requires --lora True"):
        parse_args(
            [
                "--repo-id",
                "local-model",
                "--dataset-name",
                "local-data",
                "--lora-train-embed-and-lm-head",
                "True",
            ]
        )


def test_endprompt_is_allowed_with_packing_and_resolves_logical_lengths() -> None:
    args = parse_args(
        [
            "--repo-id",
            "local-model",
            "--dataset-name",
            "local-data",
            "--packing",
            "True",
            "--max-sequence-length",
            "4096",
            "--endprompt-enable",
            "True",
            "--endprompt-logical-length",
            "8192",
            "--endprompt-logical-length-min",
            "4096",
            "--endprompt-prompts",
            "first||second",
        ]
    )
    assert args.packing is True
    assert args.endprompt_enable is True
    assert args.effective_endprompt_logical_length == 8192
    assert args.effective_endprompt_logical_length_min == 4096
    assert args.endprompt_prompt_texts == ("first", "second")


def test_endprompt_allows_messages_assistant_only_and_packing() -> None:
    args = parse_args(
        [
            "--repo-id",
            "local-model",
            "--dataset-name",
            "local-data",
            "--dataset-text-field",
            "messages",
            "--assistant-only-loss",
            "True",
            "--packing",
            "True",
            "--endprompt-enable",
            "True",
            "--lora",
            "True",
            "--lora-train-embed-and-lm-head",
            "True",
            "--lora-save-adapter-only",
            "True",
        ]
    )
    assert args.assistant_only_loss is True
    assert args.endprompt_enable is True
    assert args.packing is True
    assert args.lora_train_embed_and_lm_head is True
    assert args.lora_save_adapter_only is True


@pytest.mark.parametrize(
    ("option", "value"),
    [
        ("--endprompt-logical-length", "0"),
        ("--endprompt-logical-length", "-1"),
        ("--endprompt-logical-length-min", "0"),
    ],
)
def test_endprompt_rejects_non_positive_logical_lengths(option: str, value: str) -> None:
    with pytest.raises(ValueError, match="logical lengths"):
        parse_args(
            [
                "--repo-id",
                "local-model",
                "--dataset-name",
                "local-data",
                "--endprompt-enable",
                "True",
                option,
                value,
            ]
        )


def test_adapter_only_requires_and_accepts_lora() -> None:
    with pytest.raises(ValueError, match="requires --lora True"):
        parse_args(
            [
                "--repo-id",
                "local-model",
                "--dataset-name",
                "local-data",
                "--lora-save-adapter-only",
                "True",
            ]
        )
    args = parse_args(
        [
            "--repo-id",
            "local-model",
            "--dataset-name",
            "local-data",
            "--lora",
            "True",
            "--lora-save-adapter-only",
            "True",
        ]
    )
    assert args.lora_save_adapter_only is True


def test_lora_optimizer_checkpointing_requires_adapter_only_output() -> None:
    with pytest.raises(ValueError, match="lora_save_adapter_only True"):
        parse_args(
            [
                "--repo-id",
                "local-model",
                "--dataset-name",
                "local-data",
                "--lora",
                "True",
                "--save-optimizer-state",
                "True",
            ]
        )


def test_adapter_only_lora_accepts_optimizer_checkpointing() -> None:
    args = parse_args(
        [
            "--repo-id",
            "local-model",
            "--dataset-name",
            "local-data",
            "--lora",
            "True",
            "--lora-save-adapter-only",
            "True",
            "--save-optimizer-state",
            "True",
        ]
    )
    assert args.save_optimizer_state is True
    assert args.lora_save_adapter_only is True


def test_last_assistant_only_loss_accepts_messages_and_packing() -> None:
    args = parse_args(
        [
            "--repo-id",
            "local-model",
            "--dataset-name",
            "local-data",
            "--dataset-text-field",
            "messages",
            "--last-assistant-only-loss",
            "True",
            "--packing",
            "True",
        ]
    )

    assert args.last_assistant_only_loss is True
    assert args.assistant_only_loss is False
    assert args.uses_assistant_loss_mask is True
    assert args.packing is True


def test_last_assistant_only_loss_requires_messages_field() -> None:
    with pytest.raises(ValueError, match="last_assistant_only_loss requires"):
        parse_args(
            [
                "--repo-id",
                "local-model",
                "--dataset-name",
                "local-data",
                "--last-assistant-only-loss",
                "True",
            ]
        )


def test_assistant_loss_modes_are_mutually_exclusive() -> None:
    with pytest.raises(ValueError, match="mutually exclusive"):
        parse_args(
            [
                "--repo-id",
                "local-model",
                "--dataset-name",
                "local-data",
                "--dataset-text-field",
                "messages",
                "--assistant-only-loss",
                "True",
                "--last-assistant-only-loss",
                "True",
            ]
        )
