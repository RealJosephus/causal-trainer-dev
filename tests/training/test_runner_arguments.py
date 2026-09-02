from dataclasses import dataclass, replace
from types import SimpleNamespace

import pytest

from causal_trainer.training.arguments import parse_args
from causal_trainer.training.runner import (
    _PROCESSOR_ALLOW_PATTERNS,
    _buffered_streaming_schedule_total,
    _distributed_arguments_digest,
    _prepared_streaming_limits,
    _prepared_tokenizer_digest,
    _project_streaming_progress_total,
    _resolve_processor_source,
    _resume_signature_payload,
    _streaming_schedule_position,
)


@dataclass(frozen=True)
class _Plan:
    total_steps: int = 1


class _Config:
    @staticmethod
    def to_dict() -> dict[str, int]:
        return {"num_hidden_layers": 40}


def test_separate_processor_resolution_does_not_download_model_weights(
    tmp_path,
    monkeypatch,
) -> None:
    downloaded = tmp_path / "snapshot"
    downloaded.mkdir()
    calls = []

    def fake_snapshot_download(**kwargs):
        calls.append(kwargs)
        return str(downloaded)

    monkeypatch.setattr("huggingface_hub.snapshot_download", fake_snapshot_download)

    assert _resolve_processor_source(
        "org/processor",
        revision="processor-commit",
        token="secret",
    ) == downloaded
    assert calls == [
        {
            "repo_id": "org/processor",
            "revision": "processor-commit",
            "token": "secret",
            "allow_patterns": list(_PROCESSOR_ALLOW_PATTERNS),
        }
    ]
    assert "*.safetensors" not in _PROCESSOR_ALLOW_PATTERNS


def test_prepared_tokenizer_digest_covers_ordered_file_names_and_contents(tmp_path) -> None:
    (tmp_path / "tokenizer.json").write_bytes(b"tokenizer")
    (tmp_path / "tokenizer_config.json").write_bytes(b"config")

    digest = _prepared_tokenizer_digest(
        tmp_path,
        ["tokenizer.json", "tokenizer_config.json"],
    )

    assert len(digest) == 64
    assert digest != _prepared_tokenizer_digest(
        tmp_path,
        ["tokenizer_config.json", "tokenizer.json"],
    )


def test_prepared_tokenizer_digest_allows_huggingface_snapshot_symlinks(tmp_path) -> None:
    snapshot = tmp_path / "snapshots" / "commit"
    blob = tmp_path / "blobs" / "tokenizer-blob"
    snapshot.mkdir(parents=True)
    blob.parent.mkdir()
    blob.write_bytes(b"tokenizer")
    try:
        (snapshot / "tokenizer.json").symlink_to(blob)
    except OSError as exc:
        pytest.skip(f"symbolic links are unavailable: {exc}")

    assert len(_prepared_tokenizer_digest(snapshot, ["tokenizer.json"])) == 64


@pytest.mark.parametrize("filename", ["../tokenizer.json", "C:/tokenizer.json", "C:tokenizer.json"])
def test_prepared_tokenizer_digest_rejects_lexical_path_escape(
    tmp_path,
    filename: str,
) -> None:
    (tmp_path / "tokenizer.json").write_bytes(b"tokenizer")

    with pytest.raises(ValueError, match="escapes its source"):
        _prepared_tokenizer_digest(tmp_path, [filename])


def test_prepared_streaming_limits_use_exact_rows_and_fractional_epochs() -> None:
    assert _prepared_streaming_limits(10, 4, 1.5, -1) == (4, 2)
    assert _prepared_streaming_limits(2, 8, 0.25, -1) == (1, 1)
    assert _prepared_streaming_limits(10, 4, 0.0, 7) == (7, None)


def test_streaming_schedule_position_survives_underestimated_metadata() -> None:
    batches = [
        SimpleNamespace(epoch=0, epoch_source_examples_before=0),
        SimpleNamespace(epoch=0, epoch_source_examples_before=2),
        SimpleNamespace(epoch=0, epoch_source_examples_before=3),
        SimpleNamespace(epoch=0, epoch_source_examples_before=100),
        SimpleNamespace(epoch=1, epoch_source_examples_before=0),
        SimpleNamespace(epoch=1, epoch_source_examples_before=100),
        SimpleNamespace(epoch=2, epoch_source_examples_before=100),
    ]

    positions = [
        _streaming_schedule_position(batch, 3)
        for batch in batches
    ]

    assert positions == [0, 2, 3, 3, 3, 6, 9]
    assert positions == sorted(positions)
    assert positions[-1] < _buffered_streaming_schedule_total(3 * 3)


def test_streaming_schedule_uses_fixed_five_percent_metadata_buffer() -> None:
    assert _buffered_streaming_schedule_total(0) == 1
    assert _buffered_streaming_schedule_total(1) == 2
    assert _buffered_streaming_schedule_total(19) == 20
    assert _buffered_streaming_schedule_total(20) == 21
    assert _buffered_streaming_schedule_total(100) == 105
    assert _buffered_streaming_schedule_total(101) == 107


def test_streaming_schedule_freezes_when_real_rows_exceed_metadata() -> None:
    batches = [
        SimpleNamespace(epoch=0, epoch_source_examples_before=before)
        for before in (3, 4, 100)
    ]

    assert [
        _streaming_schedule_position(batch, 3)
        for batch in batches
    ] == [3, 3, 3]
    assert 3 < _buffered_streaming_schedule_total(3)

    zero_estimate_batch = SimpleNamespace(
        epoch=10,
        epoch_source_examples_before=100,
    )
    assert _streaming_schedule_position(zero_estimate_batch, 0) == 0


def test_real_epoch_boundaries_correct_progress_total_estimate() -> None:
    assert _project_streaming_progress_total(5, 1, 3) == 15
    assert _project_streaming_progress_total(10, 2, 3) == 15
    assert _project_streaming_progress_total(14, 3, 3) == 14


def test_resolved_layer_scan_is_part_of_distributed_and_resume_identity() -> None:
    scanned = parse_args(
        [
            "--repo-id",
            "local-model",
            "--dataset-name",
            "local-data",
            "--max-sequence-length",
            "524288",
        ]
    )
    unrolled = replace(scanned, scan_layers=False)

    assert _distributed_arguments_digest(scanned) != _distributed_arguments_digest(
        unrolled
    )

    payload = _resume_signature_payload(
        scanned,
        _Config(),
        _Plan(),
        SimpleNamespace(_fingerprint="synthetic"),
        SimpleNamespace(shape={"dp": 1, "fsdp": 1, "ep": 1, "tp": 8, "sp": 4}),
        preprocessing_process_count=8,
        prepared_real_rows=1,
        prepared_dummy_rows=0,
    )
    assert payload["arguments"]["scan_layers"] is True
    assert payload["streaming_scheduler_policy"] is None

    streaming_payload = _resume_signature_payload(
        replace(scanned, dataset_streaming=True),
        _Config(),
        _Plan(),
        SimpleNamespace(_fingerprint="synthetic"),
        SimpleNamespace(shape={"dp": 1, "fsdp": 1, "ep": 1, "tp": 8, "sp": 4}),
        preprocessing_process_count=8,
        prepared_real_rows=1,
        prepared_dummy_rows=0,
    )
    assert (
        streaming_payload["streaming_scheduler_policy"]
        == "metadata-105pct-horizon-cap-at-metadata-v1"
    )
    prepared_streaming = replace(
        scanned,
        dataset_format="prepared",
        dataset_streaming=True,
        processor_repo_id="org/processor",
        processor_revision="processor-commit",
    )
    prepared_streaming_payload = _resume_signature_payload(
        prepared_streaming,
        _Config(),
        _Plan(),
        SimpleNamespace(_fingerprint="prepared-manifest-fingerprint"),
        SimpleNamespace(shape={"dp": 1, "fsdp": 1, "ep": 1, "tp": 8, "sp": 4}),
        preprocessing_process_count=1,
        prepared_real_rows=1,
        prepared_dummy_rows=0,
        prepared_revision="dataset-commit",
    )
    assert prepared_streaming_payload["dataset_fingerprint"] == "prepared-manifest-fingerprint"
    assert prepared_streaming_payload["arguments"]["dataset_format"] == "prepared"
    assert prepared_streaming_payload["arguments"]["processor_revision"] == "processor-commit"
    assert prepared_streaming_payload["prepared_revision"] == "dataset-commit"
    assert prepared_streaming_payload["streaming_scheduler_policy"] is None
    assert prepared_streaming_payload != _resume_signature_payload(
        prepared_streaming,
        _Config(),
        _Plan(),
        SimpleNamespace(_fingerprint="prepared-manifest-fingerprint"),
        SimpleNamespace(shape={"dp": 1, "fsdp": 1, "ep": 1, "tp": 8, "sp": 4}),
        preprocessing_process_count=1,
        prepared_real_rows=1,
        prepared_dummy_rows=0,
        prepared_revision="other-dataset-commit",
    )
    assert _distributed_arguments_digest(prepared_streaming) != _distributed_arguments_digest(
        replace(prepared_streaming, dataset_format="raw")
    )
    assert "hf_retry_initial_delay" not in payload["arguments"]
    assert "hf_retry_max_delay" not in payload["arguments"]


def test_full_parameter_freezing_is_part_of_distributed_and_resume_identity() -> None:
    unfrozen = parse_args(
        ["--repo-id", "local-model", "--dataset-name", "local-data"]
    )
    frozen = replace(unfrozen, frozen_parameters="lm_head|embed|norm")

    assert _distributed_arguments_digest(unfrozen) != _distributed_arguments_digest(frozen)

    payload = _resume_signature_payload(
        frozen,
        _Config(),
        _Plan(),
        SimpleNamespace(_fingerprint="synthetic"),
        SimpleNamespace(shape={"dp": 1, "fsdp": 1, "ep": 1, "tp": 8, "sp": 4}),
        preprocessing_process_count=8,
        prepared_real_rows=1,
        prepared_dummy_rows=0,
    )
    assert payload["arguments"]["frozen_parameters"] == "lm_head|embed|norm"


def test_last_assistant_only_loss_is_part_of_distributed_and_resume_identity() -> None:
    all_tokens = parse_args(
        [
            "--repo-id",
            "local-model",
            "--dataset-name",
            "local-data",
            "--dataset-text-field",
            "messages",
        ]
    )
    last_assistant = replace(all_tokens, last_assistant_only_loss=True)

    assert _distributed_arguments_digest(all_tokens) != _distributed_arguments_digest(
        last_assistant
    )
    payload = _resume_signature_payload(
        last_assistant,
        _Config(),
        _Plan(),
        SimpleNamespace(_fingerprint="synthetic"),
        SimpleNamespace(shape={"dp": 1, "fsdp": 1, "ep": 1, "tp": 8, "sp": 4}),
        preprocessing_process_count=8,
        prepared_real_rows=1,
        prepared_dummy_rows=0,
    )

    assert payload["arguments"]["last_assistant_only_loss"] is True
