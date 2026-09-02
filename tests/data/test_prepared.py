import json
from dataclasses import replace

import pytest
from datasets import IterableDataset

import causal_trainer.data.prepared as prepared
from causal_trainer.data.prepared import (
    PREPARED_DATASET_FORMAT,
    PreparedDatasetManifest,
    PreparedShard,
    assigned_prepared_shards,
    canonical_prepared_dataset,
    iter_prepared_streaming_batches,
    load_prepared_dataset,
)


def _record(value: int, *, assistant: bool = False) -> dict[str, list[int] | list[float]]:
    record: dict[str, list[int] | list[float]] = {
        "input_ids": [value, value + 1, 0, 0],
        "attention_mask": [1, 1, 0, 0],
        "position_ids": [0, 1, 0, 0],
        "segment_ids": [1, 1, 0, 0],
        "loss_weights": [0.0, 1.0, 0.0, 0.0],
    }
    if assistant:
        record["assistant_masks"] = [0, 1, 0, 0]
    return record


def _columns(*values: int, assistant: bool = False):
    records = [_record(value, assistant=assistant) for value in values]
    return {name: [record[name] for record in records] for name in records[0]}


def _manifest(
    shard_count: int = 3,
    *,
    rows_per_shard: int = 1,
    assistant: bool = False,
) -> PreparedDatasetManifest:
    return PreparedDatasetManifest(
        max_sequence_length=4,
        split="train",
        shards=tuple(
            PreparedShard(
                path=f"data/train-{index:05d}-of-{shard_count:05d}.parquet",
                rows=rows_per_shard,
                sha256=f"{index + 1:064x}",
            )
            for index in range(shard_count)
        ),
        tokenizer={"repo_id": "org/tokenizer", "revision": "tokenizer-commit"},
        source={"repo_id": "org/raw", "revision": "source-commit", "split": "train"},
        preprocessing={"packing": True, "packing_batch_size": 1000},
        max_token_id=100,
        max_position_id=3,
        uses_assistant_loss_mask=assistant,
    )


def test_manifest_round_trip_has_strict_current_schema_and_stable_digest() -> None:
    manifest = _manifest(2, assistant=True)
    encoded = manifest.to_json()
    decoded = PreparedDatasetManifest.from_json(encoded)

    assert decoded == manifest
    assert decoded.total_rows == 2
    assert decoded.has_assistant_masks is True
    assert decoded._fingerprint == manifest.global_digest
    assert len(decoded.global_digest) == 64
    payload = json.loads(encoded)
    assert payload["format"] == PREPARED_DATASET_FORMAT
    assert "version" not in payload
    assert payload["schema"]["assistant_masks"] == {"dtype": "int32", "length": 4}


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (lambda value: value.update(total_rows=99), "total_rows"),
        (
            lambda value: value["schema"]["input_ids"].update(dtype="int64"),
            "schema",
        ),
        (lambda value: value.update(version=1), "fields mismatch"),
        (lambda value: value.update(format="some-other-format"), "format marker"),
    ],
)
def test_manifest_rejects_inconsistent_or_unknown_metadata(mutation, match: str) -> None:
    payload = _manifest().to_dict()
    mutation(payload)
    with pytest.raises(ValueError, match=match):
        PreparedDatasetManifest.from_dict(payload)


@pytest.mark.parametrize(
    "path",
    [
        "/absolute.parquet",
        "../escape.parquet",
        "C:/escape.parquet",
        "C:escape.parquet",
        "data\\file.parquet",
        "data/file.json",
    ],
)
def test_manifest_rejects_unsafe_or_non_parquet_shard_paths(path: str) -> None:
    with pytest.raises(ValueError, match="prepared shard path"):
        PreparedShard(path=path, rows=1, sha256="0" * 64)


def test_contiguous_assignment_covers_every_physical_file_once() -> None:
    manifest = _manifest(7)
    assignments = [assigned_prepared_shards(manifest, process_index=index, process_count=3) for index in range(3)]
    assert [len(value) for value in assignments] == [3, 2, 2]
    assert tuple(shard for value in assignments for shard in value) == manifest.shards

    sparse = [assigned_prepared_shards(manifest, process_index=index, process_count=9) for index in range(9)]
    assert tuple(shard for value in sparse for shard in value) == manifest.shards
    assert sum(not value for value in sparse) == 2


def test_resolve_prepared_revision_does_not_query_the_hub_for_local_paths(
    tmp_path,
    monkeypatch,
) -> None:
    manifest_file = tmp_path / prepared.PREPARED_MANIFEST_FILENAME
    manifest_file.write_text("{}", encoding="utf-8")

    monkeypatch.setattr(
        prepared,
        "HfApi",
        lambda: pytest.fail("local prepared paths must not query the Hub"),
    )

    assert prepared.resolve_prepared_revision(str(tmp_path)) is None
    assert prepared.resolve_prepared_revision(str(manifest_file)) is None


def test_resolve_prepared_revision_returns_the_hub_commit(monkeypatch) -> None:
    calls: list[dict[str, str | None]] = []

    class FakeApi:
        def dataset_info(self, **kwargs):
            calls.append(kwargs)
            return type("DatasetInfo", (), {"sha": "prepared-commit"})()

    monkeypatch.setattr(prepared, "HfApi", FakeApi)

    assert (
        prepared.resolve_prepared_revision(
            "org/prepared",
            token="secret",
            revision="data-branch",
        )
        == "prepared-commit"
    )
    assert calls == [
        {
            "repo_id": "org/prepared",
            "revision": "data-branch",
            "token": "secret",
        }
    ]


def test_canonical_dataset_has_fixed_dtypes_and_validates_semantics() -> None:
    dataset = canonical_prepared_dataset(
        _columns(10, 20, assistant=True),
        max_sequence_length=4,
        uses_assistant_loss_mask=True,
    )
    assert len(dataset) == 2
    features = dataset.features.to_dict()
    assert features["input_ids"]["feature"]["dtype"] == "int32"
    assert features["loss_weights"]["feature"]["dtype"] == "float32"
    assert features["input_ids"]["length"] == 4

    bad = _columns(10)
    bad["loss_weights"] = [[0.0, float("nan"), 0.0, 0.0]]
    with pytest.raises(ValueError, match="finite and non-negative"):
        canonical_prepared_dataset(bad, max_sequence_length=4)

    bad = _columns(10)
    bad["segment_ids"] = [[1, 0, 0, 0]]
    with pytest.raises(ValueError, match="segment_ids must be positive exactly"):
        canonical_prepared_dataset(bad, max_sequence_length=4)


@pytest.mark.parametrize(
    ("attention_mask", "position_ids", "segment_ids", "loss_weights"),
    [
        ([1, 1, 1, 1], [0, 0, 0, 1], [1, 2, 1, 1], [0.0, 0.0, 0.0, 1.0]),
        ([1, 1, 0, 1], [0, 1, 0, 0], [1, 1, 0, 1], [0.0, 1.0, 0.0, 0.0]),
    ],
)
def test_canonical_dataset_rejects_non_contiguous_segment_id_reuse(
    attention_mask: list[int],
    position_ids: list[int],
    segment_ids: list[int],
    loss_weights: list[float],
) -> None:
    columns = {
        "input_ids": [[10, 20, 30, 40]],
        "attention_mask": [attention_mask],
        "position_ids": [position_ids],
        "segment_ids": [segment_ids],
        "loss_weights": [loss_weights],
    }

    with pytest.raises(ValueError, match="reuses segment_id=1 after its contiguous run"):
        canonical_prepared_dataset(columns, max_sequence_length=4)


def test_canonical_dataset_accepts_unique_non_monotonic_segment_runs() -> None:
    dataset = canonical_prepared_dataset(
        {
            "input_ids": [[10, 20, 30, 40], [50, 60, 0, 70]],
            "attention_mask": [[1, 1, 1, 1], [1, 1, 0, 1]],
            "position_ids": [[0, 1, 0, 1], [0, 1, 0, 0]],
            "segment_ids": [[3, 3, 1, 1], [1, 1, 0, 2]],
            "loss_weights": [[0.0, 1.0, 0.0, 1.0], [0.0, 1.0, 0.0, 0.0]],
        },
        max_sequence_length=4,
    )

    assert len(dataset) == 2


def test_materialized_hub_loading_downloads_only_the_process_files(monkeypatch) -> None:
    manifest = _manifest(5)
    downloads: list[str] = []
    read_calls: list[list[str]] = []

    def fake_download(**kwargs):
        downloads.append(kwargs["filename"])
        return f"/cache/{kwargs['filename'].replace('/', '-')}"

    def fake_read_files(paths):
        read_calls.append(list(paths))
        return canonical_prepared_dataset(_columns(10, 20), max_sequence_length=4)

    monkeypatch.setattr(prepared, "hf_hub_download", fake_download)
    monkeypatch.setattr(prepared, "_read_parquet_files_in_memory", fake_read_files)
    dataset = load_prepared_dataset(
        "org/prepared",
        manifest,
        revision="prepared-commit",
        process_index=1,
        process_count=2,
        verify_checksums=False,
    )

    expected = [shard.path for shard in manifest.shards[3:]]
    assert downloads == expected
    assert len(dataset) == 2
    assert read_calls == [[f"/cache/{path.replace('/', '-')}" for path in expected]]


def test_materialized_process_with_no_files_returns_canonical_empty_dataset(monkeypatch) -> None:
    manifest = _manifest(1)
    monkeypatch.setattr(
        prepared,
        "hf_hub_download",
        lambda **kwargs: pytest.fail(f"unexpected download: {kwargs}"),
    )
    dataset = load_prepared_dataset(
        "org/prepared",
        manifest,
        process_index=2,
        process_count=3,
    )
    assert len(dataset) == 0
    assert dataset.features.to_dict() == prepared.canonical_prepared_features(4).to_dict()


def test_local_parquet_shards_round_trip_in_process_order_without_arrow_cache(tmp_path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    shards = []
    for index, values in enumerate(((10, 20), (30,), (40, 50))):
        relative = f"data/shard-{index}.parquet"
        path = tmp_path / relative
        canonical_prepared_dataset(
            _columns(*values),
            max_sequence_length=4,
        ).to_parquet(path)
        shards.append(
            PreparedShard(
                path=relative,
                rows=len(values),
                sha256=prepared.sha256_file(path),
            )
        )
    manifest = replace(_manifest(1), shards=tuple(shards))

    rank_zero = load_prepared_dataset(
        str(tmp_path),
        manifest,
        process_index=0,
        process_count=2,
    )
    rank_one = load_prepared_dataset(
        str(tmp_path),
        manifest,
        process_index=1,
        process_count=2,
    )

    assert [row[0] for row in rank_zero["input_ids"]] == [10, 20, 30]
    assert [row[0] for row in rank_one["input_ids"]] == [40, 50]
    assert not list(tmp_path.rglob("*.arrow"))

    corrupt = replace(
        manifest,
        shards=(replace(manifest.shards[0], sha256="0" * 64), *manifest.shards[1:]),
    )
    with pytest.raises(RuntimeError, match="checksum mismatch"):
        load_prepared_dataset(
            str(tmp_path),
            corrupt,
            process_index=0,
            process_count=2,
        )
    with pytest.raises(RuntimeError, match="checksum mismatch"):
        load_prepared_dataset(
            str(tmp_path),
            corrupt,
            streaming=True,
        )


@pytest.mark.parametrize(
    ("field", "values", "match"),
    [
        ("input_ids", [100, 101, 0, 0], "max_token_id=100"),
        ("position_ids", [0, 4, 0, 0], "max_position_id=3"),
    ],
)
def test_materialized_loading_rejects_values_above_manifest_maxima(
    field: str,
    values: list[int],
    match: str,
    tmp_path,
    monkeypatch,
) -> None:
    record = _record(10)
    record[field] = values
    dataset = canonical_prepared_dataset(
        {name: [column] for name, column in record.items()},
        max_sequence_length=4,
    )
    monkeypatch.setattr(
        prepared,
        "_read_parquet_files_in_memory",
        lambda paths: dataset,
    )

    with pytest.raises(ValueError, match=match):
        load_prepared_dataset(
            str(tmp_path),
            _manifest(1),
            verify_checksums=False,
        )


def test_streaming_hub_loading_explicitly_selects_every_manifest_file(monkeypatch) -> None:
    manifest = _manifest(4)
    urls: list[str] = []
    load_calls: list[tuple[tuple, dict]] = []

    def fake_url(**kwargs):
        urls.append(kwargs["filename"])
        return f"https://hub.invalid/{kwargs['filename']}"

    stream = IterableDataset.from_generator(
        lambda: iter([_record(10)]),
        features=prepared.canonical_prepared_features(4),
    )

    def fake_load_dataset(*args, **kwargs):
        load_calls.append((args, kwargs))
        return stream

    monkeypatch.setattr(prepared, "hf_hub_url", fake_url)
    monkeypatch.setattr(prepared, "load_dataset", fake_load_dataset)
    loaded = load_prepared_dataset(
        "org/prepared",
        manifest,
        revision="prepared-commit",
        streaming=True,
        process_index=3,
        process_count=8,
    )

    assert loaded is stream
    assert urls == [shard.path for shard in manifest.shards]
    assert load_calls[0][0] == ("parquet",)
    assert load_calls[0][1]["streaming"] is True
    assert len(load_calls[0][1]["data_files"]["train"]) == len(manifest.shards)


def test_prepared_streaming_is_lazy_and_emits_exact_epoch_marker() -> None:
    consumed: list[int] = []

    def source_factory(epoch: int):
        assert epoch == 0
        for value in (10, 20, 30):
            consumed.append(value)
            yield _record(value)

    iterator = iter_prepared_streaming_batches(
        source_factory,
        num_epochs=1,
        expected_rows_per_epoch=3,
        max_sequence_length=4,
        window_size=2,
    )
    first = next(iterator)
    assert consumed == [10, 20]
    assert first.source_examples == 2
    assert first.record_source_examples == (1, 1)
    assert [record["input_ids"][0] for record in first.records] == [10, 20]

    tail = next(iterator)
    marker = next(iterator)
    assert tail.source_examples == 1
    assert marker.is_epoch_end is True
    with pytest.raises(StopIteration):
        next(iterator)


def test_prepared_streaming_rejects_manifest_row_count_mismatch() -> None:
    def source_factory(epoch: int):
        del epoch
        yield _record(10)

    iterator = iter_prepared_streaming_batches(
        source_factory,
        num_epochs=1,
        expected_rows_per_epoch=2,
        max_sequence_length=4,
    )
    assert next(iterator).source_examples == 1
    with pytest.raises(ValueError, match="produced 1 rows, expected 2"):
        next(iterator)


@pytest.mark.parametrize(
    ("field", "values", "kwargs", "match"),
    [
        (
            "input_ids",
            [100, 101, 0, 0],
            {"max_token_id": 100, "max_position_id": 3},
            "max_token_id=100",
        ),
        (
            "position_ids",
            [0, 4, 0, 0],
            {"max_token_id": 100, "max_position_id": 3},
            "max_position_id=3",
        ),
    ],
)
def test_prepared_streaming_rejects_values_above_manifest_maxima(
    field: str,
    values: list[int],
    kwargs: dict[str, int],
    match: str,
) -> None:
    record = _record(10)
    record[field] = values
    iterator = iter_prepared_streaming_batches(
        lambda epoch: iter([record]),
        num_epochs=1,
        expected_rows_per_epoch=1,
        max_sequence_length=4,
        **kwargs,
    )

    with pytest.raises(ValueError, match=match):
        next(iterator)


def test_manifest_digest_changes_with_file_identity() -> None:
    manifest = _manifest(2)
    changed = replace(
        manifest,
        shards=(
            manifest.shards[0],
            replace(manifest.shards[1], sha256="f" * 64),
        ),
    )
    assert manifest.global_digest != changed.global_digest
