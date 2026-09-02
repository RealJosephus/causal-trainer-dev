"""Prepared dataset manifests, loading, validation, and streaming.

The prepared-data boundary is deliberately independent of JAX.  An offline
process can materialize the canonical fixed-shape columns, upload Parquet
files, and write one small manifest.  Training processes can then either
download only their assigned physical files or stream the complete logical
split without running the tokenizer or packer again.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import re
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from itertools import islice
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

import numpy as np
from datasets import Dataset, Features, IterableDataset, Value, load_dataset
from datasets import Sequence as DatasetSequence
from huggingface_hub import HfApi, hf_hub_download, hf_hub_url

from .pipeline import MODEL_FIELDS, contiguous_shard_bounds
from .streaming import StreamingBatchEnvelope, StreamingSourceFactory, iter_retrying_records

PREPARED_DATASET_FORMAT = "causal-trainer-prepared"
PREPARED_MANIFEST_FILENAME = "causal-trainer-prepared.json"
PREPARED_REQUIRED_FIELDS = (*MODEL_FIELDS, "loss_weights")
_MANIFEST_DIGEST_DOMAIN = b"causal-trainer-prepared-manifest\0"
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_SPLIT_RE = re.compile(r"[A-Za-z0-9_][A-Za-z0-9._-]*")


def _require_plain_int(value: Any, *, name: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer greater than or equal to {minimum}")
    return value


def _normalize_json_object(value: Any, *, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a JSON object")
    if any(not isinstance(key, str) for key in value):
        raise TypeError(f"{name} keys must be strings")
    try:
        encoded = json.dumps(
            dict(value),
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain finite JSON values") from exc
    decoded = json.loads(encoded)
    if not isinstance(decoded, dict):  # pragma: no cover - guarded above
        raise TypeError(f"{name} must be a JSON object")
    return decoded


def _validate_prepared_path(value: Any) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError("prepared shard path must be a non-empty string")
    if "\\" in value:
        raise ValueError("prepared shard paths must use forward slashes")
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or PureWindowsPath(value).drive
        or str(path) != value
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise ValueError(f"prepared shard path must be a normalized relative path, got {value!r}")
    if path.suffix.lower() != ".parquet":
        raise ValueError(f"prepared shard path must name a Parquet file, got {value!r}")
    return value


@dataclass(frozen=True, slots=True)
class PreparedShard:
    """One physical Parquet file in manifest order."""

    path: str
    rows: int
    sha256: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "path", _validate_prepared_path(self.path))
        _require_plain_int(self.rows, name="prepared shard rows")
        if not isinstance(self.sha256, str) or _SHA256_RE.fullmatch(self.sha256) is None:
            raise ValueError("prepared shard sha256 must be a lowercase SHA-256 digest")

    def to_dict(self) -> dict[str, Any]:
        return {"path": self.path, "rows": self.rows, "sha256": self.sha256}

    @classmethod
    def from_dict(cls, value: Any) -> PreparedShard:
        if not isinstance(value, Mapping):
            raise TypeError("prepared shard metadata must be a JSON object")
        expected = {"path", "rows", "sha256"}
        actual = set(value)
        if actual != expected:
            raise ValueError(
                "prepared shard metadata fields mismatch: "
                f"missing={sorted(expected - actual)}, extra={sorted(actual - expected)}"
            )
        return cls(path=value["path"], rows=value["rows"], sha256=value["sha256"])


def _schema_dict(
    max_sequence_length: int,
    *,
    uses_assistant_loss_mask: bool,
) -> dict[str, dict[str, Any]]:
    names = [*MODEL_FIELDS]
    if uses_assistant_loss_mask:
        names.append("assistant_masks")
    names.append("loss_weights")
    return {
        name: {
            "dtype": "float32" if name == "loss_weights" else "int32",
            "length": max_sequence_length,
        }
        for name in names
    }


@dataclass(frozen=True, slots=True)
class PreparedDatasetManifest:
    """Complete identity and physical layout of one prepared split."""

    max_sequence_length: int
    split: str
    shards: tuple[PreparedShard, ...]
    tokenizer: Mapping[str, Any]
    source: Mapping[str, Any]
    preprocessing: Mapping[str, Any]
    max_token_id: int
    max_position_id: int
    uses_assistant_loss_mask: bool = False

    def __post_init__(self) -> None:
        _require_plain_int(
            self.max_sequence_length,
            name="prepared max_sequence_length",
            minimum=1,
        )
        if not isinstance(self.split, str) or _SPLIT_RE.fullmatch(self.split) is None:
            raise ValueError(f"invalid prepared split name: {self.split!r}")
        if not isinstance(self.shards, tuple):
            object.__setattr__(self, "shards", tuple(self.shards))
        if not self.shards or any(not isinstance(shard, PreparedShard) for shard in self.shards):
            raise ValueError("prepared manifest requires at least one PreparedShard")
        paths = [shard.path for shard in self.shards]
        if len(paths) != len(set(paths)):
            raise ValueError("prepared manifest shard paths must be unique")
        if sum(shard.rows for shard in self.shards) <= 0:
            raise ValueError("prepared manifest must contain at least one row")
        _require_plain_int(self.max_token_id, name="prepared max_token_id")
        _require_plain_int(self.max_position_id, name="prepared max_position_id")
        if type(self.uses_assistant_loss_mask) is not bool:
            raise TypeError("uses_assistant_loss_mask must be a boolean")
        object.__setattr__(
            self,
            "tokenizer",
            _normalize_json_object(self.tokenizer, name="prepared tokenizer metadata"),
        )
        object.__setattr__(
            self,
            "source",
            _normalize_json_object(self.source, name="prepared source metadata"),
        )
        object.__setattr__(
            self,
            "preprocessing",
            _normalize_json_object(self.preprocessing, name="prepared preprocessing metadata"),
        )

    @property
    def total_rows(self) -> int:
        return sum(shard.rows for shard in self.shards)

    @property
    def has_assistant_masks(self) -> bool:
        return self.uses_assistant_loss_mask

    @property
    def schema(self) -> dict[str, dict[str, Any]]:
        return _schema_dict(
            self.max_sequence_length,
            uses_assistant_loss_mask=self.uses_assistant_loss_mask,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "format": PREPARED_DATASET_FORMAT,
            "split": self.split,
            "max_sequence_length": self.max_sequence_length,
            "total_rows": self.total_rows,
            "max_token_id": self.max_token_id,
            "max_position_id": self.max_position_id,
            "uses_assistant_loss_mask": self.uses_assistant_loss_mask,
            "schema": self.schema,
            "shards": [shard.to_dict() for shard in self.shards],
            "tokenizer": dict(self.tokenizer),
            "source": dict(self.source),
            "preprocessing": dict(self.preprocessing),
        }

    @classmethod
    def from_dict(cls, value: Any) -> PreparedDatasetManifest:
        if not isinstance(value, Mapping):
            raise TypeError("prepared manifest must be a JSON object")
        expected = {
            "format",
            "split",
            "max_sequence_length",
            "total_rows",
            "max_token_id",
            "max_position_id",
            "uses_assistant_loss_mask",
            "schema",
            "shards",
            "tokenizer",
            "source",
            "preprocessing",
        }
        actual = set(value)
        if actual != expected:
            raise ValueError(
                "prepared manifest fields mismatch: "
                f"missing={sorted(expected - actual)}, extra={sorted(actual - expected)}"
            )
        if value["format"] != PREPARED_DATASET_FORMAT:
            raise ValueError(f"unsupported prepared dataset format marker: {value['format']!r}")
        shards_value = value["shards"]
        if not isinstance(shards_value, Sequence) or isinstance(shards_value, (str, bytes)):
            raise TypeError("prepared manifest shards must be a JSON array")
        manifest = cls(
            max_sequence_length=value["max_sequence_length"],
            split=value["split"],
            shards=tuple(PreparedShard.from_dict(shard) for shard in shards_value),
            tokenizer=value["tokenizer"],
            source=value["source"],
            preprocessing=value["preprocessing"],
            max_token_id=value["max_token_id"],
            max_position_id=value["max_position_id"],
            uses_assistant_loss_mask=value["uses_assistant_loss_mask"],
        )
        if value["total_rows"] != manifest.total_rows:
            raise ValueError(
                f"prepared manifest total_rows={value['total_rows']!r} does not match "
                f"the shard total {manifest.total_rows}"
            )
        if value["schema"] != manifest.schema:
            raise ValueError("prepared manifest schema does not match its canonical settings")
        return manifest

    def to_json(self, *, indent: int | None = 2) -> str:
        return json.dumps(
            self.to_dict(),
            ensure_ascii=False,
            allow_nan=False,
            indent=indent,
            sort_keys=True,
        ) + ("\n" if indent is not None else "")

    @classmethod
    def from_json(cls, value: str | bytes | bytearray) -> PreparedDatasetManifest:
        try:
            decoded = json.loads(value)
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise ValueError(f"invalid prepared manifest JSON: {exc}") from exc
        return cls.from_dict(decoded)

    @property
    def global_digest(self) -> str:
        payload = json.dumps(
            self.to_dict(),
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        return hashlib.sha256(_MANIFEST_DIGEST_DOMAIN + payload).hexdigest()

    @property
    def digest(self) -> str:
        return self.global_digest

    @property
    def _fingerprint(self) -> str:
        """Datasets-compatible stable identity used by resume signatures."""

        return self.global_digest


def sha256_file(path: str | Path, *, chunk_size: int = 1024 * 1024) -> str:
    if type(chunk_size) is not int or chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer")
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def read_prepared_manifest(path: str | Path) -> PreparedDatasetManifest:
    return PreparedDatasetManifest.from_json(Path(path).read_bytes())


def write_prepared_manifest(
    manifest: PreparedDatasetManifest,
    path: str | Path,
) -> None:
    if not isinstance(manifest, PreparedDatasetManifest):
        raise TypeError("manifest must be a PreparedDatasetManifest")
    Path(path).write_text(manifest.to_json(), encoding="utf-8")


def load_prepared_manifest(
    name: str,
    *,
    token: str | None = None,
    revision: str | None = None,
    cache_dir: str | Path | None = None,
    filename: str = PREPARED_MANIFEST_FILENAME,
) -> PreparedDatasetManifest:
    """Load a local manifest/dataset directory or one small Hub manifest file."""

    candidate = Path(name).expanduser()
    if candidate.is_file():
        return read_prepared_manifest(candidate)
    if candidate.is_dir():
        return read_prepared_manifest(candidate / filename)
    downloaded = hf_hub_download(
        repo_id=name,
        filename=filename,
        repo_type="dataset",
        revision=revision,
        token=token,
        cache_dir=None if cache_dir is None else str(cache_dir),
    )
    return read_prepared_manifest(downloaded)


def resolve_prepared_revision(
    name: str,
    *,
    token: str | None = None,
    revision: str | None = None,
) -> str | None:
    """Resolve a Hub dataset reference once so manifest and shards cannot race."""

    if Path(name).expanduser().exists():
        return None
    info = HfApi().dataset_info(
        repo_id=name,
        revision=revision,
        token=token,
    )
    commit = getattr(info, "sha", None)
    if not isinstance(commit, str) or not commit:
        raise RuntimeError("Hugging Face did not return a prepared dataset commit SHA")
    return commit


def assigned_prepared_shards(
    manifest: PreparedDatasetManifest,
    *,
    process_index: int,
    process_count: int,
) -> tuple[PreparedShard, ...]:
    """Assign one contiguous manifest interval to a training process."""

    if not isinstance(manifest, PreparedDatasetManifest):
        raise TypeError("manifest must be a PreparedDatasetManifest")
    start, stop = contiguous_shard_bounds(
        len(manifest.shards),
        process_index,
        process_count,
    )
    return manifest.shards[start:stop]


def canonical_prepared_features(
    max_sequence_length: int,
    *,
    uses_assistant_loss_mask: bool = False,
) -> Features:
    _require_plain_int(max_sequence_length, name="max_sequence_length", minimum=1)
    if type(uses_assistant_loss_mask) is not bool:
        raise TypeError("uses_assistant_loss_mask must be a boolean")
    schema = _schema_dict(
        max_sequence_length,
        uses_assistant_loss_mask=uses_assistant_loss_mask,
    )
    return Features(
        {
            name: DatasetSequence(
                Value(field["dtype"]),
                length=max_sequence_length,
            )
            for name, field in schema.items()
        }
    )


def _canonical_field_names(*, uses_assistant_loss_mask: bool) -> tuple[str, ...]:
    return (
        *MODEL_FIELDS,
        *(("assistant_masks",) if uses_assistant_loss_mask else ()),
        "loss_weights",
    )


def _as_batch_array(
    value: Any,
    *,
    name: str,
    rows: int,
    max_sequence_length: int,
) -> np.ndarray:
    expected = (rows, max_sequence_length)
    try:
        array = np.asarray(value)
    except ValueError as exc:
        raise ValueError(f"prepared field {name!r} must be rectangular with shape {expected}") from exc
    if array.shape != expected:
        raise ValueError(f"prepared field {name!r} has shape {array.shape}, expected {expected}")
    if name == "loss_weights":
        if array.dtype.kind not in "fiu":
            raise TypeError("prepared loss_weights must be numeric")
        return np.asarray(array, dtype=np.float32)
    if array.dtype.kind not in "biu":
        raise TypeError(f"prepared field {name!r} must contain integers")
    if array.size:
        minimum = int(np.min(array))
        maximum = int(np.max(array))
        int32 = np.iinfo(np.int32)
        if minimum < int32.min or maximum > int32.max:
            raise ValueError(f"prepared field {name!r} contains a value outside int32 range")
    return np.asarray(array, dtype=np.int32)


def _validate_contiguous_segment_ids(segments: np.ndarray) -> None:
    """Require every positive segment ID to occupy one contiguous run per row."""

    for row_index, row in enumerate(segments):
        run_starts = np.empty(row.shape, dtype=np.bool_)
        run_starts[0] = True
        run_starts[1:] = row[1:] != row[:-1]
        positive_runs = row[run_starts]
        positive_runs = positive_runs[positive_runs > 0]
        if positive_runs.size == 0:
            continue
        segment_ids, counts = np.unique(positive_runs, return_counts=True)
        repeated = segment_ids[counts > 1]
        if repeated.size:
            raise ValueError(
                f"prepared row {row_index} reuses segment_id={int(repeated[0])} after its contiguous run has ended"
            )


def _validate_prepared_batch(
    columns: Mapping[str, Any],
    *,
    max_sequence_length: int,
    uses_assistant_loss_mask: bool,
    expected_rows: int | None = None,
    max_token_id: int | None = None,
    max_position_id: int | None = None,
) -> dict[str, np.ndarray]:
    if max_token_id is not None:
        _require_plain_int(max_token_id, name="prepared max_token_id")
    if max_position_id is not None:
        _require_plain_int(max_position_id, name="prepared max_position_id")
    names = _canonical_field_names(
        uses_assistant_loss_mask=uses_assistant_loss_mask,
    )
    actual = set(columns)
    required = set(names)
    if actual != required:
        raise ValueError(
            f"prepared dataset fields mismatch: missing={sorted(required - actual)}, extra={sorted(actual - required)}"
        )
    if expected_rows is None:
        try:
            rows = len(columns["input_ids"])
        except TypeError as exc:
            raise TypeError("prepared input_ids must be a sequence of rows") from exc
    else:
        rows = expected_rows
    arrays = {
        name: _as_batch_array(
            columns[name],
            name=name,
            rows=rows,
            max_sequence_length=max_sequence_length,
        )
        for name in names
    }
    attention = arrays["attention_mask"]
    segments = arrays["segment_ids"]
    positions = arrays["position_ids"]
    tokens = arrays["input_ids"]
    weights = arrays["loss_weights"]
    if tokens.size and (tokens < 0).any():
        raise ValueError("prepared input_ids must be non-negative")
    if max_token_id is not None and tokens.size and int(np.max(tokens)) > max_token_id:
        raise ValueError(f"prepared input_ids exceed manifest max_token_id={max_token_id}")
    if attention.size and not np.isin(attention, (0, 1)).all():
        raise ValueError("prepared attention_mask must contain only 0/1 values")
    if positions.size and (positions < 0).any():
        raise ValueError("prepared position_ids must be non-negative")
    if max_position_id is not None and positions.size and int(np.max(positions)) > max_position_id:
        raise ValueError(f"prepared position_ids exceed manifest max_position_id={max_position_id}")
    if segments.size and (segments < 0).any():
        raise ValueError("prepared segment_ids must be non-negative")
    if not np.array_equal(segments != 0, attention != 0):
        raise ValueError("prepared segment_ids must be positive exactly where attention_mask is one")
    if segments.size:
        _validate_contiguous_segment_ids(segments)
    if positions.size and (positions[attention == 0] != 0).any():
        raise ValueError("prepared position_ids must be zero outside the attention mask")
    if not np.isfinite(weights).all() or (weights < 0).any():
        raise ValueError("prepared loss_weights must be finite and non-negative")
    if weights.size and (weights[attention == 0] != 0).any():
        raise ValueError("prepared loss_weights must be zero outside the attention mask")
    if uses_assistant_loss_mask:
        assistant = arrays["assistant_masks"]
        if assistant.size and not np.isin(assistant, (0, 1)).all():
            raise ValueError("prepared assistant_masks must contain only 0/1 values")
        if assistant.size and (assistant[attention == 0] != 0).any():
            raise ValueError("prepared assistant_masks must be zero outside the attention mask")
    if rows:
        valid_target = (
            (attention[:, :-1] != 0)
            & (attention[:, 1:] != 0)
            & (segments[:, :-1] != 0)
            & (segments[:, :-1] == segments[:, 1:])
            & (weights[:, 1:] > 0)
        )
        invalid_rows = np.flatnonzero(~np.any(valid_target, axis=1))
        if invalid_rows.size:
            raise ValueError(
                f"prepared row {int(invalid_rows[0])} has no positive-weight causal target "
                "with a valid same-segment predecessor"
            )
    return {name: np.ascontiguousarray(arrays[name]) for name in names}


def canonical_prepared_dataset(
    columns: Mapping[str, Any],
    *,
    max_sequence_length: int,
    uses_assistant_loss_mask: bool = False,
) -> Dataset:
    """Construct an exact fixed-shape Hugging Face Dataset from canonical columns."""

    arrays = _validate_prepared_batch(
        columns,
        max_sequence_length=max_sequence_length,
        uses_assistant_loss_mask=uses_assistant_loss_mask,
    )
    return Dataset.from_dict(
        arrays,
        features=canonical_prepared_features(
            max_sequence_length,
            uses_assistant_loss_mask=uses_assistant_loss_mask,
        ),
    )


def validate_prepared_dataset(
    dataset: Dataset | IterableDataset,
    *,
    max_sequence_length: int,
    uses_assistant_loss_mask: bool = False,
    expected_rows: int | None = None,
    max_token_id: int | None = None,
    max_position_id: int | None = None,
    batch_size: int = 64,
    scan_records: bool = True,
) -> None:
    """Validate canonical features and, for materialized data, record semantics."""

    if type(batch_size) is not int or batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")
    if max_token_id is not None:
        _require_plain_int(max_token_id, name="prepared max_token_id")
    if max_position_id is not None:
        _require_plain_int(max_position_id, name="prepared max_position_id")
    expected_features = canonical_prepared_features(
        max_sequence_length,
        uses_assistant_loss_mask=uses_assistant_loss_mask,
    )
    actual_features = getattr(dataset, "features", None)
    if actual_features is not None:
        actual_dict = actual_features.to_dict() if hasattr(actual_features, "to_dict") else actual_features
        if actual_dict != expected_features.to_dict():
            raise ValueError("prepared dataset Features do not match the canonical manifest schema")
    expected_names = set(expected_features)
    column_names = getattr(dataset, "column_names", None)
    if column_names is not None and set(column_names) != expected_names:
        raise ValueError(
            "prepared dataset fields mismatch: "
            f"missing={sorted(expected_names - set(column_names))}, "
            f"extra={sorted(set(column_names) - expected_names)}"
        )
    if isinstance(dataset, IterableDataset):
        if expected_rows is not None and not scan_records:
            return
        if not scan_records:
            return
        raise TypeError("record scanning an IterableDataset would consume it; validate it while iterating")
    rows = len(dataset)
    if expected_rows is not None and rows != expected_rows:
        raise ValueError(f"prepared dataset has {rows} rows, expected {expected_rows}")
    if not scan_records:
        return
    rows_seen = 0
    for batch in dataset.iter(batch_size=batch_size):
        batch_rows = len(batch["input_ids"])
        _validate_prepared_batch(
            batch,
            max_sequence_length=max_sequence_length,
            uses_assistant_loss_mask=uses_assistant_loss_mask,
            expected_rows=batch_rows,
            max_token_id=max_token_id,
            max_position_id=max_position_id,
        )
        rows_seen += batch_rows
    if rows_seen != rows:
        raise ValueError(f"prepared validation visited {rows_seen} rows but expected {rows}")


def _empty_prepared_dataset(manifest: PreparedDatasetManifest) -> Dataset:
    return Dataset.from_dict(
        {name: [] for name in manifest.schema},
        features=canonical_prepared_features(
            manifest.max_sequence_length,
            uses_assistant_loss_mask=manifest.uses_assistant_loss_mask,
        ),
    )


def _read_parquet_files_in_memory(paths: Sequence[str]) -> Dataset:
    """Read selected Parquet files without creating a second Arrow disk cache."""

    import pyarrow as pa
    import pyarrow.parquet as pq

    if not paths:
        raise ValueError("at least one prepared Parquet path is required")
    tables = [pq.read_table(path, memory_map=True) for path in paths]
    table = tables[0] if len(tables) == 1 else pa.concat_tables(tables)
    return Dataset(table)


def load_prepared_dataset(
    name: str,
    manifest: PreparedDatasetManifest,
    *,
    token: str | None = None,
    revision: str | None = None,
    streaming: bool = False,
    process_index: int | None = None,
    process_count: int | None = None,
    cache_dir: str | Path | None = None,
    verify_checksums: bool = True,
) -> Dataset | IterableDataset:
    """Load a prepared split with explicit physical-file selection.

    Materialized loading downloads only the caller's contiguous file interval.
    Streaming deliberately selects every file so each process can reproduce the
    same global stream expected by the trainer's current placement contract.
    """

    if not isinstance(manifest, PreparedDatasetManifest):
        raise TypeError("manifest must be a PreparedDatasetManifest")
    candidate = Path(name).expanduser()
    if streaming:
        if candidate.exists():
            root = candidate if candidate.is_dir() else candidate.parent
            data_files = [str(root / shard.path) for shard in manifest.shards]
            if verify_checksums:
                for shard, local_file in zip(manifest.shards, data_files, strict=True):
                    actual = sha256_file(local_file)
                    if actual != shard.sha256:
                        raise RuntimeError(
                            f"prepared shard checksum mismatch for {shard.path!r}: "
                            f"expected {shard.sha256}, got {actual}"
                        )
        else:
            data_files = [
                hf_hub_url(
                    repo_id=name,
                    filename=shard.path,
                    repo_type="dataset",
                    revision=revision,
                )
                for shard in manifest.shards
            ]
        dataset = load_dataset(
            "parquet",
            data_files={manifest.split: data_files},
            split=manifest.split,
            features=canonical_prepared_features(
                manifest.max_sequence_length,
                uses_assistant_loss_mask=manifest.uses_assistant_loss_mask,
            ),
            token=token,
            streaming=True,
        )
        validate_prepared_dataset(
            dataset,
            max_sequence_length=manifest.max_sequence_length,
            uses_assistant_loss_mask=manifest.uses_assistant_loss_mask,
            expected_rows=manifest.total_rows,
            scan_records=False,
        )
        return dataset

    if process_index is None and process_count is None:
        process_index, process_count = 0, 1
    elif process_index is None or process_count is None:
        raise ValueError("process_index and process_count must be provided together")
    assigned = assigned_prepared_shards(
        manifest,
        process_index=process_index,
        process_count=process_count,
    )
    expected_rows = sum(shard.rows for shard in assigned)
    if not assigned:
        return _empty_prepared_dataset(manifest)

    if candidate.exists():
        root = candidate if candidate.is_dir() else candidate.parent
        local_files = [str(root / shard.path) for shard in assigned]
    else:
        local_files = [
            hf_hub_download(
                repo_id=name,
                filename=shard.path,
                repo_type="dataset",
                revision=revision,
                token=token,
                cache_dir=None if cache_dir is None else str(cache_dir),
            )
            for shard in assigned
        ]
    if verify_checksums:
        for shard, local_file in zip(assigned, local_files, strict=True):
            actual = sha256_file(local_file)
            if actual != shard.sha256:
                raise RuntimeError(
                    f"prepared shard checksum mismatch for {shard.path!r}: expected {shard.sha256}, got {actual}"
                )
    dataset = _read_parquet_files_in_memory(local_files)
    validate_prepared_dataset(
        dataset,
        max_sequence_length=manifest.max_sequence_length,
        uses_assistant_loss_mask=manifest.uses_assistant_loss_mask,
        expected_rows=expected_rows,
        max_token_id=manifest.max_token_id,
        max_position_id=manifest.max_position_id,
    )
    return dataset


def _canonical_prepared_record(
    record: Mapping[str, Any],
    *,
    max_sequence_length: int,
    uses_assistant_loss_mask: bool,
    max_token_id: int | None = None,
    max_position_id: int | None = None,
) -> dict[str, list[int] | list[float]]:
    batch = {name: [value] for name, value in record.items()}
    arrays = _validate_prepared_batch(
        batch,
        max_sequence_length=max_sequence_length,
        uses_assistant_loss_mask=uses_assistant_loss_mask,
        expected_rows=1,
        max_token_id=max_token_id,
        max_position_id=max_position_id,
    )
    return {name: arrays[name][0].tolist() for name in arrays}


def iter_prepared_streaming_batches(
    source_factory: StreamingSourceFactory,
    num_epochs: int | None,
    expected_rows_per_epoch: int,
    max_sequence_length: int,
    retry_initial_delay: float = 1.0,
    retry_max_delay: float = 60.0,
    window_size: int = 32,
    *,
    uses_assistant_loss_mask: bool = False,
    max_token_id: int | None = None,
    max_position_id: int | None = None,
) -> Iterator[StreamingBatchEnvelope]:
    """Yield validated prepared windows without tokenization or packing."""

    if not callable(source_factory):
        raise TypeError("source_factory must be callable")
    _require_plain_int(
        expected_rows_per_epoch,
        name="expected_rows_per_epoch",
        minimum=1,
    )
    _require_plain_int(max_sequence_length, name="max_sequence_length", minimum=1)
    if max_token_id is not None:
        _require_plain_int(max_token_id, name="prepared max_token_id")
    if max_position_id is not None:
        _require_plain_int(max_position_id, name="prepared max_position_id")
    if type(window_size) is not int or window_size <= 0:
        raise ValueError("window_size must be a positive integer")
    if num_epochs is not None and (type(num_epochs) is not int or num_epochs <= 0):
        raise ValueError("num_epochs must be a positive integer or None")
    epochs: Iterable[int] = itertools.count() if num_epochs is None else range(num_epochs)
    digest_fields = _canonical_field_names(
        uses_assistant_loss_mask=uses_assistant_loss_mask,
    )
    for epoch in epochs:
        source = iter_retrying_records(
            source_factory,
            epoch,
            initial_delay=retry_initial_delay,
            max_delay=retry_max_delay,
            operation="streaming prepared Hugging Face dataset",
            digest_fields=digest_fields,
        )
        iterator = iter(source)
        rows_seen = 0
        while window := list(islice(iterator, window_size)):
            normalized = tuple(
                _canonical_prepared_record(
                    record,
                    max_sequence_length=max_sequence_length,
                    uses_assistant_loss_mask=uses_assistant_loss_mask,
                    max_token_id=max_token_id,
                    max_position_id=max_position_id,
                )
                for record in window
            )
            rows_seen += len(normalized)
            if rows_seen > expected_rows_per_epoch:
                raise ValueError(
                    f"prepared streaming epoch {epoch} produced more than the manifest's {expected_rows_per_epoch} rows"
                )
            yield StreamingBatchEnvelope(
                records=normalized,
                source_examples=len(normalized),
                record_source_examples=(1,) * len(normalized),
                epoch=epoch,
            )
        if rows_seen != expected_rows_per_epoch:
            raise ValueError(
                f"prepared streaming epoch {epoch} produced {rows_seen} rows, "
                f"expected {expected_rows_per_epoch} from the manifest"
            )
        yield StreamingBatchEnvelope(
            records=(),
            source_examples=0,
            record_source_examples=(),
            epoch=epoch,
            is_epoch_end=True,
        )


__all__ = [
    "PREPARED_DATASET_FORMAT",
    "PREPARED_MANIFEST_FILENAME",
    "PREPARED_REQUIRED_FIELDS",
    "PreparedDatasetManifest",
    "PreparedShard",
    "assigned_prepared_shards",
    "canonical_prepared_dataset",
    "canonical_prepared_features",
    "iter_prepared_streaming_batches",
    "load_prepared_dataset",
    "load_prepared_manifest",
    "read_prepared_manifest",
    "resolve_prepared_revision",
    "sha256_file",
    "validate_prepared_dataset",
    "write_prepared_manifest",
]
