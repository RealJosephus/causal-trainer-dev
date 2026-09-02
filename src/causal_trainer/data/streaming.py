"""Bounded, restartable preprocessing and packing for streaming datasets."""

from __future__ import annotations

import hashlib
import itertools
import json
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from itertools import islice
from typing import Any, cast

from ..hf_retry import is_retryable_hf_error, retry_hf_call
from .packing import (
    PackedExample,
    _materialize_bin,
    _normalize_example,
    _validate_consistent_fields,
    best_fit_decreasing,
    pad_examples,
)
from .preprocessing import EndPromptSettings, preprocess_records


SourceRecord = Mapping[str, Any]
StreamingSourceFactory = Callable[[int], Iterable[SourceRecord]]
_END_OF_STREAM = object()


class _ShortReplayEof(ConnectionError):
    """A retry replay ended before reaching the already-delivered prefix."""


@dataclass(frozen=True, slots=True)
class StreamingDatasetMetadata:
    """Optional source-row count estimate from one dataset split's metadata."""

    num_examples: int | None

    def __post_init__(self) -> None:
        if self.num_examples is not None and (
            type(self.num_examples) is not int or self.num_examples < 0
        ):
            raise ValueError("num_examples must be a non-negative integer or None")

    @property
    def is_known(self) -> bool:
        return self.num_examples is not None

    def total_examples(self, num_epochs: int | None) -> int | None:
        if self.num_examples is None or num_epochs is None:
            return None
        if type(num_epochs) is not int or num_epochs < 0:
            raise ValueError("num_epochs must be a non-negative integer or None")
        return self.num_examples * num_epochs


@dataclass(frozen=True, slots=True)
class StreamingBatchEnvelope:
    """One bounded preprocessing window or an explicit epoch-end marker.

    ``source_examples`` counts raw rows consumed from the source. For a
    non-empty window, ``record_source_examples`` assigns those rows to prepared
    records, allowing a later optimizer-batch grouper to commit progress only
    when the corresponding packed rows are trained. Filtered rows are charged
    to the final prepared record. A fully filtered window has no per-record
    assignments but retains its non-zero aggregate count.
    """

    records: tuple[PackedExample, ...]
    source_examples: int
    record_source_examples: tuple[int, ...]
    epoch: int
    is_epoch_end: bool = False

    def __post_init__(self) -> None:
        if type(self.source_examples) is not int or self.source_examples < 0:
            raise ValueError("source_examples must be a non-negative integer")
        if type(self.epoch) is not int or self.epoch < 0:
            raise ValueError("epoch must be a non-negative integer")
        if len(self.record_source_examples) != len(self.records):
            raise ValueError("record_source_examples must align with records")
        if any(type(value) is not int or value < 0 for value in self.record_source_examples):
            raise ValueError("record_source_examples must contain non-negative integers")
        if self.records and sum(self.record_source_examples) != self.source_examples:
            raise ValueError("per-record source counts must sum to source_examples")
        if self.is_epoch_end and (
            self.records or self.source_examples or self.record_source_examples
        ):
            raise ValueError("an epoch-end marker cannot contain records or source examples")


def _split_num_examples(split_info: Any) -> int | None:
    if isinstance(split_info, Mapping):
        value = split_info.get("num_examples")
    else:
        value = getattr(split_info, "num_examples", None)
    return value if type(value) is int and value >= 0 else None


def streaming_dataset_metadata(
    dataset: Any,
    split: str | None = None,
) -> StreamingDatasetMetadata:
    """Read a row-count estimate from a Dataset-like object's split metadata.

    Missing, partial, or malformed metadata is represented by ``None`` rather
    than forcing a scan of the stream.
    """

    info = getattr(dataset, "info", None)
    splits = getattr(info, "splits", None)
    if splits is None:
        return StreamingDatasetMetadata(None)

    resolved_split = split
    if resolved_split is None:
        dataset_split = getattr(dataset, "split", None)
        if dataset_split is not None:
            resolved_split = str(dataset_split)
    if resolved_split is None:
        return StreamingDatasetMetadata(None)

    try:
        split_info = splits.get(resolved_split)
    except AttributeError:
        try:
            split_info = splits[resolved_split]
        except (KeyError, TypeError):
            return StreamingDatasetMetadata(None)
    return StreamingDatasetMetadata(_split_num_examples(split_info))


def _update_prefix_digest(digest: Any, record: SourceRecord) -> None:
    """Add one source row to a deterministic, bounded-memory prefix digest."""

    def encode_unknown(value: Any) -> Mapping[str, str]:
        value_type = type(value)
        return {
            "__causal_trainer_type__": (
                f"{value_type.__module__}.{value_type.__qualname__}"
            ),
            "__causal_trainer_repr__": repr(value),
        }

    encoded = json.dumps(
        record,
        default=encode_unknown,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    digest.update(len(encoded).to_bytes(8, byteorder="big", signed=False))
    digest.update(encoded)


def _record_for_digest(
    record: SourceRecord,
    digest_fields: tuple[str, ...] | None,
) -> SourceRecord:
    if digest_fields is None:
        return record
    return {
        field: record[field]
        for field in digest_fields
        if field in record
    }


def _resume_and_read_one(
    source_factory: StreamingSourceFactory,
    *,
    epoch: int,
    delivered: int,
    expected_prefix_digest: bytes,
    digest_fields: tuple[str, ...] | None,
) -> tuple[Iterator[SourceRecord], SourceRecord | object]:
    source = source_factory(epoch)
    iterator = iter(source)
    reconstructed_digest = hashlib.sha256()
    for skipped in range(delivered):
        try:
            record = next(iterator)
        except StopIteration as exc:
            message = (
                "the reconstructed streaming source ended after "
                f"{skipped} rows while {delivered} previously yielded rows had to be skipped"
            )
            raise _ShortReplayEof(
                message + "; reopening the same epoch indefinitely"
            ) from exc
        _update_prefix_digest(
            reconstructed_digest,
            _record_for_digest(record, digest_fields),
        )
    if reconstructed_digest.digest() != expected_prefix_digest:
        raise RuntimeError(
            "the reconstructed streaming source prefix changed during retry; "
            "pin dataset_revision to an immutable Hugging Face commit and ensure "
            "all hosts use the same deterministic source"
        )
    try:
        first = next(iterator)
    except StopIteration:
        return iterator, _END_OF_STREAM
    return iterator, first


def iter_retrying_records(
    source_factory: StreamingSourceFactory,
    epoch: int,
    *,
    initial_delay: float = 1.0,
    max_delay: float = 60.0,
    operation: str = "streaming Hugging Face dataset",
    digest_fields: Sequence[str] | None = None,
) -> Iterator[SourceRecord]:
    """Yield a deterministic full source while retrying network failures forever.

    If iteration itself disconnects, a fresh source for the same ``epoch`` is
    opened and exactly the number of already-yielded raw rows is skipped. This
    intentionally avoids relying on a library iterator surviving an exception.
    A natural EOF always completes the epoch; metadata never controls stream
    termination. Non-network exceptions fail immediately.
    """

    if not callable(source_factory):
        raise TypeError("source_factory must be callable")
    if type(epoch) is not int or epoch < 0:
        raise ValueError("epoch must be a non-negative integer")
    resolved_digest_fields = (
        tuple(str(field) for field in digest_fields)
        if digest_fields is not None
        else None
    )
    if resolved_digest_fields is not None and not resolved_digest_fields:
        raise ValueError("digest_fields cannot be empty")

    delivered = 0
    delivered_digest = hashlib.sha256()
    pending_error: Exception | None = None
    while True:
        delivered_snapshot = delivered
        prefix_digest_snapshot = delivered_digest.digest()

        def reopen(
            delivered_at_open: int = delivered_snapshot,
            expected_prefix_digest: bytes = prefix_digest_snapshot,
        ) -> tuple[Iterator[SourceRecord], SourceRecord | object]:
            nonlocal pending_error
            if pending_error is not None:
                error = pending_error
                pending_error = None
                raise error
            reopened = _resume_and_read_one(
                source_factory,
                epoch=epoch,
                delivered=delivered_at_open,
                expected_prefix_digest=expected_prefix_digest,
                digest_fields=resolved_digest_fields,
            )
            return reopened

        iterator, first = retry_hf_call(
            reopen,
            initial_delay=initial_delay,
            max_delay=max_delay,
            operation=f"{operation} epoch {epoch}",
        )
        if first is _END_OF_STREAM:
            return

        delivered += 1
        first_record = cast(SourceRecord, first)
        _update_prefix_digest(
            delivered_digest,
            _record_for_digest(first_record, resolved_digest_fields),
        )
        yield first_record
        while True:
            try:
                record = next(iterator)
            except StopIteration:
                return
            except Exception as exc:
                if not is_retryable_hf_error(exc):
                    raise
                pending_error = exc
                break
            delivered += 1
            _update_prefix_digest(
                delivered_digest,
                _record_for_digest(record, resolved_digest_fields),
            )
            yield record


def _batched_records(
    records: Iterable[SourceRecord],
    batch_size: int,
) -> Iterator[list[SourceRecord]]:
    iterator = iter(records)
    while batch := list(islice(iterator, batch_size)):
        yield batch


def _pack_window(
    examples: Sequence[Mapping[str, Any]],
    *,
    max_length: int,
    pad_token_id: int,
) -> tuple[list[PackedExample], list[int]]:
    if not examples:
        return [], []
    normalized = [_normalize_example(example, max_length) for example in examples]
    fields = _validate_consistent_fields(normalized)
    source_fields = tuple(field for field in fields if field != "segment_ids")
    bins = best_fit_decreasing(normalized, max_length=max_length)
    packed = [
        _materialize_bin(
            packed_bin,
            fields=source_fields,
            max_length=max_length,
            pad_token_id=pad_token_id,
            padding_side="left",
        )
        for packed_bin in bins
    ]
    return packed, [len(packed_bin) for packed_bin in bins]


def _ensure_loss_weights(
    record: PackedExample,
    *,
    assistant_only_loss: bool,
) -> PackedExample:
    output = {name: list(values) for name, values in record.items()}
    if "loss_weights" not in output:
        source_name = "assistant_masks" if assistant_only_loss else "attention_mask"
        if source_name not in output:
            raise ValueError(f"prepared streaming record is missing {source_name!r}")
        output["loss_weights"] = [float(value) for value in output[source_name]]
    else:
        output["loss_weights"] = [float(value) for value in output["loss_weights"]]
    if len(output["loss_weights"]) != len(output["input_ids"]):
        raise ValueError("prepared streaming loss_weights are not aligned with input_ids")
    if any(value < 0 for value in output["loss_weights"]):
        raise ValueError("prepared streaming loss_weights must be non-negative")
    return output


def _assign_source_counts(
    prepared_counts: Sequence[int],
    *,
    source_examples: int,
    trainable_examples: int,
) -> tuple[int, ...]:
    if not prepared_counts:
        return ()
    output = [int(value) for value in prepared_counts]
    filtered = source_examples - trainable_examples
    if filtered < 0 or sum(output) != trainable_examples:
        raise RuntimeError("streaming source provenance is inconsistent with prepared records")
    output[-1] += filtered
    return tuple(output)


def _epoch_indices(num_epochs: int | None) -> Iterable[int]:
    if num_epochs is None:
        return itertools.count()
    if type(num_epochs) is not int or num_epochs <= 0:
        raise ValueError("num_epochs must be a positive integer or None")
    return range(num_epochs)


def iter_streaming_batches(
    source_factory: StreamingSourceFactory,
    tokenizer: Any,
    *,
    num_epochs: int | None,
    dataset_text_field: str,
    max_sequence_length: int,
    pad_token_id: int,
    assistant_only_loss: bool = False,
    last_assistant_only_loss: bool = False,
    endprompt: EndPromptSettings | None = None,
    packing: bool = False,
    packing_batch_size: int = 1000,
    retry_initial_delay: float = 1.0,
    retry_max_delay: float = 60.0,
) -> Iterator[StreamingBatchEnvelope]:
    """Lazily tokenize and finalize a deterministic full stream on every host.

    The source is intentionally not process-sharded. Giving every host the same
    factory and tokenizer yields identical global record streams, preserving
    the existing ``host_global_to_array`` multi-controller contract.

    Each normal envelope materializes at most ``packing_batch_size`` raw rows.
    A separate empty marker terminates every epoch so a downstream grouper can
    flush its final partial optimizer batch without reading into the next
    epoch. ``num_epochs=None`` repeats complete epochs until the caller stops.
    """

    if not callable(source_factory):
        raise TypeError("source_factory must be callable")
    if max_sequence_length <= 0:
        raise ValueError("max_sequence_length must be positive")
    if packing_batch_size <= 0:
        raise ValueError("packing_batch_size must be positive")
    if assistant_only_loss and last_assistant_only_loss:
        raise ValueError("assistant_only_loss and last_assistant_only_loss are mutually exclusive")
    use_assistant_mask = assistant_only_loss or last_assistant_only_loss

    for epoch in _epoch_indices(num_epochs):
        source = iter_retrying_records(
            source_factory,
            epoch,
            initial_delay=retry_initial_delay,
            max_delay=retry_max_delay,
            digest_fields=(
                dataset_text_field,
                "tools",
                "chat_template_kwargs",
            ),
        )
        source_index = 0
        for raw_window in _batched_records(source, packing_batch_size):
            source_examples = len(raw_window)
            tokenized = preprocess_records(
                raw_window,
                tokenizer,
                dataset_text_field=dataset_text_field,
                max_sequence_length=max_sequence_length,
                assistant_only_loss=assistant_only_loss,
                last_assistant_only_loss=last_assistant_only_loss,
                endprompt=endprompt,
                example_index_offset=source_index,
                allow_empty=True,
            )
            source_index += source_examples
            if packing:
                prepared, prepared_counts = _pack_window(
                    tokenized,
                    max_length=max_sequence_length,
                    pad_token_id=pad_token_id,
                )
            else:
                prepared = pad_examples(
                    tokenized,
                    max_length=max_sequence_length,
                    pad_token_id=pad_token_id,
                    padding_side="left",
                )
                prepared_counts = [1] * len(prepared)

            normalized = tuple(
                _ensure_loss_weights(
                    record,
                    assistant_only_loss=use_assistant_mask,
                )
                for record in prepared
            )
            yield StreamingBatchEnvelope(
                records=normalized,
                source_examples=source_examples,
                record_source_examples=_assign_source_counts(
                    prepared_counts,
                    source_examples=source_examples,
                    trainable_examples=len(tokenized),
                ),
                epoch=epoch,
            )
        yield StreamingBatchEnvelope(
            records=(),
            source_examples=0,
            record_source_examples=(),
            epoch=epoch,
            is_epoch_end=True,
        )


__all__ = [
    "SourceRecord",
    "StreamingBatchEnvelope",
    "StreamingDatasetMetadata",
    "StreamingSourceFactory",
    "iter_retrying_records",
    "iter_streaming_batches",
    "streaming_dataset_metadata",
]
