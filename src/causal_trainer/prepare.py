"""Standalone preprocessing command for prepared Hugging Face datasets.

This module intentionally depends only on the data stack.  In particular, it
does not import JAX, the training runner, or checkpointing code, so preparing a
dataset cannot initialize an accelerator backend.
"""

from __future__ import annotations

import argparse
import hashlib
import math
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from huggingface_hub import HfApi, snapshot_download

from .data.pipeline import (
    load_training_split,
    packed_dataset_to_arrays,
    prepare_training_dataset,
    select_process_shard,
)
from .data.prepared import (
    PREPARED_MANIFEST_FILENAME,
    PreparedDatasetManifest,
    PreparedShard,
    canonical_prepared_dataset,
    sha256_file,
)
from .data.preprocessing import EndPromptSettings
from .data.tokenizer import TrainingTokenizer

_TOKENIZER_ALLOW_PATTERNS = (
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "chat_template.json",
    "chat_template.jinja",
    "chat_templates/*.jinja",
)
_TOKENIZER_LOCAL_GLOBS = (
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "chat_template.json",
    "chat_template.jinja",
    "chat_templates/*.jinja",
)


def parse_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"expected a boolean value, got {value!r}")


def _flags(name: str) -> tuple[str, ...]:
    hyphenated = name.replace("_", "-")
    if hyphenated == name:
        return (f"--{name}",)
    return (f"--{hyphenated}", f"--{name}")


def _add(parser: argparse.ArgumentParser, name: str, **kwargs: Any) -> None:
    parser.add_argument(*_flags(name), dest=name, **kwargs)


def _add_bool(
    parser: argparse.ArgumentParser,
    name: str,
    default: bool,
    help: str,
) -> None:
    _add(parser, name, nargs="?", const=True, default=default, type=parse_bool, help=help)
    parser.add_argument(
        f"--no-{name.replace('_', '-')}",
        dest=name,
        action="store_false",
    )


@dataclass(frozen=True, slots=True)
class PrepareArguments:
    dataset_name: str
    dataset_split: str
    dataset_text_field: str
    dataset_config_name: str | None
    dataset_revision: str | None
    tokenizer_name_or_path: str
    tokenizer_revision: str | None
    token: str | None
    output_dir: Path
    num_shards: int
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
    hub_repo_id: str | None
    hub_revision: str | None
    private: bool

    @property
    def uses_assistant_loss_mask(self) -> bool:
        return self.assistant_only_loss or self.last_assistant_only_loss

    @property
    def effective_endprompt_logical_length(self) -> int:
        return self.max_sequence_length if self.endprompt_logical_length is None else self.endprompt_logical_length

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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="causal-prepare",
        description=("Tokenize and optionally pack a dataset into process-local Parquet shards for causal-trainer."),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    _add(
        parser,
        "dataset_name",
        required=True,
        help="Local JSON/JSONL/Parquet/CSV/TSV file or Hugging Face dataset ID.",
    )
    _add(parser, "dataset_split", default="train")
    _add(
        parser,
        "dataset_text_field",
        default="text",
        help=(
            "Use 'text' (or another string column) for raw text, or 'messages' "
            "for role/content records rendered through the tokenizer chat template."
        ),
    )
    _add(parser, "dataset_config_name", default=None)
    _add(parser, "dataset_revision", default=None)
    _add(
        parser,
        "tokenizer_name_or_path",
        required=True,
        help="Local tokenizer directory or Hugging Face model repository ID.",
    )
    _add(parser, "tokenizer_revision", default=None)
    _add(
        parser,
        "token",
        default=None,
        help="Hugging Face token; prefer the HF_TOKEN environment variable.",
    )
    _add(parser, "output_dir", required=True, type=Path)
    _add(
        parser,
        "num_shards",
        type=int,
        default=1,
        help="Number of independent physical Parquet shards to write.",
    )
    _add(parser, "max_sequence_length", type=int, default=4096)
    _add_bool(parser, "packing", False, "Pack records into fixed-length rows.")
    _add(
        parser,
        "packing_batch_size",
        type=int,
        default=1000,
        help="Packing window used independently inside each output shard.",
    )
    _add_bool(
        parser,
        "assistant_only_loss",
        False,
        "Supervise all chat-template assistant generation spans.",
    )
    _add_bool(
        parser,
        "last_assistant_only_loss",
        False,
        "Supervise only the final assistant generation span in each record.",
    )
    _add_bool(
        parser,
        "endprompt_enable",
        False,
        "Append a terminal EndPrompt during preprocessing.",
    )
    _add(parser, "endprompt_logical_length", type=int, default=None)
    _add(parser, "endprompt_logical_length_min", type=int, default=None)
    _add(
        parser,
        "endprompt_prompts",
        default="This is the end of text, please pay attention here",
        help="Separate multiple terminal prompts with '||'.",
    )
    _add(parser, "endprompt_prompt_loss_weight", type=float, default=0.1)
    _add(parser, "endprompt_context_loss_weight", type=float, default=1.0)
    _add(parser, "preprocessing_num_workers", type=int, default=None)
    _add(
        parser,
        "hub_repo_id",
        default=None,
        help="Optional Hugging Face dataset repository to receive the output folder.",
    )
    _add(parser, "hub_revision", default=None)
    _add_bool(parser, "private", False, "Create the destination dataset repository as private.")
    return parser


def parse_args(argv: list[str] | None = None) -> PrepareArguments:
    args = PrepareArguments(**vars(build_parser().parse_args(argv)))
    if args.num_shards <= 0:
        raise ValueError("num_shards must be positive")
    if args.max_sequence_length <= 1:
        raise ValueError("max_sequence_length must be greater than one")
    if args.packing_batch_size <= 0:
        raise ValueError("packing_batch_size must be positive")
    if args.preprocessing_num_workers is not None and args.preprocessing_num_workers <= 0:
        raise ValueError("preprocessing_num_workers must be positive when provided")
    if args.assistant_only_loss and args.last_assistant_only_loss:
        raise ValueError("assistant_only_loss and last_assistant_only_loss are mutually exclusive")
    if args.uses_assistant_loss_mask and args.dataset_text_field != "messages":
        option = (
            "last_assistant_only_loss"
            if args.last_assistant_only_loss
            else "assistant_only_loss"
        )
        raise ValueError(f"{option} requires --dataset_text_field messages")
    if args.hub_revision is not None and args.hub_repo_id is None:
        raise ValueError("hub_revision requires hub_repo_id")
    if args.private and args.hub_repo_id is None:
        raise ValueError("private requires hub_repo_id")
    if args.endprompt_enable:
        logical_min = args.effective_endprompt_logical_length_min
        logical_max = args.effective_endprompt_logical_length
        if logical_min < args.max_sequence_length or logical_max < args.max_sequence_length:
            raise ValueError("EndPrompt logical lengths must be at least max_sequence_length")
        if logical_min > logical_max:
            raise ValueError("endprompt_logical_length_min cannot exceed endprompt_logical_length")
        if not args.endprompt_prompt_texts:
            raise ValueError("endprompt_prompts must contain at least one non-empty prompt")
        weights = (
            args.endprompt_context_loss_weight,
            args.endprompt_prompt_loss_weight,
        )
        if any(not math.isfinite(weight) or weight < 0.0 for weight in weights):
            raise ValueError("EndPrompt loss weights must be finite and non-negative")
        if not any(weight > 0.0 for weight in weights):
            raise ValueError("at least one EndPrompt loss weight must be positive")
    return args


def _endprompt_settings(args: PrepareArguments) -> EndPromptSettings | None:
    if not args.endprompt_enable:
        return None
    return EndPromptSettings(
        logical_length_min=args.effective_endprompt_logical_length_min,
        logical_length_max=args.effective_endprompt_logical_length,
        prompts=args.endprompt_prompt_texts,
        prompt_loss_weight=args.endprompt_prompt_loss_weight,
        context_loss_weight=args.endprompt_context_loss_weight,
    )


def _resolve_tokenizer_directory(args: PrepareArguments) -> Path:
    local = Path(args.tokenizer_name_or_path).expanduser()
    if local.is_dir():
        return local.resolve()
    if local.exists():
        raise ValueError(f"tokenizer_name_or_path must be a directory, got {local}")
    downloaded = snapshot_download(
        args.tokenizer_name_or_path,
        repo_type="model",
        revision=args.tokenizer_revision,
        token=args.token,
        allow_patterns=list(_TOKENIZER_ALLOW_PATTERNS),
    )
    return Path(downloaded)


def _tokenizer_files(directory: Path) -> tuple[Path, ...]:
    files: dict[str, Path] = {}
    for pattern in _TOKENIZER_LOCAL_GLOBS:
        for path in directory.glob(pattern):
            if not path.is_file():
                continue
            try:
                relative_path = path.relative_to(directory)
            except ValueError as error:
                raise ValueError(f"tokenizer file is outside its snapshot directory: {path}") from error
            relative = relative_path.as_posix()
            if (
                relative_path.is_absolute()
                or not relative_path.parts
                or any(part in {"", ".", ".."} for part in relative_path.parts)
                or "\\" in relative
            ):
                raise ValueError(f"unsafe tokenizer file path: {relative!r}")
            files[relative] = path
    if "tokenizer.json" not in files:
        raise FileNotFoundError(f"missing {directory / 'tokenizer.json'}")
    return tuple(files[name] for name in sorted(files))


def _tokenizer_digest(directory: Path) -> tuple[str, tuple[str, ...]]:
    digest = hashlib.sha256()
    names: list[str] = []
    for path in _tokenizer_files(directory):
        relative = path.relative_to(directory).as_posix()
        names.append(relative)
        encoded = relative.encode("utf-8")
        digest.update(len(encoded).to_bytes(4, "little", signed=False))
        digest.update(encoded)
        with path.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
    return digest.hexdigest(), tuple(names)


def _ensure_empty_output_directory(output_dir: Path) -> None:
    if output_dir.exists() and not output_dir.is_dir():
        raise ValueError(f"output_dir exists and is not a directory: {output_dir}")
    if output_dir.is_dir() and any(output_dir.iterdir()):
        raise ValueError(f"output_dir must be empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)


def _source_metadata(dataset: Any, args: PrepareArguments) -> dict[str, Any]:
    return {
        "name": args.dataset_name,
        "config_name": args.dataset_config_name,
        "revision": args.dataset_revision,
        "split": args.dataset_split,
        "rows": len(dataset),
        "fingerprint": getattr(dataset, "_fingerprint", None),
    }


def _tokenizer_metadata(
    tokenizer: TrainingTokenizer,
    tokenizer_directory: Path,
    args: PrepareArguments,
) -> dict[str, Any]:
    digest, files = _tokenizer_digest(tokenizer_directory)
    return {
        "name_or_path": args.tokenizer_name_or_path,
        "revision": args.tokenizer_revision,
        "sha256": digest,
        "files": list(files),
        "vocab_size": len(tokenizer),
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }


def _preprocessing_metadata(args: PrepareArguments) -> dict[str, Any]:
    endprompt = _endprompt_settings(args)
    return {
        "dataset_text_field": args.dataset_text_field,
        "packing": args.packing,
        "packing_batch_size": args.packing_batch_size,
        "assistant_only_loss": args.assistant_only_loss,
        "last_assistant_only_loss": args.last_assistant_only_loss,
        "preprocessing_num_workers": args.preprocessing_num_workers,
        "num_shards": args.num_shards,
        "endprompt": (
            None
            if endprompt is None
            else {
                "logical_length_min": endprompt.logical_length_min,
                "logical_length_max": endprompt.logical_length_max,
                "prompts": list(endprompt.prompts),
                "prompt_loss_weight": endprompt.prompt_loss_weight,
                "context_loss_weight": endprompt.context_loss_weight,
            }
        ),
    }


def _upload_output(args: PrepareArguments) -> None:
    if args.hub_repo_id is None:
        return
    api = HfApi()
    api.create_repo(
        args.hub_repo_id,
        token=args.token,
        private=args.private,
        repo_type="dataset",
        exist_ok=True,
    )
    if args.hub_revision is not None:
        api.create_branch(
            repo_id=args.hub_repo_id,
            branch=args.hub_revision,
            token=args.token,
            repo_type="dataset",
            exist_ok=True,
        )
    api.upload_folder(
        repo_id=args.hub_repo_id,
        folder_path=str(args.output_dir),
        path_in_repo="",
        token=args.token,
        repo_type="dataset",
        revision=args.hub_revision,
        commit_message="Upload causal-trainer prepared dataset",
    )


def prepare_dataset(args: PrepareArguments) -> PreparedDatasetManifest:
    """Prepare all future process shards sequentially and optionally upload them."""

    _ensure_empty_output_directory(args.output_dir)
    tokenizer_directory = _resolve_tokenizer_directory(args)
    tokenizer = TrainingTokenizer.from_directory(tokenizer_directory)
    source_dataset = load_training_split(
        args.dataset_name,
        args.dataset_split,
        token=args.token,
        config_name=args.dataset_config_name,
        revision=args.dataset_revision,
        streaming=False,
    )
    source = _source_metadata(source_dataset, args)
    tokenizer_info = _tokenizer_metadata(tokenizer, tokenizer_directory, args)
    endprompt = _endprompt_settings(args)
    data_dir = args.output_dir / "data"
    manifest_path = args.output_dir / PREPARED_MANIFEST_FILENAME
    temporary_manifest_path = manifest_path.with_suffix(f"{manifest_path.suffix}.incomplete")
    data_dir.mkdir()
    shards: list[PreparedShard] = []
    max_token_id = -1
    max_position_id = -1
    try:
        for shard_index in range(args.num_shards):
            source_shard, source_start = select_process_shard(
                source_dataset,
                process_index=shard_index,
                process_count=args.num_shards,
            )
            prepared = prepare_training_dataset(
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
                allow_empty=True,
            )
            columns = packed_dataset_to_arrays(
                prepared,
                max_sequence_length=args.max_sequence_length,
                assistant_only_loss=args.assistant_only_loss,
                last_assistant_only_loss=args.last_assistant_only_loss,
            )
            if columns["input_ids"].size:
                max_token_id = max(max_token_id, int(columns["input_ids"].max()))
                max_position_id = max(
                    max_position_id,
                    int(columns["position_ids"].max()),
                )
            canonical = canonical_prepared_dataset(
                columns,
                max_sequence_length=args.max_sequence_length,
                uses_assistant_loss_mask=args.uses_assistant_loss_mask,
            )
            if len(canonical) == 0:
                raise ValueError(
                    f"prepared shard {shard_index} is empty; reduce num_shards or "
                    "remove source rows that are filtered by the loss settings"
                )
            relative_path = Path("data") / (f"shard-{shard_index:05d}-of-{args.num_shards:05d}.parquet")
            output_path = args.output_dir / relative_path
            canonical.to_parquet(str(output_path))
            shards.append(
                PreparedShard(
                    path=relative_path.as_posix(),
                    rows=len(canonical),
                    sha256=sha256_file(output_path),
                )
            )
            del canonical, columns, prepared, source_shard

        if not any(shard.rows for shard in shards):
            raise ValueError("preprocessing produced no trainable records across all shards")
        manifest = PreparedDatasetManifest(
            max_sequence_length=args.max_sequence_length,
            split=args.dataset_split,
            shards=tuple(shards),
            tokenizer=tokenizer_info,
            source=source,
            preprocessing=_preprocessing_metadata(args),
            uses_assistant_loss_mask=args.uses_assistant_loss_mask,
            max_token_id=max_token_id,
            max_position_id=max_position_id,
        )
        serialized = manifest.to_json()
        temporary_manifest_path.write_text(
            serialized if serialized.endswith("\n") else f"{serialized}\n",
            encoding="utf-8",
        )
        temporary_manifest_path.replace(manifest_path)
    except Exception:
        temporary_manifest_path.unlink(missing_ok=True)
        manifest_path.unlink(missing_ok=True)
        shutil.rmtree(data_dir, ignore_errors=True)
        raise

    _upload_output(args)
    return manifest


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    manifest = prepare_dataset(args)
    destination = f" and uploaded to {args.hub_repo_id}" if args.hub_repo_id is not None else ""
    print(f"Prepared {manifest.total_rows} rows in {len(manifest.shards)} shards at {args.output_dir}{destination}.")


if __name__ == "__main__":
    main()


__all__ = (
    "PrepareArguments",
    "build_parser",
    "main",
    "parse_args",
    "prepare_dataset",
)
