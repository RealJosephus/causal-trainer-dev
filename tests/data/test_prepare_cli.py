import ast
from pathlib import Path

import pytest
from datasets import Dataset

import causal_trainer.prepare as prepare_cli
from causal_trainer.data.prepared import (
    PREPARED_MANIFEST_FILENAME,
    PreparedDatasetManifest,
)


class TinyTokenizer:
    pad_token_id = 0
    eos_token_id = None

    def __len__(self) -> int:
        return 128

    def __call__(self, text, **kwargs):
        del kwargs
        value = int(text)
        return {
            "input_ids": [value, value + 10],
            "attention_mask": [1, 1],
        }


def _write_tokenizer_stub(path: Path) -> Path:
    path.mkdir()
    (path / "tokenizer.json").write_text('{"stub": true}\n', encoding="utf-8")
    (path / "tokenizer_config.json").write_text(
        '{"pad_token": "<pad>"}\n',
        encoding="utf-8",
    )
    return path


def _arguments(tmp_path: Path, *extra: str) -> prepare_cli.PrepareArguments:
    return prepare_cli.parse_args(
        [
            "--dataset-name",
            "local-source",
            "--tokenizer-name-or-path",
            str(tmp_path / "tokenizer"),
            "--output-dir",
            str(tmp_path / "prepared"),
            "--max-sequence-length",
            "4",
            *extra,
        ]
    )


def test_prepare_module_has_no_training_or_accelerator_imports() -> None:
    source = Path(prepare_cli.__file__).read_text(encoding="utf-8")
    imports = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            imports.append(node.module)

    assert not any(name == "jax" or name.startswith("jax.") for name in imports)
    assert not any("checkpointing" in name for name in imports)
    assert not any(name == "training" or ".training" in name for name in imports)


def test_cli_accepts_hyphenated_and_underscored_preprocessing_options(tmp_path) -> None:
    args = _arguments(
        tmp_path,
        "--num_shards",
        "3",
        "--packing",
        "True",
        "--packing-batch-size",
        "17",
        "--endprompt_enable",
        "True",
        "--endprompt-logical-length",
        "8",
        "--endprompt_logical_length_min",
        "6",
        "--endprompt-prompts",
        "first||second",
        "--endprompt-prompt-loss-weight",
        "0.25",
        "--endprompt_context_loss_weight",
        "0.75",
    )

    assert args.num_shards == 3
    assert args.packing is True
    assert args.packing_batch_size == 17
    assert args.endprompt_prompt_texts == ("first", "second")
    assert args.effective_endprompt_logical_length_min == 6
    assert args.effective_endprompt_logical_length == 8


@pytest.mark.parametrize(
    ("extra", "match"),
    [
        (("--num-shards", "0"), "num_shards"),
        (("--packing-batch-size", "0"), "packing_batch_size"),
        (
            (
                "--assistant-only-loss",
                "--last-assistant-only-loss",
            ),
            "mutually exclusive",
        ),
        (("--assistant-only-loss",), "requires --dataset_text_field messages"),
        (("--hub-revision", "branch"), "requires hub_repo_id"),
    ],
)
def test_cli_rejects_invalid_independent_preprocessing_settings(
    tmp_path,
    extra,
    match,
) -> None:
    with pytest.raises(ValueError, match=match):
        _arguments(tmp_path, *extra)


def test_prepare_writes_rank_specific_parquet_shards_and_manifest(
    tmp_path,
    monkeypatch,
) -> None:
    tokenizer_dir = _write_tokenizer_stub(tmp_path / "tokenizer")
    source = Dataset.from_dict({"text": ["1", "2", "3", "4", "5"]})
    load_calls = []

    def fake_load_training_split(*args, **kwargs):
        load_calls.append((args, kwargs))
        return source

    monkeypatch.setattr(prepare_cli, "load_training_split", fake_load_training_split)
    monkeypatch.setattr(
        prepare_cli.TrainingTokenizer,
        "from_directory",
        lambda path: TinyTokenizer(),
    )
    args = _arguments(tmp_path, "--num-shards", "3")

    manifest = prepare_cli.prepare_dataset(args)

    assert load_calls == [
        (
            ("local-source", "train"),
            {
                "token": None,
                "config_name": None,
                "revision": None,
                "streaming": False,
            },
        )
    ]
    assert tokenizer_dir.is_dir()
    assert [shard.rows for shard in manifest.shards] == [2, 2, 1]
    assert manifest.total_rows == 5
    assert manifest.max_token_id == 15
    assert manifest.max_position_id == 1
    assert manifest.uses_assistant_loss_mask is False
    assert manifest.preprocessing["num_shards"] == 3
    assert manifest.source["rows"] == 5
    assert manifest.tokenizer["vocab_size"] == 128
    assert manifest.tokenizer["files"] == [
        "tokenizer.json",
        "tokenizer_config.json",
    ]

    manifest_path = args.output_dir / PREPARED_MANIFEST_FILENAME
    loaded_manifest = PreparedDatasetManifest.from_json(manifest_path.read_text(encoding="utf-8"))
    assert loaded_manifest == manifest
    assert "version" not in loaded_manifest.to_dict()

    first_tokens = []
    for shard in manifest.shards:
        shard_path = args.output_dir / shard.path
        assert shard_path.is_file()
        dataset = Dataset.from_parquet(str(shard_path))
        assert dataset.column_names == [
            "input_ids",
            "attention_mask",
            "position_ids",
            "segment_ids",
            "loss_weights",
        ]
        first_tokens.extend(row[-2] for row in dataset["input_ids"])
        assert all(len(row) == 4 for row in dataset["input_ids"])
        assert all(len(row) == 4 for row in dataset["loss_weights"])
    assert first_tokens == [1, 2, 3, 4, 5]


def test_prepare_records_assistant_and_endprompt_semantics_without_secrets(
    tmp_path,
    monkeypatch,
) -> None:
    _write_tokenizer_stub(tmp_path / "tokenizer")
    source = Dataset.from_dict({"messages": [[{"role": "user", "content": "x"}]]})

    class ChatTokenizer(TinyTokenizer):
        def apply_chat_template(self, messages, **kwargs):
            del messages, kwargs
            return {
                "input_ids": [1, 2, 3],
                "attention_mask": [1, 1, 1],
                "assistant_masks": [0, 1, 1],
            }

        def __call__(self, text, **kwargs):
            del text, kwargs
            return {"input_ids": [7]}

    monkeypatch.setattr(prepare_cli, "load_training_split", lambda *args, **kwargs: source)
    monkeypatch.setattr(
        prepare_cli.TrainingTokenizer,
        "from_directory",
        lambda path: ChatTokenizer(),
    )
    args = _arguments(
        tmp_path,
        "--dataset-text-field",
        "messages",
        "--assistant-only-loss",
        "--endprompt-enable",
        "--endprompt-logical-length",
        "8",
        "--endprompt-prompts",
        "terminal",
        "--token",
        "do-not-serialize-me",
    )

    manifest = prepare_cli.prepare_dataset(args)
    serialized = (args.output_dir / PREPARED_MANIFEST_FILENAME).read_text(encoding="utf-8")

    assert manifest.uses_assistant_loss_mask is True
    assert manifest.preprocessing["assistant_only_loss"] is True
    assert manifest.preprocessing["endprompt"] == {
        "logical_length_min": 8,
        "logical_length_max": 8,
        "prompts": ["terminal"],
        "prompt_loss_weight": 0.1,
        "context_loss_weight": 1.0,
    }
    assert "do-not-serialize-me" not in serialized

    shard = Dataset.from_parquet(str(args.output_dir / manifest.shards[0].path))
    assert "assistant_masks" in shard.column_names
    assert shard[0]["loss_weights"][-1] == pytest.approx(0.1)


def test_empty_process_shard_is_rejected_and_partial_output_is_removed(
    tmp_path,
    monkeypatch,
) -> None:
    _write_tokenizer_stub(tmp_path / "tokenizer")
    source = Dataset.from_dict({"text": ["1"]})
    monkeypatch.setattr(prepare_cli, "load_training_split", lambda *args, **kwargs: source)
    monkeypatch.setattr(
        prepare_cli.TrainingTokenizer,
        "from_directory",
        lambda path: TinyTokenizer(),
    )
    args = _arguments(tmp_path, "--num-shards", "3")

    with pytest.raises(ValueError, match="prepared shard 1 is empty"):
        prepare_cli.prepare_dataset(args)

    assert not (args.output_dir / PREPARED_MANIFEST_FILENAME).exists()
    assert not (args.output_dir / "data").exists()


def test_remote_tokenizer_download_is_narrow_and_forwards_identity(
    tmp_path,
    monkeypatch,
) -> None:
    downloaded = _write_tokenizer_stub(tmp_path / "downloaded")
    calls = []

    def fake_snapshot_download(*args, **kwargs):
        calls.append((args, kwargs))
        return str(downloaded)

    monkeypatch.setattr(prepare_cli, "snapshot_download", fake_snapshot_download)
    args = prepare_cli.parse_args(
        [
            "--dataset-name",
            "source",
            "--tokenizer-name-or-path",
            "org/tokenizer",
            "--tokenizer-revision",
            "abc123",
            "--token",
            "secret",
            "--output-dir",
            str(tmp_path / "output"),
        ]
    )

    assert prepare_cli._resolve_tokenizer_directory(args) == downloaded
    assert calls == [
        (
            ("org/tokenizer",),
            {
                "repo_type": "model",
                "revision": "abc123",
                "token": "secret",
                "allow_patterns": list(prepare_cli._TOKENIZER_ALLOW_PATTERNS),
            },
        )
    ]


def test_tokenizer_digest_uses_snapshot_paths_without_resolving_files(
    tmp_path,
    monkeypatch,
) -> None:
    tokenizer_dir = _write_tokenizer_stub(tmp_path / "tokenizer")

    def unexpected_resolve(*args, **kwargs):
        del args, kwargs
        raise AssertionError("snapshot files must be checked lexically, not resolved")

    monkeypatch.setattr(Path, "resolve", unexpected_resolve)

    digest, files = prepare_cli._tokenizer_digest(tokenizer_dir)

    assert len(digest) == 64
    assert files == ("tokenizer.json", "tokenizer_config.json")


def test_tokenizer_digest_accepts_standard_hf_snapshot_symlinks(tmp_path) -> None:
    cache = tmp_path / "models--org--tokenizer"
    snapshot = cache / "snapshots" / "commit"
    blobs = cache / "blobs"
    snapshot.mkdir(parents=True)
    blobs.mkdir()
    tokenizer_blob = blobs / "tokenizer-blob"
    tokenizer_blob.write_text('{"stub": true}\n', encoding="utf-8")
    tokenizer_link = snapshot / "tokenizer.json"
    try:
        tokenizer_link.symlink_to(Path("..") / ".." / "blobs" / tokenizer_blob.name)
    except OSError as error:
        pytest.skip(f"symbolic links are unavailable: {error}")
    (snapshot / "tokenizer_config.json").write_text(
        '{"pad_token": "<pad>"}\n',
        encoding="utf-8",
    )

    materialized = _write_tokenizer_stub(tmp_path / "materialized")
    assert prepare_cli._tokenizer_digest(snapshot) == prepare_cli._tokenizer_digest(materialized)


def test_optional_hub_upload_publishes_manifest_and_all_shards(
    tmp_path,
    monkeypatch,
) -> None:
    _write_tokenizer_stub(tmp_path / "tokenizer")
    source = Dataset.from_dict({"text": ["1", "2"]})
    calls = []

    class FakeApi:
        def create_repo(self, *args, **kwargs):
            calls.append(("create", args, kwargs))

        def create_branch(self, **kwargs):
            calls.append(("branch", (), kwargs))

        def upload_folder(self, **kwargs):
            folder = Path(kwargs["folder_path"])
            assert (folder / PREPARED_MANIFEST_FILENAME).is_file()
            assert len(list((folder / "data").glob("*.parquet"))) == 2
            calls.append(("upload", (), kwargs))

    monkeypatch.setattr(prepare_cli, "HfApi", FakeApi)
    monkeypatch.setattr(prepare_cli, "load_training_split", lambda *args, **kwargs: source)
    monkeypatch.setattr(
        prepare_cli.TrainingTokenizer,
        "from_directory",
        lambda path: TinyTokenizer(),
    )
    args = _arguments(
        tmp_path,
        "--num-shards",
        "2",
        "--hub-repo-id",
        "org/prepared",
        "--hub-revision",
        "data-branch",
        "--private",
        "--token",
        "secret",
    )

    prepare_cli.prepare_dataset(args)

    assert calls[0] == (
        "create",
        ("org/prepared",),
        {
            "token": "secret",
            "private": True,
            "repo_type": "dataset",
            "exist_ok": True,
        },
    )
    assert calls[1] == (
        "branch",
        (),
        {
            "repo_id": "org/prepared",
            "branch": "data-branch",
            "token": "secret",
            "repo_type": "dataset",
            "exist_ok": True,
        },
    )
    assert calls[2][0] == "upload"
    assert calls[2][2]["repo_id"] == "org/prepared"
    assert calls[2][2]["revision"] == "data-branch"
    assert calls[2][2]["repo_type"] == "dataset"


def test_nonempty_output_directory_is_rejected_before_external_work(
    tmp_path,
    monkeypatch,
) -> None:
    _write_tokenizer_stub(tmp_path / "tokenizer")
    output = tmp_path / "prepared"
    output.mkdir()
    (output / "owned-by-user.txt").write_text("keep", encoding="utf-8")
    called = False

    def unexpected_load(*args, **kwargs):
        nonlocal called
        called = True
        raise AssertionError("source loading should not start")

    monkeypatch.setattr(prepare_cli, "load_training_split", unexpected_load)
    args = _arguments(tmp_path)

    with pytest.raises(ValueError, match="must be empty"):
        prepare_cli.prepare_dataset(args)
    assert called is False
    assert (output / "owned-by-user.txt").read_text(encoding="utf-8") == "keep"
