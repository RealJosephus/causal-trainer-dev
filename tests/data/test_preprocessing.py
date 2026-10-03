from __future__ import annotations

import pytest

from causal_trainer.data.preprocessing import (
    EndPromptSettings,
    _scan_filter_requirement,
    preprocess_dataset,
    preprocess_example,
    preprocess_records,
    tokenize_endprompt_messages,
    tokenize_endprompt_text,
    tokenize_messages,
    tokenize_raw_text,
)
from causal_trainer.data.tokenizer import TrainingTokenizer


class FakeTokenizer:
    eos_token = "<eos>"
    pad_token_id = 0
    supports_message_loss = True

    def __init__(self) -> None:
        self.raw_calls = []
        self.chat_calls = []
        self.chat_output = {
            "input_ids": [10, 11, 12, 13],
            "attention_mask": [1, 1, 1, 1],
            "assistant_masks": [0, 0, 1, 1],
        }

    def __call__(self, text, **kwargs):
        self.raw_calls.append((text, kwargs))
        ids = list(range(1, min(len(text), kwargs["max_length"]) + 1))
        return {"input_ids": ids, "attention_mask": [1] * len(ids)}

    def apply_chat_template(self, messages, **kwargs):
        self.chat_calls.append((messages, kwargs))
        return dict(self.chat_output)


MESSAGES = [
    {"role": "user", "content": "question"},
    {"role": "assistant", "content": "answer"},
]


def test_filter_scan_avoids_rewrite_when_every_tokenized_row_is_trainable():
    class BatchedRows:
        def __init__(self, rows):
            self.rows = rows

        def __len__(self):
            return len(self.rows)

        def iter(self, *, batch_size):
            for start in range(0, len(self.rows), batch_size):
                rows = self.rows[start : start + batch_size]
                yield {key: [row[key] for row in rows] for key in rows[0]}

    rows = BatchedRows([{"keep": True}, {"keep": True}, {"keep": True}])
    assert _scan_filter_requirement(rows, lambda row: row["keep"], batch_size=2) == (
        False,
        False,
    )

    rows.rows[-1]["keep"] = False
    assert _scan_filter_requirement(rows, lambda row: row["keep"], batch_size=2) == (
        True,
        False,
    )
    assert _scan_filter_requirement(BatchedRows([]), lambda row: row["keep"]) == (
        False,
        True,
    )


def test_raw_text_never_appends_eos_and_disables_tokenizer_special_tokens():
    tokenizer = FakeTokenizer()

    output = tokenize_raw_text("hello", tokenizer, max_sequence_length=8)
    tokenize_raw_text("hello<eos>", tokenizer, max_sequence_length=8)

    assert tokenizer.raw_calls[0][0] == "hello"
    assert tokenizer.raw_calls[1][0] == "hello<eos>"
    assert tokenizer.raw_calls[0][1] == {
        "add_special_tokens": False,
        "return_attention_mask": True,
        "truncation": True,
        "max_length": 8,
    }
    assert output["attention_mask"] == [1] * len(output["input_ids"])


def test_endprompt_preserves_terminal_anchor_positions_and_fractional_weights():
    class AnchorTokenizer:
        def __init__(self):
            self.calls = []

        def __call__(self, text, **kwargs):
            self.calls.append((text, kwargs))
            if text == "anchor":
                return {"input_ids": [90, 91]}
            return {"input_ids": [1, 2, 3, 4, 5, 6][: kwargs["max_length"]]}

    tokenizer = AnchorTokenizer()
    settings = EndPromptSettings(
        logical_length_min=16,
        logical_length_max=16,
        prompts=("anchor",),
        prompt_loss_weight=0.1,
        context_loss_weight=1.0,
    )

    output = tokenize_endprompt_text(
        "raw text",
        tokenizer,
        max_sequence_length=8,
        settings=settings,
        example_index=0,
    )

    assert output == {
        "input_ids": [1, 2, 3, 4, 5, 6, 90, 91],
        "attention_mask": [1] * 8,
        "position_ids": [0, 1, 2, 3, 4, 5, 14, 15],
        "loss_weights": [1.0] * 6 + [0.1, 0.1],
    }
    assert tokenizer.calls[0] == (
        "anchor",
        {"add_special_tokens": False, "return_attention_mask": False},
    )
    assert tokenizer.calls[1][1]["add_special_tokens"] is False


def test_endprompt_dataset_mapping_uses_original_row_indices():
    class IndexedDataset:
        column_names = ["text"]

        def __init__(self):
            self.kwargs = None

        def map(self, function, **kwargs):
            self.kwargs = kwargs
            return [function({"text": "context"}, index) for index in range(2)]

    class PromptTokenizer:
        def __call__(self, text, **kwargs):
            if text == "first":
                return {"input_ids": [90]}
            if text == "second":
                return {"input_ids": [91]}
            return {"input_ids": [1]}

    dataset = IndexedDataset()
    output = preprocess_dataset(
        dataset,
        PromptTokenizer(),
        dataset_text_field="text",
        max_sequence_length=4,
        endprompt=EndPromptSettings(4, 4, ("first", "second")),
    )

    assert dataset.kwargs == {"remove_columns": ["text"], "with_indices": True}
    assert output[0]["input_ids"][-1] == 90
    assert output[1]["input_ids"][-1] == 91


def test_endprompt_dataset_mapping_applies_global_shard_offset():
    class IndexedShard:
        column_names = ["text"]

        def map(self, function, **kwargs):
            assert kwargs == {"remove_columns": ["text"], "with_indices": True}
            return [function({"text": "context"}, index) for index in range(2)]

    class PromptTokenizer:
        def __call__(self, text, **kwargs):
            if text == "first":
                return {"input_ids": [90]}
            if text == "second":
                return {"input_ids": [91]}
            return {"input_ids": [1]}

    output = preprocess_dataset(
        IndexedShard(),
        PromptTokenizer(),
        dataset_text_field="text",
        max_sequence_length=4,
        endprompt=EndPromptSettings(4, 4, ("first", "second")),
        example_index_offset=5,
    )

    assert output[0]["input_ids"][-1] == 91
    assert output[1]["input_ids"][-1] == 90


def test_local_all_filtered_shard_can_remain_empty_for_later_merge():
    datasets = pytest.importorskip("datasets")
    tokenizer = FakeTokenizer()
    tokenizer.chat_output = {
        "input_ids": [10, 11],
        "attention_mask": [1, 1],
        "assistant_masks": [0, 0],
    }
    dataset = datasets.Dataset.from_list(
        [{"messages": [{"role": "user", "content": "no assistant response"}]}]
    )

    output = preprocess_dataset(
        dataset,
        tokenizer,
        dataset_text_field="messages",
        max_sequence_length=4,
        assistant_only_loss=True,
        allow_empty=True,
    )

    assert len(output) == 0
    assert output.column_names == ["input_ids", "attention_mask", "assistant_masks"]


def test_messages_endprompt_combines_assistant_and_prompt_loss_weights():
    class ChatAnchorTokenizer(FakeTokenizer):
        def __call__(self, text, **kwargs):
            self.raw_calls.append((text, kwargs))
            assert text == "anchor"
            return {"input_ids": [90, 91]}

    tokenizer = ChatAnchorTokenizer()
    tokenizer.chat_output = {
        "input_ids": [10, 11, 12, 13, 14, 15],
        "attention_mask": [1] * 6,
        "assistant_masks": [0, 0, 1, 1, 0, 0],
    }
    output = tokenize_endprompt_messages(
        MESSAGES,
        tokenizer,
        max_sequence_length=8,
        assistant_only_loss=True,
        settings=EndPromptSettings(16, 16, ("anchor",), 0.25, 0.5),
        example_index=0,
    )

    assert output == {
        "input_ids": [10, 11, 12, 13, 90, 91],
        "attention_mask": [1] * 6,
        "assistant_masks": [0, 0, 1, 1, 0, 0],
        "position_ids": [0, 1, 2, 3, 14, 15],
        "loss_weights": [0.0, 0.0, 0.5, 0.5, 0.25, 0.25],
    }
    assert tokenizer.chat_calls[0][1]["truncation"] is False
    assert tokenizer.chat_calls[0][1]["max_length"] is None


def test_endprompt_does_not_rescue_chat_without_assistant_in_reserved_budget():
    class BudgetTokenizer(FakeTokenizer):
        def __call__(self, text, **kwargs):
            return {"input_ids": [90, 91]}

        def apply_chat_template(self, messages, **kwargs):
            if messages[0]["content"] == "late":
                return {
                    "input_ids": list(range(8)),
                    "attention_mask": [1] * 8,
                    "assistant_masks": [0, 0, 0, 0, 0, 0, 1, 1],
                }
            return {
                "input_ids": [10, 11, 12, 13],
                "attention_mask": [1] * 4,
                "assistant_masks": [0, 0, 1, 1],
            }

    records = [
        {
            "messages": [
                {"role": "user", "content": "late"},
                {"role": "assistant", "content": "outside"},
            ]
        },
        {"messages": MESSAGES},
    ]
    output = preprocess_records(
        records,
        BudgetTokenizer(),
        dataset_text_field="messages",
        max_sequence_length=8,
        assistant_only_loss=True,
        endprompt=EndPromptSettings(8, 8, ("anchor",)),
    )

    assert len(output) == 1
    assert output[0]["input_ids"][-2:] == [90, 91]
    assert output[0]["assistant_masks"][-2:] == [0, 0]


def test_real_dataset_keeps_explicit_schema_when_first_endprompt_chat_row_is_empty():
    datasets = pytest.importorskip("datasets")

    class SchemaTokenizer:
        def __call__(self, text, **kwargs):
            assert text == "anchor"
            return {"input_ids": [90]}

        def apply_chat_template(self, messages, **kwargs):
            if all(message["role"] != "assistant" for message in messages):
                return {
                    "input_ids": [10, 11],
                    "attention_mask": [1, 1],
                    "assistant_masks": [0, 0],
                }
            return {
                "input_ids": [20, 21, 22],
                "attention_mask": [1, 1, 1],
                "assistant_masks": [0, 0, 1],
            }

    dataset = datasets.Dataset.from_list(
        [
            {"messages": [{"role": "user", "content": "no answer"}]},
            {"messages": MESSAGES},
        ]
    )
    output = preprocess_dataset(
        dataset,
        SchemaTokenizer(),
        dataset_text_field="messages",
        max_sequence_length=8,
        assistant_only_loss=True,
        endprompt=EndPromptSettings(8, 8, ("anchor",)),
    )

    expected_features = datasets.Features(
        {
            "input_ids": datasets.Sequence(datasets.Value("int64")),
            "attention_mask": datasets.Sequence(datasets.Value("int64")),
            "assistant_masks": datasets.Sequence(datasets.Value("int64")),
            "position_ids": datasets.Sequence(datasets.Value("int64")),
            "loss_weights": datasets.Sequence(datasets.Value("float32")),
        }
    )
    assert output.features == expected_features
    assert output.cache_files == []
    assert len(output) == 1
    row = output[0]
    assert row["input_ids"] == [20, 21, 22, 90]
    assert row["attention_mask"] == [1, 1, 1, 1]
    assert row["assistant_masks"] == [0, 0, 1, 0]
    assert row["position_ids"] == [0, 1, 2, 7]
    assert row["loss_weights"] == pytest.approx([0.0, 0.0, 1.0, 0.1])


def test_messages_are_tokenized_directly_with_assistant_mask():
    tokenizer = FakeTokenizer()

    output = tokenize_messages(
        MESSAGES,
        tokenizer,
        max_sequence_length=128,
        assistant_only_loss=True,
    )

    assert tokenizer.chat_calls[0][0] == MESSAGES
    assert tokenizer.chat_calls[0][1] == {
        "tokenize": True,
        "return_dict": True,
        "return_attention_mask": True,
        "return_assistant_tokens_mask": True,
        "truncation": False,
        "max_length": None,
    }
    assert output["assistant_masks"] == [0, 0, 1, 1]


@pytest.mark.parametrize("loss", [0, 1, 0.0, "false", [], {}])
def test_messages_reject_non_boolean_assistant_loss(loss) -> None:
    messages = [
        {"role": "user", "content": "question"},
        {"role": "assistant", "content": "answer", "loss": loss},
    ]

    with pytest.raises(ValueError, match=r"messages\[1\]\.loss.*boolean or null"):
        tokenize_messages(
            messages,
            FakeTokenizer(),
            max_sequence_length=32,
            assistant_only_loss=True,
        )


def test_false_assistant_loss_requires_tokenizer_support() -> None:
    tokenizer = FakeTokenizer()
    tokenizer.supports_message_loss = False

    with pytest.raises(ValueError, match="does not support assistant message loss=false"):
        tokenize_messages(
            [
                {"role": "user", "content": "question"},
                {"role": "assistant", "content": "answer", "loss": False},
            ],
            tokenizer,
            max_sequence_length=32,
            assistant_only_loss=True,
        )


@pytest.mark.parametrize("loss", [True, None])
def test_messages_true_or_null_loss_keeps_the_legacy_template_call(loss) -> None:
    tokenizer = FakeTokenizer()
    messages = [
        {"role": "user", "content": "question", "loss": False},
        {"role": "assistant", "content": "answer", "loss": loss},
    ]

    tokenize_messages(
        messages,
        tokenizer,
        max_sequence_length=32,
        assistant_only_loss=True,
    )

    assert "message_loss_flags" not in tokenizer.chat_calls[0][1]


def test_message_loss_is_inert_without_an_assistant_loss_mode() -> None:
    tokenizer = FakeTokenizer()
    messages = [
        {"role": "user", "content": "question", "loss": False},
        {"role": "assistant", "content": "answer", "loss": "not used"},
    ]

    output = tokenize_messages(messages, tokenizer, max_sequence_length=32)

    assert "message_loss_flags" not in tokenizer.chat_calls[0][1]
    assert "assistant_masks" not in output


def _native_message_loss_tokenizer() -> TrainingTokenizer:
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import WhitespaceSplit

    vocabulary = {
        "unk": 0,
        "user": 1,
        "assistant": 2,
        "question": 3,
        "second": 4,
        "context": 5,
        "answer": 6,
        "eos": 7,
    }
    backend = Tokenizer(WordLevel(vocabulary, unk_token="unk"))
    backend.pre_tokenizer = WhitespaceSplit()
    return TrainingTokenizer(
        backend,
        chat_template=(
            "{% for message in messages %}{{ message.role + ' ' }}"
            "{% if message.role == 'assistant' %}"
            "{% generation %}{{ message.content + ' eos ' }}{% endgeneration %}"
            "{% else %}{{ message.content + ' ' }}{% endif %}"
            "{% endfor %}"
        ),
    )


def test_real_dataset_treats_missing_arrow_loss_as_trainable() -> None:
    datasets = pytest.importorskip("datasets")
    dataset = datasets.Dataset.from_list(
        [
            {
                "messages": [
                    {"role": "user", "content": "question"},
                    {"role": "assistant", "content": "context", "loss": False},
                ]
            },
            {
                "messages": [
                    {"role": "user", "content": "question"},
                    {"role": "assistant", "content": "answer"},
                ]
            },
        ]
    )
    assert dataset[1]["messages"][1]["loss"] is None

    output = preprocess_dataset(
        dataset,
        _native_message_loss_tokenizer(),
        dataset_text_field="messages",
        max_sequence_length=16,
        assistant_only_loss=True,
    )

    assert len(output) == 1
    assert output[0]["input_ids"] == [1, 3, 2, 6, 7]
    assert output[0]["assistant_masks"] == [0, 0, 0, 1, 1]


def test_native_loss_true_and_null_match_the_unannotated_path() -> None:
    tokenizer = _native_message_loss_tokenizer()
    base = [
        {"role": "user", "content": "question"},
        {"role": "assistant", "content": "answer"},
    ]

    expected = tokenize_messages(
        base,
        tokenizer,
        max_sequence_length=16,
        assistant_only_loss=True,
    )

    for value in (True, None):
        annotated = [dict(message) for message in base]
        annotated[1]["loss"] = value
        assert (
            tokenize_messages(
                annotated,
                tokenizer,
                max_sequence_length=16,
                assistant_only_loss=True,
            )
            == expected
        )


def test_last_assistant_loss_does_not_fall_back_from_a_disabled_final_span() -> None:
    output = tokenize_messages(
        [
            {"role": "user", "content": "question"},
            {"role": "assistant", "content": "answer"},
            {"role": "user", "content": "second"},
            {"role": "assistant", "content": "context", "loss": False},
        ],
        _native_message_loss_tokenizer(),
        max_sequence_length=16,
        last_assistant_only_loss=True,
    )

    assert output == {
        "input_ids": [],
        "attention_mask": [],
        "assistant_masks": [],
    }


def test_native_message_loss_respects_the_truncation_boundary() -> None:
    tokenizer = _native_message_loss_tokenizer()
    early_trainable = [
        {"role": "user", "content": "question"},
        {"role": "assistant", "content": "answer"},
        {"role": "user", "content": "second"},
        {"role": "assistant", "content": "context", "loss": False},
    ]
    late_trainable = [
        {"role": "user", "content": "question"},
        {"role": "assistant", "content": "context", "loss": False},
        {"role": "user", "content": "second"},
        {"role": "assistant", "content": "answer"},
    ]

    kept = tokenize_messages(
        early_trainable,
        tokenizer,
        max_sequence_length=8,
        assistant_only_loss=True,
    )
    filtered = tokenize_messages(
        late_trainable,
        tokenizer,
        max_sequence_length=8,
        assistant_only_loss=True,
    )

    assert kept["input_ids"] == [1, 3, 2, 6, 7]
    assert kept["assistant_masks"] == [0, 0, 0, 1, 1]
    assert filtered == {
        "input_ids": [],
        "attention_mask": [],
        "assistant_masks": [],
    }


def test_assistant_only_trims_all_fields_at_last_trainable_token_in_prefix():
    tokenizer = FakeTokenizer()
    tokenizer.chat_output = {
        "input_ids": list(range(10)),
        "attention_mask": [1] * 10,
        "assistant_masks": [0, 0, 1, 1, 0, 0, 1, 1, 0, 0],
    }

    output = tokenize_messages(
        MESSAGES,
        tokenizer,
        max_sequence_length=5,
        assistant_only_loss=True,
    )

    assert output == {
        "input_ids": [0, 1, 2, 3],
        "attention_mask": [1, 1, 1, 1],
        "assistant_masks": [0, 0, 1, 1],
    }


def test_last_assistant_only_loss_keeps_only_final_generation_span() -> None:
    tokenizer = FakeTokenizer()
    tokenizer.chat_output = {
        "input_ids": list(range(10)),
        "attention_mask": [1] * 10,
        "assistant_masks": [0, 0, 1, 1, 0, 0, 1, 1, 0, 0],
    }

    output = tokenize_messages(
        [
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "earlier"},
            {"role": "user", "content": "second"},
            {"role": "assistant", "content": "final"},
        ],
        tokenizer,
        max_sequence_length=10,
        last_assistant_only_loss=True,
    )

    assert output == {
        "input_ids": list(range(8)),
        "attention_mask": [1] * 8,
        "assistant_masks": [0, 0, 0, 0, 0, 0, 1, 1],
    }
    assert tokenizer.chat_calls[0][1]["return_assistant_tokens_mask"] is True
    assert tokenizer.chat_calls[0][1]["truncation"] is False
    assert tokenizer.chat_calls[0][1]["max_length"] is None


def test_messages_endprompt_combines_last_assistant_and_prompt_loss_weights() -> None:
    class ChatAnchorTokenizer(FakeTokenizer):
        def __call__(self, text, **kwargs):
            self.raw_calls.append((text, kwargs))
            assert text == "anchor"
            return {"input_ids": [90, 91]}

    tokenizer = ChatAnchorTokenizer()
    tokenizer.chat_output = {
        "input_ids": [10, 11, 12, 13, 14, 15],
        "attention_mask": [1] * 6,
        "assistant_masks": [0, 1, 0, 0, 1, 1],
    }
    output = tokenize_endprompt_messages(
        [
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "earlier"},
            {"role": "user", "content": "second"},
            {"role": "assistant", "content": "final"},
        ],
        tokenizer,
        max_sequence_length=8,
        assistant_only_loss=False,
        last_assistant_only_loss=True,
        settings=EndPromptSettings(16, 16, ("anchor",), 0.25, 0.5),
        example_index=0,
    )

    assert output == {
        "input_ids": [10, 11, 12, 13, 14, 15, 90, 91],
        "attention_mask": [1] * 8,
        "assistant_masks": [0, 0, 0, 0, 1, 1, 0, 0],
        "position_ids": [0, 1, 2, 3, 4, 5, 14, 15],
        "loss_weights": [0.0, 0.0, 0.0, 0.0, 0.5, 0.5, 0.25, 0.25],
    }


def test_last_assistant_only_loss_does_not_fall_back_when_final_span_is_truncated() -> None:
    tokenizer = FakeTokenizer()
    tokenizer.chat_output = {
        "input_ids": list(range(8)),
        "attention_mask": [1] * 8,
        "assistant_masks": [0, 1, 1, 0, 0, 0, 1, 1],
    }

    output = tokenize_messages(
        [
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "earlier"},
            {"role": "user", "content": "second"},
            {"role": "assistant", "content": "final"},
        ],
        tokenizer,
        max_sequence_length=5,
        last_assistant_only_loss=True,
    )

    assert output == {key: [] for key in tokenizer.chat_output}


def test_assistant_only_records_filter_assistant_tokens_beyond_length_limit():
    class PerConversationTokenizer(FakeTokenizer):
        def apply_chat_template(self, messages, **kwargs):
            self.chat_calls.append((messages, kwargs))
            if messages[0]["content"] == "too long":
                return {
                    "input_ids": list(range(8)),
                    "attention_mask": [1] * 8,
                    "assistant_masks": [0, 0, 0, 0, 0, 0, 1, 1],
                }
            return {
                "input_ids": [10, 11, 12, 13],
                "attention_mask": [1, 1, 1, 1],
                "assistant_masks": [0, 0, 1, 1],
            }

    records = [
        {
            "messages": [
                {"role": "user", "content": "too long"},
                {"role": "assistant", "content": "late"},
            ]
        },
        {"messages": MESSAGES},
    ]

    output = preprocess_records(
        records,
        PerConversationTokenizer(),
        dataset_text_field="messages",
        max_sequence_length=4,
        assistant_only_loss=True,
    )

    assert len(output) == 1
    assert output[0]["assistant_masks"] == [0, 0, 1, 1]


def test_messages_forward_tools_and_template_kwargs():
    tokenizer = FakeTokenizer()
    preprocess_example(
        {
            "messages": MESSAGES,
            "tools": '[{"type":"function"}]',
            "chat_template_kwargs": {"custom_flag": True},
        },
        tokenizer,
        dataset_text_field="messages",
        max_sequence_length=32,
    )
    call_kwargs = tokenizer.chat_calls[0][1]
    assert call_kwargs["tools"] == [{"type": "function"}]
    assert call_kwargs["custom_flag"] is True


@pytest.mark.parametrize(
    ("tools", "expected"),
    [
        (["{\"type\": \"function\"}"], [{"type": "function"}]),
        ([{"type": "function"}], [{"type": "function"}]),
        ({"type": "function"}, {"type": "function"}),
    ],
)
def test_messages_normalize_tools_without_changing_structured_values(tools, expected):
    tokenizer = FakeTokenizer()
    preprocess_example(
        {"messages": MESSAGES, "tools": tools},
        tokenizer,
        dataset_text_field="messages",
        max_sequence_length=32,
    )

    assert tokenizer.chat_calls[0][1]["tools"] == expected


@pytest.mark.parametrize(
    "chat_output",
    [
        {"input_ids": [10], "attention_mask": [1], "assistant_masks": [1]},
        {"input_ids": [10, 11], "attention_mask": [0, 1], "assistant_masks": [0, 1]},
    ],
)
def test_assistant_only_requires_a_valid_predecessor_for_trainable_target(chat_output):
    tokenizer = FakeTokenizer()
    tokenizer.chat_output = chat_output

    output = tokenize_messages(
        MESSAGES,
        tokenizer,
        max_sequence_length=32,
        assistant_only_loss=True,
    )

    assert output == {key: [] for key in chat_output}


@pytest.mark.parametrize(
    "chat_output, match",
    [
        (
            {"input_ids": [1, 2], "attention_mask": [1, 1]},
            "requires a chat template that returns an assistant token mask",
        ),
        (
            {"input_ids": [1, 2], "attention_mask": [1, 1], "assistant_masks": [0, 0]},
            "contains no assistant tokens",
        ),
    ],
)
def test_assistant_only_loss_fails_fast(chat_output, match):
    tokenizer = FakeTokenizer()
    tokenizer.chat_output = chat_output

    with pytest.raises(ValueError, match=match):
        tokenize_messages(
            MESSAGES,
            tokenizer,
            max_sequence_length=32,
            assistant_only_loss=True,
        )


def test_assistant_only_loss_is_rejected_for_raw_text():
    with pytest.raises(ValueError, match="only for messages"):
        preprocess_example(
            {"text": "hello"},
            FakeTokenizer(),
            dataset_text_field="text",
            max_sequence_length=32,
            assistant_only_loss=True,
        )


def test_last_assistant_only_loss_is_rejected_for_raw_text():
    with pytest.raises(ValueError, match="last_assistant_only_loss.*only for messages"):
        preprocess_example(
            {"text": "hello"},
            FakeTokenizer(),
            dataset_text_field="text",
            max_sequence_length=32,
            last_assistant_only_loss=True,
        )


def test_preprocess_example_detects_messages_from_selected_field():
    tokenizer = FakeTokenizer()
    output = preprocess_example(
        {"conversation": MESSAGES},
        tokenizer,
        dataset_text_field="conversation",
        max_sequence_length=32,
        assistant_only_loss=True,
    )
    assert output["input_ids"] == [10, 11, 12, 13]


def test_preprocess_records_returns_plain_python_rows():
    rows = preprocess_records(
        [{"text": "a"}, {"text": "b"}],
        FakeTokenizer(),
        dataset_text_field="text",
        max_sequence_length=8,
    )
    assert len(rows) == 2
    assert all(isinstance(row["input_ids"], list) for row in rows)


def test_mask_lengths_are_validated():
    tokenizer = FakeTokenizer()
    tokenizer.chat_output = {
        "input_ids": [1, 2, 3],
        "attention_mask": [1, 1, 1],
        "assistant_masks": [0, 1],
    }
    with pytest.raises(ValueError, match="same length"):
        tokenize_messages(
            MESSAGES,
            tokenizer,
            max_sequence_length=8,
            assistant_only_loss=True,
        )


def test_attention_mask_must_be_binary():
    tokenizer = FakeTokenizer()
    tokenizer.chat_output = {
        "input_ids": [1, 2, 3],
        "attention_mask": [1, 2, 1],
        "assistant_masks": [0, 1, 1],
    }
    with pytest.raises(ValueError, match="binary 0/1"):
        tokenize_messages(
            MESSAGES,
            tokenizer,
            max_sequence_length=8,
            assistant_only_loss=True,
        )


def test_training_tokenizer_chat_mask_stays_aligned_at_truncation_boundary():
    """Exercise the independent generation-span implementation offline."""

    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import WhitespaceSplit

    vocabulary = {
        "unk": 0,
        "bos": 1,
        "eos": 2,
        "pad": 3,
        "user": 4,
        "assistant": 5,
        "q": 6,
        "a": 7,
        "b": 8,
        "c": 9,
        "d": 10,
    }
    backend = Tokenizer(WordLevel(vocabulary, unk_token="unk"))
    backend.pre_tokenizer = WhitespaceSplit()
    tokenizer = TrainingTokenizer(
        backend,
        special_tokens={
            "unk_token": "unk",
            "bos_token": "bos",
            "eos_token": "eos",
            "pad_token": "pad",
        },
        chat_template=(
            "{% for message in messages %}"
            "{% if message['role'] == 'assistant' %}"
            "{{ 'assistant ' }}{% generation %}{{ message['content'] + ' ' }}{% endgeneration %}"
            "{% else %}{{ 'user ' + message['content'] + ' ' }}{% endif %}"
            "{% endfor %}{{ eos_token }}"
        ),
    )
    messages = [
        {"role": "user", "content": "q q q"},
        {"role": "assistant", "content": "a b c d"},
    ]

    full = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        return_dict=True,
        return_attention_mask=True,
        return_assistant_tokens_mask=True,
        truncation=False,
    )
    directly_truncated = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        return_dict=True,
        return_attention_mask=True,
        return_assistant_tokens_mask=True,
        truncation=True,
        max_length=8,
    )
    processed = tokenize_messages(
        messages,
        tokenizer,
        max_sequence_length=8,
        assistant_only_loss=True,
    )

    assert directly_truncated["assistant_masks"] == full["assistant_masks"][:8]
    assert processed["input_ids"] == full["input_ids"][:8]
    assert processed["assistant_masks"] == full["assistant_masks"][:8]
    assert len(processed["input_ids"]) == len(processed["attention_mask"]) == len(
        processed["assistant_masks"]
    )
