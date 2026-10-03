import json

import pytest
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import WhitespaceSplit

from causal_trainer.data.tokenizer import TrainingTokenizer


def _backend() -> Tokenizer:
    tokenizer = Tokenizer(
        WordLevel(
            {
                "unk": 0,
                "pad": 1,
                "eos": 2,
                "user": 3,
                "assistant": 4,
                "question": 5,
                "answer": 6,
                "tool": 7,
            },
            unk_token="unk",
        )
    )
    tokenizer.pre_tokenizer = WhitespaceSplit()
    return tokenizer


def test_from_directory_loads_special_tokens_and_file_templates(tmp_path) -> None:
    backend = _backend()
    backend.save(str(tmp_path / "tokenizer.json"))
    (tmp_path / "tokenizer_config.json").write_text(
        json.dumps(
            {
                "pad_token": {"content": "pad"},
                "eos_token": "eos",
                "chat_template": "ignored config template",
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "chat_template.jinja").write_text(
        "{% for message in messages %}{{ message.role + ' ' }}"
        "{% if message.role == 'assistant' %}"
        "{% generation %}{{ message.content + ' eos' }}{% endgeneration %}"
        "{% else %}{{ message.content }}{% endif %} {% endfor %}{{ eos_token }}",
        encoding="utf-8",
    )
    template_directory = tmp_path / "chat_templates"
    template_directory.mkdir()
    (template_directory / "tool_use.jinja").write_text("tool {{ tools | tojson }}", encoding="utf-8")

    tokenizer = TrainingTokenizer.from_directory(tmp_path)

    assert tokenizer.pad_token_id == 1
    assert tokenizer.eos_token_id == 2
    assert tokenizer.get_chat_template(tools=[{"type": "function"}]).startswith("tool ")
    encoded = tokenizer.apply_chat_template(
        [
            {"role": "user", "content": "question"},
            {"role": "assistant", "content": "answer"},
        ],
        return_assistant_tokens_mask=True,
    )
    assert encoded["assistant_masks"] == [0, 0, 0, 1, 1, 0]


def test_generation_trailing_whitespace_does_not_mask_a_later_user_span() -> None:
    tokenizer = TrainingTokenizer(
        _backend(),
        chat_template=(
            "{% for message in messages %}{{ message.role + ' ' }}"
            "{% if message.role == 'assistant' %}"
            "{% generation %}{{ message.content + ' ' }}{% endgeneration %}"
            "{% else %}{{ message.content + ' ' }}{% endif %}{% endfor %}"
        ),
    )

    encoded = tokenizer.apply_chat_template(
        [
            {"role": "user", "content": "question"},
            {"role": "assistant", "content": "answer"},
            {"role": "user", "content": "question"},
        ],
        return_assistant_tokens_mask=True,
    )

    assert encoded["assistant_masks"] == [0, 0, 0, 1, 0, 0]


def test_message_loss_flags_filter_spans_without_changing_rendered_tokens() -> None:
    tokenizer = TrainingTokenizer(
        _backend(),
        chat_template=(
            "{% for turn in messages %}{{ turn.role + ' ' }}"
            "{% if turn.role == 'assistant' %}"
            "{% generation %}{{ turn.content + ' eos ' }}{% endgeneration %}"
            "{% else %}{{ turn.content + ' ' }}{% endif %}{% endfor %}"
        ),
    )
    messages = [
        {"role": "user", "content": "question", "loss": False},
        {"role": "assistant", "content": "answer", "loss": False},
        {"role": "assistant", "content": "answer"},
    ]

    unfiltered = tokenizer.apply_chat_template(
        [{key: value for key, value in message.items() if key != "loss"} for message in messages],
        return_assistant_tokens_mask=True,
    )
    filtered = tokenizer.apply_chat_template(
        messages,
        return_assistant_tokens_mask=True,
    )

    assert filtered["input_ids"] == unfiltered["input_ids"]
    assert unfiltered["assistant_masks"] == [0, 0, 0, 1, 1, 0, 1, 1]
    assert filtered["assistant_masks"] == [0, 0, 0, 0, 0, 0, 1, 1]


def test_message_loss_requires_one_generation_block_per_assistant() -> None:
    tokenizer = TrainingTokenizer(
        _backend(),
        chat_template=(
            "{% generation %}{{ 'answer ' }}{% endgeneration %}"
            "{% generation %}{{ 'eos' }}{% endgeneration %}"
        ),
    )
    messages = [{"role": "assistant", "content": "answer", "loss": False}]

    with pytest.raises(ValueError, match="exactly one generation block"):
        tokenizer.apply_chat_template(
            messages,
            return_assistant_tokens_mask=True,
        )


def test_message_loss_flags_remains_an_ordinary_template_kwarg() -> None:
    tokenizer = TrainingTokenizer(
        _backend(),
        chat_template="{{ message_loss_flags }}",
    )

    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": "question"}],
        tokenize=False,
        message_loss_flags="template-owned value",
    )

    assert rendered == "template-owned value"


def test_tokenizer_rejects_ids_outside_model_embedding_table() -> None:
    tokenizer = TrainingTokenizer(_backend())
    tokenizer.validate_model_vocabulary(8)
    with pytest.raises(ValueError, match="outside model vocab_size"):
        tokenizer.validate_model_vocabulary(7)


def test_raw_tokenization_never_inserts_post_processor_special_tokens() -> None:
    tokenizer = TrainingTokenizer(_backend())
    encoded = tokenizer(
        "question answer eos",
        add_special_tokens=False,
        truncation=True,
        max_length=2,
    )
    assert encoded == {"input_ids": [5, 6], "attention_mask": [1, 1]}
    with pytest.raises(ValueError, match="automatic special-token insertion"):
        tokenizer("question", add_special_tokens=True)
