"""Dataset preprocessing for causal language-model training.

This module deliberately keeps the tokenizer boundary small. It returns plain
Python lists so that the result can be consumed by Hugging Face Datasets, a
Python iterator, or unit tests without importing JAX.
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any


TokenizedExample = dict[str, list[int] | list[float]]


def _assistant_loss_mode(
    *,
    assistant_only_loss: bool,
    last_assistant_only_loss: bool,
) -> str | None:
    if assistant_only_loss and last_assistant_only_loss:
        raise ValueError("assistant_only_loss and last_assistant_only_loss are mutually exclusive")
    if last_assistant_only_loss:
        return "last_assistant_only_loss"
    if assistant_only_loss:
        return "assistant_only_loss"
    return None


@dataclass(frozen=True, slots=True)
class EndPromptSettings:
    """Static terminal-anchor preprocessing settings."""

    logical_length_min: int
    logical_length_max: int
    prompts: tuple[str, ...]
    prompt_loss_weight: float = 0.1
    context_loss_weight: float = 1.0

    def __post_init__(self) -> None:
        if self.logical_length_min <= 0 or self.logical_length_max <= 0:
            raise ValueError("EndPrompt logical lengths must be positive")
        if self.logical_length_min > self.logical_length_max:
            raise ValueError("EndPrompt minimum logical length cannot exceed its maximum")
        if not self.prompts or any(not isinstance(prompt, str) or not prompt for prompt in self.prompts):
            raise ValueError("EndPrompt requires at least one non-empty prompt")
        weights = (self.context_loss_weight, self.prompt_loss_weight)
        if any(not math.isfinite(weight) or weight < 0.0 for weight in weights):
            raise ValueError("EndPrompt loss weights must be finite and non-negative")
        if not any(weight > 0.0 for weight in weights):
            raise ValueError("at least one EndPrompt loss weight must be positive")

    def logical_length_for_row(self, index: int) -> int:
        """Return the reference-compatible deterministic logical length."""

        span = self.logical_length_max - self.logical_length_min + 1
        if span == 1:
            return self.logical_length_min
        offset = ((int(index) * 1_103_515_245 + 12_345) & 0x7FFFFFFF) % span
        return self.logical_length_min + offset


def _as_int_list(value: Any, *, field_name: str) -> list[int]:
    """Convert a tokenizer output field to a one-dimensional integer list."""

    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, tuple):
        value = list(value)
    if not isinstance(value, list):
        raise ValueError(f"Tokenizer field {field_name!r} must be a one-dimensional sequence.")
    if value and isinstance(value[0], (list, tuple)):
        if len(value) != 1:
            raise ValueError(f"Tokenizer field {field_name!r} unexpectedly contains a batch.")
        value = list(value[0])
    try:
        return [int(item) for item in value]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Tokenizer field {field_name!r} must contain integers.") from exc


def _normalize_tokenizer_output(
    encoded: Mapping[str, Any],
    *,
    require_assistant_mask: bool,
) -> TokenizedExample:
    if not isinstance(encoded, Mapping) or "input_ids" not in encoded:
        raise ValueError("Tokenizer output must be a mapping containing 'input_ids'.")

    input_ids = _as_int_list(encoded["input_ids"], field_name="input_ids")
    if not input_ids:
        raise ValueError("Tokenization produced an empty sequence.")

    if "attention_mask" in encoded:
        attention_mask = _as_int_list(encoded["attention_mask"], field_name="attention_mask")
    else:
        attention_mask = [1] * len(input_ids)
    if len(attention_mask) != len(input_ids):
        raise ValueError("attention_mask and input_ids must have the same length.")
    if any(value not in (0, 1) for value in attention_mask):
        raise ValueError("attention_mask must contain only binary 0/1 values.")

    output: TokenizedExample = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
    }

    # Accept the common singular spellings as well as the canonical plural
    # assistant mask field.
    assistant_key = next(
        (
            key
            for key in ("assistant_masks", "assistant_mask", "assistant_tokens_mask")
            if key in encoded
        ),
        None,
    )
    if assistant_key is not None and require_assistant_mask:
        assistant_masks = _as_int_list(encoded[assistant_key], field_name=assistant_key)
        if len(assistant_masks) != len(input_ids):
            raise ValueError("assistant mask and input_ids must have the same length.")
        if any(value not in (0, 1) for value in assistant_masks):
            raise ValueError("assistant mask must contain only binary 0/1 values.")
        output["assistant_masks"] = assistant_masks

    if require_assistant_mask:
        if assistant_key is None:
            raise ValueError(
                "Assistant loss masking requires a chat template that returns an assistant token mask. "
                "Add {% generation %} blocks to the tokenizer chat template."
            )

    return output


def _validate_messages(messages: Any) -> list[Mapping[str, Any]]:
    if not isinstance(messages, Sequence) or isinstance(messages, (str, bytes)) or not messages:
        raise ValueError("A messages example must be a non-empty sequence of role/content mappings.")
    normalized = list(messages)
    for index, message in enumerate(normalized):
        if not isinstance(message, Mapping):
            raise ValueError(f"messages[{index}] must be a mapping.")
        if "role" not in message or "content" not in message:
            raise ValueError(f"messages[{index}] must contain both 'role' and 'content'.")
    return normalized


def _validate_assistant_message_loss(
    messages: Sequence[Mapping[str, Any]],
) -> bool:
    has_disabled_assistant = False
    for index, message in enumerate(messages):
        if message.get("role") != "assistant":
            continue
        value = message.get("loss")
        if value is not None and type(value) is not bool:
            raise ValueError(f"messages[{index}].loss must be a boolean or null.")
        has_disabled_assistant |= value is False
    return has_disabled_assistant


def _has_nonempty_trainable_assistant_content(
    messages: Sequence[Mapping[str, Any]],
) -> bool:
    return any(
        message.get("role") == "assistant"
        and message.get("loss") is not False
        and bool(message.get("content"))
        for message in messages
    )


def _empty_tokenized_example(example: TokenizedExample) -> TokenizedExample:
    return {key: [] for key in example}


def _keep_last_assistant_span(assistant_masks: Sequence[int]) -> list[int]:
    """Keep only the final contiguous generation span in an assistant mask."""

    output = [0] * len(assistant_masks)
    try:
        stop = next(
            index
            for index in range(len(assistant_masks) - 1, -1, -1)
            if int(assistant_masks[index]) != 0
        )
    except StopIteration:
        return output
    start = stop
    while start > 0 and int(assistant_masks[start - 1]) != 0:
        start -= 1
    output[start : stop + 1] = [1] * (stop - start + 1)
    return output


def _trainable_assistant_indices(
    example: Mapping[str, Any],
    *,
    max_sequence_length: int | None = None,
) -> list[int]:
    """Return assistant targets that have a valid in-sequence predecessor."""

    input_ids = example.get("input_ids")
    attention_mask = example.get("attention_mask")
    assistant_masks = example.get("assistant_masks")
    if input_ids is None or attention_mask is None or assistant_masks is None:
        return []
    length = min(len(input_ids), len(attention_mask), len(assistant_masks))
    if max_sequence_length is not None:
        length = min(length, max_sequence_length)
    return [
        index
        for index in range(1, length)
        if int(assistant_masks[index]) != 0
        and int(attention_mask[index - 1]) == 1
        and int(attention_mask[index]) == 1
    ]


def _trim_to_last_assistant_token(
    example: TokenizedExample,
    *,
    max_sequence_length: int,
) -> TokenizedExample:
    """Keep a bounded prefix ending at its last trainable assistant token.

    Chat templates are rendered without tokenizer-side truncation first.  This
    mirrors the reference SFT pipeline and avoids relying on the tokenizer to
    reconstruct generation spans after truncation.  An empty result is a
    deliberate sentinel which dataset preprocessing filters out.
    """

    trainable_indices = _trainable_assistant_indices(
        example,
        max_sequence_length=max_sequence_length,
    )
    if not trainable_indices:
        return _empty_tokenized_example(example)
    keep_length = trainable_indices[-1] + 1
    return {key: value[:keep_length] for key, value in example.items()}


def _has_trainable_tokens(example: Mapping[str, Any]) -> bool:
    return bool(_trainable_assistant_indices(example))


def _has_effective_loss_target(example: Mapping[str, Any]) -> bool:
    """Return whether at least one positive-weight shifted target exists."""

    input_ids = example.get("input_ids")
    attention_mask = example.get("attention_mask")
    if input_ids is None or attention_mask is None:
        return False
    length = min(len(input_ids), len(attention_mask))
    weights = example.get("loss_weights")
    if weights is not None:
        length = min(length, len(weights))
    for index in range(1, length):
        weight = float(weights[index]) if weights is not None else 1.0
        if (
            int(attention_mask[index - 1]) == 1
            and int(attention_mask[index]) == 1
            and weight > 0.0
        ):
            return True
    return False


def tokenize_raw_text(
    text: str,
    tokenizer: Any,
    *,
    max_sequence_length: int,
) -> TokenizedExample:
    """Tokenize raw PT text without manually or implicitly appending EOS."""

    if max_sequence_length <= 0:
        raise ValueError("max_sequence_length must be positive.")
    if not isinstance(text, str):
        raise ValueError("Raw text examples must be strings.")
    encoded = tokenizer(
        text,
        add_special_tokens=False,
        return_attention_mask=True,
        truncation=True,
        max_length=max_sequence_length,
    )
    return _normalize_tokenizer_output(encoded, require_assistant_mask=False)


def _token_ids(encoded: Any, *, source: str) -> list[int]:
    if not isinstance(encoded, Mapping) or "input_ids" not in encoded:
        raise ValueError(f"{source} tokenizer output must contain input_ids")
    return _as_int_list(encoded["input_ids"], field_name=f"{source}.input_ids")


def _endprompt_prompt_ids(
    tokenizer: Any,
    *,
    max_sequence_length: int,
    settings: EndPromptSettings,
    example_index: int,
) -> list[int]:
    if max_sequence_length <= 1:
        raise ValueError("max_sequence_length must be greater than one for EndPrompt")
    if settings.logical_length_min < max_sequence_length:
        raise ValueError("EndPrompt logical lengths must be at least max_sequence_length")

    prompt_text = settings.prompts[int(example_index) % len(settings.prompts)]
    prompt_ids = _token_ids(
        tokenizer(
            prompt_text,
            add_special_tokens=False,
            return_attention_mask=False,
        ),
        source="EndPrompt prompt",
    )
    if not prompt_ids:
        raise ValueError("EndPrompt terminal prompt tokenized to an empty sequence")
    if len(prompt_ids) >= max_sequence_length:
        raise ValueError(
            "EndPrompt terminal prompt must be shorter than max_sequence_length; "
            f"got {len(prompt_ids)} tokens for length {max_sequence_length}"
        )
    if len(prompt_ids) > settings.logical_length_min:
        raise ValueError("EndPrompt prompt is longer than the minimum logical length")
    return prompt_ids


def _positions_from_attention_mask(attention_mask: Sequence[int]) -> list[int]:
    positions: list[int] = []
    current = 0
    for value in attention_mask:
        if int(value) == 1:
            positions.append(current)
            current += 1
        else:
            positions.append(0)
            current = 0
    return positions


def _append_endprompt(
    context: TokenizedExample,
    prompt_ids: Sequence[int],
    *,
    settings: EndPromptSettings,
    example_index: int,
    assistant_only_loss: bool,
) -> TokenizedExample:
    """Append an anchored terminal span and construct final target weights.

    With assistant-only SFT, ``assistant_masks`` remains a role-span mask: the
    appended terminal tokens are zero there. ``loss_weights`` is already the
    final combined objective, so the chat portion is weighted only where the
    assistant mask is one while the terminal span keeps its own weight.
    """

    if assistant_only_loss and not _has_trainable_tokens(context):
        empty = _empty_tokenized_example(context)
        empty["position_ids"] = []
        empty["loss_weights"] = []
        return empty

    context_ids = [int(value) for value in context["input_ids"]]
    context_attention = [int(value) for value in context["attention_mask"]]
    logical_length = settings.logical_length_for_row(example_index)
    prompt_start = logical_length - len(prompt_ids)

    output: TokenizedExample = {
        key: list(values)
        for key, values in context.items()
    }
    output["input_ids"] = context_ids + [int(value) for value in prompt_ids]
    output["attention_mask"] = context_attention + [1] * len(prompt_ids)
    output["position_ids"] = _positions_from_attention_mask(context_attention) + list(
        range(prompt_start, logical_length)
    )

    if assistant_only_loss:
        assistant_masks = [int(value) for value in context["assistant_masks"]]
        output["assistant_masks"] = assistant_masks + [0] * len(prompt_ids)
        context_weights = [
            float(settings.context_loss_weight) * float(mask) * float(attention)
            for mask, attention in zip(assistant_masks, context_attention, strict=True)
        ]
    else:
        context_weights = [
            float(settings.context_loss_weight) * float(attention)
            for attention in context_attention
        ]
    output["loss_weights"] = context_weights + [float(settings.prompt_loss_weight)] * len(prompt_ids)

    if not _has_effective_loss_target(output):
        return _empty_tokenized_example(output)
    return output


def tokenize_endprompt_text(
    text: str,
    tokenizer: Any,
    *,
    max_sequence_length: int,
    settings: EndPromptSettings,
    example_index: int,
) -> TokenizedExample:
    """Append one terminal anchor while preserving its logical position IDs."""

    if not isinstance(text, str):
        raise ValueError("EndPrompt is supported only for raw string records")
    prompt_ids = _endprompt_prompt_ids(
        tokenizer,
        max_sequence_length=max_sequence_length,
        settings=settings,
        example_index=example_index,
    )

    context_budget = max_sequence_length - len(prompt_ids)
    context_ids = _token_ids(
        tokenizer(
            text,
            add_special_tokens=False,
            return_attention_mask=False,
            truncation=True,
            max_length=context_budget,
        ),
        source="EndPrompt context",
    )
    if len(context_ids) > context_budget:
        raise ValueError("EndPrompt context tokenizer ignored its truncation budget")

    return _append_endprompt(
        {
            "input_ids": context_ids,
            "attention_mask": [1] * len(context_ids),
        },
        prompt_ids,
        settings=settings,
        example_index=example_index,
        assistant_only_loss=False,
    )


def tokenize_endprompt_messages(
    messages: Sequence[Mapping[str, Any]],
    tokenizer: Any,
    *,
    max_sequence_length: int,
    assistant_only_loss: bool,
    last_assistant_only_loss: bool = False,
    settings: EndPromptSettings,
    example_index: int,
    tools: Any | None = None,
    chat_template_kwargs: Mapping[str, Any] | None = None,
) -> TokenizedExample:
    """Render a conversation within the space left by an anchored prompt."""

    assistant_loss_mode = _assistant_loss_mode(
        assistant_only_loss=assistant_only_loss,
        last_assistant_only_loss=last_assistant_only_loss,
    )

    prompt_ids = _endprompt_prompt_ids(
        tokenizer,
        max_sequence_length=max_sequence_length,
        settings=settings,
        example_index=example_index,
    )
    context_budget = max_sequence_length - len(prompt_ids)
    context = tokenize_messages(
        messages,
        tokenizer,
        max_sequence_length=context_budget,
        assistant_only_loss=assistant_only_loss,
        last_assistant_only_loss=last_assistant_only_loss,
        tools=tools,
        chat_template_kwargs=chat_template_kwargs,
    )
    if len(context.get("input_ids", ())) > context_budget:
        raise ValueError("EndPrompt chat tokenizer ignored its truncation budget")
    return _append_endprompt(
        context,
        prompt_ids,
        settings=settings,
        example_index=example_index,
        assistant_only_loss=assistant_loss_mode is not None,
    )


def tokenize_messages(
    messages: Sequence[Mapping[str, Any]],
    tokenizer: Any,
    *,
    max_sequence_length: int,
    assistant_only_loss: bool = False,
    last_assistant_only_loss: bool = False,
    tools: Any | None = None,
    chat_template_kwargs: Mapping[str, Any] | None = None,
) -> TokenizedExample:
    """Tokenize messages directly through the tokenizer's chat template."""

    if max_sequence_length <= 0:
        raise ValueError("max_sequence_length must be positive.")
    assistant_loss_mode = _assistant_loss_mode(
        assistant_only_loss=assistant_only_loss,
        last_assistant_only_loss=last_assistant_only_loss,
    )
    use_assistant_mask = assistant_loss_mode is not None
    messages = _validate_messages(messages)
    has_disabled_assistant = (
        _validate_assistant_message_loss(messages) if use_assistant_mask else False
    )
    apply_chat_template = getattr(tokenizer, "apply_chat_template", None)
    if not callable(apply_chat_template):
        raise ValueError("The tokenizer must provide apply_chat_template for messages training.")
    if has_disabled_assistant and not getattr(tokenizer, "supports_message_loss", False):
        raise ValueError("The tokenizer does not support assistant message loss=false.")

    if isinstance(tools, str):
        tools = json.loads(tools)
    elif isinstance(tools, list):
        tools = [json.loads(tool) if isinstance(tool, str) else tool for tool in tools]
    template_options = dict(chat_template_kwargs or {})
    template_options.update(
        {
            "tokenize": True,
            "return_dict": True,
            "return_attention_mask": True,
            "return_assistant_tokens_mask": use_assistant_mask,
            # Assistant spans are computed on the untruncated rendering and
            # then all token fields are sliced together below. This prevents a
            # tokenizer-side truncation boundary from changing span recovery.
            "truncation": not use_assistant_mask,
            "max_length": None if use_assistant_mask else max_sequence_length,
        }
    )
    if tools is not None:
        template_options["tools"] = tools
    encoded = apply_chat_template(messages, **template_options)
    output = _normalize_tokenizer_output(
        encoded,
        require_assistant_mask=use_assistant_mask,
    )
    if not use_assistant_mask:
        return output

    if not any(output["assistant_masks"]):
        if _has_nonempty_trainable_assistant_content(messages):
            raise ValueError(
                f"{assistant_loss_mode} was requested, but the tokenized example contains no assistant tokens "
                "despite having assistant content. Check that the tokenizer chat template contains "
                "{% generation %} blocks."
            )
        return _empty_tokenized_example(output)
    if last_assistant_only_loss:
        final_assistant_disabled = next(
            (
                message.get("loss") is False
                for message in reversed(messages)
                if message.get("role") == "assistant"
            ),
            False,
        )
        output["assistant_masks"] = (
            [0] * len(output["assistant_masks"])
            if final_assistant_disabled
            else _keep_last_assistant_span(output["assistant_masks"])
        )
    return _trim_to_last_assistant_token(output, max_sequence_length=max_sequence_length)


def preprocess_example(
    example: Mapping[str, Any],
    tokenizer: Any,
    *,
    dataset_text_field: str,
    max_sequence_length: int,
    assistant_only_loss: bool = False,
    last_assistant_only_loss: bool = False,
    endprompt: EndPromptSettings | None = None,
    example_index: int = 0,
) -> TokenizedExample:
    """Preprocess one raw-text or messages example.

    The data mode is inferred from the selected field's value.  A string is raw
    text; a sequence of mappings is a conversation.  Assistant-only loss is
    intentionally rejected for raw text because no role boundary exists there.
    """

    assistant_loss_mode = _assistant_loss_mode(
        assistant_only_loss=assistant_only_loss,
        last_assistant_only_loss=last_assistant_only_loss,
    )
    if dataset_text_field not in example:
        raise KeyError(f"Dataset example does not contain field {dataset_text_field!r}.")
    value = example[dataset_text_field]
    if isinstance(value, str):
        if assistant_loss_mode is not None:
            raise ValueError(
                f"{assistant_loss_mode} is supported only for messages examples, not raw text."
            )
        if endprompt is not None:
            return tokenize_endprompt_text(
                value,
                tokenizer,
                max_sequence_length=max_sequence_length,
                settings=endprompt,
                example_index=example_index,
            )
        return tokenize_raw_text(
            value,
            tokenizer,
            max_sequence_length=max_sequence_length,
        )
    if endprompt is not None:
        return tokenize_endprompt_messages(
            value,
            tokenizer,
            max_sequence_length=max_sequence_length,
            assistant_only_loss=assistant_only_loss,
            last_assistant_only_loss=last_assistant_only_loss,
            settings=endprompt,
            example_index=example_index,
            tools=example.get("tools"),
            chat_template_kwargs=example.get("chat_template_kwargs"),
        )
    return tokenize_messages(
        value,
        tokenizer,
        max_sequence_length=max_sequence_length,
        assistant_only_loss=assistant_only_loss,
        last_assistant_only_loss=last_assistant_only_loss,
        tools=example.get("tools"),
        chat_template_kwargs=example.get("chat_template_kwargs"),
    )


def preprocess_records(
    records: Iterable[Mapping[str, Any]],
    tokenizer: Any,
    *,
    dataset_text_field: str,
    max_sequence_length: int,
    assistant_only_loss: bool = False,
    last_assistant_only_loss: bool = False,
    endprompt: EndPromptSettings | None = None,
    example_index_offset: int = 0,
    allow_empty: bool = False,
) -> list[TokenizedExample]:
    """Preprocess an iterable of examples into materialized Python records."""

    use_assistant_mask = _assistant_loss_mode(
        assistant_only_loss=assistant_only_loss,
        last_assistant_only_loss=last_assistant_only_loss,
    ) is not None

    output = [
        preprocess_example(
            example,
            tokenizer,
            dataset_text_field=dataset_text_field,
            max_sequence_length=max_sequence_length,
            assistant_only_loss=assistant_only_loss,
            last_assistant_only_loss=last_assistant_only_loss,
            endprompt=endprompt,
            example_index=index,
        )
        for index, example in enumerate(records, start=example_index_offset)
    ]
    if use_assistant_mask or endprompt is not None:
        output = [
            example
            for example in output
            if (not use_assistant_mask or _has_trainable_tokens(example))
            and (endprompt is None or _has_effective_loss_target(example))
        ]
        if not output and not allow_empty:
            raise ValueError(
                "All examples were filtered out because none contained a valid positive-weight "
                "training target within max_sequence_length. Increase the limit or filter/shorten the source data."
            )
    return output


def _scan_filter_requirement(
    dataset: Any,
    keep_example: Callable[[Mapping[str, Any]], bool],
    *,
    batch_size: int = 1000,
) -> tuple[bool | None, bool]:
    """Return ``(rewrite_required, is_empty)`` without creating a new dataset.

    Hugging Face ``Dataset.filter`` always writes a new Arrow table.  Most
    prepared shards contain no rows that need dropping, so scan their existing
    column batches first.  ``None`` means that the object cannot be scanned by
    this bounded-memory path and the caller should use its normal filter API.
    """

    if batch_size <= 0:
        raise ValueError("filter scan batch_size must be positive")
    try:
        row_count = len(dataset)
    except (TypeError, NotImplementedError):
        return None, False
    if row_count == 0:
        return False, True

    batch_iterator = getattr(dataset, "iter", None)
    if callable(batch_iterator):
        batches = batch_iterator(batch_size=batch_size)
    else:
        try:
            batches = (dataset[start : start + batch_size] for start in range(0, row_count, batch_size))
        except TypeError:
            return None, False

    rows_seen = 0
    batch_iterator = iter(batches)
    while True:
        try:
            batch = next(batch_iterator)
        except StopIteration:
            break
        except (TypeError, NotImplementedError):
            return None, False
        if not isinstance(batch, Mapping) or not batch:
            return None, False
        columns = {name: values for name, values in batch.items()}
        try:
            rows_in_batch = len(next(iter(columns.values())))
        except TypeError:
            return None, False
        if any(len(values) != rows_in_batch for values in columns.values()):
            raise ValueError("tokenized dataset returned columns with inconsistent batch lengths")
        for index in range(rows_in_batch):
            rows_seen += 1
            if not keep_example({name: values[index] for name, values in columns.items()}):
                return True, False

    if rows_seen != row_count:
        raise ValueError(
            f"tokenized dataset filter scan visited {rows_seen} rows but expected {row_count}"
        )
    return False, False


def preprocess_dataset(
    dataset: Any,
    tokenizer: Any,
    *,
    dataset_text_field: str,
    max_sequence_length: int,
    assistant_only_loss: bool = False,
    last_assistant_only_loss: bool = False,
    endprompt: EndPromptSettings | None = None,
    num_proc: int | None = None,
    example_index_offset: int = 0,
    allow_empty: bool = False,
) -> Any:
    """Preprocess a Hugging Face Dataset-like object without importing datasets.

    Plain iterables are accepted as a convenience and return a list.  Dataset-like
    objects are mapped eagerly by their own implementation, with raw columns
    removed so only model inputs remain.
    """

    use_assistant_mask = _assistant_loss_mode(
        assistant_only_loss=assistant_only_loss,
        last_assistant_only_loss=last_assistant_only_loss,
    ) is not None

    map_method = getattr(dataset, "map", None)
    if not callable(map_method):
        return preprocess_records(
            dataset,
            tokenizer,
            dataset_text_field=dataset_text_field,
            max_sequence_length=max_sequence_length,
            assistant_only_loss=assistant_only_loss,
            last_assistant_only_loss=last_assistant_only_loss,
            endprompt=endprompt,
            example_index_offset=example_index_offset,
            allow_empty=allow_empty,
        )

    def tokenize_one(example: Mapping[str, Any], index: int = 0) -> TokenizedExample:
        return preprocess_example(
            example,
            tokenizer,
            dataset_text_field=dataset_text_field,
            max_sequence_length=max_sequence_length,
            assistant_only_loss=assistant_only_loss,
            last_assistant_only_loss=last_assistant_only_loss,
            endprompt=endprompt,
            example_index=example_index_offset + index,
        )

    map_kwargs: dict[str, Any] = {}
    column_names = getattr(dataset, "column_names", None)
    if column_names is not None:
        map_kwargs["remove_columns"] = list(column_names)
    if num_proc is not None:
        map_kwargs["num_proc"] = num_proc
    if endprompt is not None or example_index_offset:
        map_kwargs["with_indices"] = True
    # Empty assistant-only examples are represented by empty lists until the
    # post-map filter runs.  Without an explicit Arrow schema, a worker whose
    # first batch (or entire shard) is empty can infer ``list<null>`` and then
    # fail when it encounters integer/float rows.  Limit this Datasets-specific
    # option to actual Hugging Face objects so lightweight Dataset-like test
    # doubles remain supported without accepting an extra keyword.
    is_hf_dataset = False
    if dataset.__class__.__module__.split(".", 1)[0] == "datasets":
        from datasets import Dataset, Features, Sequence as DatasetSequence, Value

        if not isinstance(dataset, Dataset):
            raise TypeError(
                "preprocess_dataset requires a materialized datasets.Dataset; "
                "streaming datasets and DatasetDict objects are not supported"
            )
        is_hf_dataset = True

        output_features: dict[str, Any] = {
            "input_ids": DatasetSequence(Value("int64")),
            "attention_mask": DatasetSequence(Value("int64")),
        }
        if use_assistant_mask:
            output_features["assistant_masks"] = DatasetSequence(Value("int64"))
        if endprompt is not None:
            output_features["position_ids"] = DatasetSequence(Value("int64"))
            output_features["loss_weights"] = DatasetSequence(Value("float32"))
        map_kwargs["features"] = Features(output_features)
        # Tokenized data is consumed only by this run.  Keeping every
        # transformation in memory avoids writing per-worker Arrow cache files
        # that cannot be reused across the TPU hosts.
        map_kwargs["keep_in_memory"] = True
        map_kwargs["load_from_cache_file"] = False
    tokenized = map_method(tokenize_one, **map_kwargs)
    if not use_assistant_mask and endprompt is None:
        return tokenized

    def keep_example(example: Mapping[str, Any]) -> bool:
        return (
            (not use_assistant_mask or _has_trainable_tokens(example))
            and (endprompt is None or _has_effective_loss_target(example))
        )

    filter_method = getattr(tokenized, "filter", None)
    if callable(filter_method):
        rewrite_required = None
        is_empty = False
        if is_hf_dataset:
            rewrite_required, is_empty = _scan_filter_requirement(tokenized, keep_example)
            if rewrite_required is False:
                if is_empty and not allow_empty:
                    raise ValueError(
                        "All examples were filtered out because none contained a valid positive-weight "
                        "training target within max_sequence_length. Increase the limit or filter/shorten "
                        "the source data."
                    )
                return tokenized

        filter_kwargs: dict[str, Any] = {}
        if num_proc is not None:
            filter_kwargs["num_proc"] = num_proc
        if is_hf_dataset:
            filter_kwargs["keep_in_memory"] = True
            filter_kwargs["load_from_cache_file"] = False

        tokenized = filter_method(keep_example, **filter_kwargs)
        try:
            is_empty = len(tokenized) == 0
        except TypeError:
            is_empty = False
    elif isinstance(tokenized, Iterable):
        tokenized = [example for example in tokenized if keep_example(example)]
        is_empty = not tokenized
    else:
        raise TypeError("mapped dataset must provide filter() or return an iterable of tokenized records")
    if is_empty and not allow_empty:
        raise ValueError(
            "All examples were filtered out because none contained a valid positive-weight "
            "training target within max_sequence_length. Increase the limit or filter/shorten the source data."
        )
    return tokenized


__all__ = [
    "EndPromptSettings",
    "TokenizedExample",
    "preprocess_dataset",
    "preprocess_example",
    "preprocess_records",
    "tokenize_messages",
    "tokenize_endprompt_messages",
    "tokenize_endprompt_text",
    "tokenize_raw_text",
]
