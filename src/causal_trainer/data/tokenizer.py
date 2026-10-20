"""Small tokenizer/chat-template backend used by the training pipeline.

Only the capabilities needed for raw-text and messages training are exposed.
The serialized ``tokenizer.json`` is executed by the Rust ``tokenizers``
backend; repository Python files are never imported.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Any

import jinja2
from jinja2.ext import Extension
from jinja2.sandbox import ImmutableSandboxedEnvironment
from tokenizers import Encoding, Tokenizer


ChatTemplate = str | dict[str, str] | None


class _AssistantTracker(Extension):
    """Track character spans rendered inside ``generation`` blocks."""

    tags = {"generation"}

    def __init__(self, environment: ImmutableSandboxedEnvironment):
        super().__init__(environment)
        environment.extend(activate_tracker=self.activate_tracker)
        self._rendered_blocks: list[str] | None = None
        self._generation_indices: list[tuple[int, int]] | None = None
        self._generation_loss_flags: tuple[bool, ...] | None = None
        self._generation_loss_index = 0

    def parse(self, parser: Any) -> jinja2.nodes.CallBlock:
        lineno = next(parser.stream).lineno
        body = parser.parse_statements(["name:endgeneration"], drop_needle=True)
        return jinja2.nodes.CallBlock(
            self.call_method("_generation_support"),
            [],
            [],
            body,
        ).set_lineno(lineno)

    @jinja2.pass_eval_context
    def _generation_support(self, context: Any, caller: Any) -> str:
        del context
        rendered = caller()
        keep_span = True
        if self._generation_loss_flags is not None:
            if self._generation_loss_index >= len(self._generation_loss_flags):
                raise ValueError(
                    "message-level loss requires exactly one generation block "
                    "per assistant message"
                )
            keep_span = self._generation_loss_flags[self._generation_loss_index]
            self._generation_loss_index += 1
        if self._rendered_blocks is not None and self._generation_indices is not None:
            start = len("".join(self._rendered_blocks))
            if keep_span:
                self._generation_indices.append((start, start + len(rendered)))
        return rendered

    @contextmanager
    def activate_tracker(
        self,
        rendered_blocks: list[str],
        generation_indices: list[tuple[int, int]],
        generation_loss_flags: Sequence[bool] | None = None,
    ):
        if (
            self._rendered_blocks is not None
            or self._generation_indices is not None
            or self._generation_loss_flags is not None
        ):
            raise RuntimeError("assistant span tracker is already active")
        self._rendered_blocks = rendered_blocks
        self._generation_indices = generation_indices
        self._generation_loss_flags = (
            None if generation_loss_flags is None else tuple(generation_loss_flags)
        )
        self._generation_loss_index = 0
        try:
            yield
            if (
                self._generation_loss_flags is not None
                and self._generation_loss_index != len(self._generation_loss_flags)
            ):
                raise ValueError(
                    "message-level loss requires exactly one generation block "
                    "per assistant message"
                )
        finally:
            self._rendered_blocks = None
            self._generation_indices = None
            self._generation_loss_flags = None
            self._generation_loss_index = 0


def _raise_template_exception(message: str) -> None:
    raise jinja2.exceptions.TemplateError(message)


def _to_json(
    value: Any,
    ensure_ascii: bool = False,
    indent: int | None = None,
    separators: tuple[str, str] | None = None,
    sort_keys: bool = False,
) -> str:
    return json.dumps(
        value,
        ensure_ascii=ensure_ascii,
        indent=indent,
        separators=separators,
        sort_keys=sort_keys,
    )


def _strftime_now(format_string: str) -> str:
    return datetime.now().strftime(format_string)


@lru_cache(maxsize=128)
def _compile_chat_template(source: str) -> jinja2.Template:
    environment = ImmutableSandboxedEnvironment(
        trim_blocks=True,
        lstrip_blocks=True,
        extensions=[_AssistantTracker, jinja2.ext.loopcontrols],
    )
    environment.filters["tojson"] = _to_json
    environment.globals["raise_exception"] = _raise_template_exception
    environment.globals["strftime_now"] = _strftime_now
    return environment.from_string(source)


def _read_json(path: Path) -> Any:
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid JSON in {path}: {error}") from error


def _read_json_object(path: Path) -> dict[str, Any]:
    value = _read_json(path)
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _normalize_chat_template(value: Any, *, source: str) -> ChatTemplate:
    if value is None or isinstance(value, str):
        return value
    if isinstance(value, Mapping):
        if set(value) == {"chat_template"}:
            return _normalize_chat_template(value["chat_template"], source=source)
        templates: dict[str, str] = {}
        for name, template in value.items():
            if not isinstance(name, str) or not isinstance(template, str):
                raise ValueError(f"{source} chat template mapping must contain string names and templates")
            templates[name] = template
        return templates
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        templates = {}
        for entry in value:
            if not isinstance(entry, Mapping):
                raise ValueError(f"{source} chat template list entries must be mappings")
            name = entry.get("name")
            template = entry.get("template")
            if not isinstance(name, str) or not isinstance(template, str):
                raise ValueError(f"{source} chat template entries require string name/template fields")
            templates[name] = template
        return templates
    raise ValueError(f"unsupported chat template value in {source}")


def _normalize_special_token(value: Any, *, name: str) -> str | list[str]:
    if isinstance(value, str):
        return value
    if isinstance(value, Mapping) and isinstance(value.get("content"), str):
        return value["content"]
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        normalized = [_normalize_special_token(item, name=name) for item in value]
        if any(not isinstance(item, str) for item in normalized):
            raise ValueError(f"special token {name!r} must be a string or list of strings")
        return normalized
    raise ValueError(f"special token {name!r} has an unsupported serialized value")


def _special_tokens_from_config(config: Mapping[str, Any]) -> dict[str, str | list[str]]:
    tokens: dict[str, str | list[str]] = {}
    for name, value in config.items():
        if name.endswith("_token") or name == "additional_special_tokens":
            if value is not None:
                tokens[name] = _normalize_special_token(value, name=name)
    extra = config.get("extra_special_tokens")
    if extra is not None:
        if not isinstance(extra, Mapping):
            raise ValueError("extra_special_tokens must be a mapping")
        for name, value in extra.items():
            if not isinstance(name, str):
                raise ValueError("extra special-token names must be strings")
            tokens[name] = _normalize_special_token(value, name=name)
    return tokens


class TrainingTokenizer:
    """Tokenizer interface intentionally limited to the trainer's data path."""

    supports_message_loss = True

    def __init__(
        self,
        backend: Tokenizer,
        *,
        chat_template: ChatTemplate = None,
        special_tokens: Mapping[str, str | list[str]] | None = None,
    ) -> None:
        self._backend = backend
        self.chat_template = chat_template
        self.special_tokens_map = dict(special_tokens or {})
        self._special_token_id_overrides: dict[str, int] = {}

    @classmethod
    def from_directory(cls, source: str | Path) -> TrainingTokenizer:
        directory = Path(source)
        tokenizer_path = directory / "tokenizer.json"
        if not tokenizer_path.is_file():
            raise FileNotFoundError(
                f"missing {tokenizer_path}; this trainer requires a serialized fast tokenizer.json"
            )
        backend = Tokenizer.from_file(str(tokenizer_path))

        tokenizer_config: dict[str, Any] = {}
        tokenizer_config_path = directory / "tokenizer_config.json"
        if tokenizer_config_path.is_file():
            tokenizer_config = _read_json_object(tokenizer_config_path)

        special_tokens: dict[str, str | list[str]] = {}
        special_tokens_path = directory / "special_tokens_map.json"
        if special_tokens_path.is_file():
            special_tokens.update(_special_tokens_from_config(_read_json_object(special_tokens_path)))
        special_tokens.update(_special_tokens_from_config(tokenizer_config))

        chat_template = _normalize_chat_template(
            tokenizer_config.get("chat_template"),
            source=str(tokenizer_config_path),
        )
        legacy_template_path = directory / "chat_template.json"
        if chat_template is None and legacy_template_path.is_file():
            chat_template = _normalize_chat_template(
                _read_json(legacy_template_path),
                source=str(legacy_template_path),
            )

        # Standalone Jinja files take priority over tokenizer_config.json.
        file_templates: dict[str, str] = {}
        default_template_path = directory / "chat_template.jinja"
        if default_template_path.is_file():
            file_templates["default"] = default_template_path.read_text(encoding="utf-8")
        template_directory = directory / "chat_templates"
        if template_directory.is_dir():
            for path in sorted(template_directory.glob("*.jinja")):
                if path.is_file():
                    file_templates[path.stem] = path.read_text(encoding="utf-8")
        if file_templates:
            chat_template = (
                file_templates["default"]
                if set(file_templates) == {"default"}
                else file_templates
            )

        return cls(
            backend,
            chat_template=chat_template,
            special_tokens=special_tokens,
        )

    def __len__(self) -> int:
        return int(self._backend.get_vocab_size(with_added_tokens=True))

    def get_vocab(self) -> dict[str, int]:
        return self._backend.get_vocab(with_added_tokens=True)

    def decode(self, ids: list[int]) -> str:
        return self._backend.decode(ids, skip_special_tokens=False)

    def _special_token_id(self, name: str) -> int | None:
        override = self._special_token_id_overrides.get(name)
        if override is not None:
            return override
        token = self.special_tokens_map.get(name)
        if not isinstance(token, str):
            return None
        token_id = self._backend.token_to_id(token)
        return None if token_id is None else int(token_id)

    def _set_special_token_id(self, name: str, value: int | None) -> None:
        if value is None:
            self._special_token_id_overrides.pop(name, None)
            return
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{name}_id must be a non-negative integer")
        token = self._backend.id_to_token(value)
        if token is None:
            raise ValueError(f"{name}_id={value} is not present in tokenizer.json")
        self._special_token_id_overrides[name] = value
        self.special_tokens_map[name] = token

    @property
    def pad_token_id(self) -> int | None:
        return self._special_token_id("pad_token")

    @pad_token_id.setter
    def pad_token_id(self, value: int | None) -> None:
        self._set_special_token_id("pad_token", value)

    @property
    def eos_token_id(self) -> int | None:
        return self._special_token_id("eos_token")

    @eos_token_id.setter
    def eos_token_id(self, value: int | None) -> None:
        self._set_special_token_id("eos_token", value)

    def validate_model_vocabulary(self, model_vocab_size: int) -> None:
        """Ensure every tokenizer ID can index the fixed embedding table."""

        if isinstance(model_vocab_size, bool) or not isinstance(model_vocab_size, int) or model_vocab_size <= 0:
            raise ValueError("model_vocab_size must be a positive integer")
        vocabulary = self._backend.get_vocab(with_added_tokens=True)
        largest_id = max((int(token_id) for token_id in vocabulary.values()), default=-1)
        if largest_id >= model_vocab_size:
            raise ValueError(
                f"tokenizer contains token id {largest_id}, outside model vocab_size={model_vocab_size}"
            )

    @staticmethod
    def _truncate_encoding(encoding: Encoding, *, truncation: bool, max_length: int | None) -> Encoding:
        if not truncation:
            return encoding
        if isinstance(max_length, bool) or not isinstance(max_length, int) or max_length <= 0:
            raise ValueError("truncation requires a positive max_length")
        encoding.truncate(max_length, stride=0, direction="right")
        return encoding

    def _encode(
        self,
        text: str,
        *,
        add_special_tokens: bool,
        truncation: bool,
        max_length: int | None,
    ) -> Encoding:
        if not isinstance(text, str):
            raise TypeError("the training tokenizer accepts one string at a time")
        if add_special_tokens:
            raise ValueError("automatic special-token insertion is unsupported; templates must add tokens explicitly")
        encoding = self._backend.encode(text, add_special_tokens=False)
        return self._truncate_encoding(encoding, truncation=truncation, max_length=max_length)

    def __call__(
        self,
        text: str,
        *,
        add_special_tokens: bool = False,
        return_attention_mask: bool = True,
        truncation: bool = False,
        max_length: int | None = None,
        **kwargs: Any,
    ) -> dict[str, list[int]]:
        if kwargs:
            raise TypeError(f"unsupported tokenizer arguments: {', '.join(sorted(kwargs))}")
        encoding = self._encode(
            text,
            add_special_tokens=add_special_tokens,
            truncation=truncation,
            max_length=max_length,
        )
        output = {"input_ids": list(encoding.ids)}
        if return_attention_mask:
            output["attention_mask"] = list(encoding.attention_mask)
        return output

    def get_chat_template(
        self,
        chat_template: str | None = None,
        tools: Sequence[Mapping[str, Any]] | None = None,
    ) -> str:
        configured = self.chat_template
        if isinstance(configured, dict):
            if chat_template is not None and chat_template in configured:
                return configured[chat_template]
            if chat_template is None:
                if tools is not None and "tool_use" in configured:
                    return configured["tool_use"]
                if "default" in configured:
                    return configured["default"]
                names = sorted(configured)
                raise ValueError(f"multiple chat templates are available but none is default: {names}")
        elif chat_template is None and configured is not None:
            return configured
        if chat_template is None:
            raise ValueError("messages training requires a chat template")
        return chat_template

    @staticmethod
    def _render_with_assistant_indices(
        template: jinja2.Template,
        template_kwargs: Mapping[str, Any],
        *,
        generation_loss_flags: Sequence[bool] | None = None,
    ) -> tuple[str, list[tuple[int, int]]]:
        rendered_blocks: list[str] = []
        generation_indices: list[tuple[int, int]] = []
        with template.environment.activate_tracker(
            rendered_blocks,
            generation_indices,
            generation_loss_flags,
        ):
            for block in template.generate(**template_kwargs):
                rendered_blocks.append(block)
        return "".join(rendered_blocks), generation_indices

    def apply_chat_template(
        self,
        conversation: Sequence[Mapping[str, Any]],
        tools: Sequence[Mapping[str, Any]] | None = None,
        documents: Sequence[Mapping[str, Any]] | None = None,
        chat_template: str | None = None,
        add_generation_prompt: bool = False,
        continue_final_message: bool | str = False,
        tokenize: bool = True,
        padding: bool | str = False,
        truncation: bool = False,
        max_length: int | None = None,
        return_tensors: str | None = None,
        return_dict: bool = True,
        return_assistant_tokens_mask: bool = False,
        tokenizer_kwargs: Mapping[str, Any] | None = None,
        **kwargs: Any,
    ) -> str | list[int] | dict[str, list[int]]:
        if not isinstance(conversation, Sequence) or isinstance(conversation, (str, bytes)) or not conversation:
            raise ValueError("conversation must be a non-empty sequence of message mappings")
        if any(not isinstance(message, Mapping) for message in conversation):
            raise ValueError("batched conversations are not supported by the training tokenizer")
        if continue_final_message:
            raise ValueError("continue_final_message is outside the training tokenizer contract")
        if padding not in (False, "do_not_pad"):
            raise ValueError("chat-template padding is handled by the trainer, not the tokenizer")
        if return_tensors is not None:
            raise ValueError("the training tokenizer returns Python lists only")
        if return_assistant_tokens_mask and not (tokenize and return_dict):
            raise ValueError("return_assistant_tokens_mask requires tokenize=True and return_dict=True")
        if tokenizer_kwargs:
            unsupported = set(tokenizer_kwargs) - {"return_attention_mask"}
            if unsupported:
                raise TypeError(f"unsupported tokenizer_kwargs: {', '.join(sorted(unsupported))}")

        normalized_tools: list[dict[str, Any]] | None = None
        if tools is not None:
            if not isinstance(tools, Sequence) or isinstance(tools, (str, bytes)):
                raise ValueError("tools must be a sequence of JSON-schema mappings")
            normalized_tools = []
            for tool in tools:
                if not isinstance(tool, Mapping):
                    raise ValueError("tools must contain JSON-schema mappings; Python callables are unsupported")
                normalized_tools.append(dict(tool))
        if documents is not None and any(not isinstance(document, Mapping) for document in documents):
            raise ValueError("documents must contain mappings")

        source = self.get_chat_template(chat_template, normalized_tools)
        template = _compile_chat_template(source)
        assistant_loss_flags = [
            message.get("loss") is not False
            for message in conversation
            if message.get("role") == "assistant"
        ]
        generation_loss_flags = (
            assistant_loss_flags if False in assistant_loss_flags else None
        )
        template_values = {
            **self.special_tokens_map,
            **kwargs,
            "messages": list(conversation),
            "tools": normalized_tools,
            "documents": None if documents is None else list(documents),
            "add_generation_prompt": add_generation_prompt,
        }
        if return_assistant_tokens_mask:
            rendered, generation_indices = self._render_with_assistant_indices(
                template,
                template_values,
                generation_loss_flags=generation_loss_flags,
            )
        else:
            rendered = template.render(**template_values)
            generation_indices = []
        if not tokenize:
            return rendered

        encoding = self._encode(
            rendered,
            add_special_tokens=False,
            truncation=truncation,
            max_length=max_length,
        )
        input_ids = list(encoding.ids)
        if not return_dict:
            return input_ids
        output = {
            "input_ids": input_ids,
            "attention_mask": list(encoding.attention_mask),
        }
        if return_assistant_tokens_mask:
            assistant_mask = [0] * len(input_ids)
            for start_char, end_char in generation_indices:
                if end_char <= start_char:
                    continue
                # Use interval overlap instead of only mapping the final
                # character. A generation block may end in whitespace that a
                # tokenizer drops; treating an unmapped end as end-of-sequence
                # would incorrectly supervise every later user/tool token.
                for token_index, (token_start, token_end) in enumerate(encoding.offsets):
                    if token_end > token_start and token_start < end_char and token_end > start_char:
                        assistant_mask[token_index] = 1
            output["assistant_masks"] = assistant_mask
        return output


__all__ = ["TrainingTokenizer"]
