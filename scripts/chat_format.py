"""Versioned chat formatting with explicit assistant-only supervision.

The formatter deliberately works with SentencePiece directly instead of relying
on a Transformers tokenizer chat template.  Training and inference therefore
share the exact same token construction even before the final tokenizer has
atomic role-control tokens.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np


IGNORE_INDEX = -100
CHAT_TEMPLATE_NAME = "gpt-chatml"
CHAT_TEMPLATE_VERSION = 1
ROLE_MARKERS = {
    "system": "<|system|>",
    "user": "<|user|>",
    "assistant": "<|assistant|>",
}
END_MARKER = "<|end|>"


class ChatFormatError(ValueError):
    """Raised when a conversation cannot be represented safely."""


@dataclass(frozen=True)
class EncodedConversation:
    input_ids: tuple[int, ...]
    labels: tuple[int, ...]
    assistant_spans: tuple[tuple[int, int], ...]

    @property
    def supervised_tokens(self) -> int:
        return sum(stop - start for start, stop in self.assistant_spans)


def validate_messages(
    messages: Sequence[Mapping[str, Any]],
    *,
    require_final_assistant: bool = True,
) -> None:
    if not isinstance(messages, Sequence) or isinstance(messages, (str, bytes)):
        raise ChatFormatError("messages must be a sequence")
    if not messages:
        raise ChatFormatError("conversation must contain at least one message")

    previous_role: str | None = None
    saw_user = False
    for index, message in enumerate(messages):
        if not isinstance(message, Mapping):
            raise ChatFormatError(f"message {index} must be a mapping")
        role = message.get("role")
        content = message.get("content")
        if role not in ROLE_MARKERS:
            raise ChatFormatError(f"message {index} has unsupported role {role!r}")
        if not isinstance(content, str) or not content.strip():
            raise ChatFormatError(f"message {index} content must be non-empty text")
        if role == "system" and index != 0:
            raise ChatFormatError("system message is only allowed at index 0")
        if role == "system" and previous_role is not None:
            raise ChatFormatError("only one leading system message is allowed")
        if role == previous_role:
            raise ChatFormatError(f"adjacent {role!r} messages are not allowed")
        if role == "assistant" and not saw_user:
            raise ChatFormatError("assistant message must follow a user message")
        if role == "user":
            saw_user = True
        previous_role = role

    if not saw_user:
        raise ChatFormatError("conversation must contain a user message")
    if require_final_assistant and messages[-1].get("role") != "assistant":
        raise ChatFormatError("training conversation must end with assistant")


def _encode(tokenizer: Any, text: str) -> list[int]:
    return list(tokenizer.encode(text, out_type=int))


def _append(
    token_ids: list[int],
    labels: list[int],
    values: Sequence[int],
    *,
    supervised: bool,
) -> tuple[int, int]:
    start = len(token_ids)
    encoded = [int(value) for value in values]
    token_ids.extend(encoded)
    labels.extend(encoded if supervised else [IGNORE_INDEX] * len(encoded))
    return start, len(token_ids)


def encode_conversation(
    tokenizer: Any,
    messages: Sequence[Mapping[str, Any]],
    *,
    add_bos: bool = True,
    require_final_assistant: bool = True,
) -> EncodedConversation:
    """Encode messages and supervise assistant content plus its end marker."""

    validate_messages(messages, require_final_assistant=require_final_assistant)
    token_ids: list[int] = []
    labels: list[int] = []
    assistant_spans: list[tuple[int, int]] = []

    bos_id = int(tokenizer.bos_id())
    if add_bos and bos_id >= 0:
        _append(token_ids, labels, [bos_id], supervised=False)

    for message in messages:
        role = str(message["role"])
        content = str(message["content"]).strip()
        _append(
            token_ids,
            labels,
            _encode(tokenizer, f"{ROLE_MARKERS[role]}\n"),
            supervised=False,
        )
        content_tokens = _encode(tokenizer, content + "\n")
        end_tokens = _encode(tokenizer, END_MARKER + "\n")
        if role == "assistant":
            start, _ = _append(
                token_ids,
                labels,
                content_tokens,
                supervised=True,
            )
            _, stop = _append(
                token_ids,
                labels,
                end_tokens,
                supervised=True,
            )
            assistant_spans.append((start, stop))
        else:
            _append(token_ids, labels, content_tokens, supervised=False)
            _append(token_ids, labels, end_tokens, supervised=False)

    if require_final_assistant and not assistant_spans:
        raise ChatFormatError("conversation has no supervised assistant tokens")
    return EncodedConversation(
        input_ids=tuple(token_ids),
        labels=tuple(labels),
        assistant_spans=tuple(assistant_spans),
    )


def encode_generation_prompt(
    tokenizer: Any,
    messages: Sequence[Mapping[str, Any]],
    *,
    add_bos: bool = True,
) -> tuple[int, ...]:
    """Encode a conversation ending in user input and append assistant header."""

    validate_messages(messages, require_final_assistant=False)
    if messages[-1].get("role") != "user":
        raise ChatFormatError("generation prompt must end with a user message")
    encoded = encode_conversation(
        tokenizer,
        messages,
        add_bos=add_bos,
        require_final_assistant=False,
    )
    values = list(encoded.input_ids)
    values.extend(_encode(tokenizer, f"{ROLE_MARKERS['assistant']}\n"))
    return tuple(values)


def labels_to_spans(labels: Sequence[int]) -> tuple[tuple[int, int], ...]:
    """Return half-open supervised spans from an ignore-index label vector."""

    spans: list[tuple[int, int]] = []
    start: int | None = None
    for index, value in enumerate(labels):
        supervised = int(value) != IGNORE_INDEX
        if supervised and start is None:
            start = index
        elif not supervised and start is not None:
            spans.append((start, index))
            start = None
    if start is not None:
        spans.append((start, len(labels)))
    return tuple(spans)


def apply_supervision_spans(
    token_ids: np.ndarray,
    *,
    document_start: int,
    spans: Sequence[Sequence[int]],
) -> np.ndarray:
    """Build labels for a sampled document window using absolute token spans."""

    labels = np.full(token_ids.shape, IGNORE_INDEX, dtype=np.int64)
    window_stop = document_start + int(token_ids.size)
    for raw_span in spans:
        if len(raw_span) != 2:
            raise ChatFormatError("each supervision span must contain start and stop")
        span_start, span_stop = int(raw_span[0]), int(raw_span[1])
        if span_start < 0 or span_stop <= span_start:
            raise ChatFormatError("invalid supervision span")
        start = max(span_start, document_start)
        stop = min(span_stop, window_stop)
        if start < stop:
            local_start = start - document_start
            local_stop = stop - document_start
            labels[local_start:local_stop] = token_ids[local_start:local_stop]
    return labels


def template_metadata() -> dict[str, Any]:
    return {
        "name": CHAT_TEMPLATE_NAME,
        "version": CHAT_TEMPLATE_VERSION,
        "role_markers": dict(ROLE_MARKERS),
        "end_marker": END_MARKER,
        "assistant_only_loss": True,
    }
