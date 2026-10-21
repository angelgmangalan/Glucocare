"""Small, session-only conversation history helpers."""

from __future__ import annotations

from typing import Any


def trim_history(
    messages: list[dict[str, Any]],
    max_messages: int,
) -> list[dict[str, str]]:
    if isinstance(max_messages, bool) or not isinstance(max_messages, int) or max_messages < 1:
        raise ValueError("max_messages must be a positive integer.")
    trimmed = []
    for message in messages[-max_messages:]:
        if (
            isinstance(message, dict)
            and message.get("role") in {"user", "assistant"}
            and isinstance(message.get("content"), str)
            and message["content"].strip()
        ):
            trimmed.append({"role": message["role"], "content": message["content"]})
    return trimmed


def append_exchange(
    messages: list[dict[str, Any]],
    question: str,
    response: str,
    *,
    max_messages: int,
    sources: list[dict[str, Any]] | None = None,
    context_used: dict[str, bool] | None = None,
) -> list[dict[str, Any]]:
    updated = list(messages)
    updated.extend([
        {"role": "user", "content": question},
        {
            "role": "assistant",
            "content": response,
            "sources": sources or [],
            "context_used": context_used or {},
        },
    ])
    return updated[-max_messages:]
