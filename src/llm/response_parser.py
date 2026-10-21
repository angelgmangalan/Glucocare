"""Validation and extraction of chat-completions API responses."""

from __future__ import annotations

from typing import Any


class LLMResponseError(RuntimeError):
    """Raised when a provider response has an unexpected shape or no content."""


def parse_chat_completion(payload: Any) -> str:
    if not isinstance(payload, dict):
        raise LLMResponseError("The LLM service returned an unexpected response.")
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        raise LLMResponseError("The LLM service returned no response choices.")
    first = choices[0]
    if not isinstance(first, dict):
        raise LLMResponseError("The LLM service returned an unexpected response.")
    message = first.get("message")
    if not isinstance(message, dict):
        raise LLMResponseError("The LLM service returned an unexpected response.")
    content = message.get("content")
    if isinstance(content, str):
        answer = content.strip()
    elif isinstance(content, list):
        answer = "".join(
            item.get("text", "")
            for item in content
            if isinstance(item, dict) and isinstance(item.get("text"), str)
        ).strip()
    else:
        answer = ""
    if not answer:
        raise LLMResponseError("The LLM service returned an empty response.")
    return answer
