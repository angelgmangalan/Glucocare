"""Environment-backed settings for the OpenAI-compatible chat API."""

from __future__ import annotations

import os
from dataclasses import dataclass
from urllib.parse import urlparse

from src.config import DEFAULT_TOP_K


class LLMConfigurationError(ValueError):
    """Raised when LLM configuration is missing or invalid."""


def _positive_int(name: str, default: int) -> int:
    raw = os.getenv(name, str(default))
    try:
        value = int(raw)
    except ValueError as error:
        raise LLMConfigurationError(f"{name} must be a positive integer.") from error
    if value < 1:
        raise LLMConfigurationError(f"{name} must be a positive integer.")
    return value


def _temperature() -> float:
    raw = os.getenv("LLM_TEMPERATURE", "0.2")
    try:
        value = float(raw)
    except ValueError as error:
        raise LLMConfigurationError("LLM_TEMPERATURE must be between 0 and 2.") from error
    if not 0 <= value <= 2:
        raise LLMConfigurationError("LLM_TEMPERATURE must be between 0 and 2.")
    return value


def load_rag_top_k() -> int:
    return _positive_int("RAG_TOP_K", DEFAULT_TOP_K)


@dataclass(frozen=True)
class LLMSettings:
    api_key: str
    model: str
    base_url: str
    temperature: float
    max_tokens: int
    timeout_seconds: int
    max_history_messages: int
    rag_top_k: int

    @property
    def is_configured(self) -> bool:
        return bool(self.api_key and self.model)


def load_settings() -> LLMSettings:
    api_key = os.getenv("LLM_API_KEY", "").strip()
    model = os.getenv("LLM_MODEL", "gpt-4o-mini").strip()
    base_url = os.getenv("LLM_BASE_URL", "https://api.openai.com/v1").strip().rstrip("/")
    parsed = urlparse(base_url)
    if (
        not parsed.scheme
        or not parsed.netloc
        or parsed.username
        or parsed.password
        or parsed.query
        or parsed.fragment
        or (parsed.scheme != "https" and parsed.hostname not in {"localhost", "127.0.0.1"})
    ):
        raise LLMConfigurationError(
            "LLM_BASE_URL must be an HTTPS API URL (HTTP is allowed for localhost only)."
        )
    if not model:
        raise LLMConfigurationError("LLM_MODEL must not be empty.")
    return LLMSettings(
        api_key=api_key,
        model=model,
        base_url=base_url,
        temperature=_temperature(),
        max_tokens=_positive_int("LLM_MAX_TOKENS", 700),
        timeout_seconds=_positive_int("LLM_TIMEOUT_SECONDS", 30),
        max_history_messages=_positive_int("MAX_HISTORY_MESSAGES", 10),
        rag_top_k=load_rag_top_k(),
    )
