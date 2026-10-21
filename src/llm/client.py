"""LLM client facade used by the chatbot."""

from __future__ import annotations

from src.llm.config import LLMConfigurationError, load_settings
from src.llm.provider import ChatProvider, LLMProviderError, OpenAICompatibleProvider


class LLMError(RuntimeError):
    """Safe client-level configuration or provider failure."""


class LLMClient:
    def __init__(
        self,
        provider: ChatProvider,
        *,
        temperature: float = 0.2,
        max_tokens: int = 700,
    ) -> None:
        self.provider = provider
        self.temperature = temperature
        self.max_tokens = max_tokens

    @classmethod
    def from_environment(cls) -> "LLMClient":
        try:
            settings = load_settings()
        except LLMConfigurationError as error:
            raise LLMError(str(error)) from error
        if not settings.is_configured:
            raise LLMError(
                "LLM service is not configured. Please configure the API key in the environment."
            )
        return cls(
            OpenAICompatibleProvider(settings),
            temperature=settings.temperature,
            max_tokens=settings.max_tokens,
        )

    def generate_response(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        selected_temperature = self.temperature if temperature is None else temperature
        selected_max_tokens = self.max_tokens if max_tokens is None else max_tokens
        try:
            return self.provider.generate_response(
                messages,
                temperature=selected_temperature,
                max_tokens=selected_max_tokens,
            )
        except LLMProviderError as error:
            raise LLMError(str(error)) from error
