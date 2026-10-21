"""Provider protocol and OpenAI-compatible Chat Completions adapter."""

from __future__ import annotations

import json
import socket
from typing import Any, Protocol
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from src.llm.config import LLMSettings
from src.llm.response_parser import LLMResponseError, parse_chat_completion


class LLMProviderError(RuntimeError):
    """A safe, user-facing provider error with no request secrets."""


class ChatProvider(Protocol):
    def generate_response(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float,
        max_tokens: int,
    ) -> str: ...


class OpenAICompatibleProvider:
    def __init__(self, settings: LLMSettings) -> None:
        self.settings = settings

    @staticmethod
    def _rate_limit_message(error: HTTPError) -> str:
        try:
            payload: Any = json.loads(error.read().decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError, OSError):
            payload = None
        finally:
            error.close()

        provider_error = payload.get("error") if isinstance(payload, dict) else None
        if isinstance(provider_error, dict):
            error_code = provider_error.get("code") or provider_error.get("type")
            if isinstance(error_code, str) and error_code in {
                "insufficient_quota",
                "billing_hard_limit_reached",
            }:
                return (
                    "The LLM provider has no available quota. Check the account billing "
                    "and usage limits, then try again."
                )
        return (
            "The LLM service is rate limited. Please try again shortly; if the problem "
            "continues, check the provider's usage limits."
        )

    def generate_response(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float,
        max_tokens: int,
    ) -> str:
        if not self.settings.api_key:
            raise LLMProviderError(
                "LLM service is not configured. Please configure the API key in the environment."
            )
        endpoint = self.settings.base_url
        if not endpoint.endswith("/chat/completions"):
            endpoint += "/chat/completions"
        body = json.dumps({
            "model": self.settings.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }).encode("utf-8")
        request = Request(
            endpoint,
            data=body,
            headers={
                "Authorization": f"Bearer {self.settings.api_key}",
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
            method="POST",
        )
        try:
            with urlopen(request, timeout=self.settings.timeout_seconds) as response:
                payload: Any = json.loads(response.read().decode("utf-8"))
        except HTTPError as error:
            if error.code in {401, 403}:
                raise LLMProviderError("The LLM service rejected the configured API credentials.") from error
            if error.code == 429:
                raise LLMProviderError(self._rate_limit_message(error)) from error
            if error.code in {408, 504}:
                raise LLMProviderError("The LLM service timed out. Please try again.") from error
            if error.code == 404:
                raise LLMProviderError("The LLM endpoint or model is not available; check the API configuration.") from error
            if 500 <= error.code < 600:
                raise LLMProviderError("The LLM service is temporarily unavailable. Please try again.") from error
            raise LLMProviderError("The LLM service rejected the request; check its configuration.") from error
        except (TimeoutError, socket.timeout) as error:
            raise LLMProviderError("The LLM service timed out. Please try again.") from error
        except URLError as error:
            if isinstance(error.reason, (TimeoutError, socket.timeout)):
                raise LLMProviderError("The LLM service timed out. Please try again.") from error
            raise LLMProviderError("Unable to connect to the LLM service. Check the network and try again.") from error
        except OSError as error:
            raise LLMProviderError("Unable to connect to the LLM service. Check the network and try again.") from error
        except (json.JSONDecodeError, UnicodeDecodeError) as error:
            raise LLMProviderError("The LLM service returned an invalid response.") from error
        try:
            return parse_chat_completion(payload)
        except LLMResponseError as error:
            raise LLMProviderError(str(error)) from error
