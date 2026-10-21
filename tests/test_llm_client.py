from __future__ import annotations

import io
import json
import os
import unittest
from unittest.mock import patch
from urllib.error import HTTPError, URLError

from src.llm.client import LLMClient, LLMError
from src.llm.config import LLMSettings
from src.llm.provider import LLMProviderError, OpenAICompatibleProvider
from src.llm.response_parser import LLMResponseError, parse_chat_completion


class FakeHTTPResponse:
    def __init__(self, payload: object) -> None:
        self.payload = json.dumps(payload).encode("utf-8")

    def __enter__(self) -> "FakeHTTPResponse":
        return self

    def __exit__(self, *_: object) -> None:
        return None

    def read(self) -> bytes:
        return self.payload


def settings(api_key: str = "test-secret") -> LLMSettings:
    return LLMSettings(
        api_key=api_key,
        model="test-model",
        base_url="https://llm.example.test/v1",
        temperature=0.2,
        max_tokens=123,
        timeout_seconds=5,
        max_history_messages=10,
        rag_top_k=5,
    )


class LLMClientTests(unittest.TestCase):
    def test_openai_compatible_request_and_response(self) -> None:
        response = {"choices": [{"message": {"content": "A grounded answer."}}]}
        with patch("src.llm.provider.urlopen", return_value=FakeHTTPResponse(response)) as open_url:
            result = OpenAICompatibleProvider(settings()).generate_response(
                [{"role": "user", "content": "Question"}],
                temperature=0.1,
                max_tokens=50,
            )
        self.assertEqual(result, "A grounded answer.")
        request = open_url.call_args.args[0]
        payload = json.loads(request.data.decode("utf-8"))
        self.assertEqual(payload["model"], "test-model")
        self.assertEqual(payload["max_tokens"], 50)
        self.assertEqual(request.get_header("Authorization"), "Bearer test-secret")
        self.assertNotIn("test-secret", str(payload))

    def test_missing_key_is_reported_without_request(self) -> None:
        provider = OpenAICompatibleProvider(settings(api_key=""))
        with self.assertRaisesRegex(LLMProviderError, "configure the API key"):
            provider.generate_response([], temperature=0.2, max_tokens=10)

    def test_rate_limit_error_does_not_expose_response_body(self) -> None:
        error = HTTPError(
            "https://llm.example.test", 429, "limited", {},
            io.BytesIO(b'{"error":{"type":"rate_limit_exceeded","message":"private details"}}'),
        )
        with patch("src.llm.provider.urlopen", side_effect=error):
            with self.assertRaisesRegex(LLMProviderError, "rate limited") as raised:
                OpenAICompatibleProvider(settings()).generate_response(
                    [], temperature=0.2, max_tokens=10
                )
        self.assertNotIn("test-secret", str(raised.exception))
        self.assertNotIn("private details", str(raised.exception))

    def test_exhausted_quota_has_actionable_message_without_provider_details(self) -> None:
        error = HTTPError(
            "https://llm.example.test",
            429,
            "limited",
            {},
            io.BytesIO(b'{"error":{"code":"insufficient_quota","message":"private details"}}'),
        )
        with patch("src.llm.provider.urlopen", side_effect=error):
            with self.assertRaisesRegex(LLMProviderError, "no available quota") as raised:
                OpenAICompatibleProvider(settings()).generate_response(
                    [], temperature=0.2, max_tokens=10
                )
        self.assertNotIn("private details", str(raised.exception))

    def test_timeout_and_network_errors_are_safe(self) -> None:
        provider = OpenAICompatibleProvider(settings())
        with patch("src.llm.provider.urlopen", side_effect=TimeoutError()):
            with self.assertRaisesRegex(LLMProviderError, "timed out"):
                provider.generate_response([], temperature=0.2, max_tokens=10)
        with patch("src.llm.provider.urlopen", side_effect=URLError("private endpoint details")):
            with self.assertRaises(LLMProviderError) as raised:
                provider.generate_response([], temperature=0.2, max_tokens=10)
        self.assertNotIn("private endpoint details", str(raised.exception))

    def test_empty_or_malformed_response_is_rejected(self) -> None:
        with self.assertRaises(LLMResponseError):
            parse_chat_completion({"choices": []})
        with self.assertRaises(LLMResponseError):
            parse_chat_completion({"choices": [{"message": {"content": " "}}]})

    def test_client_reports_missing_environment_key(self) -> None:
        with patch.dict(os.environ, {"LLM_API_KEY": ""}, clear=False):
            with self.assertRaisesRegex(LLMError, "configure the API key"):
                LLMClient.from_environment()


if __name__ == "__main__":
    unittest.main()
