"""OpenAI-compatible vision chat-completions client with safe provider errors."""

from __future__ import annotations

import base64
import json
import os
import socket
from dataclasses import dataclass
from urllib.parse import urlparse
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from src.food_vision.exceptions import FoodVisionError
from src.food_vision.image_validation import PreparedImage
from src.food_vision.prompts import SYSTEM_PROMPT


@dataclass(frozen=True)
class VisionSettings:
    api_key: str
    model: str
    base_url: str
    timeout_seconds: int = 45

    @classmethod
    def from_environment(cls) -> "VisionSettings":
        key = os.getenv("VISION_API_KEY", os.getenv("LLM_API_KEY", "")).strip()
        model = os.getenv("VISION_MODEL", "gpt-4o-mini").strip()
        base_url = os.getenv("VISION_BASE_URL", os.getenv(
            "LLM_BASE_URL", "https://api.openai.com/v1"
        )).strip().rstrip("/")
        timeout = os.getenv("VISION_TIMEOUT_SECONDS", "45")
        if not key:
            raise FoodVisionError(
                "Food image analysis is not configured. Set VISION_API_KEY or LLM_API_KEY."
            )
        if not model:
            raise FoodVisionError("Set VISION_MODEL to a model that supports image input.")
        parsed = urlparse(base_url)
        if (
            not parsed.scheme
            or not parsed.netloc
            or parsed.username
            or parsed.password
            or parsed.query
            or parsed.fragment
            or (
                parsed.scheme != "https"
                and not (parsed.scheme == "http" and parsed.hostname in {"localhost", "127.0.0.1"})
            )
        ):
            raise FoodVisionError("VISION_BASE_URL must use HTTPS, except for a local service.")
        try:
            timeout_value = int(timeout)
        except ValueError as error:
            raise FoodVisionError("VISION_TIMEOUT_SECONDS must be a positive integer.") from error
        if timeout_value < 1:
            raise FoodVisionError("VISION_TIMEOUT_SECONDS must be a positive integer.")
        return cls(key, model, base_url, timeout_value)


class VisionClient:
    def __init__(self, settings: VisionSettings) -> None:
        self.settings = settings

    def identify_foods(self, image: PreparedImage) -> str:
        encoded = base64.b64encode(image.data).decode("ascii")
        endpoint = self.settings.base_url
        if not endpoint.endswith("/chat/completions"):
            endpoint += "/chat/completions"
        payload = {
            "model": self.settings.model,
            "temperature": 0,
            "max_tokens": 1200,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": [
                    {"type": "text", "text": "Identify only the foods visible in this image."},
                    {"type": "image_url", "image_url": {
                        "url": f"data:{image.mime_type};base64,{encoded}",
                        "detail": "high",
                    }},
                ]},
            ],
        }
        request = Request(
            endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.settings.api_key}",
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
            method="POST",
        )
        try:
            with urlopen(request, timeout=self.settings.timeout_seconds) as response:
                result = json.loads(response.read().decode("utf-8"))
        except HTTPError as error:
            error.close()
            if error.code == 429:
                raise FoodVisionError("The vision service is rate limited. Try again shortly.") from error
            if error.code in {408, 504}:
                raise FoodVisionError("Food image analysis timed out. Please try again.") from error
            if error.code in {400, 404, 422}:
                raise FoodVisionError(
                    "The configured model or endpoint may not support image input. Check VISION_MODEL and VISION_BASE_URL."
                ) from error
            if error.code in {401, 403}:
                raise FoodVisionError("The vision service rejected its configured credentials.") from error
            raise FoodVisionError("The vision service is temporarily unavailable.") from error
        except (TimeoutError, socket.timeout) as error:
            raise FoodVisionError("Food image analysis timed out. Please try again.") from error
        except URLError as error:
            if isinstance(error.reason, (TimeoutError, socket.timeout)):
                raise FoodVisionError("Food image analysis timed out. Please try again.") from error
            raise FoodVisionError("Unable to connect to the vision service. Check the network and retry.") from error
        except (OSError, json.JSONDecodeError, UnicodeDecodeError) as error:
            raise FoodVisionError("The vision service returned an invalid or unavailable response.") from error
        try:
            content = result["choices"][0]["message"]["content"]
            if not isinstance(content, str):
                raise TypeError
            return content
        except (KeyError, IndexError, TypeError) as error:
            raise FoodVisionError("The vision service returned an invalid response.") from error
