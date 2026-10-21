"""Strict validation for structured food-identification responses."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from typing import Any


class VisionResponseError(ValueError):
    """The model response did not satisfy the food prediction schema."""


@dataclass(frozen=True)
class FoodPrediction:
    item_label: str
    alternative_labels: tuple[str, ...]
    confidence: float | None
    visible_evidence: str
    portion_description: str
    portion_estimate_grams: float | None
    portion_estimate_basis: str
    ambiguities: tuple[str, ...]
    clarification_needed: bool

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _text(value: Any, field: str, *, required: bool = False, limit: int = 500) -> str:
    if value is None and not required:
        return ""
    if not isinstance(value, str) or len(value) > limit or (required and not value.strip()):
        raise VisionResponseError(f"Invalid {field} in the model response.")
    return value.strip()


def _text_list(value: Any, field: str, *, limit: int = 10) -> tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, list) or len(value) > limit:
        raise VisionResponseError(f"Invalid {field} in the model response.")
    return tuple(_text(item, field, required=True, limit=160) for item in value)


def parse_predictions(response: str | dict[str, Any]) -> list[FoodPrediction]:
    try:
        payload = json.loads(response) if isinstance(response, str) else response
    except json.JSONDecodeError as error:
        raise VisionResponseError("The vision model returned invalid structured data.") from error
    if not isinstance(payload, dict) or not isinstance(payload.get("items"), list):
        raise VisionResponseError("The vision model response is missing its food item list.")
    if len(payload["items"]) > 20:
        raise VisionResponseError("The vision model returned too many food items.")
    results: list[FoodPrediction] = []
    for item in payload["items"]:
        if not isinstance(item, dict):
            raise VisionResponseError("A food item in the model response is invalid.")
        confidence = item.get("confidence")
        if confidence is not None and (
            isinstance(confidence, bool)
            or not isinstance(confidence, (int, float))
            or not math.isfinite(confidence)
            or not 0 <= confidence <= 1
        ):
            raise VisionResponseError("Invalid confidence value in the model response.")
        grams = item.get("portion_estimate_grams")
        if grams is not None and (
            isinstance(grams, bool)
            or not isinstance(grams, (int, float))
            or not math.isfinite(grams)
            or grams <= 0
            or grams > 10000
        ):
            raise VisionResponseError("Invalid portion estimate in the model response.")
        clarification = item.get("clarification_needed", False)
        if not isinstance(clarification, bool):
            raise VisionResponseError("Invalid clarification flag in the model response.")
        results.append(FoodPrediction(
            item_label=_text(item.get("item_label"), "item_label", required=True, limit=160),
            alternative_labels=_text_list(item.get("alternative_labels"), "alternative_labels"),
            confidence=float(confidence) if confidence is not None else None,
            visible_evidence=_text(item.get("visible_evidence"), "visible_evidence"),
            portion_description=_text(item.get("portion_description"), "portion_description", limit=160),
            portion_estimate_grams=float(grams) if grams is not None else None,
            portion_estimate_basis=_text(item.get("portion_estimate_basis"), "portion_estimate_basis"),
            ambiguities=_text_list(item.get("ambiguities"), "ambiguities"),
            clarification_needed=clarification,
        ))
    return results
