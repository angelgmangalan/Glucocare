"""Per-database-serving nutrition calculations with explicit unavailable values."""

from __future__ import annotations

import math
from typing import Any


NUTRIENTS = ("calories", "carbohydrates", "protein", "fat", "fiber")


def has_verified_nutrition(record: dict[str, Any]) -> bool:
    source = str(record.get("source", "")).casefold()
    return (
        not bool(record.get("is_demo", True))
        and bool(source.strip())
        and not any(term in source for term in ("synthetic", "unverified", "illustrative", "not validated"))
    )


def calculate_item_nutrition(
    record: dict[str, Any] | None,
    serving_quantity: float,
) -> dict[str, float | None]:
    if isinstance(serving_quantity, bool) or not isinstance(serving_quantity, (int, float)):
        raise ValueError("Serving quantity must be a positive number.")
    if not math.isfinite(serving_quantity) or serving_quantity <= 0:
        raise ValueError("Serving quantity must be a positive number.")
    if record is None or not has_verified_nutrition(record):
        return {nutrient: None for nutrient in NUTRIENTS}
    result: dict[str, float | None] = {}
    for nutrient in NUTRIENTS:
        value = record.get(nutrient)
        if value is None:
            result[nutrient] = None
        elif isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f"The selected record has an invalid {nutrient} value.")
        else:
            result[nutrient] = float(value) * serving_quantity
    return result


def calculate_meal_totals(items: list[dict[str, Any]]) -> dict[str, float | None]:
    """Only report a total when every included item has a verified value for it."""
    totals: dict[str, float | None] = {}
    for nutrient in NUTRIENTS:
        values = [item.get(nutrient) for item in items]
        if not values or any(value is None for value in values):
            totals[nutrient] = None
        else:
            totals[nutrient] = sum(float(value) for value in values)
    return totals
