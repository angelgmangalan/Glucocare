"""Phase 6: Patient-aware meal recommendation engine."""

from __future__ import annotations

from typing import Any


def generate_meal_recommendations(
    patient: dict[str, Any],
    food_records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Generate food suggestions while checking recorded allergy terms."""

    recommendations = []

    # Read the patient's recorded allergies.
    allergies = [
        item.strip().lower()
        for item in str(patient.get("allergies") or "")
        .replace(";", ",")
        .split(",")
        if item.strip()
    ]

    # Review available food records.
    for food in food_records:

        food_name = str(food.get("food_name") or "").strip()
        regional_name = str(food.get("regional_name") or "").strip()

        if not food_name:
            continue

        # Check food names against recorded allergy terms.
        searchable_text = f"{food_name} {regional_name}".lower()

        if any(allergy in searchable_text for allergy in allergies):
            continue

        # Identify demo records.
        is_demo = bool(food.get("is_demo"))

        recommendations.append({
            "food_name": food_name,
            "category": food.get("category") or "Not specified",
            "calories": None if is_demo else food.get("calories"),
            "is_demo": is_demo,
            "reason": (
                "Available food database entry. Check ingredients and "
                "suitability before choosing."
            ),
        })

    return recommendations