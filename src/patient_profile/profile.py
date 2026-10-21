"""Patient-level profile and factual history summaries."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

from src.database.database import (
    DEFAULT_DATABASE_PATH,
    get_patient,
    get_patient_glucose,
    get_patient_lifestyle,
    get_patient_meals,
    get_recent_glucose,
    get_recent_lifestyle,
    get_recent_meals,
)


def get_patient_profile(
    patient_id: str,
    days: int = 7,
    database_path: str | Path = DEFAULT_DATABASE_PATH,
) -> dict[str, Any] | None:
    """Return the profile and recent history, or None for an unknown patient."""
    patient = get_patient(patient_id, database_path)
    if patient is None:
        return None
    return {
        "patient": patient,
        "recent_meals": get_recent_meals(patient_id, days, database_path),
        "glucose_history": get_recent_glucose(patient_id, days, database_path),
        "lifestyle_history": get_recent_lifestyle(patient_id, days, database_path),
    }


def generate_patient_summary(
    patient_id: str,
    database_path: str | Path = DEFAULT_DATABASE_PATH,
) -> dict[str, Any] | None:
    """Calculate descriptive record summaries; make no medical interpretations."""
    patient = get_patient(patient_id, database_path)
    if patient is None:
        return None

    meals = get_patient_meals(patient_id, database_path)
    glucose = get_patient_glucose(patient_id, database_path)
    lifestyle = get_patient_lifestyle(patient_id, database_path)
    foods = Counter(
        food.strip()
        for meal in meals
        for food in meal["food"].split(";")
        if food.strip()
    )
    most_frequent_foods = [
        {"food": name, "count": count}
        for name, count in sorted(foods.items(), key=lambda pair: (-pair[1], pair[0]))[:5]
    ]

    return {
        "patient_id": patient["patient_id"],
        "age": patient["age"],
        "diabetes_type": patient["diabetes_type"],
        "diet_preference": patient["diet_preference"],
        "cuisine_preference": patient["cuisine_preference"],
        "activity_level": patient["activity_level"],
        "meal_count": len(meals),
        "glucose_reading_count": len(glucose),
        "average_recorded_glucose_mg_dl": (
            sum(record["glucose_value"] for record in glucose) / len(glucose) if glucose else None
        ),
        "lifestyle_record_count": len(lifestyle),
        "average_sleep_hours": (
            sum(record["sleep_hours"] for record in lifestyle) / len(lifestyle) if lifestyle else None
        ),
        "average_exercise_minutes": (
            sum(record["exercise_minutes"] for record in lifestyle) / len(lifestyle)
            if lifestyle else None
        ),
        "most_frequently_recorded_foods": most_frequent_foods,
        "is_demo": bool(patient["is_demo"]),
    }
