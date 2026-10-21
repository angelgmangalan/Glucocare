"""Build structured patient context for future assistant integrations."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.database.database import DEFAULT_DATABASE_PATH
from src.patient_profile.profile import generate_patient_summary, get_patient_profile


def build_patient_context(
    patient_id: str,
    days: int = 7,
    database_path: str | Path = DEFAULT_DATABASE_PATH,
) -> dict[str, Any] | None:
    """Build an LLM-ready structured patient object without medical interpretation."""
    profile = get_patient_profile(patient_id, days, database_path)
    if profile is None:
        return None
    patient = profile["patient"]
    return {
        "patient_profile": {
            "patient_id": patient["patient_id"],
            "age": patient["age"],
            "gender": patient["gender"],
            "activity_level": patient["activity_level"],
            "is_demo": bool(patient["is_demo"]),
        },
        "diabetes_information": {
            "diabetes_type": patient["diabetes_type"],
        },
        "dietary_preferences": {
            "diet_preference": patient["diet_preference"],
            "cuisine_preference": patient["cuisine_preference"],
        },
        "allergies_restrictions": patient["allergies"],
        "health_goals": patient["health_goals"],
        "recent_meals": profile["recent_meals"],
        "recent_glucose_history": profile["glucose_history"],
        "recent_lifestyle": profile["lifestyle_history"],
        "food_preferences": {
            "diet_preference": patient["diet_preference"],
            "cuisine_preference": patient["cuisine_preference"],
        },
        "summary": generate_patient_summary(patient_id, database_path),
    }
