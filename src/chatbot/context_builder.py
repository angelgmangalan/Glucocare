"""Build a compact, date-bounded context from existing patient and analytics APIs."""

from __future__ import annotations

import re
from datetime import date, timedelta
from pathlib import Path
from typing import Any

from src.analytics.patient_analytics import get_ai_context
from src.analytics._shared import patient_history
from src.database.database import DEFAULT_DATABASE_PATH, get_all_food_information, get_patient

MEAL_TERMS = {
    "meal", "meals", "eat", "eaten", "food", "foods", "breakfast", "lunch",
    "dinner", "snack", "snacks", "carbohydrate", "carbs", "nutrition",
    "fiber", "fibre", "puttu", "appam", "idiyappam", "dosa", "idli", "rice",
    "kanji", "chapati", "sambar", "avial", "thoran", "olan",
}
GLUCOSE_TERMS = {"glucose", "blood", "sugar", "reading", "readings", "a1c"}
LIFESTYLE_TERMS = {
    "lifestyle", "exercise", "activity", "active", "sleep", "stress", "water",
}
PERSONAL_TERMS = {
    "i", "me", "my", "mine", "we", "our", "recent", "recently", "yesterday",
    "usually", "pattern", "history", "records", "data",
}
MAX_HISTORY_ROWS = 15
MAX_TEXT_LENGTH = 300


def _words(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.casefold()))


def _mentions_food(question: str, foods: list[dict[str, Any]]) -> list[dict[str, Any]]:
    matches: list[dict[str, Any]] = []
    for item in foods:
        names = (item.get("food_name", ""), item.get("regional_name", ""))
        if any(
            name and re.search(rf"(?<!\w){re.escape(str(name).casefold())}(?!\w)", question.casefold())
            for name in names
        ):
            matches.append(item)
    return matches[:5]


def _bounded_text(value: Any) -> Any:
    if isinstance(value, str):
        return value[:MAX_TEXT_LENGTH]
    return value


def _remove_identifiers(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: _remove_identifiers(item)
            for key, item in value.items()
            if key not in {"patient_id", "reading_id", "meal_id", "lifestyle_id"}
        }
    if isinstance(value, list):
        return [_remove_identifiers(item) for item in value]
    return value


def _records(
    patient_id: str,
    table: str,
    start_date: date,
    end_date: date,
    database_path: str | Path,
) -> list[dict[str, Any]]:
    _, rows, _, _ = patient_history(
        patient_id, table, start_date, end_date, database_path=database_path
    )
    fields = {
        "meals": (
            "date", "meal_type", "meal_time", "food", "portion", "calories",
            "carbohydrates", "protein", "fat", "fiber",
        ),
        "glucose": ("date", "time", "glucose_value", "context"),
        "lifestyle": (
            "date", "sleep_hours", "exercise_minutes", "water_intake", "stress_level",
        ),
    }[table]
    return [
        {field: _bounded_text(row.get(field)) for field in fields if field in row}
        for row in rows[:MAX_HISTORY_ROWS]
    ]


def _trim_analytics(
    ai_context: dict[str, Any],
    *,
    include_meals: bool,
    include_glucose: bool,
    include_lifestyle: bool,
) -> dict[str, Any]:
    analytics = ai_context["analytics"]
    result: dict[str, Any] = {}
    if include_meals:
        meal = dict(analytics["meal"])
        for name in ("food_frequency", "daily_nutrition_from_recorded_meals"):
            meal[name] = meal.get(name, [])[:14]
        result["meals"] = meal
    if include_glucose:
        glucose = dict(analytics["glucose"])
        glucose["daily_statistics"] = glucose.get("daily_statistics", [])[:14]
        glucose["by_context"] = glucose.get("by_context", [])[:10]
        result["glucose"] = glucose
    if include_lifestyle:
        result["lifestyle"] = analytics["lifestyle"]
    if include_meals or include_glucose:
        result["meal_glucose_associations"] = analytics["meal_glucose_associations"][:10]
    return _remove_identifiers(result)


def build_chat_context(
    patient_id: str,
    question: str,
    start_date: date | None = None,
    end_date: date | None = None,
    *,
    database_path: str | Path = DEFAULT_DATABASE_PATH,
) -> dict[str, Any] | None:
    """Return only question-relevant patient data, using the existing analytics context."""
    patient = get_patient(patient_id, database_path)
    if patient is None:
        return None

    today = date.today()
    selected_end = end_date or today
    selected_start = start_date or selected_end - timedelta(days=6)
    if selected_start > selected_end:
        raise ValueError("start_date must be on or before end_date.")

    foods = _mentions_food(question, get_all_food_information(database_path))
    words = _words(question)
    food_personalization = bool(foods) and bool(words.intersection({"i", "me", "my", "for"}))
    personal = bool(words.intersection(PERSONAL_TERMS)) or food_personalization
    include_meals = personal and bool(words.intersection(MEAL_TERMS))
    include_glucose = personal and bool(words.intersection(GLUCOSE_TERMS))
    include_lifestyle = personal and bool(words.intersection(LIFESTYLE_TERMS))
    if personal and not (include_meals or include_glucose or include_lifestyle):
        include_meals = include_glucose = include_lifestyle = True

    if not personal and not foods:
        return {
            "patient_context": None,
            "patient_data": {},
            "analytics": {},
            "food_nutrition": [],
            "data_quality": {},
            "date_range": {
                "start_date": selected_start.isoformat(),
                "end_date": selected_end.isoformat(),
            },
            "context_used": {
                "patient_profile": False,
                "recent_meals": False,
                "glucose_history": False,
                "lifestyle_data": False,
                "analytics": False,
                "food_nutrition": False,
            },
        }

    ai_context = get_ai_context(
        patient_id,
        selected_start,
        selected_end,
        database_path=database_path,
    )
    if ai_context is None:
        return None
    patient_profile = ai_context["patient_profile"]
    include_profile = personal
    compact_profile = {}
    if include_profile:
        compact_profile = {
            key: _bounded_text(patient_profile.get(key))
            for key in (
                "diabetes_type", "diet_preference", "cuisine_preference",
                "allergies", "health_goals", "activity_level", "is_demo",
            )
            if key in patient_profile
        }
        if not compact_profile.get("allergies"):
            compact_profile["allergies"] = "No restrictions are recorded in the profile."
        if "is_demo" in compact_profile:
            compact_profile["is_demo"] = bool(compact_profile["is_demo"])

    patient_data: dict[str, Any] = {}
    if include_meals:
        patient_data["recent_meals"] = _records(
            patient_id, "meals", selected_start, selected_end, database_path
        )
    if include_glucose:
        patient_data["glucose_history"] = _records(
            patient_id, "glucose", selected_start, selected_end, database_path
        )
    if include_lifestyle:
        patient_data["lifestyle_history"] = _records(
            patient_id, "lifestyle", selected_start, selected_end, database_path
        )

    selected_foods = []
    for food in foods:
        selected_foods.append({
            key: food.get(key)
            for key in (
                "food_name", "regional_name", "category", "serving_size", "calories",
                "carbohydrates", "protein", "fat", "fiber", "allergens", "source",
                "is_demo",
            )
            if key in food
        })

    quality = dict(ai_context["data_quality"])
    for key, value in tuple(quality.items()):
        if isinstance(value, list):
            quality[key] = value[:14]
    return {
        "patient_context": compact_profile,
        "patient_data": patient_data,
        "analytics": _remove_identifiers(_trim_analytics(
            ai_context,
            include_meals=include_meals,
            include_glucose=include_glucose,
            include_lifestyle=include_lifestyle,
        )),
        "food_nutrition": selected_foods,
        "data_quality": quality if personal else {},
        "date_range": {
            "start_date": ai_context["start_date"],
            "end_date": ai_context["end_date"],
        },
        "context_used": {
            "patient_profile": include_profile,
            "recent_meals": include_meals and bool(patient_data.get("recent_meals")),
            "glucose_history": include_glucose and bool(patient_data.get("glucose_history")),
            "lifestyle_data": include_lifestyle and bool(patient_data.get("lifestyle_history")),
            "analytics": bool(include_meals or include_glucose or include_lifestyle),
            "food_nutrition": bool(selected_foods),
        },
    }
