"""Combined patient analytics and data-quality reporting."""

from __future__ import annotations

from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

from src.analytics._shared import patient_history, resolve_date_range
from src.analytics.glucose_analytics import (
    get_daily_glucose_statistics,
    get_glucose_by_context,
    get_glucose_summary,
    get_meal_glucose_pairs,
)
from src.analytics.lifestyle_analytics import get_lifestyle_summary
from src.analytics.meal_analytics import (
    get_average_meal_nutrition,
    get_daily_nutrition_summary,
    get_food_frequency,
    get_meal_count,
    get_meal_type_distribution,
    get_meal_timing_summary,
)
from src.database.database import (
    DEFAULT_DATABASE_PATH,
    get_patient,
    get_patient_glucose,
    get_patient_lifestyle,
    get_patient_meals,
)


def _data_quality(
    patient_id: str,
    start: date,
    end: date,
    database_path: str | Path,
) -> dict[str, Any]:
    _, meals, _, _ = patient_history(patient_id, "meals", start, end, database_path=database_path)
    _, glucose, _, _ = patient_history(patient_id, "glucose", start, end, database_path=database_path)
    _, lifestyle, _, _ = patient_history(patient_id, "lifestyle", start, end, database_path=database_path)
    all_meals = get_patient_meals(patient_id, database_path)
    all_glucose = get_patient_glucose(patient_id, database_path)
    all_lifestyle = get_patient_lifestyle(patient_id, database_path)
    window_dates = {(start + timedelta(days=offset)).isoformat() for offset in range((end - start).days + 1)}
    observed_dates = {
        str(row["date"])
        for row in meals + glucose + lifestyle
        if row.get("date") and str(row["date"]) in window_dates
    }
    missing_dates = sorted(window_dates.difference(observed_dates))
    dates_with_meals = {str(row["date"]) for row in meals if row.get("date")}
    dates_with_glucose = {str(row["date"]) for row in glucose if row.get("date")}
    dates_with_lifestyle = {str(row["date"]) for row in lifestyle if row.get("date")}
    return {
        "start_date": start.isoformat(),
        "end_date": end.isoformat(),
        "meal_record_count": len(meals),
        "glucose_record_count": len(glucose),
        "lifestyle_record_count": len(lifestyle),
        "missing_dates_in_selected_period": missing_dates,
        "missing_date_count": len(missing_dates),
        "dates_without_meal_records": sorted(window_dates.difference(dates_with_meals)),
        "dates_without_glucose_records": sorted(window_dates.difference(dates_with_glucose)),
        "dates_without_lifestyle_records": sorted(window_dates.difference(dates_with_lifestyle)),
        "missing_meal_dates": sum(not row.get("date") for row in meals),
        "missing_meal_information": sum(not row.get("food") or not row.get("meal_type") for row in meals),
        "missing_glucose_dates": sum(not row.get("date") for row in glucose),
        "missing_glucose_values": sum(row.get("glucose_value") is None for row in glucose),
        "missing_glucose_timestamps": sum(not row.get("time") for row in glucose),
        "missing_meal_timestamps": sum(not row.get("meal_time") for row in meals),
        "missing_lifestyle_dates": sum(not row.get("date") for row in lifestyle),
        "duplicate_meal_ids": len(all_meals) - len({row["meal_id"] for row in all_meals}),
        "duplicate_glucose_ids": len(all_glucose) - len({row["reading_id"] for row in all_glucose}),
        "duplicate_lifestyle_ids": len(all_lifestyle) - len({row["lifestyle_id"] for row in all_lifestyle}),
        "note": "Missing dates refer to dates with no meal, glucose, or lifestyle record of any kind.",
    }


def build_patient_analytics(
    patient_id: str,
    start_date: date | datetime | str | None = None,
    end_date: date | datetime | str | None = None,
    *,
    days: int = 7,
    window_minutes: int = 120,
    database_path: str | Path = DEFAULT_DATABASE_PATH,
) -> dict[str, Any] | None:
    patient = get_patient(patient_id, database_path)
    if patient is None:
        return None
    start, end = resolve_date_range(start_date, end_date, days=days)

    meals = get_patient_meals(patient_id, database_path)
    glucose = get_patient_glucose(patient_id, database_path)
    lifestyle = get_patient_lifestyle(patient_id, database_path)
    selected_meals = [
        row for row in meals if row.get("date") and start.isoformat() <= row["date"] <= end.isoformat()
    ]

    meal_summary = {
        "recorded_meal_count": get_meal_count(patient_id, start, end, database_path=database_path),
        "meal_type_distribution": get_meal_type_distribution(patient_id, start, end, database_path=database_path),
        "food_frequency": get_food_frequency(patient_id, start, end, database_path=database_path),
        "daily_nutrition_from_recorded_meals": get_daily_nutrition_summary(patient_id, start, end, database_path=database_path),
        "average_nutrition_across_recorded_meals": get_average_meal_nutrition(patient_id, start, end, database_path=database_path),
        "timing_summary": get_meal_timing_summary(patient_id, start, end, database_path=database_path),
    }
    glucose_summary = get_glucose_summary(patient_id, start, end, database_path=database_path)
    glucose_summary["daily_statistics"] = get_daily_glucose_statistics(patient_id, start, end, database_path=database_path)
    glucose_summary["by_context"] = get_glucose_by_context(patient_id, start, end, database_path=database_path)
    lifestyle_summary = get_lifestyle_summary(patient_id, start, end, database_path=database_path)
    associations = get_meal_glucose_pairs(
        patient_id, start, end, window_minutes=window_minutes, database_path=database_path
    )
    quality = _data_quality(patient_id, start, end, database_path)
    quality["meal_time_data_available"] = any(row.get("meal_time") for row in selected_meals)
    quality["meal_glucose_association_count"] = len(associations)
    quality["meal_glucose_association_status"] = (
        "available" if associations else
        "No suitable meal timestamps and nearby glucose readings are available."
    )

    return {
        "patient_id": patient_id,
        "start_date": start.isoformat(),
        "end_date": end.isoformat(),
        "patient_profile": patient,
        "meal_summary": meal_summary,
        "glucose_summary": glucose_summary,
        "lifestyle_summary": lifestyle_summary,
        "meal_glucose_associations": associations,
        "data_quality": quality,
        "is_demo": bool(patient["is_demo"]),
    }


def get_ai_context(
    patient_id: str,
    start_date: date | datetime | str | None = None,
    end_date: date | datetime | str | None = None,
    *,
    days: int = 7,
    database_path: str | Path = DEFAULT_DATABASE_PATH,
) -> dict[str, Any] | None:
    """Provide profile plus descriptive analytics for a future AI; no model is called."""
    patient = get_patient(patient_id, database_path)
    if patient is None:
        return None
    analytics = build_patient_analytics(
        patient_id, start_date, end_date, days=days, database_path=database_path
    )
    if analytics is None:
        return None
    return {
        "patient_profile": patient,
        "analytics": {
            "meal": analytics["meal_summary"],
            "glucose": analytics["glucose_summary"],
            "lifestyle": analytics["lifestyle_summary"],
            "meal_glucose_associations": analytics["meal_glucose_associations"],
        },
        "data_quality": analytics["data_quality"],
        "start_date": analytics["start_date"],
        "end_date": analytics["end_date"],
        "is_demo": analytics["is_demo"],
    }
