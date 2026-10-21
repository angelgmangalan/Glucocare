"""Descriptive glucose reading statistics and exploratory associations."""

from __future__ import annotations

import statistics
from collections import defaultdict
from datetime import date, datetime, time
from pathlib import Path
from typing import Any

from src.analytics._shared import average, patient_history, resolve_date_range
from src.database.database import DEFAULT_DATABASE_PATH, get_patient, get_patient_glucose, get_patient_meals


def get_glucose_reading_count(
    patient_id: str,
    start_date: date | datetime | str | None = None,
    end_date: date | datetime | str | None = None,
    *,
    days: int = 7,
    database_path: str | Path = DEFAULT_DATABASE_PATH,
) -> int:
    _, readings, _, _ = patient_history(
        patient_id, "glucose", start_date, end_date, days=days, database_path=database_path
    )
    return len(readings)


def get_glucose_summary(
    patient_id: str,
    start_date: date | datetime | str | None = None,
    end_date: date | datetime | str | None = None,
    *,
    days: int = 7,
    database_path: str | Path = DEFAULT_DATABASE_PATH,
) -> dict[str, Any]:
    patient, readings, start, end = patient_history(
        patient_id, "glucose", start_date, end_date, days=days, database_path=database_path
    )
    values = [float(row["glucose_value"]) for row in readings if row.get("glucose_value") is not None]
    return {
        "patient_id": patient_id,
        "start_date": start.isoformat(),
        "end_date": end.isoformat(),
        "reading_count": len(readings),
        "average_recorded_glucose": average(values),
        "minimum_recorded_glucose": min(values) if values else None,
        "maximum_recorded_glucose": max(values) if values else None,
        "median_recorded_glucose": statistics.median(values) if values else None,
        "standard_deviation_recorded_glucose": statistics.stdev(values) if len(values) > 1 else None,
        "is_demo": bool(patient["is_demo"]) if patient else None,
    }


def get_glucose_trend(
    patient_id: str,
    start_date: date | datetime | str | None = None,
    end_date: date | datetime | str | None = None,
    *,
    days: int = 7,
    database_path: str | Path = DEFAULT_DATABASE_PATH,
) -> list[dict[str, Any]]:
    _, readings, _, _ = patient_history(
        patient_id, "glucose", start_date, end_date, days=days, database_path=database_path
    )
    return sorted(readings, key=lambda row: (row["date"], row.get("time") or ""))


def get_glucose_by_context(
    patient_id: str,
    start_date: date | datetime | str | None = None,
    end_date: date | datetime | str | None = None,
    *,
    days: int = 7,
    database_path: str | Path = DEFAULT_DATABASE_PATH,
) -> list[dict[str, Any]]:
    _, readings, _, _ = patient_history(
        patient_id, "glucose", start_date, end_date, days=days, database_path=database_path
    )
    groups: dict[str, list[float]] = defaultdict(list)
    for reading in readings:
        if reading.get("glucose_value") is not None:
            groups[str(reading["context"])].append(float(reading["glucose_value"]))
    return [
        {"context": context, "reading_count": len(values), "average_recorded_glucose": average(values)}
        for context, values in sorted(groups.items())
    ]


def get_daily_glucose_statistics(
    patient_id: str,
    start_date: date | datetime | str | None = None,
    end_date: date | datetime | str | None = None,
    *,
    days: int = 7,
    database_path: str | Path = DEFAULT_DATABASE_PATH,
) -> list[dict[str, Any]]:
    _, readings, _, _ = patient_history(
        patient_id, "glucose", start_date, end_date, days=days, database_path=database_path
    )
    grouped: dict[str, list[float]] = defaultdict(list)
    for reading in readings:
        if reading.get("glucose_value") is not None:
            grouped[str(reading["date"])].append(float(reading["glucose_value"]))
    return [
        {
            "date": day,
            "reading_count": len(values),
            "average_recorded_glucose": average(values),
            "minimum_recorded_glucose": min(values),
            "maximum_recorded_glucose": max(values),
        }
        for day, values in sorted(grouped.items())
    ]


def get_meal_glucose_pairs(
    patient_id: str,
    start_date: date | datetime | str | None = None,
    end_date: date | datetime | str | None = None,
    *,
    days: int = 7,
    window_minutes: int = 120,
    database_path: str | Path = DEFAULT_DATABASE_PATH,
) -> list[dict[str, Any]]:
    if isinstance(window_minutes, bool) or not isinstance(window_minutes, int) or window_minutes < 0:
        raise ValueError("window_minutes must be a non-negative integer.")
    start, end = resolve_date_range(start_date, end_date, days=days)
    patient = get_patient(patient_id, database_path)
    if patient is None:
        return []
    meals = get_patient_meals(patient_id, database_path)
    glucose = get_patient_glucose(patient_id, database_path)
    glucose_datetimes: list[tuple[datetime, dict[str, Any]]] = []
    for reading in glucose:
        try:
            reading_at = datetime.combine(date.fromisoformat(reading["date"]), time.fromisoformat(reading["time"]))
        except (TypeError, ValueError, KeyError):
            continue
        glucose_datetimes.append((reading_at, reading))

    window = window_minutes * 60
    associations: list[dict[str, Any]] = []
    for meal in meals:
        meal_day = str(meal.get("date", ""))
        if not start.isoformat() <= meal_day <= end.isoformat() or not meal.get("meal_time"):
            continue
        try:
            meal_at = datetime.combine(date.fromisoformat(meal_day), time.fromisoformat(meal["meal_time"]))
        except (TypeError, ValueError):
            continue
        nearby = [
            (reading_at, reading)
            for reading_at, reading in glucose_datetimes
            if abs((reading_at - meal_at).total_seconds()) <= window
        ]
        if not nearby:
            continue
        before = [
            {"time": reading["time"], "glucose_value": reading["glucose_value"], "context": reading["context"]}
            for reading_at, reading in nearby if reading_at < meal_at
        ]
        after = [
            {"time": reading["time"], "glucose_value": reading["glucose_value"], "context": reading["context"]}
            for reading_at, reading in nearby if reading_at >= meal_at
        ]
        associations.append({
            "date": meal_day,
            "meal_time": meal["meal_time"],
            "meal_type": meal["meal_type"],
            "food": meal["food"],
            "before_meal_readings": before,
            "after_meal_readings": after,
            "window_minutes": window_minutes,
            "label": "Exploratory meal-glucose association; this does not establish causation.",
        })
    return associations
