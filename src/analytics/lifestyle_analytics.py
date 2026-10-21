"""Descriptive lifestyle record summaries."""

from __future__ import annotations

from datetime import date, datetime
from pathlib import Path
from typing import Any

from src.analytics._shared import average, patient_history
from src.database.database import DEFAULT_DATABASE_PATH


STRESS_SCORES = {"low": 1, "moderate": 2, "high": 3}


def get_lifestyle_summary(
    patient_id: str,
    start_date: date | datetime | str | None = None,
    end_date: date | datetime | str | None = None,
    *,
    days: int = 7,
    database_path: str | Path = DEFAULT_DATABASE_PATH,
) -> dict[str, Any]:
    patient, records, start, end = patient_history(
        patient_id, "lifestyle", start_date, end_date, days=days, database_path=database_path
    )
    stress_scores = [
        float(STRESS_SCORES[str(row["stress_level"]).strip().lower()])
        for row in records
        if str(row.get("stress_level", "")).strip().lower() in STRESS_SCORES
    ]
    return {
        "patient_id": patient_id,
        "start_date": start.isoformat(),
        "end_date": end.isoformat(),
        "record_count": len(records),
        "average_sleep_hours": average([float(row["sleep_hours"]) for row in records if row.get("sleep_hours") is not None]),
        "average_exercise_minutes": average([float(row["exercise_minutes"]) for row in records if row.get("exercise_minutes") is not None]),
        "average_water_intake": average([float(row["water_intake"]) for row in records if row.get("water_intake") is not None]),
        "average_stress_score": average(stress_scores),
        "stress_score_scale": {"Low": 1, "Moderate": 2, "High": 3},
        "is_demo": bool(patient["is_demo"]) if patient else None,
    }


def _metric_summary(
    patient_id: str,
    metric: str,
    start_date: date | datetime | str | None,
    end_date: date | datetime | str | None,
    days: int,
    database_path: str | Path,
) -> dict[str, Any]:
    patient, records, start, end = patient_history(
        patient_id, "lifestyle", start_date, end_date, days=days, database_path=database_path
    )
    values = [float(row[metric]) for row in records if row.get(metric) is not None]
    return {
        "patient_id": patient_id,
        "start_date": start.isoformat(),
        "end_date": end.isoformat(),
        "record_count": len(values),
        "average": average(values),
        "minimum": min(values) if values else None,
        "maximum": max(values) if values else None,
        "is_demo": bool(patient["is_demo"]) if patient else None,
    }


def get_sleep_summary(patient_id: str, start_date: date | datetime | str | None = None, end_date: date | datetime | str | None = None, *, days: int = 7, database_path: str | Path = DEFAULT_DATABASE_PATH) -> dict[str, Any]:
    return _metric_summary(patient_id, "sleep_hours", start_date, end_date, days, database_path)


def get_exercise_summary(patient_id: str, start_date: date | datetime | str | None = None, end_date: date | datetime | str | None = None, *, days: int = 7, database_path: str | Path = DEFAULT_DATABASE_PATH) -> dict[str, Any]:
    return _metric_summary(patient_id, "exercise_minutes", start_date, end_date, days, database_path)


def get_water_summary(patient_id: str, start_date: date | datetime | str | None = None, end_date: date | datetime | str | None = None, *, days: int = 7, database_path: str | Path = DEFAULT_DATABASE_PATH) -> dict[str, Any]:
    return _metric_summary(patient_id, "water_intake", start_date, end_date, days, database_path)


def get_stress_summary(
    patient_id: str,
    start_date: date | datetime | str | None = None,
    end_date: date | datetime | str | None = None,
    *,
    days: int = 7,
    database_path: str | Path = DEFAULT_DATABASE_PATH,
) -> dict[str, Any]:
    patient, records, start, end = patient_history(
        patient_id, "lifestyle", start_date, end_date, days=days, database_path=database_path
    )
    scores = [
        float(STRESS_SCORES[str(row["stress_level"]).strip().lower()])
        for row in records if str(row.get("stress_level", "")).strip().lower() in STRESS_SCORES
    ]
    return {
        "patient_id": patient_id,
        "start_date": start.isoformat(),
        "end_date": end.isoformat(),
        "record_count": len(scores),
        "average_stress_score": average(scores),
        "score_scale": {"Low": 1, "Moderate": 2, "High": 3},
        "is_demo": bool(patient["is_demo"]) if patient else None,
    }
