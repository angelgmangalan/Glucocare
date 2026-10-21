"""Shared date-window and patient-history helpers for analytics."""

from __future__ import annotations

from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

from src.database.database import (
    DEFAULT_DATABASE_PATH,
    get_patient,
    get_patient_glucose,
    get_patient_lifestyle,
    get_patient_meals,
)


def normalize_date(value: date | datetime | str, name: str) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        try:
            parsed = date.fromisoformat(value)
        except ValueError as error:
            raise ValueError(f"{name} must use YYYY-MM-DD format.") from error
        if parsed.isoformat() != value:
            raise ValueError(f"{name} must use YYYY-MM-DD format.")
        return parsed
    raise TypeError(f"{name} must be a date, datetime, or YYYY-MM-DD string.")


def resolve_date_range(
    start_date: date | datetime | str | None = None,
    end_date: date | datetime | str | None = None,
    *,
    days: int = 7,
) -> tuple[date, date]:
    if isinstance(days, bool) or not isinstance(days, int) or days < 1:
        raise ValueError("days must be a positive integer.")
    end = normalize_date(end_date, "end_date") if end_date is not None else date.today()
    start = (
        normalize_date(start_date, "start_date")
        if start_date is not None
        else end - timedelta(days=days - 1)
    )
    if start > end:
        raise ValueError("start_date must be on or before end_date.")
    return start, end


def patient_history(
    patient_id: str,
    table: str,
    start_date: date | datetime | str | None = None,
    end_date: date | datetime | str | None = None,
    *,
    days: int = 7,
    database_path: str | Path = DEFAULT_DATABASE_PATH,
) -> tuple[dict[str, Any] | None, list[dict[str, Any]], date, date]:
    start, end = resolve_date_range(start_date, end_date, days=days)
    patient = get_patient(patient_id, database_path)
    if patient is None:
        return None, [], start, end
    readers = {
        "meals": get_patient_meals,
        "glucose": get_patient_glucose,
        "lifestyle": get_patient_lifestyle,
    }
    if table not in readers:
        raise ValueError(f"Unsupported patient history table: {table}")
    records = readers[table](patient_id, database_path)
    filtered = [
        record for record in records
        if record.get("date")
        and start.isoformat() <= str(record["date"]) <= end.isoformat()
    ]
    return patient, filtered, start, end


def average(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None
