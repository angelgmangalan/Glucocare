"""Descriptive meal and recorded-nutrition analytics."""

from __future__ import annotations

from collections import Counter, defaultdict
from datetime import date, datetime, time
from pathlib import Path
from typing import Any

from src.analytics._shared import average, patient_history
from src.database.database import DEFAULT_DATABASE_PATH


NUTRIENTS = ("calories", "carbohydrates", "protein", "fat", "fiber")


def get_recent_meals(
    patient_id: str,
    start_date: date | datetime | str | None = None,
    end_date: date | datetime | str | None = None,
    *,
    days: int = 7,
    database_path: str | Path = DEFAULT_DATABASE_PATH,
) -> list[dict[str, Any]]:
    _, meals, _, _ = patient_history(
        patient_id, "meals", start_date, end_date, days=days, database_path=database_path
    )
    return meals


def get_meal_count(
    patient_id: str,
    start_date: date | datetime | str | None = None,
    end_date: date | datetime | str | None = None,
    *,
    days: int = 7,
    database_path: str | Path = DEFAULT_DATABASE_PATH,
) -> int:
    return len(get_recent_meals(patient_id, start_date, end_date, days=days, database_path=database_path))


def get_meal_type_distribution(
    patient_id: str,
    start_date: date | datetime | str | None = None,
    end_date: date | datetime | str | None = None,
    *,
    days: int = 7,
    database_path: str | Path = DEFAULT_DATABASE_PATH,
) -> dict[str, int]:
    meals = get_recent_meals(patient_id, start_date, end_date, days=days, database_path=database_path)
    return dict(sorted(Counter(meal["meal_type"] for meal in meals).items()))


def get_food_frequency(
    patient_id: str,
    start_date: date | datetime | str | None = None,
    end_date: date | datetime | str | None = None,
    *,
    days: int = 7,
    database_path: str | Path = DEFAULT_DATABASE_PATH,
) -> list[dict[str, Any]]:
    meals = get_recent_meals(patient_id, start_date, end_date, days=days, database_path=database_path)
    counts: Counter[str] = Counter()
    for meal in meals:
        counts.update(food.strip() for food in str(meal.get("food", "")).split(";") if food.strip())
    return [{"food": food, "count": count} for food, count in sorted(counts.items(), key=lambda item: (-item[1], item[0]))]


def get_daily_nutrition_summary(
    patient_id: str,
    start_date: date | datetime | str | None = None,
    end_date: date | datetime | str | None = None,
    *,
    days: int = 7,
    database_path: str | Path = DEFAULT_DATABASE_PATH,
) -> list[dict[str, Any]]:
    meals = get_recent_meals(patient_id, start_date, end_date, days=days, database_path=database_path)
    daily: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "date": "",
            "meal_count": 0,
            **{key: 0.0 for key in NUTRIENTS},
            "_unavailable": set(),
        }
    )
    for meal in meals:
        row = daily[str(meal["date"])]
        row["date"] = str(meal["date"])
        row["meal_count"] += 1
        for nutrient in NUTRIENTS:
            value = meal.get(nutrient)
            if value is not None:
                row[nutrient] += float(value)
            else:
                row["_unavailable"].add(nutrient)
    result = []
    for key in sorted(daily):
        row = daily[key]
        unavailable = row.pop("_unavailable")
        for nutrient in unavailable:
            row[nutrient] = None
        result.append(row)
    return result


def get_average_meal_nutrition(
    patient_id: str,
    start_date: date | datetime | str | None = None,
    end_date: date | datetime | str | None = None,
    *,
    days: int = 7,
    database_path: str | Path = DEFAULT_DATABASE_PATH,
) -> dict[str, Any]:
    meals = get_recent_meals(patient_id, start_date, end_date, days=days, database_path=database_path)
    result: dict[str, Any] = {"recorded_meal_count": len(meals)}
    for nutrient in NUTRIENTS:
        values = [float(meal[nutrient]) for meal in meals if meal.get(nutrient) is not None]
        result[nutrient] = average(values)
    daily = get_daily_nutrition_summary(
        patient_id, start_date, end_date, days=days, database_path=database_path
    )
    result["recorded_day_count"] = len(daily)
    result["average_per_recorded_day"] = {
        nutrient: average([
            float(day[nutrient]) for day in daily if day[nutrient] is not None
        ])
        for nutrient in NUTRIENTS
    }
    result["label"] = "Average nutrition across recorded meals"
    result["daily_label"] = "Average of recorded meal sums on days with meal records; not total daily intake"
    return result


def get_meal_timing_summary(
    patient_id: str,
    start_date: date | datetime | str | None = None,
    end_date: date | datetime | str | None = None,
    *,
    days: int = 7,
    database_path: str | Path = DEFAULT_DATABASE_PATH,
) -> dict[str, Any]:
    meals = get_recent_meals(patient_id, start_date, end_date, days=days, database_path=database_path)
    timed: list[tuple[datetime, dict[str, Any]]] = []
    for meal in meals:
        if not meal.get("meal_time"):
            continue
        try:
            timestamp = datetime.combine(
                date.fromisoformat(str(meal["date"])),
                time.fromisoformat(str(meal["meal_time"])),
            )
        except (TypeError, ValueError):
            continue
        timed.append((timestamp, meal))

    times_by_type: dict[str, list[int]] = defaultdict(list)
    for timestamp, meal in timed:
        times_by_type[str(meal["meal_type"])].append(timestamp.hour * 60 + timestamp.minute)
    day_groups: dict[str, list[datetime]] = defaultdict(list)
    for timestamp, _ in timed:
        day_groups[timestamp.date().isoformat()].append(timestamp)
    gaps = [
        (later - earlier).total_seconds() / 60
        for day_meals in day_groups.values()
        for earlier, later in zip(sorted(day_meals), sorted(day_meals)[1:])
    ]

    average_times = {
        meal_type: time(hour=sum(minutes) // len(minutes) // 60, minute=sum(minutes) // len(minutes) % 60).isoformat(timespec="minutes")
        for meal_type, minutes in sorted(times_by_type.items())
    }
    return {
        "timed_meal_count": len(timed),
        "average_time_by_meal_type": average_times,
        "average_gap_between_recorded_meals_minutes": average(gaps),
        "gap_count": len(gaps),
        "status": "available" if timed else "Meal timing analysis is unavailable because no meal timestamps are recorded.",
    }
