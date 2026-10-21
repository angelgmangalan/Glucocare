"""Orchestration for validated image analysis and confirmed meal persistence."""

from __future__ import annotations

from datetime import date, time
from pathlib import Path
from typing import Any, Protocol

from src.database.database import (
    DEFAULT_DATABASE_PATH,
    get_all_food_information,
    get_food_information_by_id,
    get_patient,
    save_food_image_meal,
)
from src.food_vision.exceptions import FoodVisionError
from src.food_vision.image_validation import PreparedImage, prepare_image
from src.food_vision.nutrition_mapping import (
    NUTRIENTS,
    calculate_item_nutrition,
    calculate_meal_totals,
    has_verified_nutrition,
)
from src.food_vision.schemas import FoodPrediction, VisionResponseError, parse_predictions
from src.food_vision.vision_client import VisionClient, VisionSettings


class FoodIdentifier(Protocol):
    def identify_foods(self, image: PreparedImage) -> str: ...


def analyze_uploaded_image(
    upload: bytes,
    filename: str,
    *,
    client: FoodIdentifier | None = None,
    max_bytes: int | None = None,
) -> tuple[PreparedImage, list[FoodPrediction]]:
    image = prepare_image(upload, filename) if max_bytes is None else prepare_image(
        upload, filename, max_bytes=max_bytes
    )
    identifier = client or VisionClient(VisionSettings.from_environment())
    for attempt in range(2):
        try:
            return image, parse_predictions(identifier.identify_foods(image))
        except VisionResponseError as error:
            if attempt == 1:
                raise FoodVisionError(
                    "The vision service did not return usable food identifications. Retry with another image."
                ) from error
    raise FoodVisionError("Food image analysis could not be completed.")


def find_candidates(
    label: str,
    database_path: str | Path = DEFAULT_DATABASE_PATH,
) -> list[dict[str, Any]]:
    from src.food_vision.food_matching import rank_food_matches

    return rank_food_matches(label, get_all_food_information(database_path))


def save_confirmed_meal(
    *,
    patient_id: str,
    meal_date: date,
    meal_time: time,
    meal_type: str,
    items: list[dict[str, Any]],
    submission_id: str,
    database_path: str | Path = DEFAULT_DATABASE_PATH,
) -> None:
    if get_patient(patient_id, database_path) is None:
        raise ValueError("The selected patient is not available.")
    included = [item for item in items if item.get("include")]
    if not included:
        raise ValueError("Include at least one food item before saving.")
    food_ids: list[str] = []
    saved_labels: list[str] = []
    portions: list[str] = []
    item_nutrition: list[dict[str, float | None]] = []
    item_details: list[dict[str, Any]] = []
    for item in included:
        label = str(item.get("label", "")).strip()
        if not label:
            raise ValueError("Every included item needs a food name.")
        food_id = item.get("food_id")
        match_choice = item.get("match_choice")
        if match_choice not in {"matched", "unmatched"}:
            raise ValueError("Choose a database record or explicitly mark each item unmatched.")
        if (food_id is not None) != (match_choice == "matched"):
            raise ValueError("The selected database match is inconsistent. Review the item again.")
        record = get_food_information_by_id(food_id, database_path) if food_id else None
        if food_id and record is None:
            raise ValueError("A selected food record is no longer available. Review the meal again.")
        quantity = item.get("serving_quantity", 1.0)
        if record is not None and not item.get("quantity_confirmed"):
            raise ValueError("Confirm each selected record's serving quantity before saving.")
        nutrition = calculate_item_nutrition(record, quantity)
        item_nutrition.append(nutrition)
        if food_id:
            food_ids.append(str(food_id))
        saved_labels.append(record["food_name"] if record else label)
        portions.append(
            f"{quantity:g} x {record['serving_size']}" if record else "Serving quantity unknown"
        )
        item_details.append({
            "original_prediction": str(item.get("original_prediction") or label).strip(),
            "corrected_label": label,
            "identification_source": item.get("identification_source", "vision"),
            "selected_food_id": str(food_id) if food_id else None,
            "serving_quantity": quantity if record else None,
            "quantity_source": "user_confirmed" if record else "unknown",
            "nutrition_status": "verified" if record and has_verified_nutrition(record)
            else "unverified" if record else "unmatched",
        })
    totals = calculate_meal_totals(item_nutrition)
    nutrition_values = {nutrient: totals[nutrient] for nutrient in NUTRIENTS}
    save_food_image_meal(
        meal_id=submission_id,
        patient_id=patient_id,
        meal_date=meal_date.isoformat(),
        meal_time=meal_time.isoformat(timespec="minutes"),
        meal_type=meal_type,
        food="; ".join(saved_labels),
        portion="; ".join(portions),
        nutrition=nutrition_values,
        food_ids=food_ids,
        item_details=item_details,
        database_path=database_path,
    )
