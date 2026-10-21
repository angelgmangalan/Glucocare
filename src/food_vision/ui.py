"""Streamlit review and confirmation interface for food-image meal logs."""

from __future__ import annotations

import hashlib
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

import streamlit as st

from src.database.database import (
    DEFAULT_DATABASE_PATH,
    DatabaseError,
    get_all_food_information,
)
from src.food_vision.exceptions import FoodVisionError
from src.food_vision.food_matching import rank_food_matches
from src.food_vision.image_validation import configured_max_upload_bytes, prepare_image
from src.food_vision.nutrition_mapping import (
    calculate_item_nutrition,
    calculate_meal_totals,
    has_verified_nutrition,
)
from src.food_vision.service import analyze_uploaded_image, save_confirmed_meal


DISCLAIMER = (
    "Food identification and portion estimates are approximate. Nutrition values are calculated "
    "from selected food database records and the stated serving quantities. Hidden ingredients "
    "and preparation methods may not be visible. Review and confirm all details before saving."
)


def _clear_analysis(state: dict[str, Any]) -> None:
    state.update({"analysis_id": None, "predictions": [], "saved": False})


def _food_label(record: dict[str, Any]) -> str:
    regional = record.get("regional_name")
    suffix = f" / {regional}" if regional else ""
    status = " · unverified demo data" if record.get("is_demo") else ""
    return f"{record['food_name']}{suffix} — {record['serving_size']}{status}"


def _nutrition_text(values: dict[str, float | None]) -> str:
    labels = {
        "calories": "kcal",
        "carbohydrates": "g carbs",
        "protein": "g protein",
        "fat": "g fat",
        "fiber": "g fibre",
    }
    return " · ".join(
        f"{labels[key]}: {value:.1f}" if value is not None else f"{labels[key]}: unavailable"
        for key, value in values.items()
    )


def render_food_image_analysis(
    patient_id: str,
    database_path: str | Path = DEFAULT_DATABASE_PATH,
) -> None:
    st.title("Food Image Analysis")
    st.caption(f"Meal photo review for selected patient {patient_id}.")
    st.warning(DISCLAIMER)
    st.info(
        "This academic prototype's bundled food values are synthetic, unverified demo estimates. "
        "They may be offered as match candidates, but they are not used to calculate nutrition."
    )

    state_key = "food_image_analysis_state"
    state = st.session_state.get(state_key)
    if not isinstance(state, dict) or state.get("patient_id") != patient_id:
        state = {
            "patient_id": patient_id,
            "analysis_id": None,
            "predictions": [],
            "saved": False,
            "image_digest": None,
        }
        st.session_state[state_key] = state

    try:
        max_bytes = configured_max_upload_bytes()
    except ValueError as error:
        st.error(str(error))
        return
    uploaded = st.file_uploader(
        "Upload a meal photo (JPEG, PNG, or WEBP; maximum "
        f"{max_bytes // (1024 * 1024)} MB)",
        type=["jpg", "jpeg", "png", "webp"],
        key=f"food_image_upload_{patient_id}",
    )
    prepared = None
    upload_bytes = None
    if uploaded is not None:
        upload_bytes = uploaded.getvalue()
        digest = hashlib.sha256(upload_bytes).hexdigest()
        if state.get("image_digest") != digest:
            _clear_analysis(state)
            state["image_digest"] = digest
        try:
            prepared = prepare_image(
                upload_bytes,
                uploaded.name,
                max_bytes=max_bytes,
            )
        except ValueError as error:
            _clear_analysis(state)
            st.error(str(error))
        else:
            st.image(prepared.data, caption="Image preview (metadata removed for analysis)", width=420)
    elif state.get("image_digest") is not None:
        _clear_analysis(state)
        state["image_digest"] = None

    if st.button(
        "Analyze Meal Photo",
        type="primary",
        disabled=prepared is None,
        key=f"analyze_food_image_{patient_id}",
    ):
        try:
            with st.spinner("Analyzing the image. No meal will be saved yet."):
                _, predictions = analyze_uploaded_image(
                    upload_bytes or b"",
                    uploaded.name if uploaded else "",
                    max_bytes=max_bytes,
                )
        except (FoodVisionError, ValueError) as error:
            _clear_analysis(state)
            st.error(str(error))
        else:
            state["analysis_id"] = uuid.uuid4().hex
            state["predictions"] = [prediction.as_dict() for prediction in predictions]
            state["saved"] = False
            st.rerun()

    if not state.get("analysis_id"):
        st.caption("Select a patient, upload a photo, and analyze it to review possible food items.")
        return

    if st.button("Cancel analysis without saving", key=f"cancel_food_analysis_{patient_id}_{state['analysis_id']}"):
        _clear_analysis(state)
        st.rerun()

    st.subheader("Review identified foods")
    st.caption(
        "Model-reported confidence, when shown, is an uncalibrated estimate. Model portion estimates "
        "are informational only; calculations require a serving quantity you confirm below."
    )
    if not state["predictions"]:
        st.info("No food items were identified. Try another photo or add an item manually.")

    manual_key = f"manual_food_label_{patient_id}_{state['analysis_id']}"
    manual_label = st.text_input("Add or correct a food name", key=manual_key)
    if st.button("Add item", key=f"add_food_item_{patient_id}_{state['analysis_id']}"):
        if not manual_label.strip():
            st.error("Enter a food name before adding an item.")
        else:
            state["predictions"].append({
                "item_label": manual_label.strip(),
                "alternative_labels": [],
                "confidence": None,
                "visible_evidence": "Added by the user.",
                "portion_description": "",
                "portion_estimate_grams": None,
                "portion_estimate_basis": "No model estimate.",
                "ambiguities": [],
                "clarification_needed": False,
                "identification_source": "user",
            })
            st.rerun()

    records = get_all_food_information(database_path)
    reviewed_items: list[dict[str, Any]] = []
    for index, prediction in enumerate(state["predictions"]):
        label = prediction["item_label"]
        with st.container(border=True):
            st.markdown(f"**Visible item {index + 1}: {label}**")
            if prediction["alternative_labels"]:
                st.write("Other possibilities:", ", ".join(prediction["alternative_labels"]))
            if prediction["visible_evidence"]:
                st.write("Visual evidence:", prediction["visible_evidence"])
            if prediction["confidence"] is not None:
                st.caption(f"Uncalibrated model confidence estimate: {prediction['confidence']:.0%}")
            portion = prediction["portion_description"] or "No reliable visual portion estimate."
            st.write(f"Visual portion estimate: {portion}")
            if prediction["portion_estimate_grams"] is not None:
                st.caption(
                    f"Approximate model estimate: {prediction['portion_estimate_grams']:g} g "
                    f"({prediction['portion_estimate_basis'] or 'basis not stated'}); not used in calculations."
                )
            if prediction["ambiguities"]:
                st.warning("Uncertainty: " + "; ".join(prediction["ambiguities"]))
            elif prediction["clarification_needed"]:
                st.warning("This identification needs user clarification.")

            corrected_label = st.text_input(
                "Correct the predicted name if needed",
                value=label,
                key=f"correct_food_{patient_id}_{state['analysis_id']}_{index}",
            ).strip()
            candidates = rank_food_matches(corrected_label, records)
            option_ids = [None, "unmatched", *[str(candidate["food_id"]) for candidate in candidates]]
            candidate_by_id = {str(candidate["food_id"]): candidate for candidate in candidates}
            selection_key = (
                f"food_match_{patient_id}_{state['analysis_id']}_{index}_"
                f"{hashlib.sha1(corrected_label.encode('utf-8')).hexdigest()[:8]}"
            )
            selected_id = st.selectbox(
                "Choose a nutrition database record (none is selected automatically)",
                option_ids,
                format_func=lambda value: (
                    "Choose a record"
                    if value is None
                    else "No database match — nutrition unavailable"
                    if value == "unmatched"
                    else _food_label(candidate_by_id[value])
                ),
                key=selection_key,
            )
            include = st.checkbox(
                "Include this item in the meal record",
                value=True,
                key=f"include_food_{patient_id}_{state['analysis_id']}_{index}",
            )
            selected_record = candidate_by_id.get(selected_id) if selected_id not in (None, "unmatched") else None
            quantity = 1.0
            quantity_confirmed = False
            if selected_record is not None:
                quantity = st.number_input(
                    f"Confirmed quantity in database-defined servings ({selected_record['serving_size']})",
                    min_value=0.25,
                    max_value=50.0,
                    value=1.0,
                    step=0.25,
                    key=f"servings_{patient_id}_{state['analysis_id']}_{index}",
                )
                quantity_confirmed = st.checkbox(
                    "I confirm this serving quantity",
                    value=False,
                    key=f"confirm_quantity_{patient_id}_{state['analysis_id']}_{index}",
                )
                if quantity_confirmed and has_verified_nutrition(selected_record):
                    calculated = calculate_item_nutrition(selected_record, quantity)
                    st.write("Calculated from verified record and stated servings:", _nutrition_text(calculated))
                elif quantity_confirmed:
                    st.warning(
                        f"Selected record source: {selected_record.get('source', 'not provided')}. "
                        "This record is unverified; nutrition totals remain unavailable."
                    )
                else:
                    st.info("Confirm the serving quantity before it can be used or saved.")
            elif selected_id == "unmatched":
                st.warning("No verified database match selected. Nutrition will remain unavailable.")
            else:
                st.info("Choose a record or explicitly mark this item as unmatched.")

            if st.button(
                "Remove item",
                key=f"remove_food_{patient_id}_{state['analysis_id']}_{index}",
            ):
                state["predictions"].pop(index)
                st.rerun()
            reviewed_items.append({
                "label": corrected_label,
                "original_prediction": label,
                "identification_source": prediction.get("identification_source", "vision"),
                "include": include,
                "food_id": selected_id if selected_record else None,
                "match_choice": "matched" if selected_record else selected_id,
                "serving_quantity": quantity,
                "quantity_confirmed": quantity_confirmed,
            })

    st.subheader("Confirm portions and nutrition")
    included_items = [item for item in reviewed_items if item["include"]]
    item_nutrition = []
    for item in included_items:
        record = next(
            (record for record in records if str(record["food_id"]) == item["food_id"]),
            None,
        ) if item["food_id"] else None
        item_nutrition.append(
            calculate_item_nutrition(record, item["serving_quantity"])
            if item["match_choice"] == "matched" and item["quantity_confirmed"]
            else {nutrient: None for nutrient in ("calories", "carbohydrates", "protein", "fat", "fiber")}
        )
    totals = calculate_meal_totals(item_nutrition)
    st.write("Meal nutrition totals:", _nutrition_text(totals))
    if any(value is None for value in totals.values()):
        st.caption("Some or all totals are unavailable; no missing values are treated as zero.")
    meal_date = st.date_input(
        "Meal date", value=datetime.now().date(), key=f"meal_date_{patient_id}_{state['analysis_id']}"
    )
    meal_time = st.time_input(
        "Meal time", value=datetime.now().time().replace(second=0, microsecond=0),
        key=f"meal_time_{patient_id}_{state['analysis_id']}",
    )
    meal_type = st.selectbox(
        "Meal type",
        ["Breakfast", "Lunch", "Dinner", "Snack", "Other"],
        key=f"meal_type_{patient_id}_{state['analysis_id']}",
    )
    st.markdown("**Final meal summary**")
    st.write(f"Patient: {patient_id} · {meal_date} {meal_time:%H:%M} · {meal_type}")
    st.write("Included foods:", ", ".join(item["label"] for item in included_items) or "None")
    st.write("Nutrition:", _nutrition_text(totals))
    st.warning(
        "Food-image analysis does not determine ingredients, allergens, diabetes safety, or glucose response."
    )
    if state.get("saved"):
        st.success("Meal saved to this patient's meal history.")
        st.caption("The image is not stored with the meal.")
        return
    unresolved_review = any(
        item["match_choice"] not in {"matched", "unmatched"}
        or (item["match_choice"] == "matched" and not item["quantity_confirmed"])
        for item in included_items
    )
    if unresolved_review:
        st.info("Before saving, choose a database match or mark each item unmatched, and confirm servings for matched items.")
    if st.button(
        "Confirm and Save Meal",
        type="primary",
        disabled=not included_items or unresolved_review,
        key=f"save_food_image_meal_{patient_id}_{state['analysis_id']}",
    ):
        try:
            save_confirmed_meal(
                patient_id=patient_id,
                meal_date=meal_date,
                meal_time=meal_time,
                meal_type=meal_type,
                items=reviewed_items,
                submission_id=f"vision-{state['analysis_id']}",
                database_path=database_path,
            )
        except (DatabaseError, ValueError) as error:
            st.error(str(error))
        else:
            state["saved"] = True
            st.rerun()
