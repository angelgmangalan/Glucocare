"""Phase 6: Meal recommendation interface."""

from __future__ import annotations

from typing import Any

import streamlit as st

from src.recommendation.engine import generate_meal_recommendations


def render_meal_recommendations(
    patient: dict[str, Any],
    food_records: list[dict[str, Any]],
) -> None:
    """Display food suggestions based on the patient's recorded preferences."""

    st.subheader("Meal Recommendations")

    st.write(
        "Explore food entries from the available nutrition database. "
        "These are general suggestions, not medical or dietary prescriptions."
    )

    recommendations = generate_meal_recommendations(patient, food_records)

    if not recommendations:
        st.info("No matching food entries are currently available.")
        return

    for item in recommendations:
        with st.container(border=True):
            st.markdown(f"**{item['food_name']}**")
            st.caption(f"Category: {item['category']}")
            st.write(item["reason"])

            if item.get("is_demo"):
                st.caption(
                    "Demo entry: nutrition information has not been verified."
                )
            elif item.get("calories") is not None:
                st.write(f"Calories: {item['calories']} kcal")