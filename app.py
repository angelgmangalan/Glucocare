"""Phase 5 GlucoCare dashboard with grounded assistant and medical evidence retrieval."""

from __future__ import annotations

from datetime import date, timedelta
from typing import Any

import pandas as pd
import plotly.express as px
import streamlit as st

from src.database.database import (
    DEFAULT_DATA_DIR,
    DEFAULT_DATABASE_PATH,
    DataValidationError,
    DatabaseError,
    get_all_food_information,
    get_all_patients,
    get_patient,
    get_patient_glucose,
    get_patient_lifestyle,
    get_patient_meals,
    initialize_database,
)
from src.analytics.glucose_analytics import get_glucose_trend
from src.analytics.lifestyle_analytics import get_lifestyle_summary
from src.analytics.meal_analytics import (
    get_daily_nutrition_summary,
    get_food_frequency,
    get_meal_type_distribution,
)
from src.analytics.patient_analytics import build_patient_analytics
from src.chatbot.chatbot import Chatbot, ChatbotError
from src.chatbot.conversation import append_exchange, trim_history
from src.patient_profile.profile import generate_patient_summary
from src.llm.client import LLMError
from src.llm.config import LLMConfigurationError, load_settings
from src.rag.pipeline import initialize_rag, retrieve_context
from src.rag.vector_store import VectorStoreError
from src.food_vision.ui import render_food_image_analysis


@st.cache_resource
def _initialize_app_database() -> bool:
    return initialize_database(DEFAULT_DATA_DIR, DEFAULT_DATABASE_PATH)


@st.cache_resource
def _initialize_rag_resources() -> dict[str, Any]:
    return initialize_rag()


def _format_value(value: Any, suffix: str = "") -> str:
    return "No data" if value is None else f"{value:.1f}{suffix}"


def _history_filter(frame: pd.DataFrame, key: str) -> pd.DataFrame:
    if frame.empty:
        return frame
    dates = pd.to_datetime(frame["date"], errors="raise").dt.date
    start, end = dates.min(), dates.max()
    chosen = st.date_input(
        "Filter by date",
        value=(start, end),
        min_value=start,
        max_value=end,
        key=key,
    )
    if isinstance(chosen, (tuple, list)) and len(chosen) == 2:
        lower, upper = chosen
        return frame.loc[dates.between(lower, upper)].copy()
    return frame.loc[dates == chosen].copy()


def _show_profile(patient: dict[str, Any]) -> None:
    st.title("Patient Profile")
    st.info("Synthetic demo profile — no real patient information is used.")
    left, right = st.columns(2)
    with left:
        st.subheader("Patient profile")
        st.write(f"**Patient ID:** {patient['patient_id']}")
        st.write(f"**Age:** {patient['age']}")
        st.write(f"**Gender:** {patient['gender']}")
        st.write(f"**Diabetes type:** {patient['diabetes_type']}")
        st.write(f"**Activity:** {patient['activity_level']}")
        st.subheader("Health goals")
        st.write(patient["health_goals"] or "No health goals recorded.")
        st.write(f"**Lifestyle information:** {patient['lifestyle_notes'] or 'Not recorded.'}")
    with right:
        st.subheader("Dietary information")
        st.write(f"**Diet preference:** {patient['diet_preference']}")
        st.write(f"**Cuisine preference:** {patient['cuisine_preference']}")
        st.write(f"**Allergies / restrictions:** {patient['allergies'] or 'None recorded.'}")


def _show_dashboard(patient: dict[str, Any], database_path: str) -> None:
    st.title("GlucoCare")
    st.caption("Patient Dashboard · Phase 4")
    st.warning(
        "Synthetic demo data only — these records are fictional and are not clinical records, "
        "medical advice, or medically validated predictions."
    )
    summary = generate_patient_summary(patient["patient_id"], database_path)
    if summary is None:
        st.error("The selected patient profile could not be found.")
        return
    _show_profile(patient)

    st.subheader("Recorded history summary")
    metrics = st.columns(4)
    metrics[0].metric("Meals recorded", summary["meal_count"])
    metrics[1].metric("Glucose readings", summary["glucose_reading_count"])
    metrics[2].metric(
        "Average recorded glucose",
        _format_value(summary["average_recorded_glucose_mg_dl"], " mg/dL"),
    )
    metrics[3].metric("Lifestyle records", summary["lifestyle_record_count"])
    lifestyle_metrics = st.columns(2)
    lifestyle_metrics[0].metric("Average recorded sleep", _format_value(summary["average_sleep_hours"], " h"))
    lifestyle_metrics[1].metric(
        "Average recorded exercise",
        _format_value(summary["average_exercise_minutes"], " min"),
    )
    st.caption("These are descriptive record summaries only, not medical interpretations.")

    recent_meals = get_patient_meals(patient["patient_id"], database_path)[:6]
    recent_glucose = get_patient_glucose(patient["patient_id"], database_path)[:10]
    lifestyle = get_patient_lifestyle(patient["patient_id"], database_path)
    meal_col, glucose_col = st.columns(2)
    with meal_col:
        st.subheader("Recent meal history")
        if recent_meals:
            st.dataframe(
                pd.DataFrame(recent_meals)[["date", "meal_type", "food"]],
                hide_index=True,
                width="stretch",
            )
        else:
            st.info("No meal history is available.")
    with glucose_col:
        st.subheader("Recent glucose history")
        if recent_glucose:
            glucose_frame = pd.DataFrame(recent_glucose)
            fig = px.scatter(
                glucose_frame.sort_values(["date", "time"]),
                x="date",
                y="glucose_value",
                color="context",
                hover_data=["time"],
                labels={
                    "date": "Date",
                    "glucose_value": "Recorded glucose (mg/dL)",
                    "context": "Context",
                },
                title="Demo data — not clinical measurements",
            )
            st.plotly_chart(fig, width="stretch")
        else:
            st.info("No glucose history is available.")
    st.subheader("Lifestyle summary")
    if lifestyle:
        lifestyle_frame = pd.DataFrame(lifestyle).sort_values("date")
        sleep_col, exercise_col = st.columns(2)
        with sleep_col:
            st.plotly_chart(
                px.line(
                    lifestyle_frame,
                    x="date",
                    y="sleep_hours",
                    markers=True,
                    labels={"date": "Date", "sleep_hours": "Sleep (hours)"},
                    title="Recorded sleep",
                ),
                width="stretch",
            )
        with exercise_col:
            st.plotly_chart(
                px.bar(
                    lifestyle_frame,
                    x="date",
                    y="exercise_minutes",
                    labels={"date": "Date", "exercise_minutes": "Exercise (minutes)"},
                    title="Recorded exercise",
                ),
                width="stretch",
            )
    else:
        st.info("No lifestyle history is available.")

    frequent_foods = summary["most_frequently_recorded_foods"]
    st.subheader("Frequently recorded foods")
    if frequent_foods:
        st.dataframe(pd.DataFrame(frequent_foods), hide_index=True, width="stretch")
    else:
        st.info("No foods are recorded.")


def _show_meals(patient_id: str, database_path: str) -> None:
    st.title("Meal History")
    st.caption(
        "All meal records are synthetic demo data. Nutrition values are illustrative estimates, "
        "not clinically validated."
    )
    meals = pd.DataFrame(get_patient_meals(patient_id, database_path))
    if meals.empty:
        st.info("No meal history is available for this patient.")
        return
    meals = _history_filter(meals, "meal_date_filter")
    meal_types = sorted(meals["meal_type"].unique().tolist())
    selected_types = st.multiselect("Filter by meal type", meal_types, default=meal_types)
    meals = meals.loc[meals["meal_type"].isin(selected_types)]
    display = meals.rename(columns={
        "date": "Date",
        "meal_type": "Meal",
        "food": "Food",
        "portion": "Portion",
        "calories": "Calories",
        "carbohydrates": "Carbohydrates (g)",
        "protein": "Protein (g)",
        "fat": "Fat (g)",
        "fiber": "Fiber (g)",
    })
    columns = [
        "Date", "Meal", "Food", "Portion", "Calories", "Carbohydrates (g)",
        "Protein (g)", "Fat (g)", "Fiber (g)",
    ]
    st.dataframe(display[columns], hide_index=True, width="stretch")


def _show_glucose(patient_id: str, database_path: str) -> None:
    st.title("Glucose History")
    st.warning("Demo data — not clinical measurements.")
    glucose = pd.DataFrame(get_patient_glucose(patient_id, database_path))
    if glucose.empty:
        st.info("No glucose readings are available for this patient.")
        return
    glucose = _history_filter(glucose, "glucose_date_filter")
    contexts = sorted(glucose["context"].unique().tolist())
    selected_contexts = st.multiselect("Filter by context", contexts, default=contexts)
    glucose = glucose.loc[glucose["context"].isin(selected_contexts)]
    if glucose.empty:
        st.info("No glucose readings match the selected filters.")
        return
    fig = px.line(
        glucose.sort_values(["date", "time"]),
        x="date",
        y="glucose_value",
        color="context",
        markers=True,
        hover_data=["time"],
        labels={
            "date": "Date",
            "glucose_value": "Recorded glucose (mg/dL)",
            "context": "Context",
        },
    )
    st.plotly_chart(fig, width="stretch")
    display = glucose.rename(columns={
        "date": "Date",
        "time": "Time",
        "glucose_value": "Glucose value (mg/dL)",
        "context": "Context",
    })
    st.dataframe(
        display[["Date", "Time", "Glucose value (mg/dL)", "Context"]],
        hide_index=True,
        width="stretch",
    )


def _show_lifestyle(patient_id: str, database_path: str) -> None:
    st.title("Lifestyle History")
    st.info("All lifestyle entries are synthetic demonstration records.")
    lifestyle = pd.DataFrame(get_patient_lifestyle(patient_id, database_path))
    if lifestyle.empty:
        st.info("No lifestyle history is available for this patient.")
        return
    lifestyle = _history_filter(lifestyle, "lifestyle_date_filter")
    ordered = lifestyle.sort_values("date")
    sleep_col, exercise_col = st.columns(2)
    with sleep_col:
        st.plotly_chart(
            px.line(
                ordered,
                x="date",
                y="sleep_hours",
                markers=True,
                labels={"date": "Date", "sleep_hours": "Sleep (hours)"},
                title="Recorded sleep",
            ),
            width="stretch",
        )
    with exercise_col:
        st.plotly_chart(
            px.bar(
                ordered,
                x="date",
                y="exercise_minutes",
                labels={"date": "Date", "exercise_minutes": "Exercise (minutes)"},
                title="Recorded exercise",
            ),
            width="stretch",
        )
    display = ordered.rename(columns={
        "date": "Date",
        "sleep_hours": "Sleep (hours)",
        "exercise_minutes": "Exercise (minutes)",
        "water_intake": "Water intake (liters)",
        "stress_level": "Stress",
    })
    st.dataframe(
        display[["Date", "Sleep (hours)", "Exercise (minutes)", "Water intake (liters)", "Stress"]],
        hide_index=True,
        width="stretch",
    )


def _show_foods(database_path: str) -> None:
    st.title("Food Nutrition Reference")
    st.warning(
        "Nutrition values are synthetic illustrative estimates. They are not verified against "
        "an authoritative food-composition table and are not for clinical decisions."
    )
    foods = pd.DataFrame(get_all_food_information(database_path))
    if foods.empty:
        st.info("No food nutrition information is available.")
        return
    display = foods.rename(columns={
        "food_name": "Food",
        "regional_name": "Regional name",
        "category": "Category",
        "serving_size": "Serving size",
        "calories": "Calories",
        "carbohydrates": "Carbohydrates (g)",
        "protein": "Protein (g)",
        "fat": "Fat (g)",
        "fiber": "Fiber (g)",
        "source": "Source",
    })
    st.dataframe(
        display[[
            "Food", "Regional name", "Category", "Serving size", "Calories",
            "Carbohydrates (g)", "Protein (g)", "Fat (g)", "Fiber (g)", "Source",
        ]],
        hide_index=True,
        width="stretch",
    )


def _selected_analytics_dates() -> tuple[Any, Any]:
    periods = {
        "Last 3 days": 3,
        "Last 7 days": 7,
        "Last 14 days": 14,
        "Last 30 days": 30,
    }
    selected = st.selectbox("Analytics date range", [*periods, "Custom range"], key="analytics_period")
    today = pd.Timestamp.today().date()
    if selected == "Custom range":
        chosen = st.date_input(
            "Custom date range",
            value=(today - pd.Timedelta(days=6), today),
            key="analytics_custom_range",
        )
        if isinstance(chosen, (tuple, list)) and len(chosen) == 2:
            start, end = chosen
        elif isinstance(chosen, (tuple, list)) and len(chosen) == 1:
            start = end = chosen[0]
        else:
            st.info("Select both a start date and an end date.")
            return today, today
        if start > end:
            st.error("The start date must be on or before the end date.")
            return end, start
        return start, end
    end = today
    return end - pd.Timedelta(days=periods[selected] - 1), end


def _show_analytics(patient: dict[str, Any], database_path: str) -> None:
    st.title("Patient Analytics")
    st.warning(
        "⚠️ Demo dataset: displayed patient, meal, glucose, and lifestyle records are synthetic "
        "and are not clinical measurements."
    )
    start_date, end_date = _selected_analytics_dates()
    if start_date > end_date:
        return
    analytics = build_patient_analytics(
        patient["patient_id"], start_date, end_date, database_path=database_path
    )
    if analytics is None:
        st.error("Analytics are unavailable because the selected patient does not exist.")
        return

    meals = analytics["meal_summary"]
    glucose = analytics["glucose_summary"]
    lifestyle = analytics["lifestyle_summary"]
    overview = st.columns(4)
    overview[0].metric("Recorded meals", meals["recorded_meal_count"])
    overview[1].metric("Glucose readings", glucose["reading_count"])
    overview[2].metric("Lifestyle records", lifestyle["record_count"])
    overview[3].metric(
        "Average recorded glucose",
        _format_value(glucose["average_recorded_glucose"], " mg/dL"),
    )
    st.caption(
        f"Patient {patient['patient_id']} · {analytics['start_date']} through "
        f"{analytics['end_date']}. Descriptive record summaries only; no diagnosis or causal conclusions."
    )

    st.subheader("Meal analytics")
    meal_distribution, food_distribution = st.columns(2)
    with meal_distribution:
        if meals["meal_type_distribution"]:
            st.plotly_chart(
                px.pie(
                    names=list(meals["meal_type_distribution"]),
                    values=list(meals["meal_type_distribution"].values()),
                    title="Recorded meals by type",
                ),
                width="stretch",
            )
        else:
            st.info("No meals are available for this period.")
    with food_distribution:
        if meals["food_frequency"]:
            st.plotly_chart(
                px.bar(
                    pd.DataFrame(meals["food_frequency"]).head(12),
                    x="food",
                    y="count",
                    labels={"food": "Recorded food", "count": "Meal records"},
                    title="Most frequently recorded foods",
                ),
                width="stretch",
            )
        else:
            st.info("No food records are available for this period.")
    st.write(f"Meal type counts: {meals['meal_type_distribution'] or 'No meal records.'}")
    st.markdown("**Daily nutrition from recorded meals**")
    st.caption(
        "These are sums across meals present in the record for each date, not necessarily the "
        "patient's complete daily intake."
    )
    nutrition = meals["daily_nutrition_from_recorded_meals"]
    if nutrition:
        st.dataframe(pd.DataFrame(nutrition), hide_index=True, width="stretch")
        st.write("Average nutrition across recorded meals:")
        st.json(meals["average_nutrition_across_recorded_meals"])
    else:
        st.info("No meal nutrition records are available for this period.")
    st.info(
        "Meal timing analysis is skipped when meal timestamps are unavailable. "
        f"{meals['timing_summary']['status']} "
        f"Recorded meal timestamps in this period: {meals['timing_summary']['timed_meal_count']}."
    )
    if meals["timing_summary"]["timed_meal_count"]:
        st.write(
            "Average recorded time by meal type: "
            f"{meals['timing_summary']['average_time_by_meal_type']}"
        )
        st.write(
            "Average gap between recorded meals: "
            f"{_format_value(meals['timing_summary']['average_gap_between_recorded_meals_minutes'], ' minutes')}"
        )

    st.subheader("Glucose analytics")
    readings = get_glucose_trend(
        patient["patient_id"], start_date, end_date, database_path=database_path
    )
    if readings:
        glucose_frame = pd.DataFrame(readings)
        glucose_frame["recorded_at"] = pd.to_datetime(
            glucose_frame["date"].astype(str) + " " + glucose_frame["time"].astype(str),
            errors="coerce",
        )
        glucose_frame = glucose_frame.dropna(subset=["recorded_at"])
        if not glucose_frame.empty:
            st.plotly_chart(
                px.scatter(
                    glucose_frame,
                    x="recorded_at",
                    y="glucose_value",
                    color="context",
                    hover_data=["date", "time"],
                    labels={
                        "recorded_at": "Recorded date/time",
                        "glucose_value": "Recorded glucose (mg/dL)",
                        "context": "Context",
                    },
                    title="Glucose trend — synthetic/demo readings",
                ),
                width="stretch",
            )
            context_col, daily_col = st.columns(2)
            with context_col:
                context_stats = glucose["by_context"]
                if context_stats:
                    st.plotly_chart(
                        px.bar(
                            pd.DataFrame(context_stats),
                            x="context",
                            y="average_recorded_glucose",
                            labels={
                                "context": "Reading context",
                                "average_recorded_glucose": "Average recorded glucose (mg/dL)",
                            },
                            title="Average by recorded context",
                        ),
                        width="stretch",
                    )
                else:
                    st.info("No glucose context values are available.")
            with daily_col:
                daily_stats = glucose["daily_statistics"]
                if daily_stats:
                    st.plotly_chart(
                        px.line(
                            pd.DataFrame(daily_stats),
                            x="date",
                            y="average_recorded_glucose",
                            markers=True,
                            labels={
                                "date": "Date",
                                "average_recorded_glucose": "Average recorded glucose (mg/dL)",
                            },
                            title="Daily average of recorded readings",
                        ),
                        width="stretch",
                    )
                else:
                    st.info("No daily glucose statistics are available.")
        else:
            st.info("No timestamped glucose data is available for charting.")
    else:
        st.info("No glucose data is available for the selected period.")
    st.write(
        f"Minimum: {_format_value(glucose['minimum_recorded_glucose'], ' mg/dL')} · "
        f"Maximum: {_format_value(glucose['maximum_recorded_glucose'], ' mg/dL')} · "
        f"Median: {_format_value(glucose['median_recorded_glucose'], ' mg/dL')} · "
        f"Standard deviation: {_format_value(glucose['standard_deviation_recorded_glucose'], ' mg/dL')}"
    )

    st.subheader("Lifestyle analytics")
    lifestyle_metrics = st.columns(4)
    lifestyle_metrics[0].metric("Average sleep", _format_value(lifestyle["average_sleep_hours"], " h"))
    lifestyle_metrics[1].metric("Average exercise", _format_value(lifestyle["average_exercise_minutes"], " min"))
    lifestyle_metrics[2].metric("Average water intake", _format_value(lifestyle["average_water_intake"], " L"))
    lifestyle_metrics[3].metric("Average stress score", _format_value(lifestyle["average_stress_score"]))
    lifestyle_rows = get_patient_lifestyle(patient["patient_id"], database_path)
    lifestyle_frame = pd.DataFrame([
        row for row in lifestyle_rows if analytics["start_date"] <= row["date"] <= analytics["end_date"]
    ])
    if not lifestyle_frame.empty:
        sleep_col, exercise_col = st.columns(2)
        for column, metric, title, unit in (
            (sleep_col, "sleep_hours", "Recorded sleep", "hours"),
            (exercise_col, "exercise_minutes", "Recorded exercise", "minutes"),
        ):
            with column:
                st.plotly_chart(
                    px.line(
                        lifestyle_frame.sort_values("date"),
                        x="date",
                        y=metric,
                        markers=True,
                        labels={"date": "Date", metric: unit},
                        title=title,
                    ),
                    width="stretch",
                )
        water_col, stress_col = st.columns(2)
        with water_col:
            st.plotly_chart(
                px.line(
                    lifestyle_frame.sort_values("date"),
                    x="date",
                    y="water_intake",
                    markers=True,
                    labels={"date": "Date", "water_intake": "Liters"},
                    title="Recorded water intake",
                ),
                width="stretch",
            )
        with stress_col:
            stress_counts = lifestyle_frame["stress_level"].value_counts().rename_axis("Stress level").reset_index(name="Records")
            st.plotly_chart(
                px.bar(stress_counts, x="Stress level", y="Records", title="Recorded stress categories"),
                width="stretch",
            )
        st.caption(f"Stress score uses an ordinal descriptive mapping: {lifestyle['stress_score_scale']}.")
    else:
        st.info("No lifestyle records are available for the selected period.")

    st.subheader("Exploratory meal-glucose association")
    st.caption("Nearby readings and meals are displayed descriptively; this does not establish causation.")
    associations = analytics["meal_glucose_associations"]
    if associations:
        st.dataframe(pd.DataFrame(associations), hide_index=True, width="stretch")
    else:
        st.info(analytics["data_quality"]["meal_glucose_association_status"])

    with st.expander("Data availability and quality"):
        quality = analytics["data_quality"]
        st.write(f"Meal records: {quality['meal_record_count']}")
        st.write(f"Glucose records: {quality['glucose_record_count']}")
        st.write(f"Lifestyle records: {quality['lifestyle_record_count']}")
        st.write(f"Missing glucose timestamps: {quality['missing_glucose_timestamps']}")
        st.write(f"Missing glucose values: {quality['missing_glucose_values']}")
        st.write(f"Missing meal dates: {quality['missing_meal_dates']}")
        st.write(f"Missing meal information: {quality['missing_meal_information']}")
        st.write(f"Missing meal timestamps: {quality['missing_meal_timestamps']}")
        st.write(f"Dates without any recorded entry: {quality['missing_date_count']}")
        st.write(f"Dates without meal records: {len(quality['dates_without_meal_records'])}")
        st.write(f"Dates without glucose records: {len(quality['dates_without_glucose_records'])}")
        st.write(f"Dates without lifestyle records: {len(quality['dates_without_lifestyle_records'])}")
        st.write(
            "Duplicate records: "
            f"meals {quality['duplicate_meal_ids']}, glucose {quality['duplicate_glucose_ids']}, "
            f"lifestyle {quality['duplicate_lifestyle_ids']}"
        )
        st.caption(quality["note"])


def _show_medical_knowledge() -> None:
    st.title("Medical Knowledge")
    st.warning(
        "Retrieved source excerpts are general educational information, not personalized medical "
        "advice, diagnosis, or treatment. Consult a qualified healthcare professional for medical decisions."
    )
    try:
        report = _initialize_rag_resources()
    except (FileNotFoundError, NotADirectoryError, ValueError, RuntimeError, VectorStoreError) as error:
        st.error(f"The medical knowledge index is unavailable: {error}")
        st.info("Check that source documents and metadata are available, then rebuild the local index.")
        return
    if report["errors"]:
        with st.expander("Document loading warnings"):
            for error in report["errors"]:
                st.warning(error)
    st.caption(
        f"Indexed {report['document_count']} documents as {report['chunk_count']} chunks from "
        f"{report['source_count']} sources · embedding: {report['embedding_model']}"
    )
    st.caption(
        "Hybrid ranking combines local vector similarity with topic/title/term matches. "
        "Ranking is not evidence that a passage is medically correct or sufficient to answer a question."
    )

    with st.form("medical_knowledge_search"):
        question = st.text_area(
            "Enter a medical/nutrition question",
            placeholder="What are general principles of diabetes nutrition?",
        )
        top_k = st.slider("Top results", min_value=1, max_value=10, value=5)
        submitted = st.form_submit_button("Search Knowledge Base", type="primary")
    if st.button("Rebuild knowledge index", help="Re-read local source files and rebuild the persistent vector index."):
        try:
            report = initialize_rag(force_rebuild=True)
        except (FileNotFoundError, NotADirectoryError, ValueError, RuntimeError, VectorStoreError) as error:
            st.error(f"The knowledge index could not be rebuilt: {error}")
            return
        _initialize_rag_resources.clear()
        st.success(f"Rebuilt index: {report['chunk_count']} chunks from {report['document_count']} documents.")
        st.rerun()

    if not submitted:
        return
    if not question.strip():
        st.error("Enter a question before searching.")
        return
    try:
        result = retrieve_context(question, top_k=top_k)
    except (ValueError, RuntimeError, VectorStoreError) as error:
        st.error(f"Knowledge search failed: {error}")
        return
    st.subheader("Retrieved Knowledge")
    if not result["results"]:
        st.info(result["message"])
        return
    for index, item in enumerate(result["results"], start=1):
        metadata = item["metadata"]
        with st.expander(f"{index}. {metadata['title'] or 'Source metadata unavailable'}", expanded=index == 1):
            if not metadata["source"] and not metadata["title"] and not metadata["topic"]:
                st.write("Source metadata unavailable.")
            else:
                st.write(f"**Source:** {metadata['source'] or 'Source metadata unavailable.'}")
                st.write(f"**Title:** {metadata['title'] or 'Source metadata unavailable.'}")
                st.write(f"**Topic:** {metadata['topic'] or 'Source metadata unavailable.'}")
                if metadata["document_type"]:
                    st.write(f"**Document type:** {metadata['document_type']}")
            st.write(
                f"**Relevance score:** {item['score']:.3f} (hybrid ranking; "
                f"vector similarity {item['vector_score']:.3f})"
            )
            if metadata["url"].startswith(("https://", "http://")):
                st.markdown(f"[View original source]({metadata['url']})")
            st.markdown(item["text"])


def _render_assistant_sources(sources: list[dict[str, Any]]) -> None:
    if not sources:
        return
    with st.expander("📚 Medical Sources Used"):
        for source in sources:
            st.write(f"**Source:** {source['source'] or 'Source metadata unavailable'}")
            st.write(f"**Title:** {source['title'] or 'Source metadata unavailable'}")
            st.write(f"**Topic:** {source['topic'] or 'Source metadata unavailable'}")
            if source["url"].startswith(("https://", "http://")):
                st.markdown(f"[View original source]({source['url']})")
            if source["text"]:
                st.text(source["text"])


def _show_ai_assistant(patient_id: str) -> None:
    st.title("🤖 AI Assistant")
    st.warning(
        "Academic prototype / educational self-management assistant. Not a clinical decision-support "
        "system. Do not use it for diagnosis, emergencies, or medication decisions."
    )
    st.caption(f"Selected patient: {patient_id} · Session-only conversation history")
    default_range = (date.today() - timedelta(days=6), date.today())
    chosen_range = st.date_input(
        "Analysis period",
        value=default_range,
        key=f"assistant_date_range_{patient_id}",
    )
    if isinstance(chosen_range, (tuple, list)) and len(chosen_range) == 2:
        start_date, end_date = chosen_range
    elif isinstance(chosen_range, date):
        start_date = end_date = chosen_range
    else:
        st.error("Choose a valid analysis date range.")
        return
    scope = (patient_id, start_date.isoformat(), end_date.isoformat())
    if st.session_state.get("assistant_scope") != scope:
        st.session_state["assistant_scope"] = scope
        st.session_state["assistant_messages"] = []

    try:
        settings = load_settings()
    except LLMConfigurationError as error:
        st.error(str(error))
        return
    if not settings.is_configured:
        st.error("LLM service is not configured. Please configure the API key in the environment.")
        st.info("Set LLM_API_KEY, LLM_MODEL, and LLM_BASE_URL in a local .env file, then restart Streamlit.")
        return
    consent = st.checkbox(
        "I consent to sending my question and the minimum relevant selected-patient context, retrieved "
        "passages, and recent chat history to the configured LLM provider for this session.",
        key=f"assistant_consent_{patient_id}_{start_date.isoformat()}_{end_date.isoformat()}",
    )
    if not consent:
        st.info("Consent is required before any question or patient context is sent to the configured provider.")

    stored_messages = st.session_state.get("assistant_messages", [])
    for message in stored_messages[-settings.max_history_messages:]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant":
                _render_assistant_sources(message.get("sources", []))

    question = st.chat_input(
        "Ask about your recorded meals, glucose, lifestyle, or diabetes education",
        disabled=not consent,
    )
    if question is None:
        return
    if not question.strip():
        st.error("Enter a question before sending it.")
        return

    with st.chat_message("user"):
        st.markdown(question)
    try:
        result = Chatbot().answer(
            patient_id,
            question,
            start_date,
            end_date,
            history=trim_history(stored_messages, settings.max_history_messages),
        )
    except (ChatbotError, LLMError, DatabaseError, ValueError) as error:
        st.error(str(error))
        return
    with st.chat_message("assistant"):
        st.markdown(result["answer"])
        _render_assistant_sources(result["sources"])
        context_used = result["context_used"]
        if context_used:
            with st.expander("📊 Patient Context Used"):
                st.write(f"Patient profile: {'✓' if context_used.get('patient_profile') else 'Not used'}")
                st.write(f"Recent meals: {'✓' if context_used.get('recent_meals') else 'Not used'}")
                st.write(f"Glucose history: {'✓' if context_used.get('glucose_history') else 'Not used'}")
                st.write(f"Lifestyle data: {'✓' if context_used.get('lifestyle_data') else 'Not used'}")
                st.write(f"Analytics: {'✓' if context_used.get('analytics') else 'Not used'}")
                st.write(f"Food reference data: {'✓' if context_used.get('food_nutrition') else 'Not used'}")
                st.write(f"Medical knowledge: {'✓' if result['sources'] else 'No relevant source used'}")
    st.session_state["assistant_messages"] = append_exchange(
        stored_messages,
        question,
        result["answer"],
        max_messages=settings.max_history_messages,
        sources=result["sources"],
        context_used=result["context_used"],
    )
    st.rerun()


def main() -> None:
    st.set_page_config(page_title="GlucoCare · Phase 7", page_icon="🥗", layout="wide")
    st.sidebar.title("GlucoCare")
    st.sidebar.caption("Phases 1–5 · Phase 7 Food Image Analysis")
    try:
        _initialize_app_database()
    except (DataValidationError, DatabaseError) as error:
        st.error(f"GlucoCare could not initialize its database: {error}")
        st.info("Check that all required CSV files are present and valid, then reload the application.")
        st.stop()

    database_path = str(DEFAULT_DATABASE_PATH)
    if st.sidebar.button("Reload CSV data into database"):
        try:
            initialize_database(DEFAULT_DATA_DIR, DEFAULT_DATABASE_PATH, force=True)
        except (DataValidationError, DatabaseError) as error:
            st.sidebar.error(f"CSV data was not imported: {error}")
        else:
            st.sidebar.success("Database refreshed from CSV files.")
            st.rerun()

    try:
        patients = get_all_patients(database_path)
    except DatabaseError as error:
        st.error(f"Unable to load patient profiles: {error}")
        st.stop()
    if not patients:
        st.warning("No patient profiles are available. Add demo records to patients.csv and initialize the database.")
        st.stop()

    selected_id = st.sidebar.selectbox(
        "Select Demo Patient",
        options=[patient["patient_id"] for patient in patients],
        key="selected_patient_id",
    )
    try:
        patient = get_patient(selected_id, database_path)
    except DatabaseError as error:
        st.error(f"Unable to load the selected patient: {error}")
        st.stop()
    if patient is None:
        st.error("The selected patient profile is no longer available. Refresh the patient list.")
        st.stop()

    page = st.sidebar.radio(
        "Navigate",
        [
            "Dashboard",
            "Patient Profile",
            "Meal History",
            "Glucose Trends",
            "Lifestyle",
            "Analytics",
            "Medical Knowledge",
            "🤖 AI Assistant",
            "Meal Recommendations (Coming later)",
            "Food Image Analysis",
            "Food Nutrition",
        ],
    )
    st.sidebar.divider()
    st.sidebar.caption(
        "Education and self-management prototype only. Consult a qualified healthcare professional "
        "for medical decisions."
    )
    try:
        if page == "Dashboard":
            _show_dashboard(patient, database_path)
        elif page == "Patient Profile":
            _show_profile(patient)
        elif page == "Meal History":
            _show_meals(selected_id, database_path)
        elif page == "Glucose Trends":
            _show_glucose(selected_id, database_path)
        elif page == "Lifestyle":
            _show_lifestyle(selected_id, database_path)
        elif page == "Analytics":
            _show_analytics(patient, database_path)
        elif page == "Medical Knowledge":
            _show_medical_knowledge()
        elif page == "🤖 AI Assistant":
            _show_ai_assistant(selected_id)
        elif page == "Food Nutrition":
            _show_foods(database_path)
        elif page == "Food Image Analysis":
            render_food_image_analysis(selected_id, database_path)
        else:
            st.title(page)
            st.info("This module is reserved for a later GlucoCare phase.")
    except DatabaseError as error:
        st.error(f"Unable to load this section: {error}")


if __name__ == "__main__":
    main()
