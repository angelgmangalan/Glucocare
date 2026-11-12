"""FastAPI endpoints for the GlucoCare data and assistant services."""

from __future__ import annotations

import os
from datetime import date, timedelta
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, Query
from fastapi.encoders import jsonable_encoder
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from starlette.requests import Request
from starlette.responses import JSONResponse

from src.analytics.patient_analytics import build_patient_analytics
from src.chatbot.chatbot import Chatbot, ChatbotError
from src.database.database import (
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
from src.llm.client import LLMError
from src.rag.pipeline import retrieve_context
from src.rag.vector_store import VectorStoreError
from src.recommendation.engine import generate_meal_recommendations


def get_database_path() -> Path:
    """Initialize the default database deterministically for each API request."""
    initialize_database(database_path=DEFAULT_DATABASE_PATH)
    return DEFAULT_DATABASE_PATH


def _cors_origins() -> list[str]:
    return [
        origin.strip()
        for origin in os.getenv("GLUCOCARE_CORS_ORIGINS", "").split(",")
        if origin.strip()
    ]


app = FastAPI(title="GlucoCare API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins(),
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(DatabaseError)
async def database_error_handler(_: Request, __: DatabaseError) -> JSONResponse:
    return JSONResponse(status_code=500, content={"detail": "Database operation failed."})


@app.exception_handler(DataValidationError)
async def data_validation_error_handler(_: Request, __: DataValidationError) -> JSONResponse:
    return JSONResponse(status_code=500, content={"detail": "Database initialization failed."})


@app.exception_handler(ValueError)
async def domain_validation_error_handler(_: Request, error: ValueError) -> JSONResponse:
    return JSONResponse(status_code=400, content={"detail": str(error)})


async def optional_service_error_handler(_: Request, __: Exception) -> JSONResponse:
    return JSONResponse(status_code=503, content={"detail": "An optional GlucoCare service is unavailable."})


for service_error in (ChatbotError, LLMError, VectorStoreError):
    app.add_exception_handler(service_error, optional_service_error_handler)


class AssistantRequest(BaseModel):
    patient_id: str = Field(min_length=1, max_length=100)
    question: str = Field(min_length=1, max_length=4000)
    start_date: date | None = None
    end_date: date | None = None
    history: list[dict[str, Any]] = Field(default_factory=list, max_length=20)


def _require_patient(patient_id: str, database_path: str | Path) -> dict[str, Any]:
    patient = get_patient(patient_id, database_path)
    if patient is None:
        raise HTTPException(status_code=404, detail="Patient not found.")
    return patient


def _date_range(
    patient_id: str,
    start_date: date | None,
    end_date: date | None,
    database_path: str | Path,
) -> tuple[date, date]:
    patient = _require_patient(patient_id, database_path)
    if end_date is None:
        rows = (
            get_patient_meals(patient_id, database_path)
            + get_patient_glucose(patient_id, database_path)
            + get_patient_lifestyle(patient_id, database_path)
        )
        record_dates = [
            date.fromisoformat(str(row["date"]))
            for row in rows
            if row.get("date")
        ]
        selected_end = max(record_dates) if record_dates else date.today()
    else:
        selected_end = end_date
    selected_start = start_date or (selected_end - timedelta(days=6))
    if selected_start > selected_end:
        raise HTTPException(status_code=400, detail="start_date must be on or before end_date.")
    return selected_start, selected_end


def _records_in_range(
    rows: list[dict[str, Any]],
    start_date: date,
    end_date: date,
) -> list[dict[str, Any]]:
    start, end = start_date.isoformat(), end_date.isoformat()
    return [
        row for row in rows
        if row.get("date") and start <= str(row["date"]) <= end
    ]


@app.get("/api/health")
def health(database_path: Path = Depends(get_database_path)) -> dict[str, str]:
    del database_path
    return {"status": "ok"}


@app.get("/api/patients")
def patients(database_path: Path = Depends(get_database_path)) -> list[dict[str, Any]]:
    return jsonable_encoder(get_all_patients(database_path))


@app.get("/api/patients/{patient_id}")
def patient(
    patient_id: str,
    database_path: Path = Depends(get_database_path),
) -> dict[str, Any]:
    return jsonable_encoder(_require_patient(patient_id, database_path))


@app.get("/api/patients/{patient_id}/dashboard")
def patient_dashboard(
    patient_id: str,
    start_date: date | None = None,
    end_date: date | None = None,
    database_path: Path = Depends(get_database_path),
) -> dict[str, Any]:
    patient_row = _require_patient(patient_id, database_path)
    selected_start, selected_end = _date_range(
        patient_id, start_date, end_date, database_path
    )
    analytics = build_patient_analytics(
        patient_id, selected_start, selected_end, database_path=database_path
    )
    return jsonable_encoder({"patient": patient_row, "analytics": analytics})


@app.get("/api/patients/{patient_id}/meals")
def patient_meals(
    patient_id: str,
    start_date: date | None = None,
    end_date: date | None = None,
    database_path: Path = Depends(get_database_path),
) -> list[dict[str, Any]]:
    start, end = _date_range(patient_id, start_date, end_date, database_path)
    return jsonable_encoder(_records_in_range(get_patient_meals(patient_id, database_path), start, end))


@app.get("/api/patients/{patient_id}/glucose")
def patient_glucose(
    patient_id: str,
    start_date: date | None = None,
    end_date: date | None = None,
    database_path: Path = Depends(get_database_path),
) -> list[dict[str, Any]]:
    start, end = _date_range(patient_id, start_date, end_date, database_path)
    return jsonable_encoder(_records_in_range(get_patient_glucose(patient_id, database_path), start, end))


@app.get("/api/patients/{patient_id}/lifestyle")
def patient_lifestyle(
    patient_id: str,
    start_date: date | None = None,
    end_date: date | None = None,
    database_path: Path = Depends(get_database_path),
) -> list[dict[str, Any]]:
    start, end = _date_range(patient_id, start_date, end_date, database_path)
    return jsonable_encoder(_records_in_range(get_patient_lifestyle(patient_id, database_path), start, end))


@app.get("/api/foods")
def foods(database_path: Path = Depends(get_database_path)) -> list[dict[str, Any]]:
    return jsonable_encoder(get_all_food_information(database_path))


@app.get("/api/patients/{patient_id}/recommendations")
def patient_recommendations(
    patient_id: str,
    database_path: Path = Depends(get_database_path),
) -> list[dict[str, Any]]:
    patient_row = _require_patient(patient_id, database_path)
    return jsonable_encoder(
        generate_meal_recommendations(patient_row, get_all_food_information(database_path))
    )


@app.post("/api/assistant")
def assistant(
    payload: AssistantRequest,
    database_path: Path = Depends(get_database_path),
) -> dict[str, Any]:
    _require_patient(payload.patient_id, database_path)
    start, end = _date_range(
        payload.patient_id, payload.start_date, payload.end_date, database_path
    )
    result = Chatbot(database_path=database_path).answer(
        payload.patient_id,
        payload.question,
        start,
        end,
        history=payload.history,
    )
    return jsonable_encoder(result)


@app.get("/api/medical/search")
def medical_search(
    query: str = Query(min_length=1, max_length=2000),
    top_k: int = Query(default=5, ge=1, le=50),
) -> dict[str, Any]:
    try:
        result = retrieve_context(query, top_k)
    except (OSError, RuntimeError, ValueError) as error:
        raise HTTPException(
            status_code=503,
            detail="Medical knowledge retrieval is unavailable.",
        ) from error
    return jsonable_encoder(result)

