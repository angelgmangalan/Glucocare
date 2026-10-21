"""Orchestrate patient context, local retrieval, the LLM, and safety validation."""

from __future__ import annotations

import hashlib
import logging
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Callable

from src.chatbot.context_builder import build_chat_context
from src.chatbot.prompt_builder import build_messages
from src.chatbot.safety import (
    is_emergency_question,
    is_medication_request,
    is_prompt_injection,
    is_record_lookup_question,
    requires_medical_evidence,
    validate_response,
)
from src.database.database import DEFAULT_DATABASE_PATH
from src.llm.client import LLMClient, LLMError
from src.llm.config import LLMConfigurationError, load_rag_top_k, load_settings
from src.rag.pipeline import retrieve_context

LOGGER = logging.getLogger(__name__)
RetrievalFunction = Callable[..., dict[str, Any]]

MEDICATION_FALLBACK = (
    "I can’t advise you to start, stop, or change a diabetes medicine or dose. "
    "Please discuss medication changes with your prescribing healthcare professional."
)
EMERGENCY_FALLBACK = (
    "If you or someone else may be in immediate danger, contact your local emergency "
    "number now or go to the nearest emergency department. I can’t diagnose the cause."
)
INJECTION_FALLBACK = (
    "I can’t reveal hidden instructions or bypass safety protections. "
    "I can help with an educational health question instead."
)
NO_EVIDENCE_FALLBACK = (
    "I couldn’t find relevant medical evidence in the local knowledge base, so I can’t "
    "provide a grounded medical answer. Please consult a qualified healthcare professional "
    "or use a verified medical source."
)
UNSAFE_RESPONSE_FALLBACK = (
    "I couldn’t safely validate a response to that question. Please rephrase it or discuss "
    "medical decisions with a qualified healthcare professional."
)
EDUCATIONAL_NOTE = (
    "\n\nImportant: This is educational information, not a diagnosis or a substitute "
    "for advice from a qualified healthcare professional."
)


class ChatbotError(RuntimeError):
    """Safe application-level assistant failure."""


def _patient_log_id(patient_id: str) -> str:
    return hashlib.sha256(patient_id.encode("utf-8")).hexdigest()[:12]


def _log(level: int, message: str, *args: Any) -> None:
    timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    LOGGER.log(level, "%s " + message, timestamp, *args)


def _question_category(question: str) -> str:
    lowered = question.casefold()
    if any(word in lowered for word in ("glucose", "blood sugar", "readings")):
        return "glucose"
    if any(word in lowered for word in ("meal", "eat", "food", "breakfast", "lunch", "dinner")):
        return "nutrition"
    if any(word in lowered for word in ("exercise", "sleep", "stress", "lifestyle")):
        return "lifestyle"
    if any(word in lowered for word in ("medicine", "medication", "insulin", "dose")):
        return "medication"
    return "general_health"


def _retrieval_query(question: str) -> str:
    lowered = question.casefold()
    if any(word in lowered for word in (
        "puttu", "appam", "idiyappam", "dosa", "idli", "rice", "kanji",
        "chapati", "sambar", "avial", "thoran", "olan",
    )):
        return f"{question} diabetes nutrition carbohydrate portion meal composition"
    return question


def _source_records(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    sources: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for result in results:
        source = str(result.get("source") or result.get("metadata", {}).get("source") or "").strip()
        title = str(result.get("title") or result.get("metadata", {}).get("title") or "").strip()
        url = str(result.get("url") or result.get("metadata", {}).get("url") or "").strip()
        if not url.startswith(("https://", "http://")):
            url = ""
        key = (source, title, url)
        if key not in seen and (source or title or url):
            seen.add(key)
            sources.append({
                "source": source,
                "title": title,
                "url": url,
                "topic": str(result.get("topic") or result.get("metadata", {}).get("topic") or ""),
                "text": str(result.get("text") or ""),
            })
    return sources


def _append_sources(answer: str, sources: list[dict[str, Any]]) -> str:
    if not sources:
        return answer.strip() + EDUCATIONAL_NOTE
    lines = ["Sources:"]
    for source in sources:
        label = " — ".join(part for part in (source["source"], source["title"]) if part)
        if source["url"]:
            label = f"{label} ({source['url']})"
        lines.append(f"- {label or 'Source metadata unavailable'}")
    return answer.strip() + EDUCATIONAL_NOTE + "\n\n" + "\n".join(lines)


class Chatbot:
    def __init__(
        self,
        llm_client: LLMClient | None = None,
        *,
        retrieval_function: RetrievalFunction = retrieve_context,
        database_path: str | Path = DEFAULT_DATABASE_PATH,
    ) -> None:
        self.llm_client = llm_client
        self.retrieval_function = retrieval_function
        self.database_path = database_path

    def answer(
        self,
        patient_id: str,
        question: str,
        start_date: date | None = None,
        end_date: date | None = None,
        *,
        history: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        if not isinstance(question, str) or not question.strip():
            raise ValueError("Enter a question before sending it to the assistant.")
        if not isinstance(patient_id, str) or not patient_id.strip():
            raise ValueError("Select a patient before sending a question.")
        question = question.strip()
        patient_key = _patient_log_id(patient_id)
        category = _question_category(question)
        base_result: dict[str, Any] = {
            "answer": "",
            "sources": [],
            "context_used": {},
            "safety_status": "not_generated",
        }

        if is_emergency_question(question):
            _log(logging.WARNING, "assistant emergency escalation patient=%s category=%s", patient_key, category)
            return {**base_result, "answer": EMERGENCY_FALLBACK, "safety_status": "emergency_escalated"}
        if is_medication_request(question):
            _log(logging.INFO, "assistant medication refusal patient=%s category=%s", patient_key, category)
            return {**base_result, "answer": MEDICATION_FALLBACK, "safety_status": "medication_refused"}
        if is_prompt_injection(question):
            _log(logging.WARNING, "assistant prompt-injection refusal patient=%s category=%s", patient_key, category)
            return {**base_result, "answer": INJECTION_FALLBACK, "safety_status": "injection_refused"}

        try:
            patient_context = build_chat_context(
                patient_id,
                question,
                start_date,
                end_date,
                database_path=self.database_path,
            )
        except (OSError, ValueError, RuntimeError) as error:
            _log(logging.ERROR, "assistant context failure patient=%s category=%s", patient_key, category)
            raise ChatbotError("Patient context could not be prepared. Please check the selected records and try again.") from error
        if patient_context is None:
            raise ValueError("The selected patient is unavailable. Select a patient and try again.")

        try:
            retrieval = self.retrieval_function(
                _retrieval_query(question),
                top_k=load_rag_top_k(),
            )
            evidence = retrieval.get("results") or []
            _log(
                logging.INFO,
                "assistant retrieval success patient=%s category=%s documents=%d",
                patient_key, category, len(evidence),
            )
        except LLMConfigurationError as error:
            _log(logging.ERROR, "assistant retrieval configuration failure patient=%s category=%s", patient_key, category)
            raise ChatbotError(f"Invalid retrieval configuration: {error}") from error
        except (OSError, RuntimeError, ValueError) as error:
            _log(logging.ERROR, "assistant retrieval failure patient=%s category=%s", patient_key, category)
            if not is_record_lookup_question(question):
                raise ChatbotError(
                    "Medical knowledge retrieval is unavailable. The assistant cannot provide a grounded medical answer right now."
                ) from error
            evidence = []

        medical_claims_requested = (
            requires_medical_evidence(question) and not is_record_lookup_question(question)
        )
        if medical_claims_requested and not evidence:
            _log(logging.INFO, "assistant no-evidence response patient=%s category=%s", patient_key, category)
            return {
                **base_result,
                "answer": NO_EVIDENCE_FALLBACK,
                "context_used": patient_context["context_used"],
                "safety_status": "no_relevant_evidence",
            }

        try:
            client = self.llm_client or LLMClient.from_environment()
        except LLMError:
            _log(logging.ERROR, "assistant LLM configuration failure patient=%s category=%s", patient_key, category)
            raise
        try:
            messages = build_messages(
                question=question,
                patient_context=patient_context,
                retrieved_evidence=evidence,
                history=history,
                max_history_messages=load_settings().max_history_messages,
            )
            generated = client.generate_response(messages)
        except (LLMError, ValueError) as error:
            _log(logging.ERROR, "assistant LLM failure patient=%s category=%s", patient_key, category)
            if isinstance(error, LLMError):
                raise
            raise ChatbotError("The assistant could not prepare a safe request. Please try again.") from error

        sources = _source_records(evidence)
        is_valid, safety_status = validate_response(
            generated,
            patient_context=patient_context,
            requires_sources=medical_claims_requested,
            has_sources=bool(sources),
        )
        _log(
            logging.INFO,
            "assistant safety validation patient=%s category=%s result=%s",
            patient_key, category, safety_status,
        )
        if not is_valid:
            return {
                **base_result,
                "answer": UNSAFE_RESPONSE_FALLBACK,
                "context_used": patient_context["context_used"],
                "safety_status": safety_status,
            }
        _log(logging.INFO, "assistant LLM success patient=%s category=%s", patient_key, category)
        return {
            "answer": _append_sources(generated, sources),
            "sources": sources,
            "context_used": patient_context["context_used"],
            "safety_status": "passed",
        }
