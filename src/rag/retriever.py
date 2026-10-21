"""Retrieve relevant evidence chunks while preserving document provenance."""

from __future__ import annotations

import re
from typing import Any

from src.config import DEFAULT_TOP_K, MIN_RELEVANCE_SCORE
from src.rag.embeddings import EmbeddingProvider
from src.rag.vector_store import ChromaVectorStore, VectorStoreError

STOP_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "can", "does", "for",
    "how", "i", "in", "is", "it", "of", "on", "or", "the", "to",
    "what", "when", "where", "which", "why", "with", "you", "your",
}
TOPIC_TERMS = {
    "nutrition": {
        "nutrition", "carbohydrate", "carbohydrates", "carb", "carbs",
        "fiber", "fibre", "portion", "meal", "meals", "eating", "food",
        "foods", "diet", "dietary", "plate", "macronutrient",
    },
    "lifestyle": {
        "activity", "exercise", "physical", "sleep", "stress", "smoking",
        "tobacco", "lifestyle", "movement",
    },
}


class Retriever:
    def __init__(
        self,
        vector_store: ChromaVectorStore,
        embeddings: EmbeddingProvider,
        *,
        min_relevance_score: float = MIN_RELEVANCE_SCORE,
    ) -> None:
        self.vector_store = vector_store
        self.embeddings = embeddings
        self.min_relevance_score = min_relevance_score

    def retrieve(self, query: str, top_k: int = DEFAULT_TOP_K) -> list[dict[str, Any]]:
        if not isinstance(query, str) or not query.strip():
            raise ValueError("Enter a question to search the medical knowledge base.")
        if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 1:
            raise ValueError("top_k must be a positive integer.")
        vector = self.embeddings.embed_query(query.strip())
        results = self.vector_store.search(vector, max(top_k * 4, top_k))
        relevant: list[dict[str, Any]] = []
        query_terms = _terms(query)
        requested_topic = _requested_topic(query_terms)
        for result in results:
            metadata = result.get("metadata") or {}
            text_terms = _terms(
                result["text"] + " " + str(metadata.get("title", "")) + " "
                + str(metadata.get("topic", ""))
            )
            coverage = (
                len(query_terms.intersection(text_terms)) / len(query_terms)
                if query_terms else 0.0
            )
            title_terms = _terms(str(metadata.get("title", "")))
            title_match = (
                len(query_terms.intersection(title_terms)) / len(query_terms)
                if query_terms else 0.0
            )
            topic_match = float(
                bool(requested_topic)
                and str(metadata.get("topic", "")).casefold() == requested_topic
            )
            vector_score = float(result["score"])
            score = 0.35 * vector_score + 0.35 * coverage + 0.10 * title_match + 0.20 * topic_match
            if score < self.min_relevance_score:
                continue
            relevant.append({
                "text": result["text"],
                "score": score,
                "vector_score": vector_score,
                "metadata": {
                    "source": metadata.get("source", ""),
                    "title": metadata.get("title", ""),
                    "url": metadata.get("url", ""),
                    "topic": metadata.get("topic", ""),
                    "document_type": metadata.get("document_type", ""),
                    "source_file": metadata.get("source_file", ""),
                    "chunk_id": result.get("chunk_id", ""),
                },
            })
        return sorted(relevant, key=lambda result: result["score"], reverse=True)[:top_k]


def _terms(text: str) -> set[str]:
    return {
        term for term in re.findall(r"[a-z0-9]+", text.casefold())
        if term not in STOP_WORDS
    }


def _requested_topic(terms: set[str]) -> str:
    for topic in ("nutrition", "lifestyle"):
        if terms.intersection(TOPIC_TERMS[topic]):
            return topic
    return "diabetes" if terms.intersection({"diabetes", "glucose", "insulin", "blood", "a1c"}) else ""
