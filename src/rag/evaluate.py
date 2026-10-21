"""Run a small topic-label retrieval evaluation; does not assert factual answers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.config import DEFAULT_TOP_K, PROJECT_ROOT
from src.rag.pipeline import initialize_rag, retrieve_context


DEFAULT_QUERIES_PATH = PROJECT_ROOT / "tests" / "rag_queries.json"


def evaluate_retrieval(
    queries_path: str | Path = DEFAULT_QUERIES_PATH,
    *,
    top_k: int = DEFAULT_TOP_K,
) -> dict[str, Any]:
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 1:
        raise ValueError("top_k must be a positive integer.")
    path = Path(queries_path)
    try:
        queries = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Unable to load retrieval evaluation queries from {path}: {error}") from error
    if not isinstance(queries, list) or not queries:
        raise ValueError(f"{path} must contain a non-empty JSON array.")

    initialize_rag()
    evaluated: list[dict[str, Any]] = []
    hits = 0
    for item in queries:
        question = item.get("question")
        expected_topics = item.get("expected_topics")
        if not isinstance(question, str) or not question.strip() or not isinstance(expected_topics, list):
            raise ValueError(f"Invalid retrieval evaluation entry: {item!r}")
        result = retrieve_context(question, top_k=top_k)
        retrieved_topics = [entry["topic"] for entry in result["results"]]
        hit = any(topic in expected_topics for topic in retrieved_topics)
        hits += int(hit)
        evaluated.append({
            "question": question,
            "expected_topics": expected_topics,
            "retrieved_topics": retrieved_topics,
            "hit": hit,
        })
    return {
        "query_count": len(evaluated),
        "top_k": top_k,
        "hit_count": hits,
        "hit_rate_at_k": hits / len(evaluated),
        "queries": evaluated,
    }


def main() -> None:
    report = evaluate_retrieval()
    print(f"Queries: {report['query_count']}")
    print(f"Top-K: {report['top_k']}")
    print(f"Topic hits: {report['hit_count']}/{report['query_count']}")
    print(f"Measured Hit Rate@{report['top_k']}: {report['hit_rate_at_k']:.1%}")
    for item in report["queries"]:
        print(
            f"- {'HIT' if item['hit'] else 'MISS'} | {item['question']} "
            f"| expected={item['expected_topics']} retrieved={item['retrieved_topics']}"
        )


if __name__ == "__main__":
    main()
