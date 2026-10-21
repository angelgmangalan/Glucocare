"""Conservative candidate ranking against existing food reference records."""

from __future__ import annotations

import re
import unicodedata
from difflib import SequenceMatcher
from typing import Any


def normalize_label(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).casefold()
    return " ".join(re.findall(r"[\w]+", normalized, flags=re.UNICODE))


def rank_food_matches(
    label: str,
    records: list[dict[str, Any]],
    *,
    limit: int = 5,
) -> list[dict[str, Any]]:
    """Rank possible matches without automatically selecting or substituting one."""
    query = normalize_label(label)
    if not query:
        return []
    ranked: list[tuple[float, dict[str, Any]]] = []
    for record in records:
        names = [
            normalize_label(str(record.get(field, "")))
            for field in ("food_name", "regional_name")
        ]
        names = [name for name in names if name]
        if not names:
            continue
        if query in names:
            score = 1.0
        else:
            query_words = set(query.split())
            name_words = set(" ".join(names).split())
            overlap = len(query_words & name_words) / max(len(query_words), len(name_words))
            similarity = max(SequenceMatcher(None, query, name).ratio() for name in names)
            score = max(overlap, similarity * 0.8)
        if score >= 0.45:
            ranked.append((score, record))
    ranked.sort(key=lambda pair: (-pair[0], str(pair[1].get("food_name", "")).casefold()))
    return [record for _, record in ranked[:limit]]
