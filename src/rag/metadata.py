"""Knowledge-source metadata loading and validation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

REQUIRED_SOURCE_FIELDS = ("source", "title", "url", "topic", "document_type")


def load_source_metadata(knowledge_base_path: str | Path) -> dict[str, dict[str, Any]]:
    metadata_path = Path(knowledge_base_path) / "metadata" / "sources.json"
    if not metadata_path.is_file():
        return {}
    try:
        data = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Unable to read source metadata at {metadata_path}: {error}") from error
    documents = data.get("documents")
    if not isinstance(documents, list):
        raise ValueError(f"{metadata_path} must contain a 'documents' list.")
    by_path: dict[str, dict[str, Any]] = {}
    for index, record in enumerate(documents):
        if not isinstance(record, dict) or not isinstance(record.get("path"), str):
            raise ValueError(f"Source metadata document {index} must include a relative 'path'.")
        relative_path = Path(record["path"]).as_posix()
        if relative_path in by_path:
            raise ValueError(f"Duplicate source metadata path: {relative_path}")
        normalized = {field: record.get(field, "") for field in REQUIRED_SOURCE_FIELDS}
        normalized.update({
            key: value for key, value in record.items()
            if key not in normalized and key != "path"
        })
        by_path[relative_path] = normalized
    return by_path
