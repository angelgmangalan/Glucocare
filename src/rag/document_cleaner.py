"""Conservative cleanup for extracted source text."""

from __future__ import annotations

import re
from collections import Counter


def clean_text(text: str) -> str:
    """Normalize extraction noise while retaining headings and references."""
    if not isinstance(text, str):
        raise TypeError("Document text must be a string.")
    text = text.replace("\x00", "").replace("\ufeff", "")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    lines = [line.strip() for line in text.splitlines()]
    nonempty = [line for line in lines if line]
    counts = Counter(nonempty)
    cleaned: list[str] = []
    previous_blank = True
    for line in lines:
        if not line:
            if cleaned and not previous_blank:
                cleaned.append("")
            previous_blank = True
            continue
        if counts[line] >= 3 and not line.startswith(("#", "##", "###")):
            if len(line) < 100 and not re.search(r"\b(doi|reference|references|citation)\b", line, re.I):
                continue
        if re.fullmatch(r"(?:page\s*)?\d+\s*(?:of\s*\d+)?", line, re.I):
            continue
        cleaned.append(line)
        previous_blank = False
    return "\n".join(cleaned).strip()


def clean_documents(documents: list[dict]) -> list[dict]:
    """Clean document content without modifying provenance metadata."""
    cleaned: list[dict] = []
    for document in documents:
        text = clean_text(document.get("text", ""))
        if text:
            cleaned.append({**document, "text": text, "metadata": dict(document.get("metadata", {}))})
    return cleaned
