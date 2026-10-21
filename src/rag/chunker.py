"""Paragraph- and sentence-aware text chunking with stable provenance IDs."""

from __future__ import annotations

import hashlib
import re
from typing import Any


def _split_oversized_paragraph(paragraph: str, max_size: int) -> list[str]:
    sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9#])", paragraph)
    pieces: list[str] = []
    current = ""
    for sentence in sentences:
        if len(sentence) <= max_size:
            candidate = f"{current} {sentence}".strip()
            if current and len(candidate) > max_size:
                pieces.append(current)
                current = sentence
            else:
                current = candidate
            continue
        if current:
            pieces.append(current)
            current = ""
        for offset in range(0, len(sentence), max_size):
            piece = sentence[offset:offset + max_size].strip()
            if piece:
                pieces.append(piece)
    if current:
        pieces.append(current)
    return pieces


def chunk_documents(
    documents: list[dict[str, Any]],
    chunk_size: int = 800,
    chunk_overlap: int = 100,
) -> list[dict[str, Any]]:
    if isinstance(chunk_size, bool) or not isinstance(chunk_size, int) or chunk_size < 1:
        raise ValueError("chunk_size must be a positive integer.")
    if isinstance(chunk_overlap, bool) or not isinstance(chunk_overlap, int) or chunk_overlap < 0:
        raise ValueError("chunk_overlap must be a non-negative integer.")
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size.")

    chunks: list[dict[str, Any]] = []
    for document in documents:
        text = str(document.get("text", "")).strip()
        if not text:
            continue
        paragraphs = [part.strip() for part in re.split(r"\n\s*\n", text) if part.strip()]
        units: list[str] = []
        for paragraph in paragraphs:
            units.extend(
                [paragraph] if len(paragraph) <= chunk_size
                else _split_oversized_paragraph(paragraph, chunk_size)
            )
        current = ""
        document_chunks: list[str] = []
        for unit in units:
            candidate = f"{current}\n\n{unit}".strip() if current else unit
            if current and len(candidate) > chunk_size:
                document_chunks.append(current)
                overlap = current[-chunk_overlap:].strip() if chunk_overlap else ""
                current = f"{overlap}\n\n{unit}".strip() if overlap else unit
                if len(current) > chunk_size:
                    current = unit
            else:
                current = candidate
        if current:
            document_chunks.append(current)

        base_metadata = dict(document.get("metadata", {}))
        source = document.get("source") or base_metadata.get("source", "")
        for index, chunk_text in enumerate(document_chunks):
            digest = hashlib.sha256(
                f"{base_metadata.get('source_file', '')}\0{index}\0{chunk_text}".encode("utf-8")
            ).hexdigest()[:24]
            metadata = {
                **base_metadata,
                "source": source,
                "chunk_index": index,
            }
            chunks.append({
                "chunk_id": digest,
                "text": chunk_text,
                "source": source,
                "title": metadata.get("title", ""),
                "url": metadata.get("url", ""),
                "topic": metadata.get("topic", ""),
                "document_type": metadata.get("document_type", ""),
                "metadata": metadata,
            })
    return chunks
