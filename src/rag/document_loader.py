"""Load supported public knowledge documents with explicit provenance."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.config import KNOWLEDGE_BASE_PATH
from src.rag.metadata import load_source_metadata

SUPPORTED_SUFFIXES = {".pdf", ".txt", ".md", ".markdown"}


def load_documents(
    directory: str | Path = KNOWLEDGE_BASE_PATH,
) -> tuple[list[dict[str, Any]], list[str]]:
    """Load PDF, text, and Markdown documents; return documents and per-file errors."""
    root = Path(directory)
    if not root.exists():
        raise FileNotFoundError(f"Knowledge-base directory does not exist: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"Knowledge-base path is not a directory: {root}")
    source_metadata = load_source_metadata(root)
    documents: list[dict[str, Any]] = []
    errors: list[str] = []

    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_SUFFIXES:
            continue
        relative_path = path.relative_to(root).as_posix()
        metadata = dict(source_metadata.get(relative_path, {}))
        for field in ("source", "title", "url", "topic", "document_type"):
            metadata.setdefault(field, "")
        metadata["source_file"] = relative_path
        try:
            if path.suffix.lower() == ".pdf":
                try:
                    from pypdf import PdfReader
                except ImportError as error:
                    errors.append(f"{relative_path}: PDF support requires pypdf.")
                    continue
                try:
                    reader = PdfReader(str(path), strict=False)
                    for page_number, page in enumerate(reader.pages, start=1):
                        text = page.extract_text() or ""
                        if text.strip():
                            page_metadata = {**metadata, "page": page_number}
                            documents.append({
                                "text": text,
                                "metadata": page_metadata,
                                "source": metadata.get("source", ""),
                            })
                except Exception as error:
                    errors.append(f"{relative_path}: could not extract PDF text ({error}).")
                    continue
            else:
                text = path.read_text(encoding="utf-8-sig")
                if text.strip():
                    documents.append({
                        "text": text,
                        "metadata": metadata,
                        "source": metadata.get("source", ""),
                    })
        except (OSError, UnicodeError) as error:
            errors.append(f"{relative_path}: could not read document ({error}).")
    return documents, errors
