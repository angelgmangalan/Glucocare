"""Central configuration for the GlucoCare prototype."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")


def _configured_path(variable: str, default: Path) -> Path:
    value = Path(os.getenv(variable, str(default))).expanduser()
    return value if value.is_absolute() else PROJECT_ROOT / value


KNOWLEDGE_BASE_PATH = _configured_path("GLUCOCARE_KNOWLEDGE_BASE", PROJECT_ROOT / "knowledge_base")
VECTOR_DATABASE_PATH = _configured_path("GLUCOCARE_VECTOR_DB", PROJECT_ROOT / "vector_db")
EMBEDDING_PROVIDER = os.getenv("GLUCOCARE_EMBEDDING_PROVIDER", "hashing").strip().lower()
EMBEDDING_MODEL = os.getenv(
    "GLUCOCARE_EMBEDDING_MODEL",
    "local-feature-hashing-v1" if EMBEDDING_PROVIDER == "hashing" else "sentence-transformers/all-MiniLM-L6-v2",
)
CHUNK_SIZE = int(os.getenv("GLUCOCARE_CHUNK_SIZE", "800"))
CHUNK_OVERLAP = int(os.getenv("GLUCOCARE_CHUNK_OVERLAP", "100"))
DEFAULT_TOP_K = int(os.getenv("GLUCOCARE_TOP_K", "5"))
MIN_RELEVANCE_SCORE = float(os.getenv("GLUCOCARE_MIN_RELEVANCE_SCORE", "0.08"))
