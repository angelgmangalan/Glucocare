"""Knowledge-base indexing and evidence-only retrieval pipeline."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from src import config
from src.config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    DEFAULT_TOP_K,
    KNOWLEDGE_BASE_PATH,
    MIN_RELEVANCE_SCORE,
    VECTOR_DATABASE_PATH,
)
from src.rag.chunker import chunk_documents
from src.rag.document_cleaner import clean_documents
from src.rag.document_loader import load_documents
from src.rag.embeddings import create_embedding_provider
from src.rag.retriever import Retriever
from src.rag.vector_store import ChromaVectorStore, VectorStoreError

_retriever: Retriever | None = None
_initialization_report: dict[str, Any] | None = None
_store: ChromaVectorStore | None = None


def _corpus_fingerprint(chunks: list[dict[str, Any]], model: str) -> str:
    digest = hashlib.sha256()
    digest.update(model.encode("utf-8"))
    for chunk in chunks:
        digest.update(chunk["chunk_id"].encode("utf-8"))
        digest.update(chunk["text"].encode("utf-8"))
        digest.update(repr(sorted(chunk["metadata"].items())).encode("utf-8"))
    return digest.hexdigest()


def initialize_rag(
    *,
    force_rebuild: bool = False,
    knowledge_base_path: str | Path = KNOWLEDGE_BASE_PATH,
    vector_database_path: str | Path = VECTOR_DATABASE_PATH,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
    embedding_provider: str | None = None,
    embedding_model: str | None = None,
) -> dict[str, Any]:
    """Load and index source documents only when the local corpus/model changes."""
    global _retriever, _initialization_report, _store
    raw_documents, errors = load_documents(knowledge_base_path)
    documents = clean_documents(raw_documents)
    if not documents:
        raise ValueError(
            "The knowledge base contains no readable supported documents. "
            + ("; ".join(errors) if errors else "")
        )
    chunks = chunk_documents(documents, chunk_size, chunk_overlap)
    if not chunks:
        raise ValueError("The knowledge base contains no non-empty text chunks.")
    embeddings = create_embedding_provider(
        embedding_provider or config.EMBEDDING_PROVIDER,
        embedding_model or config.EMBEDDING_MODEL,
    )
    fingerprint = _corpus_fingerprint(chunks, embeddings.model_name)
    requested_path = Path(vector_database_path).resolve()
    if _store is not None and _store.persist_directory.resolve() == requested_path:
        store = _store
    else:
        if _store is not None:
            _store.close()
        store = ChromaVectorStore(requested_path)
    current_metadata = store.index_metadata()
    needs_rebuild = (
        force_rebuild
        or store.count() == 0
        or current_metadata.get("corpus_fingerprint") != fingerprint
        or current_metadata.get("embedding_model") != embeddings.model_name
    )
    if needs_rebuild:
        vectors = embeddings.embed_documents([chunk["text"] for chunk in chunks])
        store.rebuild_index(
            chunks,
            vectors,
            fingerprint=fingerprint,
            embedding_model=embeddings.model_name,
        )
    _store = store
    _retriever = Retriever(store, embeddings)
    _initialization_report = {
        "document_count": len(documents),
        "chunk_count": len(chunks),
        "source_count": len({
            chunk["metadata"].get("source")
            for chunk in chunks
            if chunk["metadata"].get("source")
        }),
        "index_rebuilt": needs_rebuild,
        "embedding_model": embeddings.model_name,
        "errors": errors,
        "vector_database_path": str(vector_database_path),
    }
    return dict(_initialization_report)


def retrieve_context(query: str, top_k: int = DEFAULT_TOP_K) -> dict[str, Any]:
    """Return evidence chunks and provenance; this function does not generate answers."""
    if not isinstance(query, str) or not query.strip():
        raise ValueError("Enter a question to search the medical knowledge base.")
    if _retriever is None:
        initialize_rag()
    assert _retriever is not None
    results = _retriever.retrieve(query, top_k)
    return {
        "query": query.strip(),
        "results": [
            {
                "text": result["text"],
                "source": result["metadata"]["source"],
                "title": result["metadata"]["title"],
                "url": result["metadata"]["url"],
                "topic": result["metadata"]["topic"],
                "document_type": result["metadata"]["document_type"],
                "score": result["score"],
                "vector_score": result["vector_score"],
                "metadata": result["metadata"],
            }
            for result in results
        ],
        "message": (
            None if results
            else "No sufficiently relevant information was found in the current knowledge base."
        ),
    }


def get_initialization_report() -> dict[str, Any] | None:
    return dict(_initialization_report) if _initialization_report is not None else None
