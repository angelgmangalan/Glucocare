"""Persistent ChromaDB vector store wrapper."""

from __future__ import annotations

from pathlib import Path
from typing import Any


class VectorStoreError(RuntimeError):
    """Raised when persistent vector-store operations fail."""


def _chroma_symbols() -> tuple[Any, Any]:
    try:
        import chromadb
        from chromadb.config import Settings
    except ImportError as error:
        raise VectorStoreError(
            "ChromaDB is not installed. Install project dependencies with "
            "`pip install -r requirements.txt`."
        ) from error
    return chromadb, Settings


class ChromaVectorStore:
    def __init__(self, persist_directory: str | Path, collection_name: str = "glucocare_medical_kb") -> None:
        chromadb, settings_type = _chroma_symbols()
        self.persist_directory = Path(persist_directory)
        self.collection_name = collection_name
        try:
            self.persist_directory.mkdir(parents=True, exist_ok=True)
            self.client = chromadb.PersistentClient(
                path=str(self.persist_directory),
                settings=settings_type(anonymized_telemetry=False),
            )
            self.collection = self.client.get_or_create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"},
            )
        except Exception as error:
            raise VectorStoreError(f"Unable to open Chroma vector store: {error}") from error

    def create_index(
        self,
        documents: list[dict[str, Any]],
        embeddings: list[list[float]],
        *,
        fingerprint: str,
        embedding_model: str,
    ) -> None:
        if len(documents) != len(embeddings):
            raise ValueError("Each document must have exactly one embedding.")
        try:
            try:
                self.client.delete_collection(self.collection_name)
            except Exception:
                pass
            self.collection = self.client.create_collection(
                name=self.collection_name,
                metadata={
                    "hnsw:space": "cosine",
                    "corpus_fingerprint": fingerprint,
                    "embedding_model": embedding_model,
                },
            )
            self.add_documents(documents, embeddings)
        except VectorStoreError:
            raise
        except Exception as error:
            raise VectorStoreError(f"Unable to create Chroma index: {error}") from error

    def add_documents(
        self,
        documents: list[dict[str, Any]],
        embeddings: list[list[float]],
    ) -> None:
        if len(documents) != len(embeddings):
            raise ValueError("Each document must have exactly one embedding.")
        if not documents:
            return
        try:
            self.collection.add(
                ids=[document["chunk_id"] for document in documents],
                documents=[document["text"] for document in documents],
                metadatas=[_chroma_metadata(document.get("metadata", {})) for document in documents],
                embeddings=embeddings,
            )
        except Exception as error:
            raise VectorStoreError(f"Unable to add documents to Chroma: {error}") from error

    def search(
        self,
        query_embedding: list[float],
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 1:
            raise ValueError("top_k must be a positive integer.")
        try:
            count = self.collection.count()
            if count == 0:
                return []
            result = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=min(top_k, count),
                include=["documents", "metadatas", "distances"],
            )
            documents = result["documents"][0]
            metadatas = result["metadatas"][0]
            distances = result["distances"][0]
            ids = result["ids"][0]
            return [
                {
                    "chunk_id": ids[index],
                    "text": documents[index],
                    "score": max(0.0, min(1.0, 1.0 - float(distances[index]))),
                    "metadata": metadatas[index] or {},
                }
                for index in range(len(documents))
            ]
        except Exception as error:
            raise VectorStoreError(f"Unable to search Chroma index: {error}") from error

    def rebuild_index(
        self,
        documents: list[dict[str, Any]],
        embeddings: list[list[float]],
        *,
        fingerprint: str,
        embedding_model: str,
    ) -> None:
        self.create_index(
            documents,
            embeddings,
            fingerprint=fingerprint,
            embedding_model=embedding_model,
        )

    def delete_index(self) -> None:
        try:
            self.client.delete_collection(self.collection_name)
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"},
            )
        except Exception as error:
            raise VectorStoreError(f"Unable to delete Chroma index: {error}") from error

    def index_metadata(self) -> dict[str, Any]:
        return dict(self.collection.metadata or {})

    def count(self) -> int:
        try:
            return int(self.collection.count())
        except Exception as error:
            raise VectorStoreError(f"Unable to count Chroma documents: {error}") from error

    def close(self) -> None:
        try:
            self.client.close()
        except Exception as error:
            raise VectorStoreError(f"Unable to close Chroma vector store: {error}") from error


def _chroma_metadata(metadata: dict[str, Any]) -> dict[str, str | int | float | bool]:
    result: dict[str, str | int | float | bool] = {}
    for key, value in metadata.items():
        if value is None or isinstance(value, (list, dict)):
            continue
        if isinstance(value, (str, int, float, bool)):
            result[key] = value
    return result
