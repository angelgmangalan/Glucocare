"""Configurable local embedding providers for medical document retrieval."""

from __future__ import annotations

import hashlib
import math
import re
from typing import Protocol

from src.config import EMBEDDING_MODEL, EMBEDDING_PROVIDER


class EmbeddingProvider(Protocol):
    model_name: str

    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


class LocalHashingEmbeddings:
    """Dependency-free deterministic token/character feature-hashing vectors."""

    model_name = "local-feature-hashing-v1"
    dimensions = 4096

    @staticmethod
    def _features(text: str) -> list[str]:
        normalized = text.casefold()
        words = re.findall(r"[a-z0-9]+|[\u0080-\uffff]+", normalized)
        features = [f"w:{word}" for word in words]
        features.extend(f"b:{left}_{right}" for left, right in zip(words, words[1:]))
        for word in words:
            padded = f"^{word}$"
            features.extend(f"c:{padded[index:index + 3]}" for index in range(max(1, len(padded) - 2)))
        return features

    def embed_query(self, text: str) -> list[float]:
        vector = [0.0] * self.dimensions
        for feature in self._features(text):
            index = int.from_bytes(hashlib.blake2b(feature.encode("utf-8"), digest_size=4).digest(), "big") % self.dimensions
            vector[index] += 1.0
        norm = math.sqrt(sum(value * value for value in vector))
        if norm:
            vector = [value / norm for value in vector]
        return vector

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_query(text) for text in texts]


class SentenceTransformerEmbeddings:
    """Optional offline-capable model provider when its model is installed locally."""

    def __init__(self, model_name: str) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as error:
            raise RuntimeError(
                "The sentence-transformers provider requires the optional "
                "'sentence-transformers' package."
            ) from error
        self._model = SentenceTransformer(model_name)
        self.model_name = model_name

    def embed_query(self, text: str) -> list[float]:
        vector = self._model.encode(text, normalize_embeddings=True)
        return vector.tolist()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        vectors = self._model.encode(texts, normalize_embeddings=True)
        return vectors.tolist()


def create_embedding_provider(
    provider: str = EMBEDDING_PROVIDER,
    model_name: str = EMBEDDING_MODEL,
) -> EmbeddingProvider:
    if provider == "hashing":
        return LocalHashingEmbeddings()
    if provider == "sentence-transformers":
        return SentenceTransformerEmbeddings(model_name)
    raise ValueError(
        f"Unsupported embedding provider {provider!r}. "
        "Choose 'hashing' or 'sentence-transformers'."
    )
