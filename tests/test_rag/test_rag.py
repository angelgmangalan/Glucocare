from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from src.rag.chunker import chunk_documents
from src.rag.document_cleaner import clean_text
from src.rag.document_loader import load_documents
from src.rag.embeddings import LocalHashingEmbeddings, create_embedding_provider
from src.rag.pipeline import initialize_rag, retrieve_context
from src.rag.retriever import Retriever
from src.rag.vector_store import ChromaVectorStore


class RagPipelineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.kb = self.root / "knowledge_base"
        (self.kb / "diabetes").mkdir(parents=True)
        (self.kb / "nutrition").mkdir()
        (self.kb / "metadata").mkdir()
        (self.kb / "diabetes" / "type2.txt").write_text(
            "Type 2 diabetes concerns blood glucose and insulin. "
            "The document describes glucose management topics.",
            encoding="utf-8",
        )
        (self.kb / "nutrition" / "fiber.md").write_text(
            "# Fiber in food\n\nDietary fiber is found in plant foods. "
            "This document discusses fiber and general nutrition.",
            encoding="utf-8",
        )
        metadata = {
            "documents": [
                {
                    "path": "diabetes/type2.txt",
                    "source": "Test Health Authority",
                    "title": "Type 2 overview",
                    "url": "https://example.org/type2",
                    "topic": "diabetes",
                    "document_type": "health information",
                },
                {
                    "path": "nutrition/fiber.md",
                    "source": "Test Nutrition Authority",
                    "title": "Fiber overview",
                    "url": "https://example.org/fiber",
                    "topic": "nutrition",
                    "document_type": "health information",
                },
            ]
        }
        (self.kb / "metadata" / "sources.json").write_text(
            json.dumps(metadata),
            encoding="utf-8",
        )
        self.vector_path = self.root / "vector"

    def tearDown(self) -> None:
        from src.rag import pipeline

        for store in getattr(self, "_stores", []):
            store.close()
        if pipeline._store is not None:
            pipeline._store.close()
            pipeline._store = None
            pipeline._retriever = None
            pipeline._initialization_report = None
        self.temporary_directory.cleanup()

    def _new_store(self, path: Path) -> ChromaVectorStore:
        store = ChromaVectorStore(path)
        if not hasattr(self, "_stores"):
            self._stores: list[ChromaVectorStore] = []
        self._stores.append(store)
        return store

    def test_document_loading_and_source_metadata(self) -> None:
        documents, errors = load_documents(self.kb)
        self.assertEqual(errors, [])
        self.assertEqual(len(documents), 2)
        type2 = next(document for document in documents if "Type 2" in document["text"])
        self.assertEqual(type2["metadata"]["source"], "Test Health Authority")
        self.assertEqual(type2["metadata"]["url"], "https://example.org/type2")

    def test_corrupt_pdf_is_reported_without_discarding_other_documents(self) -> None:
        (self.kb / "diabetes" / "broken.pdf").write_bytes(b"not a valid pdf")
        documents, errors = load_documents(self.kb)
        self.assertEqual(len(documents), 2)
        self.assertEqual(len(errors), 1)
        self.assertIn("could not extract PDF text", errors[0])

    def test_document_cleaning_preserves_heading_and_references(self) -> None:
        cleaned = clean_text(
            "# Fiber\n\nRepeated header\nRepeated header\nRepeated header\n"
            "A long-\nterm note.\nReferences: source 1.\nPage 2"
        )
        self.assertIn("# Fiber", cleaned)
        self.assertIn("long-\nterm", cleaned)
        self.assertIn("References: source 1.", cleaned)
        self.assertNotIn("Page 2", cleaned)
        self.assertNotIn("Repeated header", cleaned)

    def test_chunking_bounds_sizes_and_preserves_metadata(self) -> None:
        doc = {
            "text": "## Section one\n\n" + "Short sentence. " * 30,
            "source": "Test Authority",
            "metadata": {"source": "Test Authority", "title": "Test title", "topic": "nutrition"},
        }
        chunks = chunk_documents([doc], chunk_size=100, chunk_overlap=15)
        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(len(chunk["text"]) <= 100 for chunk in chunks))
        self.assertTrue(all(chunk["metadata"]["topic"] == "nutrition" for chunk in chunks))
        self.assertEqual(len({chunk["chunk_id"] for chunk in chunks}), len(chunks))

    def test_local_embedding_creation_is_deterministic(self) -> None:
        provider = create_embedding_provider("hashing", "ignored-model-name")
        vector = provider.embed_query("What are carbohydrates?")
        self.assertEqual(len(vector), LocalHashingEmbeddings.dimensions)
        self.assertEqual(vector, provider.embed_query("What are carbohydrates?"))
        self.assertEqual(len(provider.embed_documents(["fiber", "protein"])), 2)

    def test_vector_store_persistence_search_and_rebuild(self) -> None:
        documents, _ = load_documents(self.kb)
        chunks = chunk_documents(documents, chunk_size=300, chunk_overlap=20)
        embedder = LocalHashingEmbeddings()
        vectors = embedder.embed_documents([chunk["text"] for chunk in chunks])
        store = self._new_store(self.vector_path)
        store.create_index(
            chunks,
            vectors,
            fingerprint="test-corpus",
            embedding_model=embedder.model_name,
        )
        result = store.search(embedder.embed_query("fiber in foods"), 2)
        self.assertEqual(len(result), 2)
        self.assertTrue(any(item["metadata"]["topic"] == "nutrition" for item in result))
        self.assertEqual(store.index_metadata()["corpus_fingerprint"], "test-corpus")
        reopened = self._new_store(self.vector_path)
        self.assertEqual(reopened.count(), len(chunks))
        reopened.delete_index()
        self.assertEqual(reopened.count(), 0)

    def test_retrieval_metadata_and_empty_query(self) -> None:
        report = initialize_rag(
            force_rebuild=True,
            knowledge_base_path=self.kb,
            vector_database_path=self.vector_path,
            chunk_size=300,
            chunk_overlap=20,
            embedding_provider="hashing",
        )
        self.assertEqual(report["document_count"], 2)
        result = retrieve_context("What does fiber mean?", top_k=2)
        self.assertTrue(result["results"])
        self.assertTrue(any(item["metadata"]["source"] == "Test Nutrition Authority" for item in result["results"]))
        top_result = retrieve_context("What are fiber and nutrition?", top_k=1)
        self.assertEqual(len(top_result["results"]), 1)
        self.assertEqual(top_result["results"][0]["metadata"]["topic"], "nutrition")
        with self.assertRaisesRegex(ValueError, "Enter a question"):
            retrieve_context("  ")

    def test_no_results_for_unrelated_query(self) -> None:
        report = initialize_rag(
            force_rebuild=True,
            knowledge_base_path=self.kb,
            vector_database_path=self.root / "empty-results-vector",
            chunk_size=300,
            chunk_overlap=20,
            embedding_provider="hashing",
        )
        self.assertGreater(report["chunk_count"], 0)
        store = self._new_store(self.root / "irrelevant-vector")
        chunks = chunk_documents([{
            "text": "Insulin and blood glucose are topics in diabetes care.",
            "source": "Test",
            "metadata": {"source": "Test", "title": "Test", "topic": "diabetes"},
        }])
        embedder = LocalHashingEmbeddings()
        store.create_index(
            chunks, embedder.embed_documents([chunk["text"] for chunk in chunks]),
            fingerprint="irrelevant", embedding_model=embedder.model_name,
        )
        retriever = Retriever(store, embedder, min_relevance_score=0.99)
        self.assertEqual(retriever.retrieve("lunar geology and distant galaxies"), [])

    def test_missing_source_metadata_is_not_fabricated(self) -> None:
        from src.rag.document_loader import load_documents

        no_metadata = self.root / "no-metadata"
        no_metadata.mkdir()
        (no_metadata / "notes.txt").write_text("A short document about diabetes.", encoding="utf-8")
        documents, _ = load_documents(no_metadata)
        chunk = chunk_documents(documents)[0]
        self.assertEqual(chunk["metadata"]["source"], "")
        self.assertEqual(chunk["metadata"]["title"], "")


if __name__ == "__main__":
    unittest.main()
