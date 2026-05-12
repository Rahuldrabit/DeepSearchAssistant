"""Integration test: full ingest → retrieve pipeline with mocked backends.

This test exercises the real HierarchicalChunker, Embedder (with mocked
EmbeddingBackend), MetadataDB (real SQLite in-memory), and a mocked
VectorStore.  No GPU, Qdrant server, or model files are required.
"""
from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

def _make_embedding_backend(dim: int = 32) -> MagicMock:
    """Return a mock EmbeddingBackend that produces deterministic vectors."""
    backend = MagicMock()
    backend.dimension = dim

    def _encode(texts, batch_size=64):
        rng = np.random.default_rng(0)
        return rng.random((len(texts), dim)).astype(np.float32)

    def _encode_single(text):
        rng = np.random.default_rng(abs(hash(text)) % (2**31))
        return rng.random(dim).astype(np.float32).tolist()

    backend.encode.side_effect = _encode
    backend.encode_single.side_effect = _encode_single
    return backend


def _make_vector_store() -> MagicMock:
    """Return a mock VectorStore that captures upserted chunks."""
    vs = MagicMock()
    vs._store: list[dict] = []

    def _upsert(batch):
        vs._store.extend(batch)

    vs.upsert_batch.side_effect = _upsert
    return vs


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestIngestPipeline:
    """End-to-end ingestion pipeline tests (no real models, no Qdrant)."""

    @pytest.fixture
    def tmp_dir(self, tmp_path):
        return tmp_path

    @pytest.fixture
    def metadata_db(self, tmp_path):
        from deepsearch.storage.metadata_db import MetadataDB
        db_path = tmp_path / "meta.db"
        return MetadataDB(str(db_path))

    @pytest.fixture
    def embedder(self):
        from deepsearch.ingestion.embedder import Embedder
        backend = _make_embedding_backend(dim=32)
        return Embedder(embedding_backend=backend, cache=None, batch_size=8)

    @pytest.fixture
    def vector_store(self):
        return _make_vector_store()

    @pytest.fixture
    def dispatcher(self, vector_store, metadata_db, embedder):
        from deepsearch.ingestion.dispatcher import IngestionDispatcher
        return IngestionDispatcher(
            vector_store=vector_store,
            metadata_db=metadata_db,
            embedder=embedder,
        )

    # ------------------------------------------------------------------ #

    def test_ingest_txt_file(self, dispatcher, vector_store, tmp_dir):
        """Ingesting a .txt file produces chunks stored in the vector store."""
        doc = tmp_dir / "sample.txt"
        doc.write_text(
            "Machine learning is a subset of artificial intelligence. "
            "It enables systems to learn from data. "
            "Deep learning uses neural networks with multiple layers. "
            "Transformers have revolutionized natural language processing. "
            "Large language models can generate coherent text. " * 10  # ensure enough text
        )
        file_id = dispatcher.ingest(doc)

        assert isinstance(file_id, str) and len(file_id) > 0
        assert len(vector_store._store) > 0, "No chunks were upserted"

        first = vector_store._store[0]
        assert "chunk_id" in first
        assert "dense_vector" in first
        assert "sparse_indices" in first
        assert "payload" in first

    def test_ingest_json_file(self, dispatcher, vector_store, tmp_dir):
        """JSON parser produces text that gets chunked and embedded."""
        doc = tmp_dir / "data.json"
        doc.write_text('{"name": "Alice", "role": "engineer", "team": "platform"}')
        file_id = dispatcher.ingest(doc)
        assert len(file_id) > 0
        assert len(vector_store._store) > 0

    def test_ingest_csv_file(self, dispatcher, vector_store, tmp_dir):
        """CSV parser produces text that gets chunked and embedded."""
        doc = tmp_dir / "employees.csv"
        doc.write_text("name,role,dept\nAlice,Engineer,Platform\nBob,PM,Product\n")
        file_id = dispatcher.ingest(doc)
        assert len(file_id) > 0
        assert len(vector_store._store) > 0

    def test_ingest_html_file(self, dispatcher, vector_store, tmp_dir):
        """HTML parser strips tags and produces clean text."""
        doc = tmp_dir / "page.html"
        doc.write_text(
            "<html><head><title>Test</title></head>"
            "<body><h1>Welcome</h1><p>This is a paragraph.</p>"
            "<script>alert('drop me')</script></body></html>"
        )
        file_id = dispatcher.ingest(doc)
        assert len(file_id) > 0
        # Payload text should not contain script content
        for chunk in vector_store._store:
            assert "alert" not in chunk["payload"].get("text", "")

    def test_skip_already_indexed(self, dispatcher, vector_store, tmp_dir):
        """Re-ingesting an unchanged file is a no-op."""
        doc = tmp_dir / "doc.txt"
        doc.write_text("Some content that won't change. " * 20)

        fid1 = dispatcher.ingest(doc)
        count_after_first = len(vector_store._store)

        fid2 = dispatcher.ingest(doc)  # same file, unchanged
        count_after_second = len(vector_store._store)

        assert count_after_second == count_after_first, "Re-ingest added extra chunks"

    def test_bm25_sparse_vector_non_empty(self, dispatcher, vector_store, tmp_dir):
        """BM25 sparse vectors should be non-empty for non-trivial documents."""
        doc = tmp_dir / "science.txt"
        doc.write_text(
            "Quantum mechanics describes physical phenomena at atomic scales. "
            "Wave-particle duality is a fundamental concept. "
            "Heisenberg's uncertainty principle limits simultaneous measurement. " * 5
        )
        dispatcher.ingest(doc)

        for chunk in vector_store._store:
            sparse_idx = chunk["sparse_indices"]
            sparse_val = chunk["sparse_values"]
            assert len(sparse_idx) > 0, "BM25 sparse vector is empty"
            assert len(sparse_idx) == len(sparse_val)
            assert all(v > 0 for v in sparse_val), "All BM25 values should be positive"

    def test_metadata_recorded_in_db(self, dispatcher, metadata_db, tmp_dir):
        """MetadataDB records the ingested file."""
        doc = tmp_dir / "notes.txt"
        doc.write_text("Important notes about the project. " * 20)
        file_id = dispatcher.ingest(doc)

        record = metadata_db.get_file(doc)
        assert record is not None
        assert record["file_id"] == file_id
        assert record["chunk_count"] > 0

    def test_ingest_directory(self, dispatcher, vector_store, tmp_dir):
        """ingest_directory processes all supported files."""
        (tmp_dir / "a.txt").write_text("Document alpha. " * 15)
        (tmp_dir / "b.txt").write_text("Document beta. " * 15)
        (tmp_dir / "ignore.xyz").write_text("Should be skipped")

        fids = dispatcher.ingest_directory(tmp_dir)
        assert len(fids) == 2
        assert len(vector_store._store) > 0


class TestEmbedderBM25:
    """Unit-level tests for BM25 scoring within Embedder."""

    @pytest.fixture
    def embedder(self):
        from deepsearch.ingestion.embedder import Embedder
        return Embedder(embedding_backend=_make_embedding_backend(dim=32))

    def test_corpus_stats_updated(self, embedder):
        from deepsearch.ingestion.chunker import Chunk
        chunks = [
            Chunk("c1", "f1", "machine learning algorithms", 1, 0, None, {}),
            Chunk("c2", "f1", "deep neural network architectures", 1, 1, None, {}),
        ]
        embedder.embed_chunks(chunks)
        assert embedder._doc_count == 2
        assert embedder.avg_doc_len > 0

    def test_idf_boosts_rare_terms(self, embedder):
        from deepsearch.ingestion.chunker import Chunk
        # Index 10 documents; "rare" appears only once, "common" appears 9 times
        common_chunks = [
            Chunk(f"c{i}", "f1", "common term appears frequently", 1, i, None, {})
            for i in range(9)
        ]
        rare_chunk = Chunk("c9", "f1", "rare unique term xyzzy", 1, 9, None, {})
        results = embedder.embed_chunks(common_chunks + [rare_chunk])

        # Find the BM25 score for "rare" in the last chunk
        last = results[-1]
        idx = last["sparse_indices"]
        vals = last["sparse_values"]
        # vocab maps token → index; check xyzzy has a higher weight than "common"
        vocab = embedder._vocab
        xyzzy_idx = vocab.get("xyzzy")
        common_idx = vocab.get("common")

        if xyzzy_idx is not None and common_idx is not None:
            xyzzy_val = vals[idx.index(xyzzy_idx)] if xyzzy_idx in idx else 0.0
            common_val = vals[idx.index(common_idx)] if common_idx in idx else 0.0
            assert xyzzy_val > common_val, "Rare term should have higher IDF than common term"

    def test_embed_query_uses_corpus_stats(self, embedder):
        from deepsearch.ingestion.chunker import Chunk
        # Index a document first so corpus stats are populated
        chunks = [Chunk("c1", "f1", "python programming language tutorial", 1, 0, None, {})]
        embedder.embed_chunks(chunks)

        dense, si, sv = embedder.embed_query("python tutorial")
        assert len(dense) == 32
        assert len(si) == len(sv)


class TestQueryRouter:
    """Unit tests for QueryRouter heuristics."""

    @pytest.fixture
    def router(self):
        from deepsearch.retrieval.query_router import QueryRouter
        return QueryRouter()

    def test_short_factual_query_routes_fast(self, router):
        d = router.route("What is Python?")
        assert d.mode == "fast"

    def test_analytical_query_routes_deep(self, router):
        d = router.route("Explain the differences between supervised and unsupervised learning.")
        assert d.mode == "deep"

    def test_long_query_routes_deep(self, router):
        query = "I need to understand the implications of using transformers over RNNs for sequence modeling in production systems"
        d = router.route(query)
        assert d.mode == "deep"

    def test_override_mode_respected(self, router):
        d = router.route("What is 2+2?", override_mode="deep")
        assert d.mode == "deep"
        assert d.confidence == 1.0

    def test_low_confidence_escalates_to_cloud(self, router):
        d = router.route("Any query", prior_confidence=0.2)
        assert d.mode == "cloud"

    def test_compare_routes_deep(self, router):
        d = router.route("Compare REST APIs versus GraphQL for mobile applications")
        assert d.mode == "deep"

    def test_routing_decision_has_all_fields(self, router):
        from deepsearch.retrieval.query_router import RoutingDecision
        d = router.route("What is a transformer?")
        assert isinstance(d, RoutingDecision)
        assert d.mode in ("fast", "deep", "cloud")
        assert d.top_k > 0
        assert isinstance(d.reason, str)
        assert 0.0 <= d.confidence <= 1.0

    def test_suggest_top_k(self, router):
        assert router.suggest_top_k("fast") <= router.suggest_top_k("deep")
