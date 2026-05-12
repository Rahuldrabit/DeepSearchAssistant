"""Tests for Embedder logic — EmbeddingBackend is mocked."""
import numpy as np
import pytest
from unittest.mock import MagicMock, patch

from deepsearch.ingestion.chunker import HierarchicalChunker, Chunk
from deepsearch.ingestion.embedder import Embedder, _tokenize, _bm25_sparse_vector


class TestTokenizer:
    def test_lowercases(self):
        tokens = _tokenize("Hello WORLD")
        assert "hello" in tokens
        assert "world" in tokens

    def test_removes_stopwords(self):
        tokens = _tokenize("the quick brown fox")
        assert "the" not in tokens
        assert "quick" in tokens

    def test_handles_punctuation(self):
        tokens = _tokenize("hello, world!")
        assert "hello" in tokens
        assert "world" in tokens


class TestSparseVector:
    def _make_bm25(self, text: str, vocab: dict, doc_freqs: dict, doc_count: int = 5, avg_doc_len: float = 5.0):
        tokens = _tokenize(text)
        return _bm25_sparse_vector(tokens, vocab, doc_freqs, doc_count, avg_doc_len)

    def test_returns_indices_and_values(self):
        vocab = {"hello": 0, "world": 1, "foo": 2}
        doc_freqs = {"hello": 1, "world": 1}
        idx, vals = self._make_bm25("hello world", vocab, doc_freqs)
        assert 0 in idx   # hello
        assert 1 in idx   # world
        assert 2 not in idx  # foo not in text

    def test_bm25_values_positive(self):
        vocab = {"hello": 0, "world": 1}
        doc_freqs = {"hello": 2, "world": 1}
        idx, vals = self._make_bm25("hello hello world", vocab, doc_freqs)
        assert all(v > 0 for v in vals)

    def test_unknown_tokens_excluded(self):
        vocab = {"known": 0}
        doc_freqs = {}
        idx, vals = self._make_bm25("unknown word here", vocab, doc_freqs)
        assert idx == []
        assert vals == []

    def test_rare_term_higher_idf(self):
        """A term appearing in fewer documents should have a higher IDF score."""
        vocab = {"rare": 0, "common": 1}
        # rare appears in 1 of 10 docs; common appears in 9 of 10 docs
        doc_freqs = {"rare": 1, "common": 9}
        tokens_rare = _tokenize("rare")
        tokens_common = _tokenize("common")
        idx_r, vals_r = _bm25_sparse_vector(tokens_rare, vocab, doc_freqs, 10, 1.0)
        idx_c, vals_c = _bm25_sparse_vector(tokens_common, vocab, doc_freqs, 10, 1.0)
        assert vals_r[0] > vals_c[0], "Rare term IDF should be higher than common term"


class TestEmbedder:
    def _make_backend(self, dim=384):
        backend = MagicMock()
        backend.dimension = dim
        backend.encode = MagicMock(
            side_effect=lambda texts, **kw: np.random.rand(len(texts), dim).astype(np.float32)
        )
        backend.encode_single = MagicMock(
            return_value=[0.1] * dim
        )
        return backend

    def test_embed_chunks_returns_correct_keys(self):
        backend = self._make_backend()
        embedder = Embedder(backend)
        chunker = HierarchicalChunker()
        chunks = chunker.chunk("Hello world. " * 10, file_id="f1")
        level1 = [c for c in chunks if c.level == 1][:3]

        result = embedder.embed_chunks(level1)
        for item in result:
            assert "chunk_id" in item
            assert "dense_vector" in item
            assert "sparse_indices" in item
            assert "sparse_values" in item
            assert "payload" in item

    def test_embed_chunks_dense_dim(self):
        backend = self._make_backend(dim=384)
        embedder = Embedder(backend)
        chunks = [Chunk("c1", "f1", "test text here", 1, 0, None)]
        result = embedder.embed_chunks(chunks)
        assert len(result[0]["dense_vector"]) == 384

    def test_embed_query_returns_tuple(self):
        backend = self._make_backend()
        embedder = Embedder(backend)
        dense, si, sv = embedder.embed_query("what is AI?")
        assert len(dense) == 384
        assert isinstance(si, list)
        assert isinstance(sv, list)

    def test_embed_query_cache_hit(self):
        from deepsearch.storage.cache import EmbeddingCache
        backend = self._make_backend()
        cache = EmbeddingCache()
        embedder = Embedder(backend, cache=cache)

        embedder.embed_query("cached query")
        embedder.embed_query("cached query")
        # encode_single called only once (second call hits cache)
        assert backend.encode_single.call_count == 1

    def test_payload_contains_chunk_fields(self):
        backend = self._make_backend()
        embedder = Embedder(backend)
        chunk = Chunk("cid-1", "fid-1", "some content", 1, 0, "parent-id",
                      metadata={"file_name": "doc.txt"})
        result = embedder.embed_chunks([chunk])
        payload = result[0]["payload"]
        assert payload["chunk_id"] == "cid-1"
        assert payload["file_id"] == "fid-1"
        assert payload["text"] == "some content"
        assert payload["parent_id"] == "parent-id"
        assert payload["file_name"] == "doc.txt"
