"""Tests for ContextBuilder formatting and truncation."""
import pytest
from unittest.mock import MagicMock
from deepsearch.retrieval.context_builder import ContextBuilder


def make_hit(chunk_id: str, text: str, file_name: str = "doc.txt", level: int = 1) -> dict:
    return {
        "id": chunk_id,
        "score": 0.9,
        "payload": {
            "chunk_id": chunk_id,
            "text": text,
            "file_name": file_name,
            "chunk_level": level,
            "parent_id": None,
        },
    }


@pytest.fixture
def builder():
    vs = MagicMock()
    return ContextBuilder(vector_store=vs, reranker=None, max_chars=500, top_k_rerank=3)


class TestContextBuilder:
    def test_empty_hits(self, builder):
        ctx, selected = builder.build("query", [])
        assert ctx == ""
        assert selected == []

    def test_formats_source_header(self, builder):
        hits = [make_hit("c1", "This is the content.", "report.pdf")]
        ctx, selected = builder.build("query", hits)
        assert "[1] Source: report.pdf" in ctx
        assert "This is the content." in ctx

    def test_multiple_chunks_numbered(self, builder):
        hits = [
            make_hit("c1", "First passage.", "a.pdf"),
            make_hit("c2", "Second passage.", "b.pdf"),
        ]
        ctx, selected = builder.build("query", hits)
        assert "[1]" in ctx
        assert "[2]" in ctx

    def test_max_chars_truncates(self):
        vs = MagicMock()
        builder = ContextBuilder(vs, reranker=None, max_chars=50, top_k_rerank=5)
        hits = [make_hit(f"c{i}", "A" * 40, "doc.txt") for i in range(5)]
        ctx, selected = builder.build("query", hits)
        assert len(ctx) <= 100  # well within truncation range
        assert len(selected) < 5  # not all chunks included

    def test_deduplication(self, builder):
        # Same chunk_id twice
        hit = make_hit("dup", "Some text.", "f.txt")
        ctx, selected = builder.build("query", [hit, hit])
        assert ctx.count("[1] Source:") == 1
        assert len(selected) == 1

    def test_no_reranker_uses_top_k(self):
        vs = MagicMock()
        builder = ContextBuilder(vs, reranker=None, max_chars=10000, top_k_rerank=2)
        hits = [make_hit(f"c{i}", f"Text {i}", "doc.txt") for i in range(10)]
        _, selected = builder.build("query", hits)
        assert len(selected) <= 2

    def test_with_reranker(self):
        vs = MagicMock()
        reranker = MagicMock()
        reranker.is_loaded = True
        # Reranker reverses the order: index 1 scores highest
        reranker.rerank = MagicMock(return_value=[(1, 0.95), (0, 0.4)])

        builder = ContextBuilder(vs, reranker=reranker, max_chars=10000, top_k_rerank=2)
        hits = [
            make_hit("c0", "First text.", "a.txt"),
            make_hit("c1", "Second text.", "b.txt"),
        ]
        ctx, selected = builder.build("query", hits)
        # After reranking, c1 should appear first ([1])
        assert selected[0]["id"] == "c1"
