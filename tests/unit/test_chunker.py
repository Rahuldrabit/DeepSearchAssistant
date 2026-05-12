"""Tests for hierarchical text chunker."""
import pytest
from deepsearch.ingestion.chunker import HierarchicalChunker, _split_by_sentences


class TestSplitBySentences:
    def test_short_text_single_segment(self):
        result = _split_by_sentences("Hello world.", max_chars=100)
        assert len(result) == 1
        assert result[0] == "Hello world."

    def test_splits_on_sentence_boundary(self):
        text = "First sentence. Second sentence. Third sentence."
        # small max_chars forces splits
        result = _split_by_sentences(text, max_chars=30)
        assert len(result) >= 2
        assert all(len(s) <= 50 for s in result)  # none massively oversized

    def test_hard_splits_very_long_sentence(self):
        long = "A" * 200
        result = _split_by_sentences(long, max_chars=50)
        assert len(result) >= 4
        assert all(len(s) <= 50 for s in result)

    def test_empty_text(self):
        result = _split_by_sentences("", max_chars=100)
        assert result == []


class TestHierarchicalChunker:
    def setup_method(self):
        self.chunker = HierarchicalChunker(
            parent_size_chars=400,
            chunk_size_chars=200,
            child_size_chars=80,
            overlap_chars=20,
        )

    def test_returns_chunks(self):
        text = "This is a test document. " * 30
        chunks = self.chunker.chunk(text, file_id="f1")
        assert len(chunks) > 0

    def test_chunk_levels_present(self):
        text = "Sentence one. Sentence two. Sentence three. " * 20
        chunks = self.chunker.chunk(text, file_id="f2")
        levels = {c.level for c in chunks}
        # Should have at least level 0 and 1
        assert 0 in levels
        assert 1 in levels

    def test_parent_child_links(self):
        text = "Lorem ipsum dolor sit amet. " * 40
        chunks = self.chunker.chunk(text, file_id="f3")

        # All level-1 chunks have a parent_id that exists
        parent_ids = {c.chunk_id for c in chunks if c.level == 0}
        for chunk in chunks:
            if chunk.level == 1:
                assert chunk.parent_id in parent_ids

    def test_file_id_propagated(self):
        chunks = self.chunker.chunk("Some text. " * 20, file_id="abc-123")
        assert all(c.file_id == "abc-123" for c in chunks)

    def test_chunk_ids_unique(self):
        chunks = self.chunker.chunk("Word. " * 50, file_id="uniq")
        ids = [c.chunk_id for c in chunks]
        assert len(ids) == len(set(ids))

    def test_metadata_passed_through(self):
        meta = {"file_name": "test.txt", "author": "Alice"}
        chunks = self.chunker.chunk("Content. " * 20, file_id="m1", base_metadata=meta)
        for chunk in chunks:
            assert chunk.metadata["file_name"] == "test.txt"
            assert chunk.metadata["author"] == "Alice"

    def test_token_estimate(self):
        chunk = self.chunker.chunk("A B C D " * 20, file_id="t1")[0]
        # 4 chars per token heuristic
        assert chunk.token_estimate > 0

    def test_empty_text_produces_no_chunks(self):
        chunks = self.chunker.chunk("", file_id="empty")
        assert chunks == []

    def test_single_sentence(self):
        chunks = self.chunker.chunk("Just one sentence.", file_id="one")
        assert len(chunks) >= 1
        assert any("Just one sentence." in c.text for c in chunks)
