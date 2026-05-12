"""Tests for Reciprocal Rank Fusion merge."""
import pytest
from deepsearch.retrieval.hybrid_search import _rrf_merge


def make_hits(ids: list[str]) -> list[dict]:
    return [{"id": i, "score": 1.0 / (n + 1), "payload": {"text": i}} for n, i in enumerate(ids)]


class TestRRFMerge:
    def test_empty_lists(self):
        result = _rrf_merge([], [])
        assert result == []

    def test_single_list_dense_only(self):
        dense = make_hits(["a", "b", "c"])
        result = _rrf_merge(dense, [])
        ids = [r["id"] for r in result]
        assert ids == ["a", "b", "c"]

    def test_single_list_sparse_only(self):
        sparse = make_hits(["x", "y"])
        result = _rrf_merge([], sparse)
        assert result[0]["id"] == "x"

    def test_agrees_score_higher(self):
        """Doc ranked first in both lists should win."""
        dense = make_hits(["winner", "loser_d"])
        sparse = make_hits(["winner", "loser_s"])
        result = _rrf_merge(dense, sparse)
        assert result[0]["id"] == "winner"

    def test_deduplicated(self):
        """A doc appearing in both lists should appear once in output."""
        dense = make_hits(["a", "b"])
        sparse = make_hits(["b", "c"])
        result = _rrf_merge(dense, sparse)
        ids = [r["id"] for r in result]
        assert len(ids) == len(set(ids))

    def test_formula_k60(self):
        """Verify the RRF formula: score = 1/(60+rank+1) for rank 0."""
        dense = make_hits(["only"])
        result = _rrf_merge(dense, [])
        expected = 1.0 / (60 + 0 + 1)
        assert abs(result[0]["score"] - expected) < 1e-9

    def test_k_parameter(self):
        dense = make_hits(["a"])
        r60 = _rrf_merge(dense, [], k=60)
        r10 = _rrf_merge(dense, [], k=10)
        # Lower k → higher score
        assert r10[0]["score"] > r60[0]["score"]

    def test_payload_preserved(self):
        dense = [{"id": "x", "score": 0.9, "payload": {"text": "hello", "file_id": "f1"}}]
        result = _rrf_merge(dense, [])
        assert result[0]["payload"]["text"] == "hello"
        assert result[0]["payload"]["file_id"] == "f1"

    def test_sorting_descending(self):
        dense = make_hits(["a", "b", "c", "d"])
        result = _rrf_merge(dense, [])
        scores = [r["score"] for r in result]
        assert scores == sorted(scores, reverse=True)

    def test_cross_list_boost(self):
        """Doc in both lists outranks doc only in one list (same rank)."""
        # "shared" is rank-0 in dense, rank-0 in sparse → 2 * (1/61)
        # "only_dense" is rank-1 in dense → 1/62
        dense = make_hits(["shared", "only_dense"])
        sparse = make_hits(["shared", "only_sparse"])
        result = _rrf_merge(dense, sparse)
        assert result[0]["id"] == "shared"
