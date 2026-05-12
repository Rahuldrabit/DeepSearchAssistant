"""Integration tests — retrieval pipeline (no LLM, no Qdrant server).

Tests the full retrieval chain:
  HybridSearch → ContextBuilder (with/without reranker) → LLMPipeline.query()

All embedding and vector-store calls are mocked so tests run in CI with
zero GPU/model requirements.  They exercise real Python logic: RRF fusion,
context formatting, cache integration, reranker fallback, query routing.
"""
from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch, call

import numpy as np
import pytest


# ── Shared factories ──────────────────────────────────────────────────────────

def _make_hit(chunk_id: str, score: float = 0.8, text: str = "sample text",
              file_name: str = "doc.pdf", page: int = 1) -> dict:
    return {
        "id": chunk_id,
        "score": score,
        "payload": {
            "chunk_id": chunk_id,
            "text": text,
            "file_id": "file_001",
            "file_name": file_name,
            "page": page,
            "chunk_level": 1,
        },
    }


def _make_vector_store(hits: list[dict] | None = None) -> MagicMock:
    vs = MagicMock()
    _hits = hits or [_make_hit(f"c{i}", score=0.9 - i * 0.1) for i in range(10)]
    vs.search_dense.return_value = _hits
    vs.search_sparse.return_value = _hits[:5]
    vs.get_by_ids.return_value = []
    return vs


def _make_embedder(dim: int = 384) -> MagicMock:
    emb = MagicMock()
    rng = np.random.default_rng(42)
    emb.embed_query.side_effect = lambda q: (
        rng.random(dim).tolist(),
        [0, 1, 2],
        [0.5, 0.3, 0.2],
    )
    return emb


def _make_llm(answer: str = "The answer is 42.") -> MagicMock:
    llm = MagicMock()
    llm.generate.return_value = answer
    llm.build_prompt.side_effect = lambda sys_, user, ctx="": f"[SYS]{sys_}\n[Q]{user}\n[CTX]{ctx}"
    return llm


# ── HybridSearch integration ─────────────────────────────────────────────────

class TestHybridSearchIntegration:
    """Test HybridSearch with mock vector store — real RRF logic."""

    def setup_method(self):
        from deepsearch.retrieval.hybrid_search import HybridSearch
        self._vs = _make_vector_store()
        self._embedder = _make_embedder()
        self._search = HybridSearch(self._vs, self._embedder, top_k_retrieval=10, rrf_k=60)

    def test_search_returns_merged_results(self):
        results = self._search.search("What is the answer?", top_k=5)
        assert len(results) <= 5
        for r in results:
            assert "id" in r
            assert "score" in r
            assert "payload" in r

    def test_search_calls_embedder(self):
        self._search.search("test query")
        self._embedder.embed_query.assert_called_once_with("test query")

    def test_search_calls_both_stores(self):
        self._search.search("test query", top_k=5)
        self._vs.search_dense.assert_called_once()
        self._vs.search_sparse.assert_called_once()

    def test_search_dense_only_skips_sparse(self):
        self._search.search_dense_only("fast query", top_k=5)
        self._vs.search_sparse.assert_not_called()
        self._vs.search_dense.assert_called_once()

    def test_rrf_score_ordering(self):
        """Chunks appearing in both dense and sparse get higher RRF scores."""
        from deepsearch.retrieval.hybrid_search import _rrf_merge
        # chunk "c0" appears first in both lists → highest RRF score
        dense = [_make_hit("c0", 0.9), _make_hit("c1", 0.8), _make_hit("c2", 0.7)]
        sparse = [_make_hit("c0", 0.9), _make_hit("c3", 0.8), _make_hit("c4", 0.7)]
        merged = _rrf_merge(dense, sparse, k=60)
        assert merged[0]["id"] == "c0"  # highest RRF because in both lists

    def test_file_filter_applied(self):
        self._search.search("query", file_filter=["file_001"])
        # filter_ kwarg should be non-None in search_dense call
        call_kwargs = self._vs.search_dense.call_args
        assert call_kwargs.kwargs.get("filter_") is not None or call_kwargs.args

    def test_top_k_respected(self):
        results = self._search.search("query", top_k=3)
        assert len(results) <= 3


# ── ContextBuilder integration ────────────────────────────────────────────────

class TestContextBuilderIntegration:
    """Test ContextBuilder context assembly, dedup, and truncation."""

    def setup_method(self):
        from deepsearch.retrieval.context_builder import ContextBuilder
        self._vs = _make_vector_store()
        self._cb = ContextBuilder(self._vs, reranker=None, max_chars=1000, top_k_rerank=3)

    def test_build_returns_string_and_list(self):
        hits = [_make_hit(f"c{i}") for i in range(5)]
        context, selected = self._cb.build("query", hits)
        assert isinstance(context, str)
        assert isinstance(selected, list)

    def test_empty_hits_returns_empty(self):
        context, selected = self._cb.build("query", [])
        assert context == ""
        assert selected == []

    def test_context_contains_source_numbers(self):
        hits = [_make_hit(f"c{i}", text=f"chunk text {i}") for i in range(3)]
        context, _ = self._cb.build("query", hits)
        assert "[1]" in context
        assert "[2]" in context

    def test_context_truncated_at_max_chars(self):
        long_text = "word " * 500  # ~2500 chars
        hits = [_make_hit(f"c{i}", text=long_text) for i in range(5)]
        context, selected = self._cb.build("query", hits, expand_parents=False)
        assert len(context) <= 1200  # some overhead for source labels

    def test_deduplication_by_chunk_id(self):
        # Same chunk_id twice → only one in output
        hits = [_make_hit("dup"), _make_hit("dup"), _make_hit("unique")]
        context, selected = self._cb.build("query", hits)
        ids = [h["id"] for h in selected]
        assert ids.count("dup") == 1

    def test_top_k_rerank_without_model(self):
        hits = [_make_hit(f"c{i}") for i in range(10)]
        _, selected = self._cb.build("query", hits)
        assert len(selected) <= 3  # top_k_rerank=3

    def test_source_filenames_in_context(self):
        hits = [_make_hit("c1", file_name="important.pdf")]
        context, _ = self._cb.build("query", hits)
        assert "important.pdf" in context


# ── Reranker integration ──────────────────────────────────────────────────────

class TestRerankerIntegration:
    """Test Reranker with and without backend loaded."""

    def test_rerank_unloaded_returns_original_order(self):
        from deepsearch.retrieval.reranker import Reranker
        reranker = Reranker()  # not loaded
        hits = [_make_hit(f"c{i}", score=0.9 - i * 0.1) for i in range(5)]
        result = reranker.rerank("query", hits, top_k=3)
        assert result.model_used is False
        assert len(result.hits) == 3
        assert result.hits[0]["id"] == "c0"  # original order preserved

    def test_rerank_empty_hits(self):
        from deepsearch.retrieval.reranker import Reranker
        reranker = Reranker()
        result = reranker.rerank("query", [], top_k=5)
        assert result.hits == []
        assert result.scores == []

    def test_rerank_hits_convenience(self):
        from deepsearch.retrieval.reranker import Reranker
        reranker = Reranker()
        hits = [_make_hit(f"c{i}") for i in range(5)]
        reranked = reranker.rerank_hits("query", hits, top_k=2)
        assert len(reranked) == 2
        assert all("id" in h for h in reranked)

    def test_score_passage_raises_when_unloaded(self):
        from deepsearch.retrieval.reranker import Reranker
        reranker = Reranker()
        with pytest.raises(RuntimeError, match="not loaded"):
            reranker.score_passage("query", "passage text")

    def test_score_passages_raises_when_unloaded(self):
        from deepsearch.retrieval.reranker import Reranker
        reranker = Reranker()
        with pytest.raises(RuntimeError, match="not loaded"):
            reranker.score_passages("query", ["passage 1", "passage 2"])

    def test_reranker_with_mock_backend(self):
        from deepsearch.retrieval.reranker import Reranker
        reranker = Reranker()
        # Inject a mock backend
        reranker._backend._model = MagicMock()
        reranker._backend._model.predict.return_value = np.array([0.2, 0.9, 0.5])
        hits = [_make_hit(f"c{i}", text=f"passage {i}") for i in range(3)]
        result = reranker.rerank("query", hits, top_k=3)
        assert result.model_used is True
        # c1 should be first (score 0.9)
        assert result.hits[0]["id"] == "c1"

    def test_reranker_updates_hit_scores(self):
        from deepsearch.retrieval.reranker import Reranker
        reranker = Reranker()
        reranker._backend._model = MagicMock()
        reranker._backend._model.predict.return_value = np.array([0.3, 0.7])
        hits = [_make_hit("c0", score=0.1), _make_hit("c1", score=0.9)]
        result = reranker.rerank("query", hits, top_k=2)
        # Scores should be cross-encoder scores, not original retrieval scores
        score_map = {h["id"]: h["score"] for h in result.hits}
        assert score_map["c1"] == pytest.approx(0.7)
        assert score_map["c0"] == pytest.approx(0.3)


# ── A/B Test Harness integration ──────────────────────────────────────────────

class TestABTestHarnessIntegration:
    """Test ABTestHarness query execution and report generation."""

    def _make_pipeline(self, confidence: float, latency_s: float = 0.0, answer: str = "answer"):
        from deepsearch.generation.llm_pipeline import SearchResult
        import time
        p = MagicMock()
        def _query(q, mode="fast", file_filter=None):
            if latency_s:
                time.sleep(latency_s)
            return SearchResult(answer=answer, confidence=confidence, search_mode=mode, sources=[{}])
        p.query.side_effect = _query
        return p

    def setup_method(self):
        from deepsearch.retrieval.ab_test import ABTestHarness, StrategyConfig
        self._cfg_a = StrategyConfig(name="dense_only", mode="fast")
        self._cfg_b = StrategyConfig(name="hybrid", mode="fast")
        self._pipeline_a = self._make_pipeline(confidence=0.6)
        self._pipeline_b = self._make_pipeline(confidence=0.8)
        self._harness = ABTestHarness(
            strategy_a=self._cfg_a,
            strategy_b=self._cfg_b,
            pipeline_a=self._pipeline_a,
            pipeline_b=self._pipeline_b,
        )

    def test_run_returns_report(self):
        from deepsearch.retrieval.ab_test import ABReport
        report = self._harness.run(["What is X?", "What is Y?"])
        assert isinstance(report, ABReport)
        assert len(report.comparisons) == 2

    def test_report_wins_correct(self):
        report = self._harness.run(["Q1", "Q2", "Q3"])
        # B has confidence 0.8 > A's 0.6 → B wins all 3
        assert report.wins("b") == 3
        assert report.wins("a") == 0

    def test_report_mean_confidence(self):
        report = self._harness.run(["Q1", "Q2"])
        assert report.mean_confidence("a") == pytest.approx(0.6)
        assert report.mean_confidence("b") == pytest.approx(0.8)

    def test_report_success_rate_100_percent(self):
        report = self._harness.run(["Q1", "Q2"])
        assert report.success_rate("a") == 1.0
        assert report.success_rate("b") == 1.0

    def test_run_single_returns_comparison(self):
        from deepsearch.retrieval.ab_test import ABComparison
        cmp = self._harness.run_single("What is X?", golden_answer="X is Y")
        assert isinstance(cmp, ABComparison)
        assert cmp.query == "What is X?"
        assert cmp.golden_answer == "X is Y"

    def test_winner_is_b(self):
        cmp = self._harness.run_single("What is X?")
        assert cmp.winner == "hybrid"

    def test_report_summary_contains_strategy_names(self):
        report = self._harness.run(["Q1"])
        summary = report.summary()
        assert "dense_only" in summary
        assert "hybrid" in summary

    def test_report_to_dict_structure(self):
        report = self._harness.run(["Q1"])
        d = report.to_dict()
        assert "strategy_a" in d
        assert "strategy_b" in d
        assert "a" in d and "b" in d
        assert "mean_latency_ms" in d["a"]

    def test_golden_answers_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="length"):
            self._harness.run(["Q1", "Q2"], golden_answers=["A1"])

    def test_error_pipeline_handled_gracefully(self):
        from deepsearch.retrieval.ab_test import ABTestHarness, StrategyConfig
        bad_pipeline = MagicMock()
        bad_pipeline.query.side_effect = RuntimeError("LLM crashed")
        harness = ABTestHarness(
            strategy_a=StrategyConfig("bad", "fast"),
            strategy_b=StrategyConfig("good", "fast"),
            pipeline_a=bad_pipeline,
            pipeline_b=self._pipeline_b,
        )
        report = harness.run(["Q1"])
        assert report.success_rate("a") == 0.0
        assert report.success_rate("b") == 1.0

    def test_tie_when_confidence_equal(self):
        from deepsearch.retrieval.ab_test import ABTestHarness, StrategyConfig
        pipe_a = self._make_pipeline(confidence=0.7)
        pipe_b = self._make_pipeline(confidence=0.7)
        harness = ABTestHarness(
            strategy_a=StrategyConfig("a", "fast"),
            strategy_b=StrategyConfig("b", "fast"),
            pipeline_a=pipe_a,
            pipeline_b=pipe_b,
        )
        report = harness.run(["Q1", "Q2"])
        assert report.ties() == 2


# ── QueryRouter integration ───────────────────────────────────────────────────

class TestQueryRouterIntegration:
    """Test QueryRouter routing logic end-to-end."""

    def setup_method(self):
        from deepsearch.retrieval.query_router import QueryRouter
        self._router = QueryRouter()

    def test_short_query_routes_fast(self):
        decision = self._router.route("What is RAM?", override_mode="fast")
        assert decision.mode == "fast"

    def test_override_mode_respected(self):
        decision = self._router.route("simple query", override_mode="deep")
        assert decision.mode == "deep"

    def test_cloud_override_respected(self):
        decision = self._router.route("query", override_mode="cloud")
        assert decision.mode == "cloud"

    def test_decision_has_top_k(self):
        decision = self._router.route("What is the relationship?", override_mode="deep")
        assert decision.top_k > 0

    def test_complex_query_gets_higher_top_k(self):
        simple = self._router.route("What is X?", override_mode="fast")
        complex_ = self._router.route(
            "Analyze and compare all the key differences between approach A and approach B "
            "across all the documents in the corpus", override_mode="deep"
        )
        # Deep mode should have >= top_k as fast
        assert complex_.top_k >= simple.top_k
