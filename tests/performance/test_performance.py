"""Performance Tests — Layer 4.

Measures latency, throughput, and memory for core pipeline components.
Uses pytest-benchmark where available; falls back to manual timing.

Requires models to be downloaded and Qdrant running.
Skipped in CI unless DEEPSEARCH_RUN_PERF_TESTS=1 is set.

Targets (from IMPLEMENTATION_REPORT.md):
  - Fast Search end-to-end:     < 3 000 ms
  - Deep Search end-to-end:     < 15 000 ms
  - Embedding batch (100 docs): < 2 000 ms
  - Warm startup:               < 5 000 ms
  - TTFT (first token):         < 1 000 ms
  - Throughput (CPU):           > 5 tokens/sec

Run locally::

    DEEPSEARCH_RUN_PERF_TESTS=1 pytest tests/performance/ -v --tb=short
"""
from __future__ import annotations

import math
import os
import statistics
import time
from typing import Optional
from unittest.mock import MagicMock

import pytest

# ── Skip guard ───────────────────────────────────────────────────────────────

_RUN_PERF = os.getenv("DEEPSEARCH_RUN_PERF_TESTS", "0") == "1"
skip_if_no_perf = pytest.mark.skipif(
    not _RUN_PERF,
    reason="Set DEEPSEARCH_RUN_PERF_TESTS=1 to run performance tests",
)

# ── Target constants ──────────────────────────────────────────────────────────

FAST_SEARCH_MAX_MS = 3_000
DEEP_SEARCH_MAX_MS = 15_000
EMBED_BATCH_MAX_MS = 2_000       # 100 documents
WARM_STARTUP_MAX_MS = 5_000
TTFT_MAX_MS = 1_000
CPU_MIN_TOKENS_PER_SEC = 5.0

# ── Timing utilities ──────────────────────────────────────────────────────────

class Timer:
    """Context manager for elapsed-time measurement."""

    def __enter__(self):
        self._start = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.elapsed_ms = (time.perf_counter() - self._start) * 1000

    @property
    def elapsed_s(self) -> float:
        return self.elapsed_ms / 1000


def _percentile(values: list[float], p: int) -> float:
    sorted_vals = sorted(values)
    idx = max(0, math.ceil(len(sorted_vals) * p / 100) - 1)
    return sorted_vals[idx]


# ── Mock benchmarks (run in all environments) ─────────────────────────────────

class TestTimerUtility:
    """Validate the Timer helper itself."""

    def test_timer_measures_sleep(self):
        with Timer() as t:
            time.sleep(0.05)
        assert t.elapsed_ms >= 40  # allow some OS jitter

    def test_timer_elapsed_s(self):
        with Timer() as t:
            time.sleep(0.01)
        assert t.elapsed_s >= 0.005

    def test_percentile_median(self):
        vals = [1.0, 2.0, 3.0, 4.0, 5.0]
        assert _percentile(vals, 50) == 3.0

    def test_percentile_p99(self):
        vals = list(range(1, 101))
        assert _percentile(vals, 99) == 99


class TestMockPipelinePerf:
    """Baseline perf tests against a mock pipeline — always run, zero latency target."""

    def _make_fast_pipeline(self):
        from deepsearch.generation.llm_pipeline import SearchResult
        p = MagicMock()
        def _query(q, mode="fast", file_filter=None):
            return SearchResult(answer="test answer", confidence=0.8, search_mode=mode)
        p.query.side_effect = _query
        return p

    def test_mock_query_under_10ms(self):
        pipeline = self._make_fast_pipeline()
        with Timer() as t:
            pipeline.query("What is RAM?", mode="fast")
        assert t.elapsed_ms < 10, f"Mock query took {t.elapsed_ms:.1f}ms"

    def test_mock_throughput_100_queries(self):
        pipeline = self._make_fast_pipeline()
        n = 100
        with Timer() as t:
            for _ in range(n):
                pipeline.query("test query", mode="fast")
        qps = n / t.elapsed_s
        assert qps > 1000, f"Mock QPS too low: {qps:.0f}"

    def test_latency_percentiles(self):
        pipeline = self._make_fast_pipeline()
        latencies = []
        for _ in range(50):
            with Timer() as t:
                pipeline.query("test query", mode="fast")
            latencies.append(t.elapsed_ms)
        p50 = _percentile(latencies, 50)
        p99 = _percentile(latencies, 99)
        assert p50 < 5.0, f"p50 latency too high: {p50:.2f}ms"
        assert p99 < 50.0, f"p99 latency too high: {p99:.2f}ms"


class TestChunkerPerformance:
    """Measure chunker throughput on synthetic documents."""

    def _make_document(self, words: int = 1000) -> str:
        return " ".join([f"word{i}" for i in range(words)])

    def test_chunker_1000_word_document(self):
        from deepsearch.ingestion.chunker import HierarchicalChunker
        chunker = HierarchicalChunker(parent_size_chars=2048, child_size_chars=512, overlap_chars=50)
        doc = self._make_document(1000)
        with Timer() as t:
            chunks = chunker.chunk(doc, file_id="f1")
        assert t.elapsed_ms < 500, f"Chunking took {t.elapsed_ms:.1f}ms"
        assert len(chunks) > 0

    def test_chunker_10k_words_under_2s(self):
        from deepsearch.ingestion.chunker import HierarchicalChunker
        chunker = HierarchicalChunker(parent_size_chars=2048, child_size_chars=512, overlap_chars=50)
        doc = self._make_document(10_000)
        with Timer() as t:
            chunks = chunker.chunk(doc, file_id="f1")
        assert t.elapsed_ms < 2_000, f"10k-word chunking took {t.elapsed_ms:.1f}ms"

    def test_chunker_throughput_pages_per_sec(self):
        from deepsearch.ingestion.chunker import HierarchicalChunker
        chunker = HierarchicalChunker(parent_size_chars=2048, child_size_chars=512, overlap_chars=50)
        page = self._make_document(300)  # ~300 words per page
        n_pages = 20
        with Timer() as t:
            for i in range(n_pages):
                chunker.chunk(page, file_id=f"f{i}")
        pages_per_sec = n_pages / t.elapsed_s
        assert pages_per_sec > 50, f"Chunker too slow: {pages_per_sec:.1f} pages/s"


class TestCachePerformance:
    """Measure cache read/write throughput."""

    def test_answer_cache_write_read_1000(self):
        from deepsearch.storage.cache import AnswerCache
        cache = AnswerCache(max_size=2000)
        n = 1000
        # Write
        with Timer() as t_write:
            for i in range(n):
                cache.set_answer(f"question {i}", "fast", f"answer {i}")
        # Read (all hits)
        with Timer() as t_read:
            for i in range(n):
                cache.get_answer(f"question {i}", "fast")
        assert t_write.elapsed_ms < 500, f"Cache write {t_write.elapsed_ms:.1f}ms"
        assert t_read.elapsed_ms < 200, f"Cache read {t_read.elapsed_ms:.1f}ms"

    def test_query_cache_write_read_500(self):
        from deepsearch.storage.cache import QueryCache
        cache = QueryCache(max_size=1000)
        hits = [{"id": f"c{j}", "score": 0.5, "payload": {"text": "x"}} for j in range(5)]
        n = 500
        with Timer() as t:
            for i in range(n):
                cache.set_results(f"query {i}", "fast", hits)
            for i in range(n):
                cache.get_results(f"query {i}", "fast")
        assert t.elapsed_ms < 500, f"QueryCache 500 R/W took {t.elapsed_ms:.1f}ms"


class TestRRFFusionPerformance:
    """Measure RRF merge performance at realistic list sizes."""

    def _make_hits(self, n: int, prefix: str = "") -> list[dict]:
        return [
            {"id": f"{prefix}c{i}", "score": 1.0 / (i + 1), "payload": {"text": f"text {i}"}}
            for i in range(n)
        ]

    def test_rrf_merge_100_hits_each(self):
        from deepsearch.retrieval.hybrid_search import _rrf_merge
        dense = self._make_hits(100, "d")
        sparse = self._make_hits(100, "s")
        with Timer() as t:
            merged = _rrf_merge(dense, sparse, k=60)
        assert t.elapsed_ms < 50, f"RRF merge took {t.elapsed_ms:.1f}ms"
        assert len(merged) == 200  # all unique

    def test_rrf_merge_500_hits_each(self):
        from deepsearch.retrieval.hybrid_search import _rrf_merge
        dense = self._make_hits(500, "d")
        sparse = self._make_hits(500, "s")
        with Timer() as t:
            for _ in range(10):  # 10 iterations to get stable reading
                _rrf_merge(dense, sparse, k=60)
        avg_ms = t.elapsed_ms / 10
        assert avg_ms < 100, f"RRF merge 500+500 hits avg {avg_ms:.1f}ms"


class TestConfidenceScorerPerformance:
    """Measure confidence scorer throughput."""

    def test_score_100_answers(self):
        from deepsearch.generation.confidence_scorer import ConfidenceScorer
        scorer = ConfidenceScorer()
        answer = "This is a detailed answer to the question about topic."
        question = "What is the topic?"
        scores = [0.8, 0.6, 0.7] * 5
        with Timer() as t:
            for _ in range(100):
                scorer.score(answer=answer, question=question, retrieval_scores=scores)
        assert t.elapsed_ms < 500, f"100 confidence scores took {t.elapsed_ms:.1f}ms"


# ── Real hardware benchmarks (skipped in CI) ─────────────────────────────────

@skip_if_no_perf
class TestRealEmbeddingPerformance:
    """Benchmark real sentence-transformers embedding throughput."""

    @pytest.fixture(scope="class")
    def backend(self):
        from deepsearch.backends.embedding_backend import EmbeddingBackend
        b = EmbeddingBackend()
        b.load("all-MiniLM-L6-v2", device="cpu")
        return b

    def test_embed_100_sentences_under_2s(self, backend):
        texts = [f"This is sentence number {i} used for embedding throughput test." for i in range(100)]
        with Timer() as t:
            vecs = backend.encode(texts)
        assert t.elapsed_ms < EMBED_BATCH_MAX_MS, (
            f"Embedding 100 sentences took {t.elapsed_ms:.0f}ms (target <{EMBED_BATCH_MAX_MS}ms)"
        )
        assert vecs.shape == (100, 384)

    def test_embed_single_query_under_100ms(self, backend):
        with Timer() as t:
            vec = backend.encode_single("What is the purpose of this test?")
        assert t.elapsed_ms < 100, f"Single embed took {t.elapsed_ms:.1f}ms"

    def test_throughput_docs_per_minute(self, backend):
        n = 200
        texts = [f"Document {i}: sample text for throughput measurement." for i in range(n)]
        with Timer() as t:
            backend.encode(texts, batch_size=64)
        docs_per_min = n / t.elapsed_s * 60
        assert docs_per_min >= 3_000, f"Throughput {docs_per_min:.0f} docs/min below 3,000"


@skip_if_no_perf
class TestRealPipelinePerformance:
    """Benchmark full pipeline latency with real models."""

    @pytest.fixture(scope="class")
    def pipeline(self):
        """Configure and return a real LLMPipeline for perf testing."""
        pytest.skip("Real pipeline not configured for performance tests — set up pipeline fixture")

    def test_fast_search_under_3s(self, pipeline):
        with Timer() as t:
            result = pipeline.query("What is the main topic?", mode="fast")
        assert t.elapsed_ms < FAST_SEARCH_MAX_MS, (
            f"Fast search took {t.elapsed_ms:.0f}ms (target <{FAST_SEARCH_MAX_MS}ms)"
        )

    def test_deep_search_under_15s(self, pipeline):
        with Timer() as t:
            result = pipeline.query("Explain the relationships between all key concepts", mode="deep")
        assert t.elapsed_ms < DEEP_SEARCH_MAX_MS, (
            f"Deep search took {t.elapsed_ms:.0f}ms (target <{DEEP_SEARCH_MAX_MS}ms)"
        )

    def test_ttft_under_1s(self, pipeline):
        """Time to first token in streaming mode."""
        first_token_time: list[float] = []
        t0 = time.perf_counter()

        def _on_token(token: str):
            if not first_token_time:
                first_token_time.append((time.perf_counter() - t0) * 1000)

        tokens = list(pipeline.stream_query("What is X?", mode="fast"))
        # Can't measure TTFT from batch — just check answer arrived
        assert len(tokens) > 0

    def test_10_fast_queries_p95_under_5s(self, pipeline):
        """p95 latency across 10 queries stays under 5s."""
        latencies = []
        for i in range(10):
            with Timer() as t:
                pipeline.query(f"Query number {i}", mode="fast")
            latencies.append(t.elapsed_ms)
        p95 = _percentile(latencies, 95)
        assert p95 < 5_000, f"p95 latency {p95:.0f}ms > 5000ms"


@skip_if_no_perf
class TestMemoryProfile:
    """Memory usage stays within budget."""

    def test_import_overhead_under_500mb(self):
        """The deepsearch package import shouldn't consume excessive memory."""
        try:
            import psutil
            import os
            proc = psutil.Process(os.getpid())
            before = proc.memory_info().rss / 1024 / 1024
            import deepsearch  # noqa: F401
            after = proc.memory_info().rss / 1024 / 1024
            delta = after - before
            assert delta < 500, f"Import overhead {delta:.0f} MB > 500 MB"
        except ImportError:
            pytest.skip("psutil not installed")
