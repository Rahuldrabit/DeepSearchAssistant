"""A/B testing harness for retrieval strategy comparison.

Enables side-by-side comparison of two retrieval strategies (e.g.
dense-only vs. hybrid, with-reranker vs. without) using a shared set of
test queries and optional golden answers.

Usage::

    from deepsearch.retrieval.ab_test import ABTestHarness, StrategyConfig

    harness = ABTestHarness(
        strategy_a=StrategyConfig(name="dense_only", mode="fast"),
        strategy_b=StrategyConfig(name="hybrid_rerank", mode="deep"),
        pipeline_a=pipeline_a,
        pipeline_b=pipeline_b,
    )
    report = harness.run(queries=["What is X?", "How does Y work?"])
    print(report.summary())
"""
from __future__ import annotations

import logging
import statistics
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class StrategyConfig:
    """Identifies one side of the A/B test."""
    name: str
    mode: str = "fast"               # "fast" | "deep" | "cloud"
    description: str = ""


@dataclass
class QueryOutcome:
    """Result for one query under one strategy."""
    query: str
    strategy_name: str
    answer: str
    latency_ms: float
    confidence: float
    num_sources: int
    error: Optional[str] = None

    @property
    def success(self) -> bool:
        return self.error is None


@dataclass
class ABComparison:
    """Side-by-side comparison for one query."""
    query: str
    golden_answer: Optional[str]
    outcome_a: QueryOutcome
    outcome_b: QueryOutcome

    @property
    def winner(self) -> Optional[str]:
        """Return strategy name with higher confidence, or None if tied."""
        if not self.outcome_a.success and not self.outcome_b.success:
            return None
        if not self.outcome_a.success:
            return self.outcome_b.strategy_name
        if not self.outcome_b.success:
            return self.outcome_a.strategy_name
        if abs(self.outcome_a.confidence - self.outcome_b.confidence) < 0.01:
            return None  # tie
        return (
            self.outcome_a.strategy_name
            if self.outcome_a.confidence > self.outcome_b.confidence
            else self.outcome_b.strategy_name
        )

    @property
    def latency_delta_ms(self) -> float:
        """Positive = A is faster; negative = B is faster."""
        return self.outcome_a.latency_ms - self.outcome_b.latency_ms


@dataclass
class ABReport:
    """Aggregated A/B test results."""
    strategy_a: StrategyConfig
    strategy_b: StrategyConfig
    comparisons: list[ABComparison] = field(default_factory=list)

    # ── Aggregate metrics ─────────────────────────────────────────────── #

    def _outcomes(self, ab: str) -> list[QueryOutcome]:
        return [
            (c.outcome_a if ab == "a" else c.outcome_b)
            for c in self.comparisons
        ]

    def mean_latency(self, ab: str) -> float:
        vals = [o.latency_ms for o in self._outcomes(ab) if o.success]
        return statistics.mean(vals) if vals else 0.0

    def mean_confidence(self, ab: str) -> float:
        vals = [o.confidence for o in self._outcomes(ab) if o.success]
        return statistics.mean(vals) if vals else 0.0

    def success_rate(self, ab: str) -> float:
        outs = self._outcomes(ab)
        if not outs:
            return 0.0
        return sum(1 for o in outs if o.success) / len(outs)

    def wins(self, ab: str) -> int:
        strategy = self.strategy_a.name if ab == "a" else self.strategy_b.name
        return sum(1 for c in self.comparisons if c.winner == strategy)

    def ties(self) -> int:
        return sum(1 for c in self.comparisons if c.winner is None)

    # ── Human-readable summary ────────────────────────────────────────── #

    def summary(self) -> str:
        n = len(self.comparisons)
        lines = [
            f"A/B Test Report — {n} queries",
            f"  Strategy A: {self.strategy_a.name} (mode={self.strategy_a.mode})",
            f"  Strategy B: {self.strategy_b.name} (mode={self.strategy_b.mode})",
            "",
            f"{'Metric':<30} {'A':>12} {'B':>12}",
            "-" * 56,
            f"{'Mean latency (ms)':<30} {self.mean_latency('a'):>12.1f} {self.mean_latency('b'):>12.1f}",
            f"{'Mean confidence':<30} {self.mean_confidence('a'):>12.3f} {self.mean_confidence('b'):>12.3f}",
            f"{'Success rate':<30} {self.success_rate('a'):>12.1%} {self.success_rate('b'):>12.1%}",
            f"{'Wins (by confidence)':<30} {self.wins('a'):>12d} {self.wins('b'):>12d}",
            f"{'Ties':<30} {self.ties():>12d}",
        ]
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "strategy_a": self.strategy_a.name,
            "strategy_b": self.strategy_b.name,
            "n_queries": len(self.comparisons),
            "a": {
                "mean_latency_ms": self.mean_latency("a"),
                "mean_confidence": self.mean_confidence("a"),
                "success_rate": self.success_rate("a"),
                "wins": self.wins("a"),
            },
            "b": {
                "mean_latency_ms": self.mean_latency("b"),
                "mean_confidence": self.mean_confidence("b"),
                "success_rate": self.success_rate("b"),
                "wins": self.wins("b"),
            },
            "ties": self.ties(),
        }


# ─────────────────────────────────────────────────────────────────────────────
# Harness
# ─────────────────────────────────────────────────────────────────────────────

class ABTestHarness:
    """Run queries against two pipeline variants and compare results.

    Both pipelines must expose a ``query(question, mode) -> SearchResult``
    interface (i.e. ``LLMPipeline`` instances configured differently).
    """

    def __init__(
        self,
        strategy_a: StrategyConfig,
        strategy_b: StrategyConfig,
        pipeline_a,   # LLMPipeline
        pipeline_b,   # LLMPipeline
        scorer: Optional[Callable[[str, str, Optional[str]], float]] = None,
    ) -> None:
        """
        Args:
            strategy_a / strategy_b: Metadata describing each variant.
            pipeline_a / pipeline_b: LLMPipeline instances to compare.
            scorer: Optional function(answer, question, golden) → float in [0, 1].
                    Defaults to using the pipeline's own confidence score.
        """
        self._cfg_a = strategy_a
        self._cfg_b = strategy_b
        self._pipe_a = pipeline_a
        self._pipe_b = pipeline_b
        self._scorer = scorer

    # ------------------------------------------------------------------ #

    def run(
        self,
        queries: list[str],
        golden_answers: Optional[list[Optional[str]]] = None,
    ) -> ABReport:
        """Execute all queries against both strategies and return a report.

        Args:
            queries: List of natural-language questions.
            golden_answers: Optional reference answers (same length as queries).
                            Pass None entries for queries without golden answers.
        """
        golds = golden_answers or [None] * len(queries)
        if len(golds) != len(queries):
            raise ValueError("golden_answers length must match queries length")

        report = ABReport(strategy_a=self._cfg_a, strategy_b=self._cfg_b)

        for i, (q, gold) in enumerate(zip(queries, golds)):
            log.info("A/B query %d/%d: %.60s", i + 1, len(queries), q)
            oa = self._run_one(q, self._pipe_a, self._cfg_a)
            ob = self._run_one(q, self._pipe_b, self._cfg_b)
            report.comparisons.append(
                ABComparison(query=q, golden_answer=gold, outcome_a=oa, outcome_b=ob)
            )

        log.info("A/B test complete. %s", report.summary())
        return report

    def run_single(
        self,
        query: str,
        golden_answer: Optional[str] = None,
    ) -> ABComparison:
        """Run one query and return the comparison directly."""
        oa = self._run_one(query, self._pipe_a, self._cfg_a)
        ob = self._run_one(query, self._pipe_b, self._cfg_b)
        return ABComparison(query=query, golden_answer=golden_answer, outcome_a=oa, outcome_b=ob)

    # ------------------------------------------------------------------ #
    # Internal                                                             #
    # ------------------------------------------------------------------ #

    def _run_one(self, query: str, pipeline, cfg: StrategyConfig) -> QueryOutcome:
        t0 = time.perf_counter()
        try:
            result = pipeline.query(query, mode=cfg.mode)
            latency_ms = (time.perf_counter() - t0) * 1000
            return QueryOutcome(
                query=query,
                strategy_name=cfg.name,
                answer=result.answer,
                latency_ms=latency_ms,
                confidence=result.confidence,
                num_sources=len(result.sources),
            )
        except Exception as exc:
            latency_ms = (time.perf_counter() - t0) * 1000
            log.warning("Strategy %s failed for query %r: %s", cfg.name, query[:40], exc)
            return QueryOutcome(
                query=query,
                strategy_name=cfg.name,
                answer="",
                latency_ms=latency_ms,
                confidence=0.0,
                num_sources=0,
                error=str(exc),
            )
