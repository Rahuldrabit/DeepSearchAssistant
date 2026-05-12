"""Query Router — classifies queries to select optimal search mode.

Routing logic (heuristic-based, no ML model required):
  fast  — short factual lookups, single-entity queries  (< 3 s target)
  deep  — analytical, multi-aspect, or long queries     (5–15 s target)
  cloud — explicit escalation or low-confidence signal  (15–40 s target)

The router exposes a `route()` method and a `suggest_top_k()` helper that
lets downstream callers know how many candidates to retrieve.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional


# ---------------------------------------------------------------------------
# Patterns that signal query complexity
# ---------------------------------------------------------------------------

# Analytical question words that need deep reasoning
_DEEP_STARTERS = re.compile(
    r"\b(why|how|explain|compare|contrast|analyze|analyse|describe|"
    r"summarize|summarise|discuss|evaluate|what are the|what is the difference|"
    r"pros and cons|advantages|disadvantages|implications|impact)\b",
    re.IGNORECASE,
)

# Queries that reference multiple distinct concepts (conjunctions)
_MULTI_CONCEPT = re.compile(r"\b(and|both|between|versus|vs\.?|compared to|relation)\b", re.IGNORECASE)

# Queries explicitly asking for lists or steps
_LIST_REQUEST = re.compile(r"\b(list|steps|enumerate|outline|overview|summary)\b", re.IGNORECASE)

# Simple lookup patterns that suit fast mode
_FAST_PATTERNS = re.compile(
    r"\b(what is|who is|when (was|is|did)|where (is|was)|define|definition of|"
    r"meaning of|abbreviation|acronym|version of)\b",
    re.IGNORECASE,
)


@dataclass
class RoutingDecision:
    mode: str           # "fast" | "deep" | "cloud"
    top_k: int          # number of candidates to retrieve
    reason: str         # human-readable explanation
    confidence: float   # 0.0–1.0 confidence in this routing decision


class QueryRouter:
    """Heuristic query router for the DeepSearch pipeline.

    Args:
        default_mode:    fallback mode when heuristics are inconclusive
        fast_top_k:      candidate count for fast mode
        deep_top_k:      candidate count for deep mode
        cloud_top_k:     candidate count for cloud mode
        word_threshold:  queries with >= this many words go to deep mode
    """

    def __init__(
        self,
        default_mode: str = "fast",
        fast_top_k: int = 10,
        deep_top_k: int = 50,
        cloud_top_k: int = 50,
        word_threshold: int = 12,
    ) -> None:
        self._default = default_mode
        self._top_k = {"fast": fast_top_k, "deep": deep_top_k, "cloud": cloud_top_k}
        self._word_threshold = word_threshold

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def route(
        self,
        query: str,
        override_mode: Optional[str] = None,
        prior_confidence: Optional[float] = None,
    ) -> RoutingDecision:
        """Classify a query and return a RoutingDecision.

        Args:
            query:            the user's natural-language query
            override_mode:    if set, skip heuristics and use this mode
            prior_confidence: if provided and below threshold, escalate to cloud
        """
        query = query.strip()

        # 1. Explicit override
        if override_mode in ("fast", "deep", "cloud"):
            return RoutingDecision(
                mode=override_mode,
                top_k=self._top_k[override_mode],
                reason="user override",
                confidence=1.0,
            )

        # 2. Low-confidence escalation from a previous answer
        if prior_confidence is not None and prior_confidence < 0.5:
            return RoutingDecision(
                mode="cloud",
                top_k=self._top_k["cloud"],
                reason=f"prior confidence {prior_confidence:.2f} below threshold",
                confidence=0.9,
            )

        # 3. Heuristic scoring
        score = self._score(query)

        if score >= 2:
            return RoutingDecision(
                mode="deep",
                top_k=self._top_k["deep"],
                reason=f"complexity score={score} (analytical/long query)",
                confidence=min(0.6 + score * 0.1, 0.95),
            )
        elif score == 1:
            return RoutingDecision(
                mode="deep",
                top_k=self._top_k["deep"],
                reason=f"complexity score={score} (moderate complexity)",
                confidence=0.65,
            )
        else:
            return RoutingDecision(
                mode="fast",
                top_k=self._top_k["fast"],
                reason="simple lookup query",
                confidence=0.8,
            )

    def suggest_top_k(self, mode: str) -> int:
        """Return suggested candidate count for a given mode."""
        return self._top_k.get(mode, self._top_k["fast"])

    # ------------------------------------------------------------------ #
    # Internal scoring                                                     #
    # ------------------------------------------------------------------ #

    def _score(self, query: str) -> int:
        """Return a complexity score 0–N (higher = more complex)."""
        score = 0
        words = query.split()

        # Long query
        if len(words) >= self._word_threshold:
            score += 2
        elif len(words) >= 7:
            score += 1

        # Analytical question words
        if _DEEP_STARTERS.search(query):
            score += 2

        # Multi-concept query
        if _MULTI_CONCEPT.search(query):
            score += 1

        # List/overview request
        if _LIST_REQUEST.search(query):
            score += 1

        # Simple lookup — subtract to cancel complexity signals
        if _FAST_PATTERNS.search(query) and len(words) <= 8:
            score -= 1

        return max(score, 0)
