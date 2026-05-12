"""Confidence scorer for RAG-generated answers.

Combines three independent signals into a single confidence score [0, 1]:

  1. Lexical signal   — presence of hedging/refusal phrases (fast, regex)
  2. Length signal    — very short answers are often low quality
  3. Retrieval signal — how good were the retrieved chunks?
                        (uses average retrieval score when available)

The three signals are blended with configurable weights so the scorer
is transparent and can be tuned without retraining a model.

Usage:
    scorer = ConfidenceScorer()
    conf = scorer.score(
        answer="The capital of France is Paris.",
        question="What is the capital of France?",
        retrieval_scores=[0.82, 0.77, 0.61],
    )
    # conf → 0.87 (example)
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass

# ── Lexical patterns ────────────────────────────────────────────────────────
# Each tuple: (pattern, confidence_deduction)
_REFUSAL_PATTERNS: list[tuple[re.Pattern[str], float]] = [
    (re.compile(r"I couldn.t find", re.IGNORECASE), 0.7),
    (re.compile(r"not (?:mentioned|found|in(?: the)?|available) in", re.IGNORECASE), 0.5),
    (re.compile(r"no information", re.IGNORECASE), 0.5),
    (re.compile(r"I don.t know", re.IGNORECASE), 0.45),
    (re.compile(r"I.m not sure", re.IGNORECASE), 0.3),
    (re.compile(r"(?:may|might|could) (?:be|have)", re.IGNORECASE), 0.15),
    (re.compile(r"unclear|uncertain|ambiguous", re.IGNORECASE), 0.2),
    (re.compile(r"I cannot (?:find|answer|determine)", re.IGNORECASE), 0.5),
]

# Minimum answer word count for "normal" length signal
_MIN_WORDS_GOOD = 10
# Answers longer than this are almost never refusals
_MAX_WORDS_CAP = 100


@dataclass
class ConfidenceBreakdown:
    """Per-signal breakdown returned alongside the final score."""
    lexical: float    # 0–1 (1 = no hedging)
    length: float     # 0–1 (1 = adequate length)
    retrieval: float  # 0–1 (1 = high retrieval quality)
    final: float      # blended score


class ConfidenceScorer:
    """Multi-signal confidence estimator for RAG answers.

    Args:
        weight_lexical:    Weight for the lexical (hedging) signal.
        weight_length:     Weight for the length signal.
        weight_retrieval:  Weight for the retrieval quality signal.

    Weights are normalised internally so they need not sum to 1.
    """

    def __init__(
        self,
        weight_lexical: float = 0.5,
        weight_length: float = 0.2,
        weight_retrieval: float = 0.3,
    ) -> None:
        total = weight_lexical + weight_length + weight_retrieval
        if total <= 0:
            raise ValueError("Weights must be positive")
        self._w_lex = weight_lexical / total
        self._w_len = weight_length / total
        self._w_ret = weight_retrieval / total

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def score(
        self,
        answer: str,
        question: str = "",
        retrieval_scores: list[float] | None = None,
    ) -> float:
        """Return a confidence value in [0, 1].

        Args:
            answer:            The generated answer text.
            question:          The original question (reserved for future
                               question-answer alignment signal).
            retrieval_scores:  Per-chunk retrieval similarity scores [0, 1].
                               Pass None or [] when not available.
        """
        if not answer.strip():
            return 0.0
        breakdown = self.score_detailed(answer, question, retrieval_scores)
        return breakdown.final

    def score_detailed(
        self,
        answer: str,
        question: str = "",
        retrieval_scores: list[float] | None = None,
    ) -> ConfidenceBreakdown:
        """Return the full per-signal breakdown."""
        lex = self._lexical_signal(answer)
        length = self._length_signal(answer)
        ret = self._retrieval_signal(retrieval_scores)

        final = self._w_lex * lex + self._w_len * length + self._w_ret * ret
        final = max(0.0, min(1.0, final))

        return ConfidenceBreakdown(
            lexical=lex,
            length=length,
            retrieval=ret,
            final=final,
        )

    # ------------------------------------------------------------------ #
    # Individual signals                                                   #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _lexical_signal(answer: str) -> float:
        """Score based on hedging/refusal phrase presence.

        Starts at 1.0 and subtracts deductions for each matched pattern.
        Multiple patterns can stack (floor = 0.05 to keep signal smooth).
        """
        score = 1.0
        for pattern, deduction in _REFUSAL_PATTERNS:
            if pattern.search(answer):
                score -= deduction
        return max(0.05, score)

    @staticmethod
    def _length_signal(answer: str) -> float:
        """Score based on answer word count using a smooth sigmoid-like curve.

        Very short answers (< MIN_WORDS_GOOD) are penalised progressively.
        Answers beyond MAX_WORDS_CAP words receive full score.
        """
        words = len(answer.split())
        if words == 0:
            return 0.0
        if words >= _MAX_WORDS_CAP:
            return 1.0
        # Smooth ramp: tanh((words / MIN_WORDS_GOOD - 0.5) * 2)
        # At words = MIN_WORDS_GOOD → ~0.76; at 0 → ~0.02
        x = (words / _MIN_WORDS_GOOD - 0.5) * 2.0
        return (math.tanh(x) + 1.0) / 2.0

    @staticmethod
    def _retrieval_signal(scores: list[float] | None) -> float:
        """Score based on average retrieval similarity of the top chunks.

        When no scores are provided, returns a neutral mid-point (0.6)
        so the signal neither boosts nor heavily penalises the final score.
        """
        if not scores:
            return 0.6  # neutral when retrieval scores unavailable

        # Use top-3 average to reduce noise from low-quality tail chunks
        top = sorted(scores, reverse=True)[:3]
        avg = sum(top) / len(top)
        return max(0.0, min(1.0, float(avg)))
