"""Reranker — cross-encoder reranking for retrieved chunks.

Wraps RerankerBackend with pipeline-friendly interface:
  - Accepts raw hit dicts ({"id", "score", "payload"})
  - Returns same dict format with updated scores
  - Lazy-loads the model on first call if not pre-loaded
  - Gracefully degrades (returns original order) when model unavailable
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

from ..backends.embedding_backend import RerankerBackend

log = logging.getLogger(__name__)

_DEFAULT_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


@dataclass
class RerankResult:
    """Wrapper returned by Reranker.rerank() with diagnostics."""
    hits: list[dict]
    scores: list[float]
    model_used: bool = True  # False when reranker unavailable → original order


class Reranker:
    """High-level reranker: wraps RerankerBackend for use inside retrieval pipelines.

    Usage::

        reranker = Reranker()
        reranker.load()  # loads cross-encoder
        result = reranker.rerank(query, hits, top_k=5)
        top_hits = result.hits

    The reranker can also be used unloaded — it falls back to the original
    retrieval order so the pipeline never hard-crashes.
    """

    def __init__(
        self,
        model_name: str = _DEFAULT_MODEL,
        device: str = "cpu",
        auto_load: bool = False,
    ) -> None:
        self._model_name = model_name
        self._device = device
        self._backend = RerankerBackend()
        if auto_load:
            self.load()

    # ------------------------------------------------------------------ #
    # Lifecycle                                                            #
    # ------------------------------------------------------------------ #

    def load(self) -> None:
        """Load cross-encoder model (downloads on first use, ~22 MB)."""
        self._backend.load(self._model_name, device=self._device)
        log.info("Reranker loaded: %s", self._model_name)

    def unload(self) -> None:
        """Release the model from memory."""
        self._backend._model = None
        log.info("Reranker unloaded")

    @property
    def is_loaded(self) -> bool:
        return self._backend.is_loaded

    # ------------------------------------------------------------------ #
    # Core API                                                             #
    # ------------------------------------------------------------------ #

    def rerank(
        self,
        query: str,
        hits: list[dict],
        top_k: int = 5,
    ) -> RerankResult:
        """Rerank retrieved hits by cross-encoder relevance score.

        Args:
            query:  User query string.
            hits:   List of {"id", "score", "payload"} dicts from hybrid search.
            top_k:  Maximum number of hits to return.

        Returns:
            RerankResult with reordered hits and cross-encoder scores.
            If model not loaded, returns original top_k hits unchanged.
        """
        if not hits:
            return RerankResult(hits=[], scores=[], model_used=False)

        if not self._backend.is_loaded:
            log.debug("Reranker not loaded — returning original top-%d hits", top_k)
            return RerankResult(
                hits=hits[:top_k],
                scores=[h.get("score", 0.0) for h in hits[:top_k]],
                model_used=False,
            )

        passages = [h.get("payload", {}).get("text", "") for h in hits]
        ranked = self._backend.rerank(query, passages, top_k=top_k)

        reranked_hits: list[dict] = []
        reranked_scores: list[float] = []
        for orig_idx, score in ranked:
            hit = dict(hits[orig_idx])
            hit["score"] = score  # replace retrieval score with cross-encoder score
            reranked_hits.append(hit)
            reranked_scores.append(score)

        log.debug(
            "Reranked %d → %d hits (top score: %.4f)",
            len(hits), len(reranked_hits), reranked_scores[0] if reranked_scores else 0,
        )
        return RerankResult(hits=reranked_hits, scores=reranked_scores, model_used=True)

    def rerank_hits(
        self,
        query: str,
        hits: list[dict],
        top_k: int = 5,
    ) -> list[dict]:
        """Convenience wrapper — returns hit list directly (no RerankResult)."""
        return self.rerank(query, hits, top_k=top_k).hits

    # ------------------------------------------------------------------ #
    # Batch scoring                                                        #
    # ------------------------------------------------------------------ #

    def score_passage(self, query: str, passage: str) -> float:
        """Score a single query–passage pair. Returns raw cross-encoder logit."""
        if not self._backend.is_loaded:
            raise RuntimeError("Reranker not loaded. Call load() first.")
        ranked = self._backend.rerank(query, [passage], top_k=1)
        return ranked[0][1] if ranked else 0.0

    def score_passages(self, query: str, passages: list[str]) -> list[float]:
        """Score multiple passages against a query. Returns score per passage (original order)."""
        if not self._backend.is_loaded:
            raise RuntimeError("Reranker not loaded. Call load() first.")
        ranked = self._backend.rerank(query, passages, top_k=len(passages))
        # ranked is (original_idx, score) sorted by score — restore original order
        scores = [0.0] * len(passages)
        for orig_idx, score in ranked:
            scores[orig_idx] = score
        return scores
