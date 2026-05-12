"""Hybrid search: dense + sparse (BM25) with Reciprocal Rank Fusion.

RRF formula: score(d) = Σ 1 / (k + rank_i(d))
Default k=60 (per the Cormack et al. 2009 paper).
"""
from __future__ import annotations

import logging
from typing import Any, Optional

from ..ingestion.embedder import Embedder
from ..storage.vector_store import VectorStore

log = logging.getLogger(__name__)


def _rrf_merge(
    dense_hits: list[dict],
    sparse_hits: list[dict],
    k: int = 60,
) -> list[dict]:
    """Merge two ranked lists with Reciprocal Rank Fusion.

    Args:
        dense_hits:  [{"id": ..., "score": ..., "payload": ...}, ...]
        sparse_hits: same format
        k: RRF smoothing constant (default 60)

    Returns:
        Merged list sorted by descending RRF score, deduplicated.
    """
    scores: dict[str, float] = {}
    payloads: dict[str, dict] = {}

    for rank, hit in enumerate(dense_hits):
        cid = hit["id"]
        scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank + 1)
        payloads[cid] = hit.get("payload", {})

    for rank, hit in enumerate(sparse_hits):
        cid = hit["id"]
        scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank + 1)
        if cid not in payloads:
            payloads[cid] = hit.get("payload", {})

    merged = [
        {"id": cid, "score": score, "payload": payloads[cid]}
        for cid, score in sorted(scores.items(), key=lambda x: x[1], reverse=True)
    ]
    return merged


class HybridSearch:
    """Combines dense and sparse retrieval with RRF fusion."""

    def __init__(
        self,
        vector_store: VectorStore,
        embedder: Embedder,
        top_k_retrieval: int = 50,
        rrf_k: int = 60,
    ) -> None:
        self._vs = vector_store
        self._embedder = embedder
        self._top_k = top_k_retrieval
        self._rrf_k = rrf_k

    def search(
        self,
        query: str,
        top_k: Optional[int] = None,
        file_filter: Optional[list[str]] = None,
    ) -> list[dict[str, Any]]:
        """Run hybrid search and return merged results.

        Args:
            query: natural language query
            top_k: number of results to return (defaults to constructor value)
            file_filter: if set, restrict search to these file_ids

        Returns:
            List of {"id", "score", "payload"} dicts, sorted by RRF score.
        """
        k = top_k or self._top_k
        dense_vec, sparse_idx, sparse_vals = self._embedder.embed_query(query)

        filter_ = self._build_filter(file_filter)

        dense_hits = self._vs.search_dense(dense_vec, top_k=k, filter_=filter_)
        sparse_hits = self._vs.search_sparse(sparse_idx, sparse_vals, top_k=k, filter_=filter_)

        merged = _rrf_merge(dense_hits, sparse_hits, k=self._rrf_k)
        log.debug(
            "Hybrid search: %d dense + %d sparse → %d merged",
            len(dense_hits), len(sparse_hits), len(merged),
        )
        return merged[:k]

    def search_dense_only(self, query: str, top_k: int = 10) -> list[dict]:
        """Fast mode: dense only, no BM25 fusion."""
        dense_vec, _, _ = self._embedder.embed_query(query)
        return self._vs.search_dense(dense_vec, top_k=top_k)

    # ------------------------------------------------------------------ #

    @staticmethod
    def _build_filter(file_filter: Optional[list[str]]):
        if not file_filter:
            return None
        from qdrant_client.models import FieldCondition, Filter, MatchAny  # type: ignore[import]
        return Filter(
            must=[FieldCondition(key="file_id", match=MatchAny(any=file_filter))]
        )
