"""Context builder — selects and formats retrieved chunks for the LLM prompt.

Responsibilities:
1. Optionally rerank results with the cross-encoder
2. Expand level-1 chunks to their level-0 parents for richer context
3. Deduplicate and truncate to fit within the context budget
4. Format as a numbered source list
"""
from __future__ import annotations

import logging
from typing import Any, Optional

from ..backends.embedding_backend import RerankerBackend
from ..storage.vector_store import VectorStore

log = logging.getLogger(__name__)

_DEFAULT_MAX_CHARS = 6000  # ~1500 tokens — leaves room for prompt + answer


class ContextBuilder:
    def __init__(
        self,
        vector_store: VectorStore,
        reranker: Optional[RerankerBackend] = None,
        max_chars: int = _DEFAULT_MAX_CHARS,
        top_k_rerank: int = 5,
    ) -> None:
        self._vs = vector_store
        self._reranker = reranker
        self._max_chars = max_chars
        self._top_k_rerank = top_k_rerank

    # ------------------------------------------------------------------ #

    def build(
        self,
        query: str,
        hits: list[dict[str, Any]],
        expand_parents: bool = False,
    ) -> tuple[str, list[dict]]:
        """Select, (re)rank, and format chunks.

        Returns:
            (context_string, selected_chunks)
        """
        if not hits:
            return "", []

        # Rerank if available
        if self._reranker and self._reranker.is_loaded:
            hits = self._rerank(query, hits)
        else:
            hits = hits[: self._top_k_rerank]

        # Optionally expand to parent chunks for richer context
        if expand_parents:
            hits = self._expand_parents(hits)

        # Deduplicate by chunk_id
        seen: set[str] = set()
        unique_hits = []
        for h in hits:
            cid = h.get("id") or h.get("payload", {}).get("chunk_id", "")
            if cid not in seen:
                seen.add(cid)
                unique_hits.append(h)

        # Format and truncate
        context, selected = self._format(unique_hits)
        return context, selected

    def _rerank(self, query: str, hits: list[dict]) -> list[dict]:
        passages = [h.get("payload", {}).get("text", "") for h in hits]
        ranked = self._reranker.rerank(query, passages, top_k=self._top_k_rerank)
        reranked: list[dict] = []
        for idx, score in ranked:
            h = dict(hits[idx])
            h["score"] = float(score)
            reranked.append(h)
        return reranked

    def _expand_parents(self, hits: list[dict]) -> list[dict]:
        """Replace level-1 chunks with their level-0 parents where available."""
        parent_ids = {
            h["payload"].get("parent_id")
            for h in hits
            if h.get("payload", {}).get("chunk_level") == 1
            and h["payload"].get("parent_id")
        }
        if not parent_ids:
            return hits

        parent_records = self._vs.get_by_ids(list(parent_ids))
        parent_map = {r["id"]: r for r in parent_records}

        expanded: list[dict] = []
        added_parents: set[str] = set()
        for hit in hits:
            pid = hit.get("payload", {}).get("parent_id")
            if pid and pid in parent_map and pid not in added_parents:
                expanded.append(parent_map[pid])
                added_parents.add(pid)
            else:
                expanded.append(hit)
        return expanded

    def _format(self, hits: list[dict]) -> tuple[str, list[dict]]:
        """Format chunks into numbered context string, respecting max_chars."""
        parts: list[str] = []
        selected: list[dict] = []
        total_chars = 0

        for i, hit in enumerate(hits, 1):
            payload = hit.get("payload", {})
            text = payload.get("text", "").strip()
            source = payload.get("file_name", payload.get("source_path", "unknown"))
            if not text:
                continue
            chunk_str = f"[{i}] Source: {source}\n{text}"
            if total_chars + len(chunk_str) > self._max_chars:
                break
            parts.append(chunk_str)
            selected.append(hit)
            total_chars += len(chunk_str)

        return "\n\n".join(parts), selected
