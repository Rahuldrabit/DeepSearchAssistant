"""Sub-query decomposition — breaks complex queries into atomic sub-queries.

Complex questions like "What were the revenue trends and how did costs
compare to industry benchmarks?" contain multiple independent information
needs.  Answering each sub-query separately and merging the retrieved
chunks often yields better recall than a single monolithic query.

Algorithm:
  1. Classify query complexity (trivial → skip decomposition).
  2. Prompt LLM to emit numbered sub-queries.
  3. Retrieve top-K candidates for each sub-query.
  4. Merge all candidate sets with RRF deduplication.
  5. Cap at a configurable max_total_results.

Usage::

    decomposer = QueryDecomposer(llm, search)
    result = decomposer.retrieve("revenue trends and cost comparison", top_k=10)
    # result.chunks  — merged, deduplicated hits
    # result.sub_queries  — the generated sub-queries
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Optional

log = logging.getLogger(__name__)

_DECOMPOSE_SYSTEM = """\
You are a retrieval planning assistant.  Break the user's complex question
into 2–4 concise, independent sub-questions that together cover all aspects
of the original question.  Output ONLY a numbered list, one sub-question per
line, no preamble.

Example:
User: What were revenues and how do they compare to costs?
1. What were the revenues?
2. What were the costs?
3. How do revenues compare to costs?"""

_MIN_WORDS_TO_DECOMPOSE = 8   # skip for short/simple queries
_MAX_SUB_QUERIES = 4
_SUB_QUERY_TOP_K = 15         # per sub-query candidate count


@dataclass
class DecompositionResult:
    sub_queries: list[str]
    chunks: list[dict[str, Any]]   # merged, deduplicated, top-N
    skipped: bool = False           # True when decomposition was skipped


class QueryDecomposer:
    """Decomposes complex queries and multi-retrieves.

    Args:
        llm:          Any loaded LLMBackend.
        search:       HybridSearch instance.
        min_words:    Minimum word count before decomposition is attempted.
        sub_top_k:    Candidates to retrieve per sub-query.
        rrf_k:        RRF smoothing constant for merging sub-query results.
        max_results:  Cap on total returned chunks.
    """

    def __init__(
        self,
        llm,
        search,
        min_words: int = _MIN_WORDS_TO_DECOMPOSE,
        sub_top_k: int = _SUB_QUERY_TOP_K,
        rrf_k: int = 60,
        max_results: int = 50,
    ) -> None:
        self._llm = llm
        self._search = search
        self._min_words = min_words
        self._sub_top_k = sub_top_k
        self._rrf_k = rrf_k
        self._max_results = max_results

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        file_filter: Optional[list[str]] = None,
    ) -> DecompositionResult:
        """Decompose *query* and retrieve merged results.

        If the query is short or decomposition fails, falls back to a single
        hybrid search.
        """
        if len(query.split()) < self._min_words:
            log.debug("QueryDecomposer: query too short to decompose — single search")
            hits = self._search.search(query, top_k=top_k, file_filter=file_filter)
            return DecompositionResult(sub_queries=[query], chunks=hits, skipped=True)

        try:
            sub_queries = self._decompose(query)
        except Exception as exc:
            log.warning("Decomposition LLM call failed: %s — falling back", exc)
            hits = self._search.search(query, top_k=top_k, file_filter=file_filter)
            return DecompositionResult(sub_queries=[query], chunks=hits, skipped=True)

        if not sub_queries:
            hits = self._search.search(query, top_k=top_k, file_filter=file_filter)
            return DecompositionResult(sub_queries=[query], chunks=hits, skipped=True)

        # Retrieve for each sub-query
        all_ranked: list[list[dict]] = []
        for sq in sub_queries:
            try:
                hits = self._search.search(sq, top_k=self._sub_top_k, file_filter=file_filter)
                all_ranked.append(hits)
                log.debug("Sub-query '%s': %d hits", sq[:50], len(hits))
            except Exception as exc:
                log.warning("Sub-query retrieval failed for '%s': %s", sq, exc)

        merged = self._rrf_merge(all_ranked)[: min(top_k, self._max_results)]
        log.info(
            "QueryDecomposer: %d sub-queries → %d merged chunks",
            len(sub_queries), len(merged),
        )
        return DecompositionResult(sub_queries=sub_queries, chunks=merged)

    # ------------------------------------------------------------------ #
    # LLM decomposition                                                    #
    # ------------------------------------------------------------------ #

    def _decompose(self, query: str) -> list[str]:
        """Use LLM to split *query* into atomic sub-queries."""
        from ..backends.base import GenerationConfig
        cfg = GenerationConfig(
            max_tokens=200,
            temperature=0.2,
            stop=["5.", "\n\n", "<|user|>"],
        )
        prompt = self._llm.build_prompt(_DECOMPOSE_SYSTEM, query)
        raw = self._llm.generate(prompt, cfg).strip()
        return self._parse_numbered_list(raw)

    @staticmethod
    def _parse_numbered_list(text: str) -> list[str]:
        """Parse '1. foo\n2. bar\n…' → ['foo', 'bar', …]"""
        subs: list[str] = []
        for line in text.splitlines():
            m = re.match(r"^\s*\d+\.\s*(.+)", line)
            if m:
                q = m.group(1).strip()
                if q:
                    subs.append(q)
                if len(subs) >= _MAX_SUB_QUERIES:
                    break
        return subs

    # ------------------------------------------------------------------ #
    # RRF merge across multiple ranked lists                               #
    # ------------------------------------------------------------------ #

    def _rrf_merge(self, ranked_lists: list[list[dict]]) -> list[dict]:
        """Merge N ranked hit lists using Reciprocal Rank Fusion."""
        scores: dict[str, float] = {}
        payloads: dict[str, dict] = {}

        for hits in ranked_lists:
            for rank, hit in enumerate(hits):
                cid = hit["id"]
                scores[cid] = scores.get(cid, 0.0) + 1.0 / (self._rrf_k + rank + 1)
                if cid not in payloads:
                    payloads[cid] = hit.get("payload", {})

        return [
            {"id": cid, "score": score, "payload": payloads[cid]}
            for cid, score in sorted(scores.items(), key=lambda x: x[1], reverse=True)
        ]
