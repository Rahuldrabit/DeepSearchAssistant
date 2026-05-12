"""HyDE — Hypothetical Document Embeddings for retrieval.

Instead of embedding the raw user query, HyDE generates a *hypothetical*
answer document with the LLM and embeds *that* for dense retrieval.  The
intuition is that a plausible answer is closer in embedding space to real
relevant passages than the question alone.

Reference: Gao et al. (2022) "Precise Zero-Shot Dense Retrieval without
Relevance Labels" https://arxiv.org/abs/2212.10496

Usage::

    hyde = HyDERetriever(hybrid_search, llm_backend, embedder)
    hits = hyde.retrieve("What caused the 2008 financial crisis?", top_k=10)

    # Or generate the hypothesis separately for inspection:
    hyp = hyde.generate_hypothesis("What caused the 2008 financial crisis?")
    print(hyp)
"""
from __future__ import annotations

import logging
from typing import Any, Optional

log = logging.getLogger(__name__)

_HYDE_SYSTEM = (
    "You are a precise research assistant.  "
    "Write a short factual passage (2–4 sentences) that directly and "
    "authoritatively answers the following question.  "
    "Do NOT hedge or say you don't know — write as if the answer is in "
    "a document you are quoting."
)

_DEFAULT_MAX_TOKENS = 128
_DEFAULT_TEMPERATURE = 0.7   # slightly higher for diversity


class HyDERetriever:
    """Wraps a HybridSearch instance to use hypothetical document embeddings.

    Args:
        search:          Underlying HybridSearch (provides embed + dense search).
        llm:             Any loaded LLMBackend (used for hypothesis generation).
        embedder:        Embedder instance (for embedding the hypothesis).
        num_hypotheses:  How many hypotheses to generate and average (default 1).
                         Averaging multiple hypotheses reduces variance.
        fallback:        If True, fall back to normal dense search on LLM error.
    """

    def __init__(
        self,
        search,               # HybridSearch
        llm,                  # LLMBackend
        embedder,             # Embedder
        num_hypotheses: int = 1,
        fallback: bool = True,
    ) -> None:
        self._search = search
        self._llm = llm
        self._embedder = embedder
        self._num_hypotheses = max(1, num_hypotheses)
        self._fallback = fallback

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        file_filter: Optional[list[str]] = None,
    ) -> list[dict[str, Any]]:
        """Retrieve using hypothetical document embedding.

        Returns hits in the same ``{"id", "score", "payload"}`` format as
        :meth:`HybridSearch.search`.
        """
        try:
            hypothesis = self._average_hypothesis(query)
        except Exception as exc:
            log.warning("HyDE hypothesis generation failed: %s", exc)
            if self._fallback:
                log.info("HyDE falling back to normal dense search")
                return self._search.search_dense_only(query, top_k=top_k)
            raise

        dense_vec, _, _ = self._embedder.embed_query(hypothesis)
        filter_ = self._search._build_filter(file_filter)
        hits = self._search._vs.search_dense(dense_vec, top_k=top_k, filter_=filter_)
        log.debug(
            "HyDE retrieve: %d hits for query=%.40s | hypothesis=%.60s",
            len(hits), query, hypothesis,
        )
        return hits

    def generate_hypothesis(self, query: str) -> str:
        """Generate a single hypothetical answer for *query*."""
        from ..backends.base import GenerationConfig
        cfg = GenerationConfig(
            max_tokens=_DEFAULT_MAX_TOKENS,
            temperature=_DEFAULT_TEMPERATURE,
            stop=["\n\n", "<|user|>", "<|system|>"],
        )
        prompt = self._llm.build_prompt(_HYDE_SYSTEM, query)
        return self._llm.generate(prompt, cfg).strip()

    # ------------------------------------------------------------------ #
    # Internal                                                             #
    # ------------------------------------------------------------------ #

    def _average_hypothesis(self, query: str) -> str:
        """Generate ``num_hypotheses`` hypotheses and pick the longest one.

        A simple heuristic: the longest plausible answer often contains the
        most relevant content for embedding.  True averaging would require
        averaging the *vectors* (see below), but averaging text is not meaningful.
        For multi-hypothesis averaging, callers should average the *dense vectors*
        directly — this method returns a single string for simplicity.
        """
        if self._num_hypotheses == 1:
            return self.generate_hypothesis(query)

        hypotheses = [self.generate_hypothesis(query) for _ in range(self._num_hypotheses)]
        # Pick the hypothesis with the most content words
        return max(hypotheses, key=lambda h: len(h.split()))

    def retrieve_with_average_vectors(
        self,
        query: str,
        top_k: int = 10,
        file_filter: Optional[list[str]] = None,
    ) -> list[dict[str, Any]]:
        """Retrieve by averaging dense vectors across multiple hypotheses.

        This is the *true* HyDE multi-hypothesis approach: embed each
        hypothesis separately, average the vectors, then search.  Requires
        ``num_hypotheses > 1`` to be meaningful.
        """
        import numpy as np

        hypotheses = [self.generate_hypothesis(query) for _ in range(self._num_hypotheses)]
        vecs = [self._embedder.embed_query(h)[0] for h in hypotheses]
        avg_vec = np.mean([np.array(v) for v in vecs], axis=0).tolist()

        filter_ = self._search._build_filter(file_filter)
        hits = self._search._vs.search_dense(avg_vec, top_k=top_k, filter_=filter_)
        log.debug("HyDE multi-hypothesis (%d): %d hits", self._num_hypotheses, len(hits))
        return hits
