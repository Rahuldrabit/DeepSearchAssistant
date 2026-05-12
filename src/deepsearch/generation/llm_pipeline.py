"""LLM generation pipeline — RAG answer generation with confidence scoring.

Search modes:
  fast   — dense-only retrieval + small LLM (< 3 s target)
  deep   — hybrid retrieval + reranker + main LLM (5–15 s target)
  cloud  — deep retrieval + cloud LLM fallback (15–40 s target)

Hardening (Phase 3):
  - Retrieval results cached in QueryCache (avoids re-embedding repeated queries)
  - Generation retried up to max_retries on transient errors
  - Answer cache uses `is not None` guard (TTLCache truthiness fix)
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Iterator, Optional

from ..backends.base import GenerationConfig
from ..core.config import Settings, get_settings
from ..core.exceptions import CloudDisabledError, GenerationError
from ..core.model_manager import ModelManager, get_model_manager
from ..retrieval.context_builder import ContextBuilder
from ..retrieval.hybrid_search import HybridSearch
from ..retrieval.query_router import QueryRouter
from ..storage.cache import CacheBundle
from .confidence_scorer import ConfidenceScorer

# Optional advanced retrieval extensions (imported lazily to avoid circular imports)
# HyDERetriever, QueryDecomposer, AgenticRetriever, GraphRAGRetriever

log = logging.getLogger(__name__)

_SYSTEM_PROMPT = """You are a precise research assistant. Answer the user's question using ONLY the provided context.
If the answer is not in the context, say "I couldn't find that in your documents."
Be concise, accurate, and cite source numbers like [1] or [2] when referencing specific passages."""


@dataclass
class SearchResult:
    answer: str
    sources: list[dict] = field(default_factory=list)
    search_mode: str = "fast"
    latency_ms: float = 0.0
    confidence: float = 1.0
    from_cache: bool = False
    retries: int = 0


class LLMPipeline:
    """End-to-end RAG pipeline: retrieve → build context → generate."""

    def __init__(
        self,
        hybrid_search: HybridSearch,
        context_builder: ContextBuilder,
        model_manager: Optional[ModelManager] = None,
        cache: Optional[CacheBundle] = None,
        settings: Optional[Settings] = None,
        query_router: Optional[QueryRouter] = None,
        max_retries: int = 2,
        # ── Advanced retrieval extensions (all optional) ──────────────────
        hyde: Optional[object] = None,           # HyDERetriever
        decomposer: Optional[object] = None,     # QueryDecomposer
        agentic: Optional[object] = None,        # AgenticRetriever
        graph_rag: Optional[object] = None,      # GraphRAGRetriever
        language_manager: Optional[object] = None,  # LanguageManager
    ) -> None:
        self._search = hybrid_search
        self._ctx_builder = context_builder
        self._mm = model_manager or get_model_manager()
        self._cache = cache or CacheBundle()
        self._cfg = settings or get_settings()
        self._router = query_router or QueryRouter()
        self._max_retries = max_retries
        self._confidence_scorer = ConfidenceScorer()
        # Extensions
        self._hyde = hyde
        self._decomposer = decomposer
        self._agentic = agentic
        self._graph_rag = graph_rag
        self._lang_mgr = language_manager
        self._last_sources: list[dict] = []   # stored for feedback integration

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def query(
        self,
        question: str,
        mode: str = "fast",
        file_filter: Optional[list[str]] = None,
    ) -> SearchResult:
        """Blocking query.  Returns SearchResult."""
        t0 = time.perf_counter()

        # Answer cache check
        cached = self._cache.answers.get_answer(question, mode)
        if cached is not None:
            return SearchResult(
                answer=cached,
                search_mode=mode,
                latency_ms=(time.perf_counter() - t0) * 1000,
                from_cache=True,
            )

        # Route query (respect explicit mode, but let router suggest top_k)
        decision = self._router.route(question, override_mode=mode)
        top_k = decision.top_k
        log.debug("Router: mode=%s top_k=%d reason=%s", decision.mode, top_k, decision.reason)

        # Retrieval — with query-level cache
        hits = self._retrieve_cached(question, mode, top_k, file_filter)

        # Build context
        context, selected = self._ctx_builder.build(
            question, hits, expand_parents=(mode != "fast")
        )

        # Generate with retry
        retries = 0
        answer = ""
        last_exc: Optional[Exception] = None
        for attempt in range(self._max_retries + 1):
            try:
                answer = self._generate(question, context, mode)
                break
            except Exception as exc:
                last_exc = exc
                retries = attempt + 1
                log.warning("Generation attempt %d failed: %s", attempt + 1, exc)
                if attempt < self._max_retries:
                    time.sleep(0.5 * (attempt + 1))  # simple backoff
        else:
            raise GenerationError(f"Generation failed after {self._max_retries} retries: {last_exc}") from last_exc

        # Confidence scoring — multi-signal (lexical + length + retrieval)
        retrieval_scores = [h.get("score", 0.0) for h in hits if isinstance(h, dict)]
        confidence = self._confidence_scorer.score(
            answer=answer,
            question=question,
            retrieval_scores=retrieval_scores or None,
        )

        # Cloud fallback on low confidence
        if confidence < self._cfg.cloud.confidence_threshold and mode != "cloud":
            if self._cfg.cloud.enabled:
                log.info("Low confidence (%.2f) — falling back to cloud", confidence)
                return self.query(question, mode="cloud", file_filter=file_filter)
            else:
                log.debug("Cloud disabled — returning low-confidence local answer")

        latency = (time.perf_counter() - t0) * 1000
        result = SearchResult(
            answer=answer,
            sources=selected,
            search_mode=mode,
            latency_ms=latency,
            confidence=confidence,
            retries=retries,
        )
        self._cache.answers.set_answer(question, mode, answer)
        log.info(
            "Query done [%s] %.0f ms | conf=%.2f | retries=%d",
            mode, latency, confidence, retries,
        )
        return result

    def stream_query(
        self,
        question: str,
        mode: str = "fast",
        file_filter: Optional[list[str]] = None,
    ) -> Iterator[str]:
        """Streaming query — yields answer tokens one at a time."""
        decision = self._router.route(question, override_mode=mode)
        top_k = decision.top_k

        hits = self._retrieve_cached(question, mode, top_k, file_filter)
        context, _ = self._ctx_builder.build(
            question, hits, expand_parents=(mode != "fast")
        )
        yield from self._stream_tokens(question, context, mode)

    # ------------------------------------------------------------------ #
    # Retrieval with query cache                                           #
    # ------------------------------------------------------------------ #

    def _retrieve_cached(
        self,
        question: str,
        mode: str,
        top_k: int,
        file_filter: Optional[list[str]],
    ) -> list[dict]:
        # Translate query when LanguageManager is available
        effective_question = question
        if self._lang_mgr is not None:
            try:
                lang = self._lang_mgr.detect(question)
                if not lang.startswith("en"):
                    llm = (self._mm.llm_small or self._mm.llm_main)
                    if llm is not None:
                        effective_question = self._lang_mgr.translate_to_english(
                            question, lang, llm
                        )
            except Exception as exc:
                log.debug("Language detection/translation error: %s", exc)

        # Skip cache when a file filter is active (results are filter-specific)
        if file_filter is None:
            cached_hits = self._cache.queries.get_results(effective_question, mode)
            if cached_hits is not None:
                log.debug("Retrieval cache hit for: %.40s", effective_question)
                return cached_hits

        hits = self._dispatch_retrieval(effective_question, mode, top_k, file_filter)

        if file_filter is None:
            self._cache.queries.set_results(effective_question, mode, hits)

        self._last_sources = hits
        return hits

    def _dispatch_retrieval(
        self,
        question: str,
        mode: str,
        top_k: int,
        file_filter: Optional[list[str]],
    ) -> list[dict]:
        """Route retrieval to the appropriate extension based on mode/config."""
        # Agentic mode: multi-step LLM-directed retrieval (cloud/deep only)
        if self._agentic is not None and mode in ("deep", "cloud"):
            try:
                result = self._agentic.retrieve(question, top_k=top_k, file_filter=file_filter)
                return result.chunks
            except Exception as exc:
                log.warning("AgenticRetriever failed: %s — falling back", exc)

        # GraphRAG: entity-graph augmented retrieval (deep/cloud only)
        if self._graph_rag is not None and mode in ("deep", "cloud"):
            try:
                return self._graph_rag.retrieve(question, top_k=top_k, file_filter=file_filter)
            except Exception as exc:
                log.warning("GraphRAGRetriever failed: %s — falling back", exc)

        # Sub-query decomposition (deep/cloud only)
        if self._decomposer is not None and mode in ("deep", "cloud"):
            try:
                result = self._decomposer.retrieve(question, top_k=top_k, file_filter=file_filter)
                return result.chunks
            except Exception as exc:
                log.warning("QueryDecomposer failed: %s — falling back", exc)

        # HyDE (deep/cloud only — requires LLM, skip for fast mode)
        if self._hyde is not None and mode in ("deep", "cloud"):
            try:
                return self._hyde.retrieve(question, top_k=top_k, file_filter=file_filter)
            except Exception as exc:
                log.warning("HyDERetriever failed: %s — falling back", exc)

        # Standard retrieval
        if mode == "fast":
            return self._search.search_dense_only(question, top_k=top_k)
        return self._search.search(question, top_k=top_k, file_filter=file_filter)

    # ------------------------------------------------------------------ #
    # Internal generation                                                  #
    # ------------------------------------------------------------------ #

    def _generate(self, question: str, context: str, mode: str) -> str:
        gen_cfg = GenerationConfig(
            max_tokens=self._cfg.cloud.max_tokens if mode == "cloud" else 512,
            temperature=0.1 if mode == "fast" else 0.2,
            stop=["<|user|>", "<|system|>", "</s>"],
        )

        if mode == "cloud":
            return self._cloud_generate(question, context, gen_cfg)

        llm = self._mm.llm_small if mode == "fast" and self._mm.llm_small else self._mm.llm_main
        prompt = llm.build_prompt(_SYSTEM_PROMPT, question, context)
        return llm.generate(prompt, gen_cfg)

    def _stream_tokens(self, question: str, context: str, mode: str) -> Iterator[str]:
        gen_cfg = GenerationConfig(
            max_tokens=512,
            temperature=0.1 if mode == "fast" else 0.2,
            stop=["<|user|>", "<|system|>", "</s>"],
            stream=True,
        )
        if mode == "cloud":
            yield from self._cloud_stream(question, context, gen_cfg)
            return

        llm = self._mm.llm_small if mode == "fast" and self._mm.llm_small else self._mm.llm_main
        prompt = llm.build_prompt(_SYSTEM_PROMPT, question, context)
        yield from llm.stream(prompt, gen_cfg)

    def _cloud_generate(self, question: str, context: str, cfg: GenerationConfig) -> str:
        if not self._cfg.cloud.enabled:
            raise CloudDisabledError("Cloud fallback is disabled in config")
        from ..backends.cloud_backend import CloudAPIBackend
        backend = CloudAPIBackend()
        backend.load(self._cfg.cloud.model, api_key=self._cfg.cloud_api_key)
        prompt = f"{_SYSTEM_PROMPT}\n\nContext:\n{context}\n\nQuestion: {question}"
        return backend.generate(prompt, cfg)

    def _cloud_stream(self, question: str, context: str, cfg: GenerationConfig) -> Iterator[str]:
        if not self._cfg.cloud.enabled:
            raise CloudDisabledError("Cloud fallback is disabled in config")
        from ..backends.cloud_backend import CloudAPIBackend
        backend = CloudAPIBackend()
        backend.load(self._cfg.cloud.model, api_key=self._cfg.cloud_api_key)
        prompt = f"{_SYSTEM_PROMPT}\n\nContext:\n{context}\n\nQuestion: {question}"
        yield from backend.stream(prompt, cfg)
