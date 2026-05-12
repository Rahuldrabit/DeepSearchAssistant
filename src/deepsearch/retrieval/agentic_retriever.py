"""Agentic multi-step retrieval — LLM decides the next retrieval action.

Implements a simplified ReAct (Reason + Act) loop:

  Loop (up to max_steps):
    1. THINK  — LLM analyses what it knows so far and what's missing.
    2. ACT    — LLM emits a search query (or "[DONE]" to stop early).
    3. OBSERVE — Execute search, add new chunks to the knowledge pool.
  End loop.
  Final answer is built from the accumulated chunk pool.

The LLM decides when it has "enough" context by emitting ``[DONE]``.
A hard ``max_steps`` cap prevents infinite loops.

Usage::

    agent = AgenticRetriever(llm, search, max_steps=3)
    result = agent.retrieve("What is the relationship between X and Y?", top_k=15)
    print(result.steps)   # list of (search_query, n_new_chunks)
    print(result.chunks)  # accumulated, deduplicated
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Optional

log = logging.getLogger(__name__)

_AGENT_SYSTEM = """\
You are a research agent with access to a document retrieval tool.

Your task is to answer the user's question by iteratively searching for relevant information.

At each step you will see:
 - The original question
 - A summary of information already retrieved
 - The step number

You must output EXACTLY ONE of:
 a) A short search query (1–10 words) to retrieve more information, OR
 b) The text [DONE] if you have enough information to answer the question.

Output ONLY the search query or [DONE] — nothing else."""

_DONE_MARKER = "[DONE]"
_DEFAULT_MAX_STEPS = 4
_DEFAULT_PER_STEP_TOP_K = 10


@dataclass
class AgentStep:
    step_num: int
    search_query: str
    n_new_chunks: int


@dataclass
class AgentRetrievalResult:
    chunks: list[dict[str, Any]]     # all accumulated, deduplicated chunks
    steps: list[AgentStep]           # trace of each retrieval action
    terminated_early: bool = False   # True when LLM emitted [DONE]


class AgenticRetriever:
    """Multi-step retrieval agent.

    Args:
        llm:            Any loaded LLMBackend (acts as the "thinking" component).
        search:         HybridSearch (acts as the "retrieval tool").
        max_steps:      Hard cap on iterations (default 4).
        per_step_top_k: Candidates to retrieve per step (default 10).
        rrf_k:          RRF smoothing constant for dedup-merge.
    """

    def __init__(
        self,
        llm,
        search,
        max_steps: int = _DEFAULT_MAX_STEPS,
        per_step_top_k: int = _DEFAULT_PER_STEP_TOP_K,
        rrf_k: int = 60,
    ) -> None:
        self._llm = llm
        self._search = search
        self._max_steps = max_steps
        self._per_step_k = per_step_top_k
        self._rrf_k = rrf_k

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def retrieve(
        self,
        query: str,
        top_k: int = 15,
        file_filter: Optional[list[str]] = None,
    ) -> AgentRetrievalResult:
        """Run the agentic retrieval loop.

        Returns an :class:`AgentRetrievalResult` whose ``chunks`` field
        contains the top-*top_k* deduplicated hits across all steps.
        """
        all_chunks: dict[str, dict] = {}   # chunk_id → hit dict
        steps: list[AgentStep] = []
        terminated_early = False

        for step in range(1, self._max_steps + 1):
            search_query = self._decide_action(query, list(all_chunks.values()), step)
            log.debug("Agent step %d: search_query=%r", step, search_query)

            if search_query.strip().upper() == _DONE_MARKER.upper():
                terminated_early = True
                log.info("Agentic retriever: LLM signalled [DONE] at step %d", step)
                break

            try:
                new_hits = self._search.search(
                    search_query,
                    top_k=self._per_step_k,
                    file_filter=file_filter,
                )
            except Exception as exc:
                log.warning("Agent step %d search failed: %s", step, exc)
                new_hits = []

            new_count = 0
            for hit in new_hits:
                if hit["id"] not in all_chunks:
                    all_chunks[hit["id"]] = hit
                    new_count += 1

            steps.append(AgentStep(step_num=step, search_query=search_query, n_new_chunks=new_count))
            log.debug("Agent step %d: +%d new chunks (total=%d)", step, new_count, len(all_chunks))

            # Early stopping: no new information found
            if new_count == 0 and step > 1:
                log.info("Agentic retriever: no new chunks at step %d — stopping", step)
                break

        # Re-rank accumulated pool by original RRF-style score and cap at top_k
        merged = self._rank(list(all_chunks.values()))[:top_k]
        return AgentRetrievalResult(
            chunks=merged,
            steps=steps,
            terminated_early=terminated_early,
        )

    # ------------------------------------------------------------------ #
    # LLM action planning                                                  #
    # ------------------------------------------------------------------ #

    def _decide_action(
        self,
        original_query: str,
        current_chunks: list[dict],
        step: int,
    ) -> str:
        """Ask the LLM what to search for next."""
        from ..backends.base import GenerationConfig

        context_summary = self._summarise_chunks(current_chunks)
        user_prompt = (
            f"Original question: {original_query}\n\n"
            f"Step: {step}/{self._max_steps}\n\n"
            f"Information retrieved so far:\n{context_summary}\n\n"
            f"What should I search for next? (or output [DONE])"
        )
        cfg = GenerationConfig(
            max_tokens=30,
            temperature=0.1,
            stop=["\n", "<|user|>"],
        )
        prompt = self._llm.build_prompt(_AGENT_SYSTEM, user_prompt)
        raw = self._llm.generate(prompt, cfg).strip()
        return raw if raw else _DONE_MARKER

    @staticmethod
    def _summarise_chunks(chunks: list[dict]) -> str:
        """Build a brief summary of already-retrieved chunks."""
        if not chunks:
            return "(nothing retrieved yet)"
        lines = []
        for i, c in enumerate(chunks[:5], 1):   # show first 5 only
            text = c.get("payload", {}).get("text", "")
            lines.append(f"[{i}] {text[:120]}…" if len(text) > 120 else f"[{i}] {text}")
        if len(chunks) > 5:
            lines.append(f"… and {len(chunks) - 5} more passages")
        return "\n".join(lines)

    @staticmethod
    def _rank(chunks: list[dict]) -> list[dict]:
        """Sort accumulated chunks by their original retrieval score descending."""
        return sorted(chunks, key=lambda c: c.get("score", 0.0), reverse=True)
