"""GraphRAG — entity-aware retrieval using an in-memory knowledge graph.

Builds a lightweight knowledge graph from indexed chunk payloads:

  Nodes  = named entities (people, orgs, locations, concepts)
  Edges  = co-occurrence within the same chunk

At query time:
  1. Extract entities from the question.
  2. Find related entities in the graph (1-hop neighbours).
  3. Collect all chunk IDs that mention any of these entities.
  4. Retrieve those chunks from the vector store and merge with the
     standard dense search results.

Entity extraction is regex-based (fast, no ML dep) with an optional
spaCy enhancement when available.

Usage::

    graph = KnowledgeGraph()
    graph.build_from_chunks(hits)          # after any retrieval
    # or incrementally:
    graph.ingest_chunk(chunk_payload)

    grag = GraphRAGRetriever(search, graph)
    result = grag.retrieve("What is the relationship between Alice and Acme Corp?")
"""
from __future__ import annotations

import logging
import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Optional

log = logging.getLogger(__name__)

# ── Simple regex-based NER ───────────────────────────────────────────────────
# Matches Title-Case sequences of 2–4 words (heuristic for proper nouns)
_PROPER_NOUN_RE = re.compile(
    r"\b([A-Z][a-z]{1,20}(?:\s+[A-Z][a-z]{1,20}){0,3})\b"
)
# Concept keywords (frequently important in technical/business documents)
_CONCEPT_RE = re.compile(
    r"\b(revenue|profit|loss|market|product|service|API|model|algorithm|"
    r"database|network|protocol|framework|architecture|pipeline)\b",
    re.IGNORECASE,
)


def _extract_entities(text: str) -> list[str]:
    """Extract entity mentions from *text*.

    First tries spaCy (NER if available), falls back to regex heuristics.
    Returns a list of deduplicated entity strings.
    """
    try:
        import spacy  # type: ignore[import]
        # Use a small model; skip if not downloaded
        nlp = spacy.load("en_core_web_sm", disable=["parser", "textcat"])
        doc = nlp(text[:2000])  # limit for speed
        entities = [ent.text.strip() for ent in doc.ents if len(ent.text.strip()) > 2]
    except Exception:
        entities = []
        # Fallback: regex proper nouns
        entities += [m.group(1).strip() for m in _PROPER_NOUN_RE.finditer(text)]
        # Fallback: concept keywords
        entities += [m.group(1).strip() for m in _CONCEPT_RE.finditer(text)]

    # Deduplicate, normalise case
    seen: set[str] = set()
    result: list[str] = []
    for e in entities:
        key = e.lower()
        if key not in seen and len(e) > 2:
            seen.add(key)
            result.append(e)
    return result


# ── Knowledge Graph ──────────────────────────────────────────────────────────

@dataclass
class GraphStats:
    nodes: int = 0
    edges: int = 0
    chunks_indexed: int = 0


class KnowledgeGraph:
    """Lightweight in-memory knowledge graph built from chunk payloads.

    Data structures:
      _entity_chunks:  entity_lower → {chunk_id}
      _cooccurrences:  entity_lower → {co-occurring entity_lower}
      _chunk_texts:    chunk_id → text  (for hit reconstruction)
    """

    def __init__(self) -> None:
        self._entity_chunks: dict[str, set[str]] = defaultdict(set)
        self._cooccurrences: dict[str, set[str]] = defaultdict(set)
        self._chunk_store: dict[str, dict] = {}   # chunk_id → payload dict
        self._chunks_indexed: int = 0

    # ------------------------------------------------------------------ #
    # Building the graph                                                   #
    # ------------------------------------------------------------------ #

    def ingest_chunk(self, payload: dict[str, Any]) -> None:
        """Add one chunk payload to the graph."""
        chunk_id = payload.get("chunk_id") or payload.get("id", "")
        text = payload.get("text", "")
        if not text or not chunk_id:
            return

        entities = _extract_entities(text)
        entity_keys = [e.lower() for e in entities]

        # Register entity → chunk mappings
        for ek in entity_keys:
            self._entity_chunks[ek].add(chunk_id)

        # Register co-occurrences (bidirectional)
        for i, a in enumerate(entity_keys):
            for b in entity_keys[i + 1:]:
                self._cooccurrences[a].add(b)
                self._cooccurrences[b].add(a)

        self._chunk_store[chunk_id] = payload
        self._chunks_indexed += 1

    def build_from_chunks(self, hits: list[dict[str, Any]]) -> None:
        """Ingest a list of retrieval hits (each with a ``payload`` key)."""
        for hit in hits:
            payload = hit.get("payload", {})
            if payload:
                self.ingest_chunk(payload)
        log.debug("Graph built: %d nodes, %d chunks indexed", len(self._entity_chunks), self._chunks_indexed)

    def clear(self) -> None:
        """Reset the graph."""
        self._entity_chunks.clear()
        self._cooccurrences.clear()
        self._chunk_store.clear()
        self._chunks_indexed = 0

    # ------------------------------------------------------------------ #
    # Query                                                                #
    # ------------------------------------------------------------------ #

    def expand_query_entities(self, query: str, hops: int = 1) -> list[str]:
        """Return entity keys reachable from query entities within *hops* hops."""
        seed_entities = [e.lower() for e in _extract_entities(query)]
        visited: set[str] = set(seed_entities)
        frontier = set(seed_entities)

        for _ in range(hops):
            next_frontier: set[str] = set()
            for e in frontier:
                for neighbour in self._cooccurrences.get(e, []):
                    if neighbour not in visited:
                        visited.add(neighbour)
                        next_frontier.add(neighbour)
            frontier = next_frontier
            if not frontier:
                break

        return list(visited)

    def get_chunks_for_entities(self, entity_keys: list[str]) -> list[dict[str, Any]]:
        """Return all stored chunk payloads matching any of *entity_keys*."""
        chunk_ids: set[str] = set()
        for ek in entity_keys:
            chunk_ids.update(self._entity_chunks.get(ek, set()))

        result = []
        for cid in chunk_ids:
            payload = self._chunk_store.get(cid)
            if payload:
                result.append({"id": cid, "score": 0.5, "payload": payload})
        return result

    # ------------------------------------------------------------------ #
    # Stats                                                                #
    # ------------------------------------------------------------------ #

    @property
    def stats(self) -> GraphStats:
        total_edges = sum(len(v) for v in self._cooccurrences.values()) // 2
        return GraphStats(
            nodes=len(self._entity_chunks),
            edges=total_edges,
            chunks_indexed=self._chunks_indexed,
        )


# ── GraphRAG Retriever ───────────────────────────────────────────────────────

class GraphRAGRetriever:
    """Augments dense retrieval with graph-based entity expansion.

    Strategy:
      1. Run normal hybrid search for the query.
      2. Ingest the dense hits into the knowledge graph (builds/updates it).
      3. Extract entities from the query, expand via 1-hop graph traversal.
      4. Fetch graph-matched chunks not already in the dense results.
      5. Merge with RRF, return top-*top_k*.

    Args:
        search:       HybridSearch instance.
        graph:        KnowledgeGraph (shared — can be pre-populated at index time).
        graph_hops:   Entity expansion depth (1 = direct neighbours only).
        graph_weight: Score multiplier applied to graph-retrieved chunks.
    """

    def __init__(
        self,
        search,
        graph: Optional[KnowledgeGraph] = None,
        graph_hops: int = 1,
        graph_weight: float = 0.5,
        rrf_k: int = 60,
    ) -> None:
        self._search = search
        self._graph = graph or KnowledgeGraph()
        self._graph_hops = graph_hops
        self._graph_weight = graph_weight
        self._rrf_k = rrf_k

    @property
    def graph(self) -> KnowledgeGraph:
        return self._graph

    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        file_filter: Optional[list[str]] = None,
    ) -> list[dict[str, Any]]:
        """Run graph-augmented retrieval.

        Returns merged hits in ``{"id", "score", "payload"}`` format.
        """
        # Dense retrieval
        dense_hits = self._search.search(query, top_k=top_k * 2, file_filter=file_filter)

        # Update graph with these chunks (incremental)
        self._graph.build_from_chunks(dense_hits)

        # Graph expansion
        expanded_entities = self._graph.expand_query_entities(query, hops=self._graph_hops)
        graph_hits = self._graph.get_chunks_for_entities(expanded_entities)

        # Adjust graph hit scores
        for h in graph_hits:
            h["score"] = h.get("score", 0.5) * self._graph_weight

        # Merge dense + graph hits with RRF
        merged = self._rrf_merge(dense_hits, graph_hits)
        log.info(
            "GraphRAG: %d dense + %d graph → %d merged",
            len(dense_hits), len(graph_hits), len(merged),
        )
        return merged[:top_k]

    def _rrf_merge(self, a: list[dict], b: list[dict]) -> list[dict]:
        scores: dict[str, float] = {}
        payloads: dict[str, dict] = {}

        for rank, hit in enumerate(a):
            cid = hit["id"]
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (self._rrf_k + rank + 1)
            payloads[cid] = hit.get("payload", {})

        for rank, hit in enumerate(b):
            cid = hit["id"]
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (self._rrf_k + rank + 1)
            if cid not in payloads:
                payloads[cid] = hit.get("payload", {})

        return [
            {"id": cid, "score": s, "payload": payloads[cid]}
            for cid, s in sorted(scores.items(), key=lambda x: x[1], reverse=True)
        ]
