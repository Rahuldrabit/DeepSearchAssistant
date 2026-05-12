"""Qdrant vector store wrapper.

Collections use named vectors:
  - "dense"  : all-MiniLM-L6-v2 embeddings (dim=384, cosine)
  - "sparse" : BM25 sparse vectors (handled via FastEmbed)

Each point payload follows the metadata schema in ARCHITECTURE.md.
"""
from __future__ import annotations

import logging
import uuid
from pathlib import Path
from typing import Any, Optional

from qdrant_client import QdrantClient  # type: ignore[import]
from qdrant_client.models import (  # type: ignore[import]
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    NamedVector,
    PointStruct,
    SparseIndexParams,
    SparseVectorParams,
    VectorParams,
    VectorsConfig,
)

log = logging.getLogger(__name__)

COLLECTION_NAME = "documents"
DENSE_DIM = 384
DENSE_VECTOR = "dense"
SPARSE_VECTOR = "sparse"


class VectorStore:
    """Qdrant local vector store with dense + sparse named vectors."""

    def __init__(self, persist_path: str | Path) -> None:
        self._path = Path(persist_path)
        self._path.mkdir(parents=True, exist_ok=True)
        self._client: QdrantClient = QdrantClient(path=str(self._path))
        self._ensure_collection()

    # ------------------------------------------------------------------ #
    # Setup                                                                #
    # ------------------------------------------------------------------ #

    def _ensure_collection(self) -> None:
        existing = [c.name for c in self._client.get_collections().collections]
        if COLLECTION_NAME in existing:
            return

        log.info("Creating Qdrant collection: %s", COLLECTION_NAME)
        self._client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config={
                DENSE_VECTOR: VectorParams(size=DENSE_DIM, distance=Distance.COSINE),
            },
            sparse_vectors_config={
                SPARSE_VECTOR: SparseVectorParams(index=SparseIndexParams(on_disk=False)),
            },
        )
        # Payload indexes for fast filtering
        for field in ("file_id", "file_type", "chunk_level"):
            self._client.create_payload_index(
                collection_name=COLLECTION_NAME,
                field_name=field,
                field_schema="keyword",
            )

    # ------------------------------------------------------------------ #
    # Write                                                                #
    # ------------------------------------------------------------------ #

    def upsert(
        self,
        chunk_id: str,
        dense_vector: list[float],
        sparse_indices: list[int],
        sparse_values: list[float],
        payload: dict[str, Any],
    ) -> None:
        point = PointStruct(
            id=chunk_id,
            vector={
                DENSE_VECTOR: dense_vector,
                SPARSE_VECTOR: {"indices": sparse_indices, "values": sparse_values},
            },
            payload=payload,
        )
        self._client.upsert(collection_name=COLLECTION_NAME, points=[point])

    def upsert_batch(self, points: list[dict]) -> None:
        """Batch upsert. Each item: {chunk_id, dense_vector, sparse_indices,
        sparse_values, payload}."""
        structs = [
            PointStruct(
                id=p["chunk_id"],
                vector={
                    DENSE_VECTOR: p["dense_vector"],
                    SPARSE_VECTOR: {
                        "indices": p.get("sparse_indices", []),
                        "values": p.get("sparse_values", []),
                    },
                },
                payload=p["payload"],
            )
            for p in points
        ]
        self._client.upsert(collection_name=COLLECTION_NAME, points=structs)
        log.debug("Upserted %d chunks", len(structs))

    def delete_by_file_id(self, file_id: str) -> None:
        self._client.delete(
            collection_name=COLLECTION_NAME,
            points_selector=Filter(
                must=[FieldCondition(key="file_id", match=MatchValue(value=file_id))]
            ),
        )

    # ------------------------------------------------------------------ #
    # Search                                                               #
    # ------------------------------------------------------------------ #

    def search_dense(
        self,
        query_vector: list[float],
        top_k: int = 50,
        filter_: Optional[Filter] = None,
    ) -> list[dict[str, Any]]:
        results = self._client.search(
            collection_name=COLLECTION_NAME,
            query_vector=NamedVector(name=DENSE_VECTOR, vector=query_vector),
            limit=top_k,
            query_filter=filter_,
            with_payload=True,
        )
        return [
            {"id": str(r.id), "score": r.score, "payload": r.payload or {}}
            for r in results
        ]

    def search_sparse(
        self,
        sparse_indices: list[int],
        sparse_values: list[float],
        top_k: int = 50,
        filter_: Optional[Filter] = None,
    ) -> list[dict[str, Any]]:
        from qdrant_client.models import NamedSparseVector, SparseVector  # type: ignore[import]

        results = self._client.search(
            collection_name=COLLECTION_NAME,
            query_vector=NamedSparseVector(
                name=SPARSE_VECTOR,
                vector=SparseVector(indices=sparse_indices, values=sparse_values),
            ),
            limit=top_k,
            query_filter=filter_,
            with_payload=True,
        )
        return [
            {"id": str(r.id), "score": r.score, "payload": r.payload or {}}
            for r in results
        ]

    def get_by_ids(self, ids: list[str]) -> list[dict[str, Any]]:
        records = self._client.retrieve(
            collection_name=COLLECTION_NAME,
            ids=ids,
            with_payload=True,
        )
        return [{"id": str(r.id), "payload": r.payload or {}} for r in records]

    def boost_file(self, file_id: str, boost_delta: float = 0.1) -> None:
        """Increment the ``score_boost`` payload field for all chunks of a file.

        This allows positively-rated files to rank higher in future searches.
        The caller is responsible for incorporating ``score_boost`` into the
        retrieval score (e.g., by adding it to the cosine similarity).

        Args:
            file_id:     The file whose chunks should be boosted.
            boost_delta: Amount to add to the current boost value (default 0.1).
        """
        # Fetch current boosts for all chunks of this file
        results = self._client.scroll(
            collection_name=COLLECTION_NAME,
            scroll_filter=Filter(
                must=[FieldCondition(key="file_id", match=MatchValue(value=file_id))]
            ),
            limit=10_000,
            with_payload=True,
            with_vectors=False,
        )
        points, _ = results
        if not points:
            log.debug("boost_file: no chunks found for file_id=%s", file_id)
            return

        from qdrant_client.models import SetPayload  # type: ignore[import]
        for point in points:
            payload = point.payload or {}
            current = float(payload.get("score_boost", 0.0))
            new_boost = round(current + boost_delta, 4)
            self._client.set_payload(
                collection_name=COLLECTION_NAME,
                payload={"score_boost": new_boost},
                points=[point.id],
            )
        log.info(
            "Boosted %d chunks for file_id=%s by +%.2f",
            len(points), file_id, boost_delta,
        )

    # ------------------------------------------------------------------ #
    # Stats                                                                #
    # ------------------------------------------------------------------ #

    @property
    def count(self) -> int:
        info = self._client.get_collection(COLLECTION_NAME)
        return info.points_count or 0
