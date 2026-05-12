"""Partitioned vector store — separate Qdrant collections per modality.

At scale, mixing text, image, audio, and video embeddings in a single
collection degrades retrieval quality because each modality has different
embedding distributions.  This module provides a thin wrapper that routes
upsert and search operations to the correct per-modality collection.

Collections created:
  documents_text   — PDF, DOCX, TXT, HTML, etc.
  documents_image  — image chunks (OCR + caption)
  documents_audio  — audio transcript chunks
  documents_video  — video scene chunks

Cross-modality search (default) queries all four collections and merges
results with RRF.

Usage::

    store = PartitionedVectorStore(persist_path="data/qdrant")

    # Upsert a chunk — modality detected from payload["file_type"]
    store.upsert_batch([{...}])

    # Search a single modality
    hits = store.search_dense("text", query_vec, top_k=20)

    # Cross-modality search
    hits = store.search_all_modalities(query_vec, sparse_idx, sparse_vals, top_k=10)
"""
from __future__ import annotations

import logging
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
)

log = logging.getLogger(__name__)

DENSE_DIM = 384
DENSE_VECTOR = "dense"
SPARSE_VECTOR = "sparse"

# Map file_type values → modality partition names
_TYPE_TO_MODALITY: dict[str, str] = {
    # Text-family
    "pdf": "text", "docx": "text", "doc": "text",
    "txt": "text", "md": "text", "html": "text", "htm": "text",
    "json": "text", "csv": "text", "pptx": "text", "ppt": "text",
    "xlsx": "text", "xls": "text",
    # Image
    "jpg": "image", "jpeg": "image", "png": "image",
    "bmp": "image", "tiff": "image", "tif": "image",
    "webp": "image", "gif": "image",
    # Audio
    "mp3": "audio", "wav": "audio", "m4a": "audio",
    "ogg": "audio", "flac": "audio", "aac": "audio",
    # Video
    "mp4": "video", "avi": "video", "mov": "video",
    "mkv": "video", "webm": "video", "flv": "video",
}

MODALITIES = ("text", "image", "audio", "video")


def _collection_name(modality: str) -> str:
    return f"documents_{modality}"


def _modality_for(payload: dict[str, Any]) -> str:
    ft = str(payload.get("file_type", "")).lower().lstrip(".")
    return _TYPE_TO_MODALITY.get(ft, "text")


class PartitionedVectorStore:
    """Qdrant wrapper with per-modality collections.

    Shares the same ``QdrantClient`` instance (single Qdrant process) but
    routes to separate collection namespaces.

    Args:
        persist_path: Directory for Qdrant on-disk storage.
    """

    def __init__(self, persist_path: str | Path) -> None:
        self._path = Path(persist_path)
        self._path.mkdir(parents=True, exist_ok=True)
        self._client = QdrantClient(path=str(self._path))
        for m in MODALITIES:
            self._ensure_collection(m)

    # ------------------------------------------------------------------ #
    # Collection management                                                #
    # ------------------------------------------------------------------ #

    def _ensure_collection(self, modality: str) -> None:
        name = _collection_name(modality)
        existing = {c.name for c in self._client.get_collections().collections}
        if name in existing:
            return
        log.info("Creating collection: %s", name)
        self._client.create_collection(
            collection_name=name,
            vectors_config={
                DENSE_VECTOR: VectorParams(size=DENSE_DIM, distance=Distance.COSINE),
            },
            sparse_vectors_config={
                SPARSE_VECTOR: SparseVectorParams(index=SparseIndexParams(on_disk=False)),
            },
        )
        for field in ("file_id", "file_type", "chunk_level"):
            self._client.create_payload_index(
                collection_name=name, field_name=field, field_schema="keyword"
            )

    # ------------------------------------------------------------------ #
    # Write                                                                #
    # ------------------------------------------------------------------ #

    def upsert_batch(self, points: list[dict]) -> None:
        """Batch upsert.  Each item must have a ``payload`` with ``file_type``
        so the correct modality partition can be selected."""
        by_modality: dict[str, list[dict]] = {m: [] for m in MODALITIES}
        for p in points:
            m = _modality_for(p.get("payload", {}))
            by_modality[m].append(p)

        for modality, pts in by_modality.items():
            if not pts:
                continue
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
                for p in pts
            ]
            self._client.upsert(collection_name=_collection_name(modality), points=structs)
            log.debug("Upserted %d chunks to %s", len(structs), _collection_name(modality))

    def delete_by_file_id(self, file_id: str) -> None:
        """Remove all chunks for a file across all modality collections."""
        for m in MODALITIES:
            self._client.delete(
                collection_name=_collection_name(m),
                points_selector=Filter(
                    must=[FieldCondition(key="file_id", match=MatchValue(value=file_id))]
                ),
            )

    # ------------------------------------------------------------------ #
    # Search — single modality                                             #
    # ------------------------------------------------------------------ #

    def search_dense(
        self,
        modality: str,
        query_vector: list[float],
        top_k: int = 50,
        filter_: Optional[Filter] = None,
    ) -> list[dict[str, Any]]:
        results = self._client.search(
            collection_name=_collection_name(modality),
            query_vector=NamedVector(name=DENSE_VECTOR, vector=query_vector),
            limit=top_k,
            query_filter=filter_,
            with_payload=True,
        )
        return [{"id": str(r.id), "score": r.score, "payload": r.payload or {}} for r in results]

    # ------------------------------------------------------------------ #
    # Search — cross-modality                                              #
    # ------------------------------------------------------------------ #

    def search_all_modalities(
        self,
        query_vector: list[float],
        sparse_indices: list[int],
        sparse_values: list[float],
        top_k: int = 10,
        modalities: Optional[list[str]] = None,
        file_filter: Optional[list[str]] = None,
    ) -> list[dict[str, Any]]:
        """Query all (or specified) modality collections and merge with RRF."""
        targets = modalities or list(MODALITIES)
        filter_ = self._build_file_filter(file_filter)

        all_hits: list[list[dict]] = []
        for m in targets:
            hits = self.search_dense(m, query_vector, top_k=top_k * 2, filter_=filter_)
            if hits:
                all_hits.append(hits)

        return self._rrf_merge(all_hits, top_k)

    # ------------------------------------------------------------------ #
    # Stats                                                                #
    # ------------------------------------------------------------------ #

    def counts(self) -> dict[str, int]:
        """Return point count per modality."""
        result = {}
        for m in MODALITIES:
            try:
                info = self._client.get_collection(_collection_name(m))
                result[m] = info.points_count or 0
            except Exception:
                result[m] = 0
        return result

    # ------------------------------------------------------------------ #
    # Helpers                                                              #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _build_file_filter(file_filter: Optional[list[str]]) -> Optional[Filter]:
        if not file_filter:
            return None
        from qdrant_client.models import MatchAny  # type: ignore[import]
        return Filter(must=[FieldCondition(key="file_id", match=MatchAny(any=file_filter))])

    @staticmethod
    def _rrf_merge(ranked_lists: list[list[dict]], top_k: int, k: int = 60) -> list[dict]:
        scores: dict[str, float] = {}
        payloads: dict[str, dict] = {}
        for hits in ranked_lists:
            for rank, hit in enumerate(hits):
                cid = hit["id"]
                scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank + 1)
                if cid not in payloads:
                    payloads[cid] = hit.get("payload", {})
        merged = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [{"id": cid, "score": s, "payload": payloads[cid]} for cid, s in merged[:top_k]]
