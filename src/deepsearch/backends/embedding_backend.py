"""Embedding and reranking backend via sentence-transformers.

Kept separate from LLMBackend because embeddings are always loaded
(memory Slot C, ~90 MB) and never swapped out.
"""
from __future__ import annotations

import logging
from typing import Sequence

import numpy as np

log = logging.getLogger(__name__)

_EMBED_DIM = 384  # all-MiniLM-L6-v2


class EmbeddingBackend:
    """Dense embedding using sentence-transformers all-MiniLM-L6-v2."""

    def __init__(self) -> None:
        self._model = None
        self._model_name = ""

    def load(self, model_name: str = "all-MiniLM-L6-v2", device: str = "cpu") -> None:
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore[import]
        except ImportError as exc:
            raise RuntimeError(
                "sentence-transformers is not installed. "
                "Run: pip install sentence-transformers"
            ) from exc

        log.info("Loading embedding model: %s on %s", model_name, device)
        self._model = SentenceTransformer(model_name, device=device)
        self._model_name = model_name
        log.info("Embedding model loaded (%d dims)", _EMBED_DIM)

    def encode(
        self,
        texts: Sequence[str],
        batch_size: int = 64,
        normalize: bool = True,
    ) -> np.ndarray:
        """Return shape (N, 384) float32 array."""
        if self._model is None:
            raise RuntimeError("EmbeddingBackend not loaded. Call load() first.")
        return self._model.encode(
            list(texts),
            batch_size=batch_size,
            normalize_embeddings=normalize,
            show_progress_bar=False,
            convert_to_numpy=True,
        )

    def encode_single(self, text: str, normalize: bool = True) -> list[float]:
        vec = self.encode([text], normalize=normalize)
        return vec[0].tolist()

    @property
    def dimension(self) -> int:
        return _EMBED_DIM

    @property
    def is_loaded(self) -> bool:
        return self._model is not None


class RerankerBackend:
    """Cross-encoder reranker using ms-marco-MiniLM-L-6-v2."""

    def __init__(self) -> None:
        self._model = None

    def load(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        device: str = "cpu",
    ) -> None:
        try:
            from sentence_transformers import CrossEncoder  # type: ignore[import]
        except ImportError as exc:
            raise RuntimeError(
                "sentence-transformers is not installed."
            ) from exc

        log.info("Loading reranker: %s", model_name)
        self._model = CrossEncoder(model_name, device=device)
        log.info("Reranker loaded")

    def rerank(
        self,
        query: str,
        passages: list[str],
        top_k: int = 5,
    ) -> list[tuple[int, float]]:
        """Return list of (original_index, score) sorted descending."""
        if self._model is None:
            raise RuntimeError("RerankerBackend not loaded. Call load() first.")
        if not passages:
            return []
        pairs = [(query, p) for p in passages]
        scores: list[float] = self._model.predict(pairs).tolist()
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        return ranked[:top_k]

    @property
    def is_loaded(self) -> bool:
        return self._model is not None
