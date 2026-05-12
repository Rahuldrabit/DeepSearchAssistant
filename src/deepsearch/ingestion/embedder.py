"""Batch embedder — converts chunks to dense + sparse vectors.

Dense:  all-MiniLM-L6-v2 via EmbeddingBackend
Sparse: BM25 sparse vector with IDF weighting
        BM25 formula: IDF(t) * tf_norm(t,d)
        IDF(t)      = log((N - df(t) + 0.5) / (df(t) + 0.5) + 1)
        tf_norm(t,d) = tf(t,d) * (k1+1) / (tf(t,d) + k1*(1-b + b*dl/avgdl))
        k1=1.5, b=0.75  (Okapi BM25 standard parameters)
"""
from __future__ import annotations

import logging
import math
import re
from collections import Counter
from typing import TYPE_CHECKING

import numpy as np

from ..backends.embedding_backend import EmbeddingBackend
from ..storage.cache import EmbeddingCache
from .chunker import Chunk

if TYPE_CHECKING:
    pass

log = logging.getLogger(__name__)

# BM25 tuning parameters
_BM25_K1 = 1.5
_BM25_B = 0.75

# Simple English stopwords (kept minimal to avoid a heavy dep)
_STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "in", "on", "at", "to",
    "for", "of", "with", "by", "from", "is", "was", "are", "were",
    "be", "been", "being", "have", "has", "had", "do", "does", "did",
    "will", "would", "could", "should", "may", "might", "shall",
    "this", "that", "these", "those", "it", "its",
}


def _tokenize(text: str) -> list[str]:
    return [w for w in re.findall(r"\b\w+\b", text.lower()) if w not in _STOPWORDS]


def _bm25_sparse_vector(
    tokens: list[str],
    vocab: dict[str, int],
    doc_freqs: dict[str, int],
    doc_count: int,
    avg_doc_len: float,
) -> tuple[list[int], list[float]]:
    """Compute a BM25-weighted sparse vector.

    Args:
        tokens:      tokenised document/query
        vocab:       token → vocabulary index mapping
        doc_freqs:   token → document frequency (# docs containing the token)
        doc_count:   total documents indexed
        avg_doc_len: average document length in tokens

    Returns:
        (indices, values) for the sparse vector
    """
    dl = len(tokens)
    tf = Counter(tokens)
    n = max(doc_count, 1)
    adl = avg_doc_len if avg_doc_len > 0 else 1.0

    indices: list[int] = []
    values: list[float] = []

    for token, freq in tf.items():
        if token not in vocab:
            continue
        df = doc_freqs.get(token, 0)
        # Okapi IDF (with +1 smoothing to stay non-negative)
        idf = math.log((n - df + 0.5) / (df + 0.5) + 1.0)
        # BM25 TF normalisation
        tf_norm = freq * (_BM25_K1 + 1.0) / (
            freq + _BM25_K1 * (1.0 - _BM25_B + _BM25_B * dl / adl)
        )
        indices.append(vocab[token])
        values.append(idf * tf_norm)

    return indices, values


class Embedder:
    """Wraps EmbeddingBackend to produce dense + sparse vectors for chunks."""

    def __init__(
        self,
        embedding_backend: EmbeddingBackend,
        cache: EmbeddingCache | None = None,
        batch_size: int = 64,
    ) -> None:
        self._backend = embedding_backend
        self._cache = cache
        self._batch_size = batch_size

        # BM25 corpus statistics (updated as documents are indexed)
        self._vocab: dict[str, int] = {}       # token → vocab index
        self._doc_freqs: dict[str, int] = {}   # token → # docs containing it
        self._doc_count: int = 0               # total documents seen
        self._total_tokens: int = 0            # sum of all document lengths
        self._next_vocab_id: int = 0

    # ------------------------------------------------------------------ #
    # Vocabulary & statistics management                                   #
    # ------------------------------------------------------------------ #

    @property
    def avg_doc_len(self) -> float:
        return self._total_tokens / max(self._doc_count, 1)

    def _register_document(self, tokens: list[str]) -> None:
        """Update vocabulary and BM25 statistics for a new document."""
        self._doc_count += 1
        self._total_tokens += len(tokens)
        seen_in_doc: set[str] = set()
        for token in tokens:
            if token not in self._vocab:
                self._vocab[token] = self._next_vocab_id
                self._next_vocab_id += 1
            if token not in seen_in_doc:
                self._doc_freqs[token] = self._doc_freqs.get(token, 0) + 1
                seen_in_doc.add(token)

    def _register_query_tokens(self, tokens: list[str]) -> None:
        """Add new query tokens to vocab without affecting BM25 statistics."""
        for token in tokens:
            if token not in self._vocab:
                self._vocab[token] = self._next_vocab_id
                self._next_vocab_id += 1

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def embed_chunks(self, chunks: list[Chunk]) -> list[dict]:
        """Return list of dicts ready for VectorStore.upsert_batch().

        Keys: chunk_id, dense_vector, sparse_indices, sparse_values, payload
        """
        # First pass: update BM25 corpus statistics from all chunks
        chunk_tokens: list[list[str]] = []
        for chunk in chunks:
            tokens = _tokenize(chunk.text)
            chunk_tokens.append(tokens)
            self._register_document(tokens)

        # Second pass: dense embeddings (batched)
        texts = [c.text for c in chunks]
        dense_matrix = self._embed_texts_batched(texts)

        results = []
        for i, chunk in enumerate(chunks):
            sparse_i, sparse_v = _bm25_sparse_vector(
                chunk_tokens[i],
                self._vocab,
                self._doc_freqs,
                self._doc_count,
                self.avg_doc_len,
            )
            payload = {
                "file_id": chunk.file_id,
                "chunk_id": chunk.chunk_id,
                "text": chunk.text,
                "chunk_level": chunk.level,
                "chunk_index": chunk.chunk_index,
                "parent_id": chunk.parent_id,
                **chunk.metadata,
            }
            results.append({
                "chunk_id": chunk.chunk_id,
                "dense_vector": dense_matrix[i].tolist(),
                "sparse_indices": sparse_i,
                "sparse_values": sparse_v,
                "payload": payload,
            })
        return results

    def embed_query(self, query: str) -> tuple[list[float], list[int], list[float]]:
        """Embed a single query text.
        Returns (dense_vector, sparse_indices, sparse_values).
        """
        if self._cache is not None:
            cached = self._cache.get_embedding(query)
            if cached is not None:
                # Sparse is cheap to recompute
                tokens = _tokenize(query)
                si, sv = _bm25_sparse_vector(
                    tokens, self._vocab, self._doc_freqs,
                    self._doc_count, self.avg_doc_len,
                )
                return cached, si, sv

        dense = self._backend.encode_single(query)
        if self._cache is not None:
            self._cache.set_embedding(query, dense)

        tokens = _tokenize(query)
        self._register_query_tokens(tokens)
        si, sv = _bm25_sparse_vector(
            tokens, self._vocab, self._doc_freqs,
            self._doc_count, self.avg_doc_len,
        )
        return dense, si, sv

    # ------------------------------------------------------------------ #

    def _embed_texts_batched(self, texts: list[str]) -> np.ndarray:
        result = np.zeros((len(texts), self._backend.dimension), dtype=np.float32)
        miss_indices = []
        miss_texts = []

        for i, text in enumerate(texts):
            if self._cache is not None:
                cached = self._cache.get_embedding(text)
                if cached is not None:
                    result[i] = cached
                    continue
            miss_indices.append(i)
            miss_texts.append(text)

        if miss_texts:
            vecs = self._backend.encode(miss_texts, batch_size=self._batch_size)
            for j, idx in enumerate(miss_indices):
                result[idx] = vecs[j]
                if self._cache is not None:
                    self._cache.set_embedding(texts[idx], vecs[j].tolist())

        return result
