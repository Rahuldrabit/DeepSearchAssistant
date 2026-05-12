"""LRU caches with TTL for embeddings, query results, and answers."""
from __future__ import annotations

import hashlib
import time
from collections import OrderedDict
from typing import Generic, Optional, TypeVar

V = TypeVar("V")


class TTLCache(Generic[V]):
    """Thread-safe LRU cache with per-entry TTL."""

    def __init__(self, max_size: int, ttl_seconds: float) -> None:
        self._max = max_size
        self._ttl = ttl_seconds
        self._store: OrderedDict[str, tuple[V, float]] = OrderedDict()

    def get(self, key: str) -> Optional[V]:
        entry = self._store.get(key)
        if entry is None:
            return None
        value, ts = entry
        if time.time() - ts > self._ttl:
            del self._store[key]
            return None
        # Move to end (most recently used)
        self._store.move_to_end(key)
        return value

    def set(self, key: str, value: V) -> None:
        if key in self._store:
            self._store.move_to_end(key)
        self._store[key] = (value, time.time())
        if len(self._store) > self._max:
            self._store.popitem(last=False)  # evict LRU

    def invalidate(self, key: str) -> None:
        self._store.pop(key, None)

    def clear(self) -> None:
        self._store.clear()

    def __len__(self) -> int:
        return len(self._store)


def _text_key(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


class EmbeddingCache(TTLCache[list[float]]):
    """Cache for single-text dense embeddings (384-dim float lists).

    Key = SHA-256 of the text — content-addressed so equivalent text
    always hits the cache regardless of caller.
    """

    def __init__(self, max_size: int = 10_000, ttl_seconds: float = 86400) -> None:
        super().__init__(max_size, ttl_seconds)

    def get_embedding(self, text: str) -> Optional[list[float]]:
        return self.get(_text_key(text))

    def set_embedding(self, text: str, vector: list[float]) -> None:
        self.set(_text_key(text), vector)


class QueryCache(TTLCache[list[dict]]):
    """Cache for retrieved chunk lists.

    Key = SHA-256(query + search_mode).
    """

    def __init__(self, max_size: int = 500, ttl_seconds: float = 600) -> None:
        super().__init__(max_size, ttl_seconds)

    def get_results(self, query: str, mode: str = "fast") -> Optional[list[dict]]:
        return self.get(_text_key(f"{mode}::{query}"))

    def set_results(self, query: str, mode: str, results: list[dict]) -> None:
        self.set(_text_key(f"{mode}::{query}"), results)


class AnswerCache(TTLCache[str]):
    """Cache for final generated answers."""

    def __init__(self, max_size: int = 200, ttl_seconds: float = 1800) -> None:
        super().__init__(max_size, ttl_seconds)

    def get_answer(self, query: str, mode: str = "fast") -> Optional[str]:
        return self.get(_text_key(f"{mode}::{query}"))

    def set_answer(self, query: str, mode: str, answer: str) -> None:
        self.set(_text_key(f"{mode}::{query}"), answer)


class CacheBundle:
    """Convenience container for all three caches."""

    def __init__(self) -> None:
        self.embeddings = EmbeddingCache()
        self.queries = QueryCache()
        self.answers = AnswerCache()

    def clear_all(self) -> None:
        self.embeddings.clear()
        self.queries.clear()
        self.answers.clear()
