"""Tests for TTL LRU caches."""
import time
import pytest
from deepsearch.storage.cache import (
    TTLCache,
    EmbeddingCache,
    QueryCache,
    AnswerCache,
    CacheBundle,
)


class TestTTLCache:
    def test_set_and_get(self):
        cache: TTLCache[str] = TTLCache(max_size=10, ttl_seconds=60)
        cache.set("k", "v")
        assert cache.get("k") == "v"

    def test_miss_returns_none(self):
        cache: TTLCache[str] = TTLCache(max_size=10, ttl_seconds=60)
        assert cache.get("missing") is None

    def test_expired_returns_none(self):
        cache: TTLCache[str] = TTLCache(max_size=10, ttl_seconds=0.01)
        cache.set("k", "v")
        time.sleep(0.05)
        assert cache.get("k") is None

    def test_lru_eviction(self):
        cache: TTLCache[int] = TTLCache(max_size=3, ttl_seconds=60)
        cache.set("a", 1)
        cache.set("b", 2)
        cache.set("c", 3)
        cache.set("d", 4)  # evicts "a" (LRU)
        assert cache.get("a") is None
        assert cache.get("b") == 2

    def test_access_refreshes_lru(self):
        cache: TTLCache[int] = TTLCache(max_size=3, ttl_seconds=60)
        cache.set("a", 1)
        cache.set("b", 2)
        cache.set("c", 3)
        cache.get("a")   # "a" is now MRU
        cache.set("d", 4)  # evicts "b" (now LRU)
        assert cache.get("a") == 1
        assert cache.get("b") is None

    def test_invalidate(self):
        cache: TTLCache[str] = TTLCache(max_size=10, ttl_seconds=60)
        cache.set("k", "v")
        cache.invalidate("k")
        assert cache.get("k") is None

    def test_clear(self):
        cache: TTLCache[int] = TTLCache(max_size=10, ttl_seconds=60)
        cache.set("a", 1)
        cache.set("b", 2)
        cache.clear()
        assert len(cache) == 0

    def test_len(self):
        cache: TTLCache[int] = TTLCache(max_size=10, ttl_seconds=60)
        assert len(cache) == 0
        cache.set("a", 1)
        cache.set("b", 2)
        assert len(cache) == 2


class TestEmbeddingCache:
    def test_store_and_retrieve(self):
        cache = EmbeddingCache()
        vec = [0.1, 0.2, 0.3]
        cache.set_embedding("hello world", vec)
        assert cache.get_embedding("hello world") == vec

    def test_same_text_same_key(self):
        cache = EmbeddingCache()
        cache.set_embedding("test text", [1.0])
        # Same text should hit
        assert cache.get_embedding("test text") is not None

    def test_different_text_miss(self):
        cache = EmbeddingCache()
        cache.set_embedding("foo", [1.0])
        assert cache.get_embedding("bar") is None


class TestQueryCache:
    def test_cache_results(self):
        cache = QueryCache()
        results = [{"id": "1", "score": 0.9}]
        cache.set_results("what is AI?", "fast", results)
        assert cache.get_results("what is AI?", "fast") == results

    def test_mode_differentiation(self):
        cache = QueryCache()
        cache.set_results("query", "fast", [{"id": "fast"}])
        cache.set_results("query", "deep", [{"id": "deep"}])
        assert cache.get_results("query", "fast")[0]["id"] == "fast"
        assert cache.get_results("query", "deep")[0]["id"] == "deep"


class TestAnswerCache:
    def test_cache_answer(self):
        cache = AnswerCache()
        cache.set_answer("what is Python?", "fast", "Python is a programming language.")
        result = cache.get_answer("what is Python?", "fast")
        assert result == "Python is a programming language."


class TestCacheBundle:
    def test_has_all_caches(self):
        bundle = CacheBundle()
        assert isinstance(bundle.embeddings, EmbeddingCache)
        assert isinstance(bundle.queries, QueryCache)
        assert isinstance(bundle.answers, AnswerCache)

    def test_clear_all(self):
        bundle = CacheBundle()
        bundle.embeddings.set_embedding("t", [1.0])
        bundle.answers.set_answer("q", "fast", "ans")
        bundle.clear_all()
        assert bundle.embeddings.get_embedding("t") is None
        assert bundle.answers.get_answer("q", "fast") is None
