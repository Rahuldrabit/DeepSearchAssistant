"""Backend abstraction layer for LLM inference."""
from .base import BackendInfo, GenerationConfig, LLMBackend
from .embedding_backend import EmbeddingBackend, RerankerBackend
from .llamacpp_backend import LlamaCppBackend
from .speculative_backend import SpeculativeBackend, SpeculativeStats

__all__ = [
    "LLMBackend",
    "GenerationConfig",
    "BackendInfo",
    "LlamaCppBackend",
    "EmbeddingBackend",
    "RerankerBackend",
    "SpeculativeBackend",
    "SpeculativeStats",
]
