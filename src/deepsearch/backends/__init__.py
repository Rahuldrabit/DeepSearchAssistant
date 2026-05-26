"""Backend abstraction layer for LLM inference."""
from .base import BackendInfo, GenerationConfig, LLMBackend
from .embedding_backend import EmbeddingBackend, RerankerBackend
from .llamacpp_backend import LlamaCppBackend
from .local_providers import LMStudioBackend, OllamaBackend, create_local_provider
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
    # Local providers (Ollama, LM Studio)
    "OllamaBackend",
    "LMStudioBackend",
    "create_local_provider",
]
