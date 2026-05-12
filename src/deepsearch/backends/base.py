"""Abstract base protocol for all LLM backends."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import AsyncIterator, Iterator


@dataclass
class GenerationConfig:
    max_tokens: int = 512
    temperature: float = 0.2
    top_p: float = 0.95
    top_k: int = 40
    repeat_penalty: float = 1.1
    stop: list[str] = field(default_factory=list)
    stream: bool = False


@dataclass
class BackendInfo:
    name: str           # e.g. "llamacpp", "openvino", "cloud"
    model_id: str       # model name/path used to load
    device: str         # "cpu", "cuda:0", "gpu", "npu", "cloud"
    loaded: bool = False
    context_length: int = 4096
    n_gpu_layers: int = 0


class LLMBackend(ABC):
    """Protocol that every LLM backend must satisfy.

    Backends are interchangeable: LlamaCppBackend, OpenVINOBackend,
    and CloudAPIBackend all implement this interface.
    """

    @abstractmethod
    def load(self, model_path: str, **kwargs) -> None:
        """Load a model into memory.  Called once; raises on failure."""

    @abstractmethod
    def unload(self) -> None:
        """Release all memory held by this backend."""

    @abstractmethod
    def generate(self, prompt: str, config: GenerationConfig) -> str:
        """Blocking generation — returns full response string."""

    @abstractmethod
    def stream(self, prompt: str, config: GenerationConfig) -> Iterator[str]:
        """Yield tokens one at a time for streaming UI updates."""

    @abstractmethod
    async def agenerate(self, prompt: str, config: GenerationConfig) -> str:
        """Async generate for concurrent requests (cloud backend)."""

    @abstractmethod
    async def astream(
        self, prompt: str, config: GenerationConfig
    ) -> AsyncIterator[str]:
        """Async streaming variant."""

    @property
    @abstractmethod
    def info(self) -> BackendInfo:
        """Return metadata about the loaded backend/model."""

    @property
    def is_loaded(self) -> bool:
        return self.info.loaded

    # ------------------------------------------------------------------ #
    # Shared helpers                                                       #
    # ------------------------------------------------------------------ #

    def build_prompt(self, system: str, user: str, context: str = "") -> str:
        """Build a minimal instruction prompt.

        Override per-backend for chat-template aware formatting.
        """
        ctx_block = f"\n\nContext:\n{context}" if context else ""
        return (
            f"<|system|>\n{system}{ctx_block}\n"
            f"<|user|>\n{user}\n"
            f"<|assistant|>\n"
        )
