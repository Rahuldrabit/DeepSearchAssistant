"""llama-cpp-python backend — primary inference engine.

Supports CPU, CUDA, ROCm, Metal, and Intel SYCL transparently via
the n_gpu_layers parameter and the llama-cpp-python build variant.
"""
from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import AsyncIterator, Iterator

from .base import BackendInfo, GenerationConfig, LLMBackend

log = logging.getLogger(__name__)


class LlamaCppBackend(LLMBackend):
    """Wraps llama_cpp.Llama for local GGUF inference."""

    def __init__(self) -> None:
        self._model = None  # llama_cpp.Llama instance
        self._info = BackendInfo(
            name="llamacpp",
            model_id="",
            device="cpu",
            loaded=False,
        )

    # ------------------------------------------------------------------ #
    # Lifecycle                                                            #
    # ------------------------------------------------------------------ #

    def load(self, model_path: str, **kwargs) -> None:
        try:
            from llama_cpp import Llama  # type: ignore[import]
        except ImportError as exc:
            raise RuntimeError(
                "llama-cpp-python is not installed. "
                "Run: pip install llama-cpp-python"
            ) from exc

        path = Path(model_path)
        if not path.exists():
            raise FileNotFoundError(f"GGUF model not found: {model_path}")

        n_gpu_layers: int = kwargs.get("n_gpu_layers", 0)
        context_length: int = kwargs.get("context_length", 4096)
        n_threads: int = kwargs.get("n_threads", 0)  # 0 = auto
        verbose: bool = kwargs.get("verbose", False)

        log.info("Loading %s  (n_gpu_layers=%d, ctx=%d)", path.name, n_gpu_layers, context_length)

        self._model = Llama(
            model_path=str(path),
            n_ctx=context_length,
            n_gpu_layers=n_gpu_layers,
            n_threads=n_threads or None,
            verbose=verbose,
        )

        device = "cpu" if n_gpu_layers == 0 else "gpu"
        self._info = BackendInfo(
            name="llamacpp",
            model_id=path.stem,
            device=device,
            loaded=True,
            context_length=context_length,
            n_gpu_layers=n_gpu_layers,
        )
        log.info("Model loaded: %s on %s", path.name, device)

    def unload(self) -> None:
        if self._model is not None:
            del self._model
            self._model = None
        self._info.loaded = False
        log.info("LlamaCppBackend unloaded")

    # ------------------------------------------------------------------ #
    # Inference                                                            #
    # ------------------------------------------------------------------ #

    def generate(self, prompt: str, config: GenerationConfig) -> str:
        self._require_loaded()
        output = self._model(
            prompt,
            max_tokens=config.max_tokens,
            temperature=config.temperature,
            top_p=config.top_p,
            top_k=config.top_k,
            repeat_penalty=config.repeat_penalty,
            stop=config.stop or [],
            echo=False,
        )
        return output["choices"][0]["text"]

    def stream(self, prompt: str, config: GenerationConfig) -> Iterator[str]:
        self._require_loaded()
        for chunk in self._model(
            prompt,
            max_tokens=config.max_tokens,
            temperature=config.temperature,
            top_p=config.top_p,
            top_k=config.top_k,
            repeat_penalty=config.repeat_penalty,
            stop=config.stop or [],
            echo=False,
            stream=True,
        ):
            token = chunk["choices"][0]["text"]
            if token:
                yield token

    async def agenerate(self, prompt: str, config: GenerationConfig) -> str:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.generate, prompt, config)

    async def astream(
        self, prompt: str, config: GenerationConfig
    ) -> AsyncIterator[str]:
        # Run blocking iterator in a thread and bridge to async
        loop = asyncio.get_event_loop()
        queue: asyncio.Queue[str | None] = asyncio.Queue()

        def _produce() -> None:
            try:
                for token in self.stream(prompt, config):
                    asyncio.run_coroutine_threadsafe(queue.put(token), loop)
            finally:
                asyncio.run_coroutine_threadsafe(queue.put(None), loop)

        loop.run_in_executor(None, _produce)
        while True:
            token = await queue.get()
            if token is None:
                break
            yield token

    # ------------------------------------------------------------------ #
    # Chat-template aware prompt building                                  #
    # ------------------------------------------------------------------ #

    def build_prompt(self, system: str, user: str, context: str = "") -> str:
        """Uses Jinja2 chat template if model metadata provides one,
        falls back to generic format."""
        ctx_block = f"\n\nContext:\n{context}" if context else ""
        if self._model and hasattr(self._model, "metadata"):
            meta = self._model.metadata or {}
            tmpl = meta.get("tokenizer.chat_template", "")
            if tmpl:
                # Use llama_cpp's apply_chat_template if available
                try:
                    messages = [
                        {"role": "system", "content": system + ctx_block},
                        {"role": "user", "content": user},
                    ]
                    return self._model.tokenizer().apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True
                    )
                except Exception:
                    pass
        # Generic fallback
        return super().build_prompt(system, user, context)

    @property
    def info(self) -> BackendInfo:
        return self._info

    # ------------------------------------------------------------------ #

    def _require_loaded(self) -> None:
        if self._model is None:
            raise RuntimeError("LlamaCppBackend: model not loaded. Call load() first.")
