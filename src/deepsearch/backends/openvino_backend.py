"""OpenVINO backend — optional accelerator for Intel GPU/NPU.

Only imported when openvino-genai is available.  Falls back gracefully
to LlamaCppBackend if the package is missing or the device is absent.
"""
from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import AsyncIterator, Iterator

from .base import BackendInfo, GenerationConfig, LLMBackend

log = logging.getLogger(__name__)


class OpenVINOBackend(LLMBackend):
    """Wraps openvino_genai.LLMPipeline for Intel GPU / NPU inference.

    Models must be in OpenVINO IR format (int4 or int8 compressed).
    Use ``optimum-intel`` to export: ``optimum-cli export openvino ...``
    """

    def __init__(self) -> None:
        self._pipeline = None  # openvino_genai.LLMPipeline
        self._info = BackendInfo(
            name="openvino",
            model_id="",
            device="gpu",
            loaded=False,
        )

    # ------------------------------------------------------------------ #
    # Lifecycle                                                            #
    # ------------------------------------------------------------------ #

    def load(self, model_path: str, **kwargs) -> None:
        try:
            import openvino_genai as ov_genai  # type: ignore[import]
        except ImportError as exc:
            raise RuntimeError(
                "openvino-genai is not installed. "
                "Run: pip install openvino-genai"
            ) from exc

        device: str = kwargs.get("device", "GPU")
        path = Path(model_path)
        if not path.exists():
            raise FileNotFoundError(f"OpenVINO model dir not found: {model_path}")

        log.info("Loading OpenVINO model from %s on %s", path, device)
        try:
            self._pipeline = ov_genai.LLMPipeline(str(path), device)
        except Exception as exc:
            raise RuntimeError(f"OpenVINO load failed: {exc}") from exc

        self._info = BackendInfo(
            name="openvino",
            model_id=path.name,
            device=device.lower(),
            loaded=True,
            context_length=kwargs.get("context_length", 4096),
        )
        log.info("OpenVINO model loaded on %s", device)

    def unload(self) -> None:
        if self._pipeline is not None:
            del self._pipeline
            self._pipeline = None
        self._info.loaded = False
        log.info("OpenVINOBackend unloaded")

    # ------------------------------------------------------------------ #
    # Inference                                                            #
    # ------------------------------------------------------------------ #

    def generate(self, prompt: str, config: GenerationConfig) -> str:
        self._require_loaded()
        import openvino_genai as ov_genai  # type: ignore[import]

        gen_config = ov_genai.GenerationConfig()
        gen_config.max_new_tokens = config.max_tokens
        gen_config.temperature = config.temperature
        gen_config.top_p = config.top_p
        gen_config.top_k = config.top_k
        gen_config.repetition_penalty = config.repeat_penalty

        return self._pipeline.generate(prompt, gen_config)

    def stream(self, prompt: str, config: GenerationConfig) -> Iterator[str]:
        self._require_loaded()
        import openvino_genai as ov_genai  # type: ignore[import]

        tokens: list[str] = []

        def _streamer(token: str) -> bool:
            tokens.append(token)
            return False  # False = continue

        gen_config = ov_genai.GenerationConfig()
        gen_config.max_new_tokens = config.max_tokens
        gen_config.temperature = config.temperature

        # openvino_genai supports a streamer callback
        self._pipeline.generate(prompt, gen_config, _streamer)
        yield from tokens

    async def agenerate(self, prompt: str, config: GenerationConfig) -> str:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self.generate, prompt, config)

    async def astream(
        self, prompt: str, config: GenerationConfig
    ) -> AsyncIterator[str]:
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue[str | None] = asyncio.Queue()

        def _produce() -> None:
            try:
                for token in self.stream(prompt, config):
                    asyncio.run_coroutine_threadsafe(queue.put(token), loop)
            except Exception as e:
                log.error("Error in stream production: %s", e)
            finally:
                asyncio.run_coroutine_threadsafe(queue.put(None), loop)

        fut = asyncio.ensure_future(loop.run_in_executor(None, _produce))
        while True:
            token = await queue.get()
            if token is None:
                break
            yield token
        await fut

    @property
    def info(self) -> BackendInfo:
        return self._info

    def _require_loaded(self) -> None:
        if self._pipeline is None:
            raise RuntimeError("OpenVINOBackend: pipeline not loaded. Call load() first.")
