"""Shared base for OpenAI-compatible local server backends.

Both Ollama and LM Studio expose the OpenAI /v1/chat/completions API,
so all inference logic lives here.  Provider-specific health checks and
model-listing are handled by each subclass.
"""
from __future__ import annotations

import json
import logging
from typing import AsyncIterator, Iterator

from ..base import BackendInfo, GenerationConfig, LLMBackend

log = logging.getLogger(__name__)


class OpenAICompatBackend(LLMBackend):
    """Base backend for any server that speaks OpenAI /v1/chat/completions."""

    _PROVIDER_NAME = "openai_compat"
    _DEFAULT_BASE_URL = "http://localhost:8080"

    def __init__(self) -> None:
        self._client = None
        self._async_client = None
        self._model_id = ""
        self._base_url = self._DEFAULT_BASE_URL
        self._info = BackendInfo(
            name=self._PROVIDER_NAME,
            model_id="",
            device="local",
            loaded=False,
        )

    # ------------------------------------------------------------------ #
    # Lifecycle                                                            #
    # ------------------------------------------------------------------ #

    def load(self, model_path: str, **kwargs) -> None:
        """Connect to the running local server.

        Args:
            model_path: Model name as known to the server
                        (e.g. ``"llama3.2:3b"`` for Ollama,
                        ``"meta-llama-3.1-8b-instruct"`` for LM Studio).
            base_url:   Override server URL (kwarg).
            timeout:    HTTP timeout in seconds (kwarg, default 120).
        """
        import httpx  # type: ignore[import]

        self._model_id = model_path
        self._base_url = kwargs.get("base_url", self._DEFAULT_BASE_URL)
        timeout = float(kwargs.get("timeout", 120.0))

        self._client = httpx.Client(base_url=self._base_url, timeout=timeout)
        self._async_client = httpx.AsyncClient(base_url=self._base_url, timeout=timeout)

        self._health_check()

        self._info = BackendInfo(
            name=self._PROVIDER_NAME,
            model_id=self._model_id,
            device="local",
            loaded=True,
        )
        log.info(
            "%s backend ready — model: %s  url: %s",
            self._PROVIDER_NAME, self._model_id, self._base_url,
        )

    def _health_check(self) -> None:
        """Override in subclass to verify server reachability."""

    def unload(self) -> None:
        if self._client:
            self._client.close()
        if self._async_client:
            import asyncio

            try:
                loop = asyncio.get_running_loop()
                loop.create_task(self._async_client.aclose())
            except RuntimeError:
                asyncio.run(self._async_client.aclose())
        self._info.loaded = False

    # ------------------------------------------------------------------ #
    # Helpers                                                             #
    # ------------------------------------------------------------------ #

    def _build_payload(
        self, prompt: str, config: GenerationConfig, stream: bool = False
    ) -> dict:
        payload: dict = {
            "model": self._model_id,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": config.max_tokens,
            "temperature": config.temperature,
            "top_p": config.top_p,
            "stream": stream,
        }
        if config.stop:
            payload["stop"] = config.stop
        return payload

    @staticmethod
    def _extract_stream_delta(line: str) -> str:
        """Parse a single SSE ``data: ...`` line and return the text delta."""
        if not line.startswith("data: "):
            return ""
        chunk_str = line[6:].strip()
        if chunk_str == "[DONE]":
            return ""
        try:
            chunk = json.loads(chunk_str)
            delta = chunk["choices"][0].get("delta", {})
            return delta.get("content") or ""
        except Exception:
            return ""

    def _require_loaded(self) -> None:
        if not self._info.loaded:
            raise RuntimeError(
                f"{self._PROVIDER_NAME} backend not loaded. Call load() first."
            )

    # ------------------------------------------------------------------ #
    # Inference                                                           #
    # ------------------------------------------------------------------ #

    def generate(self, prompt: str, config: GenerationConfig) -> str:
        self._require_loaded()
        payload = self._build_payload(prompt, config, stream=False)
        resp = self._client.post("/v1/chat/completions", json=payload)
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"]

    def stream(self, prompt: str, config: GenerationConfig) -> Iterator[str]:
        self._require_loaded()
        payload = self._build_payload(prompt, config, stream=True)
        with self._client.stream("POST", "/v1/chat/completions", json=payload) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                if line.strip() == "data: [DONE]":
                    break
                token = self._extract_stream_delta(line)
                if token:
                    yield token

    async def agenerate(self, prompt: str, config: GenerationConfig) -> str:
        self._require_loaded()
        payload = self._build_payload(prompt, config, stream=False)
        resp = await self._async_client.post("/v1/chat/completions", json=payload)
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"]

    async def astream(
        self, prompt: str, config: GenerationConfig
    ) -> AsyncIterator[str]:
        self._require_loaded()
        payload = self._build_payload(prompt, config, stream=True)
        async with self._async_client.stream(
            "POST", "/v1/chat/completions", json=payload
        ) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if line.strip() == "data: [DONE]":
                    break
                token = self._extract_stream_delta(line)
                if token:
                    yield token

    def list_models(self) -> list[str]:
        """Return model IDs available on the server (OpenAI /v1/models)."""
        resp = self._client.get("/v1/models")
        resp.raise_for_status()
        return [m["id"] for m in resp.json().get("data", [])]

    @property
    def info(self) -> BackendInfo:
        return self._info
