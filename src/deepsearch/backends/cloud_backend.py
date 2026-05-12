"""Cloud LLM backend — fallback when local confidence is low.

Calls the Anthropic Claude API (or any OpenAI-compatible endpoint).
PII scrubbing via presidio is applied before every cloud request.
"""
from __future__ import annotations

import asyncio
import logging
from typing import AsyncIterator, Iterator

from .base import BackendInfo, GenerationConfig, LLMBackend

log = logging.getLogger(__name__)

_DEFAULT_MODEL = "claude-haiku-4-5-20251001"
_DEFAULT_BASE_URL = "https://api.anthropic.com"


class CloudAPIBackend(LLMBackend):
    """HTTP-based backend for cloud LLM APIs.

    Requires the ``cloud`` optional dependency group:
        pip install deepsearch[cloud]
    """

    def __init__(self) -> None:
        self._client = None
        self._model_id = _DEFAULT_MODEL
        self._scrubber = None  # presidio AnalyzerEngine, optional
        self._info = BackendInfo(
            name="cloud",
            model_id=_DEFAULT_MODEL,
            device="cloud",
            loaded=False,
        )

    # ------------------------------------------------------------------ #
    # Lifecycle                                                            #
    # ------------------------------------------------------------------ #

    def load(self, model_path: str, **kwargs) -> None:
        """``model_path`` is used as the model ID for cloud backends."""
        import httpx  # type: ignore[import]

        self._model_id = model_path or _DEFAULT_MODEL
        api_key: str = kwargs.get("api_key", "")
        base_url: str = kwargs.get("base_url", _DEFAULT_BASE_URL)

        self._client = httpx.Client(
            base_url=base_url,
            headers={
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            timeout=60.0,
        )
        self._async_client = httpx.AsyncClient(
            base_url=base_url,
            headers={
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            timeout=60.0,
        )

        # Try to load PII scrubber (optional)
        try:
            from presidio_analyzer import AnalyzerEngine  # type: ignore[import]
            from presidio_anonymizer import AnonymizerEngine  # type: ignore[import]

            self._scrubber = (AnalyzerEngine(), AnonymizerEngine())
            log.info("PII scrubber (presidio) loaded for cloud backend")
        except ImportError:
            log.warning("presidio not installed — PII will NOT be scrubbed before cloud calls")

        self._info = BackendInfo(
            name="cloud",
            model_id=self._model_id,
            device="cloud",
            loaded=True,
        )
        log.info("CloudAPIBackend ready — model: %s", self._model_id)

    def unload(self) -> None:
        if self._client:
            self._client.close()
        self._info.loaded = False

    # ------------------------------------------------------------------ #
    # PII scrubbing                                                        #
    # ------------------------------------------------------------------ #

    def _scrub(self, text: str) -> str:
        if self._scrubber is None:
            return text
        analyzer, anonymizer = self._scrubber
        results = analyzer.analyze(text=text, language="en")
        return anonymizer.anonymize(text=text, analyzer_results=results).text

    # ------------------------------------------------------------------ #
    # Inference                                                            #
    # ------------------------------------------------------------------ #

    def generate(self, prompt: str, config: GenerationConfig) -> str:
        self._require_loaded()
        clean = self._scrub(prompt)
        payload = {
            "model": self._model_id,
            "max_tokens": config.max_tokens,
            "temperature": config.temperature,
            "messages": [{"role": "user", "content": clean}],
        }
        resp = self._client.post("/v1/messages", json=payload)
        resp.raise_for_status()
        return resp.json()["content"][0]["text"]

    def stream(self, prompt: str, config: GenerationConfig) -> Iterator[str]:
        self._require_loaded()
        clean = self._scrub(prompt)
        payload = {
            "model": self._model_id,
            "max_tokens": config.max_tokens,
            "temperature": config.temperature,
            "stream": True,
            "messages": [{"role": "user", "content": clean}],
        }
        with self._client.stream("POST", "/v1/messages", json=payload) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                if line.startswith("data: "):
                    import json
                    data = line[6:]
                    if data == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data)
                        if chunk.get("type") == "content_block_delta":
                            yield chunk["delta"].get("text", "")
                    except Exception:
                        pass

    async def agenerate(self, prompt: str, config: GenerationConfig) -> str:
        self._require_loaded()
        clean = self._scrub(prompt)
        payload = {
            "model": self._model_id,
            "max_tokens": config.max_tokens,
            "temperature": config.temperature,
            "messages": [{"role": "user", "content": clean}],
        }
        resp = await self._async_client.post("/v1/messages", json=payload)
        resp.raise_for_status()
        return resp.json()["content"][0]["text"]

    async def astream(
        self, prompt: str, config: GenerationConfig
    ) -> AsyncIterator[str]:
        self._require_loaded()
        clean = self._scrub(prompt)
        payload = {
            "model": self._model_id,
            "max_tokens": config.max_tokens,
            "temperature": config.temperature,
            "stream": True,
            "messages": [{"role": "user", "content": clean}],
        }
        async with self._async_client.stream("POST", "/v1/messages", json=payload) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if line.startswith("data: "):
                    import json
                    data = line[6:]
                    if data == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data)
                        if chunk.get("type") == "content_block_delta":
                            yield chunk["delta"].get("text", "")
                    except Exception:
                        pass

    @property
    def info(self) -> BackendInfo:
        return self._info

    def _require_loaded(self) -> None:
        if not self._info.loaded:
            raise RuntimeError("CloudAPIBackend not loaded. Call load() first.")
