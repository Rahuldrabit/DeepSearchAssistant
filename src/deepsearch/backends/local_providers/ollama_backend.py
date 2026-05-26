"""Ollama local backend.

Connects to a running Ollama server (https://ollama.com).

Quick start
-----------
1. Install Ollama and pull a model::

       ollama pull llama3.2:3b

2. The server starts automatically, or run it manually::

       ollama serve

3. Set config (config/default.yaml)::

       local_providers:
         ollama:
           enabled: true
           base_url: "http://localhost:11434"
           model: "llama3.2:3b"
"""
from __future__ import annotations

import logging

from .base_openai_compat import OpenAICompatBackend

log = logging.getLogger(__name__)

_DEFAULT_BASE_URL = "http://localhost:11434"


class OllamaBackend(OpenAICompatBackend):
    """LLM backend that delegates to a local Ollama server.

    Ollama exposes an OpenAI-compatible ``/v1/chat/completions`` endpoint
    alongside its own ``/api/`` routes.  We use the OpenAI-compatible route
    for inference and ``/api/tags`` for model listing / health checks.
    """

    _PROVIDER_NAME = "ollama"
    _DEFAULT_BASE_URL = _DEFAULT_BASE_URL

    # ------------------------------------------------------------------ #
    # Health check                                                        #
    # ------------------------------------------------------------------ #

    def _health_check(self) -> None:
        try:
            resp = self._client.get("/api/tags")
            resp.raise_for_status()
        except Exception as exc:
            raise RuntimeError(
                f"Cannot reach Ollama server at {self._base_url}.\n"
                "Make sure Ollama is running:  ollama serve\n"
                f"Original error: {exc}"
            ) from exc

        available = [m["name"] for m in resp.json().get("models", [])]
        log.info("Ollama server reachable.  Available models: %s", available)

        if self._model_id and self._model_id not in available:
            log.warning(
                "Model '%s' not found in Ollama (available: %s). "
                "Pull it first:  ollama pull %s",
                self._model_id,
                available,
                self._model_id,
            )

    # ------------------------------------------------------------------ #
    # Ollama-specific helpers                                             #
    # ------------------------------------------------------------------ #

    def list_models(self) -> list[str]:
        """Return model names from this Ollama instance (``/api/tags``)."""
        resp = self._client.get("/api/tags")
        resp.raise_for_status()
        return [m["name"] for m in resp.json().get("models", [])]

    def pull_model(self, model_name: str, *, timeout: float = 600.0) -> None:
        """Pull *model_name* into Ollama (blocking, equivalent to ``ollama pull``).

        Args:
            model_name: E.g. ``"llama3.2:3b"`` or ``"mistral:7b-instruct"``.
            timeout:    Seconds to wait for the download (default 600 s).
        """
        log.info("Pulling Ollama model '%s' — this may take a while…", model_name)
        resp = self._client.post(
            "/api/pull",
            json={"name": model_name},
            timeout=timeout,
        )
        resp.raise_for_status()
        log.info("Model '%s' pulled successfully.", model_name)

    def delete_model(self, model_name: str) -> None:
        """Delete *model_name* from this Ollama instance."""
        resp = self._client.delete("/api/delete", json={"name": model_name})
        resp.raise_for_status()
        log.info("Ollama model '%s' deleted.", model_name)
