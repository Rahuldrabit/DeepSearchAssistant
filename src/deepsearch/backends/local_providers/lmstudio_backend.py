"""LM Studio local backend.

Connects to LM Studio's built-in local server (https://lmstudio.ai).

Quick start
-----------
1. Open LM Studio and download a model from the Discover tab.
2. Go to the **Local Server** tab (left sidebar).
3. Select and load the model, then click **Start Server**.
4. Set config (config/default.yaml)::

       local_providers:
         lmstudio:
           enabled: true
           base_url: "http://localhost:1234"
           model: "meta-llama-3.1-8b-instruct"

   The ``model`` value must match the model identifier shown in LM Studio's
   server status bar (usually the file stem of the loaded GGUF).
"""
from __future__ import annotations

import logging

from .base_openai_compat import OpenAICompatBackend

log = logging.getLogger(__name__)

_DEFAULT_BASE_URL = "http://localhost:1234"


class LMStudioBackend(OpenAICompatBackend):
    """LLM backend that delegates to LM Studio's local OpenAI-compatible server.

    LM Studio fully implements the OpenAI ``/v1/chat/completions`` and
    ``/v1/models`` endpoints, so all inference is handled by the shared
    :class:`~.base_openai_compat.OpenAICompatBackend` base.  This class only
    adds an LM-Studio-aware health check and a clearer error message.
    """

    _PROVIDER_NAME = "lmstudio"
    _DEFAULT_BASE_URL = _DEFAULT_BASE_URL

    # ------------------------------------------------------------------ #
    # Health check                                                        #
    # ------------------------------------------------------------------ #

    def _health_check(self) -> None:
        try:
            resp = self._client.get("/v1/models")
            resp.raise_for_status()
        except Exception as exc:
            raise RuntimeError(
                f"Cannot reach LM Studio server at {self._base_url}.\n"
                "Make sure the Local Server is running inside LM Studio.\n"
                f"Original error: {exc}"
            ) from exc

        loaded = [m["id"] for m in resp.json().get("data", [])]
        log.info("LM Studio server reachable.  Loaded models: %s", loaded)

        if self._model_id and not any(self._model_id in m for m in loaded):
            log.warning(
                "Model '%s' does not appear to be loaded in LM Studio "
                "(found: %s).  Load it via the Local Server tab first.",
                self._model_id,
                loaded,
            )
