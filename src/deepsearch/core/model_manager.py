"""Model manager — orchestrates backend loading with resource tracking.

Reads models.yaml, resolves GGUF paths, delegates to the appropriate
backend, and enforces memory slot constraints via ResourceManager.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import yaml

from ..backends.base import GenerationConfig, LLMBackend
from ..backends.embedding_backend import EmbeddingBackend, RerankerBackend
from ..backends.llamacpp_backend import LlamaCppBackend
from .config import Settings, get_settings
from .device_manager import DeviceManager, get_device_manager
from .exceptions import BackendNotAvailableError, ModelNotFoundError
from .resource_manager import ResourceManager, Slot, get_resource_manager

log = logging.getLogger(__name__)

_MODELS_YAML = Path(__file__).resolve().parents[3] / "config" / "models.yaml"


class ModelManager:
    """Central access point for all model instances.

    Usage:
        mm = ModelManager()
        mm.initialize()           # loads embeddings + small LLM
        mm.ensure_llm_main()      # loads main LLM on first call
        response = mm.llm_main.generate(prompt, cfg)
    """

    def __init__(
        self,
        settings: Optional[Settings] = None,
        device_manager: Optional[DeviceManager] = None,
        resource_manager: Optional[ResourceManager] = None,
    ) -> None:
        self._cfg = settings or get_settings()
        self._dev = device_manager or get_device_manager()
        self._res = resource_manager or get_resource_manager()
        self._models_meta: dict = self._load_models_yaml()

        # Backend instances
        self._embedding: EmbeddingBackend = EmbeddingBackend()
        self._reranker: RerankerBackend = RerankerBackend()
        self._llm_main: LlamaCppBackend | None = None
        self._llm_small: LlamaCppBackend | None = None
        self._vlm: LlamaCppBackend | None = None

    # ------------------------------------------------------------------ #
    # Startup                                                              #
    # ------------------------------------------------------------------ #

    def initialize(self) -> None:
        """Load always-on models (Slot B + C).  Called at app startup."""
        self._load_embeddings()
        self._load_small_llm()

    def _load_embeddings(self) -> None:
        meta = self._models_meta.get("embedding", {})
        model_name = meta.get("model_id", "all-MiniLM-L6-v2")
        self._embedding.load(model_name, device="cpu")
        self._res.mark_loaded(Slot.C, model_name)
        log.info("Embedding model loaded: %s", model_name)

        rerank_meta = self._models_meta.get("reranker", {})
        if rerank_meta:
            rmodel = rerank_meta.get("model_id", "cross-encoder/ms-marco-MiniLM-L-6-v2")
            self._reranker.load(rmodel, device="cpu")
            log.info("Reranker loaded: %s", rmodel)

    def _load_small_llm(self) -> None:
        meta = self._models_meta.get("llm_small", {})
        if not meta:
            return
        path = self._resolve_gguf(meta["filename"])
        if not path.exists():
            log.warning("Small LLM not found at %s — skipping", path)
            return
        backend = LlamaCppBackend()
        backend.load(
            str(path),
            n_gpu_layers=0,  # small LLM on CPU to leave GPU for main
            context_length=self._cfg.models.context_length,
        )
        self._llm_small = backend
        self._res.mark_loaded(Slot.B, meta["filename"])

    # ------------------------------------------------------------------ #
    # On-demand large model (Slot A — mutex)                               #
    # ------------------------------------------------------------------ #

    def ensure_llm_main(self) -> None:
        """Ensure main LLM is in Slot A (evicts VLM if needed)."""
        meta = self._models_meta.get("llm_main", {})
        model_id = meta.get("filename", "")
        if self._res.slot_a_owner == model_id:
            return  # already loaded

        path = self._resolve_gguf(model_id)
        if not path.exists():
            raise ModelNotFoundError(f"Main LLM not found: {path}")

        n_gpu_layers = self._dev.n_gpu_layers_for(meta.get("size_gb", 4.4))
        backend = LlamaCppBackend()
        backend.load(
            str(path),
            n_gpu_layers=n_gpu_layers,
            context_length=self._cfg.models.context_length,
        )

        def _unload() -> None:
            backend.unload()
            self._llm_main = None

        self._res.acquire_slot_a(model_id, _unload)
        self._llm_main = backend

    def ensure_vlm(self) -> None:
        """Ensure VLM (LLaVA) is in Slot A (evicts main LLM if needed)."""
        meta = self._models_meta.get("vlm", {})
        if not meta:
            raise BackendNotAvailableError("VLM not configured in models.yaml")
        model_id = meta.get("filename", "")
        if self._res.slot_a_owner == model_id:
            return

        path = self._resolve_gguf(model_id)
        if not path.exists():
            raise ModelNotFoundError(f"VLM not found: {path}")

        n_gpu_layers = self._dev.n_gpu_layers_for(meta.get("size_gb", 4.1))
        backend = LlamaCppBackend()
        backend.load(str(path), n_gpu_layers=n_gpu_layers)

        def _unload() -> None:
            backend.unload()
            self._vlm = None

        self._res.acquire_slot_a(model_id, _unload)
        self._vlm = backend

    # ------------------------------------------------------------------ #
    # Public accessors                                                     #
    # ------------------------------------------------------------------ #

    @property
    def embedding(self) -> EmbeddingBackend:
        return self._embedding

    @property
    def reranker(self) -> RerankerBackend:
        return self._reranker

    @property
    def llm_main(self) -> LlamaCppBackend:
        if self._llm_main is None:
            self.ensure_llm_main()
        return self._llm_main  # type: ignore[return-value]

    @property
    def llm_small(self) -> LlamaCppBackend | None:
        return self._llm_small

    @property
    def vlm(self) -> LlamaCppBackend:
        if self._vlm is None:
            self.ensure_vlm()
        return self._vlm  # type: ignore[return-value]

    def best_available_llm(self) -> LLMBackend:
        """Return main LLM if loaded, fall back to small LLM."""
        if self._llm_main and self._llm_main.is_loaded:
            return self._llm_main
        if self._llm_small and self._llm_small.is_loaded:
            return self._llm_small
        # Force-load main as last resort
        return self.llm_main

    # ------------------------------------------------------------------ #
    # Helpers                                                              #
    # ------------------------------------------------------------------ #

    def _resolve_gguf(self, filename: str) -> Path:
        models_dir = self._cfg.models_path
        return models_dir / filename

    @staticmethod
    def _load_models_yaml() -> dict:
        if not _MODELS_YAML.exists():
            log.warning("models.yaml not found at %s", _MODELS_YAML)
            return {}
        with open(_MODELS_YAML) as f:
            data = yaml.safe_load(f) or {}
        return data.get("models", data)


# Singleton
_model_manager: ModelManager | None = None


def get_model_manager() -> ModelManager:
    global _model_manager
    if _model_manager is None:
        _model_manager = ModelManager()
    return _model_manager
