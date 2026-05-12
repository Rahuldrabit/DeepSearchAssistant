"""Mutex-enforced memory slot manager.

Memory layout (16 GB machine):
  Slot A (4-5 GB) — LLM main / VLM        — mutex-protected
  Slot B (2-3 GB) — LLM small             — always loaded
  Slot C (0.5 GB) — Embedding + reranker  — always loaded
  Slot D (1.5 GB) — Whisper               — on-demand

Only one large model may occupy Slot A at a time.
"""
from __future__ import annotations

import logging
import threading
from enum import Enum
from typing import Callable

from .exceptions import ResourceBudgetExceeded

log = logging.getLogger(__name__)


class Slot(str, Enum):
    A = "A"  # Large model (LLM main / VLM) — mutex
    B = "B"  # Small LLM — always loaded
    C = "C"  # Embeddings — always loaded
    D = "D"  # Whisper — on-demand


class ResourceManager:
    """Thread-safe model slot tracker."""

    def __init__(self) -> None:
        self._slot_mutex = threading.Lock()  # guards Slot A
        self._slot_a_owner: str | None = None  # model_id currently in Slot A
        self._loaded: dict[Slot, str | None] = {s: None for s in Slot}
        self._unload_callbacks: dict[Slot, Callable[[], None]] = {}

    # ------------------------------------------------------------------ #
    # Slot A (mutex-protected large model)                                 #
    # ------------------------------------------------------------------ #

    def acquire_slot_a(self, model_id: str, unload_fn: Callable[[], None]) -> None:
        """Swap into Slot A, evicting whatever was there before."""
        with self._slot_mutex:
            current = self._loaded[Slot.A]
            if current and current != model_id:
                log.info("Evicting %s from Slot A to make room for %s", current, model_id)
                cb = self._unload_callbacks.pop(Slot.A, None)
                if cb:
                    cb()
                self._loaded[Slot.A] = None
                self._slot_a_owner = None

            if self._loaded[Slot.A] is None:
                self._loaded[Slot.A] = model_id
                self._slot_a_owner = model_id
                self._unload_callbacks[Slot.A] = unload_fn
                log.info("Slot A acquired by %s", model_id)
            # else: already loaded (same model_id), no-op

    def release_slot_a(self, model_id: str) -> None:
        with self._slot_mutex:
            if self._loaded[Slot.A] == model_id:
                cb = self._unload_callbacks.pop(Slot.A, None)
                if cb:
                    cb()
                self._loaded[Slot.A] = None
                self._slot_a_owner = None
                log.info("Slot A released by %s", model_id)

    @property
    def slot_a_owner(self) -> str | None:
        return self._slot_a_owner

    # ------------------------------------------------------------------ #
    # Other slots (no mutex needed)                                        #
    # ------------------------------------------------------------------ #

    def mark_loaded(self, slot: Slot, model_id: str) -> None:
        if slot == Slot.A:
            raise ValueError("Use acquire_slot_a() for Slot A")
        self._loaded[slot] = model_id

    def mark_unloaded(self, slot: Slot) -> None:
        self._loaded[slot] = None

    def is_loaded(self, slot: Slot) -> bool:
        return self._loaded[slot] is not None

    def loaded_model(self, slot: Slot) -> str | None:
        return self._loaded[slot]

    # ------------------------------------------------------------------ #
    # Summary                                                              #
    # ------------------------------------------------------------------ #

    def status(self) -> dict[str, str | None]:
        return {s.value: self._loaded[s] for s in Slot}


# Singleton
_resource_manager: ResourceManager | None = None


def get_resource_manager() -> ResourceManager:
    global _resource_manager
    if _resource_manager is None:
        _resource_manager = ResourceManager()
    return _resource_manager
