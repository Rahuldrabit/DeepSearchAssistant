"""Local provider backends — Ollama and LM Studio.

Both providers expose an OpenAI-compatible REST API, so they share a common
base class (:class:`~.base_openai_compat.OpenAICompatBackend`).

Usage
-----
The simplest way to get a ready-to-use backend is the factory function::

    from deepsearch.backends.local_providers import create_local_provider

    backend = create_local_provider("ollama", "llama3.2:3b")
    reply = backend.generate("Summarise this document.", GenerationConfig())

Or instantiate directly::

    from deepsearch.backends.local_providers import OllamaBackend

    backend = OllamaBackend()
    backend.load("llama3.2:3b", base_url="http://localhost:11434")
"""
from __future__ import annotations

from .lmstudio_backend import LMStudioBackend
from .ollama_backend import OllamaBackend

__all__ = [
    "OllamaBackend",
    "LMStudioBackend",
    "create_local_provider",
]

_REGISTRY: dict[str, type] = {
    "ollama": OllamaBackend,
    "lmstudio": LMStudioBackend,
}


def create_local_provider(provider: str, model: str, **kwargs):
    """Factory — create and load a local provider backend.

    Args:
        provider: ``"ollama"`` or ``"lmstudio"``.
        model:    Model name/ID as the server knows it.
        **kwargs: Forwarded to ``backend.load()``
                  (e.g. ``base_url``, ``timeout``).

    Returns:
        A loaded :class:`~deepsearch.backends.base.LLMBackend` instance.

    Raises:
        ValueError:   Unknown provider name.
        RuntimeError: Server not reachable or model not found.

    Example::

        backend = create_local_provider(
            "ollama", "mistral:7b-instruct",
            base_url="http://192.168.1.10:11434",
        )
    """
    if provider not in _REGISTRY:
        raise ValueError(
            f"Unknown local provider '{provider}'. "
            f"Available: {list(_REGISTRY)}"
        )
    backend = _REGISTRY[provider]()
    backend.load(model, **kwargs)
    return backend
