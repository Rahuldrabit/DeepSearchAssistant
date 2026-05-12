"""Tests for exception hierarchy."""
from deepsearch.core.exceptions import (
    DeepSearchError,
    ModelNotFoundError,
    ModelLoadError,
    BackendNotAvailableError,
    ResourceBudgetExceeded,
    IndexingError,
    RetrievalError,
    GenerationError,
    ConfigurationError,
    CloudDisabledError,
)


def test_all_inherit_from_base():
    for exc_cls in [
        ModelNotFoundError, ModelLoadError, BackendNotAvailableError,
        ResourceBudgetExceeded, IndexingError, RetrievalError,
        GenerationError, ConfigurationError, CloudDisabledError,
    ]:
        assert issubclass(exc_cls, DeepSearchError)
        assert issubclass(exc_cls, Exception)


def test_can_raise_and_catch_as_base():
    with pytest.raises(DeepSearchError):
        raise IndexingError("test error")


def test_message_preserved():
    exc = ModelNotFoundError("model.gguf not found")
    assert "model.gguf" in str(exc)


import pytest
