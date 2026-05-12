"""Project-wide exception hierarchy."""


class DeepSearchError(Exception):
    """Base exception for all DeepSearch errors."""


class ModelNotFoundError(DeepSearchError):
    """GGUF / model file does not exist on disk."""


class ModelLoadError(DeepSearchError):
    """Backend failed to load a model into memory."""


class BackendNotAvailableError(DeepSearchError):
    """Requested backend (e.g. OpenVINO) is not installed or device absent."""


class ResourceBudgetExceeded(DeepSearchError):
    """Cannot load model because memory slot is occupied."""


class IndexingError(DeepSearchError):
    """Error during document ingestion / indexing."""


class RetrievalError(DeepSearchError):
    """Error during vector search or reranking."""


class GenerationError(DeepSearchError):
    """Error during LLM generation."""


class ConfigurationError(DeepSearchError):
    """Invalid or missing configuration."""


class CloudDisabledError(DeepSearchError):
    """Cloud fallback was triggered but cloud is disabled in config."""
