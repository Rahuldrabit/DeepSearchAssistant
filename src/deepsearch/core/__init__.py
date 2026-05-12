"""Core engine: config, device detection, resource management, model loading."""
from .config import Settings, get_settings, load_config
from .device_manager import Device, DeviceManager, get_device_manager
from .exceptions import (
    BackendNotAvailableError,
    CloudDisabledError,
    ConfigurationError,
    DeepSearchError,
    GenerationError,
    IndexingError,
    ModelLoadError,
    ModelNotFoundError,
    ResourceBudgetExceeded,
    RetrievalError,
)
from .model_manager import ModelManager, get_model_manager
from .resource_manager import ResourceManager, Slot, get_resource_manager
from .language_manager import LanguageManager

__all__ = [
    "Settings", "get_settings", "load_config",
    "Device", "DeviceManager", "get_device_manager",
    "DeepSearchError", "ModelNotFoundError", "ModelLoadError",
    "BackendNotAvailableError", "ResourceBudgetExceeded",
    "IndexingError", "RetrievalError", "GenerationError",
    "ConfigurationError", "CloudDisabledError",
    "ModelManager", "get_model_manager",
    "ResourceManager", "Slot", "get_resource_manager",
    "LanguageManager",
]
