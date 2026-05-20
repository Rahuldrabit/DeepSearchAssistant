"""Configuration loader — merges default.yaml with user overrides."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

# Resolve config directory relative to this file
_REPO_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_CONFIG = _REPO_ROOT / "config" / "default.yaml"
_MODELS_CONFIG = _REPO_ROOT / "config" / "models.yaml"


# ------------------------------------------------------------------ #
# Sub-schemas                                                          #
# ------------------------------------------------------------------ #

class AppConfig(BaseModel):
    name: str = "DeepSearch Assistant"
    version: str = "0.1.0"
    data_dir: str = "data"
    log_level: str = "INFO"
    log_file: str = "data/deepsearch.log"


class ModelsConfig(BaseModel):
    llm_device: str = "auto"
    n_gpu_layers: int = -1
    context_length: int = 4096
    n_threads: int = 0


class RetrievalConfig(BaseModel):
    top_k_retrieval: int = 50
    top_k_rerank: int = 5
    rrf_k: int = 60
    min_score: float = 0.0
    search_mode: str = "fast"  # fast | deep | cloud


class IndexingConfig(BaseModel):
    chunk_size: int = 512
    chunk_overlap: int = 50
    batch_size: int = 32
    supported_extensions: list[str] = Field(
        default_factory=lambda: [
            ".pdf", ".docx", ".txt", ".md", ".py", ".json",
            ".csv", ".html", ".pptx", ".xlsx",
            ".png", ".jpg", ".jpeg", ".mp3", ".wav", ".mp4",
        ]
    )


class CloudConfig(BaseModel):
    enabled: bool = False
    provider: str = "anthropic"
    model: str = "claude-haiku-4-5-20251001"
    api_key_env: str = "ANTHROPIC_API_KEY"
    confidence_threshold: float = 0.65
    max_tokens: int = 1024


class UIConfig(BaseModel):
    theme: str = "dark"
    font_size: int = 13
    window_width: int = 1280
    window_height: int = 800
    streaming: bool = True


class CacheConfig(BaseModel):
    embedding_max_size: int = 10000
    query_max_size: int = 500
    query_ttl_seconds: int = 600
    answer_max_size: int = 200
    answer_ttl_seconds: int = 1800


class ResourceConfig(BaseModel):
    max_memory_gb: float = 12.0
    llm_slot_gb: float = 5.0
    small_llm_slot_gb: float = 2.5
    embedding_slot_gb: float = 0.5
    whisper_slot_gb: float = 1.5


# ------------------------------------------------------------------ #
# Top-level config                                                     #
# ------------------------------------------------------------------ #

class Settings(BaseModel):
    app: AppConfig = Field(default_factory=AppConfig)
    models: ModelsConfig = Field(default_factory=ModelsConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    indexing: IndexingConfig = Field(default_factory=IndexingConfig)
    cloud: CloudConfig = Field(default_factory=CloudConfig)
    ui: UIConfig = Field(default_factory=UIConfig)
    cache: CacheConfig = Field(default_factory=CacheConfig)
    resource: ResourceConfig = Field(default_factory=ResourceConfig)

    @property
    def data_path(self) -> Path:
        p = Path(self.app.data_dir)
        return p if p.is_absolute() else (_REPO_ROOT / p)

    @property
    def qdrant_path(self) -> Path:
        return self.data_path / "qdrant"

    @property
    def db_path(self) -> Path:
        return self.data_path / "metadata.db"

    @property
    def models_path(self) -> Path:
        return _REPO_ROOT / "models"

    @property
    def cloud_api_key(self) -> str:
        return os.getenv(self.cloud.api_key_env, "")


# ------------------------------------------------------------------ #
# Loader                                                               #
# ------------------------------------------------------------------ #

def _deep_merge(base: dict, override: dict) -> dict:
    result = base.copy()
    for key, val in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(val, dict):
            result[key] = _deep_merge(result[key], val)
        else:
            result[key] = val
    return result


def load_config(user_config_path: str | Path | None = None) -> Settings:
    """Load default.yaml, then optionally merge a user config on top."""
    raw: dict[str, Any] = {}

    if _DEFAULT_CONFIG.exists():
        with open(_DEFAULT_CONFIG) as f:
            raw = yaml.safe_load(f) or {}

    if user_config_path:
        p = Path(user_config_path)
        if p.exists():
            with open(p) as f:
                user_raw = yaml.safe_load(f) or {}
            raw = _deep_merge(raw, user_raw)

    return Settings(**raw)


# Singleton — imported by other modules
_settings: Settings | None = None


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = load_config()
    return _settings


def reset_settings(cfg: Settings | None = None) -> None:
    """Replace singleton (used in tests)."""
    global _settings
    _settings = cfg
