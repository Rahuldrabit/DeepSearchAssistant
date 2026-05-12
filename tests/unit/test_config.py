"""Tests for configuration loading."""
import pytest
from pathlib import Path


def test_default_settings_load():
    """Settings load from default.yaml without error."""
    from deepsearch.core.config import load_config
    cfg = load_config()
    assert cfg.app.name == "DeepSearch Assistant"
    assert cfg.retrieval.rrf_k == 60
    assert cfg.indexing.chunk_size == 512


def test_settings_data_paths():
    from deepsearch.core.config import load_config
    cfg = load_config()
    assert cfg.qdrant_path == cfg.data_path / "qdrant"
    assert cfg.db_path == cfg.data_path / "metadata.db"


def test_settings_override(tmp_path):
    """User config deep-merges on top of defaults."""
    import yaml
    override = tmp_path / "user.yaml"
    override.write_text(yaml.dump({"retrieval": {"rrf_k": 80, "top_k_rerank": 10}}))

    from deepsearch.core.config import load_config
    cfg = load_config(override)
    assert cfg.retrieval.rrf_k == 80
    assert cfg.retrieval.top_k_rerank == 10
    # Default values not in override remain unchanged
    assert cfg.retrieval.top_k_retrieval == 50


def test_cloud_disabled_by_default():
    from deepsearch.core.config import load_config
    cfg = load_config()
    assert cfg.cloud.enabled is False


def test_supported_extensions_include_common():
    from deepsearch.core.config import load_config
    cfg = load_config()
    exts = cfg.indexing.supported_extensions
    assert ".pdf" in exts
    assert ".docx" in exts
    assert ".txt" in exts
