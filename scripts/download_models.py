#!/usr/bin/env python3
"""Download all required GGUF models from Hugging Face Hub.

Usage:
    python scripts/download_models.py [--stage 1|2|3]

Stage 1 = MVP (LLM main + LLM small + LLM fallback)
Stage 2 = adds reranker
Stage 3 = adds VLM + Whisper
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import yaml

_REPO_ROOT = Path(__file__).resolve().parent.parent
_MODELS_DIR = _REPO_ROOT / "models"
_MODELS_YAML = _REPO_ROOT / "config" / "models.yaml"


def load_models_config() -> dict:
    with open(_MODELS_YAML) as f:
        data = yaml.safe_load(f)
    return data.get("models", data)


def download_gguf(repo_id: str, filename: str, dest: Path) -> None:
    """Download a single GGUF file using huggingface_hub."""
    try:
        from huggingface_hub import hf_hub_download  # type: ignore[import]
    except ImportError:
        print("ERROR: huggingface_hub not installed. Run: pip install huggingface-hub")
        sys.exit(1)

    target = dest / filename
    if target.exists():
        print(f"  Already exists: {filename}")
        return

    print(f"  Downloading {filename} from {repo_id}…")
    hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        local_dir=str(dest),
        local_dir_use_symlinks=False,
    )
    print(f"  Saved: {target}")


def download_sentence_transformer(model_id: str) -> None:
    """Pre-download a sentence-transformers model."""
    try:
        from sentence_transformers import SentenceTransformer  # type: ignore[import]
    except ImportError:
        print("WARNING: sentence-transformers not installed — skipping.")
        return

    print(f"  Downloading sentence-transformers model: {model_id}…")
    SentenceTransformer(model_id)
    print(f"  Cached: {model_id}")


def download_whisper(model_id: str) -> None:
    """Pre-download a faster-whisper model."""
    try:
        from faster_whisper import WhisperModel  # type: ignore[import]
    except ImportError:
        print("WARNING: faster-whisper not installed — skipping Whisper download.")
        return
    print(f"  Downloading Whisper model: {model_id}…")
    WhisperModel(model_id, device="cpu", compute_type="int8")
    print(f"  Cached: {model_id}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download DeepSearch models")
    parser.add_argument("--stage", type=int, default=1, choices=[1, 2, 3],
                        help="Download models up to this stage (default: 1)")
    args = parser.parse_args()

    _MODELS_DIR.mkdir(parents=True, exist_ok=True)
    models = load_models_config()

    print(f"\nDeepSearch Model Downloader — Stage {args.stage}\n")
    print(f"Models directory: {_MODELS_DIR}\n")

    for key, meta in models.items():
        stage = meta.get("stage", 99)
        if stage > args.stage:
            continue

        backend = meta.get("backend", "llamacpp")
        print(f"[{key}] {meta.get('model_id', key)}")

        if backend == "llamacpp":
            repo_id = meta.get("repo_id", "")
            filename = meta.get("filename", "")
            if repo_id and filename:
                download_gguf(repo_id, filename, _MODELS_DIR)
            else:
                print(f"  SKIP: missing repo_id or filename for {key}")

        elif backend == "sentence_transformers":
            model_id = meta.get("model_id", "")
            if model_id:
                download_sentence_transformer(model_id)

        elif backend == "ctranslate2":  # Whisper
            model_id = meta.get("model_id", "")
            if model_id:
                download_whisper(model_id)
        else:
            print(f"  SKIP: unknown backend '{backend}' for {key}")

    print("\nAll stage-{} models downloaded.".format(args.stage))


if __name__ == "__main__":
    main()
