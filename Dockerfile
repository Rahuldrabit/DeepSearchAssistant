# ============================================================
# DeepSearch Assistant — Multi-stage Dockerfile
# ============================================================
# Stage 1 (base)  : Python + system deps
# Stage 2 (test)  : Install lightweight test deps, run pytest
# Stage 3 (app)   : Full install for running the desktop app
#                   (requires X11/Wayland on the host)
# ============================================================

# ---- base -------------------------------------------------------
FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# System packages needed by all stages
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        gcc \
        g++ \
        libgomp1 \
        # SQLite (already in python:slim, but ensure headers)
        libsqlite3-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml ./
COPY config/ ./config/
COPY src/ ./src/

# ---- test -------------------------------------------------------
FROM base AS test

# Install Qt system libs needed for QT_QPA_PLATFORM=offscreen
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        libdbus-1-3 \
        libegl1 \
        libfontconfig1 \
        libxcb1 \
        libx11-6 \
        libxext6 \
        libxrender1 \
    && rm -rf /var/lib/apt/lists/*

# Install ONLY what unit tests actually need.
# sentence-transformers pulls torch (530 MB) — skip it.
# Unit tests mock EmbeddingBackend, so torch is not required.
RUN pip install \
        # Lightweight core
        "qdrant-client>=1.12" \
        "PyMuPDF>=1.24" \
        "python-docx>=1.1" \
        "PyQt6>=6.6" \
        "pydantic>=2.5" \
        "pydantic-settings>=2.0" \
        "pyyaml>=6.0" \
        "numpy>=1.26" \
        # Dev / test
        "pytest>=8.0" \
        "pytest-qt>=4.3" \
        "pytest-asyncio>=0.23"

COPY tests/ ./tests/

ENV PYTHONPATH=/app/src \
    QT_QPA_PLATFORM=offscreen \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1

# Run unit tests by default
CMD ["pytest", "tests/unit/", "-v", "--tb=short", "-q"]

# ---- app --------------------------------------------------------
FROM base AS app

# Extra system libs for Qt display (requires host X11 or Wayland)
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        libdbus-1-3 \
        libegl1 \
        libfontconfig1 \
        libxcb1 \
        libx11-6 \
        libxext6 \
        libxrender1 \
        libxcb-xinerama0 \
        libxcb-randr0 \
        libxcb-shape0 \
        libxcb-xfixes0 \
        libxkbcommon-x11-0 \
    && rm -rf /var/lib/apt/lists/*

# Install full deps including llama-cpp-python (CPU build)
RUN pip install \
        --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu \
        "llama-cpp-python>=0.3.0"

RUN pip install \
        "sentence-transformers>=3.0" \
        "qdrant-client>=1.12" \
        "PyMuPDF>=1.24" \
        "python-docx>=1.1" \
        "PyQt6>=6.6" \
        "pydantic>=2.5" \
        "pydantic-settings>=2.0" \
        "pyyaml>=6.0" \
        "huggingface-hub>=0.23" \
        "Pillow>=10.0" \
        "numpy>=1.26"

# Mount models/ and data/ at runtime
VOLUME ["/app/models", "/app/data"]

ENV PYTHONPATH=/app/src

ENTRYPOINT ["python", "-m", "deepsearch.app"]
