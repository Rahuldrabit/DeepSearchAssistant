# DeepSearch Assistant

A multimodal RAG (Retrieval-Augmented Generation) desktop application for intelligent personal document search. Built with llama.cpp for universal hardware support and PyQt6 for a native desktop experience.

**Find anything in your personal files — documents, images, videos, audio — using natural language.**

## Features

- **Semantic Search**: Ask questions in natural language, get accurate answers with source citations
- **Multimodal**: Index and search across DOCX, PDF, PPTX, XLSX, TXT, Markdown, images, video, and audio
- **Three Search Modes**:
  - **Fast Search** (<3s): Direct hybrid retrieval + LLM answer
  - **Deep Search** (5-15s): Query rewriting + multi-round retrieval + reranking
  - **Cloud Deep** (15-40s): Local retrieval + cloud LLM synthesis for complex queries
- **Universal Hardware**: Runs on any PC with CPU. GPU acceleration optional (NVIDIA CUDA, Intel via OpenVINO)
- **Privacy-First**: All data stays on your device. Cloud mode only sends PII-scrubbed summaries
- **Hybrid Retrieval**: Dense vectors + BM25 sparse search + Reciprocal Rank Fusion
- **Auto-Indexing**: File watcher monitors folders and re-indexes changed files automatically

## Architecture Overview

```
User Files (.docx, .pdf, .jpg, .mp4, ...)
    |
    v
[Ingestion Pipeline] -- parsers per file type
    |                    (PyMuPDF, python-docx, PaddleOCR, faster-whisper)
    v
[Chunker] -- hierarchical chunking (1024 -> 256 -> 64 tokens)
    |
    v
[Embedding] -- sentence-transformers (all-MiniLM-L6-v2)
    |
    v
[Qdrant Vector Store] -- dense + BM25 sparse + metadata
    |
    v
[Query Pipeline] -- hybrid search -> reranker -> context builder
    |
    v
[LLM Generation] -- llama.cpp (Qwen2.5-7B GGUF) / OpenVINO (optional)
    |
    v
[PyQt6 Desktop UI] -- streaming answers + source citations
```

## System Requirements

### Minimum
- **OS**: Windows 10/11, Linux (Ubuntu 20.04+), macOS 12+
- **CPU**: Any x86-64 processor (Intel or AMD)
- **RAM**: 16 GB
- **Storage**: 10 GB free (for models + index)
- **Python**: 3.10+

### Recommended
- **RAM**: 32 GB
- **GPU**: NVIDIA GPU with 6+ GB VRAM (CUDA) or Intel Arc/iGPU (OpenVINO)
- **Storage**: SSD for fast index access

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/your-username/DeepSearchAssistant.git
cd DeepSearchAssistant
```

### 2. Create virtual environment

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/macOS
source .venv/bin/activate
```

### 3. Install dependencies

```bash
# CPU-only (works everywhere)
pip install -e .

# With NVIDIA GPU support
pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu124
pip install -e .

# With Intel OpenVINO support (optional)
pip install -e ".[openvino]"
```

### 4. Download models

```bash
python scripts/download_models.py
```

This downloads the required models (~5 GB for Stage 1):
- Qwen2.5-7B-Instruct Q4_K_M (4.4 GB) — main LLM
- all-MiniLM-L6-v2 (22 MB) — embedding model

### 5. Initialize vector store

```bash
python scripts/setup_qdrant.py
```

### 6. Launch the application

```bash
python -m deepsearch
# or
python src/deepsearch/app.py
```

## Quick Start

1. **Launch** the application
2. **Drag and drop** a folder or files into the file browser panel
3. **Wait** for indexing to complete (progress bar shows status)
4. **Ask a question** in the search bar — e.g., "What was the budget for Project Alpha?"
5. **View** the streamed answer with source citations in the chat panel

## Model Stack

| Role | Model | Format | Size | Backend |
|------|-------|--------|------|---------|
| Main LLM | Qwen2.5-7B-Instruct | Q4_K_M GGUF | 4.4 GB | llama-cpp-python |
| Small LLM | Phi-3.5-mini-instruct | Q4_K_M GGUF | 2.2 GB | llama-cpp-python |
| Embedding | all-MiniLM-L6-v2 | PyTorch | 22 MB | sentence-transformers |
| Reranker | ms-marco-MiniLM-L-6-v2 | PyTorch | 22 MB | sentence-transformers |
| VLM | LLaVA-v1.6-mistral-7B | Q4_K_M GGUF | 4.1 GB | llama-cpp-python |
| ASR | faster-whisper-medium | CTranslate2 | 1.5 GB | faster-whisper |
| OCR | PaddleOCR PP-OCRv4 | PaddlePaddle | 150 MB | PaddleOCR |

## Search Modes

### Fast Search
Direct query against hybrid index. Best for simple factual questions.
```
"What is the project deadline?" → <3 seconds
```

### Deep Search
LLM rewrites the query for clarity, performs multi-round retrieval with reranking.
```
"Compare all notes about React vs Vue from 2023" → 5-15 seconds
```

### Cloud Deep (Optional)
Local retrieval + PII scrubbing + cloud LLM for complex synthesis. Requires API key.
```
"Analyze the relationship between all Q3 budget items and the strategic plan" → 15-40 seconds
```

## Configuration

Edit `config/default.yaml` to customize:

```yaml
models:
  llm_device: "auto"        # auto, cpu, cuda, openvino
  n_gpu_layers: -1           # -1 = all layers on GPU, 0 = CPU only
  context_length: 4096       # LLM context window

retrieval:
  top_k_retrieval: 50        # candidates from hybrid search
  top_k_rerank: 5            # final chunks after reranking
  rrf_k: 60                  # RRF fusion constant

indexing:
  watch_folders: []          # auto-index these folders
  chunk_size: 512            # tokens per chunk
  chunk_overlap: 50          # overlap between chunks

cloud:
  enabled: false
  provider: "openai"         # openai, anthropic
  api_key: ""                # set via env var DEEPSEARCH_CLOUD_API_KEY
  confidence_threshold: 0.65 # below this, escalate to cloud
```

## Project Structure

```
DeepSearchAssistant/
├── config/              # YAML configuration files
├── docs/                # Architecture and implementation docs
├── scripts/             # Model download and setup utilities
├── src/deepsearch/      # Main application source
│   ├── backends/        # LLM backend abstraction (llama.cpp, OpenVINO)
│   ├── core/            # Config, resource management, device detection
│   ├── ingestion/       # File parsers, chunking, embedding pipeline
│   ├── retrieval/       # Hybrid search, reranking, query routing
│   ├── generation/      # LLM pipeline, cloud fallback, confidence scoring
│   ├── storage/         # Qdrant vectors, SQLite metadata, caching
│   └── ui/              # PyQt6 desktop interface
├── tests/               # 4-layer test suite
├── models/              # Downloaded model files (gitignored)
└── data/                # Vector store and database (gitignored)
```

## Development

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run unit tests
pytest -m unit

# Run integration tests (requires models)
pytest -m integration

# Run all tests
pytest

# Lint
ruff check src/
```

## Hardware Acceleration

### NVIDIA CUDA
Install llama-cpp-python with CUDA support. The app auto-detects CUDA and offloads LLM layers to GPU.

### Intel OpenVINO (Optional)
```bash
pip install -e ".[openvino]"
```
When Intel GPU/NPU is detected and OpenVINO is installed, the app automatically uses OpenVINO for accelerated inference.

### CPU Only
Works out of the box on any x86-64 CPU. Expect ~5-10 tokens/second for LLM generation.

## License

MIT License

## Acknowledgments

- [llama.cpp](https://github.com/ggerganov/llama.cpp) — Universal LLM inference
- [Qdrant](https://qdrant.tech/) — Vector search engine
- [sentence-transformers](https://www.sbert.net/) — Text embeddings
- [OpenVINO](https://github.com/openvinotoolkit/openvino) — Intel hardware acceleration
- [faster-whisper](https://github.com/SYSTRAN/faster-whisper) — Audio transcription
- [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) — Optical character recognition
