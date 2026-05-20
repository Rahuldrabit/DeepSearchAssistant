"""Application entry point — wires all components together and launches the UI."""
from __future__ import annotations

import logging
import sys
from pathlib import Path

from PyQt6.QtWidgets import QApplication  # type: ignore[import]

from .backends.embedding_backend import EmbeddingBackend, RerankerBackend
from .core.config import get_settings
from .core.device_manager import get_device_manager
from .core.model_manager import ModelManager
from .generation.llm_pipeline import LLMPipeline
from .ingestion.chunker import HierarchicalChunker
from .ingestion.dispatcher import IngestionDispatcher
from .ingestion.embedder import Embedder
from .retrieval.context_builder import ContextBuilder
from .retrieval.hybrid_search import HybridSearch
from .storage.cache import CacheBundle
from .storage.metadata_db import MetadataDB
from .storage.vector_store import VectorStore
from .ui.main_window import MainWindow


def _setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
        ],
    )


def build_pipeline():
    """Construct the full pipeline graph and return all components."""
    cfg = get_settings()
    dev = get_device_manager()

    # Ensure data directories exist
    cfg.data_path.mkdir(parents=True, exist_ok=True)
    cfg.qdrant_path.mkdir(parents=True, exist_ok=True)

    # Storage
    vector_store = VectorStore(cfg.qdrant_path)
    metadata_db = MetadataDB(cfg.db_path)
    cache = CacheBundle()

    # Embeddings (Slot C — always loaded first)
    embedding_backend = EmbeddingBackend()
    embedding_backend.load("all-MiniLM-L6-v2", device="cpu")

    reranker_backend = RerankerBackend()
    reranker_backend.load("cross-encoder/ms-marco-MiniLM-L-6-v2", device="cpu")

    # Ingestion
    embedder = Embedder(embedding_backend, cache=cache.embeddings, data_dir=cfg.data_path)
    chunker = HierarchicalChunker(
        chunk_size_chars=cfg.indexing.chunk_size * 4,
        overlap_chars=cfg.indexing.chunk_overlap * 4,
    )
    dispatcher = IngestionDispatcher(
        vector_store=vector_store,
        metadata_db=metadata_db,
        embedder=embedder,
        chunker=chunker,
    )

    # Retrieval
    hybrid_search = HybridSearch(
        vector_store=vector_store,
        embedder=embedder,
        top_k_retrieval=cfg.retrieval.top_k_retrieval,
        rrf_k=cfg.retrieval.rrf_k,
    )
    context_builder = ContextBuilder(
        vector_store=vector_store,
        reranker=reranker_backend,
        top_k_rerank=cfg.retrieval.top_k_rerank,
    )

    # Model manager (LLM loading deferred until first query)
    model_manager = ModelManager()

    # Generation pipeline
    pipeline = LLMPipeline(
        hybrid_search=hybrid_search,
        context_builder=context_builder,
        model_manager=model_manager,
        cache=cache,
        settings=cfg,
    )

    return pipeline, dispatcher, metadata_db, model_manager, cfg


def main() -> None:
    cfg = get_settings()
    _setup_logging(cfg.app.log_level)

    log = logging.getLogger(__name__)
    log.info("Starting %s v%s", cfg.app.name, cfg.app.version)

    app = QApplication(sys.argv)
    app.setApplicationName(cfg.app.name)
    app.setApplicationVersion(cfg.app.version)

    pipeline, dispatcher, metadata_db, model_manager, settings = build_pipeline()

    app.aboutToQuit.connect(metadata_db.close)

    window = MainWindow(
        pipeline=pipeline,
        dispatcher=dispatcher,
        metadata_db=metadata_db,
        model_manager=model_manager,
        settings=settings,
    )
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
