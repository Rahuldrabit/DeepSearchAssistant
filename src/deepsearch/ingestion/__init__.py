"""Ingestion pipeline: parse → chunk → embed → store."""
from .chunker import Chunk, HierarchicalChunker
from .dispatcher import IngestionDispatcher
from .embedder import Embedder

__all__ = ["IngestionDispatcher", "HierarchicalChunker", "Embedder", "Chunk"]
