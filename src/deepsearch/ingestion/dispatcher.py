"""Ingestion dispatcher — top-level pipeline: parse → chunk → embed → store.

Usage:
    dispatcher = IngestionDispatcher(vector_store, metadata_db, embedder)
    file_id = dispatcher.ingest(Path("report.pdf"))
"""
from __future__ import annotations

import logging
import uuid
from pathlib import Path
from typing import Optional

from ..core.exceptions import IndexingError
from ..storage.metadata_db import MetadataDB
from ..storage.vector_store import VectorStore
from .chunker import HierarchicalChunker
from .embedder import Embedder
from .parsers.base import BaseParser
from .parsers.audio_parser import AudioParser
from .parsers.csv_parser import CSVParser
from .parsers.docx_parser import DocxParser
from .parsers.html_parser import HTMLParser
from .parsers.image_parser import ImageParser
from .parsers.json_parser import JSONParser
from .parsers.pdf_parser import PDFParser
from .parsers.pptx_parser import PPTXParser
from .parsers.text_parser import TextParser
from .parsers.video_parser import VideoParser
from .parsers.xlsx_parser import XLSXParser

log = logging.getLogger(__name__)

# Default parser registry (ordered: most specific first, catch-all last)
_DEFAULT_PARSERS: list[BaseParser] = [
    PDFParser(),
    DocxParser(),
    PPTXParser(),
    XLSXParser(),
    JSONParser(),
    CSVParser(),
    HTMLParser(),
    ImageParser(),   # OCR-only by default (no LLaVA path configured)
    AudioParser(),
    VideoParser(),
    TextParser(),    # catch-all last
]


class IngestionDispatcher:
    def __init__(
        self,
        vector_store: VectorStore,
        metadata_db: MetadataDB,
        embedder: Embedder,
        chunker: Optional[HierarchicalChunker] = None,
        parsers: Optional[list[BaseParser]] = None,
        skip_existing: bool = True,
    ) -> None:
        self._vs = vector_store
        self._db = metadata_db
        self._embedder = embedder
        self._chunker = chunker or HierarchicalChunker()
        self._parsers = parsers or _DEFAULT_PARSERS
        self._skip_existing = skip_existing

    # ------------------------------------------------------------------ #
    # Single file                                                          #
    # ------------------------------------------------------------------ #

    def ingest(self, path: Path) -> str:
        """Parse, chunk, embed, and store one file.  Returns file_id."""
        path = path.resolve()
        if not path.exists():
            raise IndexingError(f"File not found: {path}")

        existing = self._db.get_file(path)
        if existing:
            if self._skip_existing and self._db.is_indexed(path):
                log.debug("Skipping already-indexed file: %s", path.name)
                return existing["file_id"]
            # File changed or forced re-index: delete old chunks and reuse file_id
            self._vs.delete_by_file_id(existing["file_id"])
            file_id = existing["file_id"]
        else:
            file_id = str(uuid.uuid4())
        parser = self._find_parser(path)
        if parser is None:
            raise IndexingError(f"No parser for extension: {path.suffix}")

        log.info("Ingesting: %s", path.name)

        try:
            doc = parser.parse(path, file_id)
        except Exception as exc:
            raise IndexingError(f"Parse failed for {path}: {exc}") from exc

        chunks = self._chunker.chunk(
            doc.text,
            file_id=file_id,
            base_metadata={
                "source_path": doc.source_path,
                "file_name": path.name,
                "file_type": path.suffix.lstrip("."),
                **doc.metadata,
            },
        )

        if not chunks:
            log.warning("No chunks produced for %s", path.name)
            return file_id

        # Only embed level-1 chunks (primary retrieval units) for speed.
        # Level-0 and level-2 are stored in metadata for context expansion.
        level1_chunks = [c for c in chunks if c.level == 1]
        if not level1_chunks:
            level1_chunks = chunks  # fallback: embed everything

        log.debug("Embedding %d chunks for %s", len(level1_chunks), path.name)
        embedded = self._embedder.embed_chunks(level1_chunks)
        self._vs.upsert_batch(embedded)

        self._db.upsert_file(file_id, path, chunk_count=len(embedded))
        log.info("Indexed %s — %d chunks", path.name, len(embedded))
        return file_id

    def ingest_directory(self, directory: Path, recursive: bool = True) -> list[str]:
        """Ingest all supported files in a directory."""
        pattern = "**/*" if recursive else "*"
        supported = {ext for p in self._parsers for ext in p.supported_extensions}
        files = [p for p in directory.glob(pattern) if p.suffix.lower() in supported and p.is_file()]

        log.info("Found %d files to index in %s", len(files), directory)
        file_ids: list[str] = []
        errors: list[str] = []

        for path in files:
            try:
                fid = self.ingest(path)
                file_ids.append(fid)
            except IndexingError as exc:
                log.warning("Skipped %s: %s", path.name, exc)
                errors.append(str(path))

        if errors:
            log.warning("Failed to index %d files: %s", len(errors), errors[:5])
        return file_ids

    def delete(self, path: Path) -> None:
        """Remove a file and its chunks from the index."""
        existing = self._db.get_file(path)
        if existing:
            self._vs.delete_by_file_id(existing["file_id"])
            self._db.delete_file(existing["file_id"])
            log.info("Deleted from index: %s", path.name)

    # ------------------------------------------------------------------ #

    def _find_parser(self, path: Path) -> BaseParser | None:
        for p in self._parsers:
            if p.can_parse(path):
                return p
        return None
