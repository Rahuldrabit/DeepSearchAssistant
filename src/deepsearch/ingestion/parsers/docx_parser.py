"""DOCX parser using python-docx."""
from __future__ import annotations

import logging
from pathlib import Path

from .base import BaseParser, ParsedDocument

log = logging.getLogger(__name__)


class DocxParser(BaseParser):
    @property
    def supported_extensions(self) -> list[str]:
        return [".docx", ".doc"]

    def parse(self, path: Path, file_id: str) -> ParsedDocument:
        try:
            from docx import Document  # type: ignore[import]
        except ImportError as exc:
            raise RuntimeError("python-docx not installed. Run: pip install python-docx") from exc

        doc = Document(str(path))

        paragraphs: list[str] = []
        tables_text: list[str] = []

        for para in doc.paragraphs:
            t = para.text.strip()
            if t:
                paragraphs.append(t)

        for table in doc.tables:
            rows = []
            for row in table.rows:
                cells = [cell.text.strip() for cell in row.cells]
                rows.append(" | ".join(cells))
            tables_text.append("\n".join(rows))

        full_text = "\n\n".join(paragraphs)
        if tables_text:
            full_text += "\n\n" + "\n\n".join(tables_text)

        metadata = {
            "title": path.stem,
            "file_name": path.name,
            "paragraph_count": len(paragraphs),
            "table_count": len(tables_text),
        }
        # Try to extract core properties
        try:
            cp = doc.core_properties
            metadata["author"] = cp.author or ""
            metadata["title"] = cp.title or path.stem
        except Exception:
            pass

        log.debug("DOCX parsed: %s — %d paragraphs", path.name, len(paragraphs))
        return ParsedDocument(
            file_id=file_id,
            source_path=str(path),
            text=full_text,
            metadata=metadata,
            tables=tables_text,
        )
