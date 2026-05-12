"""Plain-text parsers: .txt, .md, .py, .json, .csv, .html."""
from __future__ import annotations

import csv
import io
import json
import logging
from pathlib import Path

from .base import BaseParser, ParsedDocument

log = logging.getLogger(__name__)


class TextParser(BaseParser):
    """Generic UTF-8 text parser for .txt, .md, .py, .js, .ts, etc."""

    @property
    def supported_extensions(self) -> list[str]:
        return [
            ".txt", ".md", ".rst", ".py", ".js", ".ts", ".tsx",
            ".java", ".cpp", ".c", ".h", ".go", ".rs", ".rb",
            ".yaml", ".yml", ".toml", ".ini", ".sh", ".bat",
        ]

    def parse(self, path: Path, file_id: str) -> ParsedDocument:
        text = path.read_text(encoding="utf-8", errors="replace")
        metadata = {
            "title": path.stem,
            "file_name": path.name,
            "extension": path.suffix,
            "size_bytes": path.stat().st_size,
        }
        return ParsedDocument(
            file_id=file_id,
            source_path=str(path),
            text=text,
            metadata=metadata,
        )


class JSONParser(BaseParser):
    @property
    def supported_extensions(self) -> list[str]:
        return [".json", ".jsonl"]

    def parse(self, path: Path, file_id: str) -> ParsedDocument:
        raw = path.read_text(encoding="utf-8", errors="replace")
        # Pretty-print so the LLM sees structure
        try:
            obj = json.loads(raw)
            text = json.dumps(obj, indent=2, ensure_ascii=False)
        except json.JSONDecodeError:
            text = raw  # fallback: pass as-is
        return ParsedDocument(
            file_id=file_id,
            source_path=str(path),
            text=text,
            metadata={"title": path.stem, "file_name": path.name},
        )


class CSVParser(BaseParser):
    @property
    def supported_extensions(self) -> list[str]:
        return [".csv", ".tsv"]

    def parse(self, path: Path, file_id: str) -> ParsedDocument:
        delimiter = "\t" if path.suffix == ".tsv" else ","
        raw = path.read_text(encoding="utf-8", errors="replace")
        reader = csv.DictReader(io.StringIO(raw), delimiter=delimiter)
        rows = list(reader)
        lines = [", ".join(f"{k}: {v}" for k, v in row.items()) for row in rows]
        text = "\n".join(lines)
        return ParsedDocument(
            file_id=file_id,
            source_path=str(path),
            text=text,
            metadata={
                "title": path.stem,
                "file_name": path.name,
                "row_count": len(rows),
                "columns": list(rows[0].keys()) if rows else [],
            },
        )


class HTMLParser(BaseParser):
    @property
    def supported_extensions(self) -> list[str]:
        return [".html", ".htm"]

    def parse(self, path: Path, file_id: str) -> ParsedDocument:
        raw = path.read_text(encoding="utf-8", errors="replace")
        try:
            from html.parser import HTMLParser as _HP

            class Stripper(_HP):
                def __init__(self):
                    super().__init__()
                    self._parts: list[str] = []

                def handle_data(self, data: str) -> None:
                    stripped = data.strip()
                    if stripped:
                        self._parts.append(stripped)

            p = Stripper()
            p.feed(raw)
            text = " ".join(p._parts)
        except Exception:
            text = raw

        return ParsedDocument(
            file_id=file_id,
            source_path=str(path),
            text=text,
            metadata={"title": path.stem, "file_name": path.name},
        )
