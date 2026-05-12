"""CSV/TSV document parser — converts tabular data to readable prose."""
from __future__ import annotations

import csv
import io
from pathlib import Path

from .base import BaseParser, ParsedDocument


_MAX_ROWS = 5_000   # hard cap to avoid massive context windows


class CSVParser(BaseParser):
    """Parses CSV and TSV files into natural language table rows."""

    @property
    def supported_extensions(self) -> list[str]:
        return [".csv", ".tsv"]

    def parse(self, path: Path, file_id: str) -> ParsedDocument:
        raw = path.read_text(encoding="utf-8-sig", errors="replace")  # BOM-safe
        delimiter = "\t" if path.suffix.lower() == ".tsv" else self._detect_delimiter(raw)

        reader = csv.DictReader(io.StringIO(raw), delimiter=delimiter)
        rows: list[str] = []
        tables: list[str] = []
        headers: list[str] = []

        try:
            headers = list(reader.fieldnames or [])
        except Exception:
            headers = []

        if headers:
            tables.append(" | ".join(headers))

        for i, row in enumerate(reader):
            if i >= _MAX_ROWS:
                rows.append(f"[truncated — showing first {_MAX_ROWS} of {i}+ rows]")
                break
            # Convert each row to "key: value, key: value, ..." prose
            parts = [f"{k}: {v}" for k, v in row.items() if v is not None and str(v).strip()]
            rows.append("; ".join(parts))
            if i < 500:  # include first 500 rows in table view
                tables.append(" | ".join(str(v) for v in row.values()))

        text = "\n".join(rows)
        return ParsedDocument(
            file_id=file_id,
            source_path=str(path),
            text=text,
            tables=tables,
            metadata={
                "title": path.stem,
                "format": path.suffix.lstrip(".").upper(),
                "file_name": path.name,
                "columns": headers,
                "row_count": len(rows),
            },
        )

    # ------------------------------------------------------------------ #

    @staticmethod
    def _detect_delimiter(raw: str) -> str:
        """Sniff the most likely delimiter from the first 4 KB."""
        sample = raw[:4096]
        sniffer = csv.Sniffer()
        try:
            dialect = sniffer.sniff(sample, delimiters=",;\t|")
            return dialect.delimiter
        except csv.Error:
            return ","
