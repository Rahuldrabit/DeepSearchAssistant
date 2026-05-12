"""Excel (.xlsx / .xls) parser — extracts sheets as prose + table rows.

Requires: openpyxl  (pip install openpyxl)
"""
from __future__ import annotations

import logging
from pathlib import Path

from .base import BaseParser, ParsedDocument

log = logging.getLogger(__name__)

_MAX_ROWS_PER_SHEET = 2_000
_MAX_COLS = 50


class XLSXParser(BaseParser):
    """Parses Excel workbooks into sheet-by-sheet text."""

    @property
    def supported_extensions(self) -> list[str]:
        return [".xlsx", ".xls", ".xlsm"]

    def parse(self, path: Path, file_id: str) -> ParsedDocument:
        try:
            import openpyxl  # type: ignore[import]
        except ImportError as exc:
            raise ImportError(
                "openpyxl is required for XLSX parsing. "
                "Run: pip install openpyxl"
            ) from exc

        try:
            wb = openpyxl.load_workbook(str(path), read_only=True, data_only=True)
        except Exception as exc:
            return ParsedDocument(
                file_id=file_id,
                source_path=str(path),
                text=f"[XLSX parse error: {exc}]",
                metadata={"title": path.stem, "format": "xlsx", "file_name": path.name},
            )

        pages: list[str] = []
        tables: list[str] = []
        all_parts: list[str] = []

        for sheet_name in wb.sheetnames:
            ws = wb[sheet_name]
            sheet_parts: list[str] = [f"=== Sheet: {sheet_name} ==="]
            headers: list[str] = []
            row_count = 0

            for row_idx, row in enumerate(ws.iter_rows(values_only=True)):
                if row_count >= _MAX_ROWS_PER_SHEET:
                    sheet_parts.append(f"[truncated — {_MAX_ROWS_PER_SHEET} row limit]")
                    break

                cells = [str(c) if c is not None else "" for c in row[:_MAX_COLS]]
                # Skip completely empty rows
                if not any(c.strip() for c in cells):
                    continue

                if row_idx == 0:
                    headers = [c.strip() for c in cells]
                    tables.append(" | ".join(headers))
                    sheet_parts.append("Headers: " + " | ".join(headers))
                else:
                    if headers:
                        # key: value prose
                        pairs = [
                            f"{h}: {v}" for h, v in zip(headers, cells)
                            if v.strip() and h.strip()
                        ]
                        sheet_parts.append("; ".join(pairs))
                    else:
                        sheet_parts.append("; ".join(c for c in cells if c.strip()))
                    tables.append(" | ".join(cells))
                    row_count += 1

            page_text = "\n".join(sheet_parts)
            pages.append(page_text)
            all_parts.append(page_text)

        wb.close()

        return ParsedDocument(
            file_id=file_id,
            source_path=str(path),
            text="\n\n".join(all_parts),
            pages=pages,
            tables=tables,
            metadata={
                "title": path.stem,
                "format": path.suffix.lstrip(".").lower(),
                "file_name": path.name,
                "sheet_count": len(wb.sheetnames),
                "sheet_names": wb.sheetnames,
            },
        )
