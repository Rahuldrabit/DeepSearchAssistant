"""JSON document parser — flattens arbitrary JSON to readable text."""
from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any

from .base import BaseParser, ParsedDocument


def _flatten(obj: Any, prefix: str = "", depth: int = 0) -> list[str]:
    """Recursively flatten a JSON object to key: value lines.

    Args:
        obj:    any JSON-parsed value
        prefix: dotted key path accumulated so far
        depth:  recursion depth guard (max 20)

    Returns:
        List of human-readable "key: value" strings.
    """
    if depth > 20:
        return [f"{prefix}: [truncated]"]

    lines: list[str] = []

    if isinstance(obj, dict):
        for k, v in obj.items():
            child_key = f"{prefix}.{k}" if prefix else k
            lines.extend(_flatten(v, child_key, depth + 1))
    elif isinstance(obj, list):
        for i, item in enumerate(obj):
            child_key = f"{prefix}[{i}]"
            lines.extend(_flatten(item, child_key, depth + 1))
    else:
        value_str = str(obj) if obj is not None else "null"
        lines.append(f"{prefix}: {value_str}" if prefix else value_str)

    return lines


class JSONParser(BaseParser):
    """Parses JSON files into flat, readable text."""

    @property
    def supported_extensions(self) -> list[str]:
        return [".json", ".jsonl"]

    def parse(self, path: Path, file_id: str) -> ParsedDocument:
        raw = path.read_text(encoding="utf-8", errors="replace")

        if path.suffix.lower() == ".jsonl":
            text = self._parse_jsonl(raw, path)
        else:
            text = self._parse_json(raw, path)

        return ParsedDocument(
            file_id=file_id,
            source_path=str(path),
            text=text,
            metadata={
                "title": path.stem,
                "format": "json",
                "file_name": path.name,
            },
        )

    # ------------------------------------------------------------------ #

    @staticmethod
    def _parse_json(raw: str, path: Path) -> str:
        try:
            obj = json.loads(raw)
        except json.JSONDecodeError as exc:
            return f"[JSON parse error in {path.name}: {exc}]\n{raw[:2000]}"
        lines = _flatten(obj)
        return "\n".join(lines)

    @staticmethod
    def _parse_jsonl(raw: str, path: Path) -> str:
        """Parse newline-delimited JSON (one JSON object per line)."""
        parts: list[str] = []
        record_num = 0
        for lineno, line in enumerate(raw.splitlines(), start=1):
            line = line.strip()
            if not line:
                continue
            record_num += 1
            try:
                obj = json.loads(line)
                parts.append(f"--- record {record_num} ---")
                parts.extend(_flatten(obj))
            except json.JSONDecodeError:
                parts.append(f"[malformed line {lineno}]: {line[:200]}")
        return "\n".join(parts)
