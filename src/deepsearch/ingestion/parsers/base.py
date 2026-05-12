"""Base parser protocol for all document types."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ParsedDocument:
    """Canonical output of every parser."""
    file_id: str
    source_path: str
    text: str                              # full extracted text
    pages: list[str] = field(default_factory=list)   # per-page text (optional)
    metadata: dict = field(default_factory=dict)     # title, author, etc.
    images: list[dict] = field(default_factory=list) # [{page, path}] for multimodal
    tables: list[str] = field(default_factory=list)  # stringified tables

    @property
    def word_count(self) -> int:
        return len(self.text.split())


class BaseParser(ABC):
    """Every parser converts a file path to ParsedDocument."""

    @abstractmethod
    def parse(self, path: Path, file_id: str) -> ParsedDocument:
        ...

    @property
    @abstractmethod
    def supported_extensions(self) -> list[str]:
        ...

    def can_parse(self, path: Path) -> bool:
        return path.suffix.lower() in self.supported_extensions
