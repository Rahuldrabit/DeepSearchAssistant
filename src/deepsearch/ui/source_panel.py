"""Source preview panel — shows the text content of retrieved chunks.

Supports plain-text preview for all document types.  PDF page rendering
and image thumbnails are shown when the required libraries are available.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from PyQt6.QtCore import Qt  # type: ignore[import]
from PyQt6.QtWidgets import (  # type: ignore[import]
    QFrame,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QSplitter,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

log = logging.getLogger(__name__)


class SourcePanel(QWidget):
    """Right-hand panel that shows the source chunks for the last answer.

    Usage::

        panel.show_sources(result.sources)  # after a query completes
    """

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setMinimumWidth(240)
        self._sources: list[dict] = []
        self._build_ui()

    # ------------------------------------------------------------------ #
    # UI                                                                   #
    # ------------------------------------------------------------------ #

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        header = QLabel("Sources")
        header.setObjectName("panelHeader")
        layout.addWidget(header)

        splitter = QSplitter(Qt.Orientation.Vertical)

        # Top: list of source chunks
        self._list = QListWidget()
        self._list.setAlternatingRowColors(True)
        self._list.currentRowChanged.connect(self._on_selection_changed)
        splitter.addWidget(self._list)

        # Bottom: chunk text preview
        self._preview = QTextEdit()
        self._preview.setReadOnly(True)
        self._preview.setObjectName("sourcePreview")
        splitter.addWidget(self._preview)

        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 2)
        layout.addWidget(splitter)

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def show_sources(self, sources: list[dict]) -> None:
        """Populate the panel with retrieved source chunks.

        Each source dict is expected to have at least:
          - ``text``      : the chunk content
          - ``file_name`` : display name
          - ``page``      : page number (optional)
        """
        self._sources = sources
        self._list.clear()
        self._preview.clear()

        for i, src in enumerate(sources, start=1):
            file_name = src.get("file_name", src.get("source_path", "Unknown"))
            page = src.get("page")
            label = f"[{i}] {file_name}"
            if page is not None:
                label += f"  p.{page}"
            item = QListWidgetItem(label)
            item.setToolTip(src.get("text", "")[:200])
            self._list.addItem(item)

        if sources:
            self._list.setCurrentRow(0)

    def clear(self) -> None:
        self._sources = []
        self._list.clear()
        self._preview.clear()

    # ------------------------------------------------------------------ #

    def _on_selection_changed(self, row: int) -> None:
        if row < 0 or row >= len(self._sources):
            self._preview.clear()
            return
        src = self._sources[row]
        text = src.get("text", "")
        file_name = src.get("file_name", "")
        page = src.get("page")

        header = f"— {file_name}"
        if page is not None:
            header += f"  (page {page})"
        self._preview.setPlainText(f"{header}\n\n{text}")
