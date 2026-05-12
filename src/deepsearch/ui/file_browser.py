"""File browser panel — shows indexed documents and allows adding new ones."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from PyQt6.QtCore import Qt, pyqtSignal  # type: ignore[import]
from PyQt6.QtWidgets import (  # type: ignore[import]
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ..storage.metadata_db import MetadataDB


class FileBrowser(QWidget):
    """Left-side panel showing indexed files."""

    files_added = pyqtSignal(list)   # list[Path]
    file_removed = pyqtSignal(str)   # file_id

    def __init__(self, metadata_db: MetadataDB, parent=None) -> None:
        super().__init__(parent)
        self._db = metadata_db

        title = QLabel("Indexed Documents")
        title.setObjectName("panelTitle")

        self._list = QListWidget()
        self._list.setObjectName("fileList")

        add_btn = QPushButton("+ Add Files")
        add_btn.setObjectName("addFilesButton")
        add_btn.clicked.connect(self._on_add_files)

        add_dir_btn = QPushButton("+ Add Folder")
        add_dir_btn.setObjectName("addFilesButton")
        add_dir_btn.clicked.connect(self._on_add_folder)

        btn_row = QHBoxLayout()
        btn_row.addWidget(add_btn)
        btn_row.addWidget(add_dir_btn)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.addWidget(title)
        layout.addWidget(self._list)
        layout.addLayout(btn_row)

        self.refresh()

    # ------------------------------------------------------------------ #

    def refresh(self) -> None:
        self._list.clear()
        for file_info in self._db.list_files():
            item = QListWidgetItem(Path(file_info["path"]).name)
            item.setData(Qt.ItemDataRole.UserRole, file_info["file_id"])
            item.setToolTip(file_info["path"])
            self._list.addItem(item)

    def _on_add_files(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(
            self,
            "Add Documents",
            str(Path.home()),
            "Documents (*.pdf *.docx *.txt *.md *.py *.json *.csv *.html);;All Files (*)",
        )
        if paths:
            self.files_added.emit([Path(p) for p in paths])

    def _on_add_folder(self) -> None:
        folder = QFileDialog.getExistingDirectory(
            self, "Add Folder", str(Path.home())
        )
        if folder:
            self.files_added.emit([Path(folder)])
