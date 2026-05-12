"""Search bar with mode selector and submit button."""
from __future__ import annotations

from PyQt6.QtCore import Qt, pyqtSignal  # type: ignore[import]
from PyQt6.QtWidgets import (  # type: ignore[import]
    QComboBox,
    QHBoxLayout,
    QPushButton,
    QTextEdit,
    QWidget,
)


class SearchBar(QWidget):
    """Multi-line query input with mode dropdown and send button."""

    submitted = pyqtSignal(str, str)  # (query_text, mode)

    _MODES = [
        ("Fast (<3s)", "fast"),
        ("Deep (5-15s)", "deep"),
        ("Cloud Deep", "cloud"),
    ]

    def __init__(self, parent=None) -> None:
        super().__init__(parent)

        self._input = QTextEdit()
        self._input.setPlaceholderText("Ask a question about your documents…")
        self._input.setMaximumHeight(80)
        self._input.setObjectName("searchInput")

        self._mode_combo = QComboBox()
        for label, _ in self._MODES:
            self._mode_combo.addItem(label)
        self._mode_combo.setObjectName("modeCombo")
        self._mode_combo.setFixedWidth(130)

        self._send_btn = QPushButton("Send")
        self._send_btn.setObjectName("sendButton")
        self._send_btn.setFixedWidth(70)
        self._send_btn.clicked.connect(self._on_submit)

        row = QHBoxLayout(self)
        row.setContentsMargins(8, 6, 8, 6)
        row.setSpacing(8)
        row.addWidget(self._input)
        row.addWidget(self._mode_combo)
        row.addWidget(self._send_btn)

        # Allow Ctrl+Enter or Enter to submit
        self._input.installEventFilter(self)

    # ------------------------------------------------------------------ #

    def eventFilter(self, obj, event) -> bool:
        from PyQt6.QtCore import QEvent
        from PyQt6.QtGui import QKeyEvent

        if obj is self._input and event.type() == QEvent.Type.KeyPress:
            key_event: QKeyEvent = event
            if key_event.key() == Qt.Key.Key_Return:
                mods = key_event.modifiers()
                # Enter (no shift) or Ctrl+Enter submits
                if not (mods & Qt.KeyboardModifier.ShiftModifier):
                    self._on_submit()
                    return True
        return super().eventFilter(obj, event)

    def _on_submit(self) -> None:
        text = self._input.toPlainText().strip()
        if not text:
            return
        mode = self._MODES[self._mode_combo.currentIndex()][1]
        self._input.clear()
        self.submitted.emit(text, mode)

    def set_enabled(self, enabled: bool) -> None:
        self._input.setEnabled(enabled)
        self._send_btn.setEnabled(enabled)
        self._mode_combo.setEnabled(enabled)

    @property
    def current_mode(self) -> str:
        return self._MODES[self._mode_combo.currentIndex()][1]

    def set_mode(self, mode: str) -> None:
        """Switch the mode combo to the given mode string (fast/deep/cloud)."""
        for i, (_, m) in enumerate(self._MODES):
            if m == mode:
                self._mode_combo.setCurrentIndex(i)
                return
