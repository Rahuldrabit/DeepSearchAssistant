"""ProgressWidget — compact progress bar + status label for indexing operations.

Designed to sit at the bottom of the FileBrowser panel.  Shows:
  - An animated progress bar (indeterminate while scanning, determinate while indexing)
  - A status label ("Indexing 3/12 files…")
  - An elapsed-time counter that ticks every second
  - A cancel button that emits cancel_requested signal
"""
from __future__ import annotations

from PyQt6.QtCore import QElapsedTimer, QTimer, pyqtSignal  # type: ignore[import]
from PyQt6.QtWidgets import (  # type: ignore[import]
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)


class ProgressWidget(QWidget):
    """Compact progress panel used during file indexing."""

    cancel_requested = pyqtSignal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._elapsed = QElapsedTimer()
        self._tick_timer = QTimer(self)
        self._tick_timer.setInterval(1000)
        self._tick_timer.timeout.connect(self._update_elapsed)
        self._total = 0
        self._done = 0
        self._setup_ui()
        self.hide()

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def start(self, total: int = 0) -> None:
        """Show the widget and begin timing.  Pass total=0 for indeterminate."""
        self._total = total
        self._done = 0
        self._elapsed.start()
        self._tick_timer.start()
        self._bar.setMaximum(max(total, 0))
        self._bar.setValue(0)
        if total == 0:
            self._bar.setRange(0, 0)  # indeterminate (animated)
        self._cancel_btn.setEnabled(True)
        self._status_label.setText("Scanning…")
        self._time_label.setText("0s")
        self.show()

    def update(self, done: int, message: str = "") -> None:
        """Update progress counter and status message."""
        self._done = done
        if self._total > 0:
            self._bar.setRange(0, self._total)
            self._bar.setValue(done)
        label = message or (
            f"Indexing {done}/{self._total} files…" if self._total > 0
            else f"Indexed {done} files…"
        )
        self._status_label.setText(label)

    def set_status(self, message: str) -> None:
        """Set status text without changing progress value."""
        self._status_label.setText(message)

    def finish(self, message: str = "") -> None:
        """Mark operation as complete and stop the timer."""
        self._tick_timer.stop()
        if self._total > 0:
            self._bar.setValue(self._total)
        else:
            self._bar.setRange(0, 1)
            self._bar.setValue(1)
        elapsed_s = self._elapsed.elapsed() // 1000
        self._status_label.setText(message or f"Done — {self._done} files in {elapsed_s}s")
        self._cancel_btn.setEnabled(False)

    def reset(self) -> None:
        """Hide the widget and reset state."""
        self._tick_timer.stop()
        self._bar.setRange(0, 1)
        self._bar.setValue(0)
        self._status_label.setText("")
        self._time_label.setText("")
        self.hide()

    # ------------------------------------------------------------------ #
    # Internal                                                             #
    # ------------------------------------------------------------------ #

    def _setup_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(4, 4, 4, 4)
        root.setSpacing(2)

        # Status row
        top_row = QHBoxLayout()
        self._status_label = QLabel("")
        self._status_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        self._time_label = QLabel("0s")
        self._cancel_btn = QPushButton("✕")
        self._cancel_btn.setFixedWidth(24)
        self._cancel_btn.setToolTip("Cancel indexing")
        self._cancel_btn.clicked.connect(self.cancel_requested)
        top_row.addWidget(self._status_label)
        top_row.addWidget(self._time_label)
        top_row.addWidget(self._cancel_btn)

        # Progress bar
        self._bar = QProgressBar()
        self._bar.setFixedHeight(6)
        self._bar.setTextVisible(False)

        root.addLayout(top_row)
        root.addWidget(self._bar)

    def _update_elapsed(self) -> None:
        elapsed_s = self._elapsed.elapsed() // 1000
        self._time_label.setText(f"{elapsed_s}s")
