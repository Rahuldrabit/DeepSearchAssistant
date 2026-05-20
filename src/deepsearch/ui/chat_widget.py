"""Chat message display widget."""
from __future__ import annotations

from PyQt6.QtCore import Qt, pyqtSignal  # type: ignore[import]
from PyQt6.QtGui import QFont  # type: ignore[import]
from PyQt6.QtWidgets import (  # type: ignore[import]
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)


class MessageBubble(QFrame):
    """A single chat message bubble (user or assistant).

    For assistant bubbles, thumbs-up / thumbs-down feedback buttons are shown
    once the message is finalised.  ``feedback_given`` is emitted with
    ``rating`` (+1 or -1) and the full answer text.
    """

    feedback_given = pyqtSignal(int, str)  # (rating, answer_text)

    def __init__(self, text: str, role: str = "assistant", parent=None) -> None:
        super().__init__(parent)
        self._role = role
        self._label = QLabel(text)
        self._label.setWordWrap(True)
        self._label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        self._label.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum
        )

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 8, 12, 8)
        layout.addWidget(self._label)

        # Feedback row — only for assistant messages
        self._feedback_row: QHBoxLayout | None = None
        if role == "assistant":
            self._feedback_row = QHBoxLayout()
            self._thumbs_up = QPushButton("👍", self)
            self._thumbs_up.setFixedSize(28, 28)
            self._thumbs_up.setToolTip("Good answer")
            self._thumbs_up.clicked.connect(lambda: self._on_feedback(1))
            self._thumbs_down = QPushButton("👎", self)
            self._thumbs_down.setFixedSize(28, 28)
            self._thumbs_down.setToolTip("Bad answer")
            self._thumbs_down.clicked.connect(lambda: self._on_feedback(-1))
            self._feedback_row.addStretch()
            self._feedback_row.addWidget(self._thumbs_up)
            self._feedback_row.addWidget(self._thumbs_down)
            layout.addLayout(self._feedback_row)
            # Hidden until message is finalised
            self._thumbs_up.setVisible(False)
            self._thumbs_down.setVisible(False)

        if role == "user":
            self.setObjectName("userBubble")
        else:
            self.setObjectName("assistantBubble")

    def append_text(self, token: str) -> None:
        self._label.setText(self._label.text() + token)

    def show_feedback_buttons(self) -> None:
        """Make feedback buttons visible (call once streaming is done)."""
        if self._role == "assistant":
            # Use show() in addition to setVisible() to ensure Qt
            # updates visibility immediately under test harnesses.
            self._thumbs_up.setVisible(True)
            self._thumbs_down.setVisible(True)
            self._thumbs_up.show()
            self._thumbs_down.show()

    def _on_feedback(self, rating: int) -> None:
        self.feedback_given.emit(rating, self._label.text())
        # Disable both buttons after one rating
        self._thumbs_up.setEnabled(False)
        self._thumbs_down.setEnabled(False)

    @property
    def text(self) -> str:
        return self._label.text()

    @text.setter
    def text(self, value: str) -> None:
        self._label.setText(value)


class ChatWidget(QWidget):
    """Scrollable chat history with streaming support.

    Signals:
        feedback_given(int, str): emitted when user rates an answer.
            int = +1 (thumbs up) or -1 (thumbs down); str = answer text.
    """

    feedback_given = pyqtSignal(int, str)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._messages: list[MessageBubble] = []
        self._current_assistant_bubble: MessageBubble | None = None

        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        self._container = QWidget()
        self._container_layout = QVBoxLayout(self._container)
        self._container_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        self._container_layout.setSpacing(8)
        self._container_layout.setContentsMargins(16, 16, 16, 16)

        self._scroll.setWidget(self._container)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(self._scroll)

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def add_user_message(self, text: str) -> None:
        bubble = MessageBubble(text, role="user")
        row = QHBoxLayout()
        row.addStretch()
        row.addWidget(bubble)
        self._container_layout.addLayout(row)
        self._messages.append(bubble)
        self._scroll_to_bottom()

    def start_assistant_message(self) -> None:
        """Begin a new streaming assistant bubble."""
        bubble = MessageBubble("", role="assistant")
        row = QHBoxLayout()
        row.addWidget(bubble)
        row.addStretch()
        self._container_layout.addLayout(row)
        self._messages.append(bubble)
        self._current_assistant_bubble = bubble
        self._scroll_to_bottom()

    def append_token(self, token: str) -> None:
        """Append a streamed token to the current assistant bubble."""
        if self._current_assistant_bubble:
            self._current_assistant_bubble.append_text(token)
            self._scroll_to_bottom()

    def finish_assistant_message(self, full_text: str | None = None) -> None:
        """Finalise the current assistant message and show feedback buttons."""
        if self._current_assistant_bubble:
            bubble = self._current_assistant_bubble
            if full_text is not None:
                bubble.text = full_text
            # Ensure the widget hierarchy is shown so QWidget.isVisible()
            # reflects the updated state immediately in tests.
            self.show()
            bubble.show()
            bubble.show_feedback_buttons()
            bubble.feedback_given.connect(self.feedback_given)
        self._current_assistant_bubble = None

    def add_error_message(self, error: str) -> None:
        bubble = MessageBubble(f"Error: {error}", role="error")
        bubble.setObjectName("errorBubble")
        row = QHBoxLayout()
        row.addWidget(bubble)
        row.addStretch()
        self._container_layout.addLayout(row)
        self._scroll_to_bottom()

    def add_status_message(self, text: str) -> None:
        label = QLabel(text)
        label.setObjectName("statusLabel")
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._container_layout.addWidget(label)
        self._scroll_to_bottom()

    def clear(self) -> None:
        while self._container_layout.count():
            item = self._container_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
            elif item.layout():
                self._clear_layout(item.layout())
        self._messages.clear()
        self._current_assistant_bubble = None

    # ------------------------------------------------------------------ #

    def _scroll_to_bottom(self) -> None:
        vsb = self._scroll.verticalScrollBar()
        vsb.setValue(vsb.maximum())

    @staticmethod
    def _clear_layout(layout) -> None:
        while layout.count():
            item = layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
