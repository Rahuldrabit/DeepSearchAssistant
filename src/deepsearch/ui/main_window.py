"""Main application window."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from PyQt6.QtCore import Qt, QSize  # type: ignore[import]
from PyQt6.QtGui import QAction, QIcon, QKeySequence, QShortcut  # type: ignore[import]
from PyQt6.QtWidgets import (  # type: ignore[import]
    QApplication,
    QLabel,
    QMainWindow,
    QProgressBar,
    QSplitter,
    QStatusBar,
    QSystemTrayIcon,
    QToolBar,
    QVBoxLayout,
    QWidget,
    QMenu,
)

from ..core.config import Settings
from ..core.model_manager import ModelManager
from ..generation.llm_pipeline import LLMPipeline
from ..ingestion.dispatcher import IngestionDispatcher
from ..storage.metadata_db import MetadataDB
from .chat_widget import ChatWidget
from .file_browser import FileBrowser
from .search_bar import SearchBar
from .source_panel import SourcePanel
from .workers import IndexWorker, ModelLoadWorker, QueryWorker

log = logging.getLogger(__name__)

_DARK_QSS = Path(__file__).parent / "resources" / "styles" / "dark.qss"
_LIGHT_QSS = Path(__file__).parent / "resources" / "styles" / "light.qss"


class MainWindow(QMainWindow):
    def __init__(
        self,
        pipeline: LLMPipeline,
        dispatcher: IngestionDispatcher,
        metadata_db: MetadataDB,
        model_manager: ModelManager,
        settings: Settings,
    ) -> None:
        super().__init__()
        self._pipeline = pipeline
        self._dispatcher = dispatcher
        self._db = metadata_db
        self._mm = model_manager
        self._cfg = settings
        self._dark_mode = (settings.ui.theme == "dark")

        self._active_query_worker: Optional[QueryWorker] = None
        self._active_index_worker: Optional[IndexWorker] = None
        self._model_load_worker: Optional[ModelLoadWorker] = None
        self._last_query: str = ""
        self._last_answer: str = ""

        self._setup_ui()
        self._setup_shortcuts()
        self._setup_tray()
        self._load_stylesheet()
        self._start_model_load()

    # ------------------------------------------------------------------ #
    # UI setup                                                             #
    # ------------------------------------------------------------------ #

    def _setup_ui(self) -> None:
        self.setWindowTitle("DeepSearch Assistant")
        self.resize(self._cfg.ui.window_width, self._cfg.ui.window_height)

        # Status bar
        self._status_bar = QStatusBar()
        self.setStatusBar(self._status_bar)

        self._progress = QProgressBar()
        self._progress.setMaximumWidth(200)
        self._progress.setMaximumHeight(16)
        self._progress.setVisible(False)
        self._status_bar.addPermanentWidget(self._progress)

        self._status_label = QLabel("Initialising…")
        self._status_bar.addWidget(self._status_label)

        # Menu bar
        self._build_menu_bar()

        # Toolbar
        toolbar = QToolBar("Main")
        toolbar.setMovable(False)
        self.addToolBar(toolbar)

        clear_action = QAction("Clear Chat", self)
        clear_action.triggered.connect(self._on_clear_chat)
        toolbar.addAction(clear_action)

        theme_action = QAction("Toggle Theme", self)
        theme_action.triggered.connect(self._toggle_theme)
        toolbar.addAction(theme_action)

        settings_action = QAction("Settings…", self)
        settings_action.triggered.connect(self._open_settings)
        toolbar.addAction(settings_action)

        # Central splitter: file browser | chat area | source panel
        outer_splitter = QSplitter(Qt.Orientation.Horizontal)

        self._file_browser = FileBrowser(self._db)
        self._file_browser.setMinimumWidth(200)
        self._file_browser.setMaximumWidth(300)
        self._file_browser.files_added.connect(self._on_files_added)
        outer_splitter.addWidget(self._file_browser)

        # Centre: chat + search bar
        centre = QWidget()
        centre_layout = QVBoxLayout(centre)
        centre_layout.setContentsMargins(0, 0, 0, 0)
        centre_layout.setSpacing(0)

        self._chat = ChatWidget()
        self._chat.feedback_given.connect(self._on_feedback)
        self._search_bar = SearchBar()
        self._search_bar.submitted.connect(self._on_query_submitted)

        centre_layout.addWidget(self._chat)
        centre_layout.addWidget(self._search_bar)
        outer_splitter.addWidget(centre)

        # Right: source preview panel
        self._source_panel = SourcePanel()
        self._source_panel.setMaximumWidth(360)
        outer_splitter.addWidget(self._source_panel)

        outer_splitter.setStretchFactor(0, 0)
        outer_splitter.setStretchFactor(1, 1)
        outer_splitter.setStretchFactor(2, 0)

        self.setCentralWidget(outer_splitter)

    def _build_menu_bar(self) -> None:
        menubar = self.menuBar()

        # File menu
        file_menu = menubar.addMenu("&File")
        clear_action = QAction("Clear Chat", self)
        clear_action.setShortcut(QKeySequence("Ctrl+Shift+C"))
        clear_action.triggered.connect(self._on_clear_chat)
        file_menu.addAction(clear_action)

        file_menu.addSeparator()
        quit_action = QAction("Quit", self)
        quit_action.setShortcut(QKeySequence("Ctrl+Q"))
        quit_action.triggered.connect(QApplication.quit)
        file_menu.addAction(quit_action)

        # View menu
        view_menu = menubar.addMenu("&View")
        theme_action = QAction("Toggle Dark/Light Theme", self)
        theme_action.triggered.connect(self._toggle_theme)
        view_menu.addAction(theme_action)

        # Settings menu
        settings_menu = menubar.addMenu("&Settings")
        prefs_action = QAction("Preferences…", self)
        prefs_action.setShortcut(QKeySequence("Ctrl+,"))
        prefs_action.triggered.connect(self._open_settings)
        settings_menu.addAction(prefs_action)

    def _setup_shortcuts(self) -> None:
        """Register global keyboard shortcuts."""
        # Ctrl+1/2/3 — switch search mode
        QShortcut(QKeySequence("Ctrl+1"), self).activated.connect(
            lambda: self._search_bar.set_mode("fast")
        )
        QShortcut(QKeySequence("Ctrl+2"), self).activated.connect(
            lambda: self._search_bar.set_mode("deep")
        )
        QShortcut(QKeySequence("Ctrl+3"), self).activated.connect(
            lambda: self._search_bar.set_mode("cloud")
        )

    def _setup_tray(self) -> None:
        """Configure system tray icon with context menu."""
        if not QSystemTrayIcon.isSystemTrayAvailable():
            log.debug("System tray not available on this platform")
            return

        self._tray = QSystemTrayIcon(self)
        # Use a placeholder icon — replace with a real icon file in production
        icon_path = Path(__file__).parent / "resources" / "icons" / "app.png"
        if icon_path.exists():
            self._tray.setIcon(QIcon(str(icon_path)))

        tray_menu = QMenu()
        show_action = tray_menu.addAction("Show")
        show_action.triggered.connect(self.show)
        show_action.triggered.connect(self.raise_)
        tray_menu.addSeparator()
        quit_action = tray_menu.addAction("Quit")
        quit_action.triggered.connect(QApplication.quit)
        self._tray.setContextMenu(tray_menu)

        self._tray.activated.connect(self._on_tray_activated)
        self._tray.show()

    def _load_stylesheet(self) -> None:
        qss_path = _DARK_QSS if self._dark_mode else _LIGHT_QSS
        if qss_path.exists():
            with open(qss_path) as f:
                self.setStyleSheet(f.read())

    # ------------------------------------------------------------------ #
    # Theme                                                                #
    # ------------------------------------------------------------------ #

    def _toggle_theme(self) -> None:
        self._dark_mode = not self._dark_mode
        self._load_stylesheet()
        log.debug("Theme switched to %s", "dark" if self._dark_mode else "light")

    # ------------------------------------------------------------------ #
    # Settings                                                             #
    # ------------------------------------------------------------------ #

    def _open_settings(self) -> None:
        from .settings_dialog import SettingsDialog
        dlg = SettingsDialog(self._cfg, parent=self)
        dlg.settings_saved.connect(self._on_settings_saved)
        dlg.exec()

    def _on_settings_saved(self, settings: Settings) -> None:
        self._cfg = settings
        self._dark_mode = (settings.ui.theme == "dark")
        log.info("Settings updated")
        self._load_stylesheet()

    # ------------------------------------------------------------------ #
    # Model loading                                                        #
    # ------------------------------------------------------------------ #

    def _start_model_load(self) -> None:
        """Load models asynchronously at startup."""
        self._search_bar.set_enabled(False)
        self._model_load_worker = ModelLoadWorker(self._mm)
        self._model_load_worker.progress.connect(self._status_label.setText)
        self._model_load_worker.loaded.connect(self._on_model_loaded)
        self._model_load_worker.error.connect(self._on_model_load_error)
        self._model_load_worker.start()

    def _on_model_loaded(self) -> None:
        self._status_label.setText("Ready")
        self._search_bar.set_enabled(True)
        self._chat.add_status_message("Models loaded. Ask a question to get started.")
        self._model_load_worker = None

    def _on_model_load_error(self, error: str) -> None:
        self._status_label.setText(f"Model load error: {error}")
        self._chat.add_error_message(f"Failed to load model: {error}")
        self._search_bar.set_enabled(True)
        self._model_load_worker = None

    # ------------------------------------------------------------------ #
    # Query handling                                                       #
    # ------------------------------------------------------------------ #

    def _on_query_submitted(self, query: str, mode: str) -> None:
        if self._active_query_worker and self._active_query_worker.isRunning():
            return  # Debounce

        self._last_query = query
        self._source_panel.clear()
        self._chat.add_user_message(query)
        self._chat.start_assistant_message()
        self._search_bar.set_enabled(False)
        self._status_label.setText(f"Searching [{mode}]…")

        worker = QueryWorker(
            self._pipeline,
            query,
            mode=mode,
            stream=self._cfg.ui.streaming,
        )
        worker.token_received.connect(self._chat.append_token)
        worker.result_ready.connect(self._on_query_result)
        worker.error.connect(self._on_query_error)
        worker.finished.connect(lambda: self._search_bar.set_enabled(True))
        worker.start()
        self._active_query_worker = worker

    def _on_query_result(self, result) -> None:
        self._last_answer = result.answer
        self._chat.finish_assistant_message()
        self._source_panel.show_sources(result.sources)
        conf_str = f"{result.confidence:.0%}" if result.confidence else ""
        self._status_label.setText(
            f"Done [{result.search_mode}] — {result.latency_ms:.0f} ms"
            + (f" | conf {conf_str}" if conf_str else "")
        )

    def _on_query_error(self, error: str) -> None:
        self._chat.finish_assistant_message()
        self._chat.add_error_message(error)
        self._status_label.setText("Error during query")
        self._search_bar.set_enabled(True)

    # ------------------------------------------------------------------ #
    # File indexing                                                        #
    # ------------------------------------------------------------------ #

    def _on_files_added(self, paths: list[Path]) -> None:
        if self._active_index_worker and self._active_index_worker.isRunning():
            self._chat.add_status_message("Indexing already in progress — please wait.")
            return

        self._progress.setVisible(True)
        self._progress.setRange(0, 0)  # indeterminate

        worker = IndexWorker(self._dispatcher, paths)
        worker.progress.connect(self._status_label.setText)
        worker.file_done.connect(lambda p, fid: None)
        worker.finished.connect(self._on_indexing_done)
        worker.error.connect(self._on_indexing_error)
        worker.start()
        self._active_index_worker = worker

    def _on_indexing_done(self, count: int) -> None:
        self._progress.setVisible(False)
        self._status_label.setText(f"Indexed {count} file(s)")
        self._file_browser.refresh()
        self._chat.add_status_message(f"Indexed {count} file(s). You can now ask questions.")

    def _on_indexing_error(self, error: str) -> None:
        self._progress.setVisible(False)
        self._status_label.setText("Indexing error")
        self._chat.add_error_message(f"Indexing failed: {error}")

    # ------------------------------------------------------------------ #
    # Feedback                                                             #
    # ------------------------------------------------------------------ #

    def _on_feedback(self, rating: int, answer: str) -> None:
        """Store user feedback in SQLite; boost positively-rated files."""
        self._db.add_feedback(
            query=self._last_query,
            answer=answer,
            rating=rating,
        )
        log.info("Feedback recorded: rating=%d query=%.40s", rating, self._last_query)

        if rating > 0:
            # Boost source files that contributed to this answer
            sources = getattr(self._pipeline, "_last_sources", []) or []
            for src in sources:
                file_id = src.get("file_id")
                if file_id:
                    try:
                        store = getattr(self._pipeline._search, "_vs", None) or getattr(
                            self._pipeline._search, "_vector_store", None
                        )
                        if store is not None:
                            store.boost_file(file_id)
                    except Exception as exc:
                        log.debug("score_boost failed for %s: %s", file_id, exc)

    # ------------------------------------------------------------------ #
    # System tray                                                          #
    # ------------------------------------------------------------------ #

    def _on_tray_activated(self, reason) -> None:
        if reason == QSystemTrayIcon.ActivationReason.DoubleClick:
            self.show()
            self.raise_()
            self.activateWindow()

    def closeEvent(self, event) -> None:  # type: ignore[override]
        """Minimise to tray instead of quitting when tray is available."""
        tray = getattr(self, "_tray", None)
        if tray and tray.isVisible():
            self.hide()
            event.ignore()
        else:
            event.accept()

    # ------------------------------------------------------------------ #

    def _on_clear_chat(self) -> None:
        self._chat.clear()
        self._source_panel.clear()
