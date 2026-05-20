"""File watcher — monitors directories and triggers re-indexing on changes.

Uses the ``watchdog`` library (pip install watchdog).  Falls back gracefully
when watchdog is not installed (watcher simply does nothing).

Usage:
    watcher = FileWatcher(dispatcher, debounce_seconds=2.0)
    watcher.watch(Path("/my/docs"))
    watcher.start()
    ...
    watcher.stop()

Or as a context manager:
    with FileWatcher(dispatcher) as watcher:
        watcher.watch(Path("/my/docs"))
        time.sleep(60)
"""
from __future__ import annotations

import logging
import threading
import time
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)


class FileWatcher:
    """Monitors one or more directories and re-indexes changed files.

    Args:
        dispatcher:       IngestionDispatcher to call for each changed file
        debounce_seconds: wait this long after a change before re-indexing
                          (avoids repeated triggers during large file saves)
        recursive:        watch subdirectories too
    """

    def __init__(
        self,
        dispatcher,  # IngestionDispatcher — avoid circular import
        debounce_seconds: float = 2.0,
        recursive: bool = True,
    ) -> None:
        self._dispatcher = dispatcher
        self._debounce = debounce_seconds
        self._recursive = recursive
        self._observer: Optional[object] = None
        self._pending: dict[str, float] = {}   # path → event time
        self._lock = threading.Lock()
        self._running = False
        self._flush_thread: Optional[threading.Thread] = None
        self._watched_paths: list[Path] = []

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def watch(self, path: Path) -> None:
        """Add a directory to the watch list.  Must be called before start()."""
        path = path.resolve()
        if not path.is_dir():
            raise ValueError(f"Not a directory: {path}")
        self._watched_paths.append(path)

    def start(self) -> None:
        """Start the file watcher background threads."""
        if self._running:
            return
        try:
            from watchdog.observers import Observer  # type: ignore[import]
            from watchdog.events import FileSystemEventHandler  # type: ignore[import]
        except ImportError:
            log.warning(
                "watchdog not installed — file watcher disabled. "
                "Run: pip install watchdog"
            )
            return

        handler = _ChangeHandler(self._on_event)
        self._observer = Observer()
        for path in self._watched_paths:
            self._observer.schedule(handler, str(path), recursive=self._recursive)
            log.info("Watching: %s (recursive=%s)", path, self._recursive)

        self._running = True
        self._observer.start()

        self._flush_thread = threading.Thread(
            target=self._flush_loop, daemon=True, name="FileWatcherFlush"
        )
        self._flush_thread.start()
        log.info("FileWatcher started")

    def stop(self) -> None:
        """Stop the file watcher."""
        self._running = False
        if self._observer is not None:
            try:
                self._observer.stop()
                self._observer.join(timeout=5)
            except Exception:
                pass
            self._observer = None
        if self._flush_thread is not None:
            self._flush_thread.join(timeout=5)
            self._flush_thread = None
        log.info("FileWatcher stopped")

    def __enter__(self) -> "FileWatcher":
        self.start()
        return self

    def __exit__(self, *args) -> None:
        self.stop()

    # ------------------------------------------------------------------ #
    # Internal                                                             #
    # ------------------------------------------------------------------ #

    def _on_event(self, path: str) -> None:
        """Called by watchdog handler on any file change."""
        with self._lock:
            self._pending[path] = time.monotonic()

    def _flush_loop(self) -> None:
        """Background thread: drain pending events after debounce window."""
        while self._running:
            time.sleep(0.5)
            now = time.monotonic()
            ready: list[str] = []
            with self._lock:
                for path, ts in list(self._pending.items()):
                    if now - ts >= self._debounce:
                        ready.append(path)
                        del self._pending[path]

            for path_str in ready:
                self._process(Path(path_str))

    def _process(self, path: Path) -> None:
        """Re-index a single changed file."""
        if not path.exists():
            # Deleted file — remove from index
            try:
                self._dispatcher.delete(path)
                log.info("FileWatcher: removed from index: %s", path.name)
            except Exception as exc:
                log.warning("FileWatcher: delete failed for %s: %s", path, exc)
            return

        try:
            self._dispatcher.ingest(path)
            log.info("FileWatcher: re-indexed: %s", path.name)
        except Exception as exc:
            log.warning("FileWatcher: ingest failed for %s: %s", path, exc)


# ---------------------------------------------------------------------------
# Watchdog event handler
# ---------------------------------------------------------------------------

try:
    from watchdog.events import FileSystemEventHandler as _Base  # type: ignore[import]
except ImportError:
    _Base = object  # type: ignore[assignment,misc]


class _ChangeHandler(_Base):
    """Minimal watchdog event handler that routes events to FileWatcher."""

    def __init__(self, callback) -> None:
        if _Base is not object:
            super().__init__()
        self._cb = callback

    def on_modified(self, event) -> None:
        if not event.is_directory:
            self._cb(event.src_path)

    def on_created(self, event) -> None:
        if not event.is_directory:
            self._cb(event.src_path)

    def on_moved(self, event) -> None:
        if not event.is_directory:
            # Old path → remove, new path → index
            self._cb(event.src_path)
            self._cb(event.dest_path)

    def on_deleted(self, event) -> None:
        if not event.is_directory:
            self._cb(event.src_path)
