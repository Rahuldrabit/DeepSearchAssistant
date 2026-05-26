"""QThread workers — keep all blocking operations off the main thread."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

from PyQt6.QtCore import QThread, pyqtSignal  # type: ignore[import]

from ..generation.llm_pipeline import LLMPipeline, SearchResult
from ..ingestion.dispatcher import IngestionDispatcher


class QueryWorker(QThread):
    """Runs LLMPipeline.query() in a background thread.

    Signals:
        token_received(str) — emitted for each streamed token
        result_ready(SearchResult) — emitted when generation is complete
        error(str) — emitted on exception
    """

    token_received = pyqtSignal(str)
    result_ready = pyqtSignal(object)  # SearchResult
    error = pyqtSignal(str)

    def __init__(
        self,
        pipeline: LLMPipeline,
        question: str,
        mode: str = "fast",
        stream: bool = True,
        file_filter: Optional[list[str]] = None,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._pipeline = pipeline
        self._question = question
        self._mode = mode
        self._stream = stream
        self._file_filter = file_filter

    def run(self) -> None:
        try:
            if self._stream:
                full_answer = ""
                for token in self._pipeline.stream_query(
                    self._question, mode=self._mode, file_filter=self._file_filter
                ):
                    full_answer += token
                    self.token_received.emit(token)
                # Emit a minimal result object so callers can save feedback
                result = SearchResult(
                    answer=full_answer,
                    search_mode=self._mode,
                    sources=getattr(self._pipeline, "_last_sources", []) or [],
                )
                self.result_ready.emit(result)
            else:
                result = self._pipeline.query(
                    self._question, mode=self._mode, file_filter=self._file_filter
                )
                self.result_ready.emit(result)
        except Exception as exc:
            self.error.emit(str(exc))


class IndexWorker(QThread):
    """Runs IngestionDispatcher.ingest() or ingest_directory() in background.

    Signals:
        progress(str) — human-readable progress message
        file_done(str, str) — (file_path, file_id) when a file is indexed
        finished(int) — total files indexed count
        error(str) — emitted on exception
    """

    progress = pyqtSignal(str)
    file_done = pyqtSignal(str, str)
    finished = pyqtSignal(int)
    error = pyqtSignal(str)

    def __init__(
        self,
        dispatcher: IngestionDispatcher,
        paths: list[Path],
        recursive: bool = True,
        max_workers: int = 1,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._dispatcher = dispatcher
        self._paths = paths
        self._recursive = recursive
        self._max_workers = max(1, int(max_workers))

    def _expand_paths(self) -> list[Path]:
        """Expand directories into supported files; de-duplicate paths."""
        supported = self._dispatcher.supported_extensions()
        pattern = "**/*" if self._recursive else "*"

        files: list[Path] = []
        seen: set[Path] = set()

        for p in self._paths:
            try:
                p = p.resolve()
            except Exception:
                # resolve() can fail on some broken symlinks; keep original
                pass

            if p.is_dir():
                for fp in p.glob(pattern):
                    if not fp.is_file():
                        continue
                    if fp.suffix.lower() not in supported:
                        continue
                    rp = fp.resolve()
                    if rp not in seen:
                        seen.add(rp)
                        files.append(rp)
            else:
                # Keep prior UX: if the user explicitly picked a path,
                # try to ingest it (missing/unsupported will raise a clear error).
                try:
                    rp = p.resolve()
                except Exception:
                    rp = p
                if rp not in seen:
                    seen.add(rp)
                    files.append(rp)

        return files

    def run(self) -> None:
        try:
            files = self._expand_paths()
            if not files:
                self.finished.emit(0)
                return

            count = 0
            if self._max_workers <= 1 or len(files) == 1:
                for path in files:
                    self.progress.emit(f"Indexing {path.name}…")
                    fid = self._dispatcher.ingest(path)
                    self.file_done.emit(str(path), fid)
                    count += 1
                self.finished.emit(count)
                return

            self.progress.emit(
                f"Indexing {len(files)} file(s) ({self._max_workers} workers)…"
            )

            with ThreadPoolExecutor(max_workers=self._max_workers) as ex:
                future_to_path = {
                    ex.submit(self._dispatcher.ingest, path): path
                    for path in files
                }

                for fut in as_completed(future_to_path):
                    path = future_to_path[fut]
                    try:
                        fid = fut.result()
                    except Exception:
                        # Best-effort cancellation of pending tasks
                        for f in future_to_path:
                            f.cancel()
                        raise
                    self.file_done.emit(str(path), fid)
                    count += 1
                    self.progress.emit(f"Indexed {count}/{len(files)}: {path.name}")

            self.finished.emit(count)
        except Exception as exc:
            self.error.emit(str(exc))


class ModelLoadWorker(QThread):
    """Loads main LLM in background on first use.

    Signals:
        progress(str) — status message
        loaded() — model is ready
        error(str) — load failed
    """

    progress = pyqtSignal(str)
    loaded = pyqtSignal()
    error = pyqtSignal(str)

    def __init__(self, model_manager, parent=None) -> None:
        super().__init__(parent)
        self._mm = model_manager

    def run(self) -> None:
        try:
            self.progress.emit("Loading language model…")
            self._mm.ensure_llm_main()
            self.loaded.emit()
        except Exception as exc:
            self.error.emit(str(exc))
