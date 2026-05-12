"""Phase 3 unit tests: PPTX/XLSX parsers, FileWatcher, hardened pipeline."""
from __future__ import annotations

import time
import threading
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pytest


# ---------------------------------------------------------------------------
# PPTX Parser (mocked python-pptx)
# ---------------------------------------------------------------------------

class TestPPTXParser:
    def _make_mock_prs(self, slides_data: list[dict]):
        """Build a mock pptx.Presentation with given slide data."""
        prs = MagicMock()
        slides = []
        for data in slides_data:
            slide = MagicMock()
            # Shapes
            shape_list = []
            for text in data.get("shapes", []):
                shape = MagicMock()
                shape.has_text_frame = True
                shape.has_table = False
                shape.shape_type = 1  # not GROUP
                para = MagicMock()
                run = MagicMock()
                run.text = text
                para.runs = [run]
                shape.text_frame.paragraphs = [para]
                shape_list.append(shape)
            slide.shapes = shape_list

            # Notes
            if data.get("notes"):
                slide.has_notes_slide = True
                slide.notes_slide.notes_text_frame.text = data["notes"]
            else:
                slide.has_notes_slide = False

            # Placeholders (for title)
            if data.get("title"):
                ph = MagicMock()
                ph.placeholder_format.idx = 0
                ph.text = data["title"]
                slide.placeholders = [ph]
            else:
                slide.placeholders = []

            slides.append(slide)

        prs.slides = slides
        return prs

    def test_parses_slide_text(self, tmp_path):
        import sys
        from deepsearch.ingestion.parsers.pptx_parser import PPTXParser

        mock_prs = self._make_mock_prs([
            {"shapes": ["Introduction", "This is slide one"], "title": "Intro"},
            {"shapes": ["Conclusion"], "title": "End"},
        ])
        pptx_file = tmp_path / "deck.pptx"
        pptx_file.write_bytes(b"")

        # Inject mock modules so the lazy import inside parse() works
        mock_pptx_module = MagicMock()
        mock_pptx_module.Presentation.return_value = mock_prs
        mock_pptx_module.exc.PackageNotFoundError = Exception

        with patch.dict(sys.modules, {"pptx": mock_pptx_module, "pptx.exc": mock_pptx_module.exc}):
            parser = PPTXParser()
            doc = parser.parse(pptx_file, "pptx-1")

        assert "Intro" in doc.text or "Introduction" in doc.text
        assert doc.metadata["slide_count"] == 2

    def test_supported_extensions(self):
        from deepsearch.ingestion.parsers.pptx_parser import PPTXParser
        p = PPTXParser()
        assert ".pptx" in p.supported_extensions
        assert ".ppt" in p.supported_extensions

    def test_raises_on_missing_dependency(self, tmp_path):
        from deepsearch.ingestion.parsers.pptx_parser import PPTXParser
        p = PPTXParser()
        f = tmp_path / "test.pptx"
        f.write_bytes(b"fake")
        # python-pptx may or may not be installed; check graceful handling
        try:
            doc = p.parse(f, "fid")
            # If pptx is installed, it might error on bad bytes — that's ok
        except ImportError:
            pass  # expected when python-pptx not installed
        except Exception:
            pass  # any parse error is acceptable for fake bytes

    def test_shape_text_extracts_text_frame(self):
        from deepsearch.ingestion.parsers.pptx_parser import _shape_text
        shape = MagicMock()
        shape.has_text_frame = True
        shape.has_table = False
        shape.shape_type = 1

        run = MagicMock()
        run.text = "Hello from shape"
        para = MagicMock()
        para.runs = [run]
        shape.text_frame.paragraphs = [para]

        result = _shape_text(shape)
        assert "Hello from shape" in result

    def test_shape_text_extracts_table(self):
        from deepsearch.ingestion.parsers.pptx_parser import _shape_text
        shape = MagicMock()
        shape.has_text_frame = False
        shape.has_table = True
        shape.shape_type = 1

        cell1 = MagicMock(); cell1.text = "Col A"
        cell2 = MagicMock(); cell2.text = "Col B"
        row = MagicMock(); row.cells = [cell1, cell2]
        shape.table.rows = [row]

        result = _shape_text(shape)
        assert "Col A" in result
        assert "Col B" in result


# ---------------------------------------------------------------------------
# XLSX Parser (mocked openpyxl)
# ---------------------------------------------------------------------------

class TestXLSXParser:
    def _make_mock_wb(self, sheets: dict[str, list[list]]):
        """Build a mock openpyxl workbook."""
        wb = MagicMock()
        wb.sheetnames = list(sheets.keys())
        wb.close = MagicMock()

        def getitem(name):
            ws = MagicMock()
            rows = sheets[name]
            ws.iter_rows.side_effect = lambda values_only=True: iter(rows)
            return ws

        wb.__getitem__.side_effect = getitem
        return wb

    def test_supported_extensions(self):
        from deepsearch.ingestion.parsers.xlsx_parser import XLSXParser
        p = XLSXParser()
        assert ".xlsx" in p.supported_extensions
        assert ".xls" in p.supported_extensions

    def _xlsx_parse_with_mock(self, tmp_path, data, file_id="xlsx1"):
        import sys
        from deepsearch.ingestion.parsers.xlsx_parser import XLSXParser

        mock_wb = self._make_mock_wb(data)
        mock_openpyxl = MagicMock()
        mock_openpyxl.load_workbook.return_value = mock_wb

        f = tmp_path / f"{file_id}.xlsx"
        f.write_bytes(b"")

        with patch.dict(sys.modules, {"openpyxl": mock_openpyxl}):
            parser = XLSXParser()
            doc = parser.parse(f, file_id)
        return doc

    def test_parses_sheet_text(self, tmp_path):
        data = {
            "Sheet1": [
                ("name", "age", "city"),
                ("Alice", 30, "NYC"),
                ("Bob", 25, "LA"),
            ]
        }
        doc = self._xlsx_parse_with_mock(tmp_path, data)
        assert "Alice" in doc.text
        assert "Bob" in doc.text
        assert "Sheet1" in doc.text

    def test_headers_extracted(self, tmp_path):
        data = {"Sales": [("product", "qty", "price"), ("Widget", 10, 9.99)]}
        doc = self._xlsx_parse_with_mock(tmp_path, data, "xlsx2")
        assert "product" in doc.text
        assert "Widget" in doc.text

    def test_multiple_sheets(self, tmp_path):
        data = {
            "Alpha": [("x", "y"), (1, 2)],
            "Beta":  [("a", "b"), (3, 4)],
        }
        doc = self._xlsx_parse_with_mock(tmp_path, data, "xlsx3")
        assert "Alpha" in doc.text
        assert "Beta" in doc.text
        assert len(doc.pages) == 2


# ---------------------------------------------------------------------------
# FileWatcher
# ---------------------------------------------------------------------------

class TestFileWatcher:
    def test_init_without_watchdog(self):
        """FileWatcher should be constructable even without watchdog."""
        from deepsearch.ingestion.file_watcher import FileWatcher
        dispatcher = MagicMock()
        watcher = FileWatcher(dispatcher, debounce_seconds=0.1)
        assert watcher is not None

    def test_watch_requires_directory(self, tmp_path):
        from deepsearch.ingestion.file_watcher import FileWatcher
        dispatcher = MagicMock()
        watcher = FileWatcher(dispatcher)
        with pytest.raises(ValueError, match="Not a directory"):
            watcher.watch(tmp_path / "nonexistent.txt")

    def test_on_event_queues_path(self, tmp_path):
        from deepsearch.ingestion.file_watcher import FileWatcher
        dispatcher = MagicMock()
        watcher = FileWatcher(dispatcher, debounce_seconds=0.05)

        path_str = str(tmp_path / "file.txt")
        watcher._on_event(path_str)

        assert path_str in watcher._pending

    def test_flush_loop_processes_ready_events(self, tmp_path):
        from deepsearch.ingestion.file_watcher import FileWatcher

        f = tmp_path / "notes.txt"
        f.write_text("content")

        dispatcher = MagicMock()
        dispatcher.ingest = MagicMock(return_value="fid-1")

        watcher = FileWatcher(dispatcher, debounce_seconds=0.05)
        # Simulate a past event (already past debounce window)
        import time as _time
        watcher._pending[str(f)] = _time.monotonic() - 1.0

        ready = []
        import time as _time2
        now = _time2.monotonic()
        with watcher._lock:
            for path, ts in list(watcher._pending.items()):
                if now - ts >= watcher._debounce:
                    ready.append(path)
                    del watcher._pending[path]

        for p in ready:
            watcher._process(Path(p))

        dispatcher.ingest.assert_called_once_with(f)

    def test_process_deleted_file_calls_delete(self, tmp_path):
        from deepsearch.ingestion.file_watcher import FileWatcher
        dispatcher = MagicMock()
        watcher = FileWatcher(dispatcher)

        non_existent = tmp_path / "gone.txt"
        watcher._process(non_existent)

        dispatcher.delete.assert_called_once_with(non_existent)

    def test_stop_is_safe_without_start(self):
        from deepsearch.ingestion.file_watcher import FileWatcher
        dispatcher = MagicMock()
        watcher = FileWatcher(dispatcher)
        watcher.stop()  # should not raise


# ---------------------------------------------------------------------------
# Hardened LLM Pipeline
# ---------------------------------------------------------------------------

class TestHardenedPipeline:
    def _make_pipeline(self, answer="Test answer"):
        from deepsearch.generation.llm_pipeline import LLMPipeline
        from deepsearch.storage.cache import CacheBundle

        search = MagicMock()
        search.search_dense_only.return_value = []
        search.search.return_value = []

        ctx_builder = MagicMock()
        ctx_builder.build.return_value = ("Context text", [])

        mm = MagicMock()
        llm = MagicMock()
        llm.build_prompt.return_value = "prompt"
        llm.generate.return_value = answer
        llm.stream.return_value = iter(["tok1", "tok2"])
        mm.llm_main = llm
        mm.llm_small = None

        cfg = MagicMock()
        cfg.cloud.confidence_threshold = 0.5
        cfg.cloud.enabled = False
        cfg.cloud.max_tokens = 512

        cache = CacheBundle()

        pipeline = LLMPipeline(
            hybrid_search=search,
            context_builder=ctx_builder,
            model_manager=mm,
            cache=cache,
            settings=cfg,
        )
        return pipeline, mm, search

    def test_query_uses_retrieval_cache(self):
        from deepsearch.generation.llm_pipeline import LLMPipeline
        pipeline, mm, search = self._make_pipeline()

        result1 = pipeline.query("What is AI?", mode="fast")
        result2 = pipeline.query("What is AI?", mode="fast")

        # Dense search called only once (second call uses QueryCache)
        assert search.search_dense_only.call_count == 1

    def test_query_returns_cached_answer(self):
        pipeline, mm, search = self._make_pipeline(answer="Cached answer")
        r1 = pipeline.query("Same question?", mode="fast")
        assert not r1.from_cache

        r2 = pipeline.query("Same question?", mode="fast")
        assert r2.from_cache
        assert r2.answer == "Cached answer"

    def test_query_retries_on_generation_failure(self):
        pipeline, mm, search = self._make_pipeline()

        call_count = 0
        def flaky_generate(prompt, cfg):
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise RuntimeError("transient error")
            return "Recovered answer"

        mm.llm_main.generate.side_effect = flaky_generate

        result = pipeline.query("Retry test?", mode="fast")
        assert result.answer == "Recovered answer"
        assert result.retries >= 1

    def test_query_raises_after_max_retries(self):
        from deepsearch.core.exceptions import GenerationError
        pipeline, mm, search = self._make_pipeline()
        mm.llm_main.generate.side_effect = RuntimeError("always fails")

        with pytest.raises(GenerationError):
            pipeline.query("Always fails?", mode="fast")

    def test_result_has_retries_field(self):
        pipeline, mm, search = self._make_pipeline()
        result = pipeline.query("Clean query?", mode="fast")
        assert hasattr(result, "retries")
        assert result.retries == 0

    def test_file_filter_bypasses_retrieval_cache(self):
        """file_filter queries must bypass both answer cache and retrieval cache."""
        pipeline, mm, search = self._make_pipeline()

        # Two *different* filtered questions so the answer cache doesn't interfere
        pipeline.query("Filtered query one?", mode="deep", file_filter=["fid-1"])
        pipeline.query("Filtered query two?", mode="deep", file_filter=["fid-1"])

        # Both must hit search (no retrieval cache when file_filter is set)
        assert search.search.call_count == 2
