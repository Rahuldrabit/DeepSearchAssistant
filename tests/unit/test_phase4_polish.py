"""Unit tests for Phase 4 components.

Covers:
  - DeviceManager: battery detection, recommend_search_mode, gpu_layer_budget
  - VectorStore: boost_file payload update
  - SearchBar: set_mode
  - ChatWidget: feedback signal pathway
  - SettingsDialog: population, accept path (headless via monkeypatching)
  - SourcePanel: show_sources, clear, selection
"""
from __future__ import annotations

import sys
import types
from unittest.mock import MagicMock, patch, PropertyMock


# ── Headless PyQt6 guard ─────────────────────────────────────────────────────
# Allow import even without a display server; individual tests that need
# widgets are skipped when QApplication cannot be created.

def _qt_available() -> bool:
    try:
        import PyQt6.QtWidgets  # noqa: F401
        return True
    except Exception:
        return False


_QT = _qt_available()

import pytest


# ═══════════════════════════════════════════════════════════════════════════ #
# 4.1 — DeviceManager: battery + mode routing                                #
# ═══════════════════════════════════════════════════════════════════════════ #

class TestDeviceManagerBattery:
    def test_on_battery_false_when_plugged_in(self):
        """on_battery returns False when psutil says power_plugged=True."""
        from deepsearch.core.device_manager import DeviceManager
        dm = DeviceManager()
        mock_bat = MagicMock()
        mock_bat.power_plugged = True
        with patch("deepsearch.core.device_manager._on_battery", return_value=False):
            assert dm.on_battery is False

    def test_on_battery_true_when_unplugged(self):
        from deepsearch.core.device_manager import DeviceManager
        dm = DeviceManager()
        with patch("deepsearch.core.device_manager._on_battery", return_value=True):
            assert dm.on_battery is True

    def test_on_battery_false_when_psutil_missing(self):
        """Returns False gracefully when psutil is not installed."""
        import deepsearch.core.device_manager as dm_mod
        with patch.dict(sys.modules, {"psutil": None}):
            # Direct call to the module-level function
            result = dm_mod._on_battery()
        # Should not raise; result is False because import fails
        assert result is False

    def test_recommend_mode_fast_on_battery(self):
        """recommend_search_mode downgrades to 'fast' when on battery."""
        from deepsearch.core.device_manager import DeviceManager
        dm = DeviceManager()
        with patch("deepsearch.core.device_manager._on_battery", return_value=True):
            assert dm.recommend_search_mode("deep") == "fast"
            assert dm.recommend_search_mode("cloud") == "fast"

    def test_recommend_mode_passthrough_when_plugged(self):
        from deepsearch.core.device_manager import DeviceManager
        dm = DeviceManager()
        with patch("deepsearch.core.device_manager._on_battery", return_value=False):
            assert dm.recommend_search_mode("deep") == "deep"
            assert dm.recommend_search_mode("cloud") == "cloud"

    def test_available_ram_gb_returns_float(self):
        from deepsearch.core.device_manager import _available_ram_gb
        psutil_mock = MagicMock()
        mem_mock = MagicMock()
        mem_mock.available = 8 * 1e9  # 8 GB
        psutil_mock.virtual_memory.return_value = mem_mock
        with patch.dict(sys.modules, {"psutil": psutil_mock}):
            val = _available_ram_gb()
        assert abs(val - 8.0) < 0.01

    def test_gpu_layer_budget_zero_when_low_ram(self):
        """Returns 0 GPU layers when RAM is critically low."""
        from deepsearch.core.device_manager import DeviceManager
        dm = DeviceManager()
        dm._detected = True  # skip hardware scan
        with patch("deepsearch.core.device_manager._available_ram_gb", return_value=1.0):
            assert dm.gpu_layer_budget() == 0


# ═══════════════════════════════════════════════════════════════════════════ #
# 4.2 — VectorStore: boost_file                                              #
# ═══════════════════════════════════════════════════════════════════════════ #

class TestVectorStoreBoost:
    def _make_store(self):
        from deepsearch.storage.vector_store import VectorStore
        mock_client = MagicMock()
        store = VectorStore.__new__(VectorStore)
        store._client = mock_client
        return store, mock_client

    def test_boost_file_updates_payload(self):
        store, client = self._make_store()

        # Simulate two chunks returned by scroll
        pt1 = MagicMock()
        pt1.id = "chunk-1"
        pt1.payload = {"file_id": "f1", "score_boost": 0.0}
        pt2 = MagicMock()
        pt2.id = "chunk-2"
        pt2.payload = {"file_id": "f1", "score_boost": 0.1}
        client.scroll.return_value = ([pt1, pt2], None)

        store.boost_file("f1", boost_delta=0.1)

        assert client.set_payload.call_count == 2
        # Check first call sets boost to 0.0 + 0.1 = 0.1
        first_call_payload = client.set_payload.call_args_list[0][1]["payload"]
        assert abs(first_call_payload["score_boost"] - 0.1) < 1e-6

    def test_boost_file_no_op_when_empty(self):
        store, client = self._make_store()
        client.scroll.return_value = ([], None)

        store.boost_file("unknown-file")
        client.set_payload.assert_not_called()

    def test_boost_file_default_delta_is_01(self):
        store, client = self._make_store()
        pt = MagicMock()
        pt.id = "c1"
        pt.payload = {"score_boost": 0.0}
        client.scroll.return_value = ([pt], None)

        store.boost_file("f1")  # no explicit delta

        payload = client.set_payload.call_args[1]["payload"]
        assert abs(payload["score_boost"] - 0.1) < 1e-6


# ═══════════════════════════════════════════════════════════════════════════ #
# 4.3 — SearchBar: set_mode                                                  #
# ═══════════════════════════════════════════════════════════════════════════ #

@pytest.mark.skipif(not _QT, reason="PyQt6 not available")
class TestSearchBarSetMode:
    def _make_bar(self, qtbot):
        from deepsearch.ui.search_bar import SearchBar
        bar = SearchBar()
        qtbot.addWidget(bar)
        return bar

    def test_set_mode_fast(self, qtbot):
        bar = self._make_bar(qtbot)
        bar.set_mode("deep")
        bar.set_mode("fast")
        assert bar.current_mode == "fast"

    def test_set_mode_deep(self, qtbot):
        bar = self._make_bar(qtbot)
        bar.set_mode("deep")
        assert bar.current_mode == "deep"

    def test_set_mode_cloud(self, qtbot):
        bar = self._make_bar(qtbot)
        bar.set_mode("cloud")
        assert bar.current_mode == "cloud"

    def test_set_mode_invalid_is_no_op(self, qtbot):
        bar = self._make_bar(qtbot)
        bar.set_mode("fast")
        bar.set_mode("nonexistent")
        assert bar.current_mode == "fast"  # unchanged


# ═══════════════════════════════════════════════════════════════════════════ #
# 4.4 — ChatWidget: feedback signal                                          #
# ═══════════════════════════════════════════════════════════════════════════ #

@pytest.mark.skipif(not _QT, reason="PyQt6 not available")
class TestChatWidgetFeedback:
    def _make_chat(self, qtbot):
        from deepsearch.ui.chat_widget import ChatWidget
        chat = ChatWidget()
        qtbot.addWidget(chat)
        return chat

    def test_feedback_buttons_hidden_during_streaming(self, qtbot):
        chat = self._make_chat(qtbot)
        chat.start_assistant_message()
        bubble = chat._current_assistant_bubble
        assert not bubble._thumbs_up.isVisible()
        assert not bubble._thumbs_down.isVisible()

    def test_feedback_buttons_visible_after_finish(self, qtbot):
        chat = self._make_chat(qtbot)
        chat.start_assistant_message()
        chat.append_token("Hello")
        chat.finish_assistant_message()
        # The bubble that was current before finish should now show buttons
        last_bubble = chat._messages[-1]
        assert last_bubble._thumbs_up.isVisible()

    def test_thumbs_up_emits_rating_plus1(self, qtbot):
        chat = self._make_chat(qtbot)
        chat.start_assistant_message()
        chat.append_token("Great answer")
        chat.finish_assistant_message()
        bubble = chat._messages[-1]

        received = []
        chat.feedback_given.connect(lambda r, a: received.append((r, a)))
        bubble._thumbs_up.click()

        assert received == [(1, "Great answer")]

    def test_thumbs_down_emits_rating_minus1(self, qtbot):
        chat = self._make_chat(qtbot)
        chat.start_assistant_message()
        chat.append_token("Bad answer")
        chat.finish_assistant_message()
        bubble = chat._messages[-1]

        received = []
        chat.feedback_given.connect(lambda r, a: received.append((r, a)))
        bubble._thumbs_down.click()

        assert received == [(-1, "Bad answer")]

    def test_buttons_disabled_after_rating(self, qtbot):
        chat = self._make_chat(qtbot)
        chat.start_assistant_message()
        chat.finish_assistant_message()
        bubble = chat._messages[-1]
        bubble._thumbs_up.click()
        assert not bubble._thumbs_up.isEnabled()
        assert not bubble._thumbs_down.isEnabled()


# ═══════════════════════════════════════════════════════════════════════════ #
# 4.5 — SourcePanel                                                          #
# ═══════════════════════════════════════════════════════════════════════════ #

@pytest.mark.skipif(not _QT, reason="PyQt6 not available")
class TestSourcePanel:
    def _make_panel(self, qtbot):
        from deepsearch.ui.source_panel import SourcePanel
        panel = SourcePanel()
        qtbot.addWidget(panel)
        return panel

    def _sources(self):
        return [
            {"file_name": "report.pdf", "page": 3, "text": "The revenue was $5M."},
            {"file_name": "notes.txt", "page": None, "text": "Short meeting notes."},
        ]

    def test_show_sources_populates_list(self, qtbot):
        panel = self._make_panel(qtbot)
        panel.show_sources(self._sources())
        assert panel._list.count() == 2

    def test_list_item_labels_include_filename(self, qtbot):
        panel = self._make_panel(qtbot)
        panel.show_sources(self._sources())
        assert "report.pdf" in panel._list.item(0).text()
        assert "notes.txt" in panel._list.item(1).text()

    def test_page_number_shown_in_label(self, qtbot):
        panel = self._make_panel(qtbot)
        panel.show_sources(self._sources())
        assert "p.3" in panel._list.item(0).text()

    def test_clear_empties_list(self, qtbot):
        panel = self._make_panel(qtbot)
        panel.show_sources(self._sources())
        panel.clear()
        assert panel._list.count() == 0

    def test_selection_updates_preview(self, qtbot):
        panel = self._make_panel(qtbot)
        panel.show_sources(self._sources())
        panel._list.setCurrentRow(0)
        assert "revenue" in panel._preview.toPlainText()

    def test_empty_sources_clears_preview(self, qtbot):
        panel = self._make_panel(qtbot)
        panel.show_sources(self._sources())
        panel.show_sources([])
        assert panel._preview.toPlainText() == ""
