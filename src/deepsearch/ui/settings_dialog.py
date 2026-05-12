"""Settings dialog for configuring DeepSearch Assistant.

Sections:
  - Models    : main LLM / small LLM selection, RAM budget slider
  - Folders   : list of watched / indexed directories
  - Cloud     : API key + confidence threshold
  - Appearance: theme toggle
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from PyQt6.QtCore import Qt, pyqtSignal  # type: ignore[import]
from PyQt6.QtWidgets import (  # type: ignore[import]
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QPushButton,
    QSlider,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from ..core.config import Settings

log = logging.getLogger(__name__)

# Available GGUF model presets (name → HF repo slug for display only)
_MAIN_LLM_OPTIONS = [
    "Qwen2.5-7B-Instruct-Q4_K_M",
    "Phi-3.5-mini-instruct-Q4_K_M",
    "Qwen2.5-3B-Instruct-Q4_K_M",
    "custom…",
]

_SMALL_LLM_OPTIONS = [
    "Phi-3.5-mini-instruct-Q4_K_M",
    "Qwen2.5-3B-Instruct-Q4_K_M",
    "none (disable fast mode)",
    "custom…",
]


class SettingsDialog(QDialog):
    """Modal settings dialog.  Emits ``settings_saved`` with the updated
    :class:`Settings` when the user clicks OK."""

    settings_saved = pyqtSignal(object)  # Settings

    def __init__(self, settings: Settings, parent=None) -> None:
        super().__init__(parent)
        self._cfg = settings
        self.setWindowTitle("Settings")
        self.setMinimumWidth(520)
        self.setMinimumHeight(420)

        self._build_ui()
        self._populate()

    # ------------------------------------------------------------------ #
    # UI construction                                                      #
    # ------------------------------------------------------------------ #

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)

        self._tabs = QTabWidget()
        self._tabs.addTab(self._make_models_tab(), "Models")
        self._tabs.addTab(self._make_folders_tab(), "Folders")
        self._tabs.addTab(self._make_cloud_tab(), "Cloud")
        self._tabs.addTab(self._make_appearance_tab(), "Appearance")
        layout.addWidget(self._tabs)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self._on_accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _make_models_tab(self) -> QWidget:
        w = QWidget()
        form = QFormLayout(w)
        form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)

        self._main_llm_combo = QComboBox()
        self._main_llm_combo.addItems(_MAIN_LLM_OPTIONS)
        self._main_llm_combo.currentTextChanged.connect(self._on_main_llm_changed)
        form.addRow("Main LLM:", self._main_llm_combo)

        self._main_llm_path = QLineEdit()
        self._main_llm_path.setPlaceholderText("Path to .gguf file…")
        self._main_llm_path.setVisible(False)
        self._main_llm_browse = QPushButton("Browse…")
        self._main_llm_browse.setVisible(False)
        self._main_llm_browse.clicked.connect(
            lambda: self._browse_model(self._main_llm_path)
        )
        main_row = QHBoxLayout()
        main_row.addWidget(self._main_llm_path)
        main_row.addWidget(self._main_llm_browse)
        form.addRow("", main_row)

        self._small_llm_combo = QComboBox()
        self._small_llm_combo.addItems(_SMALL_LLM_OPTIONS)
        self._small_llm_combo.currentTextChanged.connect(self._on_small_llm_changed)
        form.addRow("Small LLM:", self._small_llm_combo)

        self._small_llm_path = QLineEdit()
        self._small_llm_path.setPlaceholderText("Path to .gguf file…")
        self._small_llm_path.setVisible(False)
        self._small_llm_browse = QPushButton("Browse…")
        self._small_llm_browse.setVisible(False)
        self._small_llm_browse.clicked.connect(
            lambda: self._browse_model(self._small_llm_path)
        )
        small_row = QHBoxLayout()
        small_row.addWidget(self._small_llm_path)
        small_row.addWidget(self._small_llm_browse)
        form.addRow("", small_row)

        # RAM budget slider (4–32 GB in 1 GB steps)
        self._ram_slider = QSlider(Qt.Orientation.Horizontal)
        self._ram_slider.setMinimum(4)
        self._ram_slider.setMaximum(32)
        self._ram_slider.setSingleStep(1)
        self._ram_slider.setTickInterval(4)
        self._ram_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self._ram_label = QLabel("8 GB")
        self._ram_slider.valueChanged.connect(
            lambda v: self._ram_label.setText(f"{v} GB")
        )
        ram_row = QHBoxLayout()
        ram_row.addWidget(self._ram_slider)
        ram_row.addWidget(self._ram_label)
        form.addRow("RAM budget:", ram_row)

        return w

    def _make_folders_tab(self) -> QWidget:
        w = QWidget()
        layout = QVBoxLayout(w)

        layout.addWidget(QLabel("Indexed / watched folders:"))
        self._folders_list = QListWidget()
        layout.addWidget(self._folders_list)

        btn_row = QHBoxLayout()
        add_btn = QPushButton("Add folder…")
        add_btn.clicked.connect(self._add_folder)
        remove_btn = QPushButton("Remove selected")
        remove_btn.clicked.connect(self._remove_folder)
        btn_row.addWidget(add_btn)
        btn_row.addWidget(remove_btn)
        btn_row.addStretch()
        layout.addLayout(btn_row)

        return w

    def _make_cloud_tab(self) -> QWidget:
        w = QWidget()
        form = QFormLayout(w)

        self._cloud_enabled = QCheckBox("Enable cloud LLM fallback")
        form.addRow(self._cloud_enabled)

        self._cloud_api_key = QLineEdit()
        self._cloud_api_key.setEchoMode(QLineEdit.EchoMode.Password)
        self._cloud_api_key.setPlaceholderText("sk-… (stored in memory only)")
        form.addRow("API Key:", self._cloud_api_key)

        self._cloud_threshold = QDoubleSpinBox()
        self._cloud_threshold.setMinimum(0.0)
        self._cloud_threshold.setMaximum(1.0)
        self._cloud_threshold.setSingleStep(0.05)
        self._cloud_threshold.setDecimals(2)
        form.addRow("Confidence threshold:", self._cloud_threshold)

        self._cloud_model = QLineEdit()
        self._cloud_model.setPlaceholderText("e.g. claude-sonnet-4-6 or gpt-4o")
        form.addRow("Cloud model:", self._cloud_model)

        return w

    def _make_appearance_tab(self) -> QWidget:
        w = QWidget()
        form = QFormLayout(w)

        self._dark_theme = QCheckBox("Use dark theme")
        form.addRow(self._dark_theme)

        self._streaming = QCheckBox("Stream tokens (shows answer word-by-word)")
        form.addRow(self._streaming)

        return w

    # ------------------------------------------------------------------ #
    # Populate from current settings                                       #
    # ------------------------------------------------------------------ #

    def _populate(self) -> None:
        # Models
        self._ram_slider.setValue(
            getattr(self._cfg.resources, "ram_budget_gb", 8)
        )

        # Cloud
        self._cloud_enabled.setChecked(self._cfg.cloud.enabled)
        self._cloud_api_key.setText(getattr(self._cfg, "cloud_api_key", "") or "")
        self._cloud_threshold.setValue(self._cfg.cloud.confidence_threshold)
        self._cloud_model.setText(self._cfg.cloud.model or "")

        # Appearance
        self._streaming.setChecked(getattr(self._cfg.ui, "streaming", True))

    # ------------------------------------------------------------------ #
    # Slot helpers                                                         #
    # ------------------------------------------------------------------ #

    def _on_main_llm_changed(self, text: str) -> None:
        custom = text == "custom…"
        self._main_llm_path.setVisible(custom)
        self._main_llm_browse.setVisible(custom)

    def _on_small_llm_changed(self, text: str) -> None:
        custom = text == "custom…"
        self._small_llm_path.setVisible(custom)
        self._small_llm_browse.setVisible(custom)

    def _browse_model(self, line_edit: QLineEdit) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Select GGUF model", "", "GGUF models (*.gguf);;All files (*)"
        )
        if path:
            line_edit.setText(path)

    def _add_folder(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Select folder to index")
        if folder and not self._folder_already_listed(folder):
            self._folders_list.addItem(folder)

    def _remove_folder(self) -> None:
        for item in self._folders_list.selectedItems():
            self._folders_list.takeItem(self._folders_list.row(item))

    def _folder_already_listed(self, folder: str) -> bool:
        for i in range(self._folders_list.count()):
            if self._folders_list.item(i).text() == folder:
                return True
        return False

    # ------------------------------------------------------------------ #
    # Accept / save                                                        #
    # ------------------------------------------------------------------ #

    def _on_accept(self) -> None:
        # Apply cloud settings (mutable attrs)
        self._cfg.cloud.enabled = self._cloud_enabled.isChecked()
        self._cfg.cloud.confidence_threshold = self._cloud_threshold.value()
        if self._cloud_model.text().strip():
            self._cfg.cloud.model = self._cloud_model.text().strip()

        # API key — stored on Settings directly (not persisted to disk)
        api_key = self._cloud_api_key.text().strip()
        if api_key:
            object.__setattr__(self._cfg, "cloud_api_key", api_key)

        # RAM budget
        if hasattr(self._cfg, "resources"):
            object.__setattr__(
                self._cfg.resources, "ram_budget_gb", self._ram_slider.value()
            )

        # Streaming
        if hasattr(self._cfg, "ui"):
            object.__setattr__(self._cfg.ui, "streaming", self._streaming.isChecked())

        self.settings_saved.emit(self._cfg)
        self.accept()

    # ------------------------------------------------------------------ #
    # Accessors for tests                                                  #
    # ------------------------------------------------------------------ #

    @property
    def dark_theme_checked(self) -> bool:
        return self._dark_theme.isChecked()

    @property
    def cloud_enabled_checked(self) -> bool:
        return self._cloud_enabled.isChecked()

    @property
    def ram_budget(self) -> int:
        return self._ram_slider.value()

    def listed_folders(self) -> list[str]:
        return [self._folders_list.item(i).text() for i in range(self._folders_list.count())]
