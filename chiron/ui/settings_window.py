"""Dual-agent settings form: explicit Observer, Responder, and fixed capture."""

from __future__ import annotations

import asyncio
import logging
import math
import os
from typing import Any

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QKeyEvent
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QSlider,
    QSpinBox,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from chiron.capture.frames import describe_monitors
from chiron.config.settings import OverlaySettings, Settings, default_settings_path
from chiron.live.estimate import TOKENS_PER_FRAME
from chiron.models.catalogue import (
    STATIC_MODELS,
    available_models,
    find_model,
    observer_models,
    responder_models,
)
from chiron.models.providers import OPENROUTER
from chiron.nonlive.compaction import resolve_context_window
from chiron.ui.hotkeys import HotkeyError, normalise_hotkey
from chiron.ui.model_picker import ModelPicker, format_context
from chiron.ui.theme import PALETTE, settings_stylesheet

logger = logging.getLogger(__name__)

BATCHED_TOKENS_PER_FRAME = {"low": 66, "medium": 258, "high": 258}


def _version() -> str:
    try:
        from importlib.metadata import version

        return version("chiron")
    except Exception:
        return "dev"


class HotkeyEdit(QLineEdit):
    """Read-only field that records the next pressed key combination."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setReadOnly(True)
        self.setPlaceholderText("Click, then press a combination")

    def keyPressEvent(self, event: QKeyEvent) -> None:
        if event.key() in (
            Qt.Key.Key_Control,
            Qt.Key.Key_Alt,
            Qt.Key.Key_Shift,
            Qt.Key.Key_Meta,
        ):
            return
        if event.key() in (Qt.Key.Key_Backspace, Qt.Key.Key_Delete):
            self.setText("")
            self.textEdited.emit("")
            return
        from PySide6.QtGui import QKeySequence

        sequence = QKeySequence(event.keyCombination()).toString(
            QKeySequence.SequenceFormat.PortableText
        )
        try:
            spec = normalise_hotkey(sequence.lower().replace("meta+", "super+"))
        except HotkeyError:
            return
        self.setText(spec)
        self.textEdited.emit(spec)


class SettingsWindow(QWidget):
    """Edit a complete Settings copy and emit it only on Save."""

    settingsSaved = Signal(object)
    appearanceChanged = Signal(object)
    quitRequested = Signal()

    def __init__(self, settings: Settings, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.settings = settings.copy_deep()
        self._models: list[Any] = list(STATIC_MODELS)
        self._loading = False
        self._ready = False
        self.setWindowTitle("Chiron Settings")
        self.setStyleSheet(settings_stylesheet())
        self.resize(900, 680)
        self._build_ui()
        self._ready = True
        self.set_models(self._models)
        self.load(self.settings)

    # ---------------------------------------------------------------- build

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)
        body = QHBoxLayout()
        body.setContentsMargins(0, 0, 0, 0)
        body.setSpacing(0)
        root.addLayout(body, stretch=1)
        self.nav = QListWidget(self)
        self.nav.setObjectName("nav")
        self.nav.setFixedWidth(190)
        body.addWidget(self.nav)
        self.pages = QStackedWidget(self)
        body.addWidget(self.pages, stretch=1)
        for title, builder in (
            ("Agents", self._build_agents_page),
            ("Capture", self._build_capture_page),
            ("Journal", self._build_journal_page),
            ("Overlay", self._build_overlay_page),
            ("Hotkeys", self._build_hotkeys_page),
            ("Privacy", self._build_privacy_page),
            ("About", self._build_about_page),
        ):
            self.nav.addItem(QListWidgetItem(title))
            self.pages.addWidget(builder())
        self.nav.currentRowChanged.connect(self.pages.setCurrentIndex)
        self.nav.setCurrentRow(0)
        root.addWidget(self._button_bar())

    def _button_bar(self) -> QWidget:
        bar = QWidget(self)
        layout = QHBoxLayout(bar)
        layout.setContentsMargins(18, 12, 18, 14)
        quit_button = QPushButton("Quit Chiron")
        quit_button.clicked.connect(self.quitRequested.emit)
        layout.addWidget(quit_button)
        self.restart_badge = QLabel("")
        self.restart_badge.setObjectName("restartBadge")
        layout.addWidget(self.restart_badge)
        layout.addStretch(1)
        revert = QPushButton("Revert")
        revert.clicked.connect(lambda: self.load(self.settings))
        layout.addWidget(revert)
        close = QPushButton("Close")
        close.clicked.connect(self.close)
        layout.addWidget(close)
        self.save_button = QPushButton("Save")
        self.save_button.setObjectName("primary")
        self.save_button.clicked.connect(self._save)
        layout.addWidget(self.save_button)
        return bar

    def _page(self, title: str, blurb: str) -> tuple[QWidget, QVBoxLayout]:
        page = QWidget(self)
        outer = QVBoxLayout(page)
        outer.setContentsMargins(0, 0, 0, 0)
        scroll = QScrollArea(page)
        scroll.setWidgetResizable(True)
        outer.addWidget(scroll)
        content = QWidget(scroll)
        layout = QVBoxLayout(content)
        layout.setContentsMargins(28, 24, 28, 24)
        layout.setSpacing(10)
        heading = QLabel(title)
        heading.setObjectName("pageTitle")
        layout.addWidget(heading)
        subtitle = QLabel(blurb)
        subtitle.setObjectName("pageBlurb")
        subtitle.setWordWrap(True)
        layout.addWidget(subtitle)
        scroll.setWidget(content)
        return page, layout

    def _section(self, layout: QVBoxLayout, title: str) -> QFormLayout:
        label = QLabel(title.upper())
        label.setObjectName("sectionTitle")
        layout.addWidget(label)
        line = QFrame()
        line.setObjectName("separator")
        line.setFrameShape(QFrame.Shape.HLine)
        layout.addWidget(line)
        form = QFormLayout()
        form.setContentsMargins(0, 6, 0, 12)
        form.setSpacing(9)
        form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)
        layout.addLayout(form)
        return form

    @staticmethod
    def _hint(form: QFormLayout, text: str = "") -> QLabel:
        label = QLabel(text)
        label.setObjectName("hint")
        label.setWordWrap(True)
        form.addRow("", label)
        return label

    def _wire(self, widget: Any, signal: str) -> Any:
        getattr(widget, signal).connect(self._on_field_changed)
        return widget

    def _model_row(self, picker: ModelPicker) -> QHBoxLayout:
        row = QHBoxLayout()
        row.addWidget(picker, stretch=1)
        self.refresh_models_button = QPushButton("Refresh")
        self.refresh_models_button.clicked.connect(
            lambda: self.refresh_catalogue(force=True)
        )
        row.addWidget(self.refresh_models_button)
        return row

    def _build_agents_page(self) -> QWidget:
        page, layout = self._page(
            "Two agents",
            "Chiron-Observer writes the journal through Live checkpoints or "
            "buffered video reviews. Chiron-Responder alone answers the player.",
        )
        form = self._section(layout, "Credentials")
        self.api_key_edit = self._wire(QLineEdit(), "textEdited")
        self.api_key_edit.setEchoMode(QLineEdit.EchoMode.Password)
        self.api_key_edit.setPlaceholderText("AIza… (for Live/Gemini models)")
        form.addRow("Google AI Studio key", self.api_key_edit)
        self.key_source_label = self._hint(form)
        self._hint(
            form,
            "Required only when either selected agent uses Live or Google AI Studio. "
            "Environment fallbacks: GEMINI_API_KEY, GOOGLE_API_KEY.",
        )
        self.openrouter_key_edit = self._wire(QLineEdit(), "textEdited")
        self.openrouter_key_edit.setEchoMode(QLineEdit.EchoMode.Password)
        self.openrouter_key_edit.setPlaceholderText("sk-or-v1-… (optional)")
        form.addRow("OpenRouter key", self.openrouter_key_edit)
        self._hint(
            form,
            "Required when either selected agent uses OpenRouter. An OpenRouter-only "
            "installation does not need a Google key.",
        )

        form = self._section(layout, "Chiron-Observer")
        self.observer_model_picker = self._wire(ModelPicker(), "selectionChanged")
        form.addRow("Observer model", self._model_row(self.observer_model_picker))
        self.observer_note_label = self._hint(
            form, "Live and known video-capable batched models are offered."
        )
        self.observer_prompt_edit = self._wire(QPlainTextEdit(), "textChanged")
        self.observer_prompt_edit.setFixedHeight(70)
        self.observer_prompt_edit.setPlaceholderText(
            "Optional extra rules for durable observations."
        )
        form.addRow("Observer instructions", self.observer_prompt_edit)

        form = self._section(layout, "Chiron-Responder")
        self.responder_model_picker = self._wire(ModelPicker(), "selectionChanged")
        form.addRow("Responder model", self.responder_model_picker)
        self.responder_note_label = self._hint(form)
        self.responder_mode_combo = self._wire(QComboBox(), "currentIndexChanged")
        self.responder_mode_combo.addItem(
            "Fixed-horizon (low latency)", "fixed_horizon"
        )
        self.responder_mode_combo.addItem("ReAct (required journal tool)", "react")
        form.addRow("Responder mode", self.responder_mode_combo)
        self._hint(
            form,
            "ReAct exposes only read_journal and forces it once on every request. "
            "Known models without function calling are hidden in that mode.",
        )
        self.responder_reasoning_combo = self._wire(QComboBox(), "currentIndexChanged")
        self.responder_reasoning_combo.addItem("Provider default", None)
        for label, effort in (
            ("None", "none"),
            ("Minimal", "minimal"),
            ("Low", "low"),
            ("Medium", "medium"),
            ("High", "high"),
        ):
            self.responder_reasoning_combo.addItem(label, effort)
        form.addRow("Reasoning effort", self.responder_reasoning_combo)
        self._hint(
            form,
            "Supported levels vary by model and provider. Provider default is safest.",
        )
        self.game_name_edit = self._wire(QLineEdit(), "textEdited")
        self.game_name_edit.setPlaceholderText("e.g. Elden Ring; blank = detect window")
        form.addRow("Game", self.game_name_edit)
        self.responder_prompt_edit = self._wire(QPlainTextEdit(), "textChanged")
        self.responder_prompt_edit.setFixedHeight(85)
        self.responder_prompt_edit.setPlaceholderText(
            "e.g. I am playing blind; do not spoil undiscovered content."
        )
        form.addRow("Responder instructions", self.responder_prompt_edit)
        layout.addStretch(1)
        return page

    def _build_capture_page(self) -> QWidget:
        page, layout = self._page(
            "Fixed capture",
            "Every scheduled frame is an Observer opportunity. The client no "
            "longer guesses whether screen motion is interesting.",
        )
        form = self._section(layout, "Source")
        self.watch_on_launch_check = self._wire(
            QCheckBox("Start Watch on launch"), "toggled"
        )
        form.addRow("On launch", self.watch_on_launch_check)
        self.monitor_combo = self._wire(QComboBox(), "currentIndexChanged")
        labels = describe_monitors()
        for index, label in enumerate(
            labels or ["All monitors", "Monitor 1", "Monitor 2"]
        ):
            self.monitor_combo.addItem(label, index)
        form.addRow("Monitor", self.monitor_combo)

        form = self._section(layout, "Cadence")
        self.interval_spin = self._wire(QDoubleSpinBox(), "valueChanged")
        self.interval_spin.setRange(1.0, 60.0)
        self.interval_spin.setSingleStep(0.5)
        self.interval_spin.setSuffix(" s")
        form.addRow("Capture every", self.interval_spin)
        self._hint(form, "Every accepted capture is preserved for the Observer.")
        self.process_interval_spin = self._wire(QSpinBox(), "valueChanged")
        self.process_interval_spin.setRange(30, 1800)
        self.process_interval_spin.setSingleStep(30)
        self.process_interval_spin.setSuffix(" s")
        form.addRow("Process every", self.process_interval_spin)
        self.process_interval_hint = self._hint(
            form,
            "Batched Observer only. Watch off stops new captures but drains an "
            "already buffered final review.",
        )
        self.question_policy_combo = self._wire(QComboBox(), "currentIndexChanged")
        self.question_policy_combo.addItem("Latest scheduled frame", "latest")
        self.question_policy_combo.addItem("Immediate new frame", "immediate")
        form.addRow("Question screenshot", self.question_policy_combo)
        self._hint(
            form,
            "Immediate captures exactly one extra frame and does not move the next "
            "periodic deadline.",
        )
        self.question_answer_combo = self._wire(QComboBox(), "currentIndexChanged")
        self.question_answer_combo.addItem("Answer immediately", "immediate")
        self.question_answer_combo.addItem(
            "Flush Observer before answering", "flush_observer"
        )
        form.addRow("Question answer", self.question_answer_combo)
        self._hint(
            form,
            "Flush waits for the buffered Observer to review pending frames and "
            "commit its journal entries. It does not apply to Live observation.",
        )
        self.burn_label = self._hint(form)
        self.burn_label.setStyleSheet(f"color: {PALETTE['accent']};")

        form = self._section(layout, "Image")
        self.frame_width_spin = self._wire(QSpinBox(), "valueChanged")
        self.frame_width_spin.setRange(256, 1920)
        self.frame_width_spin.setSingleStep(64)
        self.frame_width_spin.setSuffix(" px")
        form.addRow("Capture width", self.frame_width_spin)
        self.media_res_combo = self._wire(QComboBox(), "currentTextChanged")
        self.media_res_combo.addItems(["low", "medium", "high"])
        form.addRow("Observer detail", self.media_res_combo)
        self._hint(
            form,
            "Width controls captured JPEGs. Detail controls Live visual budget or "
            "the batched time-lapse target (512/768/1152 px, never upscaled).",
        )
        self.jpeg_quality_spin = self._wire(QSpinBox(), "valueChanged")
        self.jpeg_quality_spin.setRange(10, 95)
        form.addRow("JPEG quality", self.jpeg_quality_spin)
        self.stamp_check = self._wire(QCheckBox("Stamp capture time"), "toggled")
        form.addRow("Timestamp", self.stamp_check)
        layout.addStretch(1)
        return page

    def _build_journal_page(self) -> QWidget:
        page, layout = self._page(
            "Observer journal",
            "One write path serves Observer tool calls, application events, the "
            "drawer, the recorder, and Responder read-only snapshots.",
        )
        form = self._section(layout, "Automatic memory")
        self.journal_observer_context_label = QLabel()
        self.journal_observer_context_label.setObjectName("hint")
        form.addRow("Observer context", self.journal_observer_context_label)
        self.journal_responder_context_label = QLabel()
        self.journal_responder_context_label.setObjectName("hint")
        form.addRow("Responder context", self.journal_responder_context_label)
        self._hint(
            form,
            "Every raw journal entry remains in the gameplay session. Chiron "
            "automatically summarizes only the model-facing copy when it approaches "
            "the selected model's token budget, while keeping recent entries verbatim.",
        )
        layout.addStretch(1)
        return page

    def _build_overlay_page(self) -> QWidget:
        page, layout = self._page("Overlay", "Appearance changes preview live.")
        form = self._section(layout, "Size and appearance")
        self.width_spin = self._wire(QSpinBox(), "valueChanged")
        self.width_spin.setRange(280, 1600)
        self.width_spin.setSuffix(" px")
        form.addRow("Width", self.width_spin)
        self.height_spin = self._wire(QSpinBox(), "valueChanged")
        self.height_spin.setRange(240, 1400)
        self.height_spin.setSuffix(" px")
        form.addRow("Height", self.height_spin)
        row = QHBoxLayout()
        self.opacity_slider = self._wire(
            QSlider(Qt.Orientation.Horizontal), "valueChanged"
        )
        self.opacity_slider.setRange(20, 100)
        row.addWidget(self.opacity_slider)
        self.opacity_label = QLabel()
        row.addWidget(self.opacity_label)
        form.addRow("Opacity", row)
        self.font_spin = self._wire(QSpinBox(), "valueChanged")
        self.font_spin.setRange(7, 24)
        self.font_spin.setSuffix(" pt")
        form.addRow("Font", self.font_spin)
        self.on_top_check = self._wire(QCheckBox("Keep above other windows"), "toggled")
        form.addRow("Stacking", self.on_top_check)
        self.start_hidden_check = self._wire(QCheckBox("Start hidden"), "toggled")
        form.addRow("On launch", self.start_hidden_check)
        layout.addStretch(1)
        return page

    def _build_hotkeys_page(self) -> QWidget:
        page, layout = self._page("Hotkeys", "Global X11 shortcuts.")
        form = self._section(layout, "Bindings")
        self.toggle_hotkey_edit = self._wire(HotkeyEdit(), "textEdited")
        form.addRow("Show / hide", self.toggle_hotkey_edit)
        self.settings_hotkey_edit = self._wire(HotkeyEdit(), "textEdited")
        form.addRow("Settings", self.settings_hotkey_edit)
        self.watch_toggle_hotkey_edit = self._wire(HotkeyEdit(), "textEdited")
        form.addRow("Toggle Watch", self.watch_toggle_hotkey_edit)
        self.start_watch_hotkey_edit = self._wire(HotkeyEdit(), "textEdited")
        form.addRow("Start Watch", self.start_watch_hotkey_edit)
        self.stop_watch_hotkey_edit = self._wire(HotkeyEdit(), "textEdited")
        form.addRow("Stop Watch", self.stop_watch_hotkey_edit)
        self.journal_hotkey_edit = self._wire(HotkeyEdit(), "textEdited")
        form.addRow("Journal", self.journal_hotkey_edit)
        self.hotkey_backend_label = self._hint(form)
        layout.addStretch(1)
        return page

    def _build_privacy_page(self) -> QWidget:
        page, layout = self._page(
            "Privacy",
            "Capture runs only when Watch is requested and the Gemini Live "
            "Observer is connected.",
        )
        for text in (
            "Observer loss stops capture immediately and clears cached frames.",
            "A stale journal-only answer never receives an image from an older Watch span.",
            "With an OpenRouter Responder, an immediate/current question screenshot "
            "is sent to both Google Live and OpenRouter.",
            "Gameplay sessions live in the data directory and survive --fresh-install.",
        ):
            label = QLabel(f"• {text}")
            label.setWordWrap(True)
            layout.addWidget(label)
        layout.addStretch(1)
        return page

    def _build_about_page(self) -> QWidget:
        page, layout = self._page("About", f"Chiron {_version()}")
        label = QLabel(
            f"Settings: {default_settings_path()}\n"
            "Linux/X11 only. Chiron-Observer uses Live or ephemeral in-memory "
            "batched video; Chiron-Responder uses Google AI Studio or OpenRouter. "
            "A crash can lose an unfinished in-memory Observer batch."
        )
        label.setWordWrap(True)
        layout.addWidget(label)
        layout.addStretch(1)
        return page

    # ------------------------------------------------------------ load/save

    def load(self, settings: Settings) -> None:
        self._loading = True
        try:
            self.api_key_edit.setText(settings.api_key)
            self.openrouter_key_edit.setText(settings.openrouter_api_key)
            self.observer_model_picker.set_selection(settings.observer_model)
            self.responder_model_picker.set_selection(settings.responder_model)
            self.responder_mode_combo.setCurrentIndex(
                max(0, self.responder_mode_combo.findData(settings.responder_mode))
            )
            self.responder_reasoning_combo.setCurrentIndex(
                max(
                    0,
                    self.responder_reasoning_combo.findData(
                        settings.responder_reasoning_effort
                    ),
                )
            )
            self.game_name_edit.setText(settings.game_name)
            self.observer_prompt_edit.setPlainText(settings.observer_system_prompt)
            self.responder_prompt_edit.setPlainText(settings.responder_system_prompt)
            capture = settings.capture
            self.watch_on_launch_check.setChecked(capture.watch_on_launch)
            self.monitor_combo.setCurrentIndex(
                max(0, self.monitor_combo.findData(capture.monitor_index))
            )
            self.interval_spin.setValue(capture.interval_seconds)
            self.process_interval_spin.setValue(round(capture.process_interval_seconds))
            self.question_policy_combo.setCurrentIndex(
                max(
                    0,
                    self.question_policy_combo.findData(capture.question_frame_policy),
                )
            )
            self.question_answer_combo.setCurrentIndex(
                max(
                    0,
                    self.question_answer_combo.findData(capture.question_answer_policy),
                )
            )
            self.frame_width_spin.setValue(capture.frame_width)
            self.media_res_combo.setCurrentText(capture.media_resolution)
            self.jpeg_quality_spin.setValue(capture.jpeg_quality)
            self.stamp_check.setChecked(capture.stamp_timestamp)
            overlay = settings.overlay
            self.width_spin.setValue(overlay.width)
            self.height_spin.setValue(overlay.height)
            self.opacity_slider.setValue(round(overlay.opacity * 100))
            self.font_spin.setValue(overlay.font_size)
            self.on_top_check.setChecked(overlay.always_on_top)
            self.start_hidden_check.setChecked(overlay.start_hidden)
            hotkeys = settings.hotkeys
            self.toggle_hotkey_edit.setText(hotkeys.toggle_overlay)
            self.settings_hotkey_edit.setText(hotkeys.open_settings)
            self.watch_toggle_hotkey_edit.setText(hotkeys.toggle_watching)
            self.start_watch_hotkey_edit.setText(hotkeys.start_watching)
            self.stop_watch_hotkey_edit.setText(hotkeys.stop_watching)
            self.journal_hotkey_edit.setText(hotkeys.toggle_journal)
        finally:
            self._loading = False
        self._refresh_derived()

    def collect(self) -> Settings:
        settings = self.settings.copy_deep()
        settings.api_key = self.api_key_edit.text().strip()
        settings.openrouter_api_key = self.openrouter_key_edit.text().strip()
        settings.observer_model = (
            self.observer_model_picker.selection() or settings.observer_model
        )
        settings.responder_model = (
            self.responder_model_picker.selection() or settings.responder_model
        )
        settings.responder_mode = self.responder_mode_combo.currentData()
        settings.responder_reasoning_effort = (
            self.responder_reasoning_combo.currentData()
        )
        settings.game_name = self.game_name_edit.text().strip()
        settings.observer_system_prompt = (
            self.observer_prompt_edit.toPlainText().strip()
        )
        settings.responder_system_prompt = (
            self.responder_prompt_edit.toPlainText().strip()
        )
        settings.capture.watch_on_launch = self.watch_on_launch_check.isChecked()
        monitor = self.monitor_combo.currentData()
        if monitor is not None:
            settings.capture.monitor_index = int(monitor)
        settings.capture.interval_seconds = self.interval_spin.value()
        settings.capture.process_interval_seconds = self.process_interval_spin.value()
        settings.capture.question_frame_policy = (
            self.question_policy_combo.currentData()
        )
        settings.capture.question_answer_policy = (
            self.question_answer_combo.currentData()
        )
        settings.capture.frame_width = self.frame_width_spin.value()
        settings.capture.media_resolution = self.media_res_combo.currentText()
        settings.capture.jpeg_quality = self.jpeg_quality_spin.value()
        settings.capture.stamp_timestamp = self.stamp_check.isChecked()
        settings.overlay.width = self.width_spin.value()
        settings.overlay.height = self.height_spin.value()
        settings.overlay.opacity = self.opacity_slider.value() / 100.0
        settings.overlay.font_size = self.font_spin.value()
        settings.overlay.always_on_top = self.on_top_check.isChecked()
        settings.overlay.start_hidden = self.start_hidden_check.isChecked()
        settings.hotkeys.toggle_overlay = self.toggle_hotkey_edit.text().strip()
        settings.hotkeys.open_settings = self.settings_hotkey_edit.text().strip()
        settings.hotkeys.toggle_watching = self.watch_toggle_hotkey_edit.text().strip()
        settings.hotkeys.start_watching = self.start_watch_hotkey_edit.text().strip()
        settings.hotkeys.stop_watching = self.stop_watch_hotkey_edit.text().strip()
        settings.hotkeys.toggle_journal = self.journal_hotkey_edit.text().strip()
        return settings

    def _save(self) -> None:
        try:
            settings = Settings.model_validate(self.collect().model_dump())
        except ValueError as error:
            self.restart_badge.setText(f"Cannot save: {error}")
            return
        missing = [
            role
            for role, model_id in (
                ("Observer", settings.observer_model),
                ("Responder", settings.responder_model),
            )
            if not settings.key_for_model(model_id)
        ]
        if missing:
            self.restart_badge.setText(
                "Cannot save: missing API key for " + " and ".join(missing) + "."
            )
            return
        self.settings = settings.copy_deep()
        self.settingsSaved.emit(settings)
        self._refresh_derived()

    # -------------------------------------------------------------- reactive

    def _on_field_changed(self, *_: Any) -> None:
        if self._loading or not self._ready:
            return
        self._refresh_derived()
        self.appearanceChanged.emit(self._overlay_preview())

    def _overlay_preview(self) -> OverlaySettings:
        preview = self.settings.overlay.model_copy()
        preview.width = self.width_spin.value()
        preview.height = self.height_spin.value()
        preview.opacity = self.opacity_slider.value() / 100.0
        preview.font_size = self.font_spin.value()
        preview.always_on_top = self.on_top_check.isChecked()
        return preview

    def _refresh_derived(self) -> None:
        if not self._ready:
            return
        edited = self.collect()
        self.opacity_label.setText(f"{self.opacity_slider.value()}%")
        source = edited.api_key_source()
        self.key_source_label.setText(
            "Using the key saved here."
            if source == "settings"
            else (
                "No key found; the Observer cannot connect."
                if source == "none"
                else f"Using {source} from the environment."
            )
        )
        interval = edited.capture.interval_seconds
        process_interval = edited.capture.process_interval_seconds
        batched = not edited.observer_model.startswith("live/")
        self.process_interval_spin.setEnabled(batched)
        self.process_interval_hint.setVisible(batched)
        token_table = BATCHED_TOKENS_PER_FRAME if batched else TOKENS_PER_FRAME
        tokens = token_table.get(edited.capture.media_resolution, 260)
        per_minute = (60.0 / interval) * tokens
        model = find_model(self._models, edited.observer_model)
        price = (
            f"Observer: ${model.prompt_price * 1_000_000:.2f}/M input, "
            f"${model.completion_price * 1_000_000:.2f}/M output."
            if model and model.pricing_known
            else "Observer pricing unavailable for this selection."
        )
        if batched:
            estimated_frames = math.ceil(process_interval / interval)
            cap_note = (
                " The 300-frame early seal applies." if estimated_frames > 300 else ""
            )
            observer_estimate = (
                f"Normal batch: ~{estimated_frames} frames, "
                f"{3600 / process_interval:.1f} calls/hour; about "
                f"{per_minute:,.0f} visual tokens/minute before prompt/output."
                f"{cap_note}"
            )
        else:
            observer_estimate = (
                f"Live cadence: about {per_minute:,.0f} visual tokens/minute "
                f"at {interval:g}s."
            )
        self.burn_label.setText(f"{observer_estimate} {price}")
        known = find_model(self._models, edited.responder_model)
        if known is None:
            note = (
                f"{edited.responder_model} is not in a catalogue; it will be used "
                "as typed and may reject images or tools."
            )
        elif edited.responder_mode == "react" and not known.supports_tools:
            note = "This known model does not support ReAct function calling."
        else:
            note = ""
        self.responder_note_label.setText(note)
        observer = find_model(self._models, edited.observer_model)
        if edited.observer_model.startswith("live/"):
            observer_note = (
                "Only record_event is exposed; content/audio output is discarded."
            )
        elif observer is None:
            observer_note = (
                "Video and structured output are unverified for this typed model id."
            )
        elif not observer.supports_video or not observer.supports_structured_output:
            observer_note = "This route does not advertise all batched capabilities."
        else:
            observer_note = "Batched: video input and structured output advertised."
        self.observer_note_label.setText(observer_note)
        observer_window = resolve_context_window(edited.observer_model)
        responder_window = resolve_context_window(edited.responder_model)
        self.journal_observer_context_label.setText(
            _context_window_text(observer_window)
        )
        self.journal_responder_context_label.setText(
            _context_window_text(responder_window)
        )
        notices = []
        if self.settings.requires_observer_reconnect(edited):
            notices.append("reconnect Observer")
        if self.settings.requires_responder_rebuild(edited):
            notices.append("rebuild Responder (conversation kept)")
        self.restart_badge.setText(
            "Saving will " + " and ".join(notices) + "." if notices else ""
        )
        # Re-filter known choices as the function-calling requirement changes.
        self._set_picker_models()

    # ------------------------------------------------------------- catalogue

    def _set_picker_models(self) -> None:
        loading, self._loading = self._loading, True
        try:
            self.observer_model_picker.set_models(observer_models(self._models))
            responders = responder_models(
                self._models, mode=self.responder_mode_combo.currentData()
            )
            has_openrouter_key = bool(
                self.openrouter_key_edit.text().strip()
                or os.environ.get("OPENROUTER_API_KEY", "").strip()
            )
            if not has_openrouter_key:
                responders = [
                    model for model in responders if model.provider != OPENROUTER
                ]
            self.responder_model_picker.set_models(responders)
        finally:
            self._loading = loading

    def set_models(self, models: list[Any]) -> None:
        self._models = list(models)
        self._set_picker_models()
        self._refresh_derived()

    def refresh_catalogue(self, *, force: bool = False) -> None:
        edited = self.collect()

        async def fetch() -> None:
            self.refresh_models_button.setEnabled(False)
            try:
                models = await asyncio.to_thread(
                    available_models,
                    google_key=edited.resolved_api_key(),
                    openrouter_key=edited.resolved_openrouter_key(),
                    force=force,
                )
            except Exception as error:  # noqa: BLE001
                logger.warning("Could not list models: %s", error)
                models = []
            finally:
                self.refresh_models_button.setEnabled(True)
            if models:
                self.set_models(models)

        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return
        asyncio.ensure_future(fetch())

    def set_detected_game(self, name: str) -> None:
        self.game_name_edit.setPlaceholderText(
            f"auto-detected: {name}" if name else "e.g. Elden Ring"
        )

    def set_hotkey_backend(self, backend: str) -> None:
        messages = {
            "xlib": "Using X11 key grabs (python-xlib).",
            "pynput": "Using pynput.",
            "none": "No global hotkey backend is available on this display server.",
        }
        self.hotkey_backend_label.setText(messages.get(backend, backend))


__all__ = ["HotkeyEdit", "ModelPicker", "SettingsWindow", "TOKENS_PER_FRAME"]


def _context_window_text(info: Any) -> str:
    suffix = " · assumed" if info.assumed else ""
    return f"{format_context(info.tokens)} · {info.source}{suffix}"
