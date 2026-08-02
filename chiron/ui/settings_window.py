"""The settings page: everything Chiron can be told, in one window.

This is an ordinary window rather than part of the overlay — it is read and
edited between sessions, not glanced at mid-fight, so it can afford space, a
sidebar and full-length explanations. The overlay stays a chat panel.

Editing works on a copy. Widgets are populated from a :class:`Settings` snapshot,
:meth:`SettingsWindow.collect` reads them back into a brand new one, and only
**Save** hands that object to the application. Nothing is half-applied, so there
is no state where the capture thread has the new frame rate but the session still
has the old model.

Appearance is the deliberate exception: opacity, size and font apply live while
you drag the slider, because choosing an overlay's transparency blind is a
guessing game. Those changes still are not persisted until Save.

Two things are surfaced rather than hidden, because both are load-bearing and
both are invisible in a chat window otherwise: the estimated token burn implied
by the capture settings, and which journal strategy is in force with an honest
account of what each one costs you.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QKeyEvent
from PySide6.QtWidgets import (
    QButtonGroup,
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
    QRadioButton,
    QScrollArea,
    QSlider,
    QSpinBox,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from chiron.capture.frames import describe_monitors
from chiron.config.settings import (
    LIVE_CONTEXT_TOKENS,
    LIVE_MODEL_CHOICES,
    SIDECAR_MODEL_CHOICES,
    OverlaySettings,
    Settings,
    default_settings_path,
)
from chiron.ui.hotkeys import HotkeyError, normalise_hotkey
from chiron.ui.theme import PALETTE, settings_stylesheet

logger = logging.getLogger(__name__)

#: Tokens one frame costs at each media resolution, from the API documentation.
#: Used only for the burn estimate shown on the capture page.
TOKENS_PER_FRAME = {"low": 260, "medium": 560, "high": 1120}


def _version() -> str:
    """Chiron's installed version, or ``dev`` when it cannot be determined."""
    try:
        from importlib.metadata import version

        return version("chiron")
    except Exception:
        return "dev"


class HotkeyEdit(QLineEdit):
    """A field that records the next key combination pressed into it.

    Typing a hotkey by hand invites typos that only show up as "the shortcut does
    nothing"; pressing the combination cannot be spelt wrong.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setReadOnly(True)
        self.setPlaceholderText("Click, then press a combination")

    def keyPressEvent(self, event: QKeyEvent) -> None:
        """Record the pressed combination, ignoring modifier-only presses."""
        key = event.key()
        modifier_keys = (
            Qt.Key.Key_Control,
            Qt.Key.Key_Alt,
            Qt.Key.Key_Shift,
            Qt.Key.Key_Meta,
        )
        if key in modifier_keys:
            return
        if key in (Qt.Key.Key_Backspace, Qt.Key.Key_Delete):
            self.setText("")
            self.textEdited.emit("")
            return
        from PySide6.QtGui import QKeySequence

        sequence = QKeySequence(event.keyCombination()).toString(
            QKeySequence.SequenceFormat.PortableText
        )
        spec = sequence.lower().replace("meta+", "super+")
        try:
            spec = normalise_hotkey(spec)
        except HotkeyError:
            return
        self.setText(spec)
        self.textEdited.emit(spec)


class SettingsWindow(QWidget):
    """The dedicated settings page.

    Signals:
        settingsSaved (object): A complete new :class:`Settings` was saved.
        appearanceChanged (object): Live preview of :class:`OverlaySettings`
            while the overlay page is being edited.
        quitRequested (): The user asked to close Chiron entirely.

    Attributes:
        settings (Settings): The last saved snapshot this window was given.
    """

    settingsSaved = Signal(object)
    appearanceChanged = Signal(object)
    quitRequested = Signal()

    def __init__(self, settings: Settings, parent: QWidget | None = None) -> None:
        """Build the window and populate it from `settings`."""
        super().__init__(parent)
        self.settings = settings.copy_deep()
        self._loading = False
        # Populating a combo box emits its change signal, so fields on pages that
        # do not exist yet would be read while the window is still being built.
        # Nothing reacts until every page is in place.
        self._ready = False
        self._widgets: dict[str, Any] = {}

        self.setWindowTitle("Chiron — Settings")
        self.setStyleSheet(settings_stylesheet())
        self.resize(880, 640)
        self._build_ui()
        self._ready = True
        self.load(self.settings)

    # ----------------------------------------------------------------- build

    def _build_ui(self) -> None:
        """Assemble the sidebar, the pages and the button bar."""
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
            ("Session", self._build_session_page),
            ("Capture", self._build_capture_page),
            ("Journal", self._build_journal_page),
            ("Overlay", self._build_overlay_page),
            ("Hotkeys", self._build_hotkeys_page),
            ("About", self._build_about_page),
        ):
            self.nav.addItem(QListWidgetItem(title))
            self.pages.addWidget(builder())
        self.nav.currentRowChanged.connect(self.pages.setCurrentIndex)
        self.nav.setCurrentRow(0)

        root.addWidget(self._build_button_bar())

    def _build_button_bar(self) -> QWidget:
        """The footer: restart notice on the left, actions on the right."""
        bar = QWidget(self)
        layout = QHBoxLayout(bar)
        layout.setContentsMargins(18, 12, 18, 14)
        layout.setSpacing(10)

        quit_button = QPushButton("Quit Chiron", bar)
        quit_button.setToolTip("Stop Chiron completely. The overlay's ✕ does the same.")
        quit_button.clicked.connect(self.quitRequested.emit)
        layout.addWidget(quit_button)

        self.restart_badge = QLabel("", bar)
        self.restart_badge.setObjectName("restartBadge")
        layout.addWidget(self.restart_badge)
        layout.addStretch(1)

        self.revert_button = QPushButton("Revert", bar)
        self.revert_button.clicked.connect(lambda: self.load(self.settings))
        layout.addWidget(self.revert_button)

        close_button = QPushButton("Close", bar)
        close_button.clicked.connect(self.close)
        layout.addWidget(close_button)

        self.save_button = QPushButton("Save", bar)
        self.save_button.setObjectName("primary")
        self.save_button.setDefault(True)
        self.save_button.clicked.connect(self._save)
        layout.addWidget(self.save_button)
        return bar

    def _page(self, title: str, blurb: str) -> tuple[QWidget, QVBoxLayout]:
        """Return a scrollable page shell and the layout to fill it with."""
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

        heading = QLabel(title, content)
        heading.setObjectName("pageTitle")
        layout.addWidget(heading)

        subtitle = QLabel(blurb, content)
        subtitle.setObjectName("pageBlurb")
        subtitle.setWordWrap(True)
        layout.addWidget(subtitle)
        layout.addSpacing(6)

        scroll.setWidget(content)
        return page, layout

    def _section(self, layout: QVBoxLayout, title: str) -> QFormLayout:
        """Add a titled form section and return its form layout."""
        label = QLabel(title.upper(), layout.parentWidget())
        label.setObjectName("sectionTitle")
        layout.addWidget(label)

        line = QFrame(layout.parentWidget())
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
    def _hint(form: QFormLayout, text: str) -> QLabel:
        """Add a dim explanatory line under the previous field."""
        label = QLabel(text)
        label.setObjectName("hint")
        label.setWordWrap(True)
        form.addRow("", label)
        return label

    def _register(self, key: str, widget: Any, signal_name: str) -> Any:
        """Track a widget and wire its change signal to the dirty check."""
        self._widgets[key] = widget
        getattr(widget, signal_name).connect(self._on_field_changed)
        return widget

    # ----------------------------------------------------------------- pages

    def _build_session_page(self) -> QWidget:
        """Credential, model and the standing instructions given to the model."""
        page, layout = self._page(
            "Session",
            "Which model Chiron thinks with, how it authenticates, and what it is "
            "told before the first frame arrives.",
        )

        form = self._section(layout, "Credentials")
        key_row = QHBoxLayout()
        self.api_key_edit = self._register("api_key", QLineEdit(), "textEdited")
        self.api_key_edit.setEchoMode(QLineEdit.EchoMode.Password)
        self.api_key_edit.setPlaceholderText("AIza…")
        key_row.addWidget(self.api_key_edit, stretch=1)

        reveal = QPushButton("Show")
        reveal.setCheckable(True)
        reveal.toggled.connect(
            lambda shown: self.api_key_edit.setEchoMode(
                QLineEdit.EchoMode.Normal if shown else QLineEdit.EchoMode.Password
            )
        )
        key_row.addWidget(reveal)

        self.check_key_button = QPushButton("Test")
        self.check_key_button.clicked.connect(self._test_api_key)
        key_row.addWidget(self.check_key_button)
        form.addRow("Gemini API key", key_row)

        self.key_source_label = self._hint(form, "")
        self._hint(
            form,
            "Leave blank to use GEMINI_API_KEY (or GOOGLE_API_KEY) from the "
            "environment. Saved keys are written to the settings file with "
            "owner-only permissions. Get one at aistudio.google.com/apikey.",
        )

        form = self._section(layout, "Model")
        self.live_model_combo = self._register(
            "live_model", QComboBox(), "currentTextChanged"
        )
        self.live_model_combo.setEditable(True)
        self.live_model_combo.addItems(LIVE_MODEL_CHOICES)
        form.addRow("Live model", self.live_model_combo)
        self._hint(
            form,
            "Must be a Live API model. These models answer in speech; Chiron "
            "reads their own transcription, so the panel stays text and no audio "
            "is ever played. Output is billed as audio tokens.",
        )

        self.media_res_combo = self._register(
            "media_resolution", QComboBox(), "currentTextChanged"
        )
        self.media_res_combo.addItems(["low", "medium", "high"])
        form.addRow("Frame detail", self.media_res_combo)
        self._hint(
            form,
            "How many tokens each frame costs the context window: roughly 260 at "
            "low, and proportionally more above that.",
        )

        form = self._section(layout, "Instructions")
        self.game_name_edit = self._register("game_name", QLineEdit(), "textEdited")
        self.game_name_edit.setPlaceholderText("e.g. Elden Ring")
        form.addRow("Game", self.game_name_edit)
        self._hint(
            form,
            "Told to the model up front, so it knows what it is looking at. "
            "Leave empty and Chiron works it out from the window you are in "
            "when watching starts.",
        )

        self.extra_prompt_edit = self._register(
            "extra_prompt", QPlainTextEdit(), "textChanged"
        )
        self.extra_prompt_edit.setPlaceholderText(
            "e.g. I'm playing blind — never spoil anything I haven't found yet."
        )
        self.extra_prompt_edit.setFixedHeight(90)
        form.addRow("Extra instructions", self.extra_prompt_edit)

        layout.addStretch(1)
        return page

    def _build_capture_page(self) -> QWidget:
        """The adaptive shutter, and what it costs."""
        page, layout = self._page(
            "Capture",
            "Chiron watches slowly by default and speeds up when it matters. "
            "Slower baselines stretch how far back the model can see.",
        )

        form = self._section(layout, "Source")
        self.watch_on_launch_check = self._register(
            "watch_on_launch",
            QCheckBox("Start watching as soon as Chiron opens"),
            "toggled",
        )
        form.addRow("On launch", self.watch_on_launch_check)
        self._hint(
            form,
            "Off by default. Chiron reads nothing until you start watching, from "
            "the hotkey or the eye button in the overlay.",
        )

        self.monitor_combo = self._register(
            "monitor", QComboBox(), "currentIndexChanged"
        )
        for index, label in enumerate(describe_monitors()):
            self.monitor_combo.addItem(label, index)
        if self.monitor_combo.count() == 0:
            for index in range(0, 4):
                name = "All monitors" if index == 0 else f"Monitor {index}"
                self.monitor_combo.addItem(name, index)
        form.addRow("Monitor", self.monitor_combo)

        form = self._section(layout, "Shutter")
        self.baseline_spin = self._register(
            "baseline", QDoubleSpinBox(), "valueChanged"
        )
        self.baseline_spin.setRange(0.5, 60.0)
        self.baseline_spin.setSingleStep(0.5)
        self.baseline_spin.setSuffix(" s")
        form.addRow("Baseline interval", self.baseline_spin)
        self._hint(form, "Time between keepalive frames when nothing is happening.")

        self.burst_interval_spin = self._register(
            "burst_interval", QDoubleSpinBox(), "valueChanged"
        )
        self.burst_interval_spin.setRange(1.0, 10.0)
        self.burst_interval_spin.setSingleStep(0.5)
        self.burst_interval_spin.setSuffix(" s")
        form.addRow("Burst interval", self.burst_interval_spin)
        self._hint(form, "1 second is the API's ceiling; nothing faster is possible.")

        self.burst_duration_spin = self._register(
            "burst_duration", QDoubleSpinBox(), "valueChanged"
        )
        self.burst_duration_spin.setRange(1.0, 120.0)
        self.burst_duration_spin.setSuffix(" s")
        form.addRow("Burst duration", self.burst_duration_spin)
        self._hint(
            form,
            "How long the shutter stays fast after a question or a scene change.",
        )

        self.burn_label = QLabel("")
        self.burn_label.setObjectName("hint")
        self.burn_label.setWordWrap(True)
        self.burn_label.setStyleSheet(f"color: {PALETTE['accent']};")
        form.addRow("Estimated burn", self.burn_label)

        form = self._section(layout, "Scene detection")
        self.scene_check = self._register(
            "scene_enabled", QCheckBox("Burst when the screen changes hard"), "toggled"
        )
        form.addRow("Detection", self.scene_check)
        self._hint(
            form,
            "A cheap pixel diff catches loading screens, new areas and death "
            "screens — the moments most worth recording.",
        )

        threshold_row = QHBoxLayout()
        self.scene_slider = self._register(
            "scene_threshold", QSlider(Qt.Orientation.Horizontal), "valueChanged"
        )
        self.scene_slider.setRange(1, 60)
        threshold_row.addWidget(self.scene_slider, stretch=1)
        self.scene_value_label = QLabel("")
        self.scene_value_label.setObjectName("hint")
        self.scene_value_label.setFixedWidth(48)
        threshold_row.addWidget(self.scene_value_label)
        form.addRow("Sensitivity", threshold_row)
        self._hint(form, "Lower fires more often. 12% suits most games.")

        form = self._section(layout, "Encoding")
        self.frame_width_spin = self._register(
            "frame_width", QSpinBox(), "valueChanged"
        )
        self.frame_width_spin.setRange(256, 1920)
        self.frame_width_spin.setSingleStep(64)
        self.frame_width_spin.setSuffix(" px")
        form.addRow("Frame width", self.frame_width_spin)

        self.jpeg_quality_spin = self._register(
            "jpeg_quality", QSpinBox(), "valueChanged"
        )
        self.jpeg_quality_spin.setRange(10, 95)
        form.addRow("JPEG quality", self.jpeg_quality_spin)

        self.stamp_check = self._register(
            "stamp", QCheckBox("Stamp capture time onto each frame"), "toggled"
        )
        form.addRow("Timestamps", self.stamp_check)
        self._hint(
            form,
            "Gives the model an explicit clock, so 'the chest you saw earlier' "
            "has a when. Recommended.",
        )

        layout.addStretch(1)
        return page

    def _build_journal_page(self) -> QWidget:
        """Which writer fills the journal, and how often it is folded back in."""
        page, layout = self._page(
            "Journal",
            "The model's visual memory is a few minutes long whatever you do. The "
            "journal turns what it saw into text before the frames are evicted.",
        )

        form = self._section(layout, "Strategy")
        self.strategy_group = QButtonGroup(self)
        self.tool_call_radio = QRadioButton("In-session tool call")
        self.tool_call_radio.setObjectName("strategy")
        self.sidecar_radio = QRadioButton("Sidecar summariser")
        self.sidecar_radio.setObjectName("strategy")
        self.strategy_group.addButton(self.tool_call_radio, 0)
        self.strategy_group.addButton(self.sidecar_radio, 1)
        self.strategy_group.idToggled.connect(self._on_strategy_changed)

        form.addRow("", self.tool_call_radio)
        self._hint(
            form,
            "The live model gets a record_event function and calls it when "
            "something notable happens. No extra API calls, and one model holds "
            "the whole picture — but journalling competes with the conversation.",
        )
        form.addRow("", self.sidecar_radio)
        self._hint(
            form,
            "A separate cheap model reads the recent transcript and a few kept "
            "frames on a timer. Cleaner separation and easier to tune, at the "
            "cost of extra calls.",
        )

        form = self._section(layout, "Sidecar")
        self.sidecar_model_combo = self._register(
            "sidecar_model", QComboBox(), "currentTextChanged"
        )
        self.sidecar_model_combo.setEditable(True)
        self.sidecar_model_combo.addItems(SIDECAR_MODEL_CHOICES)
        form.addRow("Summariser model", self.sidecar_model_combo)
        self._hint(form, "A litellm model id, routed with the same Google key.")

        self.sidecar_interval_spin = self._register(
            "sidecar_interval", QDoubleSpinBox(), "valueChanged"
        )
        self.sidecar_interval_spin.setRange(30.0, 1800.0)
        self.sidecar_interval_spin.setSingleStep(30.0)
        self.sidecar_interval_spin.setSuffix(" s")
        form.addRow("Summarise every", self.sidecar_interval_spin)

        self.sidecar_frames_spin = self._register(
            "sidecar_frames", QSpinBox(), "valueChanged"
        )
        self.sidecar_frames_spin.setRange(0, 8)
        form.addRow("Frames shown", self.sidecar_frames_spin)

        form = self._section(layout, "Folding")
        self.fold_interval_spin = self._register(
            "fold_interval", QDoubleSpinBox(), "valueChanged"
        )
        self.fold_interval_spin.setRange(30.0, 900.0)
        self.fold_interval_spin.setSingleStep(30.0)
        self.fold_interval_spin.setSuffix(" s")
        form.addRow("Fold into session every", self.fold_interval_spin)
        self._hint(
            form,
            "New journal lines are pushed back into the live session as text, so "
            "facts survive the eviction of the frames they came from.",
        )

        self.fold_limit_spin = self._register("fold_limit", QSpinBox(), "valueChanged")
        self.fold_limit_spin.setRange(1, 200)
        form.addRow("Entries per fold", self.fold_limit_spin)

        self.max_entries_spin = self._register(
            "max_entries", QSpinBox(), "valueChanged"
        )
        self.max_entries_spin.setRange(10, 5000)
        form.addRow("Entries kept", self.max_entries_spin)
        self._hint(
            form,
            "The journal lives in memory only — v0 does not persist it between runs.",
        )

        layout.addStretch(1)
        return page

    def _build_overlay_page(self) -> QWidget:
        """Size, opacity and typography of the floating panel."""
        page, layout = self._page(
            "Overlay",
            "The panel that floats over your game. Changes here preview live.",
        )

        form = self._section(layout, "Size")
        self.width_spin = self._register("overlay_width", QSpinBox(), "valueChanged")
        self.width_spin.setRange(280, 1600)
        self.width_spin.setSingleStep(20)
        self.width_spin.setSuffix(" px")
        form.addRow("Width", self.width_spin)

        self.height_spin = self._register("overlay_height", QSpinBox(), "valueChanged")
        self.height_spin.setRange(240, 1400)
        self.height_spin.setSingleStep(20)
        self.height_spin.setSuffix(" px")
        form.addRow("Height", self.height_spin)
        self._hint(form, "You can also drag the panel's corner grip.")

        form = self._section(layout, "Appearance")
        opacity_row = QHBoxLayout()
        self.opacity_slider = self._register(
            "opacity", QSlider(Qt.Orientation.Horizontal), "valueChanged"
        )
        self.opacity_slider.setRange(20, 100)
        opacity_row.addWidget(self.opacity_slider, stretch=1)
        self.opacity_label = QLabel("")
        self.opacity_label.setObjectName("hint")
        self.opacity_label.setFixedWidth(48)
        opacity_row.addWidget(self.opacity_label)
        form.addRow("Opacity", opacity_row)

        self.font_spin = self._register("font_size", QSpinBox(), "valueChanged")
        self.font_spin.setRange(7, 24)
        self.font_spin.setSuffix(" pt")
        form.addRow("Font size", self.font_spin)

        self.on_top_check = self._register(
            "always_on_top", QCheckBox("Keep above other windows"), "toggled"
        )
        form.addRow("Stacking", self.on_top_check)
        self._hint(
            form,
            "Run games in borderless-windowed mode. A true-fullscreen X11 game "
            "grabs the display and nothing can draw over it.",
        )

        self.start_hidden_check = self._register(
            "start_hidden", QCheckBox("Start hidden, wait for the hotkey"), "toggled"
        )
        form.addRow("On launch", self.start_hidden_check)

        layout.addStretch(1)
        return page

    def _build_hotkeys_page(self) -> QWidget:
        """Global shortcuts that work while a game holds focus."""
        page, layout = self._page(
            "Hotkeys",
            "Grabbed from the display server, so they work while the game has "
            "keyboard focus.",
        )

        form = self._section(layout, "Watching")
        self.watch_toggle_hotkey_edit = self._register(
            "hotkey_watch_toggle", HotkeyEdit(), "textEdited"
        )
        form.addRow("Start / stop watching", self.watch_toggle_hotkey_edit)
        self._hint(form, "One key, pressed twice. Bound by default.")

        self.start_watch_hotkey_edit = self._register(
            "hotkey_watch_start", HotkeyEdit(), "textEdited"
        )
        form.addRow("Start watching", self.start_watch_hotkey_edit)

        self.stop_watch_hotkey_edit = self._register(
            "hotkey_watch_stop", HotkeyEdit(), "textEdited"
        )
        form.addRow("Stop watching", self.stop_watch_hotkey_edit)
        self._hint(
            form,
            "Optional separate keys, for when you would rather not have to know "
            "the current state — particularly to be certain you have stopped.",
        )

        form = self._section(layout, "Window")
        self.toggle_hotkey_edit = self._register(
            "hotkey_toggle", HotkeyEdit(), "textEdited"
        )
        form.addRow("Show / hide overlay", self.toggle_hotkey_edit)

        self.settings_hotkey_edit = self._register(
            "hotkey_settings", HotkeyEdit(), "textEdited"
        )
        form.addRow("Open settings", self.settings_hotkey_edit)
        self._hint(
            form,
            "Click a field and press the combination. Backspace clears a binding. "
            "If a shortcut is already owned by your desktop environment, Chiron "
            "reports it in the overlay instead of silently doing nothing.",
        )
        self.hotkey_backend_label = self._hint(form, "")

        layout.addStretch(1)
        return page

    def _build_about_page(self) -> QWidget:
        """Version, file locations and the things that trip people up."""
        page, layout = self._page(
            "About", "Chiron — an AI gaming assistant that watches your screen."
        )

        form = self._section(layout, "Installation")
        form.addRow("Version", QLabel(_version()))
        path_label = QLabel(str(default_settings_path()))
        path_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        form.addRow("Settings file", path_label)

        form = self._section(layout, "Known limits")
        for text in (
            "Run games borderless-windowed — a true-fullscreen X11 game grabs the "
            "display and no overlay can draw over it.",
            "Frames arrive at most once a second, so Chiron coaches strategy and "
            "orientation. It cannot call out a dodge in time.",
            "The journal lives in memory only; it is gone when Chiron exits.",
            "X11 is the supported display server. Wayland capture and hotkeys go "
            "through different machinery and are not wired up yet.",
        ):
            label = QLabel(f"• {text}")
            label.setObjectName("hint")
            label.setWordWrap(True)
            form.addRow("", label)

        layout.addStretch(1)
        return page

    # ------------------------------------------------------------ load/save

    def load(self, settings: Settings) -> None:
        """Populate every widget from `settings` without marking the form dirty."""
        self._loading = True
        try:
            self.api_key_edit.setText(settings.api_key)
            self.live_model_combo.setCurrentText(settings.live_model)
            self.media_res_combo.setCurrentText(settings.capture.media_resolution)
            self.game_name_edit.setText(settings.game_name)
            self.extra_prompt_edit.setPlainText(settings.extra_system_prompt)

            capture = settings.capture
            self.watch_on_launch_check.setChecked(capture.watch_on_launch)
            index = self.monitor_combo.findData(capture.monitor_index)
            self.monitor_combo.setCurrentIndex(max(0, index))
            self.baseline_spin.setValue(capture.baseline_interval_seconds)
            self.burst_interval_spin.setValue(capture.burst_interval_seconds)
            self.burst_duration_spin.setValue(capture.burst_duration_seconds)
            self.scene_check.setChecked(capture.scene_change_enabled)
            self.scene_slider.setValue(round(capture.scene_change_threshold * 100))
            self.frame_width_spin.setValue(capture.frame_width)
            self.jpeg_quality_spin.setValue(capture.jpeg_quality)
            self.stamp_check.setChecked(capture.stamp_timestamp)

            journal = settings.journal
            self.tool_call_radio.setChecked(journal.strategy == "tool_call")
            self.sidecar_radio.setChecked(journal.strategy == "sidecar")
            self.sidecar_model_combo.setCurrentText(journal.sidecar_model)
            self.sidecar_interval_spin.setValue(journal.sidecar_interval_seconds)
            self.sidecar_frames_spin.setValue(journal.sidecar_frame_count)
            self.fold_interval_spin.setValue(journal.fold_interval_seconds)
            self.fold_limit_spin.setValue(journal.fold_entry_limit)
            self.max_entries_spin.setValue(journal.max_entries)

            overlay = settings.overlay
            self.width_spin.setValue(overlay.width)
            self.height_spin.setValue(overlay.height)
            self.opacity_slider.setValue(round(overlay.opacity * 100))
            self.font_spin.setValue(overlay.font_size)
            self.on_top_check.setChecked(overlay.always_on_top)
            self.start_hidden_check.setChecked(overlay.start_hidden)

            self.toggle_hotkey_edit.setText(settings.hotkeys.toggle_overlay)
            self.settings_hotkey_edit.setText(settings.hotkeys.open_settings)
            self.watch_toggle_hotkey_edit.setText(settings.hotkeys.toggle_watching)
            self.start_watch_hotkey_edit.setText(settings.hotkeys.start_watching)
            self.stop_watch_hotkey_edit.setText(settings.hotkeys.stop_watching)
        finally:
            self._loading = False
        self._refresh_derived()

    def collect(self) -> Settings:
        """Read the form into a new :class:`Settings`.

        Returns:
            Settings: The edited configuration. Position is carried over from the
                loaded snapshot, since the overlay owns its own placement.
        """
        settings = self.settings.copy_deep()
        settings.api_key = self.api_key_edit.text().strip()
        settings.live_model = self.live_model_combo.currentText().strip()
        settings.game_name = self.game_name_edit.text().strip()
        settings.extra_system_prompt = self.extra_prompt_edit.toPlainText().strip()

        settings.capture.watch_on_launch = self.watch_on_launch_check.isChecked()
        monitor = self.monitor_combo.currentData()
        settings.capture.monitor_index = (
            int(monitor) if monitor is not None else settings.capture.monitor_index
        )
        settings.capture.baseline_interval_seconds = self.baseline_spin.value()
        settings.capture.burst_interval_seconds = self.burst_interval_spin.value()
        settings.capture.burst_duration_seconds = self.burst_duration_spin.value()
        settings.capture.scene_change_enabled = self.scene_check.isChecked()
        settings.capture.scene_change_threshold = self.scene_slider.value() / 100.0
        settings.capture.frame_width = self.frame_width_spin.value()
        settings.capture.jpeg_quality = self.jpeg_quality_spin.value()
        settings.capture.stamp_timestamp = self.stamp_check.isChecked()
        settings.capture.media_resolution = self.media_res_combo.currentText()

        settings.journal.strategy = (
            "sidecar" if self.sidecar_radio.isChecked() else "tool_call"
        )
        settings.journal.sidecar_model = self.sidecar_model_combo.currentText().strip()
        settings.journal.sidecar_interval_seconds = self.sidecar_interval_spin.value()
        settings.journal.sidecar_frame_count = self.sidecar_frames_spin.value()
        settings.journal.fold_interval_seconds = self.fold_interval_spin.value()
        settings.journal.fold_entry_limit = self.fold_limit_spin.value()
        settings.journal.max_entries = self.max_entries_spin.value()

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
        return settings

    def _save(self) -> None:
        """Emit the edited settings and adopt them as the new baseline."""
        settings = self.collect()
        self.settings = settings.copy_deep()
        self.settingsSaved.emit(settings)
        self._refresh_derived()

    def set_detected_game(self, name: str) -> None:
        """Show the auto-detected game as the Game field's placeholder.

        A placeholder, not a value: the field stays empty so Save never turns a
        guess into the player's explicit choice.

        Args:
            name (str): The detected game, or empty to restore the example.
        """
        if name:
            self.game_name_edit.setPlaceholderText(f"auto-detected: {name}")
        else:
            self.game_name_edit.setPlaceholderText("e.g. Elden Ring")

    def set_hotkey_backend(self, backend: str) -> None:
        """Report which hotkey mechanism is active, for the hotkeys page."""
        messages = {
            "xlib": "Using X11 key grabs (python-xlib).",
            "pynput": "Using pynput.",
            "none": "No global hotkey backend available on this display server — "
            "the overlay can only be reached from the taskbar.",
        }
        self.hotkey_backend_label.setText(messages.get(backend, backend))

    # -------------------------------------------------------------- reactive

    def _on_field_changed(self, *_: Any) -> None:
        """React to any edit: refresh derived labels and live previews."""
        if self._loading or not self._ready:
            return
        self._refresh_derived()
        self.appearanceChanged.emit(self._overlay_preview())

    def _on_strategy_changed(self, *_: Any) -> None:
        """Enable the sidecar fields only when the sidecar is selected."""
        if not self._ready:
            return
        sidecar = self.sidecar_radio.isChecked()
        for widget in (
            self.sidecar_model_combo,
            self.sidecar_interval_spin,
            self.sidecar_frames_spin,
        ):
            widget.setEnabled(sidecar)
        self._on_field_changed()

    def _overlay_preview(self) -> OverlaySettings:
        """The overlay appearance implied by the form right now."""
        preview = self.settings.overlay.model_copy()
        preview.width = self.width_spin.value()
        preview.height = self.height_spin.value()
        preview.opacity = self.opacity_slider.value() / 100.0
        preview.font_size = self.font_spin.value()
        preview.always_on_top = self.on_top_check.isChecked()
        return preview

    def _refresh_derived(self) -> None:
        """Update computed labels: token burn, slider readouts, restart notice."""
        if not self._ready:
            return
        resolution = self.media_res_combo.currentText()
        per_frame = TOKENS_PER_FRAME.get(resolution, 260)
        baseline = self.baseline_spin.value()
        burst = self.burst_interval_spin.value()
        baseline_rate = (60.0 / baseline) * per_frame if baseline else 0.0
        burst_rate = (60.0 / burst) * per_frame if burst else 0.0
        minutes = LIVE_CONTEXT_TOKENS / baseline_rate if baseline_rate else 0.0
        self.burn_label.setText(
            f"≈{baseline_rate / 1000:.1f}k tokens/min idle, "
            f"≈{burst_rate / 1000:.1f}k while bursting — about "
            f"{minutes:.0f} minutes of frames fit in a "
            f"{LIVE_CONTEXT_TOKENS // 1000}k context before the oldest are evicted."
        )
        self.opacity_label.setText(f"{self.opacity_slider.value()}%")
        self.scene_value_label.setText(f"{self.scene_slider.value()}%")

        for widget in (
            self.sidecar_model_combo,
            self.sidecar_interval_spin,
            self.sidecar_frames_spin,
        ):
            widget.setEnabled(self.sidecar_radio.isChecked())

        source = self.collect().api_key_source()
        messages = {
            "settings": "Using the key saved here.",
            "none": "No key found — Chiron cannot connect until one is set.",
        }
        self.key_source_label.setText(
            messages.get(source, f"Using {source} from the environment.")
        )

        needs_restart = self.settings.requires_session_restart(self.collect())
        self.restart_badge.setText(
            "Saving will reconnect the live session." if needs_restart else ""
        )

    def _test_api_key(self) -> None:
        """Check the credential by listing models, without leaving the window."""
        key = self.collect().resolved_api_key()
        if not key:
            self.key_source_label.setText("No key to test.")
            return
        self.check_key_button.setEnabled(False)
        self.key_source_label.setText("Checking…")

        async def check() -> None:
            message = "Key works."
            try:
                from google import genai

                client = genai.Client(api_key=key)
                await client.aio.models.list(config={"page_size": 1})
            except Exception as error:
                message = f"Key rejected: {error}"
            self.key_source_label.setText(message)
            self.check_key_button.setEnabled(True)

        try:
            asyncio.ensure_future(check())
        except RuntimeError:
            # No running loop (a bare widget test, say) — say so rather than crash.
            self.key_source_label.setText("Cannot test without a running event loop.")
            self.check_key_button.setEnabled(True)


__all__ = ["TOKENS_PER_FRAME", "HotkeyEdit", "SettingsWindow"]
