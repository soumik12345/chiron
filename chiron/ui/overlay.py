"""The overlay: a frameless, always-on-top chat panel that floats over the game.

`Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint` is the pair that behaves
reliably on X11, and `Qt.Tool` keeps the panel out of the taskbar and the
alt-tab list so it reads as an instrument rather than another application.
Because the frame is gone, the header doubles as a drag handle and a
:class:`QSizeGrip` in the corner does the resizing.

The transcript is re-rendered from a small message list rather than appended to.
Streaming makes the last message grow a few times a second, and re-rendering a
bounded list is both simpler and less fragile than surgically editing a rich-text
document — the list is capped, so the cost stays flat however long a session runs.

Journal entries appear inline in a dimmer style. That is a deliberate piece of
honesty: the journal is what lets Chiron remember anything beyond the last few
minutes, and a player who can see it being written can tell when it is wrong.
"""

from __future__ import annotations

import html
import re

from PySide6.QtCore import QEvent, QPoint, Qt, Signal
from PySide6.QtGui import (
    QCloseEvent,
    QKeyEvent,
    QKeySequence,
    QMouseEvent,
    QShortcut,
)
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSizeGrip,
    QTextBrowser,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from chiron.config.settings import OverlaySettings
from chiron.journal.log import JournalEntry
from chiron.ui.theme import PALETTE, STATUS_COLOURS, overlay_stylesheet

#: Messages retained in the transcript. Older ones scroll out of existence; the
#: journal, not the transcript, is what remembers.
MAX_MESSAGES = 80

_BOLD = re.compile(r"\*\*(.+?)\*\*", re.DOTALL)
_ITALIC = re.compile(r"(?<!\*)\*([^*]+)\*(?!\*)")
_CODE = re.compile(r"`([^`]+)`")


def format_message_text(text: str) -> str:
    """Escape `text` and apply the sliver of markdown worth supporting.

    The model writes short coaching answers, so bold, italics, inline code and
    line breaks cover essentially everything that shows up. Anything else is left
    as literal text rather than risking mangled HTML in an always-on-top window.

    Args:
        text (str): Raw model or user text.

    Returns:
        str: HTML-safe markup.
    """
    escaped = html.escape(text)
    escaped = _CODE.sub(rf'<code style="color:{PALETTE["accent"]}">\1</code>', escaped)
    escaped = _BOLD.sub(r"<b>\1</b>", escaped)
    escaped = _ITALIC.sub(r"<i>\1</i>", escaped)
    return escaped.replace("\n", "<br>")


class _DragHandle(QWidget):
    """The header strip; dragging it moves the frameless window."""

    def __init__(self, window: QWidget, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._window = window
        self._press_offset: QPoint | None = None
        self.setCursor(Qt.CursorShape.OpenHandCursor)

    def mousePressEvent(self, event: QMouseEvent) -> None:
        """Remember where in the window the drag started."""
        if event.button() == Qt.MouseButton.LeftButton:
            self._press_offset = (
                event.globalPosition().toPoint()
                - self._window.frameGeometry().topLeft()
            )
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            event.accept()

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        """Move the window with the pointer."""
        if self._press_offset is not None:
            self._window.move(event.globalPosition().toPoint() - self._press_offset)
            event.accept()

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        """End the drag."""
        self._press_offset = None
        self.setCursor(Qt.CursorShape.OpenHandCursor)
        event.accept()


class OverlayWindow(QWidget):
    """The always-on-top chat panel.

    Signals:
        promptSubmitted (str): The player pressed Enter on a non-empty prompt.
        settingsRequested (): The gear button was clicked.
        panelHidden (): The panel was hidden by Escape or the – button.
        quitRequested (): The user asked to leave — the ✕, Ctrl+Q, or the window
            manager closing the window.
        watchToggled (bool): The watch button was clicked, carrying the state the
            user is asking for.

    Attributes:
        settings (OverlaySettings): Appearance currently applied.
        watching (bool): Whether Chiron is currently reading the screen.
    """

    promptSubmitted = Signal(str)
    settingsRequested = Signal()
    panelHidden = Signal()
    quitRequested = Signal()
    watchToggled = Signal(bool)

    def __init__(
        self, settings: OverlaySettings, parent: QWidget | None = None
    ) -> None:
        """Build the panel and apply `settings`."""
        super().__init__(parent)
        self.settings = settings
        self._messages: list[tuple[str, str]] = []
        self._streaming = False
        self.watching = False

        self.setWindowTitle("Chiron")
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self._apply_window_flags(settings.always_on_top)

        self._build_ui()
        self.apply_settings(settings)
        self.set_watching(False)

        quit_shortcut = QShortcut(QKeySequence("Ctrl+Q"), self)
        quit_shortcut.activated.connect(self.quitRequested.emit)

    # ----------------------------------------------------------------- build

    def _apply_window_flags(self, always_on_top: bool) -> None:
        """Set the frameless/tool/on-top flag combination."""
        flags = Qt.WindowType.FramelessWindowHint | Qt.WindowType.Tool
        if always_on_top:
            flags |= Qt.WindowType.WindowStaysOnTopHint
        self.setWindowFlags(flags)

    def _build_ui(self) -> None:
        """Assemble the panel's widgets."""
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)

        panel = QFrame(self)
        panel.setObjectName("panel")
        outer.addWidget(panel)

        layout = QVBoxLayout(panel)
        layout.setContentsMargins(12, 10, 12, 10)
        layout.setSpacing(8)

        layout.addWidget(self._build_header(panel))
        layout.addWidget(self._build_transcript(panel), stretch=1)
        layout.addLayout(self._build_input_row(panel))
        layout.addLayout(self._build_footer_row(panel))

    def _build_header(self, parent: QWidget) -> QWidget:
        """The draggable title strip with status, settings and hide buttons."""
        header = _DragHandle(self, parent)
        row = QHBoxLayout(header)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(8)

        brand = QLabel("CHIRON", header)
        brand.setObjectName("brand")
        row.addWidget(brand)

        self.watch_button = QToolButton(header)
        self.watch_button.setObjectName("watch")
        self.watch_button.clicked.connect(
            lambda: self.watchToggled.emit(not self.watching)
        )
        row.addWidget(self.watch_button)

        self.status_dot = QLabel("●", header)
        self.status_dot.setObjectName("statusDot")
        row.addWidget(self.status_dot)

        self.status_text = QLabel("idle", header)
        self.status_text.setObjectName("statusText")
        row.addWidget(self.status_text)
        row.addStretch(1)

        self.settings_button = QToolButton(header)
        self.settings_button.setText("⚙")
        self.settings_button.setToolTip("Settings")
        self.settings_button.clicked.connect(self.settingsRequested.emit)
        row.addWidget(self.settings_button)

        self.hide_button = QToolButton(header)
        self.hide_button.setText("–")
        self.hide_button.clicked.connect(self.hide_panel)
        row.addWidget(self.hide_button)
        self.set_hide_hint("")

        self.close_button = QToolButton(header)
        self.close_button.setObjectName("close")
        self.close_button.setText("✕")
        self.close_button.setToolTip("Quit Chiron (Ctrl+Q)")
        self.close_button.clicked.connect(self.quitRequested.emit)
        row.addWidget(self.close_button)
        return header

    def _build_transcript(self, parent: QWidget) -> QWidget:
        """The scrolling conversation view."""
        self.transcript = QTextBrowser(parent)
        self.transcript.setObjectName("transcript")
        self.transcript.setOpenExternalLinks(True)
        self.transcript.setFrameShape(QFrame.Shape.NoFrame)
        return self.transcript

    def _build_input_row(self, parent: QWidget) -> QHBoxLayout:
        """The prompt line and send button."""
        row = QHBoxLayout()
        row.setSpacing(8)

        self.input = QLineEdit(parent)
        self.input.setObjectName("input")
        self.input.setPlaceholderText("Ask about what's on screen…")
        self.input.returnPressed.connect(self._submit)
        row.addWidget(self.input, stretch=1)

        self.send_button = QPushButton("Ask", parent)
        self.send_button.setObjectName("send")
        self.send_button.clicked.connect(self._submit)
        row.addWidget(self.send_button)
        return row

    def _build_footer_row(self, parent: QWidget) -> QHBoxLayout:
        """The status footer and the resize grip."""
        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        self.footer = QLabel("", parent)
        self.footer.setObjectName("footer")
        row.addWidget(self.footer, stretch=1)
        row.addWidget(QSizeGrip(parent), 0, Qt.AlignmentFlag.AlignBottom)
        return row

    # ------------------------------------------------------------ appearance

    def apply_settings(self, settings: OverlaySettings) -> None:
        """Apply appearance settings, resizing and repositioning as needed.

        Args:
            settings (OverlaySettings): The new appearance configuration.
        """
        previous_on_top = self.settings.always_on_top
        self.settings = settings
        self.setStyleSheet(overlay_stylesheet(settings.font_size))
        self.setWindowOpacity(settings.opacity)
        self.resize(settings.width, settings.height)
        if settings.position_x is not None and settings.position_y is not None:
            self.move(settings.position_x, settings.position_y)
        if settings.always_on_top != previous_on_top:
            visible = self.isVisible()
            self._apply_window_flags(settings.always_on_top)
            if visible:
                self.show()
        self._render()

    def current_geometry(self) -> tuple[int, int, int, int]:
        """Return ``(x, y, width, height)`` so the app can persist placement."""
        geometry = self.geometry()
        return geometry.x(), geometry.y(), geometry.width(), geometry.height()

    # --------------------------------------------------------------- content

    def append_user(self, text: str) -> None:
        """Add a player message to the transcript."""
        self._append("user", text)

    def append_system(self, text: str) -> None:
        """Add a Chiron-side notice (connection state, errors, hints)."""
        self._append("system", text)

    def append_journal(self, entry: JournalEntry) -> None:
        """Show a journal entry inline, dimmed."""
        self._append("journal", entry.render())

    def start_response(self) -> None:
        """Open an empty assistant message for streaming deltas into."""
        self._streaming = True
        self._append("assistant", "")

    def append_delta(self, text: str) -> None:
        """Append a streamed chunk to the open assistant message."""
        if not self._streaming or not self._messages:
            self.start_response()
        role, existing = self._messages[-1]
        self._messages[-1] = (role, existing + text)
        self._render()

    def end_response(self, text: str = "") -> None:
        """Close the streaming message, replacing it with `text` when given."""
        if self._streaming and self._messages and text:
            self._messages[-1] = ("assistant", text)
        self._streaming = False
        self._render()

    def clear_transcript(self) -> None:
        """Empty the conversation view."""
        self._messages.clear()
        self._streaming = False
        self._render()

    def _append(self, role: str, text: str) -> None:
        """Add a message and re-render, trimming to :data:`MAX_MESSAGES`."""
        self._messages.append((role, text))
        if len(self._messages) > MAX_MESSAGES:
            del self._messages[: len(self._messages) - MAX_MESSAGES]
        self._render()

    def _render(self) -> None:
        """Rebuild the transcript HTML and scroll to the newest message."""
        blocks: list[str] = []
        for role, text in self._messages:
            if role == "user":
                blocks.append(
                    f'<p style="margin:6px 0 2px 0;color:{PALETTE["user"]}">'
                    f"<b>You</b></p>"
                    f'<p style="margin:0 0 8px 0">{format_message_text(text)}</p>'
                )
            elif role == "assistant":
                body = format_message_text(text) or (
                    f'<span style="color:{PALETTE["text_faint"]}">…</span>'
                )
                blocks.append(
                    f'<p style="margin:6px 0 2px 0;color:{PALETTE["accent"]}">'
                    f"<b>Chiron</b></p>"
                    f'<p style="margin:0 0 8px 0">{body}</p>'
                )
            elif role == "journal":
                blocks.append(
                    f'<p style="margin:2px 0;color:{PALETTE["text_faint"]}">'
                    f"<i>✎ {format_message_text(text)}</i></p>"
                )
            else:
                blocks.append(
                    f'<p style="margin:4px 0;color:{PALETTE["text_dim"]}">'
                    f"{format_message_text(text)}</p>"
                )
        self.transcript.setHtml("".join(blocks))
        scrollbar = self.transcript.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    # ---------------------------------------------------------------- status

    def set_watching(self, watching: bool, hotkey: str = "") -> None:
        """Reflect whether Chiron is reading the screen.

        The button says what pressing it will *do*, and the eye says what is
        happening right now — a control whose label is also its state reads
        ambiguously on the one setting where being sure matters.

        Args:
            watching (bool): The current state.
            hotkey (str): The binding to mention in the tooltip.
        """
        self.watching = watching
        suffix = f" ({hotkey})" if hotkey else ""
        if watching:
            self.watch_button.setText("👁 Watching")
            self.watch_button.setToolTip(
                f"Chiron is reading your screen — stop{suffix}"
            )
            self.watch_button.setStyleSheet(f"color: {PALETTE['live']};")
        else:
            # U+2298 (⊘) is a standalone glyph; the combining U+20E0 used here
            # previously has zero width and draws itself over the next letter.
            self.watch_button.setText("⊘ Not watching")
            self.watch_button.setToolTip(
                f"Chiron cannot see your screen — start watching{suffix}"
            )
            self.watch_button.setStyleSheet(f"color: {PALETTE['text_faint']};")

    def set_status(self, status: str, detail: str = "") -> None:
        """Update the header status dot and label.

        Args:
            status (str): One of the session statuses.
            detail (str): Extra context shown beside the status name.
        """
        colour = STATUS_COLOURS.get(status, PALETTE["text_faint"])
        self.status_dot.setStyleSheet(f"color: {colour};")
        self.status_text.setText(f"{status} · {detail}" if detail else status)

    def set_footer(self, text: str) -> None:
        """Set the small status line under the input."""
        self.footer.setText(text)

    # ------------------------------------------------------------ visibility

    def show_and_focus(self) -> None:
        """Show the panel, raise it and put the caret in the prompt."""
        self.show()
        self.raise_()
        self.activateWindow()
        self.input.setFocus(Qt.FocusReason.OtherFocusReason)

    def hide_panel(self) -> None:
        """Hide the panel and announce it, so the app can persist geometry."""
        self.hide()
        self.panelHidden.emit()

    def set_hide_hint(self, hotkey: str) -> None:
        """Say on the hide button which key brings the panel back.

        Hiding a frameless, taskbar-less window is only safe if the way back is
        obvious at the moment you hide it.

        Args:
            hotkey (str): The show/hide binding, or empty if there isn't one.
        """
        back = f" {hotkey} brings it back." if hotkey.strip() else ""
        self.hide_button.setToolTip(f"Hide the panel (Esc).{back}")

    def toggle(self) -> None:
        """Show the panel if hidden, hide it if visible."""
        if self.isVisible():
            self.hide_panel()
        else:
            self.show_and_focus()

    # ------------------------------------------------------------------ misc

    def _submit(self) -> None:
        """Emit the typed prompt and clear the input."""
        text = self.input.text().strip()
        if not text:
            return
        self.input.clear()
        self.promptSubmitted.emit(text)

    def keyPressEvent(self, event: QKeyEvent) -> None:
        """Escape hides the panel; everything else behaves normally."""
        if event.key() == Qt.Key.Key_Escape:
            self.hide_panel()
            event.accept()
            return
        super().keyPressEvent(event)

    def closeEvent(self, event: QCloseEvent) -> None:
        """Closing the window quits Chiron.

        Reached by the ✕, by Alt+F4, and by anything else the desktop counts as
        closing the window. Hiding is what the – button and Escape are for; a
        close that leaves the process running is a close button that lies.
        """
        self.quitRequested.emit()
        event.accept()

    def changeEvent(self, event: QEvent) -> None:
        """Keep the prompt focused when the panel is activated."""
        if event.type() == QEvent.Type.ActivationChange and self.isActiveWindow():
            self.input.setFocus(Qt.FocusReason.ActiveWindowFocusReason)
        super().changeEvent(event)


__all__ = ["MAX_MESSAGES", "OverlayWindow", "format_message_text"]
