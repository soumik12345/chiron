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

v2 makes the panel a three-view stack rather than one transcript: **play**
(above), **history** (the browsable list of recorded sessions) and **viewer**
(one of them read back). Navigation is a small back-stack — Escape walks back a
view and only hides the panel from play — and an incoming answer snaps to play,
because a response missed while browsing last Tuesday is a response missed.
Entering the two review views lets the panel grow and returning shrinks it back:
reviewing happens between fights, and a list of sessions in a 420x560 panel is
not a list anybody reads.
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
    QInputDialog,
    QLabel,
    QLineEdit,
    QPushButton,
    QSizeGrip,
    QSizePolicy,
    QStackedWidget,
    QTextBrowser,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from chiron.config.settings import OverlaySettings
from chiron.journal.log import JournalEntry
from chiron.ui.history_view import HistoryView, SessionViewer
from chiron.ui.theme import PALETTE, STATUS_COLOURS, overlay_stylesheet

#: Messages retained in the transcript. Older ones scroll out of existence; the
#: journal, not the transcript, is what remembers.
MAX_MESSAGES = 80

#: The three pages of the stack, in the order they are added.
PLAY_VIEW, HISTORY_VIEW, SESSION_VIEW = 0, 1, 2

#: Minimum panel size while reviewing. The play panel is deliberately small; a
#: session list at that size shows three rows and a scroll bar.
REVIEW_MIN_WIDTH = 560
REVIEW_MIN_HEIGHT = 620

#: Characters of session title shown in the header before it is elided. The full
#: title is still in the tooltip.
TITLE_DISPLAY_LIMIT = 44

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
        newSessionRequested (): The ``+`` was clicked — close the current record
            and reset memory.
        historyRequested (): The history button was clicked.
        sessionOpened (str): A row in the history list was tapped.
        sessionRenamed (str, str): A session id and its new title.
        sessionDeleted (str): A session should be removed entirely.
        sessionThumbnailsDeleted (str): A session's thumbnails should go, its
            text stay.

    Attributes:
        settings (OverlaySettings): Appearance currently applied.
        watching (bool): Whether Chiron is currently reading the screen.
    """

    promptSubmitted = Signal(str)
    settingsRequested = Signal()
    panelHidden = Signal()
    quitRequested = Signal()
    watchToggled = Signal(bool)
    newSessionRequested = Signal()
    historyRequested = Signal()
    sessionOpened = Signal(str)
    sessionRenamed = Signal(str, str)
    sessionDeleted = Signal(str)
    sessionThumbnailsDeleted = Signal(str)

    def __init__(
        self, settings: OverlaySettings, parent: QWidget | None = None
    ) -> None:
        """Build the panel and apply `settings`."""
        super().__init__(parent)
        self.settings = settings
        self._messages: list[tuple[str, str]] = []
        self._streaming = False
        self.watching = False
        self._session_id = ""
        self._footer_text = ""
        self._cost_text = ""
        #: Where the panel was before review made it bigger, so returning to
        #: play puts it back rather than leaving a game-sized window over a game.
        self._play_size: tuple[int, int] | None = None

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
        layout.addWidget(self._build_session_row(panel))
        layout.addWidget(self._build_views(panel), stretch=1)
        self.input_row = self._build_input_row(panel)
        layout.addLayout(self.input_row)
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

    def _build_session_row(self, parent: QWidget) -> QWidget:
        """The session title, the ``+``, and the way into history.

        A row of its own rather than more buttons in the header: the header
        already carries the three controls whose meaning must never be in doubt
        (watching, settings, quit), and a rename target has to be wide enough to
        read a title in.
        """
        row = QWidget(parent)
        layout = QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        self.session_button = QToolButton(row)
        self.session_button.setObjectName("sessionTitle")
        self.session_button.setToolTip("Click to rename this session")
        self.session_button.clicked.connect(self._rename_current)
        # `Preferred` rather than a QToolButton's default `Fixed`, so a long
        # title gives way under pressure instead of shoving the two controls
        # off the panel.
        self.session_button.setSizePolicy(
            QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed
        )
        layout.addWidget(self.session_button)
        # An explicit stretch, not a stretch factor on the button: a Fixed-policy
        # widget never absorbs its allocation, so the layout would hand the slack
        # out as gaps *between* the three controls and leave the two buttons
        # floating in the middle of the row.
        layout.addStretch(1)

        self.new_session_button = QToolButton(row)
        self.new_session_button.setText("+")
        self.new_session_button.setToolTip(
            "Start a new session. Chiron forgets this one's journal and "
            "conversation; the record is kept."
        )
        self.new_session_button.clicked.connect(self.newSessionRequested.emit)
        layout.addWidget(self.new_session_button)

        self.history_button = QToolButton(row)
        self.history_button.setText("🕘")
        self.history_button.setToolTip("Past sessions")
        self.history_button.clicked.connect(self.historyRequested.emit)
        layout.addWidget(self.history_button)

        self.set_session("", "")
        return row

    def _build_views(self, parent: QWidget) -> QWidget:
        """The three-page stack: play, history, one session read back."""
        self.views = QStackedWidget(parent)

        self.transcript = QTextBrowser(self.views)
        self.transcript.setObjectName("transcript")
        self.transcript.setOpenExternalLinks(True)
        self.transcript.setFrameShape(QFrame.Shape.NoFrame)
        self.views.addWidget(self.transcript)

        self.history = HistoryView(self.views)
        self.history.backRequested.connect(self.show_play)
        self.history.sessionOpened.connect(self.sessionOpened.emit)
        self.history.renameRequested.connect(self._rename_session)
        self.history.deleteRequested.connect(self.sessionDeleted.emit)
        self.history.thumbnailsDeleteRequested.connect(
            self.sessionThumbnailsDeleted.emit
        )
        self.views.addWidget(self.history)

        self.viewer = SessionViewer(self.views)
        self.viewer.backRequested.connect(self.show_history)
        self.views.addWidget(self.viewer)

        self.views.setCurrentIndex(PLAY_VIEW)
        return self.views

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
        if self._play_size is None:
            self.resize(settings.width, settings.height)
        else:
            # A review view is holding the panel open; remember the new play
            # size for when it closes rather than shrinking mid-read.
            self._play_size = (settings.width, settings.height)
        if settings.position_x is not None and settings.position_y is not None:
            self.move(settings.position_x, settings.position_y)
        if settings.always_on_top != previous_on_top:
            visible = self.isVisible()
            self._apply_window_flags(settings.always_on_top)
            if visible:
                self.show()
        self._render()

    def current_geometry(self) -> tuple[int, int, int, int]:
        """Return ``(x, y, width, height)`` so the app can persist placement.

        Reports the *play* size while a review view has the panel temporarily
        enlarged. Otherwise opening history once would permanently resize the
        thing that floats over the game.
        """
        geometry = self.geometry()
        if self._play_size is not None:
            width, height = self._play_size
            return geometry.x(), geometry.y(), width, height
        return geometry.x(), geometry.y(), geometry.width(), geometry.height()

    # ------------------------------------------------------------------ views

    @property
    def current_view(self) -> int:
        """Which page of the stack is showing."""
        return self.views.currentIndex()

    def show_play(self) -> None:
        """Return to the game view, shrinking the panel back if review grew it."""
        self.views.setCurrentIndex(PLAY_VIEW)
        self._set_review_mode(False)

    def show_history(self) -> None:
        """Show the session list."""
        self.views.setCurrentIndex(HISTORY_VIEW)
        self._set_review_mode(True)

    def show_session_viewer(self) -> None:
        """Show whatever the viewer was last given."""
        self.views.setCurrentIndex(SESSION_VIEW)
        self._set_review_mode(True)

    def back(self) -> bool:
        """Walk one step back through the view stack.

        Returns:
            bool: True when a step was taken, False when already on play — which
                is how Escape knows whether it should hide the panel instead.
        """
        index = self.views.currentIndex()
        if index == SESSION_VIEW:
            self.show_history()
            return True
        if index == HISTORY_VIEW:
            self.show_play()
            return True
        return False

    def _set_review_mode(self, reviewing: bool) -> None:
        """Grow the panel for review and put it back afterwards.

        The prompt line goes with it: there is nothing to ask a past session.
        """
        for index in range(self.input_row.count()):
            widget = self.input_row.itemAt(index).widget()
            if widget is not None:
                widget.setVisible(not reviewing)
        if reviewing:
            if self._play_size is None:
                self._play_size = (self.width(), self.height())
                self.resize(
                    max(self.width(), REVIEW_MIN_WIDTH),
                    max(self.height(), REVIEW_MIN_HEIGHT),
                )
        elif self._play_size is not None:
            width, height = self._play_size
            self._play_size = None
            self.resize(width, height)

    # ---------------------------------------------------------------- session

    def set_session(self, session_id: str, title: str) -> None:
        """Show which session is being recorded, if any.

        Args:
            session_id (str): The open session's id, or empty for none.
            title (str): Its display title.
        """
        self._session_id = session_id
        if session_id:
            # Elided here rather than left to the widget: a QToolButton clips
            # rather than ellipsises, and a renamed session can be any length.
            shown = (
                title
                if len(title) <= TITLE_DISPLAY_LIMIT
                else title[: TITLE_DISPLAY_LIMIT - 1].rstrip() + "…"
            )
            self.session_button.setText(f"▮ {shown}")
            self.session_button.setToolTip(f"{title}\nClick to rename this session")
            self.session_button.setEnabled(True)
        else:
            # Not an error state: Chiron launches with no session, and one is
            # created the moment the player does something worth recording.
            self.session_button.setText("▯ no session yet")
            self.session_button.setEnabled(False)

    def set_cost(self, usd: float, estimated: bool = False) -> None:
        """Show the running spend beside the other footer counters."""
        self._cost_text = f"{'~' if estimated else ''}${usd:.2f}" if usd else ""
        self.set_footer(self._footer_text)

    def _rename_current(self) -> None:
        """Ask for a new title for the session being recorded."""
        self._rename_session(self._session_id)

    def _rename_session(self, session_id: str) -> None:
        """Ask for a new title for any session and announce the answer."""
        if not session_id:
            return
        title, accepted = QInputDialog.getText(self, "Rename session", "Title:")
        if accepted and title.strip():
            self.sessionRenamed.emit(session_id, title.strip())

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
        """Open an empty assistant message for streaming deltas into.

        Snaps back to the play view first: an answer that arrives while the
        player is reading last Tuesday is an answer they never see, and a
        question was asked to be answered now.
        """
        self.show_play()
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
                f"Chiron is reading your screen. Stop watching{suffix}"
            )
            self.watch_button.setStyleSheet(f"color: {PALETTE['live']};")
        else:
            # U+2298 (⊘) is a standalone glyph; the combining U+20E0 used here
            # previously has zero width and draws itself over the next letter.
            self.watch_button.setText("⊘ Not watching")
            self.watch_button.setToolTip(
                f"Chiron cannot see your screen. Start watching{suffix}"
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
        """Set the small status line under the input, keeping the cost on it.

        Cost is appended here rather than passed in by the caller, because the
        two are refreshed by different things at different rates: the counters
        tick once a second off a timer, the total changes only when a call is
        billed. Whichever moved last, the line reads whole.
        """
        self._footer_text = text
        self.footer.setText(
            f"{text}  ·  {self._cost_text}" if self._cost_text else text
        )

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
        """Escape walks back a view, and hides the panel from the play view."""
        if event.key() == Qt.Key.Key_Escape:
            if not self.back():
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
