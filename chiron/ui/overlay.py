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

Journal entries used to appear inline in a dimmer style, for a reason worth
keeping: the journal is what lets Chiron remember anything beyond the last few
minutes, and a player who can see it being written can tell when it is wrong. v3
keeps the honesty and moves the position — the entries now live in a collapsible
right-hand column (:class:`~chiron.ui.journal_drawer.JournalDrawer`) rather than
interleaved with the conversation, because a memory is scanned and a conversation
is read, and the two do not belong on one axis. The cost of moving them out is
that a closed drawer is a silent one, which is what the unread count on the
toggle exists to pay: `✎ 3` says three things were written while you were
playing, and opening it clears the count.

Opening the drawer *widens the window* rather than narrowing the transcript. The
play panel is deliberately small, and a 420px panel split two ways is two columns
of nothing. `width` in :class:`~chiron.config.settings.OverlaySettings` therefore
always means the un-drawered width, and :meth:`OverlayWindow.current_geometry`
subtracts the column back out before the app persists it.

The panel is a two-view stack: **play** (the transcript) and **viewer** (one
recorded session read back). Getting *to* a session is a popup —
:class:`~chiron.ui.history_view.SessionPicker` drops under the history button
rather than taking a page of its own — so the panel only ever resizes for
reading, not for choosing. Escape walks back a view and only hides the panel from
play, and an incoming answer snaps to play, because a response missed while
browsing last Tuesday is a response missed.
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
from chiron.ui.history_view import SessionPicker, SessionViewer
from chiron.ui.journal_drawer import JournalDrawer
from chiron.ui.theme import PALETTE, STATUS_COLOURS, overlay_stylesheet

#: Messages retained in the transcript. Older ones scroll out of existence; the
#: journal, not the transcript, is what remembers.
MAX_MESSAGES = 80

#: The two pages of the stack, in the order they are added.
PLAY_VIEW, SESSION_VIEW = 0, 1

#: Minimum panel size while reading a session back. The play panel is
#: deliberately small; an evening of transcript and thumbnails at that size is
#: not something anybody reads.
REVIEW_MIN_WIDTH = 560
REVIEW_MIN_HEIGHT = 620

#: Gap between the transcript and the journal column, in the body row.
BODY_SPACING = 10

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
        historyRequested (): The history button was clicked; fill the picker.
        sessionOpened (str): A row in the session picker was tapped.
        sessionRenamed (str, str): A session id and its new title.
        sessionDeleted (str): A session should be removed entirely.
        sessionThumbnailsDeleted (str): A session's thumbnails should go, its
            text stay.
        journalToggled (bool): The drawer was opened or closed.

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
    journalToggled = Signal(bool)

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
        #: Stored as the literal window size, drawer included.
        self._play_size: tuple[int, int] | None = None
        #: Drawer state. `_journal_width` is the column alone; the window is
        #: that much wider while it is open, and `current_geometry()` takes it
        #: back off before anything persists a width.
        self._journal_open = False
        self._journal_width = settings.journal_width
        self._journal_unread = 0

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
        layout.addLayout(self._build_body_row(panel), stretch=1)
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

        self.journal_button = QToolButton(row)
        self.journal_button.setObjectName("journalToggle")
        self.journal_button.clicked.connect(self.toggle_journal)
        layout.addWidget(self.journal_button)

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
        self._refresh_journal_button()
        return row

    def _build_body_row(self, parent: QWidget) -> QHBoxLayout:
        """The view stack and the journal column beside it.

        The drawer sits *outside* the stack rather than being a third page,
        which is what lets it stay open while a recorded session is read back:
        the journal being written now and the evening being reviewed are two
        different things, and there is no reason looking at one should close the
        other.
        """
        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(BODY_SPACING)
        row.addWidget(self._build_views(parent), stretch=1)

        self.journal_drawer = JournalDrawer(parent)
        self.journal_drawer.setObjectName("journalDrawer")
        self.journal_drawer.setFixedWidth(self._journal_width)
        self.journal_drawer.closeRequested.connect(lambda: self.set_journal_open(False))
        self.journal_drawer.setVisible(False)
        row.addWidget(self.journal_drawer)
        return row

    def _build_views(self, parent: QWidget) -> QWidget:
        """The two-page stack: play, and one session read back."""
        self.views = QStackedWidget(parent)

        self.transcript = QTextBrowser(self.views)
        self.transcript.setObjectName("transcript")
        self.transcript.setOpenExternalLinks(True)
        self.transcript.setFrameShape(QFrame.Shape.NoFrame)
        self.views.addWidget(self.transcript)

        self.viewer = SessionViewer(self.views)
        self.viewer.backRequested.connect(self.show_play)
        self.viewer.renameRequested.connect(self._rename_session)
        self.viewer.deleteRequested.connect(self.sessionDeleted.emit)
        self.viewer.thumbnailsDeleteRequested.connect(
            self.sessionThumbnailsDeleted.emit
        )
        self.views.addWidget(self.viewer)

        self.picker = SessionPicker(self)
        self.picker.sessionOpened.connect(self.sessionOpened.emit)

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
        # The picker is a top-level popup, so it is not styled by the panel's
        # sheet and needs its own copy of it.
        self.picker.setStyleSheet(overlay_stylesheet(settings.font_size))
        self.setWindowOpacity(settings.opacity)

        # The drawer settles before the resize below, so the width it asks for
        # is the width it gets — `settings.width` never includes the column.
        self._journal_width = settings.journal_width
        self.journal_drawer.setFixedWidth(self._journal_width)
        self._show_journal(settings.journal_open)
        offset = self._journal_offset()

        if self._play_size is None:
            self.resize(settings.width + offset, settings.height)
        else:
            # A review view is holding the panel open; remember the new play
            # size for when it closes rather than shrinking mid-read.
            self._play_size = (settings.width + offset, settings.height)
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

        Two things temporarily make the window wider or taller than the size
        worth remembering, and both are undone here: a review view enlarging the
        panel, and the journal drawer widening it. Otherwise reading one session
        back, or leaving the drawer open, would permanently resize the thing
        that floats over the game.
        """
        geometry = self.geometry()
        width, height = (
            self._play_size
            if self._play_size is not None
            else (geometry.width(), geometry.height())
        )
        return geometry.x(), geometry.y(), width - self._journal_offset(), height

    # ------------------------------------------------------------------ views

    @property
    def current_view(self) -> int:
        """Which page of the stack is showing."""
        return self.views.currentIndex()

    def show_play(self) -> None:
        """Return to the game view, shrinking the panel back if review grew it."""
        self.views.setCurrentIndex(PLAY_VIEW)
        self._set_review_mode(False)

    def open_session_picker(self) -> None:
        """Drop the session list under the history button.

        A popup rather than a page: choosing which evening to look at should not
        cost a panel resize, and the panel is floating over a game.
        """
        self.picker.open_under(self.history_button)

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
        if self.views.currentIndex() == SESSION_VIEW:
            self.show_play()
            return True
        return False

    def _set_review_mode(self, reviewing: bool) -> None:
        """Grow the panel for review and put it back afterwards.

        The prompt line goes with it: there is nothing to ask a past session.
        The drawer stays exactly as it was — the journal being written now is
        still being written while an old evening is read.
        """
        for index in range(self.input_row.count()):
            widget = self.input_row.itemAt(index).widget()
            if widget is not None:
                widget.setVisible(not reviewing)
        if reviewing:
            if self._play_size is None:
                self._play_size = (self.width(), self.height())
                self.resize(
                    max(self.width(), REVIEW_MIN_WIDTH + self._journal_offset()),
                    max(self.height(), REVIEW_MIN_HEIGHT),
                )
        elif self._play_size is not None:
            width, height = self._play_size
            self._play_size = None
            self.resize(width, height)

    # ---------------------------------------------------------------- journal

    @property
    def journal_open(self) -> bool:
        """Whether the journal column is showing."""
        return self._journal_open

    def _journal_offset(self) -> int:
        """How much wider the window is for having the drawer open."""
        return self._journal_width + BODY_SPACING if self._journal_open else 0

    def set_journal_open(self, opening: bool) -> None:
        """Open or collapse the journal column, resizing the window with it.

        The width changes rather than the split, so the transcript keeps the
        size it was given. Repeating the current state does nothing, which is
        what makes this safe to call from a hotkey, a button and settings alike.

        Args:
            opening (bool): The state to move to.
        """
        if opening == self._journal_open:
            return
        before = self._journal_offset()
        self._show_journal(opening)
        delta = self._journal_offset() - before
        self.resize(self.width() + delta, self.height())
        if self._play_size is not None:
            # Reviewing: the size play will be restored to has to move too, or
            # closing the drawer here would reopen a gap there.
            self._play_size = (self._play_size[0] + delta, self._play_size[1])
        self.journalToggled.emit(opening)

    def toggle_journal(self) -> None:
        """Open the journal if closed, close it if open."""
        self.set_journal_open(not self._journal_open)

    def _show_journal(self, opening: bool) -> None:
        """Set drawer state and visibility without touching the window size."""
        self._journal_open = opening
        self.journal_drawer.setVisible(opening)
        if opening:
            # Opening is reading: whatever arrived while it was shut has now
            # been seen.
            self._journal_unread = 0
        self._refresh_journal_button()

    def _refresh_journal_button(self) -> None:
        """Label the toggle with the unread count, and restyle it if it changed."""
        count = self._journal_unread
        self.journal_button.setText(f"✎ {count}" if count else "✎")
        if self._journal_open:
            tip = "Hide the journal"
        elif count:
            noun = "entry" if count == 1 else "entries"
            tip = f"Show the journal — {count} new {noun}"
        else:
            tip = "Show the journal: what Chiron has written down"
        self.journal_button.setToolTip(tip)
        self.journal_button.setProperty("unread", "true" if count else "false")
        # Qt does not re-evaluate a property selector on its own.
        self.journal_button.style().unpolish(self.journal_button)
        self.journal_button.style().polish(self.journal_button)

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
        """Add a journal entry to the drawer, counting it if the drawer is shut.

        Args:
            entry (JournalEntry): What was just written down.
        """
        self.journal_drawer.append(entry)
        if not self._journal_open:
            self._journal_unread += 1
            self._refresh_journal_button()

    def set_journal(self, entries: list[JournalEntry]) -> None:
        """Replace everything in the drawer, oldest first, and clear the count."""
        self.journal_drawer.set_entries(entries)
        self._journal_unread = 0
        self._refresh_journal_button()

    def clear_journal(self) -> None:
        """Empty the drawer, for a new gameplay session."""
        self.journal_drawer.clear()
        self._journal_unread = 0
        self._refresh_journal_button()

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


__all__ = [
    "BODY_SPACING",
    "MAX_MESSAGES",
    "PLAY_VIEW",
    "SESSION_VIEW",
    "OverlayWindow",
    "format_message_text",
]
