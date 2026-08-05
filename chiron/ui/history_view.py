"""Getting to a past session, and reading one back.

Picking and reading are different sizes of thing, and they used to be the same
size: two full pages in the overlay's stack, both of which grew the panel and
both of which had to be walked back out of. But choosing which evening to look at
is a two-second act — you know which one you want — and paying a page transition,
a panel resize and two presses of Escape for it made the common case cost what
the rare one costs.

So :class:`SessionPicker` is a popup: it drops under the history button, takes
the keyboard while it is up, and closes the moment a row is tapped or Escape is
pressed. The panel never resizes for it, which matters for a window that is
floating over a game. It is still built entirely from ``index.json`` — it never
opens an event file, which is what keeps opening it instant however many evenings
are recorded — and the pinned disk total comes with it, because retention here is
unbounded and manual, so the number *is* the policy.

:class:`SessionViewer` is the one thing that reads events, and only for the row
that was tapped. It is also where a session is now managed: renaming, deleting
and dropping thumbnails live in a ``⋯`` menu in its header rather than in a
right-click on a list row. Destructive actions belong on the thing you are
looking at, not on a row in a list you are skimming — and "delete thumbnails,
keep the text" is a decision nobody can make well without seeing the thumbnails
first.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

from PySide6.QtCore import QPoint, Qt, Signal
from PySide6.QtGui import QGuiApplication, QKeyEvent
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMenu,
    QTextBrowser,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from chiron.sessions.events import SessionEvent
from chiron.sessions.render import (
    cost_breakdown,
    describe_row,
    format_bytes,
    render_events,
    summary_line,
)
from chiron.sessions.store import SessionRow
from chiron.ui.theme import PALETTE

#: Role the session id is stashed under on each list item.
_ID_ROLE = Qt.ItemDataRole.UserRole

#: Popup size. Narrower than the panel would be silly, and taller than this
#: turns a picker back into the page it replaced.
PICKER_MIN_WIDTH = 340
PICKER_HEIGHT = 380


class SessionPicker(QWidget):
    """A dropdown listing every recorded session, newest first.

    A top-level ``Qt.Popup`` rather than a page in the overlay's stack: it takes
    the keyboard while it is up, closes on Escape or an outside click, and
    leaves the panel's geometry alone.

    Signals:
        sessionOpened (str): A row was tapped; open it in the viewer.
    """

    sessionOpened = Signal(str)

    def __init__(self, parent: QWidget | None = None) -> None:
        """Build an empty, hidden picker."""
        super().__init__(parent)
        self._rows: list[SessionRow] = []

        self.setWindowFlags(Qt.WindowType.Popup)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        panel = QFrame(self)
        panel.setObjectName("panel")
        outer.addWidget(panel)

        layout = QVBoxLayout(panel)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        self.search = QLineEdit(panel)
        self.search.setObjectName("input")
        self.search.setPlaceholderText("Search sessions…")
        self.search.textChanged.connect(self._refilter)
        self.search.returnPressed.connect(self._open_first)
        layout.addWidget(self.search)

        self.list = QListWidget(panel)
        self.list.setObjectName("sessions")
        self.list.setFrameShape(QFrame.Shape.NoFrame)
        self.list.setWordWrap(True)
        # Rows already wrap, so a horizontal bar is never the way to read one —
        # it only ever appears because a row is a pixel wider than the viewport.
        self.list.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.list.itemActivated.connect(self._open)
        self.list.itemClicked.connect(self._open)
        layout.addWidget(self.list, stretch=1)

        self.total = QLabel("", panel)
        self.total.setObjectName("footer")
        layout.addWidget(self.total)

    # ------------------------------------------------------------- content

    def set_sessions(self, rows: Iterable[SessionRow], *, total_bytes: int = 0) -> None:
        """Replace the list, preserving the current search term.

        Args:
            rows (Iterable[SessionRow]): Rows straight from the index, newest
                first.
            total_bytes (int): Disk used by every session, for the pinned line.
        """
        self._rows = list(rows)
        self._refilter()
        count = len(self._rows)
        noun = "session" if count == 1 else "sessions"
        self.total.setText(
            f"{count} {noun}  ·  {format_bytes(total_bytes)} on disk"
            if count
            else "No sessions recorded yet."
        )

    def selected_id(self) -> str:
        """The id of the highlighted row, or empty."""
        item = self.list.currentItem()
        return str(item.data(_ID_ROLE)) if item is not None else ""

    def _refilter(self) -> None:
        """Rebuild the list from the rows that match the search box."""
        query = self.search.text()
        self.list.clear()
        for row in self._rows:
            if not row.matches(query):
                continue
            item = QListWidgetItem(f"{row.title or row.id}\n{describe_row(row)}")
            item.setData(_ID_ROLE, row.id)
            item.setToolTip(row.id)
            self.list.addItem(item)

    # ----------------------------------------------------------- popping up

    def open_under(self, anchor: QWidget) -> None:
        """Show the picker beneath `anchor`, kept on screen.

        Args:
            anchor (QWidget): The button that was clicked. The popup is aligned
                to its right edge, since the history button lives at the right
                edge of the session row and a left-aligned popup would hang off
                a panel that is often near the screen edge itself.
        """
        window = anchor.window()
        self.resize(max(PICKER_MIN_WIDTH, window.width() - 24), PICKER_HEIGHT)

        below = anchor.mapToGlobal(QPoint(anchor.width(), anchor.height() + 4))
        position = QPoint(below.x() - self.width(), below.y())

        screen = QGuiApplication.screenAt(below) or QGuiApplication.primaryScreen()
        if screen is not None:
            available = screen.availableGeometry()
            position.setX(
                max(
                    available.left(),
                    min(position.x(), available.right() - self.width()),
                )
            )
            if position.y() + self.height() > available.bottom():
                # No room below: flip above the button rather than off-screen.
                above = anchor.mapToGlobal(QPoint(0, 0)).y() - self.height() - 4
                position.setY(max(available.top(), above))

        self.move(position)
        self.search.clear()
        self.show()
        self.raise_()
        self.search.setFocus(Qt.FocusReason.PopupFocusReason)

    def _open(self, item: QListWidgetItem) -> None:
        """Emit the id of an activated row and close."""
        session_id = str(item.data(_ID_ROLE) or "")
        if session_id:
            self.close()
            self.sessionOpened.emit(session_id)

    def _open_first(self) -> None:
        """Enter in the search box opens the top match."""
        item = self.list.currentItem() or self.list.item(0)
        if item is not None:
            self._open(item)

    def keyPressEvent(self, event: QKeyEvent) -> None:
        """Escape closes; Down walks from the search box into the list."""
        if event.key() == Qt.Key.Key_Escape:
            self.close()
            event.accept()
            return
        if event.key() == Qt.Key.Key_Down and self.search.hasFocus():
            if self.list.count():
                self.list.setCurrentRow(0)
                self.list.setFocus(Qt.FocusReason.OtherFocusReason)
            event.accept()
            return
        super().keyPressEvent(event)


class SessionViewer(QWidget):
    """One recorded session, read-only, thumbnails inline.

    Signals:
        backRequested (): Return to the play view.
        renameRequested (str): Rename was chosen from the ``⋯`` menu.
        deleteRequested (str): Delete was chosen.
        thumbnailsDeleteRequested (str): Delete-thumbnails-keep-text was chosen.
    """

    backRequested = Signal()
    renameRequested = Signal(str)
    deleteRequested = Signal(str)
    thumbnailsDeleteRequested = Signal(str)

    def __init__(self, parent: QWidget | None = None) -> None:
        """Build an empty viewer."""
        super().__init__(parent)
        self.session_id = ""

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        header = QHBoxLayout()
        header.setSpacing(6)
        self.back_button = QToolButton(self)
        self.back_button.setText("‹")
        self.back_button.setToolTip("Back to the game (Esc)")
        self.back_button.clicked.connect(self.backRequested.emit)
        header.addWidget(self.back_button)
        self.title = QLabel("", self)
        self.title.setObjectName("brand")
        header.addWidget(self.title, stretch=1)

        self.manage_button = QToolButton(self)
        self.manage_button.setText("⋯")
        self.manage_button.setToolTip("Rename or delete this session")
        self.manage_button.clicked.connect(self._manage)
        header.addWidget(self.manage_button)
        layout.addLayout(header)

        self.summary = QLabel("", self)
        self.summary.setObjectName("footer")
        self.summary.setWordWrap(True)
        layout.addWidget(self.summary)

        self.body = QTextBrowser(self)
        self.body.setObjectName("transcript")
        self.body.setFrameShape(QFrame.Shape.NoFrame)
        self.body.setOpenExternalLinks(False)
        layout.addWidget(self.body, stretch=1)

        self.clear()

    def show_session(
        self,
        row: SessionRow,
        events: Iterable[SessionEvent],
        *,
        frames_directory: Path | None = None,
    ) -> None:
        """Render one recorded session.

        Args:
            row (SessionRow): The index row, for the title and the pinned line.
            events (Iterable[SessionEvent]): Its events.
            frames_directory (Path | None): Where its thumbnails live, if any
                survive.
        """
        self.session_id = row.id
        self.title.setText(row.title or row.id)
        breakdown = cost_breakdown(row)
        detail = f"  ·  {' · '.join(breakdown[:3])}" if breakdown else ""
        self.summary.setText(f"{summary_line(row)}{detail}")
        self.body.setHtml(render_events(events, frames_directory=frames_directory))
        self.body.verticalScrollBar().setValue(0)
        self.manage_button.setEnabled(True)

    def clear(self) -> None:
        """Forget whatever was being shown."""
        self.session_id = ""
        self.title.setText("")
        self.summary.setText("")
        self.body.setHtml(
            f'<p style="color:{PALETTE["text_faint"]}">Nothing selected.</p>'
        )
        self.manage_button.setEnabled(False)

    def _manage(self) -> None:
        """Offer rename and the two kinds of delete for the open session."""
        if not self.session_id:
            return
        menu = QMenu(self)
        rename = menu.addAction("Rename…")
        thumbs = menu.addAction("Delete thumbnails, keep the text")
        menu.addSeparator()
        delete = menu.addAction("Delete this session")
        chosen = menu.exec(
            self.manage_button.mapToGlobal(QPoint(0, self.manage_button.height()))
        )
        if chosen is rename:
            self.renameRequested.emit(self.session_id)
        elif chosen is thumbs:
            self.thumbnailsDeleteRequested.emit(self.session_id)
        elif chosen is delete:
            self.deleteRequested.emit(self.session_id)


__all__ = ["PICKER_HEIGHT", "PICKER_MIN_WIDTH", "SessionPicker", "SessionViewer"]
