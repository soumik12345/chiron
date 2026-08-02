"""The two views that are not the game: the session list, and one session read back.

Both live inside the overlay rather than in a window of their own. A separate
window would need its own placement, its own always-on-top decision and its own
way back; a third page in the panel that is already floating over the game needs
none of those, and Escape walking back through a small stack is a navigation
model nobody has to be taught.

:class:`HistoryView` is built entirely from ``index.json`` — it never opens an
event file, which is what keeps opening the list instant however many evenings
are recorded. :class:`SessionViewer` is the one thing that does read events, and
only for the row that was tapped.

The pinned total at the bottom of the list is deliberate: retention here is
unbounded and manual, so the disk figure *is* the policy. A number the player
sees growing is what makes "delete thumbnails, keep the text" a decision they
can make in time.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

from PySide6.QtCore import Qt, Signal
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


class HistoryView(QWidget):
    """A searchable list of every recorded session.

    Signals:
        sessionOpened (str): A row was activated; open it in the viewer.
        renameRequested (str): Rename was chosen from the context menu.
        deleteRequested (str): Delete was chosen.
        thumbnailsDeleteRequested (str): Delete-thumbnails-keep-text was chosen.
        backRequested (): The player wants the play view again.
    """

    sessionOpened = Signal(str)
    renameRequested = Signal(str)
    deleteRequested = Signal(str)
    thumbnailsDeleteRequested = Signal(str)
    backRequested = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        """Build an empty history view."""
        super().__init__(parent)
        self._rows: list[SessionRow] = []

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

        self.search = QLineEdit(self)
        self.search.setObjectName("input")
        self.search.setPlaceholderText("Search sessions…")
        self.search.textChanged.connect(self._refilter)
        header.addWidget(self.search, stretch=1)
        layout.addLayout(header)

        self.list = QListWidget(self)
        self.list.setObjectName("sessions")
        self.list.setFrameShape(QFrame.Shape.NoFrame)
        self.list.setWordWrap(True)
        self.list.itemActivated.connect(self._open)
        self.list.itemClicked.connect(self._open)
        self.list.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.list.customContextMenuRequested.connect(self._menu)
        layout.addWidget(self.list, stretch=1)

        self.total = QLabel("", self)
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

    def _open(self, item: QListWidgetItem) -> None:
        """Emit the id of an activated row."""
        session_id = str(item.data(_ID_ROLE) or "")
        if session_id:
            self.sessionOpened.emit(session_id)

    def _menu(self, position) -> None:
        """Offer rename and the two kinds of delete for the row under the cursor."""
        item = self.list.itemAt(position)
        if item is None:
            return
        session_id = str(item.data(_ID_ROLE) or "")
        menu = QMenu(self)
        rename = menu.addAction("Rename…")
        thumbs = menu.addAction("Delete thumbnails, keep the text")
        menu.addSeparator()
        delete = menu.addAction("Delete this session")
        chosen = menu.exec(self.list.mapToGlobal(position))
        if chosen is rename:
            self.renameRequested.emit(session_id)
        elif chosen is thumbs:
            self.thumbnailsDeleteRequested.emit(session_id)
        elif chosen is delete:
            self.deleteRequested.emit(session_id)


class SessionViewer(QWidget):
    """One recorded session, read-only, thumbnails inline.

    Signals:
        backRequested (): Return to the history list.
    """

    backRequested = Signal()

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
        self.back_button.setToolTip("Back to history (Esc)")
        self.back_button.clicked.connect(self.backRequested.emit)
        header.addWidget(self.back_button)
        self.title = QLabel("", self)
        self.title.setObjectName("brand")
        header.addWidget(self.title, stretch=1)
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

    def clear(self) -> None:
        """Forget whatever was being shown."""
        self.session_id = ""
        self.title.setText("")
        self.summary.setText("")
        self.body.setHtml(
            f'<p style="color:{PALETTE["text_faint"]}">Nothing selected.</p>'
        )


__all__ = ["HistoryView", "SessionViewer"]
