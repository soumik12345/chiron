"""The journal drawer: what Chiron has written down, beside what it is saying.

The journal used to appear inline in the transcript, dimmed, interleaved with the
conversation. That put the two most different things in the panel on the same
axis: the transcript is a conversation you read once, and the journal is a state
you *scan* — "does it know I already killed the boss?" is a question about the
list as a whole, not about the entry that happened to arrive last. Interleaved,
answering it meant scrolling past your own questions, and a long answer could
push the last three entries out of sight entirely.

So it is a column of its own, newest first, and the transcript is purely the
conversation again. Two consequences are deliberate:

* **Newest first.** The drawer is the one place in the panel that is not a
  chronology being followed — it is a memory being audited, and the entry that
  matters most is the one that just landed. Nothing has to be scrolled to see it.
* **The count survives being closed.** Moving the journal out of the transcript
  removes the one signal that it was being written at all, so the toggle carries
  an unread count. That is :class:`~chiron.ui.overlay.OverlayWindow`'s job, not
  this widget's — the drawer only ever renders what it holds.

Rendering is pure, in the same way :mod:`chiron.sessions.render` is: entries in,
HTML out, no display needed to assert on it.
"""

from __future__ import annotations

import html

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QTextBrowser,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from chiron.journal.log import JournalEntry
from chiron.ui.theme import JOURNAL_CATEGORY_COLOURS, PALETTE

#: Entries the drawer renders. The log itself holds far more (500 by default);
#: this is the point past which scrolling stops being how anyone finds anything,
#: and the recorded session is the place to read a whole evening back.
MAX_DRAWER_ENTRIES = 200

#: Point size of the blank paragraph that separates two entries. Entries are
#: told apart by the space between them rather than by rules, so this is the
#: one number that decides whether the column reads as a list or as a wall.
GAP_POINTS = 7


def category_colour(category: str) -> str:
    """The accent colour for a journal category.

    Categories are free-form strings — :data:`~chiron.journal.log.CATEGORIES` is
    a hint to the model, not a schema — so anything unrecognised falls back to
    the dim text colour rather than being dropped or coloured at random.

    Args:
        category (str): The entry's category.

    Returns:
        str: A hex colour from the palette.
    """
    return JOURNAL_CATEGORY_COLOURS.get(category.strip().lower(), PALETTE["text_dim"])


def render_empty() -> str:
    """The drawer with nothing in it yet.

    An empty column is the first thing most players will see here, and a lone
    dim sentence in a wide void is what made the first version read as leftover
    space rather than an instrument. So it says what the column is *for* — the
    answer to "why is this here" is the only useful thing an empty state has.
    """
    # No watermark glyph: whatever font X11 falls back to for ✎ will not scale
    # up, so a large one renders at body size and reads as a stray paperclip.
    # Two lines of type say the same thing and say it cleanly.
    return (
        f'<p align="center" style="margin:34px 0 0 0;color:{PALETTE["text_dim"]}">'
        f"Nothing written down yet.</p>"
        f'<p align="center" style="margin:6px 10px 0 10px;'
        f'color:{PALETTE["text_faint"]}">'
        f"Chiron writes down what will still matter in an hour.</p>"
    )


def render_entries(entries: list[JournalEntry]) -> str:
    """Render journal entries as the drawer's HTML, newest first.

    Each entry is a two-column row — a coloured category on the left, its clock
    time flush right — over the note itself. The colour is carried by a bullet
    and the category word rather than by the note, so a column of six entries
    reads as six *things that happened* at a glance, and the notes stay plain
    text at full contrast. Right-aligning the time takes a table because Qt's
    rich text has no flexbox; it is one table per entry, unnested, which is the
    shape QTextDocument lays out most reliably.

    Args:
        entries (list[JournalEntry]): Entries oldest first, as the log holds
            them. They are reversed here so the newest needs no scrolling.

    Returns:
        str: HTML for the drawer body, or the empty state for no entries.
    """
    if not entries:
        return render_empty()
    blocks: list[str] = []
    for entry in reversed(entries[-MAX_DRAWER_ENTRIES:]):
        colour = category_colour(entry.category)
        blocks.append(
            '<table width="100%" cellspacing="0" cellpadding="0">'
            "<tr>"
            f'<td style="color:{colour}">'
            f"<b>● {html.escape(entry.category.upper())}</b></td>"
            f'<td align="right" style="color:{PALETTE["text_faint"]}">'
            f"{entry.clock}</td>"
            "</tr></table>"
            f'<p style="margin:1px 0 0 0;color:{PALETTE["text"]}">'
            f"{html.escape(entry.note)}</p>"
            # An empty paragraph rather than a bottom margin on the note:
            # QTextDocument drops the margin between a paragraph and the table
            # that follows it, so the entries render packed against each other.
            f'<p style="margin:0;font-size:{GAP_POINTS}pt">&nbsp;</p>'
        )
    return "".join(blocks)


class JournalDrawer(QWidget):
    """The right-hand column listing journal entries as they are written.

    Signals:
        closeRequested (): The ``›`` was clicked — collapse the column.

    Attributes:
        entries (list[JournalEntry]): What is currently shown, oldest first.
    """

    closeRequested = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        """Build an empty drawer."""
        super().__init__(parent)
        self.entries: list[JournalEntry] = []
        # A plain QWidget draws no stylesheet background or border without
        # this, which is what the hairline separating the column from the
        # transcript is made of.
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)

        layout = QVBoxLayout(self)
        # Its own padding rather than the panel's: the column is a surface, and
        # text touching the edge of a surface is what made the first version
        # look like a hairline with things floating beside it.
        layout.setContentsMargins(11, 9, 9, 9)
        layout.setSpacing(0)

        header = QHBoxLayout()
        header.setSpacing(6)
        self.title = QLabel("JOURNAL", self)
        self.title.setObjectName("drawerTitle")
        header.addWidget(self.title)
        self.count = QLabel("", self)
        self.count.setObjectName("drawerCount")
        header.addWidget(self.count)
        header.addStretch(1)

        self.close_button = QToolButton(self)
        self.close_button.setObjectName("drawerClose")
        self.close_button.setText("›")
        self.close_button.setToolTip("Collapse the journal")
        self.close_button.clicked.connect(self.closeRequested.emit)
        header.addWidget(self.close_button)
        layout.addLayout(header)

        # A rule under the title, so the stream below reads as its contents
        # rather than as more things in the same space.
        self.rule = QFrame(self)
        self.rule.setObjectName("drawerRule")
        self.rule.setFixedHeight(1)
        layout.addSpacing(7)
        layout.addWidget(self.rule)
        layout.addSpacing(4)

        self.body = QTextBrowser(self)
        self.body.setObjectName("journalBody")
        self.body.setFrameShape(QFrame.Shape.NoFrame)
        self.body.setOpenExternalLinks(False)
        layout.addWidget(self.body, stretch=1)

        self._render()

    def append(self, entry: JournalEntry) -> None:
        """Add one entry and re-render.

        Args:
            entry (JournalEntry): The entry just written.
        """
        self.entries.append(entry)
        if len(self.entries) > MAX_DRAWER_ENTRIES:
            del self.entries[: len(self.entries) - MAX_DRAWER_ENTRIES]
        self._render()

    def set_entries(self, entries: list[JournalEntry]) -> None:
        """Replace everything shown, oldest first.

        Used to backfill the drawer from the log — opening it for the first time
        in an evening should not show an empty column when the journal is full.
        """
        self.entries = list(entries)[-MAX_DRAWER_ENTRIES:]
        self._render()

    def clear(self) -> None:
        """Forget every entry, for a new gameplay session."""
        self.entries.clear()
        self._render()

    def _render(self) -> None:
        """Rebuild the body and pin the newest entry in view."""
        # Hidden rather than blanked: the count is a styled chip, and an empty
        # one still paints a stub beside the title.
        self.count.setVisible(bool(self.entries))
        self.count.setText(str(len(self.entries)))
        self.body.setHtml(render_entries(self.entries))
        # Newest first, so the top is where the interesting end is.
        self.body.verticalScrollBar().setValue(0)


__all__ = [
    "GAP_POINTS",
    "MAX_DRAWER_ENTRIES",
    "JournalDrawer",
    "category_colour",
    "render_empty",
    "render_entries",
]
