"""The game journal: timestamped text that outlives the frames it came from.

The live model's visual memory is structurally a few minutes long — at any
capture rate, a 32k context holds only so many frames before the sliding window
starts evicting them. The journal is how meaning survives that eviction. Text is
roughly a hundred times cheaper per unit of meaning than pixels, so a line like::

    12:05 — [death] died to the skeleton on the bridge, lost 2400 souls

keeps mattering long after the frames that produced it are gone.

The format is deliberately dumb: a timestamp, a category, a sentence. That makes
it trivial to render into a session as text, trivial to show in the overlay, and
trivial to persist to SQLite when cross-session memory arrives — none of which
would be true of a richer structure.

The log tracks which entries have already been folded into the live session, so a
periodic fold sends only what is new, while a session rotation can replay the
whole thing as a reconnect seed.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from datetime import datetime

#: Categories the model is asked to choose from. Free-form strings are accepted —
#: this is a hint for consistency, not a schema to reject entries against.
CATEGORIES = (
    "location",
    "objective",
    "combat",
    "death",
    "item",
    "npc",
    "progress",
    "note",
)


@dataclass(frozen=True)
class JournalEntry:
    """One recorded event.

    Attributes:
        timestamp (float): Unix time the event was recorded.
        note (str): What happened, in one sentence.
        category (str): Loose bucket from :data:`CATEGORIES`.
        source (str): Which writer produced this — ``tool_call`` or ``sidecar``.
    """

    timestamp: float
    note: str
    category: str = "note"
    source: str = "tool_call"

    @property
    def clock(self) -> str:
        """The entry time as ``HH:MM``."""
        return datetime.fromtimestamp(self.timestamp).strftime("%H:%M")

    def render(self) -> str:
        """The entry as one journal line."""
        return f"{self.clock} — [{self.category}] {self.note}"


@dataclass
class JournalLog:
    """An in-memory, append-only log of journal entries.

    Both journal writers append here, so the strategy in force is invisible to
    everything downstream: the overlay renders the same list either way, and the
    session manager folds the same text.

    Attributes:
        max_entries (int): Entries retained before the oldest are dropped.
        entries (list[JournalEntry]): Everything currently held, oldest first.
    """

    max_entries: int = 500
    entries: list[JournalEntry] = field(default_factory=list)
    _folded_count: int = field(default=0, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def append(
        self,
        note: str,
        *,
        category: str = "note",
        source: str = "tool_call",
        timestamp: float | None = None,
    ) -> JournalEntry | None:
        """Record one event.

        Args:
            note (str): What happened. Blank notes are ignored.
            category (str): Loose bucket from :data:`CATEGORIES`.
            source (str): Which writer produced the entry.
            timestamp (float | None): Event time; defaults to now.

        Returns:
            JournalEntry | None: The stored entry, or None when `note` was blank.
        """
        text = note.strip()
        if not text:
            return None
        entry = JournalEntry(
            timestamp=time.time() if timestamp is None else timestamp,
            note=text,
            category=(category or "note").strip() or "note",
            source=source,
        )
        with self._lock:
            self.entries.append(entry)
            overflow = len(self.entries) - self.max_entries
            if overflow > 0:
                del self.entries[:overflow]
                self._folded_count = max(0, self._folded_count - overflow)
        return entry

    def recent(self, limit: int = 40) -> list[JournalEntry]:
        """The `limit` most recent entries, oldest first."""
        with self._lock:
            return list(self.entries[-limit:]) if limit > 0 else []

    def unfolded(self) -> list[JournalEntry]:
        """Entries appended since the last :meth:`mark_folded`."""
        with self._lock:
            return list(self.entries[self._folded_count :])

    def mark_folded(self) -> None:
        """Record that everything currently held has been sent to the session."""
        with self._lock:
            self._folded_count = len(self.entries)

    def render(self, entries: list[JournalEntry] | None = None) -> str:
        """Render entries as newline-separated journal lines.

        Args:
            entries (list[JournalEntry] | None): What to render. Defaults to the
                whole log.

        Returns:
            str: One line per entry, oldest first; empty string for no entries.
        """
        source = self.entries if entries is None else entries
        with self._lock:
            rows = list(source)
        return "\n".join(entry.render() for entry in rows)

    def clear(self) -> None:
        """Drop every entry (used when starting a fresh play session)."""
        with self._lock:
            self.entries.clear()
            self._folded_count = 0

    def __len__(self) -> int:
        """Number of entries currently held."""
        with self._lock:
            return len(self.entries)


__all__ = ["CATEGORIES", "JournalEntry", "JournalLog"]
