"""The single write path and read-only snapshot seam for the gameplay journal."""

from __future__ import annotations

import threading
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from chiron.journal.log import CATEGORIES, JournalEntry, JournalLog

RECORD_EVENT_DECLARATION: dict[str, Any] = {
    "name": "record_event",
    "description": (
        "Record one durable gameplay fact that will still matter later. Ignore "
        "routine motion, transient UI state, uncertainty, and repeats."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "note": {
                "type": "string",
                "description": "One concrete, self-contained sentence.",
            },
            "category": {
                "type": "string",
                "enum": list(CATEGORIES),
                "description": "Loose event category.",
            },
        },
        "required": ["note"],
        "additionalProperties": False,
    },
}

EntryCallback = Callable[[JournalEntry], None]


@dataclass(frozen=True)
class JournalSnapshot:
    """Immutable model-facing summary plus verbatim recent journal tail."""

    entries: tuple[JournalEntry, ...]
    summary: str = ""
    total_entries: int = 0
    summarized_entries: int = 0
    generation: int = 0

    def render(self) -> str:
        parts: list[str] = []
        if self.summary.strip():
            parts.append(f"Earlier journal summary:\n{self.summary.strip()}")
        if self.entries:
            label = "Recent journal entries:" if parts else "Gameplay journal:"
            parts.append(
                label + "\n" + "\n".join(entry.render() for entry in self.entries)
            )
        return "\n\n".join(parts)

    def __len__(self) -> int:
        return self.total_entries


class JournalReader:
    """The Responder's deliberately mutation-free journal capability."""

    def __init__(self, service: JournalService) -> None:
        self._service = service

    def snapshot(self) -> JournalSnapshot:
        return self._service.snapshot()


class JournalService:
    """Append once, notify once, and expose no mutation to the Responder."""

    def __init__(
        self,
        log: JournalLog,
        on_entry: EntryCallback | None = None,
    ) -> None:
        self.log = log
        self.on_entry = on_entry
        self._summary = ""
        self._summarized_entries = 0
        self._generation = 0
        self._memory_lock = threading.Lock()

    @property
    def function_declarations(self) -> list[dict[str, Any]]:
        """The Observer's only callable function."""
        return [RECORD_EVENT_DECLARATION]

    def record(
        self,
        note: str,
        category: str = "note",
        *,
        source: str = "observer",
        timestamp: float | None = None,
    ) -> JournalEntry | None:
        entry = self.log.append(
            note,
            category=category,
            source=source,
            timestamp=timestamp,
        )
        if entry is not None and self.on_entry is not None:
            self.on_entry(entry)
        return entry

    async def handle_observer_tool(
        self, name: str, args: dict[str, Any]
    ) -> dict[str, Any]:
        """Execute the Observer's write-only tool call."""
        if name != "record_event":
            return {"error": f"Unknown Observer function: {name}"}
        entry = self.record(
            str(args.get("note") or ""),
            str(args.get("category") or "note"),
        )
        if entry is None:
            return {"status": "ignored", "reason": "empty note"}
        return {"status": "recorded", "at": entry.clock}

    def snapshot(self) -> JournalSnapshot:
        """Return the derived summary and unsummarised tail used by models."""
        entries = self.log.snapshot()
        with self._memory_lock:
            cursor = min(self._summarized_entries, len(entries))
            summary = self._summary
            generation = self._generation
        return JournalSnapshot(
            entries=tuple(entries[cursor:]),
            summary=summary,
            total_entries=len(entries),
            summarized_entries=cursor,
            generation=generation,
        )

    def apply_summary(
        self, summary: str, until: int, *, generation: int | None = None
    ) -> bool:
        """Replace derived context through `until` without deleting raw entries."""
        text = summary.strip()
        if not text:
            return False
        total = len(self.log)
        with self._memory_lock:
            if generation is not None and generation != self._generation:
                return False
            self._summary = text
            self._summarized_entries = max(
                self._summarized_entries, min(int(until), total)
            )
        return True

    def reader(self) -> JournalReader:
        """Grant snapshot access without granting the write handler."""
        return JournalReader(self)

    def clear(self) -> None:
        self.log.clear()
        with self._memory_lock:
            self._summary = ""
            self._summarized_entries = 0
            self._generation += 1


__all__ = [
    "CATEGORIES",
    "JournalService",
    "JournalReader",
    "JournalSnapshot",
    "RECORD_EVENT_DECLARATION",
]
