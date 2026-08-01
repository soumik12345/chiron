"""Durable agent sessions: append-only JSONL with resume and branching.

A session is a **tree**, not a list. Every entry records its `parent_id`, so the
file is append-only even when you rewind: branching from an earlier entry simply
moves the active leaf back, and the next append forks a new path. Replaying a
session means walking the root→leaf path and applying each entry in order.

Three entry types are written:

* `message` — one OpenAI-format message dict, exactly as the agent stored it.
* `compaction` — records that history was summarised, so replay reproduces the
  compacted list rather than the raw one.
* `info` — free-form metadata (model id, labels) written at session start.

Sessions are cheap, per-run files under `.chiron_data/sessions/`, holding *one agent
conversation* each.
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, ValidationError

logger = logging.getLogger(__name__)

DEFAULT_SESSION_DIR = Path(".chiron_data") / "sessions"

EntryType = Literal["info", "message", "compaction"]

_UNSET: Any = object()


class SessionTreeError(ValueError):
    """Raised when entries do not form a valid traversable tree."""


def _new_id() -> str:
    """Return a short unique entry id."""
    return uuid.uuid4().hex[:12]


class SessionEntry(BaseModel):
    """One append-only record in a session file.

    Attributes:
        id (str): Unique id for this entry.
        parent_id (str | None): The entry this one follows, or None for a root.
        entry_type (EntryType): Which kind of record this is.
        payload (dict): Type-specific data.
        timestamp (float): Unix timestamp when the entry was created.
    """

    id: str = Field(default_factory=_new_id)
    parent_id: str | None = None
    entry_type: EntryType
    payload: dict[str, Any] = Field(default_factory=dict)
    timestamp: float = Field(default_factory=time.time)


def entries_by_id(entries: list[SessionEntry]) -> dict[str, SessionEntry]:
    """Return entries keyed by id, rejecting duplicates.

    Raises:
        SessionTreeError: If two entries share an id.
    """
    result: dict[str, SessionEntry] = {}
    for entry in entries:
        if entry.id in result:
            raise SessionTreeError(f"Duplicate session entry id: {entry.id}")
        result[entry.id] = entry
    return result


def path_to_entry(entries: list[SessionEntry], leaf_id: str) -> list[SessionEntry]:
    """Return the root→leaf path ending at `leaf_id`.

    Raises:
        SessionTreeError: If the leaf is missing or the parent chain cycles.
    """
    by_id = entries_by_id(entries)
    path: list[SessionEntry] = []
    seen: set[str] = set()
    current: str | None = leaf_id

    while current is not None:
        if current in seen:
            raise SessionTreeError(f"Cycle detected at session entry: {current}")
        seen.add(current)
        entry = by_id.get(current)
        if entry is None:
            raise SessionTreeError(f"Missing session entry: {current}")
        path.append(entry)
        current = entry.parent_id

    path.reverse()
    return path


@dataclass
class SessionState:
    """Agent state reconstructed by replaying a session path.

    Attributes:
        messages (list[dict]): The message list as the agent last had it.
        active_leaf_id (str | None): The entry the next append should follow.
        info (dict): Merged payloads of all `info` entries on the path.
        entries (list[SessionEntry]): The replayed path.
    """

    messages: list[dict[str, Any]] = field(default_factory=list)
    active_leaf_id: str | None = None
    info: dict[str, Any] = field(default_factory=dict)
    entries: list[SessionEntry] = field(default_factory=list)

    @classmethod
    def replay(
        cls, entries: list[SessionEntry], leaf_id: str | None | Any = _UNSET
    ) -> SessionState:
        """Rebuild state from entries.

        Args:
            entries (list[SessionEntry]): All entries read from storage.
            leaf_id (str | None): Replay only the root→leaf path ending here. Omit to
                replay every entry in storage order (the common linear case); pass
                None explicitly for the empty pre-root state.

        Returns:
            SessionState: The reconstructed state.
        """
        if leaf_id is _UNSET:
            path = list(entries)
        elif leaf_id is None:
            path = []
        else:
            path = path_to_entry(entries, leaf_id)

        messages: list[dict[str, Any]] = []
        info: dict[str, Any] = {}
        for entry in path:
            if entry.entry_type == "message":
                messages.append(dict(entry.payload["message"]))
            elif entry.entry_type == "compaction":
                messages = _apply_compaction(messages, entry.payload)
            elif entry.entry_type == "info":
                info.update(entry.payload)

        return cls(
            messages=messages,
            active_leaf_id=path[-1].id if path else None,
            info=info,
            entries=path,
        )


def _apply_compaction(
    messages: list[dict[str, Any]], payload: dict[str, Any]
) -> list[dict[str, Any]]:
    """Reproduce a recorded compaction against replayed messages."""
    from chiron.core.context import summary_message

    kept = int(payload.get("kept_tail_count", 0))
    system = messages[:1] if messages and messages[0].get("role") == "system" else []
    tail = messages[-kept:] if kept else []
    return [*system, summary_message(payload.get("summary", "")), *tail]


class JsonlSessionStore:
    """Append-only JSONL session storage with tree-shaped history.

    Attributes:
        path (Path): The JSONL file backing this session.
        active_leaf_id (str | None): The entry the next append will attach to.
    """

    def __init__(self, path: str | Path, *, active_leaf_id: str | None = None) -> None:
        """Open (or create) a session file.

        Existing entries are loaded immediately so `state()` works without an extra
        read. Parent directories are created on demand.

        Args:
            path (str | Path): Path to the `.jsonl` session file.
            active_leaf_id (str | None): Attach point for the next append. Defaults to
                the last entry in the file, i.e. resume where the file left off.
        """
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._entries: list[SessionEntry] = self._read()
        self.active_leaf_id = (
            active_leaf_id
            if active_leaf_id is not None
            else (self._entries[-1].id if self._entries else None)
        )

    @classmethod
    def for_session(
        cls, session_id: str, *, directory: str | Path = DEFAULT_SESSION_DIR
    ) -> JsonlSessionStore:
        """Open the store for `session_id` under `directory`."""
        return cls(Path(directory) / f"{session_id}.jsonl")

    @property
    def entries(self) -> list[SessionEntry]:
        """All entries in the file, in write order."""
        return list(self._entries)

    def _read(self) -> list[SessionEntry]:
        """Parse the backing file, skipping unparseable lines.

        A session that was killed mid-append leaves a truncated final line, which is
        precisely the crash durable sessions exist to survive — so a line that will
        not parse is dropped with a warning rather than being allowed to make the
        whole session unopenable. Dropping a line from the *middle* of a file will
        still surface later as a :class:`SessionTreeError` when the broken parent
        chain is walked, which is the honest failure for genuine corruption.
        """
        if not self.path.exists():
            return []
        entries: list[SessionEntry] = []
        with self.path.open("r", encoding="utf-8") as handle:
            for number, line in enumerate(handle, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    entries.append(SessionEntry.model_validate(json.loads(line)))
                except (json.JSONDecodeError, ValidationError) as e:
                    logger.warning(
                        "Skipping unparseable entry at %s:%d: %s", self.path, number, e
                    )
        return entries

    def append(self, entry_type: EntryType, payload: dict[str, Any]) -> SessionEntry:
        """Append an entry attached to the active leaf and advance the leaf.

        Args:
            entry_type (EntryType): The kind of record to write.
            payload (dict): Type-specific data.

        Returns:
            SessionEntry: The entry that was written.
        """
        entry = SessionEntry(
            parent_id=self.active_leaf_id, entry_type=entry_type, payload=payload
        )
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(entry.model_dump_json() + "\n")
        self._entries.append(entry)
        self.active_leaf_id = entry.id
        return entry

    def append_message(self, message: dict[str, Any]) -> SessionEntry:
        """Record one agent message."""
        return self.append("message", {"message": message})

    def append_compaction(
        self,
        *,
        summary: str,
        tokens_before: int,
        tokens_after: int,
        kept_tail_count: int,
        dropped_count: int,
    ) -> SessionEntry:
        """Record that history was compacted, so replay reproduces the compacted list."""
        return self.append(
            "compaction",
            {
                "summary": summary,
                "tokens_before": tokens_before,
                "tokens_after": tokens_after,
                "kept_tail_count": kept_tail_count,
                "dropped_count": dropped_count,
            },
        )

    def append_info(self, **payload: Any) -> SessionEntry:
        """Record free-form session metadata (model id, labels, …)."""
        return self.append("info", payload)

    def state(self, leaf_id: str | None | Any = _UNSET) -> SessionState:
        """Replay the session.

        Defaults to the active leaf, which is the resume point. Pass an explicit
        `leaf_id` to inspect any other point in the tree without moving the leaf.
        """
        if leaf_id is _UNSET:
            leaf_id = self.active_leaf_id
        return SessionState.replay(self._entries, leaf_id)

    def branch(self, entry_id: str) -> SessionState:
        """Rewind the active leaf to `entry_id` so the next append forks a branch.

        Nothing is deleted — the abandoned path stays in the file and can be replayed
        or branched from again.

        Args:
            entry_id (str): The entry to continue from.

        Returns:
            SessionState: The state as of that entry.

        Raises:
            SessionTreeError: If `entry_id` is not in this session.
        """
        state = SessionState.replay(self._entries, entry_id)
        self.active_leaf_id = entry_id
        return state

    def leaf_ids(self) -> list[str]:
        """Return the ids of entries that have no children (one per branch tip)."""
        parents = {e.parent_id for e in self._entries if e.parent_id is not None}
        return [e.id for e in self._entries if e.id not in parents]


__all__ = [
    "DEFAULT_SESSION_DIR",
    "JsonlSessionStore",
    "SessionEntry",
    "SessionState",
    "SessionTreeError",
    "entries_by_id",
    "path_to_entry",
]
