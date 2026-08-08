"""The recorder: an observer of the app that writes the session down.

One object, owned by :class:`~chiron.app.ChironApp` and subscribed in its
``_connect()`` to signals that already existed. That relationship is the whole
design constraint: **the recorder observes, it never participates.** It cannot
refuse a frame, delay an answer, or raise into gameplay—every public method
swallows its own failures—and it outlives Observer reconnects and Responder
rebuilds because a gameplay session spans both agents.

Two consequences worth stating:

* **Nothing is written until something happens.** Launching Chiron opens no
  session; the first watch-start or first message creates one. An overlay left
  running all day while nobody plays leaves no trace, which is the honest
  behaviour for a screen recorder.
* **Appends are buffered.** Events land in a list and are flushed on a short
  timer, on session close and on shutdown. A hard crash therefore loses at most
  a couple of seconds, and the torn-tail-tolerant reader in
  :mod:`chiron.sessions.store` shrugs at the half-written line it leaves.

The counters kept here mirror :func:`~chiron.sessions.store.summarise` exactly —
incrementally while a session is open, by rescan afterwards — which is what lets
the index be a cache rather than a second source of truth.
"""

from __future__ import annotations

import logging
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

from PySide6.QtCore import QObject, QTimer, Signal

from chiron.capture.frames import Frame
from chiron.journal.log import JournalEntry
from chiron.sessions import events as ev
from chiron.sessions.store import (
    CostRollup,
    SessionIndex,
    SessionRow,
    SessionStore,
    default_title,
    new_session_id,
    sessions_root,
)

logger = logging.getLogger(__name__)

#: How often buffered events reach the disk. Short enough that a crash costs
#: seconds, long enough that several agent events share one write.
FLUSH_INTERVAL_MS = 2000

#: How often ``index.json`` is rewritten while a session is recording. Slower
#: than the event flush on purpose: the index is a whole-file rewrite and only
#: matters to a browser the player is not looking at mid-fight.
INDEX_INTERVAL_MS = 15_000

#: Frames remembered, so one image delivered to both agents reuses its thumbnail
#: rather than writing a second copy. The
#: ring buffer upstream is smaller than this, so in practice every repeat is
#: caught.
_FRAME_MEMORY = 64


def _frame_key(frame: Frame) -> tuple[float, int, int, int]:
    """A value identity for a frame, safe to remember after it is released.

    Deliberately *not* :func:`id`, which was the first and wrong answer: a frame
    that goes out of scope frees its address, CPython hands the same address to
    the next frame, and the memo then reports a brand-new capture as one it has
    already stored. Capture time plus shape is unique per grab and stays true.
    """
    return (frame.captured_at, frame.width, frame.height, len(frame.jpeg))


class SessionRecorder(QObject):
    """Records one gameplay session at a time to disk.

    Signals:
        sessionChanged (str, str): A session was opened, renamed or closed,
            carrying its id and title (both empty when none is open).
        costChanged (float, bool): The running total in USD, and whether any of
            it is estimated rather than measured.

    Attributes:
        root (Path): The sessions directory being written to.
        index (SessionIndex): The browsable list, kept current as work happens.
    """

    sessionChanged = Signal(str, str)
    costChanged = Signal(float, bool)

    def __init__(
        self,
        root: str | Path | None = None,
        parent: QObject | None = None,
        *,
        flush_interval_ms: int = FLUSH_INTERVAL_MS,
        thumbnail_min_interval: float = 0.0,
    ) -> None:
        """Create a recorder with no session open.

        Args:
            root (str | Path | None): Sessions directory. Defaults to
                :func:`~chiron.sessions.store.sessions_root`.
            parent (QObject | None): Qt parent.
            flush_interval_ms (int): How often buffered events are written.
            thumbnail_min_interval (float): Minimum seconds between stored
                thumbnails. 0 stores every frame that reached a model, which is
                v2's accepted default; raising it is the escape hatch if real
                sessions bloat, since live mode can send a frame every couple of
                seconds for hours.
        """
        super().__init__(parent)
        self.root = Path(root) if root is not None else sessions_root()
        self.index = SessionIndex(self.root)
        self.thumbnail_min_interval = max(0.0, thumbnail_min_interval)

        self._store: SessionStore | None = None
        self._row: SessionRow | None = None
        self._rollup = CostRollup()
        self._buffer: list[ev.SessionEvent] = []
        self._frame_ids: dict[tuple[float, int, int, int], str] = {}
        self._frame_order: list[tuple[float, int, int, int]] = []
        self._frame_counter = 0
        self._last_thumbnail_at = 0.0
        self._watch_started: float | None = None

        self._flush_timer = QTimer(self)
        self._flush_timer.setInterval(max(200, flush_interval_ms))
        self._flush_timer.timeout.connect(self.flush)
        self._index_timer = QTimer(self)
        self._index_timer.setInterval(INDEX_INTERVAL_MS)
        self._index_timer.timeout.connect(self._save_index)

    # ------------------------------------------------------------ lifecycle

    @property
    def active(self) -> bool:
        """Whether a session is currently open."""
        return self._store is not None

    @property
    def session_id(self) -> str:
        """The open session's id, or empty."""
        return self._store.session_id if self._store is not None else ""

    @property
    def title(self) -> str:
        """The open session's title, or empty."""
        return self._row.title if self._row is not None else ""

    @property
    def cost(self) -> CostRollup:
        """What the open session has spent so far."""
        return self._rollup

    def ensure_session(self, *, game: str = "", mode: str = "dual_agent") -> str:
        """Open a session if none is open, and return its id.

        Called from the two moments that count as the player actually starting
        something: watching the screen, and asking a question. Idle launch is
        deliberately not one of them.

        Args:
            game (str): The detected or configured game, for the auto-title.
            mode (str): Runtime topology at creation; v3 uses ``dual_agent``.

        Returns:
            str: The open session's id, which may be one that already existed.
        """
        if self._store is not None:
            return self._store.session_id
        now = time.time()
        try:
            session_id = new_session_id(game=game, when=now)
            store = SessionStore.create(self.root, session_id)
        except OSError as error:
            logger.warning("Could not create a session directory: %s", error)
            return ""

        title = default_title(game=game, when=now)
        self._store = store
        self._row = SessionRow(
            id=session_id,
            title=title,
            game=game,
            mode=mode,
            created_at=now,
            last_active_at=now,
            # Nothing has been seen yet, so nothing has been stored. The flag
            # turns on with the first thumbnail and off again if they are
            # deleted to reclaim disk.
            has_thumbnails=False,
        )
        self._rollup = CostRollup()
        self._frame_ids.clear()
        self._frame_order.clear()
        self._frame_counter = 0
        self._last_thumbnail_at = 0.0
        self._watch_started = None

        self._append(
            "session_meta",
            ev.session_meta(
                session_id=session_id,
                created_at=now,
                title=title,
                mode=mode,
                game=game,
            ),
            ts=now,
        )
        self.flush()
        self._save_index()
        self._flush_timer.start()
        self._index_timer.start()
        self.sessionChanged.emit(session_id, title)
        logger.info("Opened gameplay session %s (%s)", session_id, mode)
        return session_id

    def close_session(self) -> None:
        """Flush, finalise the index row, and forget the open session.

        Safe to call when nothing is open, which is what makes it usable as the
        first half of "start a new session" without a state check at every call
        site.
        """
        if self._store is None:
            return
        if self._watch_started is not None:
            self.record_watch(False)
        self._flush_timer.stop()
        self._index_timer.stop()
        self.flush()
        self._save_index()
        logger.info("Closed gameplay session %s", self._store.session_id)
        self._store = None
        self._row = None
        self._rollup = CostRollup()
        self.sessionChanged.emit("", "")
        self.costChanged.emit(0.0, False)

    def rename(self, title: str) -> None:
        """Retitle the open session."""
        text = (title or "").strip()
        if self._row is None or not text:
            return
        self._row.title = text
        self._save_index()
        self.sessionChanged.emit(self._row.id, text)

    def shutdown(self) -> None:
        """Persist everything on the way out."""
        self.close_session()

    # ---------------------------------------------------------------- events

    def record_watch(self, watching: bool) -> None:
        """Note that watching started or stopped.

        Watch spans and sessions are independent axes — one session covers a
        whole evening of starting and stopping — so this is an event inside the
        record rather than a boundary of it.
        """
        if self._store is None:
            return
        now = time.time()
        if watching:
            if self._watch_started is not None:
                return
            self._watch_started = now
            self._append("watch_started", ev.watch_span(watching=True), ts=now)
        else:
            if self._watch_started is None:
                return
            if self._row is not None:
                self._row.watched_seconds += max(0.0, now - self._watch_started)
            self._watch_started = None
            self._append("watch_stopped", ev.watch_span(watching=False), ts=now)

    def record_game(
        self, *, label: str, identity: str = "", described: str = "", changed: bool
    ) -> None:
        """Note the game the active-window tracker is reporting."""
        if self._store is None:
            return
        if self._row is not None and not self._row.game:
            self._row.game = label
        self._append(
            "game_changed" if changed else "game_detected",
            ev.game(label=label, identity=identity, described=described),
        )

    def record_message(
        self,
        role: str,
        text: str,
        *,
        frames: list[str] | None = None,
        call_id: str | None = None,
        agent_id: str | None = None,
    ) -> None:
        """Record one conversation turn.

        Frames attached to a question are separate, agent-attributed ``frame``
        events. The viewer interleaves them by timestamp.
        """
        if self._store is None or not (text or "").strip():
            return
        if self._row is not None:
            self._row.message_count += 1
        self._append(
            "message",
            ev.message(
                role=role,
                text=text,
                frames=frames,
                call_id=call_id,
                agent_id=agent_id,
            ),
        )

    def record_journal_entry(self, entry: JournalEntry) -> None:
        """Record a journal entry, verbatim."""
        if self._store is None:
            return
        if self._row is not None:
            self._row.journal_count += 1
        self._append(
            "journal_entry",
            ev.journal_entry(
                timestamp=entry.timestamp,
                note=entry.note,
                category=entry.category,
                source=entry.source,
            ),
            ts=entry.timestamp,
        )

    def record_observer_run(self, payload: dict[str, Any]) -> None:
        """Record one successful buffered review and mirror rescan counters."""
        if self._store is None:
            return
        if self._row is not None:
            self._row.observer_runs += 1
        self._append("observer_run", ev.observer_run(**dict(payload)))

    def record_frame(
        self, frame: Frame, reason: str, *, agent_id: str | None = None
    ) -> str:
        """Store a thumbnail of a frame that reached a model, and note it.

        Args:
            frame (Frame): The frame that was sent.
            reason (str): ``question``, ``scheduled`` or ``immediate``.

        Returns:
            str: The frame id, or empty when nothing was stored — either no
                session is open, or the thumbnail rate cap declined this one.
        """
        if self._store is None:
            return ""
        known = self._frame_ids.get(_frame_key(frame))
        if known:
            self._append_frame_event(
                frame,
                known,
                reason,
                f"{known}.jpg",
                agent_id=agent_id,
            )
            return known
        now = time.time()
        if (
            self.thumbnail_min_interval
            and self._last_thumbnail_at
            and now - self._last_thumbnail_at < self.thumbnail_min_interval
        ):
            return ""

        self._frame_counter += 1
        frame_id = f"{self._frame_counter:06d}"
        path = self._store.write_thumbnail(frame_id, frame.jpeg)
        if path is None:
            return ""
        self._last_thumbnail_at = now
        self._remember_frame(frame, frame_id)
        self._append_frame_event(frame, frame_id, reason, path.name, agent_id=agent_id)
        return frame_id

    def _append_frame_event(
        self,
        frame: Frame,
        frame_id: str,
        reason: str,
        thumbnail: str,
        *,
        agent_id: str | None,
    ) -> None:
        """Attribute one delivery while allowing its thumbnail to be reused."""
        if self._row is not None:
            self._row.frame_count += 1
            self._row.has_thumbnails = True
        self._append(
            "frame",
            ev.frame(
                frame_id=frame_id,
                captured_at=frame.captured_at,
                width=frame.width,
                height=frame.height,
                reason=reason,
                thumbnail=thumbnail,
                agent_id=agent_id,
            ),
            ts=frame.captured_at,
        )

    def record_llm_call(self, record: Any) -> None:
        """Record one priced call and fold it into the running cost.

        Accepts a :class:`~chiron.models.usage.LLMCallRecord` or a plain dict, so
        the live provider's estimated records and litellm's measured ones take
        the same route in.
        """
        if self._store is None:
            return
        payload = (
            record.model_dump() if hasattr(record, "model_dump") else dict(record or {})
        )
        # Whoever installed the sink may not have known which gameplay session
        # was open, so the recorder—which does know—fills it in.
        payload.setdefault("session_id", None)
        if not payload["session_id"]:
            payload["session_id"] = self._store.session_id
        self._rollup.add(payload)
        if self._row is not None:
            self._row.cost = asdict(self._rollup)
        self._append("llm_call", payload)
        self.costChanged.emit(self._rollup.total_usd, self._rollup.is_estimated)

    def record_compaction(self, payload: dict[str, Any]) -> None:
        """Record that memory was deliberately consolidated."""
        if self._store is None:
            return
        if self._row is not None:
            self._row.compactions += 1
        self._append(
            "compaction",
            ev.compaction(
                mode=str(payload.get("mode") or "nonlive"),
                summary=str(payload.get("summary") or ""),
                tokens_before=int(payload.get("tokens_before") or 0),
                tokens_after=int(payload.get("tokens_after") or 0),
                kept_tail_count=int(payload.get("kept_tail_count") or 0),
                dropped_count=int(payload.get("dropped_count") or 0),
                reason=str(payload.get("reason") or ""),
                call_id=payload.get("call_id"),
                agent_id=payload.get("agent_id"),
            ),
        )

    def record_status(
        self, status: str, detail: str = "", *, agent_id: str | None = None
    ) -> None:
        """Record a provider status transition."""
        if self._store is None:
            return
        self._append(
            "status",
            ev.status(status=status, detail=detail, agent_id=agent_id),
        )

    def record_agent_trace(self, payload: dict[str, Any]) -> None:
        """Record diagnostic ReAct events without adding transcript turns."""
        if self._store is None:
            return
        self._append(
            "agent_trace",
            ev.agent_trace(
                agent_id=str(payload.get("agent_id") or "responder"),
                event=payload.get("event"),
            ),
        )

    def record_settings_changed(self, fields: list[str], mode: str) -> None:
        """Record which settings changed mid-session, and the mode afterwards."""
        if self._store is None or not fields:
            return
        self._append("settings_changed", ev.settings_changed(fields=fields, mode=mode))

    # -------------------------------------------------------------- browsing

    def sessions(self) -> list[SessionRow]:
        """Every recorded session, newest first, with the open one current."""
        snapshot = self._row_snapshot()
        if snapshot is not None:
            self.index.put(snapshot)
        return self.index.rows()

    def _row_snapshot(self) -> SessionRow | None:
        """The open session's row as it stands right now, safe to hand out.

        A copy, and a copy with the *in-progress* watch span already added: the
        stored row only accumulates on a watch stop, so a browser opened
        mid-evening would otherwise report the session as watched for zero
        seconds. Adding it to the copy rather than the row is what keeps it from
        being counted twice when watching actually stops.
        """
        if self._row is None or self._store is None:
            return None
        self._row.disk_bytes = self._store.disk_bytes()
        self._row.cost = asdict(self._rollup)
        snapshot = self._row.model_copy(deep=True)
        if self._watch_started is not None:
            snapshot.watched_seconds += max(0.0, time.time() - self._watch_started)
        return snapshot

    def row_for(self, session_id: str) -> SessionRow | None:
        """The summary row for a session, current if it is the open one.

        ``index.json`` is only rewritten every few seconds — it is a whole-file
        write, and nobody is looking at the browser mid-fight — so reading it
        directly for the session being recorded would show the counts as they
        were at the last save, which for a session opened a minute ago means all
        zeroes. Anything that displays a row goes through here.
        """
        if self._store is not None and session_id == self._store.session_id:
            snapshot = self._row_snapshot()
            if snapshot is not None:
                return snapshot
        return self.index.get(session_id)

    def store_for(self, session_id: str) -> SessionStore | None:
        """The store for a recorded session, or None when it is not on disk."""
        directory = self.root / session_id
        return SessionStore(directory) if directory.is_dir() else None

    def read_session(self, session_id: str) -> list[ev.SessionEvent]:
        """Every event of a recorded session, flushing first if it is the open one."""
        if self._store is not None and session_id == self._store.session_id:
            self.flush()
        store = self.store_for(session_id)
        return store.read_events() if store is not None else []

    def delete_session(self, session_id: str) -> None:
        """Delete a session, closing it first if it is the one being recorded."""
        if self._store is not None and session_id == self._store.session_id:
            self.close_session()
        self.index.delete_session(session_id)

    def delete_thumbnails(self, session_id: str) -> int:
        """Delete a session's thumbnails, keeping its text.

        Reclaiming disk from the session currently being recorded is allowed —
        it is the one whose thumbnails are still growing — so the in-memory row
        is corrected too, or the next index write would put the deleted
        thumbnails straight back into the browser's idea of the world.
        """
        if self._store is not None and session_id == self._store.session_id:
            self.flush()
            if self._row is not None:
                self._row.has_thumbnails = False
            self._frame_ids.clear()
            self._frame_order.clear()
        return self.index.delete_thumbnails(session_id)

    def rename_session(self, session_id: str, title: str) -> None:
        """Retitle any session, open or not."""
        if self._store is not None and session_id == self._store.session_id:
            self.rename(title)
            return
        self.index.rename(session_id, title)

    def total_bytes(self) -> int:
        """Disk used by every session."""
        return self.index.total_bytes()

    # ------------------------------------------------------------- internals

    def _append(
        self, event_type: str, payload: dict[str, Any], *, ts: float | None = None
    ) -> None:
        """Buffer one event and keep the row's activity clock current."""
        when = time.time() if ts is None else ts
        self._buffer.append(
            ev.SessionEvent(ts=when, type=event_type, payload=payload)  # type: ignore[arg-type]
        )
        if self._row is not None:
            self._row.last_active_at = max(self._row.last_active_at, when)

    def flush(self) -> None:
        """Write buffered events to disk. Never raises."""
        if self._store is None or not self._buffer:
            return
        pending, self._buffer = self._buffer, []
        try:
            self._store.append(pending)
        except OSError as error:
            logger.warning("Could not write session events: %s", error)

    def _save_index(self) -> None:
        """Update this session's index row and rewrite ``index.json``."""
        snapshot = self._row_snapshot()
        if snapshot is None:
            return
        self.index.put(snapshot)
        self.index.save()

    def _remember_frame(self, frame: Frame, frame_id: str) -> None:
        """Remember a frame's id by value, bounded."""
        key = _frame_key(frame)
        self._frame_ids[key] = frame_id
        self._frame_order.append(key)
        while len(self._frame_order) > _FRAME_MEMORY:
            self._frame_ids.pop(self._frame_order.pop(0), None)


__all__ = ["FLUSH_INTERVAL_MS", "INDEX_INTERVAL_MS", "SessionRecorder"]
