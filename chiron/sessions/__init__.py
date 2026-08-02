"""Gameplay sessions: the part of an evening that outlives the process.

v0 and v1 gave Chiron a memory that dies on exit — the journal, the conversation,
the frames and every cent spent evaporate with the process. This package is the
durable half: a **session** is one user-managed stretch of play, recorded as a
typed, append-only event stream plus the thumbnails of every frame that reached a
model, and summarised in one index file the browser reads instead of opening any
of them.

Three modules, in dependency order:

* :mod:`chiron.sessions.events` — the event vocabulary. One line per event,
  ``{ts, type, payload}``, in wall-clock order.
* :mod:`chiron.sessions.store` — where those lines live: one directory per
  session under ``$XDG_DATA_HOME/chiron/sessions/``, an ``index.json`` beside
  them, and torn-tail-tolerant reads so a hard crash costs the last line rather
  than the file.
* :mod:`chiron.sessions.recorder` — :class:`~chiron.sessions.recorder.SessionRecorder`,
  the one object :class:`~chiron.app.ChironApp` owns. It observes the app through
  signals that already exist; it never participates, so it cannot block or break
  gameplay, and it outlives the session provider across a live/non-live swap
  exactly as the journal does.

**Not to be confused with** :mod:`chiron.session` (singular), which is the
*provider* seam — "how Chiron thinks" — and has nothing to do with persistence.
A gameplay session spans many provider connections, many watch spans, and both
modes if the player switches models mid-evening.

The on-disk format is deliberately replayable even though v2 does not replay it:
``journal_entry`` events fold back into a :class:`~chiron.journal.log.JournalLog`,
``message`` events into the non-live history, ``compaction`` events as their
summary. Resume is a v3 feature the format is already shaped for.
"""

from __future__ import annotations

from chiron.sessions.events import EVENT_TYPES, EventType, SessionEvent
from chiron.sessions.recorder import SessionRecorder
from chiron.sessions.store import (
    SessionIndex,
    SessionRow,
    SessionStore,
    new_session_id,
    sessions_root,
)

__all__ = [
    "EVENT_TYPES",
    "EventType",
    "SessionEvent",
    "SessionIndex",
    "SessionRecorder",
    "SessionRow",
    "SessionStore",
    "new_session_id",
    "sessions_root",
]
