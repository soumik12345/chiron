"""The event vocabulary a gameplay session is recorded in.

Every line of ``events.jsonl`` is one :class:`SessionEvent` — a wall-clock
timestamp, a type, and a payload — appended in the order things happened. There
is no tree, no parent chain and no branching: a recording of an evening is a
list, and the closed vocabulary below is what makes it replayable later without
guessing at what a payload meant.

The types divide into three families:

* **Structure** — ``session_meta`` (always the first line), ``watch_started`` /
  ``watch_stopped``, ``game_detected`` / ``game_changed``, ``status``,
  ``settings_changed``. Sessions and watching are independent axes: one session
  spans many watch spans, and without ``settings_changed`` the cost data is
  uninterpretable a week later.
* **Content** — ``message``, ``journal_entry``, ``observer_run``, ``frame``,
  ``compaction`` and ``agent_trace``. Live v3 sessions emitted no
  ``observer_run``; buffered v4 sessions use it again with an expanded payload.
* **Money** — ``llm_call``, one serialised
  :class:`~chiron.models.usage.LLMCallRecord` per call, linked by ``call_id``
  from the attributed agent activity that caused it.

Payloads are plain JSON dicts rather than per-type models on purpose. A reader
five versions from now has to tolerate fields it has never heard of and fields
that have gone away; a dict does that natively, and the builders here are the
single place each shape is written down.
"""

from __future__ import annotations

import time
from typing import Any, Literal

from pydantic import BaseModel, Field

EventType = Literal[
    "session_meta",
    "watch_started",
    "watch_stopped",
    "game_detected",
    "game_changed",
    "message",
    "journal_entry",
    "observer_run",
    "frame",
    "llm_call",
    "compaction",
    "status",
    "settings_changed",
    "agent_trace",
]

#: The vocabulary as a tuple, for validation and for tests that assert the
#: recorder cannot invent a type the reader has never heard of.
EVENT_TYPES: tuple[str, ...] = (
    "session_meta",
    "watch_started",
    "watch_stopped",
    "game_detected",
    "game_changed",
    "message",
    "journal_entry",
    "observer_run",
    "frame",
    "llm_call",
    "compaction",
    "status",
    "settings_changed",
    "agent_trace",
)

#: Why a frame was kept. New v3 records use ``question``, ``scheduled`` and
#: ``immediate``; ``observer`` and ``burst`` remain for old-session readers.
FrameReason = Literal["question", "observer", "burst", "scheduled", "immediate"]

#: How memory was consolidated. New records use ``responder`` and
#: ``observer_rotate``; ``nonlive`` and ``live_rotate`` remain readable.
CompactionMode = Literal["nonlive", "live_rotate", "responder", "observer_rotate"]


class SessionEvent(BaseModel):
    """One line of a session's event stream.

    Attributes:
        ts (float): Unix time the event happened — wall clock, not monotonic,
            because the viewer's job is to say *when*.
        type (EventType): Which kind of event this is.
        payload (dict[str, Any]): Type-specific data, built by the helpers in
            this module.
    """

    ts: float = Field(default_factory=time.time)
    type: EventType
    payload: dict[str, Any] = Field(default_factory=dict)

    @property
    def clock(self) -> str:
        """The event time as ``HH:MM:SS`` local time, for the viewer."""
        return time.strftime("%H:%M:%S", time.localtime(self.ts))


# --------------------------------------------------------------- structure


def session_meta(
    *, session_id: str, created_at: float, title: str, mode: str, game: str = ""
) -> dict[str, Any]:
    """The opening line: who this session is and what it started as.

    ``mode`` is the topology at creation. v3 writes ``dual_agent`` while old
    ``live`` and ``nonlive`` values remain valid historical data.
    """
    return {
        "id": session_id,
        "created_at": created_at,
        "title": title,
        "mode": mode,
        "game": game,
    }


def watch_span(*, watching: bool) -> dict[str, Any]:
    """A watch start or stop. The type carries the direction; this carries none."""
    return {"watching": watching}


def game(*, label: str, identity: str = "", described: str = "") -> dict[str, Any]:
    """A detected or changed game, as the active-window tracker saw it.

    Args:
        label (str): The short human name ("Elden Ring").
        identity (str): The tracker's stable key — appid or window class — which
            is what actually decides whether the game *changed*.
        described (str): The longer phrasing that goes into a system prompt.
    """
    return {"label": label, "identity": identity, "described": described or label}


def status(
    *, status: str, detail: str = "", agent_id: str | None = None
) -> dict[str, Any]:
    """A provider status transition, from the shared status vocabulary."""
    return {"status": status, "detail": detail, "agent_id": agent_id}


def settings_changed(*, fields: list[str], mode: str) -> dict[str, Any]:
    """Which settings fields changed mid-session, and the mode afterwards.

    Only the field *names* are recorded. The values would frequently include an
    API key, and a play history is not a place to keep one.
    """
    return {"fields": sorted(fields), "mode": mode}


# ----------------------------------------------------------------- content


def message(
    *,
    role: str,
    text: str,
    frames: list[str] | None = None,
    call_id: str | None = None,
    agent_id: str | None = None,
) -> dict[str, Any]:
    """One turn of the conversation.

    Images are stored as frame *references* — the ids of thumbnails written
    beside this file — never as base64. An evening of inline images would make
    the event stream unreadable and unbounded, and the thumbnail is the thing
    worth keeping anyway.
    """
    return {
        "role": role,
        "text": text,
        "frames": list(frames or []),
        "call_id": call_id,
        "agent_id": agent_id,
    }


def journal_entry(
    *, timestamp: float, note: str, category: str, source: str
) -> dict[str, Any]:
    """A journal entry, field for field as :class:`~chiron.journal.log.JournalEntry`.

    Deliberately verbatim: replay folds these straight back into a fresh
    :class:`~chiron.journal.log.JournalLog`, and any translation here would be a
    place for the two shapes to drift apart.
    """
    return {
        "timestamp": timestamp,
        "note": note,
        "category": category,
        "source": source,
    }


def observer_run(**payload: Any) -> dict[str, Any]:
    """One successful buffered Observer batch, preserving extensible v4 fields."""
    result = dict(payload)
    result.setdefault("agent_id", "observer")
    result.setdefault("seal_reason", str(result.get("reason") or "interval"))
    result.setdefault("journal_entry_count", int(result.get("entries") or 0))
    result.setdefault("source_frame_count", 0)
    result.setdefault("encoded_part_count", 0)
    result.setdefault("successful_call_ids", [])
    return result


def frame(
    *,
    frame_id: str,
    captured_at: float,
    width: int,
    height: int,
    reason: str,
    thumbnail: str,
    agent_id: str | None = None,
) -> dict[str, Any]:
    """A frame that reached a model, and why it was kept.

    What the record shows is exactly what the AI saw — the thumbnail is written
    only for frames that were actually sent, never for the ones the ring buffer
    held and discarded.
    """
    return {
        "id": frame_id,
        "captured_at": captured_at,
        "width": width,
        "height": height,
        "reason": reason,
        "thumbnail": thumbnail,
        "agent_id": agent_id,
    }


def compaction(
    *,
    mode: str,
    summary: str = "",
    tokens_before: int = 0,
    tokens_after: int = 0,
    kept_tail_count: int = 0,
    dropped_count: int = 0,
    reason: str = "",
    call_id: str | None = None,
    agent_id: str | None = None,
) -> dict[str, Any]:
    """Memory was deliberately consolidated.

    The fields are the ones :mod:`chiron.core.session`'s dormant store already
    defined for the same event, kept identical so the two can be read by one
    reader if the branching store ever wakes up.
    """
    return {
        "mode": mode,
        "summary": summary,
        "tokens_before": tokens_before,
        "tokens_after": tokens_after,
        "kept_tail_count": kept_tail_count,
        "dropped_count": dropped_count,
        "reason": reason,
        "call_id": call_id,
        "agent_id": agent_id,
    }


def agent_trace(*, agent_id: str, event: Any) -> dict[str, Any]:
    """One diagnostic ReAct event, intentionally outside the transcript."""
    return {"agent_id": agent_id, "event": event}


__all__ = [
    "EVENT_TYPES",
    "CompactionMode",
    "EventType",
    "FrameReason",
    "SessionEvent",
    "agent_trace",
    "compaction",
    "frame",
    "game",
    "journal_entry",
    "message",
    "observer_run",
    "session_meta",
    "settings_changed",
    "status",
    "watch_span",
]
