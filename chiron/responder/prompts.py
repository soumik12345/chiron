"""Prompt assembly for the answer-only Chiron-Responder."""

from __future__ import annotations

import time

from chiron.capture.frames import Frame
from chiron.config.settings import Settings
from chiron.journal.service import JournalSnapshot
from chiron.session import ObserverStatus

RESPONDER_SYSTEM_PROMPT = """\
You are Chiron-Responder, a concise gaming assistant. You answer the player's \
questions; you do not claim to be the process watching the screen. Treat the \
Observer journal as durable evidence. When visual observation is stale, say so \
when it affects the answer and never imply that you can see the current screen. \
Do not invent game state, item names, objectives, or events.\
"""

REACT_INSTRUCTION = """\
This request uses tools. read_journal is the only tool and must succeed before \
you give a final answer. Tool protocol and intermediate reasoning are diagnostic; \
the player sees only your final answer.\
"""


def build_responder_instruction(
    settings: Settings,
    *,
    detected_game: str = "",
    react: bool = False,
) -> str:
    chunks = [RESPONDER_SYSTEM_PROMPT]
    game = settings.game_name.strip()
    if game:
        chunks.append(f"\nThe player is playing: {game}.\n")
    elif detected_game.strip():
        chunks.append(
            "\nThe focused window suggests the player is playing: "
            f"{detected_game.strip()}. Trust direct evidence over this label.\n"
        )
    extra = settings.responder_system_prompt.strip()
    if extra:
        chunks.append(f"\nAdditional player instructions:\n{extra}\n")
    if react:
        chunks.append("\n" + REACT_INSTRUCTION)
    return "".join(chunks)


def observer_context(status: ObserverStatus) -> str:
    """Render connection/freshness metadata without overstating currency."""
    if status.last_observed_at is None:
        observed = "never in this gameplay session"
    else:
        age = max(0.0, time.time() - status.last_observed_at)
        observed = f"{age:.1f} seconds ago"
    if status.last_sampled_at is None:
        sampled = "never in this gameplay session"
    else:
        age = max(0.0, time.time() - status.last_sampled_at)
        sampled = f"{age:.1f} seconds ago"
    return "\n".join(
        [
            f"mode: {status.mode}",
            f"state: {status.state}",
            f"watch requested: {'yes' if status.watch_requested else 'no'}",
            f"accepting frames: {'yes' if status.accepting_frames else 'no'}",
            f"newest frame processed by Observer: {observed}",
            f"newest frame sampled for Observer: {sampled}",
            f"pending Observer frames: {status.pending_frames}",
            f"stale: {'yes' if status.stale else 'no'}",
            f"detail: {status.detail or '(none)'}",
        ]
    )


def build_fixed_turn(
    question: str,
    journal: JournalSnapshot,
    status: ObserverStatus,
    frame: Frame | None,
) -> str:
    journal_text = journal.render() or "(empty)"
    frame_line = frame_context(frame)
    return (
        "<observer-status>\n"
        f"{observer_context(status)}\n"
        "</observer-status>\n\n"
        "<journal>\n"
        f"{journal_text}\n"
        "</journal>\n\n"
        f"{frame_line}\n\nPlayer question: {question.strip()}"
    )


def frame_context(frame: Frame | None) -> str:
    """Describe current visual evidence with its timestamp and age."""
    if frame is None:
        return "No current screenshot accompanies this question."
    age = max(0.0, time.time() - frame.captured_at)
    return (
        f"A screenshot captured at {frame.clock} ({age:.1f} seconds ago) "
        "accompanies this question."
    )


__all__ = [
    "REACT_INSTRUCTION",
    "RESPONDER_SYSTEM_PROMPT",
    "build_fixed_turn",
    "build_responder_instruction",
    "frame_context",
    "observer_context",
]
