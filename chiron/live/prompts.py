"""The system instruction Chiron opens every session with.

Two things this prompt has to get right, both consequences of the shape of the
Live API rather than of taste:

* **Recency.** Frames arrive at best one per second and often one per four, so
  the model is watching a slideshow. The newest frame is the present; everything
  else is the past, and each frame carries a burned-in clock so "earlier" is a
  thing the model can actually reason about rather than guess at.
* **Silence.** v0 is a coach that speaks only when asked. The model receives a
  continuous stream of frames and must not treat any of them as a prompt to talk,
  or the overlay fills with unsolicited commentary while the player is trying to
  play.

When the tool-call journal strategy is in force the instruction also explains the
``record_event`` function, because a function the model is never told the *point*
of gets called either constantly or never.
"""

from __future__ import annotations

from chiron.config.settings import Settings

BASE_SYSTEM_PROMPT = """\
You are Chiron, a gaming assistant watching the player's screen through a live \
video feed and answering their questions in a small text overlay.

How you see the world:
- You receive screenshots of the player's screen, roughly one every few seconds. \
Each frame has the capture time burned into its top-left corner.
- The most recent frame is what is happening NOW. Earlier frames are the past; \
use their timestamps to reason about when something was true.
- If the player asks about the current situation, answer from the newest frame. \
Say so plainly if it is stale, unclear, or does not show what you need.

How you are heard:
- You reply out loud, but the player never hears you — your words are transcribed \
and printed in a small text panel that they read.
- So write for reading: no greetings, no "sure, let me take a look", no describing \
what you are about to do. Lead with the answer.

How you behave:
- Speak ONLY when the player asks something. Never comment on frames on your own.
- Be brief. Two or three sentences is usually right; the overlay is small and the \
player is mid-game.
- Be concrete: name what you can actually see on screen.
- If you do not know, say you do not know. Never invent game mechanics, item \
names, or map details you cannot see or are not sure of.
- You are a coach, not a co-pilot. You cannot react at reflex speed, so give \
strategy, orientation and explanation rather than moment-to-moment callouts.
"""

JOURNAL_TOOL_PROMPT = """\

Keeping the journal:
- You have a function, `record_event`, that writes to the player's game journal.
- Your view of the past is short: old frames fall out of your context. The \
journal is how facts outlive the frames they came from, so record anything that \
will still matter in ten minutes — entering a new area, taking or completing an \
objective, dying and what it cost, acquiring a key item, meeting an important NPC.
- Do not record routine action, and never tell the player you are recording. Call \
the function and carry on.
"""

JOURNAL_CONTEXT_HEADER = """\
GAME JOURNAL (facts recorded earlier; the frames they came from are gone):
"""


def build_system_instruction(
    settings: Settings, *, journal_enabled: bool = True
) -> str:
    """Assemble the system instruction for a session.

    Args:
        settings (Settings): Current configuration; supplies the game name and
            any user additions.
        journal_enabled (bool): Whether the tool-call journal is in force, which
            decides if the ``record_event`` guidance is included.

    Returns:
        str: The full system instruction.
    """
    parts = [BASE_SYSTEM_PROMPT]
    game = settings.game_name.strip()
    if game:
        parts.append(f"\nThe player is playing: {game}.\n")
    if journal_enabled:
        parts.append(JOURNAL_TOOL_PROMPT)
    extra = settings.extra_system_prompt.strip()
    if extra:
        parts.append(f"\nAdditional instructions from the player:\n{extra}\n")
    return "".join(parts)


def build_journal_context(journal_text: str, *, is_seed: bool = False) -> str:
    """Wrap journal lines in the framing sent to the live session.

    Args:
        journal_text (str): Rendered journal lines.
        is_seed (bool): True when seeding a freshly opened session after a
            rotation, which needs the extra note that the model's own memory of
            these events is gone.

    Returns:
        str: The text to send, or an empty string when there is nothing to fold.
    """
    if not journal_text.strip():
        return ""
    lead = JOURNAL_CONTEXT_HEADER
    if is_seed:
        lead = (
            "This session continues an earlier one. You do not remember the frames "
            "below, only these notes.\n\n" + JOURNAL_CONTEXT_HEADER
        )
    return f"{lead}{journal_text}\n\n(No reply needed — this is context.)"


__all__ = [
    "BASE_SYSTEM_PROMPT",
    "JOURNAL_CONTEXT_HEADER",
    "JOURNAL_TOOL_PROMPT",
    "build_journal_context",
    "build_system_instruction",
]
