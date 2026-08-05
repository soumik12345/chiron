"""Instructions for the silent, journal-only Live Observer."""

from __future__ import annotations

from chiron.config.settings import Settings

OBSERVER_SYSTEM_PROMPT = """\
You are Chiron-Observer. You watch periodic screenshots from the player's game \
and may affect the application only by calling record_event. Never answer the \
player, narrate, coach, or produce useful speech. Any audio or content you emit \
is discarded.

Record only durable facts likely to matter later: progress, objectives, named \
locations or entities, key resources, discoveries, deaths, and player decisions. \
Ignore animation, ordinary movement, repeated HUD state, transient menus, and \
anything uncertain. Do not repeat facts already present in the journal. Call \
record_event zero or more times, then finish the checkpoint silently.\
"""

CHECKPOINT_INSTRUCTION = """\
Inspect the new screenshot. Call record_event only for durable new facts not \
already in the journal; otherwise finish silently.\
"""


def build_observer_instruction(settings: Settings, *, detected_game: str = "") -> str:
    parts = [OBSERVER_SYSTEM_PROMPT]
    game = settings.game_name.strip()
    if game:
        parts.append(f"\nThe player is playing: {game}.\n")
    elif detected_game.strip():
        parts.append(
            "\nThe focused window suggests: "
            f"{detected_game.strip()}. Trust the frames if they disagree.\n"
        )
    extra = settings.observer_system_prompt.strip()
    if extra:
        parts.append(f"\nAdditional Observer instructions:\n{extra}\n")
    return "".join(parts)


def build_journal_context(journal_text: str) -> str:
    if not journal_text.strip():
        return ""
    return (
        "Earlier gameplay journal (context only; do not reply or repeat):\n"
        + journal_text.strip()
    )


__all__ = [
    "CHECKPOINT_INSTRUCTION",
    "OBSERVER_SYSTEM_PROMPT",
    "build_journal_context",
    "build_observer_instruction",
]
