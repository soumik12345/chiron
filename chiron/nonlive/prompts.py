"""What a request/response model is told, and why it differs from the live one.

The live prompt (:mod:`chiron.live.prompts`) is written for a model swimming in a
stream of frames: its job is to say *when the present is*, and to keep the model
quiet between questions. Neither problem exists here. A non-live model is handed
a small, explicit set of frames per call and cannot speak unless spoken to, so
the instruction spends its words on the two things that *are* different:

* **The journal is the memory.** In live mode the journal insures against frame
  eviction; here it is the only thing that survives a call at all. Everything the
  model knows about the last hour that is not in the current frames came from the
  journal, and it has to be told that so it neither invents continuity nor
  disclaims knowledge it has been given.
* **The observer is silent.** It writes journal entries and nothing else. Given a
  chat endpoint's overwhelming inclination to answer the person, the observer
  prompt has to say plainly that there is nobody listening — its entire output is
  parsed as JSON, and prose around it is at best discarded.

The observer's reply shape is deliberately the sidecar's, so
:func:`~chiron.journal.writers.parse_sidecar_entries` reads both. JSON rather
than a tool call because it is transport-agnostic: OpenRouter's several hundred
models disagree about function calling in ways they do not disagree about
producing a JSON object.
"""

from __future__ import annotations

from chiron.config.settings import Settings
from chiron.journal.log import CATEGORIES

QA_SYSTEM_PROMPT = """\
You are Chiron, a gaming assistant. The player asks you questions in a small text \
overlay while they play, and you answer from screenshots of their screen.

How you see the world:
- Attached to the player's message are the most recent screenshots of their \
screen, with the capture time burned into the top-left corner of each.
- The last attached frame is what is happening NOW. Any earlier ones are the \
immediate past.
- You do not see the screen between questions. Anything you know about earlier in \
this session comes from the game journal below, which is written for you as you \
play. Treat it as reliable, and say plainly when something is not in it.

How you answer:
- Lead with the answer. No greetings, no "let me take a look", no describing what \
you are about to do — the player is mid-game and reading a small panel.
- Be brief: two or three sentences is usually right.
- Be concrete: name what you can actually see on screen.
- If you do not know, say so. Never invent game mechanics, item names or map \
details you cannot see and are not sure of.
- You are a coach, not a co-pilot. You cannot react at reflex speed, so give \
strategy, orientation and explanation rather than moment-to-moment callouts.
"""

OBSERVER_SYSTEM_PROMPT = """\
You maintain a terse game journal for a player. You are not talking to anyone: \
your entire reply is parsed by a program, and any prose outside the JSON object \
is discarded.

You are shown the journal so far, the recent conversation between the player and \
their assistant, and the newest frames of the player's screen. Record only what \
will still matter in ten minutes: location changes, objectives taken or \
completed, deaths and what they cost, key items acquired, important NPCs, \
decisive progress.

Rules:
- Write each entry as one concrete, self-contained, past-tense sentence.
- Do not record routine moment-to-moment action, speculation, or anything you \
are unsure of.
- Do not repeat anything already in the journal shown to you.
- Most of the time nothing has happened. An empty list is the correct answer and \
costs nothing; a padded one poisons the memory this player is relying on.

Respond with JSON only, in exactly this shape:
{"entries": [{"category": "location", "note": "Entered Firelink Shrine."}]}

Valid categories: %s
""" % ", ".join(CATEGORIES)

#: What the observer is asked on each tick, in the unified conversation.
OBSERVER_TURN_INSTRUCTION = (
    "Journal update. Look at the frames above and reply with the JSON object "
    "described in your instructions — entries for anything newly worth "
    "remembering, or an empty list. Do not address the player."
)

JOURNAL_HEADER = "GAME JOURNAL (what has happened so far this session):"


def build_qa_instruction(settings: Settings, *, detected_game: str = "") -> str:
    """The system prompt for the question-answering path.

    Args:
        settings (Settings): Supplies the game name and any user additions.
        detected_game (str): The focused window, phrased by
            :meth:`~chiron.capture.active_window.WindowInfo.describe`. Used only
            when the player has not named the game, and hedged because a focused
            window is evidence, not certainty.

    Returns:
        str: The full system instruction.
    """
    return "".join(
        [QA_SYSTEM_PROMPT, _game_clause(settings, detected_game), _extra(settings)]
    )


def build_observer_instruction(settings: Settings, *, detected_game: str = "") -> str:
    """The system prompt for the observer.

    The player's extra instructions are deliberately **not** included. They are
    written to shape how Chiron talks to them ("never spoil anything I haven't
    found yet"), and an observer that applied them would quietly stop recording
    the very facts the player will later ask about.
    """
    return "".join([OBSERVER_SYSTEM_PROMPT, _game_clause(settings, detected_game)])


def _game_clause(settings: Settings, detected_game: str) -> str:
    """The "what is being played" sentence, or nothing when it is unknown."""
    game = settings.game_name.strip()
    if game:
        return f"\nThe player is playing: {game}.\n"
    detected = detected_game.strip()
    if detected:
        return (
            f"\nThe player appears to be playing: {detected} — detected from the "
            "window in focus. Trust the frames if they show something else.\n"
        )
    return ""


def _extra(settings: Settings) -> str:
    """The player's own additions to the instruction, if any."""
    extra = settings.extra_system_prompt.strip()
    return f"\nAdditional instructions from the player:\n{extra}\n" if extra else ""


def build_journal_block(journal_text: str) -> str:
    """Wrap rendered journal lines for inclusion in a request.

    Args:
        journal_text (str): Rendered journal lines, oldest first.

    Returns:
        str: The block to prepend to the user turn, or empty when there is
            nothing recorded yet.
    """
    if not journal_text.strip():
        return ""
    return f"{JOURNAL_HEADER}\n{journal_text}"


def build_observer_prompt(journal_text: str, conversation: str) -> str:
    """The text half of an observer call in split mode.

    Args:
        journal_text (str): Rendered journal lines the observer must not repeat.
        conversation (str): Recent player/Chiron lines, or empty.

    Returns:
        str: The user-turn text accompanying the frames.
    """
    lines = [
        "Existing journal (do not repeat these):",
        journal_text.strip() or "(empty)",
        "",
        "Recent conversation:",
        conversation.strip() or "(no conversation since the last journal update)",
        "",
        "The frames below are the newest views of the player's screen.",
    ]
    return "\n".join(lines)


__all__ = [
    "JOURNAL_HEADER",
    "OBSERVER_SYSTEM_PROMPT",
    "OBSERVER_TURN_INSTRUCTION",
    "QA_SYSTEM_PROMPT",
    "build_journal_block",
    "build_observer_instruction",
    "build_observer_prompt",
    "build_qa_instruction",
]
