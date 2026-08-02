"""Ways to fill the journal, behind one interface.

Every writer appends to the same :class:`~chiron.journal.log.JournalLog`, so the
choice is invisible to the rest of the app — it is a setting, not an
architecture:

* :class:`ToolCallJournal` hands the live model a ``record_event`` function and
  asks it to call it when something notable happens. Zero extra API calls and one
  model holding the whole picture, at the risk of journaling competing with the
  conversation for the model's attention.
* :class:`SidecarJournal` leaves the live model alone and, every few minutes,
  asks a cheap regular Gemini model to distil the recent transcript and a handful
  of kept frames into entries. Cleaner separation and easier to tune, at the cost
  of extra calls.
* :class:`ObserverJournal` does nothing on its own. It is what non-live mode
  uses, where the observer already *is* the journal writer — it runs on the
  screen's own events rather than a timer, and it has the frames and the
  conversation in hand — so the writer's remaining job is the one thing the
  observer cannot do for itself: hand each entry to the UI callback.

The strategy setting therefore applies to live mode only. In non-live mode there
is no session to evict frames from and no second summariser worth paying for;
the journal is primary memory rather than insurance, and the observer fills it.

The sidecar reuses the repo's existing :class:`~chiron.models.litellm_model.LiteLLMModel`
wrapper rather than opening a second Google client, which means its spend lands
in the same usage accounting as everything else that goes through litellm.
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import re
import time
from abc import ABC
from collections import deque
from typing import Any, Callable

from chiron.capture.frames import Frame
from chiron.config.settings import JournalSettings
from chiron.journal.log import CATEGORIES, JournalEntry, JournalLog
from chiron.models.litellm_model import LiteLLMModel

logger = logging.getLogger(__name__)

EntryCallback = Callable[[JournalEntry], None]

#: The function declaration handed to the live model by :class:`ToolCallJournal`.
RECORD_EVENT_DECLARATION: dict[str, Any] = {
    "name": "record_event",
    "description": (
        "Record a notable event in the player's game journal. Call this whenever "
        "something happens that would still matter in ten minutes: entering a new "
        "area, picking up a key item, dying, accepting or completing an objective, "
        "meeting an NPC, or a decisive change in the player's situation. Do not "
        "record routine moment-to-moment action, and do not announce that you are "
        "recording — just call the function and carry on."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "note": {
                "type": "string",
                "description": (
                    "One sentence, past tense, concrete and self-contained. "
                    "Example: 'Entered Firelink Shrine and lit the bonfire.'"
                ),
            },
            "category": {
                "type": "string",
                "enum": list(CATEGORIES),
                "description": "Which kind of event this is.",
            },
        },
        "required": ["note"],
    },
}

SIDECAR_SYSTEM_PROMPT = """\
You maintain a terse game journal for a player being coached by an AI assistant.

You are given the recent conversation between the player and the assistant, and \
the most recent frames of the player's screen. Extract only events that will \
still matter in ten minutes: location changes, objectives taken or completed, \
deaths and their cost, key items acquired, important NPCs, decisive progress.

Rules:
- Write each entry as one concrete, self-contained, past-tense sentence.
- Do not record routine action, speculation, or anything you are unsure of.
- Do not repeat events already present in the existing journal shown to you.
- If nothing worth recording happened, return an empty list.

Respond with JSON only, in exactly this shape:
{"entries": [{"category": "location", "note": "Entered Firelink Shrine."}]}

Valid categories: %s
""" % ", ".join(CATEGORIES)


class JournalWriter(ABC):
    """Common interface for the journal strategies.

    Attributes:
        name (str): Strategy id, matching
            :attr:`~chiron.config.settings.JournalSettings.strategy`.
        log (JournalLog): The log entries are appended to.
        on_entry (EntryCallback | None): Called with each new entry, so the UI
            can react without knowing which writer produced it.
    """

    name: str = "base"

    def __init__(self, log: JournalLog, on_entry: EntryCallback | None = None) -> None:
        self.log = log
        self.on_entry = on_entry

    def function_declarations(self) -> list[dict[str, Any]]:
        """Function declarations to install in the live session. Empty by default."""
        return []

    async def handle_tool_call(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        """Handle a live-model function call addressed to this writer.

        Args:
            name (str): The called function's name.
            args (dict[str, Any]): Parsed arguments.

        Returns:
            dict[str, Any]: The payload to return to the model.
        """
        return {"error": f"Unknown function: {name}"}

    def observe_user_message(self, text: str) -> None:
        """Note something the player said."""

    def observe_model_message(self, text: str) -> None:
        """Note something Chiron answered."""

    def observe_frame(self, frame: Frame) -> None:
        """Note a captured frame."""

    async def start(self) -> None:
        """Begin any background work. No-op by default."""

    async def stop(self) -> None:
        """Stop background work and release resources. No-op by default."""

    def record(
        self, note: str, category: str = "note", *, timestamp: float | None = None
    ) -> JournalEntry | None:
        """Append to the log and notify the UI callback.

        Public because the non-live observer produces entries from outside any
        writer — it is the one strategy whose work happens in the session
        manager — and still needs them to reach the overlay the same way.

        Args:
            note (str): What happened, in one sentence. Blank notes are ignored.
            category (str): Loose bucket from
                :data:`~chiron.journal.log.CATEGORIES`.
            timestamp (float | None): Event time; defaults to now.

        Returns:
            JournalEntry | None: The stored entry, or None for a blank note.
        """
        entry = self.log.append(
            note, category=category, source=self.name, timestamp=timestamp
        )
        if entry is not None and self.on_entry is not None:
            self.on_entry(entry)
        return entry


class ToolCallJournal(JournalWriter):
    """Journalling by function call, done by the live model itself."""

    name = "tool_call"

    def function_declarations(self) -> list[dict[str, Any]]:
        """The single ``record_event`` declaration."""
        return [RECORD_EVENT_DECLARATION]

    async def handle_tool_call(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        """Record the event and acknowledge it.

        The acknowledgement is deliberately terse: the model needs to know the
        call succeeded, and nothing more, or it will start narrating its own
        bookkeeping to the player.
        """
        if name != "record_event":
            return await super().handle_tool_call(name, args)
        note = str(args.get("note") or "")
        category = str(args.get("category") or "note")
        entry = self.record(note, category)
        if entry is None:
            return {"status": "ignored", "reason": "empty note"}
        return {"status": "recorded", "at": entry.clock}


class ObserverJournal(JournalWriter):
    """The non-live journal: entries arrive from the observer, not from here.

    A writer with nothing to write may look like a hole in the design, but the
    alternative is worse. Non-live mode's observer already holds everything a
    journal strategy would need — the frame ring buffer, the conversation, the
    trigger that decides when looking is worth paying for — so a second thing
    summarising the same material on a timer would be duplicated cost for
    duplicated entries. What survives is the part the observer genuinely cannot
    supply itself: a shared route to the log and the overlay callback, so an
    entry looks the same in the transcript whichever mode produced it.

    Installing nothing into the session is also load-bearing: a non-live
    provider has no function declarations to install and no background loop to
    stop, and both of those are inherited no-ops.
    """

    name = "observer"


class SidecarJournal(JournalWriter):
    """Journalling by a separate summariser model on a timer.

    Attributes:
        settings (JournalSettings): Cadence and model configuration.
        model (LiteLLMModel): The summariser. Injectable so tests can drive the
            distillation logic without a network call.
    """

    name = "sidecar"

    def __init__(
        self,
        log: JournalLog,
        settings: JournalSettings,
        *,
        api_key: str = "",
        on_entry: EntryCallback | None = None,
        model: LiteLLMModel | None = None,
        usage_sink: Callable[[Any], None] | None = None,
    ) -> None:
        """Create a stopped sidecar.

        Args:
            log (JournalLog): Where entries land.
            settings (JournalSettings): Model id and interval.
            api_key (str): Google credential, passed through to litellm. Empty
                leaves litellm to find one in the environment.
            on_entry (EntryCallback | None): Per-entry UI callback.
            model (LiteLLMModel | None): Override the summariser, for tests.
            usage_sink (Callable | None): Where this writer's priced calls go.
                The sidecar is the one part of live mode that *does* reach
                litellm, so its spend is measured rather than estimated — and
                without a sink it would be the one real number nobody records.
        """
        super().__init__(log, on_entry)
        self.settings = settings
        self.model = model or LiteLLMModel(
            model_id=settings.sidecar_model,
            api_key=api_key or None,
            temperature=0.2,
            max_tokens=800,
            usage_sink=usage_sink,
        )
        self._transcript: deque[tuple[float, str, str]] = deque(maxlen=40)
        self._frames: deque[Frame] = deque(maxlen=max(1, settings.sidecar_frame_count))
        self._task: asyncio.Task[None] | None = None
        self._last_run = 0.0

    def observe_user_message(self, text: str) -> None:
        """Buffer a player message for the next summarisation."""
        if text.strip():
            self._transcript.append((time.time(), "player", text.strip()))

    def observe_model_message(self, text: str) -> None:
        """Buffer a Chiron answer for the next summarisation."""
        if text.strip():
            self._transcript.append((time.time(), "chiron", text.strip()))

    def observe_frame(self, frame: Frame) -> None:
        """Keep the most recent frames so the summariser can see the screen."""
        if self.settings.sidecar_frame_count > 0:
            self._frames.append(frame)

    async def start(self) -> None:
        """Start the periodic summarisation loop."""
        if self._task is None or self._task.done():
            self._last_run = time.time()
            self._task = asyncio.create_task(self._loop(), name="chiron-sidecar")

    async def stop(self) -> None:
        """Cancel the loop and wait for it to unwind."""
        task, self._task = self._task, None
        if task is not None and not task.done():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

    async def _loop(self) -> None:
        """Summarise on the configured interval until cancelled."""
        while True:
            await asyncio.sleep(self.settings.sidecar_interval_seconds)
            try:
                await self.summarise_once()
            except asyncio.CancelledError:
                raise
            except Exception as error:
                logger.warning("Sidecar summarisation failed: %s", error)

    async def summarise_once(self) -> list[JournalEntry]:
        """Distil the buffered transcript and frames into journal entries.

        Nothing is sent when there is nothing new to look at, so an idle player
        costs nothing. The buffered transcript is cleared on success only — a
        failed call leaves the material in place for the next attempt.

        Returns:
            list[JournalEntry]: The entries recorded by this pass.
        """
        if not self._transcript and not self._frames:
            return []

        messages = self._build_messages()
        response = await self.model.acompletion(messages=messages)
        content = _response_text(response)
        try:
            self.model.record_usage(
                getattr(response, "usage", None),
                response=response,
                context={"kind": "journal_sidecar"},
            )
        except Exception:  # pragma: no cover - accounting must never break the app
            logger.debug("Usage accounting failed for sidecar call", exc_info=True)

        entries: list[JournalEntry] = []
        for note, category in parse_sidecar_entries(content):
            entry = self.record(note, category)
            if entry is not None:
                entries.append(entry)

        self._transcript.clear()
        self._last_run = time.time()
        logger.info("Sidecar recorded %d journal entries", len(entries))
        return entries

    def _build_messages(self) -> list[dict[str, Any]]:
        """Assemble the summariser request: existing journal, transcript, frames."""
        existing = self.log.render(self.log.recent(20))
        lines = [
            "Existing journal (do not repeat these):",
            existing or "(empty)",
            "",
            "Recent conversation:",
        ]
        if self._transcript:
            for when, role, text in self._transcript:
                clock = time.strftime("%H:%M", time.localtime(when))
                lines.append(f"{clock} {role}: {text}")
        else:
            lines.append("(no conversation since the last journal update)")

        content: list[dict[str, Any]] = [{"type": "text", "text": "\n".join(lines)}]
        for frame in list(self._frames):
            encoded = base64.b64encode(frame.jpeg).decode("ascii")
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{encoded}"},
                }
            )
        return [
            {"role": "system", "content": SIDECAR_SYSTEM_PROMPT},
            {"role": "user", "content": content},
        ]


def _response_text(response: Any) -> str:
    """Extract the assistant text from a litellm response, tolerating shapes."""
    try:
        choice = response.choices[0]
    except (AttributeError, IndexError, TypeError):
        return ""
    message = getattr(choice, "message", None) or {}
    content = (
        message.get("content")
        if isinstance(message, dict)
        else getattr(message, "content", None)
    )
    return content or ""


def parse_sidecar_entries(content: str) -> list[tuple[str, str]]:
    """Parse the summariser's reply into ``(note, category)`` pairs.

    Models wrap JSON in prose and code fences more often than anyone would like,
    so the first balanced-looking JSON object in the reply is used. A reply that
    yields nothing parseable is treated as "no events", not as an error — the
    next pass will try again with the same material.

    Args:
        content (str): The raw model reply.

    Returns:
        list[tuple[str, str]]: Extracted notes and their categories.
    """
    if not content or not content.strip():
        return []
    text = content.strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if fenced:
        text = fenced.group(1).strip()
    if not text.startswith("{"):
        start, end = text.find("{"), text.rfind("}")
        if start == -1 or end <= start:
            return []
        text = text[start : end + 1]

    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        logger.debug("Unparseable sidecar reply: %s", content[:200])
        return []

    raw_entries = payload.get("entries") if isinstance(payload, dict) else payload
    if not isinstance(raw_entries, list):
        return []

    results: list[tuple[str, str]] = []
    for item in raw_entries:
        if isinstance(item, str):
            results.append((item, "note"))
        elif isinstance(item, dict):
            note = str(item.get("note") or item.get("text") or "").strip()
            if note:
                results.append((note, str(item.get("category") or "note")))
    return results


def build_journal_writer(
    settings: JournalSettings,
    log: JournalLog,
    *,
    api_key: str = "",
    on_entry: EntryCallback | None = None,
    live: bool = True,
    usage_sink: Callable[[Any], None] | None = None,
) -> JournalWriter:
    """Create the writer the current mode and strategy call for.

    Args:
        settings (JournalSettings): Journal configuration.
        log (JournalLog): The shared log.
        api_key (str): Google credential for the sidecar model.
        on_entry (EntryCallback | None): Per-entry UI callback.
        live (bool): Whether a Live API session is in force. The strategy
            setting only means something there; a non-live provider always
            journals through its observer, so the setting is ignored rather
            than being allowed to install a sidecar that would duplicate it.
        usage_sink (Callable | None): Where the sidecar's priced calls go. The
            other two strategies make no calls of their own and ignore it.

    Returns:
        JournalWriter: An :class:`ObserverJournal`, :class:`SidecarJournal` or
            :class:`ToolCallJournal`.
    """
    if not live:
        return ObserverJournal(log, on_entry)
    if settings.strategy == "sidecar":
        return SidecarJournal(
            log, settings, api_key=api_key, on_entry=on_entry, usage_sink=usage_sink
        )
    return ToolCallJournal(log, on_entry)


__all__ = [
    "RECORD_EVENT_DECLARATION",
    "SIDECAR_SYSTEM_PROMPT",
    "JournalWriter",
    "ObserverJournal",
    "SidecarJournal",
    "ToolCallJournal",
    "build_journal_writer",
    "parse_sidecar_entries",
]
