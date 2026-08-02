"""The non-live provider: no socket, no ambient context, one call at a time.

A Live session hands you awareness for free. Frames stream into a context window
and simply *are* there; the journal exists as insurance against their eviction. A
request/response endpoint gives nothing away. Every call starts blank, and the
context has to be assembled from scratch: system prompt, journal, conversation
history, and whichever frames are worth attaching right now.

That inverts the memory architecture. **The journal stops being insurance and
becomes primary memory** — in live mode the model watches and the journal
remembers; here the journal watches, and the model only ever sees what the
observer chose to distil into it plus the frames of the current moment. So
:meth:`NonLiveSessionManager.fold_journal` has no work to do (there is no
stateful session to fold into, and the journal is in every request already) and
the observer's quality is genuinely load-bearing, which is why its trigger gets
a module of its own.

Two paths run over the same manager:

* **The observer** — silent, journal-only. Fires on the trigger policy in
  :mod:`chiron.nonlive.observer`, never emits overlay text. In ``unified`` mode
  its ticks go into the same conversation the player's questions do, so answers
  come from a model that has actually been watching; in ``split`` mode it is a
  near-stateless call against its own, typically cheaper, model.
* **The Q&A path** — a question bursts the shutter, waits briefly for a frame
  newer than the question, and streams the answer back through the same three
  signals the live manager emits. The overlay cannot tell the two providers
  apart, which is the point of the seam.

**Images ride in the current request only.** Once a turn ages, its frames are
replaced in history by a ``[frame 14:32:07]`` placeholder, so conversation cost
stays linear in text rather than compounding in pixels. The timestamps burned
into the frames remain the temporal ground truth either way. The journal block
gets the same treatment for the same reason: it is prepended to every question
and would otherwise survive into history twenty times over, which is a five-digit
token bill for saying one thing repeatedly. Only the current request carries it.

What is left over after that deduplication is handled by
:mod:`chiron.nonlive.compaction`: history is measured against the model's window
before every call and summarised above 85%, with the typed context-overflow error
as a backstop that compacts and retries once.
"""

from __future__ import annotations

import asyncio
import base64
import logging
import time
from collections import deque
from typing import Any, Awaitable, Callable

from PySide6.QtCore import QObject, Signal

from chiron.capture.frames import Frame
from chiron.config.settings import Settings, litellm_model_id
from chiron.journal.log import JournalLog
from chiron.journal.writers import JournalWriter, parse_sidecar_entries
from chiron.live.session import is_permanent_error
from chiron.models.litellm_model import LiteLLMModel
from chiron.models.usage import LLMCallRecord
from chiron.nonlive import compaction as ctx
from chiron.nonlive.observer import ObserverTrigger
from chiron.nonlive.prompts import (
    OBSERVER_TURN_INSTRUCTION,
    build_journal_block,
    build_observer_instruction,
    build_observer_prompt,
    build_qa_instruction,
)

logger = logging.getLogger(__name__)

#: How long a question waits for a frame captured after it was asked. Long
#: enough for the burst the question itself triggered to produce one, short
#: enough not to read as the assistant being slow.
FRESH_FRAME_WAIT_SECONDS = 1.5

#: A frame older than this means nobody is watching, so a question should not
#: wait for a fresh one that is never coming.
CAPTURE_IDLE_SECONDS = 10.0

#: Frames attached to a question: the fresh one, plus a little of the run-up.
QA_FRAME_COUNT = 2

#: Frames retained in the ring buffer. The observer only ever shows a handful,
#: but keeping a short history costs nothing and makes the choice of which ones.
FRAME_BUFFER_SIZE = 30

#: Conversation turns kept in history. Beyond this the oldest are dropped —
#: they are text-only by then, but a conversation that only grows eventually
#: costs more per question than the frames do.
MAX_HISTORY_MESSAGES = 40

#: Journal entries included in a request.
JOURNAL_CONTEXT_ENTRIES = 40

#: How often the observer loop wakes to ask whether it is due.
_OBSERVER_TICK_SECONDS = 1.0


class NonLiveSessionManager(QObject):
    """Runs Chiron against a request/response multimodal endpoint.

    Emits exactly the signals :class:`~chiron.live.session.LiveSessionManager`
    does, and keeps its status vocabulary — including ``live``, which here means
    *armed* rather than *connected*, since there is no connection to hold open.
    Everything downstream of the seam is written against that vocabulary and has
    no business knowing which provider is behind it.

    Signals:
        statusChanged (str, str): Status (``idle``, ``live``, ``error``,
            ``stopped``) and a detail line.
        responseStarted (): The model began answering.
        responseDelta (str): A chunk of the answer.
        responseCompleted (str): The answer is complete; carries the full text.
        journalFolded (int): Emitted for interface parity; never fires here.
        errorOccurred (str): Something went wrong, in words fit for the overlay.
        frameSent (object, str): A frame was attached to a request, with why —
            ``question`` or ``observer``. This is the moment a frame becomes
            something the model has seen, which is the only kind worth recording.
        observerRan (object): One observer tick, as a dict the recorder turns
            into an ``observer_run`` event.
        llmCall (object): One priced :class:`~chiron.models.usage.LLMCallRecord`.
        compacted (object): History was summarised, as a dict.

    Attributes:
        settings (Settings): Configuration used to build the next request.
        journal (JournalLog): The shared journal — primary memory in this mode.
        writer (JournalWriter): The route observer entries take to the log and
            the overlay.
        trigger (ObserverTrigger): When looking is worth paying for.
        status (str): Current status.
        detected_game (str): The focused window when watching started.
        session_id (str): The gameplay session every call is billed to. Runtime
            state set by the app, like ``detected_game``.
    """

    statusChanged = Signal(str, str)
    responseStarted = Signal()
    responseDelta = Signal(str)
    responseCompleted = Signal(str)
    journalFolded = Signal(int)
    errorOccurred = Signal(str)
    frameSent = Signal(object, str)
    observerRan = Signal(object)
    llmCall = Signal(object)
    compacted = Signal(object)

    def __init__(
        self,
        settings: Settings,
        journal: JournalLog,
        writer: JournalWriter,
        parent: QObject | None = None,
    ) -> None:
        """Create a stopped manager. Nothing is called until :meth:`start`."""
        super().__init__(parent)
        self.settings = settings
        self.journal = journal
        self.writer = writer
        self.status = "idle"
        self.detected_game = ""
        self.session_id = ""
        self.trigger = ObserverTrigger(settings=settings.observer)

        self._frames: deque[Frame] = deque(maxlen=FRAME_BUFFER_SIZE)
        self._latest_frame: Frame | None = None
        self._frame_event = asyncio.Event()
        self._history: list[dict[str, Any]] = []
        self._conversation: deque[tuple[float, str, str]] = deque(maxlen=20)
        self._observer_task: asyncio.Task[None] | None = None
        self._answer_task: asyncio.Task[None] | None = None
        self._lock = asyncio.Lock()
        self._frames_sent = 0
        self._observer_runs = 0
        self._compactions = 0
        self._stopping = False
        #: The most recent ledger row's id, so an observer run or a compaction
        #: can name the call it paid for without threading one through.
        self._last_call_id: str | None = None

    # ------------------------------------------------------------------ API

    @property
    def is_connected(self) -> bool:
        """Whether the manager is armed. There is no socket to be connected to."""
        return self.status == "live"

    @property
    def frames_sent(self) -> int:
        """How many frames this manager has attached to a request."""
        return self._frames_sent

    @property
    def observer_runs(self) -> int:
        """How many observer calls have been made this session."""
        return self._observer_runs

    @property
    def compactions(self) -> int:
        """How many times history has been summarised this session."""
        return self._compactions

    def start(self) -> None:
        """Arm the provider and start the observer loop.

        Cheap, unlike opening a websocket: nothing is sent until a question
        arrives or the observer decides the screen is worth looking at.
        """
        if not self.settings.key_for_model(self.settings.selected_model):
            self._set_status("error", "No API key. Open Settings")
            self.errorOccurred.emit(
                "No API key for the selected model. Open Settings and paste one."
            )
            return
        self._stopping = False
        self.trigger.settings = self.settings.observer
        self.trigger.reset(time.time())
        self._set_status("live", litellm_model_id(self.settings.selected_model))
        if self._observer_task is None or self._observer_task.done():
            self._observer_task = asyncio.ensure_future(self._observe_loop())

    async def stop(self, timeout: float = 5.0) -> None:
        """Stop observing and abandon any answer still streaming.

        Args:
            timeout (float): Seconds to wait for each task to unwind.
        """
        self._stopping = True
        for name in ("_observer_task", "_answer_task"):
            task = getattr(self, name)
            setattr(self, name, None)
            if task is not None and not task.done():
                task.cancel()
                try:
                    await asyncio.wait_for(task, timeout)
                except (asyncio.CancelledError, asyncio.TimeoutError):
                    pass
                except Exception as error:  # pragma: no cover - defensive
                    logger.warning("Error stopping %s: %s", name, error)
        await self.writer.stop()
        self._set_status("stopped", "")

    def send_text(self, text: str) -> None:
        """Answer a question about what is on screen right now.

        Args:
            text (str): What the player typed. Blank input is ignored.
        """
        if not text.strip():
            return
        self.writer.observe_user_message(text)
        self._conversation.append((time.time(), "player", text.strip()))
        self._answer_task = asyncio.ensure_future(
            self._answer(text.strip(), time.time())
        )

    def send_frame(self, frame: Frame) -> None:
        """Take a captured frame into the ring buffer and the trigger.

        Nothing is sent here. Unlike the live provider, where a frame not sent is
        a frame the model never sees, frames are held until something — a
        question, or the observer's own judgement — decides they are worth
        paying to look at.

        Args:
            frame (Frame): The newest frame.
        """
        self._frames.append(frame)
        self._latest_frame = frame
        self.writer.observe_frame(frame)
        self.trigger.observe(frame, frame.captured_at)
        self._frame_event.set()

    def fold_journal(self, *, force: bool = False) -> int:
        """Nothing to fold: the journal is in every request already.

        Folding exists because a live session's context is stateful and the
        journal has to be pushed *into* it. Here each call is assembled from the
        journal from scratch, so there is no session to fall behind. Kept as a
        no-op rather than removed, because the fold timer is wired once in
        :class:`~chiron.app.ChironApp` for whichever provider is running.

        Returns:
            int: Always 0.
        """
        return 0

    def reset_observation(self) -> None:
        """Forget what the screen looked like before now.

        Called when watching starts and when capture settings change — the same
        two moments the live provider's scene-change baseline is reset for, and
        for the same reason: comparing across a gap nobody watched reads as an
        event every time.
        """
        self.trigger.settings = self.settings.observer
        self.trigger.reset(time.time())

    def reset_memory(self) -> None:
        """Forget the conversation entirely — a new gameplay session started.

        The session boundary *is* the memory boundary, so this drops history,
        the recent-conversation buffer the split observer reads, and the held
        frames. The journal is cleared by the app, which owns it; between them
        the next question is answered by a model that knows nothing but what is
        on screen.
        """
        self._history.clear()
        self._conversation.clear()
        self._frames.clear()
        self._latest_frame = None
        self._compactions = 0
        self.reset_observation()

    def apply_settings(self, settings: Settings) -> None:
        """Adopt new settings. Cadence numbers apply from the next tick."""
        self.settings = settings
        self.trigger.settings = settings.observer

    # ------------------------------------------------------------ the observer

    async def _observe_loop(self) -> None:
        """Ask the trigger, on a slow tick, whether a look is worth paying for."""
        while not self._stopping:
            await asyncio.sleep(_OBSERVER_TICK_SECONDS)
            reason = self.trigger.due(time.time())
            if reason is None:
                continue
            try:
                await self._run_observer(reason)
            except asyncio.CancelledError:
                raise
            except Exception as error:
                logger.warning("Observer call failed: %s", error)
                self.errorOccurred.emit(f"Observer: {error}")
                # Still count it as a run. A failing model must not become a
                # tight retry loop against the same doomed configuration.
                self.trigger.note_run(time.time())

    async def _run_observer(self, reason: str) -> None:
        """One observer call: frames in, journal entries out, nothing said."""
        frames = self.trigger.frames_for_call()
        if not frames:
            self.trigger.note_run(time.time())
            self.observerRan.emit(
                {"reason": reason, "frames": [], "entries": 0, "skipped": True}
            )
            return
        logger.info("Observer running (%s) with %d frames", reason, len(frames))

        self._last_call_id = None
        if self.settings.agent_mode == "split":
            content = await self._observe_split(frames)
        else:
            content = await self._observe_unified(frames)

        entries = parse_sidecar_entries(content)
        for note, category in entries:
            self.writer.record(note, category)
        self._observer_runs += 1
        self.trigger.note_run(time.time())
        self.observerRan.emit(
            {
                "reason": reason,
                "frames": list(frames),
                "entries": len(entries),
                "call_id": self._last_call_id,
            }
        )
        logger.info("Observer recorded %d entries", len(entries))

    async def _observe_split(self, frames: list[Frame]) -> str:
        """A stateless observer call against its own model."""
        model = self._model(self.settings.observer_model_id(), max_tokens=800)
        messages = [
            {
                "role": "system",
                "content": build_observer_instruction(
                    self.settings, detected_game=self.detected_game
                ),
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": build_observer_prompt(
                            self._journal_text(), self._conversation_text()
                        ),
                    },
                    *self._image_parts(frames, "observer"),
                ],
            },
        ]
        return await self._complete(model, messages, kind="nonlive_observer")

    async def _observe_unified(self, frames: list[Frame]) -> str:
        """An observer tick inside the conversation the player's questions use.

        The cost is real — every tick pays for the whole conversation context —
        and so is the benefit: when the player finally asks something, they are
        asking a model that has been watching, not one reading notes about it.
        """
        model = self._model(litellm_model_id(self.settings.selected_model))
        instruction = build_observer_instruction(
            self.settings, detected_game=self.detected_game
        )
        turn = {
            "role": "user",
            "content": [
                {"type": "text", "text": OBSERVER_TURN_INSTRUCTION},
                *self._image_parts(frames, "observer"),
            ],
        }
        async with self._lock:
            content = await self._call_with_history(
                instruction,
                turn,
                lambda messages: self._complete(
                    model, messages, kind="nonlive_observer"
                ),
            )
            self._remember(turn, content or "(nothing new)", frames)
        return content

    # --------------------------------------------------------------- the Q&A

    async def _answer(self, text: str, asked_at: float) -> None:
        """Answer one question, streaming the reply into the overlay."""
        try:
            await self._wait_for_fresh_frame(asked_at)
            frames = self._recent_frames(QA_FRAME_COUNT)
            model = self._model(litellm_model_id(self.settings.selected_model))
            instruction = build_qa_instruction(
                self.settings, detected_game=self.detected_game
            )
            turn = {
                "role": "user",
                "content": [
                    {"type": "text", "text": self._question_text(text)},
                    *self._image_parts(frames, "question"),
                ],
            }
            async with self._lock:
                answer = await self._call_with_history(
                    instruction, turn, lambda messages: self._stream(model, messages)
                )
                # The journal block rides in the request and not in history: it
                # is prepended to *every* question, and remembering twenty copies
                # of it is the single largest avoidable cost in this mode.
                self._remember(turn, answer, frames, history_text=text)
        except asyncio.CancelledError:
            raise
        except Exception as error:
            logger.warning("Answer failed: %s", error)
            self.errorOccurred.emit(str(error))
            if is_permanent_error(error):
                self._set_status("error", "check the model and key in Settings")
            return

        if answer.strip():
            self._conversation.append((time.time(), "chiron", answer.strip()))
            self.writer.observe_model_message(answer)
        self.responseCompleted.emit(answer)

    def _question_text(self, text: str) -> str:
        """The player's question, preceded by the journal it should be read with."""
        block = build_journal_block(self._journal_text())
        frames = "" if self._latest_frame else "\n(No screenshot is available.)"
        return f"{block}\n\n{text}{frames}" if block else f"{text}{frames}"

    async def _wait_for_fresh_frame(self, asked_at: float) -> None:
        """Wait briefly for a frame captured after the question was asked.

        The question already burst the shutter, so one is usually milliseconds
        away and waiting buys an answer about *now* rather than about four
        seconds ago. When nothing is watching, there is no frame coming and the
        wait is skipped outright rather than spent.
        """
        latest = self._latest_frame
        if latest is None or time.time() - latest.captured_at > CAPTURE_IDLE_SECONDS:
            return
        deadline = time.monotonic() + FRESH_FRAME_WAIT_SECONDS
        while True:
            frame = self._latest_frame
            if frame is not None and frame.captured_at >= asked_at:
                return
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return
            self._frame_event.clear()
            try:
                await asyncio.wait_for(self._frame_event.wait(), timeout=remaining)
            except asyncio.TimeoutError:
                return

    # ---------------------------------------------------------- context budget

    async def _call_with_history(
        self,
        instruction: str,
        turn: dict[str, Any],
        send: Callable[[list[dict[str, Any]]], Awaitable[str]],
    ) -> str:
        """Assemble ``[system, *history, turn]``, keep it in the window, and send.

        Both call paths go through here, because both grow the same history and
        both fail the same way when it gets too big. The order is proactive first
        and reactive second: measure and summarise before sending, and if the
        provider disagrees with the measurement anyway, compact and try once
        more. A rejected request bills nothing, so the backstop costs latency
        rather than money — and it is what makes an unknown pasted model id safe
        to select, since nothing offline can know that model's real window.

        Must be called with :attr:`_lock` held: it mutates history.
        """
        model_id = litellm_model_id(self.settings.selected_model)
        messages = [{"role": "system", "content": instruction}, *self._history, turn]
        if ctx.should_compact(model_id, messages):
            await self._compact_history("threshold")
            messages = [
                {"role": "system", "content": instruction},
                *self._history,
                turn,
            ]
        try:
            return await send(messages)
        except asyncio.CancelledError:
            raise
        except Exception as error:
            if not ctx.is_context_overflow(error):
                raise
            logger.info("Context window exceeded; compacting and retrying once")
            outcome = await self._compact_history("overflow")
            if outcome is None:
                raise
            messages = [
                {"role": "system", "content": instruction},
                *self._history,
                turn,
            ]
            return await send(messages)

    async def _compact_history(self, reason: str) -> ctx.CompactionOutcome | None:
        """Summarise older history into one message and keep a recent tail.

        The summariser runs on the selected model, or on ``observer_model`` in
        split mode — same distil-don't-converse job the observer already does
        there, and the same reason to want it cheap. Recorded as
        ``kind="compaction"``, because compaction spend that hides inside the
        question totals is spend nobody can find later.

        Returns:
            CompactionOutcome | None: What happened, or None when there was
                nothing safe to drop or the summariser failed — in which case
                history is left exactly as it was and the caller carries on.
        """
        model_id = litellm_model_id(self.settings.selected_model)
        head, tail = ctx.split_history(self._history)
        if not head:
            return None
        previous, head = ctx.extract_previous_summary(head)
        if not head:
            return None

        before = ctx.count_tokens(model_id, self._history)
        summariser_id = self.settings.observer_model_id()
        model = self._model(summariser_id, max_tokens=900)
        try:
            summary = await self._complete(
                model,
                ctx.build_summary_request(
                    head, previous=previous, journal=self._journal_text()
                ),
                kind="compaction",
            )
        except asyncio.CancelledError:
            raise
        except Exception as error:  # noqa: BLE001 — never let this kill an answer
            logger.warning("Compaction failed; keeping history as it is: %s", error)
            return None
        if not summary.strip():
            return None

        self._history = [ctx.summary_message(summary.strip()), *tail]
        self._compactions += 1
        outcome = ctx.CompactionOutcome(
            messages=self._history,
            summary=summary.strip(),
            tokens_before=before,
            tokens_after=ctx.count_tokens(model_id, self._history),
            kept_tail_count=len(tail),
            dropped_count=len(head) + (1 if previous else 0),
            reason=reason,
        )
        logger.info(
            "Compacted history (%s): %d → %d tokens, %d kept",
            reason,
            outcome.tokens_before,
            outcome.tokens_after,
            outcome.kept_tail_count,
        )
        self.compacted.emit(
            {
                "mode": "nonlive",
                "summary": outcome.summary,
                "tokens_before": outcome.tokens_before,
                "tokens_after": outcome.tokens_after,
                "kept_tail_count": outcome.kept_tail_count,
                "dropped_count": outcome.dropped_count,
                "reason": reason,
                "call_id": self._last_call_id,
            }
        )
        return outcome

    # -------------------------------------------------------------- the model

    def _model(self, model_id: str, *, max_tokens: int = 600) -> LiteLLMModel:
        """A litellm wrapper for one call, credentialed for its own provider.

        Built per call rather than held, so a settings change takes effect on the
        next request with nothing to invalidate. The cumulative counters a
        long-lived instance would carry are not the ledger — but that only holds
        if something is actually listening, so the usage sink is installed here,
        on every instance, closing over the gameplay session the call belongs to.
        A throwaway model with no sink is a call that was made and never booked.
        """
        return LiteLLMModel(
            model_id=model_id,
            api_key=self.settings.key_for_model(model_id) or None,
            temperature=0.3,
            max_tokens=max_tokens,
            usage_sink=self._on_usage_record,
            usage_labels={"session_id": self.session_id or None},
        )

    def _on_usage_record(self, record: LLMCallRecord) -> None:
        """Publish one priced call, and remember its id for the event that caused it."""
        self._last_call_id = record.id
        self.llmCall.emit(record)

    async def _complete(
        self, model: LiteLLMModel, messages: list[dict[str, Any]], *, kind: str
    ) -> str:
        """One non-streaming call, returning its text."""
        response = await model.acompletion(messages=messages)
        self._record_usage(model, getattr(response, "usage", None), response, kind)
        try:
            message = response.choices[0].message
        except (AttributeError, IndexError, TypeError):
            return ""
        content = (
            message.get("content")
            if isinstance(message, dict)
            else getattr(message, "content", None)
        )
        return content or ""

    async def _stream(self, model: LiteLLMModel, messages: list[dict[str, Any]]) -> str:
        """One streaming call, emitting deltas as they arrive.

        Returns:
            str: The complete answer.
        """
        chunks: list[str] = []
        usage: Any = None
        last_chunk: Any = None
        started = False
        stream = await model.acompletion(messages=messages, stream=True)
        async for chunk in stream:
            last_chunk = chunk
            usage = getattr(chunk, "usage", None) or usage
            text = _delta_text(chunk)
            if not text:
                continue
            if not started:
                started = True
                self.responseStarted.emit()
            chunks.append(text)
            self.responseDelta.emit(text)
        if not started:
            self.responseStarted.emit()
        self._record_usage(model, usage, last_chunk, "nonlive_qa", streamed=True)
        return "".join(chunks)

    def _record_usage(
        self,
        model: LiteLLMModel,
        usage: Any,
        response: Any,
        kind: str,
        *,
        streamed: bool = False,
    ) -> None:
        """Book one call's spend, never letting accounting break the app.

        `kind` is a full :data:`~chiron.models.usage.LLMCallKind` value, not a
        fragment to be prefixed. It used to be the latter, which is how the app
        came to pass values the Literal rejected — every record raised on
        construction and was swallowed here, so nothing was ever recorded and
        nothing said so.
        """
        try:
            model.record_usage(
                usage,
                response=response,
                context={"kind": kind, "streamed": streamed},
            )
        except Exception:  # pragma: no cover - accounting is observability
            logger.debug("Usage accounting failed for a %s call", kind, exc_info=True)

    # ------------------------------------------------------------- assembling

    def _recent_frames(self, count: int) -> list[Frame]:
        """The `count` newest frames, oldest first."""
        return list(self._frames)[-max(1, count) :] if self._frames else []

    def _image_parts(self, frames: list[Frame], reason: str) -> list[dict[str, Any]]:
        """Frames as data-URI image parts, counting and announcing them as sent.

        This is the one place a held frame becomes a frame the model has seen, so
        it is also where the recorder is told — what the session record shows is
        exactly what the AI looked at, never the frames the ring buffer held and
        threw away.
        """
        parts: list[dict[str, Any]] = []
        for frame in frames:
            encoded = base64.b64encode(frame.jpeg).decode("ascii")
            parts.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{encoded}"},
                }
            )
            self._frames_sent += 1
            self.frameSent.emit(frame, reason)
        return parts

    def _journal_text(self) -> str:
        """The recent journal, rendered."""
        return self.journal.render(self.journal.recent(JOURNAL_CONTEXT_ENTRIES))

    def _conversation_text(self) -> str:
        """Recent player/Chiron lines, for the split observer."""
        return "\n".join(
            f"{time.strftime('%H:%M', time.localtime(when))} {role}: {text}"
            for when, role, text in self._conversation
        )

    def _remember(
        self,
        turn: dict[str, Any],
        reply: str,
        frames: list[Frame],
        *,
        history_text: str | None = None,
    ) -> None:
        """Append a completed exchange to history, stripped of what repeats.

        This is where history cost is kept linear. Two things come out. The
        images go, each replaced by the clock it was captured at, so the model
        keeps the thread of what was asked and when it was looking without paying
        again for pixels it has already read. And `history_text` replaces the
        turn's text when the version that was *sent* carried the journal block —
        the journal is prepended to every question and is in the next request
        already, so remembering it here would accumulate one stale copy per
        exchange, which was the largest single avoidable cost in this mode.

        Args:
            turn (dict): The user turn as it was sent.
            reply (str): The model's answer.
            frames (list[Frame]): Frames attached to the turn, for their clocks.
            history_text (str | None): Text to remember instead of the turn's
                own. None keeps what was sent, which is right for the observer,
                whose prompt carries no journal block.
        """
        remembered = _without_images(turn, frames)
        if history_text is not None:
            remembered = _with_text(remembered, history_text)
        self._history.append(remembered)
        self._history.append({"role": "assistant", "content": reply})
        overflow = len(self._history) - MAX_HISTORY_MESSAGES
        if overflow > 0:
            del self._history[:overflow]

    def _set_status(self, status: str, detail: str = "") -> None:
        """Record and announce a status change."""
        self.status = status
        self.statusChanged.emit(status, detail)


def _without_images(
    message: dict[str, Any], frames: list[Frame] | None = None
) -> dict[str, Any]:
    """A copy of `message` with each image replaced by the clock it was taken at.

    The placeholder is the frame's own capture time rather than "an image was
    here", because the whole reason frames carry a burned-in clock is so the
    model can reason about when something was true. Dropping the image should
    not also drop the *when*.
    """
    content = message.get("content")
    if not isinstance(content, list):
        return dict(message)
    clocks = iter(frame.clock for frame in frames or [])
    parts: list[dict[str, Any]] = []
    for part in content:
        if isinstance(part, dict) and part.get("type") == "image_url":
            clock = next(clocks, time.strftime("%H:%M:%S", time.localtime()))
            parts.append({"type": "text", "text": f"[frame {clock}]"})
        else:
            parts.append(part)
    return {**message, "content": parts}


def _with_text(message: dict[str, Any], text: str) -> dict[str, Any]:
    """A copy of `message` whose first text part is replaced by `text`.

    Used to drop the journal block from a remembered question while leaving the
    frame placeholders that came after it untouched.
    """
    content = message.get("content")
    if not isinstance(content, list):
        return {**message, "content": text}
    parts: list[dict[str, Any]] = []
    replaced = False
    for part in content:
        if not replaced and isinstance(part, dict) and part.get("type") == "text":
            parts.append({"type": "text", "text": text})
            replaced = True
        else:
            parts.append(part)
    if not replaced:
        parts.insert(0, {"type": "text", "text": text})
    return {**message, "content": parts}


def _delta_text(chunk: Any) -> str:
    """The text in one streaming chunk, tolerating shapes."""
    try:
        choice = chunk.choices[0]
    except (AttributeError, IndexError, TypeError):
        return ""
    delta = getattr(choice, "delta", None)
    if delta is None and isinstance(choice, dict):
        delta = choice.get("delta")
    if delta is None:
        return ""
    content = (
        delta.get("content")
        if isinstance(delta, dict)
        else getattr(delta, "content", None)
    )
    return content or ""


__all__ = [
    "CAPTURE_IDLE_SECONDS",
    "FRESH_FRAME_WAIT_SECONDS",
    "MAX_HISTORY_MESSAGES",
    "QA_FRAME_COUNT",
    "NonLiveSessionManager",
]
