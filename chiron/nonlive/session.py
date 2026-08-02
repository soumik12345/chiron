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
into the frames remain the temporal ground truth either way.
"""

from __future__ import annotations

import asyncio
import base64
import logging
import time
from collections import deque
from typing import Any

from PySide6.QtCore import QObject, Signal

from chiron.capture.frames import Frame
from chiron.config.settings import Settings, litellm_model_id
from chiron.journal.log import JournalLog
from chiron.journal.writers import JournalWriter, parse_sidecar_entries
from chiron.live.session import is_permanent_error
from chiron.models.litellm_model import LiteLLMModel
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

    Attributes:
        settings (Settings): Configuration used to build the next request.
        journal (JournalLog): The shared journal — primary memory in this mode.
        writer (JournalWriter): The route observer entries take to the log and
            the overlay.
        trigger (ObserverTrigger): When looking is worth paying for.
        status (str): Current status.
        detected_game (str): The focused window when watching started.
    """

    statusChanged = Signal(str, str)
    responseStarted = Signal()
    responseDelta = Signal(str)
    responseCompleted = Signal(str)
    journalFolded = Signal(int)
    errorOccurred = Signal(str)

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
        self._stopping = False

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
            return
        logger.info("Observer running (%s) with %d frames", reason, len(frames))

        if self.settings.agent_mode == "split":
            content = await self._observe_split(frames)
        else:
            content = await self._observe_unified(frames)

        entries = parse_sidecar_entries(content)
        for note, category in entries:
            self.writer.record(note, category)
        self._observer_runs += 1
        self.trigger.note_run(time.time())
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
                    *self._image_parts(frames),
                ],
            },
        ]
        return await self._complete(model, messages, kind="observer")

    async def _observe_unified(self, frames: list[Frame]) -> str:
        """An observer tick inside the conversation the player's questions use.

        The cost is real — every tick pays for the whole conversation context —
        and so is the benefit: when the player finally asks something, they are
        asking a model that has been watching, not one reading notes about it.
        """
        model = self._model(litellm_model_id(self.settings.selected_model))
        turn = {
            "role": "user",
            "content": [
                {"type": "text", "text": OBSERVER_TURN_INSTRUCTION},
                *self._image_parts(frames),
            ],
        }
        async with self._lock:
            messages = [
                {
                    "role": "system",
                    "content": build_observer_instruction(
                        self.settings, detected_game=self.detected_game
                    ),
                },
                *self._history,
                turn,
            ]
            content = await self._complete(model, messages, kind="observer")
            self._remember(turn, content or "(nothing new)", frames)
        return content

    # --------------------------------------------------------------- the Q&A

    async def _answer(self, text: str, asked_at: float) -> None:
        """Answer one question, streaming the reply into the overlay."""
        try:
            await self._wait_for_fresh_frame(asked_at)
            frames = self._recent_frames(QA_FRAME_COUNT)
            model = self._model(litellm_model_id(self.settings.selected_model))
            turn = {
                "role": "user",
                "content": [
                    {"type": "text", "text": self._question_text(text)},
                    *self._image_parts(frames),
                ],
            }
            async with self._lock:
                messages = [
                    {
                        "role": "system",
                        "content": build_qa_instruction(
                            self.settings, detected_game=self.detected_game
                        ),
                    },
                    *self._history,
                    turn,
                ]
                answer = await self._stream(model, messages)
                self._remember(turn, answer, frames)
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

    # -------------------------------------------------------------- the model

    def _model(self, model_id: str, *, max_tokens: int = 600) -> LiteLLMModel:
        """A litellm wrapper for one call, credentialed for its own provider.

        Built per call rather than held, so a settings change takes effect on the
        next request with nothing to invalidate. The cumulative counters a
        long-lived instance would carry are not the ledger — every call reports
        itself through :meth:`~chiron.models.litellm_model.LiteLLMModel.record_usage`.
        """
        return LiteLLMModel(
            model_id=model_id,
            api_key=self.settings.key_for_model(model_id) or None,
            temperature=0.3,
            max_tokens=max_tokens,
        )

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
        self._record_usage(model, usage, last_chunk, "qa", streamed=True)
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
        """Book one call's spend, never letting accounting break the app."""
        try:
            model.record_usage(
                usage,
                response=response,
                context={"kind": f"nonlive_{kind}", "streamed": streamed},
            )
        except Exception:  # pragma: no cover - accounting is observability
            logger.debug("Usage accounting failed for a %s call", kind, exc_info=True)

    # ------------------------------------------------------------- assembling

    def _recent_frames(self, count: int) -> list[Frame]:
        """The `count` newest frames, oldest first."""
        return list(self._frames)[-max(1, count) :] if self._frames else []

    def _image_parts(self, frames: list[Frame]) -> list[dict[str, Any]]:
        """Frames as data-URI image parts, counting them as sent."""
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

    def _remember(self, turn: dict[str, Any], reply: str, frames: list[Frame]) -> None:
        """Append a completed exchange to history, with the frames taken out.

        This is where history cost is kept linear: the turn goes in as text,
        each image replaced by the clock it was captured at. The model keeps the
        thread of what was asked and when it was looking, and stops paying for
        pixels it has already read.
        """
        self._history.append(_without_images(turn, frames))
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
