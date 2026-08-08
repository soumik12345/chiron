"""Gemini Live Observer restricted to scheduled journal-writing checkpoints."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from PySide6.QtCore import QObject, Signal

from chiron.capture.frames import Frame
from chiron.config.settings import Settings, live_model_id
from chiron.journal.compaction import JournalCompactor
from chiron.journal.service import JournalService
from chiron.live import estimate
from chiron.nonlive import compaction as ctx
from chiron.observer.prompts import (
    CHECKPOINT_INSTRUCTION,
    build_journal_context,
    build_observer_instruction,
)

logger = logging.getLogger(__name__)

MEDIA_RESOLUTIONS = {
    "low": "MEDIA_RESOLUTION_LOW",
    "medium": "MEDIA_RESOLUTION_MEDIUM",
    "high": "MEDIA_RESOLUTION_HIGH",
}
BACKOFF_SECONDS = (1.0, 2.0, 4.0, 8.0, 15.0, 30.0)
STOP_TIMEOUT_SECONDS = 5.0
COMPACTION_TRIGGER_RATIO = 0.78
COMPACTION_WAIT_SECONDS = 10.0


def is_permanent_error(error: BaseException) -> bool:
    """Whether retrying the same Live configuration cannot help."""
    text = str(error).lower()
    return any(
        marker in text
        for marker in (
            "not supported by the model",
            "response modalities",
            "api key not valid",
            "api_key_invalid",
            "permission_denied",
            "was not found",
            "is not found",
        )
    )


class LiveObserverSessionManager(QObject):
    """Own one reconnecting Live socket with no transcript output surface."""

    statusChanged = Signal(str, str)
    errorOccurred = Signal(str)
    frameSent = Signal(object, str)
    compacted = Signal(object)
    llmCall = Signal(object)
    observerRan = Signal(object)

    def __init__(
        self,
        settings: Settings,
        journal: JournalService,
        journal_compactor: JournalCompactor | None = None,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self.settings = settings
        self.journal = journal
        self.journal_compactor = journal_compactor
        self.status = "idle"
        self.status_detail = ""
        self.mode = "live"
        self.detected_game = ""
        self.session_id = ""
        self.last_sampled_at: float | None = None
        self.last_observed_at: float | None = None
        self.next_process_at: float | None = None

        self._task: asyncio.Task[None] | None = None
        self._current_cycle: asyncio.Task[Any] | None = None
        self._active_session: Any = None
        self._stopping = False
        self._rotate_reason = ""
        self._resumption_handle: str | None = None

        self._wake = asyncio.Event()
        self._send_lock = asyncio.Lock()
        self._turn_done = asyncio.Event()
        self._turn_done.set()
        self._pending_frame: tuple[Frame, str] | None = None
        self._observation_in_flight = False
        self._inflight_frame: Frame | None = None
        self._inflight_started_at: str | None = None
        self._inflight_timer = None
        self._inflight_prompt_text = ""
        self._inflight_output_text = ""
        self._inflight_output_tokens = 0

        self._frames_sent = 0
        self._context_frames = 0
        self._context_chars = 0
        self._server_tokens: int | None = None
        self._compacting = False
        self._compactions = 0

    @property
    def frames_sent(self) -> int:
        return self._frames_sent

    @property
    def accepting_frames(self) -> bool:
        return self.status == "live" and not self._stopping

    @property
    def pending_frames(self) -> int:
        return int(self._pending_frame is not None) + int(self._observation_in_flight)

    @property
    def compactions(self) -> int:
        return self._compactions

    @property
    def is_connected(self) -> bool:
        return self.status == "live"

    @property
    def estimated_context_tokens(self) -> int:
        if self._server_tokens is not None:
            return self._server_tokens
        return estimate.estimate_context_tokens(
            frames=self._context_frames,
            media_resolution=self.settings.capture.media_resolution,
            transcript_chars=self._context_chars,
        )

    # ------------------------------------------------------------------ API

    def start(self) -> None:
        if self._task is not None and not self._task.done():
            return
        self._stopping = False
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            # Construction and synchronous Qt tests happen before qasync owns the
            # thread. The real application starts Watch from that running loop;
            # declining to manufacture an orphaned coroutine keeps this boundary
            # safe for embedders too.
            self._set_status("connecting", "waiting for the application event loop")
            return
        self._set_status("connecting", "opening Observer")
        self._task = loop.create_task(self._run())

    async def stop(self, timeout: float = STOP_TIMEOUT_SECONDS) -> None:
        self._stopping = True
        self._pending_frame = None
        self._wake.set()
        task, self._task = self._task, None
        if task is not None and not task.done():
            task.cancel()
            try:
                await asyncio.wait_for(task, timeout=timeout)
            except (asyncio.CancelledError, asyncio.TimeoutError):
                pass
        self._active_session = None
        self._observation_in_flight = False
        self._inflight_frame = None
        self._inflight_started_at = None
        self._inflight_timer = None
        self._inflight_prompt_text = ""
        self._inflight_output_text = ""
        self._inflight_output_tokens = 0
        self._turn_done.set()
        self._set_status("stopped", "")

    def observe(self, frame: Frame, reason: str = "scheduled") -> None:
        """Offer one checkpoint, coalescing any unsent older frame."""
        if self.status != "live" or self._stopping:
            return
        self._pending_frame = (frame, reason)
        self.last_sampled_at = frame.captured_at
        self._wake.set()

    def reset_memory(self) -> None:
        self._resumption_handle = None
        self.last_sampled_at = None
        self.last_observed_at = None
        self._pending_frame = None
        self._observation_in_flight = False
        self._inflight_frame = None
        self._inflight_started_at = None
        self._inflight_timer = None
        self._inflight_prompt_text = ""
        self._inflight_output_text = ""
        self._inflight_output_tokens = 0
        self._turn_done.set()
        self._frames_sent = 0
        self._compactions = 0
        self._reset_context_accounting()
        if self._task is not None and not self._task.done():
            self.rotate("new session", fresh=True)

    def apply_settings(self, settings: Settings) -> None:
        self.settings = settings

    def rotate(self, reason: str, *, fresh: bool = False) -> None:
        if fresh:
            self._resumption_handle = None
        self._rotate_reason = reason
        cycle = self._current_cycle
        if cycle is not None and not cycle.done():
            cycle.cancel()

    # ------------------------------------------------------------- config

    def _build_config(self) -> Any:
        from google.genai import types

        declarations = [
            types.FunctionDeclaration(**declaration)
            for declaration in self.journal.function_declarations
        ]
        resolution = MEDIA_RESOLUTIONS.get(
            self.settings.capture.media_resolution, "MEDIA_RESOLUTION_LOW"
        )
        return types.LiveConnectConfig(
            response_modalities=[types.Modality.AUDIO],
            # Audio is unavoidable for the served model family, but no output
            # transcription or content route exists on this object.
            system_instruction=build_observer_instruction(
                self.settings, detected_game=self.detected_game
            ),
            media_resolution=getattr(types.MediaResolution, resolution),
            tools=[types.Tool(function_declarations=declarations)],
            realtime_input_config=types.RealtimeInputConfig(
                automatic_activity_detection=types.AutomaticActivityDetection(
                    disabled=True
                )
            ),
            history_config=types.HistoryConfig(initial_history_in_client_content=True),
            context_window_compression=types.ContextWindowCompressionConfig(
                sliding_window=types.SlidingWindow()
            ),
            session_resumption=types.SessionResumptionConfig(
                handle=self._resumption_handle
            ),
        )

    # ------------------------------------------------------------- lifecycle

    async def _run(self) -> None:
        failures = 0
        while not self._stopping:
            api_key = self.settings.resolved_api_key()
            if not api_key:
                message = "No Gemini API key configured. Open Settings."
                self._set_status("error", message)
                self.errorOccurred.emit(message)
                return
            reason, self._rotate_reason = self._rotate_reason, ""
            self._set_status(
                "connecting",
                f"reconnecting after {reason}" if reason else "opening Observer",
            )
            try:
                await self._session_cycle(api_key)
                failures = 0
            except asyncio.CancelledError:
                if self._stopping:
                    raise
                continue
            except Exception as error:  # noqa: BLE001 - reconnect boundary
                failures += 1
                logger.warning("Observer Live error (%d): %s", failures, error)
                self.errorOccurred.emit(str(error))
                if is_permanent_error(error):
                    self._set_status("error", "check Observer model and Google key")
                    return
            if self._stopping:
                break
            delay = BACKOFF_SECONDS[min(failures, len(BACKOFF_SECONDS) - 1)]
            self._set_status(
                "reconnecting", f"retrying in {delay:.0f}s" if failures else "rotating"
            )
            await asyncio.sleep(delay if failures else 0.2)
        self._set_status("stopped", "")

    async def _session_cycle(self, api_key: str) -> None:
        from google import genai

        resuming = bool(self._resumption_handle)
        client = genai.Client(api_key=api_key)
        async with client.aio.live.connect(
            model=live_model_id(self.settings.observer_model),
            config=self._build_config(),
        ) as session:
            self._active_session = session
            self._reset_context_accounting()
            self._set_status("live", live_model_id(self.settings.observer_model))
            if not resuming:
                await self._seed(session)

            receive_task = asyncio.ensure_future(self._receive(session))
            send_task = asyncio.ensure_future(self._send_pump(session))
            self._current_cycle = asyncio.ensure_future(
                asyncio.wait(
                    {receive_task, send_task}, return_when=asyncio.FIRST_COMPLETED
                )
            )
            try:
                done, _ = await self._current_cycle
            finally:
                self._current_cycle = None
                self._active_session = None
                for task in (receive_task, send_task):
                    if not task.done():
                        task.cancel()
                await asyncio.gather(receive_task, send_task, return_exceptions=True)
            for task in done:
                error = task.exception()
                if error is not None:
                    raise error

    async def _seed(self, session: Any) -> None:
        snapshot = (
            await self.journal_compactor.prepare(
                self.settings.observer_model, reason="observer_seed"
            )
            if self.journal_compactor is not None
            else self.journal.snapshot()
        )
        text = snapshot.render()
        if not text:
            return
        await self._send_context(session, build_journal_context(text))

    # --------------------------------------------------------------- sending

    async def _send_pump(self, session: Any) -> None:
        while True:
            pending, self._pending_frame = self._pending_frame, None
            if pending is not None:
                await self._send_observation(session, *pending)
                continue
            self._wake.clear()
            # Close the lost-wakeup window after clearing the Event.
            if self._pending_frame is not None:
                self._wake.set()
                continue
            await self._wake.wait()

    async def _send_observation(self, session: Any, frame: Frame, reason: str) -> None:
        """Send one non-interleavable activity/video/text/activity transaction."""
        from google.genai import types

        async with self._send_lock:
            self._turn_done.clear()
            self._observation_in_flight = True
            self._inflight_frame = frame
            self._inflight_started_at = estimate.utc_now_iso()
            self._inflight_timer = estimate.call_timer()
            self._inflight_prompt_text = CHECKPOINT_INSTRUCTION
            self._inflight_output_text = ""
            self._inflight_output_tokens = 0
            await session.send_realtime_input(activity_start=types.ActivityStart())
            await session.send_realtime_input(
                video=types.Blob(data=frame.jpeg, mime_type="image/jpeg")
            )
            await session.send_realtime_input(text=CHECKPOINT_INSTRUCTION)
            await session.send_realtime_input(activity_end=types.ActivityEnd())
            self._frames_sent += 1
            self._context_frames += 1
            self._context_chars += len(CHECKPOINT_INSTRUCTION)
            self.frameSent.emit(frame, reason)
            await self._turn_done.wait()
        self._consider_compaction()

    async def _send_context(self, session: Any, text: str) -> None:
        from google.genai import types

        if not text:
            return
        started_at = estimate.utc_now_iso()
        async with self._send_lock:
            try:
                await session.send_client_content(
                    turns=types.Content(role="user", parts=[types.Part(text=text)]),
                    turn_complete=False,
                )
            except Exception as error:
                logger.info("Observer client-history replay rejected: %s", error)
                await session.send_realtime_input(text=text)
            self._context_chars += len(text)
            self.llmCall.emit(
                estimate.estimate_turn(
                    model_id=self.settings.observer_model,
                    frames=0,
                    media_resolution=self.settings.capture.media_resolution,
                    prompt_text=text,
                    session_id=self.session_id or None,
                    started_at=started_at,
                    agent_id="observer",
                    kind="observer_context",
                )
            )

    # ------------------------------------------------------------- receiving

    async def _receive(self, session: Any) -> None:
        while True:
            received = False
            async for message in session.receive():
                received = True
                reported = estimate.read_usage_metadata(message)
                if reported is not None:
                    self._server_tokens = reported
                update = getattr(message, "session_resumption_update", None)
                if update is not None and update.resumable and update.new_handle:
                    self._resumption_handle = update.new_handle
                if getattr(message, "go_away", None) is not None:
                    self._set_status("reconnecting", "server is recycling Observer")
                    return
                tool_call = getattr(message, "tool_call", None)
                if tool_call is not None:
                    await self._handle_tool_call(session, tool_call)
                content = getattr(message, "server_content", None)
                if content is not None:
                    turn = getattr(content, "model_turn", None)
                    for part in getattr(turn, "parts", None) or []:
                        inline = getattr(part, "inline_data", None)
                        if inline is not None:
                            data = getattr(inline, "data", b"")
                            if isinstance(data, str):
                                # Some transports expose wire base64 rather than
                                # the SDK's usual decoded PCM bytes.
                                byte_count = len(data) * 3 // 4
                            else:
                                try:
                                    byte_count = len(data)
                                except TypeError:
                                    byte_count = 0
                            self._inflight_output_tokens += (
                                estimate.audio_output_tokens(byte_count)
                            )
                if content is not None and getattr(content, "turn_complete", False):
                    self._finish_observation()
            # session.receive() covers one turn. A turn may contain only tool
            # protocol messages, so iterator completion is also a valid boundary.
            if received:
                self._finish_observation()
                continue
            return

    async def _handle_tool_call(self, session: Any, tool_call: Any) -> None:
        from google.genai import types

        responses = []
        for call in tool_call.function_calls or []:
            self._inflight_output_text += f"{call.name} {dict(call.args or {})}"
            try:
                result = await self.journal.handle_observer_tool(
                    call.name, dict(call.args or {})
                )
            except Exception as error:  # noqa: BLE001 - acknowledge tool failure
                result = {"error": str(error)}
            responses.append(
                types.FunctionResponse(id=call.id, name=call.name, response=result)
            )
        if responses:
            await session.send_tool_response(function_responses=responses)
            acknowledgement = str(
                [getattr(response, "response", {}) for response in responses]
            )
            self._inflight_prompt_text += acknowledgement
            self._context_chars += len(acknowledgement)

    def _finish_observation(self) -> None:
        if not self._observation_in_flight:
            return
        frame = self._inflight_frame
        if frame is not None:
            self.last_observed_at = frame.captured_at
            timer = self._inflight_timer
            self.llmCall.emit(
                estimate.estimate_turn(
                    model_id=self.settings.observer_model,
                    frames=1,
                    media_resolution=self.settings.capture.media_resolution,
                    prompt_text=self._inflight_prompt_text,
                    output_text=self._inflight_output_text,
                    output_tokens=self._inflight_output_tokens,
                    session_id=self.session_id or None,
                    started_at=self._inflight_started_at,
                    duration_ms=timer() if timer is not None else None,
                    agent_id="observer",
                    kind="observer_checkpoint",
                )
            )
        self._observation_in_flight = False
        self._inflight_frame = None
        self._inflight_started_at = None
        self._inflight_timer = None
        self._inflight_prompt_text = ""
        self._inflight_output_text = ""
        self._inflight_output_tokens = 0
        self._turn_done.set()

    # ------------------------------------------------------------- compaction

    def _consider_compaction(self) -> None:
        if (
            self._compacting
            or self.status != "live"
            or self.estimated_context_tokens
            < int(
                ctx.context_window_for(self.settings.observer_model)
                * COMPACTION_TRIGGER_RATIO
            )
        ):
            return
        self._compacting = True
        asyncio.get_running_loop().create_task(self._compact_and_rotate())

    async def _compact_and_rotate(self) -> None:
        before = self.estimated_context_tokens
        try:
            try:
                await asyncio.wait_for(
                    self._turn_done.wait(), timeout=COMPACTION_WAIT_SECONDS
                )
            except asyncio.TimeoutError:
                logger.warning("Observer compaction deadline reached mid-transaction")
            self._compactions += 1
            self.compacted.emit(
                {
                    "agent_id": "observer",
                    "mode": "observer_rotate",
                    "tokens_before": before,
                    "tokens_after": 0,
                    "reason": "context threshold",
                }
            )
            self.rotate("context compaction", fresh=True)
        finally:
            self._compacting = False

    def _reset_context_accounting(self) -> None:
        self._context_frames = 0
        self._context_chars = 0
        self._server_tokens = None

    def _set_status(self, status: str, detail: str) -> None:
        self.status = status
        self.status_detail = detail
        self.statusChanged.emit(status, detail)


# Compatibility for imports from the v3 Live-only architecture. New application
# wiring uses the explicit class name through the Observer factory.
ObserverSessionManager = LiveObserverSessionManager


__all__ = [
    "COMPACTION_TRIGGER_RATIO",
    "LiveObserverSessionManager",
    "MEDIA_RESOLUTIONS",
    "ObserverSessionManager",
    "is_permanent_error",
]
