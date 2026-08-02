"""The Live API session: one websocket, kept alive across its own deaths.

A Live session with video is a short-lived thing. Without context-window
compression it is terminated at two minutes; with a sliding window it lives
indefinitely but silently forgets its oldest frames, and the connection itself
still gets recycled — the server sends `GoAway` shortly before it does. On top of
that, laptops suspend and wifi drops. So the manager is written around the
assumption that **the session will end and that is normal**:

* Compression is on, so duration is unbounded and eviction is the API's problem.
* Every resumption handle the server offers is kept, so an unexpected close
  reconnects into the *same* conversation rather than a blank one.
* `GoAway` triggers a proactive rotation rather than a wait for the axe.
* When resumption is not possible, a fresh session is opened and seeded with the
  system prompt plus the journal — which is exactly what the journal is for.

Everything runs on the Qt event loop through ``qasync``, so there is one thread
and no cross-thread bridging: the UI can call :meth:`send_text` directly, and the
signals this object emits are ordinary direct connections.

Sends are funnelled through a queue owned by the connection task, so nothing can
write to a socket that is in the middle of being swapped. Frames are the one
exception to ordinary queueing: only the newest one is worth sending, so a frame
waiting behind another frame replaces it instead of stacking up.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from PySide6.QtCore import QObject, Signal

from chiron.capture.frames import Frame
from chiron.config.settings import Settings
from chiron.journal.log import JournalLog
from chiron.journal.writers import JournalWriter
from chiron.live.prompts import build_journal_context, build_system_instruction

logger = logging.getLogger(__name__)

#: Media resolution setting -> API enum name.
MEDIA_RESOLUTIONS = {
    "low": "MEDIA_RESOLUTION_LOW",
    "medium": "MEDIA_RESOLUTION_MEDIUM",
    "high": "MEDIA_RESOLUTION_HIGH",
}

#: Reconnect backoff, in seconds, indexed by consecutive failure count.
_BACKOFF_SECONDS = (1.0, 2.0, 4.0, 8.0, 15.0, 30.0)

#: How long shutdown waits for the session, and then the journal writer, to
#: unwind before abandoning them. Quitting has to finish.
STOP_TIMEOUT_SECONDS = 5.0


async def _settle(awaitable: Any, timeout: float, what: str) -> bool:
    """Await something that is shutting down, giving up after `timeout`.

    `CancelledError` is swallowed because the caller is the one who asked for the
    cancellation — this helper is only ever used on the shutdown path, where a
    task ending as cancelled is success, not an error to propagate.

    Args:
        awaitable (Any): Task or coroutine being wound down.
        timeout (float): Seconds to wait before abandoning it.
        what (str): Name for the log line if it has to be abandoned.

    Returns:
        bool: True if it finished, False if it was abandoned.
    """
    try:
        await asyncio.wait_for(awaitable, timeout)
    except asyncio.CancelledError:
        pass
    except asyncio.TimeoutError:
        logger.warning(
            "Timed out stopping %s after %.1fs; abandoning it", what, timeout
        )
        return False
    except Exception as error:
        logger.warning("Error while stopping %s: %s", what, error)
    return True


class LiveSessionManager(QObject):
    """Owns the Live API websocket and the conversation running over it.

    Signals:
        statusChanged (str, str): Connection status (``idle``, ``connecting``,
            ``live``, ``reconnecting``, ``error``, ``stopped``) and a detail line.
        responseStarted (): The model began answering.
        responseDelta (str): A chunk of the answer.
        responseCompleted (str): The answer is complete; carries the full text.
        journalFolded (int): N journal entries were folded into the session.
        errorOccurred (str): Something went wrong, in words fit for the overlay.

    Attributes:
        settings (Settings): Configuration used for the next connection.
        journal (JournalLog): The shared journal.
        writer (JournalWriter): Strategy handling tool calls and summarisation.
        status (str): Current connection status.
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
        """Create a disconnected manager."""
        super().__init__(parent)
        self.settings = settings
        self.journal = journal
        self.writer = writer
        self.status = "idle"
        #: The focused window when watching started, already phrased for the
        #: instruction. Runtime state, not a setting: it is discovered, changes
        #: per play session, and must never be persisted as the player's choice.
        self.detected_game = ""

        self._task: asyncio.Task[None] | None = None
        self._queue: asyncio.Queue[tuple[str, Any]] = asyncio.Queue()
        self._latest_frame: Frame | None = None
        self._frame_queued = False
        self._resumption_handle: str | None = None
        self._model_has_spoken = False
        self._stopping = False
        self._rotate_reason = ""
        self._response_buffer: list[str] = []
        self._frames_sent = 0
        self._current_cycle: asyncio.Task[None] | None = None

    # ------------------------------------------------------------------ API

    @property
    def is_connected(self) -> bool:
        """Whether a session is currently established."""
        return self.status == "live"

    @property
    def frames_sent(self) -> int:
        """How many frames this manager has pushed to the API."""
        return self._frames_sent

    def start(self) -> None:
        """Open a session and keep it open until :meth:`stop`."""
        if self._task is not None and not self._task.done():
            return
        self._stopping = False
        self._task = asyncio.ensure_future(self._run())

    async def stop(self, timeout: float = STOP_TIMEOUT_SECONDS) -> None:
        """Close the session and stop reconnecting.

        Bounded on purpose. This runs on the way out of the application, and an
        await that never returns there is indistinguishable to the user from the
        app refusing to close — the exact complaint a quit button exists to
        answer. Cancelling a task that is suspended inside a websocket's cleanup
        is *usually* instant, but "usually" is not a guarantee worth betting the
        only exit on, so anything still unwinding after `timeout` is abandoned
        and the process carries on shutting down.

        Args:
            timeout (float): Seconds to wait for the session task, and then for
                the journal writer, before giving up on each.
        """
        self._stopping = True
        task, self._task = self._task, None
        if task is not None and not task.done():
            task.cancel()
            await _settle(task, timeout, "live session")
        await _settle(self.writer.stop(), timeout, "journal writer")
        self._set_status("stopped", "")

    def send_text(self, text: str) -> None:
        """Queue a player message for the model.

        Args:
            text (str): What the player typed. Blank input is ignored.
        """
        if not text.strip():
            return
        self.writer.observe_user_message(text)
        self._queue.put_nowait(("text", text))

    def send_frame(self, frame: Frame) -> None:
        """Queue a captured frame, superseding any frame still waiting.

        Args:
            frame (Frame): The newest frame. If an earlier one has not been sent
                yet it is dropped — a stale screenshot is worse than no
                screenshot, and the API is rate-limited to 1 fps regardless.
        """
        self._latest_frame = frame
        self.writer.observe_frame(frame)
        if not self._frame_queued:
            self._frame_queued = True
            self._queue.put_nowait(("frame", None))

    def fold_journal(self, *, force: bool = False) -> int:
        """Send journal entries the session has not seen yet.

        This is the other half of the memory story: the model's frames age out,
        but the text distilled from them gets pushed back in as context, where it
        costs a hundredth as much and lasts far longer.

        Args:
            force (bool): Fold the whole recent journal rather than only entries
                added since the last fold. Used when seeding a new session.

        Returns:
            int: How many entries were folded (0 when there was nothing new).
        """
        limit = self.settings.journal.fold_entry_limit
        entries = self.journal.recent(limit) if force else self.journal.unfolded()
        if not entries:
            return 0
        entries = entries[-limit:]
        text = build_journal_context(self.journal.render(entries), is_seed=force)
        if not text:
            return 0
        self._queue.put_nowait(("context", text))
        self.journal.mark_folded()
        self.journalFolded.emit(len(entries))
        return len(entries)

    def rotate(self, reason: str = "manual") -> None:
        """Tear the current session down so the loop opens a fresh one."""
        self._rotate_reason = reason
        task = self._current_cycle
        if task is not None and not task.done():
            task.cancel()

    def reset_observation(self) -> None:
        """Nothing to reset: the Live API decides for itself what it has seen.

        Part of the :class:`~chiron.session.SessionProvider` surface, where the
        non-live provider uses it to restart its novelty detector's warm-up. A
        live session has no such state — the model watches continuously, and the
        capture service's own scene-change baseline is reset separately.
        """

    def apply_settings(self, settings: Settings) -> None:
        """Adopt new settings for subsequent connections.

        The caller decides whether a restart is needed — see
        :meth:`~chiron.config.settings.Settings.requires_session_restart` — since
        model id, credential and system instruction are fixed at connect time.
        """
        self.settings = settings

    # ------------------------------------------------------------- internals

    def _set_status(self, status: str, detail: str = "") -> None:
        """Record and announce a status change."""
        self.status = status
        self.statusChanged.emit(status, detail)

    def _build_config(self) -> Any:
        """Build the ``LiveConnectConfig`` for a connection."""
        from google.genai import types

        declarations = self.writer.function_declarations()
        tools = (
            [
                types.Tool(
                    function_declarations=[
                        types.FunctionDeclaration(**d) for d in declarations
                    ]
                )
            ]
            if declarations
            else None
        )
        resolution = MEDIA_RESOLUTIONS.get(
            self.settings.capture.media_resolution, "MEDIA_RESOLUTION_LOW"
        )
        # `settings.live_model` is derived from the provider-qualified selection,
        # so a non-live selection reaching here (it should not) still connects
        # with a real Live model id rather than a 404.
        return types.LiveConnectConfig(
            # Every Live model still served is a native-audio one, and those
            # accept only the AUDIO response modality — asking for TEXT is
            # rejected outright with a 1007 close. The documented route to text
            # is to let the model speak and read its own transcription, which is
            # what `output_audio_transcription` turns on. The generated audio is
            # never played; the overlay renders the transcript.
            response_modalities=[types.Modality.AUDIO],
            output_audio_transcription=types.AudioTranscriptionConfig(),
            system_instruction=build_system_instruction(
                self.settings,
                journal_enabled=bool(declarations),
                detected_game=self.detected_game,
            ),
            media_resolution=getattr(types.MediaResolution, resolution),
            # Unbounded session duration, at the price of old frames being
            # evicted — which is the trade the journal exists to make survivable.
            context_window_compression=types.ContextWindowCompressionConfig(
                sliding_window=types.SlidingWindow()
            ),
            # Ask for handles even on the first connection: the one we need is
            # always the one issued before the disconnect we did not expect.
            session_resumption=types.SessionResumptionConfig(
                handle=self._resumption_handle
            ),
            tools=tools,
        )

    async def _run(self) -> None:
        """Connect, serve, reconnect — until stopped."""
        failures = 0
        await self.writer.start()
        while not self._stopping:
            api_key = self.settings.resolved_api_key()
            if not api_key:
                self._set_status("error", "No Gemini API key. Open Settings")
                self.errorOccurred.emit(
                    "No Gemini API key configured. Open Settings and paste one, "
                    "or export GEMINI_API_KEY."
                )
                return

            reason = self._rotate_reason
            self._rotate_reason = ""
            self._set_status(
                "connecting",
                f"reconnecting after {reason}" if reason else "opening session",
            )
            try:
                await self._session_cycle(api_key)
                failures = 0
            except asyncio.CancelledError:
                if self._stopping:
                    raise
                # A rotation cancelled the cycle; loop round and reconnect.
                logger.info("Session rotated: %s", self._rotate_reason or "requested")
                continue
            except Exception as error:
                failures += 1
                logger.warning("Live session error (%d): %s", failures, error)
                self.errorOccurred.emit(str(error))
                if is_permanent_error(error):
                    self._set_status("error", "check the model and key in Settings")
                    return
            if self._stopping:
                break
            delay = _BACKOFF_SECONDS[min(failures, len(_BACKOFF_SECONDS) - 1)]
            self._set_status(
                "reconnecting", f"retrying in {delay:.0f}s" if failures else "rotating"
            )
            await asyncio.sleep(delay if failures else 0.2)
        self._set_status("stopped", "")

    async def _session_cycle(self, api_key: str) -> None:
        """One connection: connect, seed, pump until either side finishes."""
        from google import genai

        client = genai.Client(api_key=api_key)
        async with client.aio.live.connect(
            model=self.settings.live_model, config=self._build_config()
        ) as session:
            self._model_has_spoken = False
            self._set_status("live", self.settings.live_model)
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
                for task in (receive_task, send_task):
                    if not task.done():
                        task.cancel()
                await asyncio.gather(receive_task, send_task, return_exceptions=True)
            for task in done:
                error = task.exception()
                if error is not None:
                    raise error

    async def _seed(self, session: Any) -> None:
        """Give a new session its starting context.

        Client content with ``turn_complete=False`` appends to the conversation
        without asking for a reply, which is exactly what seeding needs: the model
        should know what has happened, not narrate it back.
        """
        entries = self.journal.recent(self.settings.journal.fold_entry_limit)
        if not entries:
            return
        text = build_journal_context(self.journal.render(entries), is_seed=True)
        await self._send_context(session, text)
        self.journal.mark_folded()
        logger.info("Seeded new session with %d journal entries", len(entries))

    async def _send_pump(self, session: Any) -> None:
        """Drain the outbound queue onto the socket until cancelled."""
        while True:
            kind, payload = await self._queue.get()
            try:
                if kind == "text":
                    await self._send_user_text(session, payload)
                elif kind == "frame":
                    self._frame_queued = False
                    frame = self._latest_frame
                    if frame is not None:
                        await self._send_frame(session, frame)
                elif kind == "context":
                    await self._send_context(session, payload)
            except asyncio.CancelledError:
                raise
            except Exception as error:
                logger.warning("Failed to send %s: %s", kind, error)
                raise

    async def _send_user_text(self, session: Any, text: str) -> None:
        """Send a player message, respecting the API's seeding restriction.

        Newer Live models accept ``send_client_content`` only for initial context
        seeding; once the model has spoken, conversational text has to go through
        ``send_realtime_input``. Tracking whether the model has answered yet keeps
        both cases correct without sniffing model ids.
        """
        from google.genai import types

        if self._model_has_spoken:
            await session.send_realtime_input(text=text)
        else:
            await session.send_client_content(
                turns=types.Content(role="user", parts=[types.Part(text=text)]),
                turn_complete=True,
            )

    async def _send_frame(self, session: Any, frame: Frame) -> None:
        """Push one JPEG frame as realtime video input."""
        from google.genai import types

        await session.send_realtime_input(
            video=types.Blob(data=frame.jpeg, mime_type="image/jpeg")
        )
        self._frames_sent += 1

    async def _send_context(self, session: Any, text: str) -> None:
        """Append text to the conversation without requesting a reply."""
        from google.genai import types

        if not text:
            return
        try:
            await session.send_client_content(
                turns=types.Content(role="user", parts=[types.Part(text=text)]),
                turn_complete=False,
            )
        except Exception as error:
            # Some Live models refuse client content after the first turn. Falling
            # back to realtime text costs a short acknowledgement from the model
            # but keeps the journal reaching the session, which is the point.
            logger.info("Client-content fold rejected (%s); using realtime text", error)
            await session.send_realtime_input(text=text)

    async def _receive(self, session: Any) -> None:
        """Consume server messages until the session ends or rotation is due.

        ``session.receive()`` covers exactly **one** model turn — the SDK's
        iterator stops as soon as a turn completes. Chiron's session outlives any
        one answer, so each finished turn simply re-enters the iterator and waits
        for the next thing the server has to say. An iterator that ends without
        producing anything means the socket is gone, which returns and lets the
        connection loop reconnect.
        """
        while True:
            if not await self._receive_turn(session):
                return

    async def _receive_turn(self, session: Any) -> bool:
        """Consume one model turn.

        Returns:
            bool: True to keep listening, False when the session is finished —
                either the server said `GoAway` or the stream produced nothing.
        """
        received_anything = False
        async for message in session.receive():
            received_anything = True
            if message.session_resumption_update is not None:
                update = message.session_resumption_update
                if update.resumable and update.new_handle:
                    self._resumption_handle = update.new_handle

            if message.go_away is not None:
                left = getattr(message.go_away, "time_left", None)
                logger.info("GoAway received (time left: %s); rotating session", left)
                self._rotate_reason = "server GoAway"
                self._set_status("reconnecting", "server is recycling the session")
                return False

            if message.tool_call is not None:
                await self._handle_tool_call(session, message.tool_call)

            content = message.server_content
            if content is None:
                continue

            for text in _text_parts(content):
                if not self._response_buffer:
                    self._model_has_spoken = True
                    self.responseStarted.emit()
                self._response_buffer.append(text)
                self.responseDelta.emit(text)

            if content.turn_complete:
                answer = "".join(self._response_buffer)
                self._response_buffer.clear()
                if answer.strip():
                    self.writer.observe_model_message(answer)
                    self.responseCompleted.emit(answer)

        if not received_anything:
            logger.info("Receive stream closed by the server")
        return received_anything

    async def _handle_tool_call(self, session: Any, tool_call: Any) -> None:
        """Route function calls to the journal writer and answer them."""
        from google.genai import types

        responses = []
        for call in tool_call.function_calls or []:
            args = dict(call.args or {})
            try:
                result = await self.writer.handle_tool_call(call.name, args)
            except Exception as error:
                logger.warning("Tool call %s failed: %s", call.name, error)
                result = {"error": str(error)}
            responses.append(
                types.FunctionResponse(id=call.id, name=call.name, response=result)
            )
        if responses:
            await session.send_tool_response(function_responses=responses)


def _text_parts(content: Any) -> list[str]:
    """Extract the visible text from a ``LiveServerContent`` message.

    Two sources, because which one a model uses is not ours to choose. Native
    audio models answer in speech and report ``output_transcription`` chunks;
    a text-capable model puts text parts in its ``model_turn``. Reading both
    means the overlay renders whichever arrives, and audio payloads — which
    Chiron never plays — are simply ignored.

    Args:
        content (Any): The ``server_content`` field of a server message.

    Returns:
        list[str]: Non-empty text fragments, in order.
    """
    texts: list[str] = []

    transcription = getattr(content, "output_transcription", None)
    spoken = getattr(transcription, "text", None) if transcription else None
    if spoken:
        texts.append(spoken)

    turn = getattr(content, "model_turn", None)
    for part in (getattr(turn, "parts", None) or []) if turn else []:
        text = getattr(part, "text", None)
        if text:
            texts.append(text)
    return texts


def is_permanent_error(error: BaseException) -> bool:
    """Whether reconnecting could not possibly help.

    A bad key or a model that will not serve the requested modalities fails the
    same way every time, and retrying it forever buries the one message that
    would tell the player what to fix under a stream of reconnect notices.

    Args:
        error (BaseException): The failure raised by a connection attempt.

    Returns:
        bool: True when the failure is configuration, not connectivity.
    """
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


__all__ = ["MEDIA_RESOLUTIONS", "LiveSessionManager", "is_permanent_error"]
