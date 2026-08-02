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

**A live context cannot overflow — it forgets instead**, and v2 turns that into a
decision rather than an accident. Sliding-window compression silently evicts the
oldest content as 128k approaches; since frames dominate the budget, what
evaporates is the visual history, with no error and no notice of what was
dropped. So this manager watches its own estimated usage and, crossing ~100k,
does the deliberate version of what the reconnect path already does: force the
journal current, then rotate to a fresh session seeded from it — **dropping the
resumption handle first**, because handle-based resumption would faithfully
restore the very context being shed. Sliding-window compression stays on
underneath as the backstop; if the estimate drifts low or a rotation fails,
silent eviction remains infinitely better than termination.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

from PySide6.QtCore import QObject, Signal

from chiron.capture.frames import Frame
from chiron.capture.novelty import NoveltyDetector
from chiron.config.settings import LIVE_CONTEXT_TOKENS, Settings
from chiron.journal.log import JournalLog
from chiron.journal.writers import JournalWriter
from chiron.live import estimate
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

#: Estimated context tokens at which consolidation becomes due — ~78% of the
#: window. Well below it, because the trigger is allowed to wait for a quiet
#: moment and waiting has to be affordable.
COMPACTION_TOKENS = int(LIVE_CONTEXT_TOKENS * 0.78)

#: The point at which waiting stops being affordable and the rotation happens
#: mid-fight anyway. Gating must never be able to postpone into eviction, which
#: is the exact failure compaction exists to replace.
COMPACTION_DEADLINE_TOKENS = int(LIVE_CONTEXT_TOKENS * 0.90)

#: Seconds since the last novelty spike after which the screen counts as quiet.
#: A rotation is a few seconds of blindness; the whole point of gating is to
#: spend them between fights rather than during the boss.
QUIET_SECONDS = 4.0


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
        frameSent (object, str): A frame reached the model, always with the
            reason ``burst`` — in live mode every frame is streamed rather than
            chosen, which is the difference the reason exists to record.
        observerRan (object): Emitted for interface parity; never fires here.
            There is no observer in live mode; the model is watching.
        llmCall (object): One *estimated* :class:`~chiron.models.usage.LLMCallRecord`.
            The Live API reports nothing billable, so these are arithmetic and
            say so — see :mod:`chiron.live.estimate`.
        compacted (object): The session was consolidated and rotated.

    Attributes:
        settings (Settings): Configuration used for the next connection.
        journal (JournalLog): The shared journal.
        writer (JournalWriter): Strategy handling tool calls and summarisation.
        status (str): Current connection status.
        session_id (str): The gameplay session estimates are billed to. Runtime
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
        self.session_id = ""

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

        # --- context accounting, reset with every connection -----------------
        #: Frames and transcript in the *current* connection's context, which is
        #: not the same thing as the session totals: a rotation empties the
        #: server's context, and an estimate that kept counting would trigger
        #: another rotation moments later.
        self._context_frames = 0
        self._context_chars = 0
        #: The server's own token count, when a message carries one. Trusted over
        #: the estimate — it knows what it is actually holding — but absent often
        #: enough that the estimate cannot be retired.
        self._server_tokens: int | None = None
        self._compacting = False
        self._compactions = 0

        # --- what one estimated turn covers ---------------------------------
        self._turn_frames = 0
        self._turn_prompt: list[str] = []
        self._turn_started_at: str | None = None

        # --- quiet-moment gating --------------------------------------------
        #: A cheap novelty read over the frames already being sent, used for one
        #: decision only: whether now is a bad moment to blink.
        self._novelty = NoveltyDetector()
        self._last_spike_at = 0.0
        self._pending_question = False

    # ------------------------------------------------------------------ API

    @property
    def is_connected(self) -> bool:
        """Whether a session is currently established."""
        return self.status == "live"

    @property
    def frames_sent(self) -> int:
        """How many frames this manager has pushed to the API."""
        return self._frames_sent

    @property
    def compactions(self) -> int:
        """How many times this manager has consolidated and rotated."""
        return self._compactions

    @property
    def estimated_context_tokens(self) -> int:
        """How full the model's context is, as best as can be known from here.

        The server's ``usage_metadata`` when it has sent one, and the client-side
        estimate otherwise. The deferral in the design doc resolved this way
        round because the two disagree in a specific direction: the estimate
        cannot see the system instruction, the seeded journal or the audio the
        model generated, so it reads low, and reading low is the failure mode
        that ends in silent eviction.
        """
        if self._server_tokens is not None:
            return self._server_tokens
        return estimate.estimate_context_tokens(
            frames=self._context_frames,
            media_resolution=self.settings.capture.media_resolution,
            transcript_chars=self._context_chars,
        )

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
        # Frames sent since the last answer are real spend even though no turn
        # ever came to attribute them to — a watched evening with no questions
        # asked would otherwise estimate at zero, which is the wrong shape of
        # wrong for a cost figure.
        self._emit_turn_estimate("")
        self._set_status("stopped", "")

    def send_text(self, text: str) -> None:
        """Queue a player message for the model.

        Args:
            text (str): What the player typed. Blank input is ignored.
        """
        if not text.strip():
            return
        self.writer.observe_user_message(text)
        self._pending_question = True
        self._turn_prompt.append(text)
        self._context_chars += len(text)
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
        reading = self._novelty.observe(frame.signature, frame.captured_at)
        if reading.spike:
            self._last_spike_at = frame.captured_at
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

    def rotate(self, reason: str = "manual", *, fresh: bool = False) -> None:
        """Tear the current session down so the loop opens a fresh one.

        Args:
            reason (str): What prompted the rotation, for the status line.
            fresh (bool): Drop the resumption handle first, so the new
                connection starts blank and :meth:`_seed` replays the journal
                into it. This is what separates a *compaction* rotate from a
                reconnect: keeping the handle restores the server-side context,
                which would faithfully resurrect everything being shed. Every
                other caller wants the handle kept, which is why it is off by
                default.
        """
        if fresh:
            self._resumption_handle = None
        self._rotate_reason = reason
        task = self._current_cycle
        if task is not None and not task.done():
            task.cancel()

    def reset_observation(self) -> None:
        """Forget what the screen looked like before now.

        The Live API decides for itself what it has seen, so there is no model
        state to reset — but the novelty read that decides whether *now* is a
        quiet moment is this manager's own, and comparing across a gap nobody
        watched reads as an event every time.
        """
        self._novelty.reset(time.time())
        self._last_spike_at = 0.0

    def reset_memory(self) -> None:
        """Forget the conversation entirely — a new gameplay session started.

        The live context lives on the server, so the only way to clear it is to
        connect again without the handle that would restore it. A session that
        is not running simply drops the handle, and the next connection is blank
        by construction.
        """
        self._resumption_handle = None
        self._response_buffer.clear()
        self._model_has_spoken = False
        self._reset_context_accounting()
        self.reset_observation()
        if self._task is not None and not self._task.done():
            self.rotate("new session", fresh=True)

    def apply_settings(self, settings: Settings) -> None:
        """Adopt new settings for subsequent connections.

        The caller decides whether a restart is needed — see
        :meth:`~chiron.config.settings.Settings.requires_session_restart` — since
        model id, credential and system instruction are fixed at connect time.
        """
        self.settings = settings

    # ----------------------------------------------------------- compaction

    def is_quiet(self, now: float | None = None) -> bool:
        """Whether this is a good moment to blink.

        Two conditions, both cheap and both already measured: no question is
        waiting on an answer, and the screen has not spiked recently. A player
        mid-boss is exactly who a few seconds of blindness costs the most.
        """
        current = time.time() if now is None else now
        if self._pending_question:
            return False
        return current - self._last_spike_at >= QUIET_SECONDS

    def compaction_due(self, now: float | None = None) -> str:
        """Whether consolidation should happen now, and why.

        Returns:
            str: ``"quiet"`` when the threshold is crossed and the moment is
                right, ``"deadline"`` when it is crossed hard enough that the
                moment no longer gets a vote, and empty otherwise.
        """
        tokens = self.estimated_context_tokens
        if tokens >= COMPACTION_DEADLINE_TOKENS:
            return "deadline"
        if tokens >= COMPACTION_TOKENS and self.is_quiet(now):
            return "quiet"
        return ""

    def _consider_compaction(self) -> None:
        """Start a consolidation if one is due and none is already running."""
        if self._stopping or self._compacting or self.status != "live":
            return
        reason = self.compaction_due()
        if not reason:
            return
        self._compacting = True
        asyncio.ensure_future(self._compact_and_rotate(reason))

    async def _compact_and_rotate(self, reason: str) -> None:
        """Consolidate memory into the journal, then reconnect without a handle.

        The live equivalent of compaction, in the only shape a server-side,
        immutable context allows. In live mode the journal *is* the running
        compaction summary, so the work is making it current — a sidecar pass
        over the recent transcript when that strategy is active, then a forced
        fold so the facts reach the session that is about to end, in case the
        rotation fails. Then a fresh connection, which :meth:`_seed` opens by
        replaying the journal.

        Net effect: instead of amnesia, the model blinks for a few seconds and
        comes back knowing a distilled version of the whole evening.
        """
        before = self.estimated_context_tokens
        try:
            summarise = getattr(self.writer, "summarise_once", None)
            if callable(summarise):
                try:
                    await summarise()
                except asyncio.CancelledError:
                    raise
                except Exception as error:  # noqa: BLE001
                    logger.warning("Sidecar pass before compaction failed: %s", error)
            folded = self.fold_journal(force=True)
            entries = len(self.journal)
            logger.info(
                "Compacting live session (%s) at ~%d tokens; %d entries folded",
                reason,
                before,
                folded,
            )
            self.compacted.emit(
                {
                    "mode": "live_rotate",
                    "summary": self.journal.render(
                        self.journal.recent(self.settings.journal.fold_entry_limit)
                    ),
                    "tokens_before": before,
                    "tokens_after": 0,
                    "kept_tail_count": entries,
                    "dropped_count": self._context_frames,
                    "reason": reason,
                }
            )
            self._compactions += 1
            self.rotate("compaction", fresh=True)
        finally:
            self._compacting = False

    def _reset_context_accounting(self) -> None:
        """Start the context estimate again — a new connection holds nothing."""
        self._context_frames = 0
        self._context_chars = 0
        self._server_tokens = None

    # ------------------------------------------------------------- estimates

    def _emit_turn_estimate(self, answer: str) -> None:
        """Publish an estimated ledger row covering everything since the last one.

        Arithmetic, not measurement: the Live API bills over a websocket and
        reports nothing chiron can invoice against, so every row this emits is
        tagged ``pricing_source="estimated"`` and every surface that renders a
        total puts a ``~`` in front of it.
        """
        frames, self._turn_frames = self._turn_frames, 0
        prompt = " ".join(self._turn_prompt)
        self._turn_prompt.clear()
        started_at, self._turn_started_at = self._turn_started_at, None
        if not frames and not prompt and not answer.strip():
            return
        record = estimate.estimate_turn(
            model_id=self.settings.live_model,
            frames=frames,
            media_resolution=self.settings.capture.media_resolution,
            prompt_text=prompt,
            output_text=answer,
            session_id=self.session_id or None,
            started_at=started_at,
        )
        self.llmCall.emit(record)

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
            # A new connection holds nothing yet — including one opened by a
            # compaction rotate, which is the whole point of having rotated.
            self._reset_context_accounting()
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
        self._context_frames += 1
        self._turn_frames += 1
        if self._turn_started_at is None:
            self._turn_started_at = estimate.utc_now_iso()
        self.frameSent.emit(frame, "burst")
        # Frames are what fill a live context, so this is the natural place to
        # ask whether it is getting full.
        self._consider_compaction()

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
            # The server has always sent this and the loop has always ignored
            # it. It is the only authoritative answer to "how full is the
            # context", which is the question compaction turns on.
            reported = estimate.read_usage_metadata(message)
            if reported is not None:
                self._server_tokens = reported

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
                self._pending_question = False
                self._context_chars += len(answer)
                if answer.strip():
                    self.writer.observe_model_message(answer)
                    self.responseCompleted.emit(answer)
                self._emit_turn_estimate(answer)
                # A finished answer is the other natural checkpoint, and by
                # definition a moment with no question waiting on one.
                self._consider_compaction()

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


__all__ = [
    "COMPACTION_DEADLINE_TOKENS",
    "COMPACTION_TOKENS",
    "MEDIA_RESOLUTIONS",
    "QUIET_SECONDS",
    "LiveSessionManager",
    "is_permanent_error",
]
