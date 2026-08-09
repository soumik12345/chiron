"""Buffered request/response implementation of the write-only Observer role."""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from functools import partial
from typing import Callable

from PySide6.QtCore import QObject, QTimer, Signal

from chiron.capture.frames import Frame
from chiron.config.settings import Settings
from chiron.journal.compaction import JournalCompactor
from chiron.journal.service import JournalService
from chiron.models.litellm_model import LiteLLMModel
from chiron.models.usage import LLMCallRecord, call_timer, utc_now_iso
from chiron.nonlive import compaction as ctx
from chiron.observer.media import (
    EncodedVideo,
    VideoEncodingError,
    encode_split_mp4,
    reduced_detail,
)
from chiron.observer.session import is_permanent_error
from chiron.observer.structured import (
    StructuredJournalEntry,
    StructuredObserverError,
    build_batch_messages,
    response_format,
    response_text,
    validate_result,
)

logger = logging.getLogger(__name__)

SAMPLE_CAP = 300
MAX_QUEUED_BATCHES = 3
PROVIDER_BACKOFF_SECONDS = (1.0, 2.0, 4.0)
DEFAULT_DRAIN_TIMEOUT = 90.0


def is_permanent_nonlive_error(error: BaseException) -> bool:
    """Whether retrying the same model/key/media contract cannot help."""
    text = str(error).lower()
    return is_permanent_error(error) or any(
        marker in text
        for marker in (
            "unsupported video",
            "video input is not supported",
            "unsupported modality",
            "response_format is not supported",
            "structured output is not supported",
            "invalid schema",
            "invalid api key",
            "authentication failed",
            "unauthorized",
            "status code: 401",
            "status code: 403",
            "model not found",
        )
    )


@dataclass(frozen=True)
class BufferedFrame:
    frame: Frame
    reason: str


@dataclass
class SealedBatch:
    id: str
    items: tuple[BufferedFrame, ...]
    seal_reason: str
    settings: Settings
    detected_game: str
    session_id: str
    encoded: list[EncodedVideo] | None = None
    next_part: int = 0
    journal_entry_count: int = 0
    successful_call_ids: list[str] = field(default_factory=list)
    delivered_parts: set[int] = field(default_factory=set)
    validated_parts: dict[int, tuple[StructuredJournalEntry, ...]] = field(
        default_factory=dict
    )
    committed_entries: dict[int, int] = field(default_factory=dict)
    commit_attempts: set[tuple[int, int]] = field(default_factory=set)

    @property
    def frames(self) -> tuple[Frame, ...]:
        return tuple(item.frame for item in self.items)

    @property
    def remaining_frame_count(self) -> int:
        if self.encoded is None:
            return len(self.items)
        return sum(len(part.frames) for part in self.encoded[self.next_part :])


class RetainedBatchError(RuntimeError):
    """A sealed batch remains available for an explicit retry or discard."""


class NonLiveObserverSessionManager(QObject):
    """Collect frames, review sealed videos sequentially, and write the journal."""

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
        *,
        model_factory: Callable[..., LiteLLMModel] = LiteLLMModel,
        encoder: Callable[..., list[EncodedVideo]] = encode_split_mp4,
        max_queued_batches: int = MAX_QUEUED_BATCHES,
    ) -> None:
        super().__init__(parent)
        self.settings = settings
        self.journal = journal
        self.journal_compactor = journal_compactor
        self.mode = "nonlive"
        self.status = "idle"
        self.status_detail = ""
        self._detected_game = ""
        self.session_id = ""
        self.last_sampled_at: float | None = None
        self.last_observed_at: float | None = None
        self.next_process_at: float | None = None

        self._model_factory = model_factory
        self._encoder = encoder
        self._max_queued_batches = max(1, max_queued_batches)
        self._active: list[BufferedFrame] = []
        self._active_context: tuple[Settings, str, str] | None = None
        self._sealed: deque[SealedBatch] = deque()
        self._processing_task: asyncio.Task[None] | None = None
        self._watch_requested = False
        self._accepting_frames = False
        self._draining = False
        self._restart_requested = False
        self._retained_error = False
        self._retained_detail = ""
        self._frames_sent = 0

        self._process_timer = QTimer(self)
        self._process_timer.setSingleShot(True)
        self._process_timer.timeout.connect(self._on_process_deadline)

    # ---------------------------------------------------------------- state

    @property
    def accepting_frames(self) -> bool:
        return self._accepting_frames

    @property
    def pending_frames(self) -> int:
        return len(self._active) + sum(
            batch.remaining_frame_count for batch in self._sealed
        )

    @property
    def frames_sent(self) -> int:
        return self._frames_sent

    @property
    def detected_game(self) -> str:
        return self._detected_game

    @detected_game.setter
    def detected_game(self, value: str) -> None:
        text = str(value or "")
        if text != self._detected_game and self._active:
            # A game identity is prompt context. Keep old and new frames out of
            # the same logical request without inventing another public seal kind.
            self._seal("final_flush")
        self._detected_game = text

    # ------------------------------------------------------------------ API

    def start(self) -> None:
        self._watch_requested = True
        if self._retained_error:
            self._restart_requested = True
            self._set_status("error", self._retained_detail)
            return
        if self._draining:
            self._restart_requested = True
            self._set_status("draining", f"{self.pending_frames} frames pending")
            return
        key = self.settings.key_for_model(self.settings.observer_model)
        if not key:
            self._accepting_frames = False
            self._set_status("error", "no API key for the selected Observer model")
            self.errorOccurred.emit(self.status_detail)
            return
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            self._set_status("preparing", "waiting for the application event loop")
            return
        self._accepting_frames = True
        self._arm_deadline()
        self._set_status("watching", "buffering frames")
        self._ensure_processor()

    async def stop(self, timeout: float = DEFAULT_DRAIN_TIMEOUT) -> None:
        """Stop acceptance immediately and drain every already accepted frame."""
        self.begin_drain()
        task = self._processing_task
        if task is not None and not task.done():
            try:
                await asyncio.wait_for(asyncio.shield(task), timeout=timeout)
            except asyncio.TimeoutError:
                task.cancel()
                self._set_status(
                    "error",
                    f"drain timed out; {self.pending_frames} frames were not observed",
                )
                self.errorOccurred.emit(self.status_detail)
                return
        if self._watch_requested:
            return
        self._draining = False
        if self._sealed:
            return
        self._set_status("stopped", "")

    def begin_drain(self) -> None:
        """Synchronously close the privacy gate before the async drain awaits."""
        if self._draining:
            return
        self._watch_requested = False
        self._restart_requested = False
        self._accepting_frames = False
        self._process_timer.stop()
        self.next_process_at = None
        self._draining = True
        self._seal("final_flush")
        self._set_status("draining", f"{self.pending_frames} frames pending")
        self._ensure_processor()

    def observe(self, frame: Frame, reason: str = "scheduled") -> None:
        if not self._accepting_frames:
            return
        if not self._active:
            self._active_context = (
                self.settings.copy_deep(),
                self.detected_game,
                self.session_id,
            )
        self._active.append(BufferedFrame(frame, reason))
        self.last_sampled_at = frame.captured_at
        if len(self._active) >= SAMPLE_CAP:
            self._seal("sample_cap")

    def reset_memory(self) -> None:
        if self.pending_frames:
            raise RuntimeError("Cannot reset Observer memory while frames are pending")
        self.last_sampled_at = None
        self.last_observed_at = None
        self._frames_sent = 0

    def apply_settings(self, settings: Settings) -> None:
        # An active buffer owns the context captured with its first frame; only a
        # newly started batch sees these settings.
        self.settings = settings

    def retry_retained(self) -> None:
        """Retry the oldest retained batch after the player fixes its configuration."""
        if not self._sealed:
            return
        if self.settings.observer_model.startswith("live/"):
            raise RetainedBatchError(
                "A buffered video cannot be retried through a Live model; select "
                "a batched model or explicitly discard it"
            )
        batch = self._sealed[0]
        retry = batch.settings.copy_deep()
        retry.observer_model = self.settings.observer_model
        retry.api_key = self.settings.api_key
        retry.openrouter_api_key = self.settings.openrouter_api_key
        batch.settings = retry
        self._retained_error = False
        self._retained_detail = ""
        self._set_status("retrying", f"retrying {batch.id}")
        self._ensure_processor()

    async def retry_retained_and_wait(
        self,
        settings: Settings | None = None,
        *,
        timeout: float = DEFAULT_DRAIN_TIMEOUT,
    ) -> bool:
        """Apply repaired provider settings, retry, and report full completion."""
        if settings is not None:
            self.apply_settings(settings)
        self.retry_retained()
        task = self._processing_task
        if task is not None and not task.done():
            try:
                await asyncio.wait_for(asyncio.shield(task), timeout=timeout)
            except asyncio.TimeoutError:
                task.cancel()
                self._retained_error = True
                self._retained_detail = "Observer batch retry timed out"
                self._set_status("error", self._retained_detail)
                self.errorOccurred.emit(self._retained_detail)
                return False
        return not self._sealed

    def discard_retained(self) -> int:
        """Explicitly discard retained frames; callers must obtain user confirmation."""
        count = self.pending_frames
        self._active.clear()
        self._active_context = None
        self._sealed.clear()
        self._retained_error = False
        self._retained_detail = ""
        self._draining = False
        self._restart_requested = False
        if self._watch_requested:
            self._accepting_frames = True
            self._arm_deadline()
            self._set_status("watching", "buffering frames")
        else:
            self._set_status("stopped", "")
        return count

    # ------------------------------------------------------------- scheduling

    def _arm_deadline(self) -> None:
        if not self._accepting_frames:
            self.next_process_at = None
            return
        seconds = self.settings.capture.process_interval_seconds
        self.next_process_at = time.time() + seconds
        self._process_timer.start(max(1, round(seconds * 1000)))

    def _on_process_deadline(self) -> None:
        if not self._accepting_frames:
            return
        self._seal("interval")
        if self._accepting_frames:
            self._arm_deadline()

    def _seal(self, reason: str) -> None:
        if not self._active:
            return
        context = self._active_context or (
            self.settings.copy_deep(),
            self.detected_game,
            self.session_id,
        )
        batch = SealedBatch(
            id=uuid.uuid4().hex,
            items=tuple(self._active),
            seal_reason=reason,
            settings=context[0],
            detected_game=context[1],
            session_id=context[2],
        )
        self._active = []
        self._active_context = None
        self._sealed.append(batch)
        if len(self._sealed) >= self._max_queued_batches and not self._draining:
            self._accepting_frames = False
            self._process_timer.stop()
            self.next_process_at = None
            self._set_status("backpressure", f"{self.pending_frames} frames pending")
        self._ensure_processor()

    def _ensure_processor(self) -> None:
        if self._processing_task is not None and not self._processing_task.done():
            return
        if not self._sealed:
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        self._processing_task = loop.create_task(self._process_loop())

    async def _process_loop(self) -> None:
        try:
            while self._sealed:
                batch = self._sealed[0]
                self._set_status(
                    "reviewing", f"Reviewing {batch.remaining_frame_count} frames"
                )
                try:
                    await self._process_batch(batch)
                except Exception as error:  # retained intentionally
                    state = (
                        "incompatible"
                        if isinstance(error, StructuredObserverError)
                        or is_permanent_nonlive_error(error)
                        else "error"
                    )
                    self._accepting_frames = False
                    self._retained_error = True
                    self._retained_detail = str(error)
                    self._set_status(state, str(error))
                    self.errorOccurred.emit(str(error))
                    return
                self._sealed.popleft()
                self._retained_error = False
                self._retained_detail = ""
                if (
                    self._watch_requested
                    and not self._draining
                    and len(self._sealed) < self._max_queued_batches
                ):
                    self._accepting_frames = True
                    if not self._process_timer.isActive():
                        self._arm_deadline()
            if self._draining:
                self._draining = False
                if self._restart_requested:
                    self._restart_requested = False
                    self.start()
                elif not self._watch_requested:
                    self._set_status("stopped", "")
            elif self._watch_requested and self._accepting_frames:
                self._set_status("watching", "buffering frames")
        finally:
            self._processing_task = None

    # --------------------------------------------------------------- pipeline

    async def _process_batch(self, batch: SealedBatch) -> None:
        if batch.encoded is None:
            detail = batch.settings.capture.media_resolution
            try:
                batch.encoded = await self._encode(batch.frames, detail)
            except Exception:
                try:
                    batch.encoded = await self._encode(
                        batch.frames, reduced_detail(detail)
                    )
                except Exception as error:
                    raise VideoEncodingError(
                        "Observer video encoding failed twice; the JPEG batch is retained"
                    ) from error

        while batch.next_part < len(batch.encoded):
            part_index = batch.next_part
            part = batch.encoded[part_index]
            if part_index not in batch.delivered_parts:
                reasons = {item.frame.captured_at: item.reason for item in batch.items}
                for frame in part.frames:
                    self.frameSent.emit(
                        frame, reasons.get(frame.captured_at, "scheduled")
                    )
                    self._frames_sent += 1
                batch.delivered_parts.add(part_index)
            entries = batch.validated_parts.get(part_index)
            if entries is None:
                entries = await self._review_part(batch, part)
                batch.validated_parts[part_index] = entries
            # All validation completed before the first write for this response.
            committed = batch.committed_entries.get(part_index, 0)
            for position, entry in enumerate(entries[committed:], start=committed):
                frame = part.frames[entry.video_second]
                attempt_key = (part_index, position)
                if attempt_key in batch.commit_attempts and any(
                    existing.timestamp == frame.captured_at
                    and existing.note == entry.note
                    and existing.category == entry.category
                    and existing.source == "observer"
                    for existing in self.journal.snapshot().entries
                ):
                    batch.committed_entries[part_index] = position + 1
                    continue
                batch.commit_attempts.add(attempt_key)
                self.journal.record(
                    entry.note,
                    entry.category,
                    source="observer",
                    timestamp=frame.captured_at,
                )
                batch.committed_entries[part_index] = position + 1
            batch.journal_entry_count += len(entries) - committed
            batch.next_part += 1
            self.last_observed_at = part.frames[-1].captured_at

        payload = {
            "agent_id": "observer",
            "batch_id": batch.id,
            "seal_reason": batch.seal_reason,
            "first_captured_at": batch.frames[0].captured_at,
            "last_captured_at": batch.frames[-1].captured_at,
            "source_frame_count": len(batch.frames),
            "encoded_part_count": len(batch.encoded),
            "journal_entry_count": batch.journal_entry_count,
            "successful_call_ids": list(batch.successful_call_ids),
            "last_observed_at": self.last_observed_at,
        }
        self.observerRan.emit(payload)

    async def _encode(
        self, frames: tuple[Frame, ...], detail: str
    ) -> list[EncodedVideo]:
        """Run PyAV outside Qt/asyncio without leaking a loop-owned executor."""
        executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="chiron-video")
        try:
            return await asyncio.get_running_loop().run_in_executor(
                executor, partial(self._encoder, frames, detail)
            )
        finally:
            # On normal completion the worker is already done. On a bounded quit
            # timeout, do not block the shared event loop waiting for local media.
            executor.shutdown(wait=False, cancel_futures=True)

    async def _review_part(self, batch: SealedBatch, part: EncodedVideo):
        force = False
        repair = ""
        repaired = False
        overflow_retried = False
        provider_attempt = 0
        while True:
            snapshot = (
                await self.journal_compactor.prepare(
                    batch.settings.observer_model,
                    force=force,
                    reason="observer_overflow" if force else "observer_threshold",
                )
                if self.journal_compactor is not None
                else self.journal.snapshot()
            )
            messages = build_batch_messages(
                batch.settings,
                part,
                snapshot,
                detected_game=batch.detected_game,
                repair=repair,
            )
            model = self._model(batch)
            started_at = utc_now_iso()
            elapsed = call_timer()
            try:
                response = await model.acompletion(
                    messages=messages,
                    response_format=response_format(len(part.frames)),
                    extra_body=(
                        {"provider": {"require_parameters": True}}
                        if batch.settings.observer_model.startswith("openrouter/")
                        else None
                    ),
                )
            except Exception as error:
                if ctx.is_context_overflow(error) and not overflow_retried:
                    overflow_retried = True
                    force = True
                    model.record_failure(
                        error,
                        context={
                            "kind": "nonlive_observer",
                            "status": "retry",
                            "attempt": provider_attempt + 1,
                            "started_at": started_at,
                            "duration_ms": elapsed(),
                        },
                    )
                    continue
                if ctx.is_context_overflow(error):
                    raise RetainedBatchError(
                        "The Observer still exceeded its context after compaction; "
                        "choose a larger-context model"
                    ) from error
                if is_permanent_nonlive_error(error):
                    model.record_failure(
                        error,
                        context={
                            "kind": "nonlive_observer",
                            "attempt": provider_attempt + 1,
                            "started_at": started_at,
                            "duration_ms": elapsed(),
                        },
                    )
                    raise
                if provider_attempt < len(PROVIDER_BACKOFF_SECONDS):
                    model.record_failure(
                        error,
                        context={
                            "kind": "nonlive_observer",
                            "status": "retry",
                            "attempt": provider_attempt + 1,
                            "started_at": started_at,
                            "duration_ms": elapsed(),
                        },
                    )
                    delay = PROVIDER_BACKOFF_SECONDS[provider_attempt]
                    provider_attempt += 1
                    self._set_status("retrying", f"retrying in {delay:g}s")
                    await asyncio.sleep(delay)
                    continue
                model.record_failure(
                    error,
                    context={
                        "kind": "nonlive_observer",
                        "attempt": provider_attempt + 1,
                        "started_at": started_at,
                        "duration_ms": elapsed(),
                    },
                )
                raise RetainedBatchError(
                    "Observer provider retries were exhausted; the batch is retained"
                ) from error

            choice = response.choices[0]
            model.record_usage(
                getattr(response, "usage", None),
                response=response,
                context={
                    "kind": "nonlive_observer",
                    "attempt": provider_attempt + 1,
                    "started_at": started_at,
                    "duration_ms": elapsed(),
                    "finish_reason": getattr(choice, "finish_reason", None),
                },
            )
            try:
                return validate_result(response_text(response), len(part.frames))
            except StructuredObserverError as error:
                if repaired:
                    raise StructuredObserverError(
                        "Observer returned invalid structured output twice; "
                        "the selected model/route is incompatible and the batch is retained"
                    ) from error
                repaired = True
                repair = str(error)
                provider_attempt = 0

    def _model(self, batch: SealedBatch) -> LiteLLMModel:
        def usage_sink(record: LLMCallRecord) -> None:
            self.llmCall.emit(record)
            if record.status == "ok":
                batch.successful_call_ids.append(record.id)

        model_id = batch.settings.observer_model
        return self._model_factory(
            model_id=model_id,
            api_key=batch.settings.key_for_model(model_id),
            temperature=0.2,
            max_tokens=1200,
            reasoning_effort="none",
            usage_sink=usage_sink,
            usage_labels={
                "session_id": batch.session_id or None,
                "agent_id": "observer",
                "run_id": batch.id,
            },
        )

    def _set_status(self, status: str, detail: str) -> None:
        self.status = status
        self.status_detail = detail
        self.statusChanged.emit(status, detail)


__all__ = [
    "DEFAULT_DRAIN_TIMEOUT",
    "MAX_QUEUED_BATCHES",
    "PROVIDER_BACKOFF_SECONDS",
    "SAMPLE_CAP",
    "BufferedFrame",
    "NonLiveObserverSessionManager",
    "RetainedBatchError",
    "SealedBatch",
    "is_permanent_nonlive_error",
]
