"""Fixed-interval screen capture with a cadence-neutral immediate path."""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from typing import Any

from PySide6.QtCore import QObject, Signal

from chiron.capture.frames import ScreenGrabber, encode_frame
from chiron.capture.scheduler import FixedIntervalScheduler
from chiron.config.settings import CaptureSettings

logger = logging.getLogger(__name__)

_TICK_SECONDS = 0.05
_MAX_CONSECUTIVE_ERRORS = 5


class CaptureService(QObject):
    """Capture only while effective Watch is active.

    Scheduled frames use :attr:`frameCaptured`. An immediate question capture
    uses :attr:`immediateFrameCaptured` with the caller's opaque token and does
    not update :class:`FixedIntervalScheduler`.
    """

    frameCaptured = Signal(object)
    immediateFrameCaptured = Signal(object, object)
    errorOccurred = Signal(str)

    def __init__(
        self, settings: CaptureSettings, parent: QObject | None = None
    ) -> None:
        super().__init__(parent)
        self._lock = threading.Lock()
        self._settings = settings
        self._stop = threading.Event()
        self._watching = threading.Event()
        self._thread: threading.Thread | None = None
        self._immediate: deque[Any] = deque()
        self.scheduler = FixedIntervalScheduler(settings.interval_seconds)

    @property
    def settings(self) -> CaptureSettings:
        with self._lock:
            return self._settings

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    @property
    def is_watching(self) -> bool:
        return self._watching.is_set()

    def set_watching(self, watching: bool) -> None:
        if watching == self.is_watching:
            return
        if watching:
            with self._lock:
                self.scheduler.reset()
                self._immediate.clear()
            self._watching.set()
        else:
            self._watching.clear()
            with self._lock:
                self._immediate.clear()
        logger.info("Capture %s", "started" if watching else "stopped")

    def start(self) -> None:
        if self.is_running:
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run, name="chiron-capture", daemon=True
        )
        self._thread.start()

    def stop(self, timeout: float = 2.0) -> None:
        self._stop.set()
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=timeout)
        self._thread = None

    def apply_settings(self, settings: CaptureSettings) -> None:
        with self._lock:
            self._settings = settings
            self.scheduler.interval_seconds = settings.interval_seconds

    def request_immediate(self, token: Any = None) -> bool:
        """Request exactly one extra capture without moving the periodic deadline."""
        if not self.is_watching:
            return False
        with self._lock:
            self._immediate.append(token)
        return True

    def clear_pending(self) -> None:
        with self._lock:
            self._immediate.clear()

    def _run(self) -> None:
        grabber = ScreenGrabber(self.settings.monitor_index)
        errors = 0
        try:
            while not self._stop.is_set():
                if not self._watching.is_set():
                    self._stop.wait(_TICK_SECONDS)
                    continue

                now = time.time()
                with self._lock:
                    settings = self._settings
                    immediate = bool(self._immediate)
                    token = self._immediate.popleft() if immediate else None
                    due = self.scheduler.is_due(now)
                if not immediate and not due:
                    self._stop.wait(_TICK_SECONDS)
                    continue

                try:
                    image = grabber.grab(settings.monitor_index)
                    frame = encode_frame(
                        image,
                        width=settings.frame_width,
                        quality=settings.jpeg_quality,
                        stamp=settings.stamp_timestamp,
                        captured_at=now,
                    )
                    errors = 0
                except Exception as error:  # noqa: BLE001 - worker boundary
                    errors += 1
                    logger.warning("Screen capture failed (%d): %s", errors, error)
                    if errors >= _MAX_CONSECUTIVE_ERRORS:
                        self.errorOccurred.emit(f"Screen capture failed: {error}")
                        return
                    self._stop.wait(1.0)
                    continue

                if immediate:
                    self.immediateFrameCaptured.emit(frame, token)
                else:
                    with self._lock:
                        self.scheduler.note_capture(now)
                    self.frameCaptured.emit(frame)
        finally:
            grabber.close()


__all__ = ["CaptureService"]
