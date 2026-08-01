"""The capture worker: a background thread that turns the screen into frames.

Grabbing and JPEG-encoding a screen costs tens of milliseconds; doing it on the
Qt thread would show up as a stuttering overlay on exactly the machine that is
also running a game. So the shutter lives on its own thread, and finished frames
cross back as Qt signals — queued automatically, because the service object
itself belongs to the main thread.

The thread does three things in a loop: ask the :class:`~chiron.capture.scheduler.AdaptiveScheduler`
whether a frame is due, take one if it is, and compare it against the previous
frame's signature to decide whether the world just changed enough to be worth
bursting over. Settings can be replaced at any time from the UI thread; the loop
picks up the new snapshot on its next pass.
"""

from __future__ import annotations

import logging
import threading
import time

from PySide6.QtCore import QObject, Signal

from chiron.capture.frames import Frame, ScreenGrabber, encode_frame, signature_distance
from chiron.capture.scheduler import AdaptiveScheduler
from chiron.config.settings import CaptureSettings

logger = logging.getLogger(__name__)

#: How long the loop sleeps between "is a frame due yet?" checks. Short enough
#: that a burst requested by a question starts within a blink.
_TICK_SECONDS = 0.05

#: Consecutive grab failures tolerated before the service gives up and reports.
_MAX_CONSECUTIVE_ERRORS = 5


class CaptureService(QObject):
    """Adaptive screen capture on a worker thread.

    Signals:
        frameCaptured (object): A finished :class:`~chiron.capture.frames.Frame`.
        sceneChanged (float): A hard scene change was detected, with its
            normalised difference score.
        modeChanged (str, str): The shutter switched between ``baseline`` and
            ``burst``, with the reason for a burst (empty at baseline).
        errorOccurred (str): Capture failed and the thread stopped.

    Attributes:
        settings (CaptureSettings): The snapshot the loop is currently using.
        scheduler (AdaptiveScheduler): The shutter policy.
    """

    frameCaptured = Signal(object)
    sceneChanged = Signal(float)
    modeChanged = Signal(str, str)
    errorOccurred = Signal(str)

    def __init__(
        self, settings: CaptureSettings, parent: QObject | None = None
    ) -> None:
        """Create a stopped, not-yet-watching service configured by `settings`."""
        super().__init__(parent)
        self._lock = threading.Lock()
        self._settings = settings
        self._stop = threading.Event()
        # Running and watching are different things. The thread may be alive and
        # idle; only `_watching` decides whether the screen is ever read, and it
        # starts false so launching Chiron never captures anything by itself.
        self._watching = threading.Event()
        self._thread: threading.Thread | None = None
        self._last_signature: bytes = b""
        self._last_mode: str = "baseline"
        self.scheduler = AdaptiveScheduler(
            baseline_interval=settings.baseline_interval_seconds,
            burst_interval=settings.burst_interval_seconds,
            burst_duration=settings.burst_duration_seconds,
        )

    @property
    def settings(self) -> CaptureSettings:
        """The capture settings currently in force."""
        with self._lock:
            return self._settings

    @property
    def is_running(self) -> bool:
        """Whether the worker thread is alive."""
        return self._thread is not None and self._thread.is_alive()

    @property
    def is_watching(self) -> bool:
        """Whether the screen is actually being read."""
        return self._watching.is_set()

    def set_watching(self, watching: bool) -> None:
        """Start or stop reading the screen.

        The thread keeps running either way, so toggling costs nothing and the
        next frame after a start arrives immediately rather than after a full
        baseline interval.

        Args:
            watching (bool): True to capture, False to go idle.
        """
        if watching == self.is_watching:
            return
        if watching:
            with self._lock:
                # Forget the last frame, so the first frame after a pause is not
                # diffed against whatever was on screen before it — that stale
                # comparison would read as a scene change every single time.
                self._last_signature = b""
                self.scheduler.last_capture = None
                self.scheduler.end_burst(time.time())
            self._watching.set()
        else:
            self._watching.clear()
        logger.info("Capture %s", "started" if watching else "stopped")

    def start(self) -> None:
        """Start the capture thread if it is not already running."""
        if self.is_running:
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run, name="chiron-capture", daemon=True
        )
        self._thread.start()
        logger.info("Capture thread started")

    def stop(self, timeout: float = 2.0) -> None:
        """Ask the thread to finish and wait briefly for it.

        Args:
            timeout (float): Seconds to wait for the thread to exit.
        """
        self._stop.set()
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=timeout)
        self._thread = None

    def apply_settings(self, settings: CaptureSettings) -> None:
        """Replace the settings the loop reads on its next pass.

        Args:
            settings (CaptureSettings): The new configuration. Scheduler timings
                are updated in place so an in-flight burst is preserved.
        """
        with self._lock:
            self._settings = settings
            self.scheduler.baseline_interval = settings.baseline_interval_seconds
            self.scheduler.burst_interval = settings.burst_interval_seconds
            self.scheduler.burst_duration = settings.burst_duration_seconds

    def request_burst(self, reason: str = "") -> None:
        """Burst the shutter to 1 fps for the configured window.

        Args:
            reason (str): Short label ("question", "scene change") surfaced in
                the overlay status line.
        """
        with self._lock:
            self.scheduler.request_burst(time.time(), reason)
        self._emit_mode()

    def _emit_mode(self) -> None:
        """Emit :attr:`modeChanged` when the shutter mode actually changed."""
        with self._lock:
            mode = self.scheduler.mode(time.time())
            reason = self.scheduler.burst_reason
        if mode != self._last_mode:
            self._last_mode = mode
            self.modeChanged.emit(mode, reason)

    def _run(self) -> None:
        """Worker loop: capture when due, diff, encode, emit."""
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
                    self.scheduler.expire_burst(now)
                    due = self.scheduler.is_due(now)
                    if due:
                        self.scheduler.note_capture(now)
                self._emit_mode()

                if not due:
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
                except Exception as error:
                    errors += 1
                    logger.warning("Screen capture failed (%d): %s", errors, error)
                    if errors >= _MAX_CONSECUTIVE_ERRORS:
                        self.errorOccurred.emit(f"Screen capture failed: {error}")
                        return
                    self._stop.wait(1.0)
                    continue

                self._handle_frame(frame, settings)
        finally:
            grabber.close()
            logger.info("Capture thread stopped")

    def _handle_frame(self, frame: Frame, settings: CaptureSettings) -> None:
        """Emit a captured frame and burst if the scene changed hard."""
        if settings.scene_change_enabled and self._last_signature:
            distance = signature_distance(self._last_signature, frame.signature)
            if distance >= settings.scene_change_threshold:
                self.sceneChanged.emit(distance)
                with self._lock:
                    self.scheduler.request_burst(frame.captured_at, "scene change")
                self._emit_mode()
        self._last_signature = frame.signature
        self.frameCaptured.emit(frame)


__all__ = ["CaptureService"]
