"""The adaptive shutter.

Continuous 1 fps is the API's ceiling and also a good way to spend a 32k context
window in six minutes. Most of the time nothing on screen is worth a frame, so
the scheduler idles at a slow keepalive rate and *bursts* to 1 fps for a short
window when something makes the next few seconds matter:

* the player asked a question, so the answer should be about the screen *now*;
* a pixel diff says the screen changed hard — a loading screen, a new area, a
  death screen — which are exactly the moments worth journaling.

That stretches effective visual memory three to five times versus streaming
continuously, and cuts token burn by the same factor.

The class is deliberately clock-free: every method takes the current time as an
argument. Nothing here sleeps, captures or touches Qt, so the policy can be
tested by handing it a sequence of timestamps.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

CaptureMode = Literal["baseline", "burst"]


@dataclass
class AdaptiveScheduler:
    """Decides when the next frame is due.

    Attributes:
        baseline_interval (float): Seconds between keepalive frames.
        burst_interval (float): Seconds between frames while bursting.
        burst_duration (float): How long one burst lasts.
        last_capture (float | None): When a frame was last taken, or None if
            none has been.
        burst_until (float): Timestamp the current burst expires at; 0.0 when
            not bursting.
        burst_reason (str): Why the current burst was requested, for the UI.
    """

    baseline_interval: float = 4.0
    burst_interval: float = 1.0
    burst_duration: float = 15.0
    last_capture: float | None = None
    burst_until: float = 0.0
    burst_reason: str = ""
    _pending: bool = field(default=False, repr=False)

    def request_burst(self, now: float, reason: str = "") -> None:
        """Start (or extend) a burst window ending `burst_duration` from `now`.

        The next frame is also marked due immediately: a burst is requested
        because *right now* matters, and waiting out the remainder of a four
        second keepalive gap would defeat the point.

        Args:
            now (float): Current time.
            reason (str): Short label shown in the overlay's status line.
        """
        self.burst_until = max(self.burst_until, now + self.burst_duration)
        self.burst_reason = reason or self.burst_reason
        self._pending = True

    def end_burst(self, now: float) -> None:
        """Drop back to baseline immediately."""
        self.burst_until = 0.0
        self.burst_reason = ""

    def mode(self, now: float) -> CaptureMode:
        """Whether the shutter is currently bursting or idling."""
        return "burst" if now < self.burst_until else "baseline"

    def interval(self, now: float) -> float:
        """The frame interval that applies at `now`."""
        return (
            self.burst_interval if self.mode(now) == "burst" else self.baseline_interval
        )

    def seconds_until_due(self, now: float) -> float:
        """How long until the next frame should be taken (0.0 when due now)."""
        if self._pending or self.last_capture is None:
            return 0.0
        elapsed = now - self.last_capture
        return max(0.0, self.interval(now) - elapsed)

    def is_due(self, now: float) -> bool:
        """Whether a frame should be taken at `now`."""
        return self.seconds_until_due(now) <= 0.0

    def note_capture(self, now: float) -> None:
        """Record that a frame was just taken."""
        self.last_capture = now
        self._pending = False

    def expire_burst(self, now: float) -> bool:
        """Clear a burst whose window has passed.

        Returns:
            bool: True when this call ended a burst, so a caller can emit a
                mode-changed notification exactly once.
        """
        if self.burst_until and now >= self.burst_until:
            self.burst_until = 0.0
            self.burst_reason = ""
            return True
        return False


__all__ = ["AdaptiveScheduler", "CaptureMode"]
