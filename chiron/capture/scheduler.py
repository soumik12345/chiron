"""Clock-free fixed capture scheduling."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class FixedIntervalScheduler:
    """One deadline at a time, with no catch-up after missed ticks."""

    interval_seconds: float = 5.0
    last_capture: float | None = None

    def seconds_until_due(self, now: float) -> float:
        if self.last_capture is None:
            return 0.0
        return max(0.0, self.interval_seconds - (now - self.last_capture))

    def is_due(self, now: float) -> bool:
        return self.seconds_until_due(now) <= 0.0

    def note_capture(self, now: float) -> None:
        """Advance from the actual capture time, never by missed intervals."""
        self.last_capture = now

    def reset(self) -> None:
        """Make exactly one periodic capture due immediately."""
        self.last_capture = None


__all__ = ["FixedIntervalScheduler"]
