"""Narrow contracts for Chiron's two permanent runtime agents.

The Observer and Responder deliberately do not share a provider interface. The
Observer owns either a Gemini Live socket or a buffered request pipeline and can
only write journal observations; the Responder owns user-visible answers and can
only read a journal snapshot.
``ChironApp`` constructs both, so changing a Responder model can never turn off
or replace visual observation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from chiron.capture.frames import Frame
from chiron.config.settings import Settings


@dataclass(frozen=True)
class ObserverStatus:
    """Responder-facing snapshot of visual freshness."""

    state: str
    watch_requested: bool
    mode: str = "live"
    accepting_frames: bool = True
    last_sampled_at: float | None = None
    last_observed_at: float | None = None
    pending_frames: int = 0
    next_process_at: float | None = None
    detail: str = ""

    @property
    def stale(self) -> bool:
        """Whether a question cannot be grounded in a current live span."""
        if not self.watch_requested or self.last_observed_at is None:
            return True
        if self.mode == "live":
            return not self.accepting_frames
        return not self.accepting_frames or self.pending_frames > 0


@runtime_checkable
class ObserverAgent(Protocol):
    """Write-only Observer surface used by the application."""

    status: str
    status_detail: str
    mode: str
    detected_game: str
    session_id: str
    accepting_frames: bool
    last_sampled_at: float | None
    last_observed_at: float | None
    pending_frames: int
    next_process_at: float | None

    @property
    def frames_sent(self) -> int: ...

    def start(self) -> None: ...

    async def stop(self, timeout: float = ...) -> None: ...

    def observe(self, frame: Frame, reason: str = ...) -> None: ...

    def reset_memory(self) -> None: ...

    def apply_settings(self, settings: Settings) -> None: ...


@runtime_checkable
class ResponderAgent(Protocol):
    """Question-answering surface; only this contract emits transcript output."""

    session_id: str
    detected_game: str

    def ask(
        self,
        text: str,
        frame: Frame | None,
        observer_status: ObserverStatus,
    ) -> None: ...

    def reset_memory(self) -> None: ...

    def apply_settings(self, settings: Settings) -> None: ...

    async def stop(self, timeout: float = ...) -> None: ...


__all__ = ["ObserverAgent", "ObserverStatus", "ResponderAgent"]
