"""Narrow contracts for Chiron's two permanent runtime agents.

The Observer and Responder deliberately do not share a provider interface. The
Observer owns the Gemini Live socket and can only write journal observations;
the Responder owns user-visible answers and can only read a journal snapshot.
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
    last_observed_at: float | None = None
    detail: str = ""

    @property
    def stale(self) -> bool:
        """Whether a question cannot be grounded in a current live span."""
        return not (
            self.watch_requested
            and self.state == "live"
            and self.last_observed_at is not None
        )


@runtime_checkable
class ObserverAgent(Protocol):
    """Tool-only Live observer surface used by the application."""

    status: str
    status_detail: str
    detected_game: str
    session_id: str
    last_observed_at: float | None

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
