"""Cooperative cancellation for the agent loop.

A single token is created per run and threaded through the loop, the LLM stream,
and every tool call. Cancellation is **cooperative**: setting the flag stops the
agent at the next checkpoint (mid-stream, between tool calls, at a turn boundary)
rather than killing an in-flight coroutine, so message history is never left with
an assistant turn whose tool calls have no matching results.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class CancellationToken(Protocol):
    """Anything the loop can poll to decide whether it should stop."""

    def is_cancelled(self) -> bool:
        """Return True when the current run should stop as soon as possible."""
        ...


class SimpleCancellationToken:
    """The default in-process :class:`CancellationToken` implementation."""

    def __init__(self) -> None:
        self._cancelled = False

    def cancel(self) -> None:
        """Request cancellation. Idempotent."""
        self._cancelled = True

    def is_cancelled(self) -> bool:
        """Return True once :meth:`cancel` has been called."""
        return self._cancelled
