"""Console rendering as an event subscriber.

The agent loop never prints. This module turns the event stream into the Rich
console output the agent used to emit inline, which keeps presentation swappable:
attach :class:`ConsoleRenderer` for a terminal, or write your own subscriber to
push the same events down an SSE channel.
"""

from __future__ import annotations

import json
from typing import Any

from chiron.core.events import (
    AgentEvent,
    CompactionEndEvent,
    CompactionStartEvent,
    MessageEndEvent,
    MessageStartEvent,
    MessageUpdateEvent,
    RetryEvent,
    ToolExecutionEndEvent,
    ToolExecutionStartEvent,
    ToolExecutionUpdateEvent,
)


def short_text(value: Any, limit: int = 200) -> str:
    """Truncate a value's string form for compact console logging."""
    text = value if isinstance(value, str) else json.dumps(value, default=str)
    return text if len(text) <= limit else text[: limit - 1] + "…"


class ConsoleRenderer:
    """Render agent events to a Rich console.

    Assistant text is printed as it streams; when a run is not streaming, the final
    assistant message is printed once instead. Tool activity and compaction are
    reported as dim status lines.

    Attributes:
        console: The Rich `Console` written to.
        show_tools (bool): Whether tool activity lines are printed.
        show_reasoning (bool): Whether streamed thinking is printed, dimmed, above the
            answer. Off by default — reasoning is usually noise in a transcript.
    """

    def __init__(
        self,
        console: Any = None,
        *,
        show_tools: bool = True,
        show_reasoning: bool = False,
    ) -> None:
        """Initialise the renderer, creating a default console when none is given."""
        if console is None:
            from rich.console import Console

            console = Console()
        self.console = console
        self.show_tools = show_tools
        self.show_reasoning = show_reasoning
        self._streamed = False
        self._reasoning_open = False

    def __call__(self, event: AgentEvent) -> None:
        """Render one event. Safe to register via `ReactAgent.subscribe`."""
        if isinstance(event, MessageStartEvent):
            if event.message.get("role") == "assistant":
                self._streamed = False
        elif isinstance(event, MessageUpdateEvent):
            self._render_update(event)
        elif isinstance(event, MessageEndEvent):
            self._render_message_end(event)
        elif isinstance(event, ToolExecutionStartEvent) and self.show_tools:
            self.console.print(
                f"[dim]→ {event.tool_name}({short_text(event.args)})[/dim]"
            )
        elif isinstance(event, ToolExecutionUpdateEvent) and self.show_tools:
            self.console.print(f"[dim]  … {short_text(event.output)}[/dim]")
        elif isinstance(event, ToolExecutionEndEvent) and self.show_tools:
            tag = "error" if event.is_error else "ok"
            self.console.print(f"[dim]  {tag}: {short_text(event.output)}[/dim]")
            if event.added_tool_names:
                self.console.print(
                    f"[dim]  + tools available: {', '.join(event.added_tool_names)}[/dim]"
                )
        elif isinstance(event, RetryEvent):
            self.console.print(
                f"[dim]· retrying {event.attempt}/{event.max_attempts} "
                f"in {event.delay_seconds:g}s ({short_text(event.reason, 80)})[/dim]"
            )
        elif isinstance(event, CompactionStartEvent):
            self.console.print(
                f"[dim]· compacting history (~{event.tokens_before} tokens)[/dim]"
            )
        elif isinstance(event, CompactionEndEvent):
            self.console.print(
                f"[dim]· compacted to ~{event.tokens_after} tokens[/dim]"
            )

    def _render_update(self, event: MessageUpdateEvent) -> None:
        """Print a streamed delta, keeping thinking visually distinct from the answer."""
        if event.reasoning_delta and self.show_reasoning:
            if not self._reasoning_open:
                self.console.print("[dim]thinking: [/dim]", end="")
                self._reasoning_open = True
            self.console.print(
                f"[dim]{event.reasoning_delta}[/dim]", end="", highlight=False
            )
        if event.delta:
            if self._reasoning_open:
                self.console.print()
                self._reasoning_open = False
            self._streamed = True
            self.console.print(event.delta, end="", markup=False, highlight=False)

    def _render_message_end(self, event: MessageEndEvent) -> None:
        """Close out a finished assistant message."""
        if event.message.get("role") != "assistant":
            return
        if self._reasoning_open:
            self.console.print()
            self._reasoning_open = False
        content = event.message.get("content")
        if self._streamed:
            self.console.print()  # terminate the streamed line
        elif content:
            self.console.print(content, markup=False, highlight=False)
        self._streamed = False


__all__ = ["ConsoleRenderer", "short_text"]
