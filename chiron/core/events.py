"""Typed events emitted by the agent loop.

The loop is UI-free: instead of printing, it emits events that any number of
subscribers consume — a Rich console renderer, an SSE feed, a session recorder, a
test assertion. Every event carries plain OpenAI-format message dicts, the same
shape the loop stores in `ReactAgent.messages`.

Event ordering for one run::

    agent_start
      turn_start
        message_start(assistant)  [message_update…]  message_end(assistant)
        tool_execution_start  tool_execution_end
        message_start(tool)  message_end(tool)
      turn_end
      …
    agent_end

`compaction_start` / `compaction_end` are emitted between turns when history
is summarised to fit the context window, and `retry` when a provider call fails
transiently and is about to be re-issued.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

Message = dict[str, Any]


class AgentStartEvent(BaseModel):
    """Emitted once, before the first turn of a run."""

    type: Literal["agent_start"] = "agent_start"


class AgentEndEvent(BaseModel):
    """Emitted once, after the run settles (including on cancellation)."""

    type: Literal["agent_end"] = "agent_end"
    messages: list[Message] = Field(default_factory=list)
    stop_reason: str = "completed"


class TurnStartEvent(BaseModel):
    """Emitted before each LLM call. `turn` is 1-based."""

    type: Literal["turn_start"] = "turn_start"
    turn: int


class TurnEndEvent(BaseModel):
    """Emitted after a turn's assistant message and all its tool results."""

    type: Literal["turn_end"] = "turn_end"
    turn: int
    message: Message
    tool_results: list[Message] = Field(default_factory=list)


class MessageStartEvent(BaseModel):
    """A message was added to history (or has begun streaming)."""

    type: Literal["message_start"] = "message_start"
    message: Message


class MessageUpdateEvent(BaseModel):
    """A streamed assistant delta. `message` is the partial message so far.

    Exactly one of `delta` / `reasoning_delta` is non-empty, so a UI can render
    visible output and the model's thinking in separate places.
    """

    type: Literal["message_update"] = "message_update"
    delta: str = ""
    reasoning_delta: str = ""
    message: Message


class MessageEndEvent(BaseModel):
    """A message is final and present in `ReactAgent.messages`."""

    type: Literal["message_end"] = "message_end"
    message: Message


class ToolExecutionStartEvent(BaseModel):
    """A tool call is about to execute (after approval, before `forward`)."""

    type: Literal["tool_execution_start"] = "tool_execution_start"
    tool_call_id: str
    tool_name: str
    args: dict[str, Any] = Field(default_factory=dict)


class ToolExecutionUpdateEvent(BaseModel):
    """Progress reported by a still-running tool via its `on_update` callback."""

    type: Literal["tool_execution_update"] = "tool_execution_update"
    tool_call_id: str
    tool_name: str
    output: str
    details: Any = None


class ToolExecutionEndEvent(BaseModel):
    """A tool call finished, was rejected at approval, or failed.

    `details` carries the tool's structured payload, which never reaches the model
    but is available to renderers and logs.
    """

    type: Literal["tool_execution_end"] = "tool_execution_end"
    tool_call_id: str
    tool_name: str
    output: str
    is_error: bool = False
    details: Any = None
    added_tool_names: list[str] = Field(default_factory=list)
    terminate: bool = False


class RetryEvent(BaseModel):
    """A provider call failed transiently and is about to be retried."""

    type: Literal["retry"] = "retry"
    attempt: int
    max_attempts: int
    delay_seconds: float
    reason: str


class CompactionStartEvent(BaseModel):
    """History exceeded the compaction threshold; summarisation is starting."""

    type: Literal["compaction_start"] = "compaction_start"
    tokens_before: int


class CompactionEndEvent(BaseModel):
    """History was replaced by `summary` plus the retained recent messages."""

    type: Literal["compaction_end"] = "compaction_end"
    tokens_before: int
    tokens_after: int
    summary: str


AgentEvent = (
    AgentStartEvent
    | AgentEndEvent
    | TurnStartEvent
    | TurnEndEvent
    | MessageStartEvent
    | MessageUpdateEvent
    | MessageEndEvent
    | ToolExecutionStartEvent
    | ToolExecutionUpdateEvent
    | ToolExecutionEndEvent
    | CompactionStartEvent
    | CompactionEndEvent
    | RetryEvent
)

__all__ = [
    "AgentEndEvent",
    "AgentEvent",
    "AgentStartEvent",
    "CompactionEndEvent",
    "CompactionStartEvent",
    "Message",
    "MessageEndEvent",
    "MessageStartEvent",
    "MessageUpdateEvent",
    "RetryEvent",
    "ToolExecutionEndEvent",
    "ToolExecutionStartEvent",
    "ToolExecutionUpdateEvent",
    "TurnEndEvent",
    "TurnStartEvent",
]
