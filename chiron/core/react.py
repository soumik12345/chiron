"""A stateful ReAct agent over chiron's :class:`LiteLLMModel`.

The agent drives a native tool-calling loop: per turn it calls the model with the
registered tool schemas (`tool_choice="auto"`), executes any tool calls, feeds the
results back, and repeats. **A turn ends when the model replies with no tool calls**
— `final_answer` is an optional convenience tool, not a requirement.

That rule cannot, on its own, distinguish a model that has finished from one that has
merely gone quiet, so two guards stand in front of it: an empty reply is retried
rather than obeyed (see :func:`_is_empty_reply`), and an optional `completion_guard`
can push a settling run onward when the work it exists to do is not done. Both are
bounded, and a run stopped by either ends with a stop reason that says so rather than
reporting success.

The loop itself emits nothing to a terminal. It yields typed
:mod:`~chiron.core.events` that any number of subscribers consume, which is what
lets the same agent drive a Rich console, an SSE stream, and a session recorder at
once. :class:`~chiron.core.rendering.ConsoleRenderer` is the built-in subscriber
that reproduces terminal output.

Capabilities:

* **Stateful history** — `messages` persists across calls, so `prompt` /
  `continue_` build a real conversation. `reset` starts over.
* **Cancellation** — `cancel()` stops the run at the next checkpoint (mid-stream,
  between tool calls, at a turn boundary) and repairs history by answering every
  orphaned tool call, so the transcript stays valid for the next request.
* **Steering & follow-ups** — `steer()` injects a message at the next turn
  boundary of a *running* agent; `follow_up()` queues work for after it settles.
* **Automatic compaction** — when the estimated context crosses the model's
  threshold, older history is replaced by a structured summary.
* **Durable sessions** — with a :class:`~chiron.core.session.JsonlSessionStore`,
  every message is appended to disk and a session can be resumed or branched.
* **Max-turn guard**, **LLM-call retries** with backoff, and **optional per-tool
  approval**.
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections import deque
from collections.abc import AsyncIterator, Awaitable, Sequence
from dataclasses import dataclass, field
from inspect import isawaitable
from typing import Any, Callable, Literal

from pydantic import BaseModel

from chiron.core.cancellation import CancellationToken, SimpleCancellationToken
from chiron.core.context import (
    ContextCompactor,
    ContextUsageEstimate,
    estimate_context_tokens,
    estimate_context_usage,
)
from chiron.core.events import (
    AgentEndEvent,
    AgentEvent,
    AgentStartEvent,
    CompactionEndEvent,
    CompactionStartEvent,
    MessageEndEvent,
    MessageStartEvent,
    MessageUpdateEvent,
    RetryEvent,
    ToolExecutionEndEvent,
    ToolExecutionStartEvent,
    ToolExecutionUpdateEvent,
    TurnEndEvent,
    TurnStartEvent,
)
from chiron.core.prompts import SYSTEM_PROMPT
from chiron.core.rendering import short_text
from chiron.core.results import ToolResult, image_followup_message
from chiron.core.router import ToolRouter
from chiron.core.session import JsonlSessionStore
from chiron.core.tool import Tool
from chiron.models.litellm_model import LiteLLMModel, extract_reasoning
from chiron.models.usage import call_timer, read_provider_field, utc_now_iso

logger = logging.getLogger(__name__)

_MAX_LLM_RETRIES = 3
_RETRY_DELAYS = [5, 15, 30]
_RATE_LIMIT_DELAYS = [30, 60]

INTERRUPTED_TOOL_RESULT = "Tool call interrupted by user"

EventListener = Callable[[AgentEvent], "Awaitable[None] | None"]
QueueMode = Literal["one_at_a_time", "all"]


@dataclass(frozen=True)
class ToolInvocation:
    """One tool call as the loop is about to run it.

    Attributes:
        tool_call_id (str): The provider's id for this call.
        name (str): The tool being invoked.
        arguments (dict): The parsed arguments.
    """

    tool_call_id: str
    name: str
    arguments: dict[str, Any]


# A gate: return `(blocked, reason)`. Blocking substitutes `reason` as a failed
# tool result, so the model learns why rather than silently losing the call.
BeforeToolCall = Callable[
    [ToolInvocation], "Awaitable[tuple[bool, str | None]] | tuple[bool, str | None]"
]
# A rewriter: return the result to use in place of what the tool produced.
AfterToolCall = Callable[
    [ToolInvocation, ToolResult], "Awaitable[ToolResult] | ToolResult"
]
# A completion check: return None when the run may settle, or a sentence explaining
# what is still missing, which is injected as a user message so the run continues.
CompletionGuard = Callable[[], "Awaitable[str | None] | str | None"]


class ReactAgentResult(BaseModel):
    """The outcome of one `run` / `prompt` call.

    Attributes:
        final_answer (str | None): The last assistant reply that carried no tool
            calls. None when the run was cancelled, hit the turn limit, or the model
            returned empty content.
        completed (bool): Whether the run ended because the model was done. True only
            for `completed` and `tool_terminated` — every other stop reason is a
            run that ended with work outstanding.
        stop_reason (str): `completed`, `tool_terminated`, `max_iterations`,
            `cancelled`, `length` (the provider truncated the reply),
            `content_filter`, `empty_response` (the provider kept returning a
            reply with neither content nor tool calls), or `incomplete` (the
            completion guard still had outstanding work when the nudge budget ran
            out). The last two exist so that a run which stopped short can no longer
            be mistaken for one that finished.
        steps (int): How many model turns were executed.
        messages (list): Full history after the run.
        usage (dict): The model's cumulative token counters.
        cost_usd (float): The model's cumulative spend.
    """

    final_answer: str | None = None
    completed: bool = False
    stop_reason: str = "completed"
    steps: int = 0
    messages: list = []
    usage: dict = {}
    cost_usd: float = 0.0


@dataclass(frozen=True)
class QueuedMessages:
    """A snapshot of the steering and follow-up queues.

    Attributes:
        steering (tuple[dict, ...]): Messages that will be injected at the next turn
            boundary of the running agent.
        follow_up (tuple[dict, ...]): Messages that will start a new turn once the
            agent would otherwise have settled.
    """

    steering: tuple[dict[str, Any], ...] = ()
    follow_up: tuple[dict[str, Any], ...] = ()

    @property
    def count(self) -> int:
        """Total number of queued messages."""
        return len(self.steering) + len(self.follow_up)


@dataclass
class _RunState:
    """Mutable bookkeeping shared between the loop body and its caller."""

    turn: int = 0
    stop_reason: str = "completed"
    final_answer: str | None = None
    completed: bool = False
    #: Empty replies retried since the last turn that actually did something.
    empty_replies: int = 0
    #: Nudges the completion guard has spent this run.
    nudges_spent: int = 0
    #: The guard wanted to push the run onward but had no nudges left.
    guard_unsatisfied: bool = False


def _status_code(error: Exception) -> int | None:
    """Return the HTTP status carried by a provider exception, if any."""
    for attribute in ("status_code", "code", "http_status"):
        value = getattr(error, attribute, None)
        if isinstance(value, int):
            return value
        if isinstance(value, str) and value.isdigit():
            return int(value)
    return None


def _litellm_exception_classes() -> tuple[tuple[type, ...], tuple[type, ...]]:
    """Return `(rate_limit_types, transient_types)` from litellm, if importable.

    Classifying on exception type is far more reliable than matching substrings of
    `str(error)`, which misfires on messages that merely quote a status code.
    The substring heuristic remains as a fallback for wrapped or bare exceptions.
    """
    try:
        import litellm
    except ImportError:  # pragma: no cover - litellm is a hard dependency
        return (), ()

    def pick(*names: str) -> tuple[type, ...]:
        found = [getattr(litellm, name, None) for name in names]
        return tuple(c for c in found if isinstance(c, type))

    rate_limit = pick("RateLimitError")
    transient = pick(
        "Timeout",
        "APIConnectionError",
        "APIError",
        "InternalServerError",
        "ServiceUnavailableError",
    )
    # Bare transport failures reach us unwrapped when the stream dies mid-flight.
    return rate_limit, transient + (TimeoutError, ConnectionError)


_RATE_LIMIT_TYPES, _TRANSIENT_TYPES = _litellm_exception_classes()

_TRANSIENT_PATTERNS = (
    "timeout",
    "timed out",
    "503",
    "service unavailable",
    "502",
    "bad gateway",
    "500",
    "internal server error",
    "overloaded",
    "capacity",
    "connection reset",
    "connection refused",
    "connection error",
    "eof",
    "broken pipe",
)


def _is_rate_limit_error(error: Exception) -> bool:
    """Return True when the error is an API rate-limit response."""
    if _RATE_LIMIT_TYPES and isinstance(error, _RATE_LIMIT_TYPES):
        return True
    if _status_code(error) == 429:
        return True
    s = str(error).lower()
    return any(
        p in s
        for p in ("429", "rate limit", "rate_limit", "too many requests", "throttl")
    )


def _is_transient_error(error: Exception) -> bool:
    """Return True when the error is transient and safe to retry (5xx/timeout/conn).

    Client errors (4xx other than 429 and 408) are never retried: a malformed request
    or a bad key will fail identically on every attempt.

    The **type** is consulted before the status code, because the two disagree on the
    single most common retryable failure in a long run: `litellm.Timeout` carries
    `status_code=408`, which the 5xx rule would read as a non-retryable client error
    and give up on. None of litellm's 4xx classes derive from the transient ones, so
    checking types first cannot make a genuine client error look retryable.
    """
    if _is_rate_limit_error(error):
        return True
    if _TRANSIENT_TYPES and isinstance(error, _TRANSIENT_TYPES):
        return True
    status = _status_code(error)
    if status is not None:
        # 408 Request Timeout and 409 Conflict are the two 4xx the provider is asking
        # you to try again on; everything else in that range is the caller's fault.
        return status >= 500 or status in (408, 409)
    s = str(error).lower()
    return any(p in s for p in _TRANSIENT_PATTERNS)


def _retry_delay_for(error: Exception, attempt: int) -> int | None:
    """Return the backoff delay (seconds) for an error/attempt, or None if not retryable.

    Args:
        error (Exception): The exception that triggered a potential retry.
        attempt (int): Zero-based attempt index (0 = first failure).

    Returns:
        int | None: Seconds to wait before retrying, or None if the error should not
            be retried (or the schedule is exhausted).
    """
    schedule = (
        _RATE_LIMIT_DELAYS
        if _is_rate_limit_error(error)
        else (_RETRY_DELAYS if _is_transient_error(error) else None)
    )
    if schedule is None or attempt >= len(schedule):
        return None
    return schedule[attempt]


# Provider finish reasons that mean the turn did not end on the model's own terms.
# "stop" is normal completion; "tool_calls"/"function_call" mean the loop continues.
_ABNORMAL_FINISH_REASONS = {
    "length": "length",
    "max_tokens": "length",
    "content_filter": "content_filter",
}


def _abnormal_stop_reason(finish_reason: str | None) -> str | None:
    """Map a provider finish reason to a stop reason, or None when it was normal."""
    return _ABNORMAL_FINISH_REASONS.get((finish_reason or "").lower())


def _last_tool_text(tool_results: list[dict[str, Any]]) -> str | None:
    """Return the text of the last tool result, used when a tool ends the run."""
    return tool_results[-1].get("content") if tool_results else None


#: How each stop reason reads to someone who is not holding this file open. Used by
#: the agents when a run ends without the artifact they exist to produce — where the
#: stop reason is the whole explanation, and `stop_reason=incomplete` explains
#: nothing to the person whose book didn't get a moodboard.
_STOP_REASON_PHRASES: dict[str, str] = {
    "completed": "the model ended the run without submitting anything",
    "empty_response": (
        "the model went quiet — it returned an empty reply several times running"
    ),
    "incomplete": (
        "the model stopped early and didn't pick the work back up when asked to"
    ),
    "max_iterations": "the run hit its turn limit",
    "length": "the model's reply was cut off by its output limit",
    "content_filter": "the provider's content filter stopped the reply",
    "cancelled": "the run was cancelled",
}


def describe_stop_reason(stop_reason: str) -> str:
    """Render a stop reason as a phrase, falling back to the raw value."""
    return _STOP_REASON_PHRASES.get(stop_reason, f"stop_reason={stop_reason}")


def _is_empty_reply(message: dict[str, Any]) -> bool:
    """True when the provider returned neither content nor tool calls.

    **An empty reply is not a decision.** The loop's normal termination rule — a turn
    ends when the model returns no tool calls — cannot by itself tell "the model is
    finished" apart from "the provider returned nothing", and the second happens: an
    empty candidate comes back with `finish_reason: "stop"`, no usage, and no
    diagnostics. Treating that as a finished turn silently ends a run mid-task, which
    is worst for an agent whose output is an artifact rather than a closing message.

    Deliberately counts a reasoning-only reply as empty too: thinking with nothing
    acted on leaves the run exactly where it was.
    """
    if message.get("tool_calls"):
        return False
    return not str(message.get("content") or "").strip()


# --------------------------------------------------------------------------- #
# Message helpers + normalised LLM result
# --------------------------------------------------------------------------- #
def _user_message(content: Any) -> dict[str, Any]:
    """Build a user message."""
    return {"role": "user", "content": content}


def _assistant_message(
    content: str | None,
    tool_calls: list[dict] | None,
    *,
    reasoning_content: str | None = None,
    thinking_blocks: list | None = None,
) -> dict[str, Any]:
    """Build an assistant message, attaching optional parts only when present.

    `thinking_blocks` are Anthropic's signed reasoning blocks. They are stored on
    the message and replayed verbatim, because a conversation that drops them is
    rejected once extended thinking is in play.
    """
    msg: dict[str, Any] = {"role": "assistant", "content": content}
    if tool_calls:
        msg["tool_calls"] = tool_calls
    if reasoning_content:
        msg["reasoning_content"] = reasoning_content
    if thinking_blocks:
        msg["thinking_blocks"] = thinking_blocks
    return msg


def _tool_message(
    content: str, tool_call_id: str, name: str, *, is_error: bool = False
) -> dict[str, Any]:
    """Build a `role: tool` result message for one executed tool call.

    `is_error` is loop-only bookkeeping: it stays in durable history and on events,
    and is stripped before the message reaches the provider (Chat Completions has no
    field for it).
    """
    message: dict[str, Any] = {
        "role": "tool",
        "content": content,
        "tool_call_id": tool_call_id,
        "name": name,
    }
    if is_error:
        message["is_error"] = True
    return message


@dataclass
class LLMResult:
    """Normalised result of one LLM call (streaming or non-streaming).

    Attributes:
        content (str | None): Assistant text, or None if only tool calls were emitted.
        tool_calls_acc (dict[int, dict]): Tool calls keyed by their index, each with
            `id`, `type`, and `function` (`name` + `arguments`) sub-keys.
        finish_reason (str | None): Model's stop reason, or None if not reported.
        reasoning_content (str | None): The model's thinking text, when reported.
        thinking_blocks (list | None): Provider-signed thinking blocks to replay.
        usage (dict): Per-call usage slice as returned by `model.record_usage`.
    """

    content: str | None
    tool_calls_acc: dict[int, dict]
    finish_reason: str | None
    reasoning_content: str | None = None
    thinking_blocks: list | None = None
    usage: dict = field(default_factory=dict)

    @property
    def tool_calls(self) -> list[dict]:
        """Tool calls in index order."""
        return [self.tool_calls_acc[i] for i in sorted(self.tool_calls_acc)]

    def to_message(self) -> dict[str, Any]:
        """Build the assistant message this result represents."""
        return _assistant_message(
            self.content,
            self.tool_calls or None,
            reasoning_content=self.reasoning_content,
            thinking_blocks=self.thinking_blocks,
        )


class ReactAgent:
    """A stateful ReAct agent over chiron's async :class:`LiteLLMModel`.

    Attributes:
        model (LiteLLMModel): The LLM wrapper used for every completion.
        tool_router (ToolRouter): Registry/dispatcher for the agent's tools.
        system_prompt (str): The base system prompt (plus any `instructions`).
        messages (list[dict]): Live conversation history, system message first.
        max_iterations (int): Turn ceiling per run (`-1` or None = unbounded).
        yolo_mode (bool): When True, tools requiring approval are auto-approved.
        session (JsonlSessionStore | None): Durable append-only session storage.
        compactor (ContextCompactor | None): Automatic history compaction.
        last_result (ReactAgentResult | None): Outcome of the most recent run.
    """

    def __init__(
        self,
        tools: list[Tool],
        *,
        model: LiteLLMModel | None = None,
        model_id: str = "openrouter/openai/gpt-4o-mini",
        system_prompt: str = SYSTEM_PROMPT,
        instructions: str | None = None,
        temperature: float | None = 0.7,
        max_tokens: int | None = None,
        max_iterations: int | None = 25,
        yolo_mode: bool = True,
        enable_prompt_caching: bool = True,
        approval_callback: Callable[[str, dict], bool] | None = None,
        weave_project: str | None = None,
        session: JsonlSessionStore | None = None,
        auto_compact: bool = True,
        compactor: ContextCompactor | None = None,
        context_window_tokens: int | None = None,
        queue_mode: QueueMode = "one_at_a_time",
        deferred_tools: list[Tool] | None = None,
        before_tool_call: BeforeToolCall | None = None,
        after_tool_call: AfterToolCall | None = None,
        preserve_reasoning: bool = True,
        completion_guard: CompletionGuard | None = None,
        max_completion_nudges: int = 2,
        max_empty_replies: int = 2,
        first_tool_choice: str | dict[str, Any] | None = None,
        usage_kind: str = "turn",
    ) -> None:
        """Initialise the agent with its tool set and configuration.

        Args:
            tools (list[Tool]): The tools made available to the agent.
            model (LiteLLMModel | None): LLM wrapper to use. Built from `model_id` and
                the sampling args when omitted.
            model_id (str): litellm model id used when `model` is not supplied.
            system_prompt (str): Base system prompt. Defaults to `SYSTEM_PROMPT`.
            instructions (str | None): Extra instructions appended to the system prompt.
            temperature (float): Sampling temperature (when building the model).
            max_tokens (int | None): Completion token cap (when building the model).
            max_iterations (int | None): Turn ceiling per run. `-1` or None disables it.
            yolo_mode (bool): Auto-approve tools that declare `requires_approval`.
            enable_prompt_caching (bool): Pass-through to the model wrapper.
            approval_callback (Callable[[str, dict], bool] | None): Called as
                `(tool_name, arguments) -> bool` to resolve approval when a tool
                requires it and auto-approval is off.
            weave_project (str | None): When set, initialise W&B Weave tracing.
            session (JsonlSessionStore | None): Durable session storage. When it
                already holds messages they become this agent's starting history.
            auto_compact (bool): Summarise older history when it nears the context
                window. Defaults to True.
            compactor (ContextCompactor | None): Custom compaction policy. Built from
                the model when omitted and `auto_compact` is on.
            context_window_tokens (int | None): Override the model's context window
                for compaction purposes.
            queue_mode (QueueMode): Whether a turn boundary drains one queued message
                or all of them.
            deferred_tools (list[Tool] | None): Tools registered but hidden from the
                model until a tool result activates them via `added_tool_names`.
            before_tool_call (BeforeToolCall | None): Gate called as
                `(invocation) -> (blocked, reason)` before every tool call. May be
                sync or async. Runs after `approval_callback`.
            after_tool_call (AfterToolCall | None): Rewriter called as
                `(invocation, result) -> result` after every tool call, including
                blocked and failed ones. May be sync or async.
            preserve_reasoning (bool): Keep reasoning/thinking blocks in history and
                replay the signed blocks to the provider. Defaults to True; set False
                to strip them (smaller context, but Anthropic extended thinking then
                cannot be continued across turns).
            completion_guard (CompletionGuard | None): Consulted when a turn ends with
                no tool calls — the loop's implicit exit. Return None to let the run
                settle, or a sentence naming what is still missing, which is injected
                as a user message so the run continues instead of ending. This is what
                closes the gap for an agent whose deliverable is a **submitted
                artifact** rather than a closing message: for those, "the model stopped
                talking" is not evidence of success, and the right answer is to say so
                and carry on rather than to fail the run. May be sync or async.
            max_completion_nudges (int): How many times `completion_guard` may push a
                settling run onward before it is allowed to stop as `incomplete`.
                Bounded explicitly rather than left to `max_iterations`, because each
                nudge replays the whole transcript and a model that has decided it is
                finished rarely changes its mind on the fifth telling.
            max_empty_replies (int): How many times a reply carrying neither content
                nor tool calls is retried before the run stops as `empty_response`.
                The retry is nearly free — an empty candidate is billed nothing and
                nothing was executed — so this defends the common transient case
                without the loop being able to spin on it.
        """
        self.model = model or LiteLLMModel(
            model_id=model_id,
            temperature=temperature,
            max_tokens=max_tokens,
            enable_prompt_caching=enable_prompt_caching,
        )
        self.tools = list(tools)
        self.deferred_tools = list(deferred_tools or [])
        self.tool_router = ToolRouter(self.tools, self.deferred_tools)
        self.system_prompt = system_prompt + (
            f"\n\n{instructions}" if instructions else ""
        )
        self.max_iterations = max_iterations
        self.yolo_mode = yolo_mode
        self.approval_callback = approval_callback
        self.before_tool_call = before_tool_call
        self.after_tool_call = after_tool_call
        self.preserve_reasoning = preserve_reasoning
        self.completion_guard = completion_guard
        self.max_completion_nudges = max_completion_nudges
        self.max_empty_replies = max_empty_replies
        self.first_tool_choice = first_tool_choice
        self.usage_kind = usage_kind
        self.queue_mode: QueueMode = queue_mode
        self.session = session
        self.auto_compact = auto_compact
        self.compactor = compactor
        if self.compactor is None and auto_compact:
            self.compactor = ContextCompactor(
                self.model, context_window_tokens=context_window_tokens
            )
        self.last_result: ReactAgentResult | None = None

        self.messages: list[dict[str, Any]] = []
        self._listeners: list[EventListener] = []
        self._steering: deque[dict[str, Any]] = deque()
        self._follow_up: deque[dict[str, Any]] = deque()
        self._signal: SimpleCancellationToken | None = None
        self._running = False

        self._restore_or_seed()
        self._maybe_init_weave(weave_project)

    @staticmethod
    def _maybe_init_weave(weave_project: str | None) -> None:
        """Initialise W&B Weave tracing if a project name is given (best-effort)."""
        if not weave_project:
            return
        try:
            import weave

            weave.init(weave_project)
        except Exception as e:  # noqa: BLE001
            logger.warning("weave.init failed (continuing without tracing): %s", e)

    def _restore_or_seed(self) -> None:
        """Adopt an existing session's history, or start a fresh one."""
        if self.session is not None:
            state = self.session.state()
            if state.messages:
                self.messages = list(state.messages)
                return
            self.session.append_info(model_id=getattr(self.model, "model_id", None))
        self._append({"role": "system", "content": self.system_prompt})

    # ----------------------------------------------------------------------- #
    # Subscriptions, queues, and control
    # ----------------------------------------------------------------------- #
    def subscribe(self, listener: EventListener) -> Callable[[], None]:
        """Register an event listener and return a function that unsubscribes it.

        The listener may be sync or async and is called for every event, in
        registration order, before the event is yielded to the run's consumer.
        """
        self._listeners.append(listener)

        def unsubscribe() -> None:
            if listener in self._listeners:
                self._listeners.remove(listener)

        return unsubscribe

    @property
    def is_running(self) -> bool:
        """Whether a run is currently in progress."""
        return self._running

    @property
    def queued_messages(self) -> QueuedMessages:
        """A snapshot of both queues."""
        return QueuedMessages(tuple(self._steering), tuple(self._follow_up))

    def cancel(self) -> None:
        """Request cancellation of the running agent (no-op when idle).

        The run stops at its next checkpoint and history is repaired so every
        outstanding tool call receives a result.
        """
        if self._signal is not None:
            self._signal.cancel()

    def steer(self, content: str) -> QueuedMessages:
        """Queue a user message to be injected at the next turn boundary."""
        return self.steer_message(_user_message(content))

    def steer_message(self, message: dict[str, Any]) -> QueuedMessages:
        """Queue a raw message to be injected at the next turn boundary."""
        self._steering.append(message)
        return self.queued_messages

    def follow_up(self, content: str) -> QueuedMessages:
        """Queue a user message to run once the agent would otherwise settle."""
        return self.follow_up_message(_user_message(content))

    def follow_up_message(self, message: dict[str, Any]) -> QueuedMessages:
        """Queue a raw message to run once the agent would otherwise settle."""
        self._follow_up.append(message)
        return self.queued_messages

    def clear_queues(self) -> QueuedMessages:
        """Drop everything from both queues and return what was discarded."""
        snapshot = self.queued_messages
        self._steering.clear()
        self._follow_up.clear()
        return snapshot

    def reset(self) -> None:
        """Clear history back to the system message and drop queued messages.

        With a session store this starts a **new root** in the same file rather than
        appending to the old conversation, so replaying the session reproduces exactly
        what the agent now holds. Nothing is deleted — the previous conversation
        remains as its own branch. Use :meth:`branch` to rewind instead of starting over.
        """
        self._ensure_not_running()
        self.clear_queues()
        self.messages = []
        if self.session is not None:
            self.session.active_leaf_id = None
        self._append({"role": "system", "content": self.system_prompt})

    def branch(self, entry_id: str) -> None:
        """Rewind to a point in the session tree and continue from there.

        Args:
            entry_id (str): The session entry to continue from.

        Raises:
            RuntimeError: If no session store is configured, or a run is in progress.
        """
        self._ensure_not_running()
        if self.session is None:
            raise RuntimeError("branch() requires a session store")
        state = self.session.branch(entry_id)
        self.messages = list(state.messages)

    def context_usage(self) -> ContextUsageEstimate:
        """Return the estimated context size of the next request."""
        return estimate_context_usage(self.messages, self._tool_specs())

    # ----------------------------------------------------------------------- #
    # Public run API
    # ----------------------------------------------------------------------- #
    def stream_events(
        self,
        prompt: str | list[dict[str, Any]] | None = None,
        *,
        auto_approve: bool | None = None,
        provider_stream: bool = False,
    ) -> AsyncIterator[AgentEvent]:
        """Run the agent and yield its events as they happen.

        This is the primitive the other entry points are built on. `is_running`
        flips before the first event, so a caller can hand the agent to a UI and
        immediately `steer` or `cancel` it.

        Args:
            prompt (str | None): A user message to start with. None continues the
                existing conversation without adding anything.
            auto_approve (bool | None): Override `yolo_mode` for this run.
            provider_stream (bool): Request a streaming completion, which produces
                `message_update` events for assistant text deltas.

        Returns:
            AsyncIterator[AgentEvent]: The run's event stream.

        Raises:
            RuntimeError: If a run is already in progress.
        """
        self._ensure_not_running()
        self._append_interrupted_tool_results()
        self._running = True
        prompts = [_user_message(prompt)] if prompt is not None else []
        return self._run(
            prompts=prompts,
            auto_approve=auto_approve,
            provider_stream=provider_stream,
        )

    async def run(
        self,
        prompt: str | list[dict[str, Any]] | None = None,
        *,
        stream: bool = False,
        auto_approve: bool | None = None,
        console: Any = None,
    ) -> ReactAgentResult:
        """Run one task to completion and return the result.

        History is preserved across calls, so calling `run` twice continues the same
        conversation. Use :meth:`reset` to start fresh.

        Args:
            prompt (str | None): The task/question for the agent. None continues from
                the existing history (equivalent to :meth:`continue_`).
            stream (bool): Request a streaming completion and print assistant text
                deltas plus tool activity to a Rich console as the agent works.
            auto_approve (bool | None): Override `yolo_mode` for this run. None uses
                the agent's `yolo_mode`.
            console (Any): Optional Rich `Console` for rendered output. Supplying
                one attaches the console renderer even when `stream` is False.

        Returns:
            ReactAgentResult: The final answer, stop reason, step count, full message
                history, and cumulative usage/cost.
        """
        unsubscribe: Callable[[], None] | None = None
        if stream or console is not None:
            from chiron.core.rendering import ConsoleRenderer

            unsubscribe = self.subscribe(ConsoleRenderer(console))
        try:
            async for _ in self.stream_events(
                prompt, auto_approve=auto_approve, provider_stream=stream
            ):
                pass
        finally:
            if unsubscribe is not None:
                unsubscribe()
        assert self.last_result is not None  # set in _run's finally
        return self.last_result

    async def continue_(self, **kwargs: Any) -> ReactAgentResult:
        """Resume the existing conversation without adding a new user message."""
        return await self.run(None, **kwargs)

    def run_sync(self, prompt: str | None = None, **kwargs: Any) -> ReactAgentResult:
        """Blocking convenience wrapper around :meth:`run`."""
        return asyncio.run(self.run(prompt, **kwargs))

    # ----------------------------------------------------------------------- #
    # Run lifecycle
    # ----------------------------------------------------------------------- #
    async def _run(
        self,
        *,
        prompts: Sequence[dict[str, Any]],
        auto_approve: bool | None,
        provider_stream: bool,
    ) -> AsyncIterator[AgentEvent]:
        """Own the run's lifecycle: cancellation token, notification, teardown."""
        signal = SimpleCancellationToken()
        self._signal = signal
        state = _RunState()
        try:
            async for event in self._loop(
                state=state,
                prompts=prompts,
                auto_approve=auto_approve,
                provider_stream=provider_stream,
                signal=signal,
            ):
                await self._notify(event)
                yield event
        finally:
            if signal.is_cancelled():
                self._append_interrupted_tool_results()
            if self._signal is signal:
                self._signal = None
            self._running = False
            self.last_result = ReactAgentResult(
                final_answer=state.final_answer,
                completed=state.completed,
                stop_reason=state.stop_reason,
                steps=state.turn,
                messages=list(self.messages),
                usage=dict(self.model.cumulative),
                cost_usd=round(self.model.cumulative.get("cost_usd", 0.0), 6),
            )

    async def _loop(
        self,
        *,
        state: _RunState,
        prompts: Sequence[dict[str, Any]],
        auto_approve: bool | None,
        provider_stream: bool,
        signal: SimpleCancellationToken,
    ) -> AsyncIterator[AgentEvent]:
        """The ReAct loop proper. Yields events; never touches a console."""
        yield AgentStartEvent()
        for message in prompts:
            for event in self._add_message(message):
                yield event

        pending: tuple[dict[str, Any], ...] = self._drain(self._steering)
        active = True
        while active:
            has_more_tools = True
            while has_more_tools or pending:
                for message in pending:
                    for event in self._add_message(message):
                        yield event
                pending = ()

                # An earlier turn may have already settled; ending this way overrides
                # that, so completion is cleared alongside the stop reason.
                if signal.is_cancelled():
                    state.stop_reason = "cancelled"
                    state.completed = False
                    active = False
                    break
                if self._turn_limit_reached(state.turn):
                    state.stop_reason = "max_iterations"
                    state.completed = False
                    active = False
                    break

                async for event in self._maybe_compact():
                    yield event

                state.turn += 1
                yield TurnStartEvent(turn=state.turn)

                outcome: dict[str, Any] = {}
                async for event in self._assistant_events(
                    outcome, provider_stream, signal, turn=state.turn
                ):
                    yield event

                assistant = outcome["message"]
                calls = list(assistant.get("tool_calls") or [])
                has_more_tools = bool(calls)

                abnormal = _abnormal_stop_reason(outcome.get("finish_reason"))
                if abnormal is not None:
                    # A reply cut short mid-generation is not a finished turn — and
                    # its tool calls, if any, may have truncated arguments. Stop
                    # rather than executing them or reporting success.
                    state.final_answer = assistant.get("content")
                    state.stop_reason = abnormal
                    state.completed = False
                    yield TurnEndEvent(turn=state.turn, message=assistant)
                    active = False
                    break

                if calls:
                    # The provider is answering properly again; a couple of empty
                    # replies early in a long run shouldn't spend the budget that
                    # protects a blip much later on.
                    state.empty_replies = 0

                if not calls:
                    # The implicit exit. Before honouring it, rule out the two ways a
                    # quiet turn lies about being a finished one. Held locally so a
                    # blip on this turn can't mislabel a run that goes on to finish
                    # properly several turns later.
                    settle_reason = "completed"
                    if _is_empty_reply(assistant):
                        if state.empty_replies < self.max_empty_replies:
                            state.empty_replies += 1
                            # The empty turn stays in history for diagnostics and is
                            # already filtered out of the wire format by
                            # _provider_messages(), so the retry sees the transcript
                            # exactly as the failed attempt did.
                            logger.warning(
                                "Empty reply from the provider (turn %d) — retrying "
                                "(%d/%d)",
                                state.turn,
                                state.empty_replies,
                                self.max_empty_replies,
                            )
                            yield RetryEvent(
                                attempt=state.empty_replies + 1,
                                max_attempts=self.max_empty_replies + 1,
                                delay_seconds=0.0,
                                reason="the model returned an empty reply",
                            )
                            yield TurnEndEvent(turn=state.turn, message=assistant)
                            has_more_tools = True  # keep the loop alive
                            continue
                        settle_reason = "empty_response"

                    nudge = await self._completion_nudge(state)
                    if nudge is not None:
                        yield TurnEndEvent(turn=state.turn, message=assistant)
                        pending = (_user_message(nudge),)
                        continue
                    if state.guard_unsatisfied:
                        # The guard still has something outstanding but has no nudges
                        # left: the run is stopping with work undone, which is not the
                        # same thing as having finished it.
                        settle_reason = "incomplete"

                    state.final_answer = assistant.get("content")
                    # Only a run that settled with nothing outstanding may call itself
                    # completed. The other two endings are failures that were
                    # previously indistinguishable from success.
                    state.completed = settle_reason == "completed"
                    state.stop_reason = settle_reason

                tool_results: list[dict[str, Any]] = []
                terminated = False
                for call in calls:
                    async for event in self._execute_tool_call(
                        call, auto_approve, signal
                    ):
                        yield event
                        # Only the `role: tool` message is a tool result. A result
                        # carrying images also emits a follow-up *user* message whose
                        # content is a list of parts, and collecting that would hand a
                        # non-string to `final_answer` when a tool ends the run.
                        if (
                            isinstance(event, MessageEndEvent)
                            and event.message.get("role") == "tool"
                        ):
                            tool_results.append(event.message)
                        elif isinstance(event, ToolExecutionEndEvent):
                            terminated = terminated or event.terminate

                yield TurnEndEvent(
                    turn=state.turn, message=assistant, tool_results=tool_results
                )

                if terminated:
                    # A tool declared itself the end of the run.
                    state.final_answer = _last_tool_text(tool_results)
                    state.completed = True
                    state.stop_reason = "tool_terminated"
                    active = False
                    break

                pending = self._drain(self._steering)

            if not active:
                break
            follow_ups = self._drain(self._follow_up)
            if follow_ups:
                pending = follow_ups
                continue
            break

        yield AgentEndEvent(messages=list(self.messages), stop_reason=state.stop_reason)

    async def _notify(self, event: AgentEvent) -> None:
        """Deliver one event to every subscriber, awaiting async listeners."""
        for listener in list(self._listeners):
            result = listener(event)
            if isawaitable(result):
                await result

    def _ensure_not_running(self) -> None:
        """Guard against re-entrant runs.

        Raises:
            RuntimeError: If a run is already in progress.
        """
        if self._running:
            raise RuntimeError(
                "ReactAgent is already running; use steer() or follow_up() to queue "
                "messages, or cancel() to stop it."
            )

    def _turn_limit_reached(self, turn: int) -> bool:
        """Return True when `max_iterations` has been consumed."""
        limit = self.max_iterations
        if limit is None or limit == -1:
            return False
        return turn >= limit

    # ----------------------------------------------------------------------- #
    # History
    # ----------------------------------------------------------------------- #
    def _append(self, message: dict[str, Any]) -> None:
        """Add a message to history and record it in the session, if any."""
        self.messages.append(message)
        if self.session is not None:
            self.session.append_message(message)

    def _add_message(self, message: dict[str, Any]) -> list[AgentEvent]:
        """Append a message and return its start/end events."""
        self._append(message)
        return [MessageStartEvent(message=message), MessageEndEvent(message=message)]

    async def _completion_nudge(self, state: _RunState) -> str | None:
        """Ask the completion guard whether the run may settle.

        Returns:
            str | None: A message to inject and continue with, or None to let the run
                settle. When the guard reports outstanding work but the nudge budget is
                spent, this returns None *and* sets `state.guard_unsatisfied`, so the
                caller stops with `incomplete` rather than claiming success.
        """
        if self.completion_guard is None:
            return None
        outcome = self.completion_guard()
        reason = await outcome if isawaitable(outcome) else outcome
        if not reason:
            return None
        if state.nudges_spent >= self.max_completion_nudges:
            logger.warning(
                "Completion guard still unsatisfied after %d nudges: %s",
                state.nudges_spent,
                reason,
            )
            state.guard_unsatisfied = True
            return None
        state.nudges_spent += 1
        return str(reason)

    def _provider_messages(self) -> list[dict[str, Any]]:
        """Return history filtered and cleaned to what a provider will accept.

        Two kinds of divergence between durable history and the wire format:

        * An assistant message with neither content nor tool calls (nor thinking) is
          kept for diagnostics but must not be replayed — providers reject it.
        * Loop-only bookkeeping (`is_error` on tool results, `reasoning_content`,
          which is output-only) is stripped. `thinking_blocks` are *kept*: Anthropic
          requires the signed blocks back to continue an extended-thinking turn.
        """
        cleaned: list[dict[str, Any]] = []
        for message in self.messages:
            if (
                message.get("role") == "assistant"
                and not message.get("content")
                and not message.get("tool_calls")
                and not message.get("thinking_blocks")
            ):
                continue
            drop = {"is_error", "reasoning_content"}
            if not self.preserve_reasoning:
                drop.add("thinking_blocks")
            if drop & message.keys():
                message = {k: v for k, v in message.items() if k not in drop}
            cleaned.append(message)
        return cleaned

    def _append_interrupted_tool_results(self) -> int:
        """Answer every tool call that never received a result.

        A cancelled (or crashed) run can leave an assistant turn whose `tool_calls`
        have no matching `role: tool` messages, which every provider rejects on the
        next request. This closes those gaps with an error result.

        Returns:
            int: How many placeholder results were appended.
        """
        returned = {
            message.get("tool_call_id")
            for message in self.messages
            if message.get("role") == "tool"
        }
        added = 0
        for message in list(self.messages):
            if message.get("role") != "assistant":
                continue
            for call in message.get("tool_calls") or []:
                call_id = call.get("id")
                if call_id in returned:
                    continue
                returned.add(call_id)
                self._append(
                    _tool_message(
                        INTERRUPTED_TOOL_RESULT,
                        call_id,
                        (call.get("function") or {}).get("name", "unknown"),
                    )
                )
                added += 1
        return added

    def _drain(self, queue: deque[dict[str, Any]]) -> tuple[dict[str, Any], ...]:
        """Pop queued messages according to `queue_mode`."""
        if not queue:
            return ()
        if self.queue_mode == "all":
            messages = tuple(queue)
            queue.clear()
            return messages
        return (queue.popleft(),)

    # ----------------------------------------------------------------------- #
    # Compaction
    # ----------------------------------------------------------------------- #
    def _tool_specs(self) -> list[dict[str, Any]] | None:
        """Return the tool schemas sent to the model, or None when there are none."""
        return self.tool_router.get_tool_specs_for_llm() or None

    async def _maybe_compact(self) -> AsyncIterator[AgentEvent]:
        """Summarise older history when the next request would overflow the window."""
        if not self.auto_compact or self.compactor is None:
            return
        specs = self._tool_specs()
        if not self.compactor.should_compact(self.messages, specs):
            return

        tokens_before = estimate_context_tokens(self.messages, specs)
        yield CompactionStartEvent(tokens_before=tokens_before)
        result = await self.compactor.compact(self.messages, specs)
        if result is None:
            # Nothing safe to drop (or summarisation failed) — carry on uncompacted.
            yield CompactionEndEvent(
                tokens_before=tokens_before,
                tokens_after=tokens_before,
                summary="",
            )
            return

        self.messages = result.messages
        if self.session is not None:
            self.session.append_compaction(
                summary=result.summary,
                tokens_before=result.tokens_before,
                tokens_after=result.tokens_after,
                kept_tail_count=result.kept_tail_count,
                dropped_count=result.dropped_count,
            )
        yield CompactionEndEvent(
            tokens_before=result.tokens_before,
            tokens_after=result.tokens_after,
            summary=result.summary,
        )

    # ----------------------------------------------------------------------- #
    # Tool execution + approval
    # ----------------------------------------------------------------------- #
    async def _execute_tool_call(
        self,
        call: dict[str, Any],
        auto_approve: bool | None,
        signal: CancellationToken,
    ) -> AsyncIterator[AgentEvent]:
        """Execute one tool call, appending its result and emitting its events.

        Order of gates: argument parsing → approval → cancellation →
        `before_tool_call` → the tool itself. `after_tool_call` then sees every
        outcome, including blocked and failed ones, so a policy can rewrite results
        uniformly.
        """
        name = call["function"]["name"]
        call_id = call["id"]
        raw_args = call["function"].get("arguments") or "{}"

        try:
            args = json.loads(raw_args) if raw_args.strip() else {}
            if not isinstance(args, dict):
                raise ValueError("arguments must be a JSON object")
        except (json.JSONDecodeError, ValueError, TypeError):
            args = {}
            yield ToolExecutionStartEvent(
                tool_call_id=call_id, tool_name=name, args=args
            )
            for event in self._finish_tool_call(
                call_id,
                name,
                ToolResult.error(
                    f"ERROR: arguments for '{name}' were not a valid JSON object."
                ),
            ):
                yield event
            return

        yield ToolExecutionStartEvent(tool_call_id=call_id, tool_name=name, args=args)
        invocation = ToolInvocation(tool_call_id=call_id, name=name, arguments=args)

        result: ToolResult | None = None
        tool = self.tool_router.get(name)
        if (
            tool is not None
            and tool.requires_approval
            and not self._approve(name, args, auto_approve)
        ):
            result = ToolResult.error(
                f"Tool '{name}' was not approved by the user; it was skipped."
            )
        elif signal.is_cancelled():
            result = ToolResult.error(INTERRUPTED_TOOL_RESULT)
        elif self.before_tool_call is not None:
            blocked, reason = await _maybe_await(self.before_tool_call(invocation))
            if blocked:
                result = ToolResult.error(
                    reason or f"Tool '{name}' was blocked before execution."
                )

        if result is None:
            async for event, finished in self._run_tool_with_progress(
                invocation, signal
            ):
                if event is not None:
                    yield event
                else:
                    result = finished

        assert result is not None  # every branch above produces one
        if self.after_tool_call is not None:
            result = ToolResult.coerce(
                await _maybe_await(self.after_tool_call(invocation, result))
            )

        for event in self._finish_tool_call(call_id, name, result):
            yield event

    async def _run_tool_with_progress(
        self, invocation: ToolInvocation, signal: CancellationToken
    ) -> AsyncIterator[tuple[AgentEvent | None, ToolResult | None]]:
        """Run a tool, surfacing its `on_update` reports as events while it works.

        The tool runs as a task while this generator drains a queue the progress
        callback writes to, so a long-running tool's output reaches subscribers live
        instead of arriving in a batch once it returns.

        Yields:
            tuple[AgentEvent | None, ToolResult | None]: `(event, None)` per progress
                report, then `(None, result)` exactly once at the end.
        """
        queue: asyncio.Queue[Any] = asyncio.Queue()
        loop = asyncio.get_running_loop()

        def on_update(partial: Any) -> None:
            # Called from inside the tool, possibly from another thread.
            loop.call_soon_threadsafe(queue.put_nowait, partial)

        task = asyncio.ensure_future(
            self.tool_router.call_tool(
                invocation.name,
                invocation.arguments,
                tool_call_id=invocation.tool_call_id,
                signal=signal,
                on_update=on_update,
            )
        )

        while True:
            drain = asyncio.ensure_future(queue.get())
            done, _ = await asyncio.wait(
                {task, drain}, return_when=asyncio.FIRST_COMPLETED
            )
            if drain in done:
                partial = ToolResult.coerce(drain.result())
                yield (
                    ToolExecutionUpdateEvent(
                        tool_call_id=invocation.tool_call_id,
                        tool_name=invocation.name,
                        output=partial.model_text(),
                        details=partial.details,
                    ),
                    None,
                )
                continue
            drain.cancel()
            break

        # Flush anything reported between the last drain and the tool returning.
        while not queue.empty():
            partial = ToolResult.coerce(queue.get_nowait())
            yield (
                ToolExecutionUpdateEvent(
                    tool_call_id=invocation.tool_call_id,
                    tool_name=invocation.name,
                    output=partial.model_text(),
                    details=partial.details,
                ),
                None,
            )
        yield (None, await task)

    def _finish_tool_call(
        self, call_id: str, name: str, result: ToolResult
    ) -> list[AgentEvent]:
        """Emit the end-of-execution event and append the resulting message(s).

        Any tools the result unlocks are activated here, so they appear in the schemas
        sent on the very next turn. Images become a follow-up user message, since the
        Chat Completions tool-result slot is text-only.
        """
        activated = (
            self.tool_router.activate(*result.added_tool_names)
            if result.added_tool_names
            else []
        )
        events: list[AgentEvent] = [
            ToolExecutionEndEvent(
                tool_call_id=call_id,
                tool_name=name,
                output=result.model_text(),
                is_error=result.is_error,
                details=result.details,
                added_tool_names=activated,
                terminate=result.terminate,
            )
        ]
        events.extend(
            self._add_message(
                _tool_message(
                    result.model_text(), call_id, name, is_error=result.is_error
                )
            )
        )
        followup = image_followup_message(result, name)
        if followup is not None:
            events.extend(self._add_message(followup))
        return events

    def _approve(self, name: str, args: dict, auto_approve: bool | None) -> bool:
        """Decide whether a tool requiring approval may run.

        Resolution order: explicit `auto_approve` / `yolo_mode` →
        `approval_callback` → interactive Rich prompt (only on a TTY) → reject.

        Args:
            name (str): The tool name awaiting approval.
            args (dict): The parsed arguments the tool would be called with.
            auto_approve (bool | None): Per-run override of `yolo_mode`.

        Returns:
            bool: True to execute the tool, False to skip it.
        """
        effective = self.yolo_mode if auto_approve is None else auto_approve
        if effective:
            return True
        if self.approval_callback is not None:
            try:
                return bool(self.approval_callback(name, args))
            except Exception:  # noqa: BLE001 — a failing callback means "do not run"
                return False
        try:
            import sys

            if not sys.stdin.isatty():
                return False
            from rich.prompt import Confirm

            return Confirm.ask(
                f"Approve tool '{name}' with args {short_text(args)}?",
                default=False,
            )
        except Exception:  # noqa: BLE001
            return False

    # ----------------------------------------------------------------------- #
    # LLM call (with retry) — streaming + non-streaming
    # ----------------------------------------------------------------------- #
    async def _assistant_events(
        self,
        outcome: dict[str, Any],
        provider_stream: bool,
        signal: CancellationToken,
        turn: int | None = None,
    ) -> AsyncIterator[AgentEvent]:
        """Call the model, append the assistant message, and emit its events.

        Also carries the per-call cost-tracking provenance: each attempt is timed, and
        every outcome — success, transient failure that will be retried, terminal
        failure — is reported to the model wrapper, which prices it and forwards a
        ledger row to whatever usage sink is installed. Failed attempts cost no tokens
        but are recorded anyway, so the call table shows what the provider actually saw.

        Args:
            outcome (dict): Filled with `message` (the appended assistant message)
                and `finish_reason` before the generator ends.
            provider_stream (bool): Whether to request a streaming completion.
            signal (CancellationToken): Polled between chunks so a cancelled run stops
                consuming the stream promptly.
            turn (int | None): The 1-based turn number, stamped onto the ledger row.
        """
        partial: dict[str, Any] = {"role": "assistant", "content": ""}
        yield MessageStartEvent(message=dict(partial))

        result: LLMResult | None = None
        raw_usage: Any = None
        provider: str | None = None
        response: Any = None
        attempts = 0
        elapsed_ms = 0
        for attempt in range(_MAX_LLM_RETRIES):
            accumulator = _StreamAccumulator()
            attempts = attempt + 1
            started_at = utc_now_iso()
            elapsed = call_timer()
            try:
                response = await self.model.acompletion(
                    messages=self._provider_messages(),
                    tools=self._tool_specs(),
                    stream=provider_stream,
                    tool_choice=self.first_tool_choice if turn == 1 else None,
                )
                if not provider_stream:
                    result, raw_usage = self._parse_response(response)
                    elapsed_ms = elapsed()
                    break

                async for chunk in response:
                    if signal.is_cancelled():
                        break
                    text, reasoning = accumulator.consume(chunk)
                    if text or reasoning:
                        partial["content"] = accumulator.content
                        yield MessageUpdateEvent(
                            delta=text,
                            reasoning_delta=reasoning,
                            message=dict(partial),
                        )
                result, raw_usage = accumulator.finish()
                provider = accumulator.provider
                elapsed_ms = elapsed()
                break
            except Exception as e:  # noqa: BLE001
                # A stream that already produced output cannot be retried: the deltas
                # are gone and re-issuing would duplicate them.
                if accumulator.received:
                    self.model.record_failure(
                        e,
                        context={
                            "kind": self.usage_kind,
                            "turn": turn,
                            "attempt": attempts,
                            "started_at": started_at,
                            "duration_ms": elapsed(),
                            "streamed": provider_stream,
                            "provider": accumulator.provider,
                        },
                    )
                    raise
                delay = _retry_delay_for(e, attempt)
                terminal = attempt >= _MAX_LLM_RETRIES - 1 or delay is None
                self.model.record_failure(
                    e,
                    context={
                        "kind": self.usage_kind,
                        "status": "error" if terminal else "retry",
                        "turn": turn,
                        "attempt": attempts,
                        "started_at": started_at,
                        "duration_ms": elapsed(),
                        "streamed": provider_stream,
                    },
                )
                if terminal:
                    raise
                logger.warning(
                    "Transient LLM error (attempt %d): %s — retrying in %ds",
                    attempt + 1,
                    e,
                    delay,
                )
                yield RetryEvent(
                    attempt=attempt + 2,
                    max_attempts=_MAX_LLM_RETRIES,
                    delay_seconds=float(delay),
                    reason=str(e),
                )
                if not await _sleep_unless_cancelled(delay, signal):
                    raise

        assert result is not None  # the loop either breaks with a result or raises
        result.usage = self.model.record_usage(
            raw_usage,
            response=response,
            context={
                "kind": self.usage_kind,
                "turn": turn,
                "attempt": attempts,
                "started_at": started_at,
                "duration_ms": elapsed_ms,
                "streamed": provider_stream,
                "finish_reason": result.finish_reason,
                "provider": provider,
            },
        )
        if not self.preserve_reasoning:
            result.reasoning_content = None
            result.thinking_blocks = None
        message = result.to_message()
        self._append(message)
        outcome["message"] = message
        outcome["finish_reason"] = result.finish_reason
        yield MessageEndEvent(message=message)

    @staticmethod
    def _parse_response(response: Any) -> tuple[LLMResult, Any]:
        """Normalise a non-streaming litellm response into `(LLMResult, raw_usage)`."""
        choice = response.choices[0]
        message = choice.message
        content = message.content or None
        finish_reason = choice.finish_reason
        reasoning_content, thinking_blocks = extract_reasoning(message)

        tool_calls_acc: dict[int, dict] = {}
        if getattr(message, "tool_calls", None):
            for idx, tc in enumerate(message.tool_calls):
                tool_calls_acc[idx] = {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
        result = LLMResult(
            content,
            tool_calls_acc,
            finish_reason,
            reasoning_content=reasoning_content,
            thinking_blocks=thinking_blocks,
        )
        return result, getattr(response, "usage", None)


async def _sleep_unless_cancelled(
    delay: float, signal: CancellationToken | None
) -> bool:
    """Sleep in short steps, returning False as soon as cancellation is requested."""
    remaining = float(delay)
    while remaining > 0:
        if signal is not None and signal.is_cancelled():
            return False
        step = min(1.0, remaining)
        await asyncio.sleep(step)
        remaining -= step
    return signal is None or not signal.is_cancelled()


class _StreamAccumulator:
    """Reassembles a streaming completion chunk by chunk.

    Kept separate from the event-emitting generator so the (fiddly) delta-merging
    logic stays testable on its own and a cancelled stream still yields whatever
    arrived before the interruption.
    """

    def __init__(self) -> None:
        self._content_parts: list[str] = []
        self._reasoning_parts: list[str] = []
        self._thinking_blocks: list = []
        self._tool_calls: dict[int, dict] = {}
        self._finish_reason: str | None = None
        self._usage: Any = None
        self._provider: str | None = None
        self.received = False

    @property
    def content(self) -> str:
        """The assistant text received so far."""
        return "".join(self._content_parts)

    @property
    def provider(self) -> str | None:
        """The upstream provider this stream reported, if any chunk carried one.

        A streaming response has no single object to read provenance off, and only
        some chunks repeat the field — so it is captured as the stream goes by rather
        than reconstructed afterwards.
        """
        return self._provider

    def consume(self, chunk: Any) -> tuple[str, str]:
        """Merge one chunk and return its `(text_delta, reasoning_delta)`.

        Either may be empty; a chunk carrying only tool-call fragments returns both
        empty but still marks the stream as having produced output.
        """
        if getattr(chunk, "usage", None):
            self._usage = chunk.usage
        if self._provider is None:
            self._provider = read_provider_field(chunk)
        choice = chunk.choices[0] if getattr(chunk, "choices", None) else None
        if choice is None:
            return "", ""

        delta = choice.delta
        if choice.finish_reason:
            self._finish_reason = choice.finish_reason

        text = getattr(delta, "content", None) or ""
        if text:
            self._content_parts.append(text)
            self.received = True

        reasoning, blocks = extract_reasoning(delta)
        if reasoning:
            self._reasoning_parts.append(reasoning)
            self.received = True
        if blocks:
            self._thinking_blocks.extend(blocks)
            self.received = True

        for tc_delta in getattr(delta, "tool_calls", None) or []:
            self.received = True
            slot = self._tool_calls.setdefault(
                tc_delta.index,
                {
                    "id": "",
                    "type": "function",
                    "function": {"name": "", "arguments": ""},
                },
            )
            if tc_delta.id:
                slot["id"] = tc_delta.id
            if tc_delta.function:
                if tc_delta.function.name:
                    slot["function"]["name"] += tc_delta.function.name
                if tc_delta.function.arguments:
                    slot["function"]["arguments"] += tc_delta.function.arguments
        return text, reasoning or ""

    def finish(self) -> tuple[LLMResult, Any]:
        """Return the normalised `(LLMResult, raw_usage)` for the stream."""
        return (
            LLMResult(
                self.content or None,
                self._tool_calls,
                self._finish_reason,
                reasoning_content="".join(self._reasoning_parts) or None,
                thinking_blocks=self._thinking_blocks or None,
            ),
            self._usage,
        )


async def _maybe_await(value: Any) -> Any:
    """Await `value` when it is awaitable, so hooks may be sync or async."""
    return await value if isawaitable(value) else value
