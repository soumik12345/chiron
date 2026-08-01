"""Context accounting and automatic history compaction.

The agent's message list grows without bound as it works; left alone it eventually
exceeds the model's context window and the provider rejects the request. This module
provides the two halves of the fix:

* **Accounting** — a deterministic, offline token estimate for the system prompt,
  the message list, and the tool schemas (which are re-sent on every call and are
  easy to forget). No tokenizer dependency: characters ÷ 4 plus per-item overhead.
* **Compaction** — when the estimate crosses a threshold derived from the model's
  context window, the older half of history is replaced by a structured summary and
  the recent tail is kept verbatim.

The tail is chosen so it never begins with a `role: tool` message, which would
orphan a tool result from its assistant tool call and be rejected by the provider.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import litellm

from chiron.models.usage import call_timer, utc_now_iso

logger = logging.getLogger(__name__)

CHARS_PER_TOKEN = 4
MESSAGE_OVERHEAD_TOKENS = 4
TOOL_OVERHEAD_TOKENS = 16
# A flat stand-in for one image part. Providers bill images by tile count, which has
# no relation to the length of the base64 payload we would otherwise measure.
IMAGE_TOKENS = 1_500

DEFAULT_CONTEXT_WINDOW_TOKENS = 128_000
DEFAULT_COMPACTION_RESERVE_TOKENS = 16_384
DEFAULT_KEEP_RECENT_TOKENS = 20_000

COMPACTION_SUMMARY_PREFIX = "Previous conversation summary:\n"

SUMMARIZATION_SYSTEM_PROMPT = (
    "You are a context summarization assistant. Your task is to read a conversation "
    "between a user and an AI agent, then produce a structured summary following the "
    "exact format specified.\n\n"
    "Do NOT continue the conversation. Do NOT respond to any questions in the "
    "conversation. ONLY output the structured summary."
)

SUMMARIZATION_PROMPT = """The messages above are a conversation to summarize. Create a \
structured context checkpoint summary that another LLM will use to continue the work.

Use this EXACT format:

## Goal
[What is the user trying to accomplish?]

## Constraints & Preferences
- [Any constraints, preferences, or requirements mentioned by the user, or "(none)"]

## Progress
### Done
- [x] [Completed tasks/changes]

### In Progress
- [ ] [Current work]

### Blocked
- [Issues preventing progress, if any]

## Key Decisions
- **[Decision]**: [Brief rationale]

## Next Steps
1. [Ordered list of what should happen next]

## Critical Context
- [Any data, examples, or references needed to continue, or "(none)"]

Keep each section concise. Preserve exact file paths, identifiers, and error messages."""

UPDATE_SUMMARIZATION_PROMPT = """The messages above are NEW conversation messages to \
incorporate into the existing summary provided in <previous-summary> tags.

Update the existing structured summary with new information. RULES:
- PRESERVE all existing information from the previous summary
- ADD new progress, decisions, and context from the new messages
- UPDATE the Progress section: move items from "In Progress" to "Done" when completed
- UPDATE "Next Steps" based on what was accomplished
- PRESERVE exact file paths, identifiers, and error messages
- If something is no longer relevant, you may remove it

Use the same section structure as the previous summary (Goal, Constraints & \
Preferences, Progress, Key Decisions, Next Steps, Critical Context).

Keep each section concise."""


@dataclass(frozen=True)
class ContextUsageEstimate:
    """Deterministic context-size accounting for one provider request.

    Attributes:
        total_tokens (int): Estimated tokens for the whole request.
        system_tokens (int): Estimated tokens contributed by the system message.
        message_tokens (int): Estimated tokens contributed by all messages.
        tool_tokens (int): Estimated tokens contributed by the tool schemas.
        message_count (int): Number of messages counted.
        tool_count (int): Number of tool schemas counted.
    """

    total_tokens: int
    system_tokens: int
    message_tokens: int
    tool_tokens: int
    message_count: int
    tool_count: int


@dataclass(frozen=True)
class CompactionResult:
    """The outcome of one compaction pass.

    Attributes:
        messages (list[dict]): The replacement message list.
        summary (str): The generated summary text.
        tokens_before (int): Estimated context size before compaction.
        tokens_after (int): Estimated context size after compaction.
        kept_tail_count (int): How many trailing messages were retained verbatim.
        dropped_count (int): How many messages the summary replaced.
    """

    messages: list[dict[str, Any]]
    summary: str
    tokens_before: int
    tokens_after: int
    kept_tail_count: int
    dropped_count: int


def estimate_text_tokens(text: str | None) -> int:
    """Return a rough, deterministic token estimate for a piece of text."""
    if not text:
        return 0
    return max(1, (len(text) + CHARS_PER_TOKEN - 1) // CHARS_PER_TOKEN)


def estimate_content_tokens(content: Any) -> int:
    """Return a rough token estimate for a message's `content`.

    Multimodal content is a list of parts. Text parts are measured normally; images
    get a flat estimate, because their base64 payload is enormous relative to what a
    vision model actually charges for it.
    """
    if content is None:
        return 0
    if isinstance(content, str):
        return estimate_text_tokens(content)
    if isinstance(content, list):
        total = 0
        for part in content:
            if not isinstance(part, dict):
                total += estimate_text_tokens(str(part))
            elif part.get("type") == "image_url":
                total += IMAGE_TOKENS
            else:
                total += estimate_text_tokens(part.get("text", ""))
        return total
    return estimate_text_tokens(str(content))


def estimate_message_tokens(message: dict[str, Any]) -> int:
    """Return a rough token estimate for one OpenAI-format message dict.

    Counts the message content plus, for assistant messages, each tool call's name
    and serialised arguments — which are frequently larger than the text content —
    and any reasoning the model returned.
    """
    tokens = MESSAGE_OVERHEAD_TOKENS + estimate_content_tokens(message.get("content"))
    tokens += estimate_text_tokens(message.get("reasoning_content"))
    for block in message.get("thinking_blocks") or []:
        if isinstance(block, dict):
            tokens += estimate_text_tokens(block.get("thinking", ""))
    for call in message.get("tool_calls") or []:
        function = call.get("function") or {}
        tokens += estimate_text_tokens(function.get("name"))
        tokens += estimate_text_tokens(function.get("arguments"))
    if message.get("role") == "tool":
        tokens += estimate_text_tokens(message.get("name"))
    return tokens


def estimate_tool_tokens(spec: dict[str, Any]) -> int:
    """Return a rough token estimate for one OpenAI function-calling tool schema."""
    function = spec.get("function") or {}
    return (
        TOOL_OVERHEAD_TOKENS
        + estimate_text_tokens(function.get("name"))
        + estimate_text_tokens(function.get("description"))
        + estimate_text_tokens(str(function.get("parameters", "")))
    )


def estimate_context_usage(
    messages: list[dict[str, Any]],
    tool_specs: list[dict[str, Any]] | None = None,
) -> ContextUsageEstimate:
    """Return deterministic context accounting for a would-be provider request.

    Args:
        messages (list[dict]): The message list, including the system message.
        tool_specs (list[dict] | None): Tool schemas sent alongside the messages.

    Returns:
        ContextUsageEstimate: The per-section and total token estimate.
    """
    specs = tool_specs or []
    system_tokens = sum(
        estimate_message_tokens(m) for m in messages if m.get("role") == "system"
    )
    message_tokens = sum(estimate_message_tokens(m) for m in messages)
    tool_tokens = sum(estimate_tool_tokens(s) for s in specs)
    return ContextUsageEstimate(
        total_tokens=message_tokens + tool_tokens,
        system_tokens=system_tokens,
        message_tokens=message_tokens,
        tool_tokens=tool_tokens,
        message_count=len(messages),
        tool_count=len(specs),
    )


def estimate_context_tokens(
    messages: list[dict[str, Any]],
    tool_specs: list[dict[str, Any]] | None = None,
) -> int:
    """Return the estimated total context size for a would-be provider request."""
    return estimate_context_usage(messages, tool_specs).total_tokens


def resolve_context_window(model_id: str) -> int:
    """Return the model's input context window in tokens, best-effort.

    Asks litellm for the model's metadata, trying the provider-stripped forms that
    litellm actually knows about (it does not carry entries for `openrouter/`
    prefixed ids). Falls back to :data:`DEFAULT_CONTEXT_WINDOW_TOKENS`.
    """
    candidates = [model_id]
    if model_id.startswith("openrouter/"):
        candidates.append(model_id[len("openrouter/") :])
    if "/" in model_id:
        candidates.append(model_id.rsplit("/", 1)[1])

    for candidate in candidates:
        try:
            info = litellm.get_model_info(candidate)
        except Exception:  # noqa: BLE001 - unknown model ids raise
            continue
        window = info.get("max_input_tokens") or info.get("max_tokens")
        if window:
            return int(window)
    return DEFAULT_CONTEXT_WINDOW_TOKENS


def compaction_threshold(context_window_tokens: int, reserve_tokens: int) -> int:
    """Return the token count at which compaction should trigger."""
    return max(1, context_window_tokens - reserve_tokens)


def split_for_compaction(
    messages: list[dict[str, Any]], keep_recent_tokens: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Split messages into `(to_summarise, to_keep)` at a provider-safe boundary.

    The system message (index 0) is excluded from both halves — the caller re-attaches
    it. The boundary is advanced past any leading `role: tool` message so a tool
    result is never separated from the assistant tool call that produced it.

    Args:
        messages (list[dict]): Full history, system message first.
        keep_recent_tokens (int): Approximate token budget for the retained tail.

    Returns:
        tuple[list[dict], list[dict]]: The head to summarise and the tail to keep.
    """
    body = (
        messages[1:] if messages and messages[0].get("role") == "system" else messages
    )
    if not body:
        return [], []

    total = 0
    index = 0
    for i in range(len(body) - 1, -1, -1):
        total += estimate_message_tokens(body[i])
        if total > keep_recent_tokens:
            index = i + 1
            break

    # Never start the retained tail on an orphaned tool result.
    while index < len(body) and body[index].get("role") == "tool":
        index += 1
    return body[:index], body[index:]


def split_previous_summary(
    messages: list[dict[str, Any]],
) -> tuple[str | None, list[dict[str, Any]]]:
    """Peel off an earlier compaction summary so re-compaction can update it.

    Args:
        messages (list[dict]): The head about to be summarised.

    Returns:
        tuple[str | None, list[dict]]: The previous summary text (or None) and the
            remaining messages that are new since it was written.
    """
    if not messages:
        return None, messages
    first = messages[0]
    content = first.get("content")
    if first.get("role") != "user" or not isinstance(content, str):
        return None, messages
    if not content.startswith(COMPACTION_SUMMARY_PREFIX):
        return None, messages
    return content[len(COMPACTION_SUMMARY_PREFIX) :], messages[1:]


def serialize_for_compaction(messages: list[dict[str, Any]]) -> str:
    """Render messages into the tagged transcript handed to the summarizer."""
    if not messages:
        return "(no new messages)"
    lines: list[str] = []
    for index, message in enumerate(messages, start=1):
        role = message.get("role", "unknown")
        attributes = f"index={index} role={role}"
        if role == "tool":
            attributes += f" name={message.get('name')}"
        lines.append(f"<message {attributes}>")
        content = message.get("content")
        if content:
            lines.append(str(content))
        for call in message.get("tool_calls") or []:
            function = call.get("function") or {}
            lines.append(
                f"- tool call {function.get('name')}: {function.get('arguments')}"
            )
        lines.append("</message>")
    return "\n".join(lines)


def build_compaction_prompt(
    messages: list[dict[str, Any]], *, instructions: str | None = None
) -> str:
    """Build the summarizer prompt for a head of history.

    Uses the "update" variant when the head already opens with a previous compaction
    summary, so summaries accumulate rather than being rewritten from scratch.
    """
    previous, new_messages = split_previous_summary(messages)
    prompt = (
        f"<conversation>\n{serialize_for_compaction(new_messages)}\n</conversation>\n\n"
    )
    base = UPDATE_SUMMARIZATION_PROMPT if previous is not None else SUMMARIZATION_PROMPT
    if previous is not None:
        prompt += f"<previous-summary>\n{previous}\n</previous-summary>\n\n"
    if instructions and instructions.strip():
        base = f"{base}\n\nAdditional focus: {instructions.strip()}"
    return f"{prompt}{base}"


def summary_message(summary: str) -> dict[str, Any]:
    """Wrap a summary as the user message that stands in for compacted history."""
    return {"role": "user", "content": f"{COMPACTION_SUMMARY_PREFIX}{summary}"}


class ContextCompactor:
    """Decides when to compact and performs the summarisation LLM call.

    Attributes:
        model: The model wrapper used for the summarisation call (usually the agent's).
        context_window_tokens (int): The model's input context window.
        reserve_tokens (int): Headroom left for the next response.
        keep_recent_tokens (int): Approximate budget for the verbatim tail.
        instructions (str | None): Extra focus appended to the summarizer prompt.
    """

    def __init__(
        self,
        model: Any,
        *,
        context_window_tokens: int | None = None,
        reserve_tokens: int = DEFAULT_COMPACTION_RESERVE_TOKENS,
        keep_recent_tokens: int = DEFAULT_KEEP_RECENT_TOKENS,
        instructions: str | None = None,
    ) -> None:
        """Initialise the compactor.

        Args:
            model: Object exposing `acompletion` and `record_usage` (a
                :class:`~chiron.models.litellm_model.LiteLLMModel`, or a stand-in).
            context_window_tokens (int | None): Override the model's context window.
                Resolved from the model id when omitted.
            reserve_tokens (int): Tokens reserved for the next completion.
            keep_recent_tokens (int): Approximate token budget for retained messages.
            instructions (str | None): Extra summarizer guidance.
        """
        self.model = model
        if context_window_tokens is None:
            model_id = getattr(model, "model_id", "") or ""
            context_window_tokens = (
                resolve_context_window(model_id)
                if model_id
                else DEFAULT_CONTEXT_WINDOW_TOKENS
            )
        self.context_window_tokens = context_window_tokens
        self.reserve_tokens = reserve_tokens
        self.keep_recent_tokens = keep_recent_tokens
        self.instructions = instructions

    @property
    def threshold(self) -> int:
        """The estimated token count at which :meth:`compact` should be called."""
        return compaction_threshold(self.context_window_tokens, self.reserve_tokens)

    def should_compact(
        self,
        messages: list[dict[str, Any]],
        tool_specs: list[dict[str, Any]] | None = None,
    ) -> bool:
        """Return True when the next request would exceed the compaction threshold."""
        return estimate_context_tokens(messages, tool_specs) >= self.threshold

    async def compact(
        self,
        messages: list[dict[str, Any]],
        tool_specs: list[dict[str, Any]] | None = None,
    ) -> CompactionResult | None:
        """Summarise older history and return the replacement message list.

        Args:
            messages (list[dict]): Full history, system message first.
            tool_specs (list[dict] | None): Tool schemas, counted in the estimate.

        Returns:
            CompactionResult | None: The compaction outcome, or None when there is
                nothing safe to drop (the tail alone already fills the budget) or the
                summarisation call failed.
        """
        tokens_before = estimate_context_tokens(messages, tool_specs)
        head, tail = split_for_compaction(messages, self.keep_recent_tokens)
        if not head:
            return None

        try:
            summary = await self._summarize(head)
        except Exception as e:  # noqa: BLE001 - compaction must never kill a run
            logger.warning("Compaction summarisation failed; keeping history: %s", e)
            return None

        system = (
            messages[:1] if messages and messages[0].get("role") == "system" else []
        )
        compacted = [*system, summary_message(summary), *tail]
        return CompactionResult(
            messages=compacted,
            summary=summary,
            tokens_before=tokens_before,
            tokens_after=estimate_context_tokens(compacted, tool_specs),
            kept_tail_count=len(tail),
            dropped_count=len(head),
        )

    async def _summarize(self, head: list[dict[str, Any]]) -> str:
        """Run the summarisation completion and return its text.

        The call sends no tools and does not stream, so it cannot recurse into the
        agent loop. Its usage is folded into the model's cumulative totals, and it is
        recorded as its own `kind="compaction"` ledger row — compaction is a real
        charge that would otherwise hide inside the turn totals, and knowing what it
        costs is the whole point of a per-call ledger.
        """
        started_at = utc_now_iso()
        elapsed = call_timer()
        response = await self.model.acompletion(
            messages=[
                {"role": "system", "content": SUMMARIZATION_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": build_compaction_prompt(
                        head, instructions=self.instructions
                    ),
                },
            ],
            tools=None,
            stream=False,
        )
        self.model.record_usage(
            getattr(response, "usage", None),
            response=response,
            context={
                "kind": "compaction",
                "started_at": started_at,
                "duration_ms": elapsed(),
            },
        )
        content = response.choices[0].message.content
        if not content or not content.strip():
            raise ValueError("summarizer returned empty content")
        return content.strip()
