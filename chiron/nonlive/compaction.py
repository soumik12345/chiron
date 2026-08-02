"""Keeping a non-live conversation inside the model's window, on purpose.

Before this existed, non-live mode had guards but no accounting. The caps were
count-denominated proxies — forty messages, forty journal entries, images demoted
to placeholders — and none of them knows what a token is. On a small-context
model (and the picker deliberately accepts any pasted OpenRouter id) the
assembled request can exceed the window; when it did, litellm's error matched
nothing in :func:`~chiron.live.session.is_permanent_error`'s marker list, so it
was treated as transient, and since history is only appended to on *success* the
oversized history was never trimmed. Every subsequent question re-assembled the
same doomed request. The wall was permanent until the provider was rebuilt.

Three things fix it, in the order they matter:

1. **Deduplicate first.** The journal block is prepended to every question and
   used to survive into history as text, so twenty remembered exchanges carried
   up to twenty stale copies of it — on the order of 12k tokens of duplication
   before any real conversation was counted. Stripping it as a turn enters
   history (the same demotion images already get) means most sessions on 128k
   models never compact at all, which is the correct outcome. That lives in
   :mod:`chiron.nonlive.session`; this module is what happens when it is not
   enough.
2. **Proactive.** Price the assembled messages against the model's window before
   each call and summarise above :data:`COMPACTION_TRIGGER_RATIO`.
3. **Reactive.** Catch the typed :class:`litellm.ContextWindowExceededError` —
   which also cures the misclassification above — compact, and retry once. A
   rejected request bills nothing, so the backstop costs one round of latency
   that the proactive path makes rare.

Everything here is pure: counting, splitting, and building the summariser's
request. The call itself belongs to the session manager, which owns the lock
that history is mutated under.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

#: Fraction of the context window at which history is summarised proactively.
#: The same ratio the compaction trigger in chiron's agent harness settled on;
#: the remaining 15% has to hold the system prompt, the current turn's frames and
#: the answer, none of which are in the history being measured.
COMPACTION_TRIGGER_RATIO = 0.85

#: Messages kept verbatim after a compaction. History here is strictly
#: user/assistant pairs, so an even number keeps whole exchanges — a tail that
#: opens on an assistant reply to a question that has been summarised away reads
#: as a non-sequitur to the next model that sees it.
KEEP_RECENT_MESSAGES = 8

#: Tokens charged to one image part when measuring. The base64 payload is
#: enormous relative to what a vision model actually bills, so counting it as
#: text would overstate a request by two orders of magnitude. Frames only ride in
#: the current turn anyway; this exists so the current turn is not free.
IMAGE_TOKEN_ALLOWANCE = 300

#: Window assumed when nothing can say what the model's really is.
DEFAULT_CONTEXT_WINDOW = 128_000

#: Marks the message that stands in for compacted history, and is what lets a
#: second compaction *update* the first summary rather than summarising it again.
COMPACTION_SUMMARY_PREFIX = "EARLIER THIS SESSION (compacted summary):\n"

SUMMARY_SYSTEM_PROMPT = """\
You compress the earlier part of a gaming session into a briefing for an \
assistant that will keep helping the player. You are not talking to the player: \
your reply is inserted into a conversation as context and nothing else.

Keep what will still matter an hour from now — where the player is, what they \
are trying to do, what they have tried, what failed and why, decisions and \
preferences they have stated, and anything the assistant told them that they are \
acting on. Drop pleasantries, repeated questions, and descriptions of frames \
that have long since scrolled past.

Write compact prose under these headings, omitting any that would be empty:

Where they are:
What they are trying to do:
What has been tried:
What the player prefers:
Open questions:

Preserve exact names — areas, bosses, items, NPCs, quests. Never invent one.\
"""

UPDATE_SUMMARY_INSTRUCTION = """\
An earlier summary is included first, followed by the conversation since. \
Produce one merged summary under the same headings: keep everything from the \
earlier summary that is still true, fold in what has happened since, and drop \
what has been superseded.\
"""


@dataclass(frozen=True)
class CompactionOutcome:
    """The result of one compaction pass.

    Attributes:
        messages (list[dict]): The replacement history.
        summary (str): The summary text that stands in for what was dropped.
        tokens_before (int): Measured size of the history before.
        tokens_after (int): Measured size of the replacement.
        kept_tail_count (int): Messages retained verbatim.
        dropped_count (int): Messages the summary replaced.
        reason (str): ``threshold`` for the proactive path, ``overflow`` for the
            reactive one — which is the difference between working as designed
            and having been saved.
    """

    messages: list[dict[str, Any]]
    summary: str
    tokens_before: int
    tokens_after: int
    kept_tail_count: int
    dropped_count: int
    reason: str = "threshold"


# ------------------------------------------------------------------ measuring


def text_projection(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Messages with image parts replaced by a short stand-in.

    Token counters measure text, and a data-URI image part *is* text as far as
    they are concerned — several hundred thousand characters of base64 for one
    screenshot. Projecting them out and charging
    :data:`IMAGE_TOKEN_ALLOWANCE` separately is the only way to get a number that
    resembles what the provider will count.
    """
    projected: list[dict[str, Any]] = []
    for message in messages:
        content = message.get("content")
        if not isinstance(content, list):
            projected.append(message)
            continue
        parts: list[str] = []
        for part in content:
            if not isinstance(part, dict):
                parts.append(str(part))
            elif part.get("type") == "image_url":
                parts.append("[image]")
            else:
                parts.append(str(part.get("text") or ""))
        projected.append({**message, "content": "\n".join(parts)})
    return projected


def count_images(messages: list[dict[str, Any]]) -> int:
    """How many image parts the messages carry."""
    total = 0
    for message in messages:
        content = message.get("content")
        if isinstance(content, list):
            total += sum(
                1
                for part in content
                if isinstance(part, dict) and part.get("type") == "image_url"
            )
    return total


def count_tokens(model_id: str, messages: list[dict[str, Any]]) -> int:
    """Measure a would-be request, images charged flat.

    Uses ``litellm.token_counter`` with the model's own tokenizer where litellm
    knows it, and falls back to characters ÷ 4 when it does not — a pasted id
    from a provider litellm has never heard of must still yield a number, or the
    proactive path silently stops running for exactly the models most likely to
    have a small window.
    """
    projected = text_projection(messages)
    images = count_images(messages) * IMAGE_TOKEN_ALLOWANCE
    try:
        import litellm

        return int(litellm.token_counter(model=model_id, messages=projected)) + images
    except Exception:  # noqa: BLE001 — an unknown id is normal, not exceptional
        characters = sum(len(str(m.get("content") or "")) for m in projected)
        return characters // 4 + 4 * len(projected) + images


def context_window_for(model_id: str) -> int:
    """The model's input window in tokens, best-effort.

    The catalogue is asked first, and only its **disk cache** — the settings page
    already fetched it, and a token count computed before every question is not
    a place to make an HTTP request. litellm's bundled model info is the
    fallback, which is the right order rather than the convenient one: litellm
    carries no entries for ``openrouter/``-prefixed ids, and those are precisely
    the ones a user pastes in.
    """
    from chiron.models.catalogue import cached_context_length

    cached = cached_context_length(model_id)
    if cached:
        return cached

    try:
        import litellm

        for candidate in _id_candidates(model_id):
            try:
                info = litellm.get_model_info(candidate)
            except Exception:  # noqa: BLE001 — unknown ids raise
                continue
            window = info.get("max_input_tokens") or info.get("max_tokens")
            if window:
                return int(window)
    except Exception:  # noqa: BLE001 - pragma: no cover
        logger.debug("Could not resolve a context window for %s", model_id)
    return DEFAULT_CONTEXT_WINDOW


def _id_candidates(model_id: str) -> list[str]:
    """Forms of a model id to try against litellm's tables, most specific first."""
    candidates = [model_id]
    if model_id.startswith("openrouter/"):
        candidates.append(model_id[len("openrouter/") :])
    if "/" in model_id:
        candidates.append(model_id.rsplit("/", 1)[1])
    seen: set[str] = set()
    return [c for c in candidates if c and not (c in seen or seen.add(c))]


def should_compact(
    model_id: str,
    messages: list[dict[str, Any]],
    *,
    ratio: float = COMPACTION_TRIGGER_RATIO,
    window: int | None = None,
) -> bool:
    """Whether the next request is close enough to the window to summarise first.

    Args:
        model_id (str): The model the request will be sent to.
        messages (list[dict]): The full assembled request, system message and
            current turn included — the thing the provider will actually see.
        ratio (float): Fraction of the window that triggers compaction.
        window (int | None): Override the resolved context window, for tests.

    Returns:
        bool: True when history should be compacted before sending.
    """
    limit = window if window is not None else context_window_for(model_id)
    return count_tokens(model_id, messages) >= max(1, int(limit * ratio))


def is_context_overflow(error: BaseException) -> bool:
    """Whether a failure means "the request was too big for the window".

    Prefers litellm's typed exception and falls back to the message, because the
    same condition arrives wrapped in different classes depending on how far the
    request got before someone counted.
    """
    try:
        import litellm

        if isinstance(error, litellm.ContextWindowExceededError):
            return True
    except Exception:  # noqa: BLE001 - pragma: no cover
        pass
    text = str(error).lower()
    return any(
        marker in text
        for marker in (
            "context window",
            "context_length_exceeded",
            "maximum context length",
            "too many tokens",
            "input token count",
        )
    )


# ------------------------------------------------------------------ splitting


def summary_message(summary: str) -> dict[str, Any]:
    """Wrap a summary as the user message that stands in for dropped history."""
    return {"role": "user", "content": f"{COMPACTION_SUMMARY_PREFIX}{summary}"}


def extract_previous_summary(
    history: list[dict[str, Any]],
) -> tuple[str | None, list[dict[str, Any]]]:
    """Peel an earlier summary off the front of history, so it can be updated.

    Without this, a second compaction summarises the first summary alongside the
    new conversation, and each pass compresses what came before it again —
    detail decays geometrically for no reason. With it, summaries accumulate.
    """
    if not history:
        return None, history
    first = history[0]
    content = first.get("content")
    if first.get("role") != "user" or not isinstance(content, str):
        return None, history
    if not content.startswith(COMPACTION_SUMMARY_PREFIX):
        return None, history
    return content[len(COMPACTION_SUMMARY_PREFIX) :], history[1:]


def split_history(
    history: list[dict[str, Any]], keep_recent: int = KEEP_RECENT_MESSAGES
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Split history into ``(to_summarise, to_keep_verbatim)``.

    The tail is nudged to start on a ``user`` message so a kept assistant reply
    is never orphaned from the question it answered.

    Args:
        history (list[dict]): The conversation, without the system message.
        keep_recent (int): How many trailing messages to keep verbatim.

    Returns:
        tuple[list[dict], list[dict]]: The head to summarise and the tail to
            keep. The head is empty when there is nothing safe to drop, which is
            the caller's signal to leave history alone.
    """
    keep = max(0, keep_recent)
    if len(history) <= keep:
        return [], list(history)
    index = len(history) - keep
    while index < len(history) and history[index].get("role") != "user":
        index += 1
    if index >= len(history):
        # Everything after the boundary is assistant text; keeping nothing
        # verbatim is worse than keeping the lot, so decline to compact.
        return [], list(history)
    return list(history[:index]), list(history[index:])


def serialise_history(messages: list[dict[str, Any]]) -> str:
    """Render history as the plain transcript the summariser is shown."""
    if not messages:
        return "(nothing)"
    lines: list[str] = []
    for message in messages:
        role = str(message.get("role") or "unknown")
        content = message.get("content")
        if isinstance(content, list):
            text = " ".join(
                str(part.get("text") or "")
                for part in content
                if isinstance(part, dict) and part.get("type") != "image_url"
            )
        else:
            text = str(content or "")
        text = text.strip()
        if text:
            lines.append(f"{role}: {text}")
    return "\n".join(lines) or "(nothing)"


def build_summary_request(
    head: list[dict[str, Any]],
    *,
    previous: str | None = None,
    journal: str = "",
) -> list[dict[str, Any]]:
    """The messages sent to summarise a head of history.

    Args:
        head (list[dict]): The conversation being compacted.
        previous (str | None): An earlier summary to merge into, if any.
        journal (str): The current journal. Included so the summariser does not
            spend its output repeating facts that are already primary memory in
            this mode — the journal rides in every request regardless.

    Returns:
        list[dict]: A two-message request for the summariser model.
    """
    parts: list[str] = []
    if previous:
        parts.append(f"<earlier-summary>\n{previous}\n</earlier-summary>\n")
    if journal.strip():
        parts.append(
            "<journal>\n"
            "Already recorded elsewhere and sent with every request — do not "
            f"repeat these:\n{journal.strip()}\n</journal>\n"
        )
    parts.append(f"<conversation>\n{serialise_history(head)}\n</conversation>\n")
    parts.append(UPDATE_SUMMARY_INSTRUCTION if previous else "Summarise the above.")
    return [
        {"role": "system", "content": SUMMARY_SYSTEM_PROMPT},
        {"role": "user", "content": "\n".join(parts)},
    ]


__all__ = [
    "COMPACTION_SUMMARY_PREFIX",
    "COMPACTION_TRIGGER_RATIO",
    "DEFAULT_CONTEXT_WINDOW",
    "IMAGE_TOKEN_ALLOWANCE",
    "KEEP_RECENT_MESSAGES",
    "SUMMARY_SYSTEM_PROMPT",
    "CompactionOutcome",
    "build_summary_request",
    "context_window_for",
    "count_images",
    "count_tokens",
    "extract_previous_summary",
    "is_context_overflow",
    "serialise_history",
    "should_compact",
    "split_history",
    "summary_message",
    "text_projection",
]
