"""The per-LLM-call ledger row, and the plumbing that fills one in.

chiron's cost accounting used to be a single number per book: `BookRecord.cost_usd`,
read off :attr:`~chiron.models.litellm_model.LiteLLMModel.cumulative` once a run had
finished. That answers "what did this book cost" and nothing else — not which model,
not which upstream provider served the request, not how much of the bill was cache
reads versus completions, and not which of the run's forty calls was the expensive one.

This module defines :class:`LLMCallRecord`, the durable row written once per LLM call,
and the small extraction helpers that populate its provenance fields. Records are
emitted by :meth:`LiteLLMModel.record_usage` to whatever *sink* the caller installed —
the backend's per-book JSONL ledger in the app, a list in tests, nothing at all in the
default case, which is what keeps the core agent framework free of any storage concern.

**Route versus provider.** Two different questions, two fields. `route` is the API
chiron actually authenticated against and pays (`openrouter`, or `gemini` for a
direct Google AI Studio call); `provider` is the upstream that served the tokens,
which OpenRouter reports per request and which varies call to call as it
load-balances. Pricing follows `route`, because that is who bills you; attribution
follows `provider`, because that is who ran the model. The split is what keeps
Gemini-via-OpenRouter and Gemini-billed-direct as the different line items they are —
same model, same upstream, two separate invoices.
"""

from __future__ import annotations

import time
import uuid
from datetime import datetime, timezone
from typing import Any, Callable, Literal

from pydantic import BaseModel, Field

#: Why a call was made. `turn` is the agent loop asking the model what to do next;
#: `compaction` is history being summarised to stay inside the context window, in
#: either :class:`~chiron.core.context.ContextCompactor` or the non-live provider.
#: Compaction spend is invisible in a cumulative total but is real money, so it gets
#: its own kind rather than being folded into turns.
#:
#: The overlay's own kinds are here too, and their absence used to be a live bug: the
#: app passed `nonlive_qa` and `journal_sidecar` into a two-value Literal, so every
#: record raised on construction and died inside the `try/except` that wraps usage
#: accounting. Nothing was ever written, and nothing said so. A closed vocabulary is
#: still worth keeping — it is what stops a typo becoming a cost category nobody
#: reads — but it has to actually contain the kinds the application emits.
LLMCallKind = Literal[
    "turn",
    "compaction",
    "nonlive_qa",
    "nonlive_observer",
    "journal_sidecar",
    "live_turn",
]

#: `ok` billed normally. `retry` is an attempt that failed transiently and *was*
#: re-issued; `error` is one that failed terminally. Failed attempts usually carry no
#: usage, but recording them keeps the call table honest — a run that shows six calls
#: when the provider saw nine is a misleading trace, even when the extra three were free.
LLMCallStatus = Literal["ok", "retry", "error"]

#: Which rate table priced the call. `openrouter_live` is the live per-token-type
#: table (cache reads and writes priced separately); `google_static` is chiron's
#: hand-maintained Gemini table, which Google gives no live equivalent for;
#: `litellm_static` is litellm's bundled static cost map, read per token type
#: (:func:`~chiron.models.pricing.litellm_pricing`); `actual` means
#: the route reported its real charge and no estimate was needed; `unpriced` means
#: nothing could price it.
#:
#: `estimated` is the odd one out and deliberately distinct: the Live API never
#: reports usage in a shape chiron can bill from, so live-mode rows are arithmetic
#: — frames times a per-frame token figure, plus transcript length — rather than
#: measurement. Every surface that renders a total checks for this and prefixes a
#: `~`, because a number derived from a token estimate and a number a provider
#: charged should never look alike.
PricingSource = Literal[
    "openrouter_live",
    "google_static",
    "litellm_static",
    "actual",
    "estimated",
    "unpriced",
]

#: Where a finished record goes. Sync by design: it is called from
#: :meth:`LiteLLMModel.record_usage`, which is a plain method on the hot path of an
#: async loop, so a sink must be cheap (an append, a list push) and must not block.
UsageSink = Callable[["LLMCallRecord"], None]

#: Display names for provider slugs OpenRouter doesn't already capitalise for us.
_PROVIDER_LABELS: dict[str, str] = {
    "openai": "OpenAI",
    "anthropic": "Anthropic",
    "google": "Google",
    "google-vertex": "Google Vertex",
    "google-ai-studio": "Google AI Studio",
    # litellm's own slug for a direct AI Studio call, which is what `route` and
    # (absent an upstream report) `provider` both come out as for a `gemini/` id.
    "gemini": "Google AI Studio",
    "meta-llama": "Meta",
    "mistralai": "Mistral",
    "deepseek": "DeepSeek",
    "x-ai": "xAI",
    "qwen": "Qwen",
    "cohere": "Cohere",
    "deepinfra": "DeepInfra",
    "together": "Together",
    "fireworks": "Fireworks",
    "groq": "Groq",
    "openrouter": "OpenRouter",
    "azure": "Azure",
    "bedrock": "Bedrock",
    "perplexity": "Perplexity",
    "amazon": "Amazon",
    "nvidia": "NVIDIA",
    "moonshotai": "Moonshot AI",
}


def utc_now_iso() -> str:
    """The current UTC time as an ISO-8601 string, matching the rest of the store."""
    return datetime.now(timezone.utc).isoformat()


def new_call_id() -> str:
    """A short unique id for one ledger row."""
    return uuid.uuid4().hex[:16]


def new_run_id() -> str:
    """A short unique id grouping every call made by one agent run."""
    return uuid.uuid4().hex[:12]


def call_timer() -> Callable[[], int]:
    """Start a wall-clock timer; the returned callable reports elapsed milliseconds.

    Uses :func:`time.perf_counter`, so a clock adjustment mid-call can't produce a
    negative latency in the ledger.
    """
    started = time.perf_counter()
    return lambda: int((time.perf_counter() - started) * 1000)


def provider_label(slug: str | None) -> str:
    """A display name for a provider slug (`"openai"` → `"OpenAI"`).

    OpenRouter already sends nicely-cased names (`"OpenAI"`, `"Together"`) for the
    upstream it routed to, so anything that isn't all-lowercase is passed through
    untouched; only slugs derived from a model id need the lookup.

    Args:
        slug (str | None): A provider slug or display name, or None.

    Returns:
        str: The display name, or `"Unknown"` when the slug is missing.
    """
    if not slug:
        return "Unknown"
    if slug != slug.lower():
        return slug  # already a display name from the provider
    return _PROVIDER_LABELS.get(slug, slug.replace("-", " ").replace("_", " ").title())


def split_model_id(model_id: str) -> tuple[str, str, str]:
    """Split a litellm model id into `(route, vendor, model)`.

    litellm ids come in three shapes, and the vendor sits in a different slot in each::

        openrouter/openai/gpt-4o-mini  → ("openrouter", "openai",    "gpt-4o-mini")
        anthropic/claude-sonnet-4      → ("anthropic",  "anthropic", "claude-sonnet-4")
        gpt-4o-mini                    → ("",           "",          "gpt-4o-mini")

    The two-segment form is a *direct* provider call, where the route and the vendor are
    the same party — which is exactly the case the ledger is shaped to absorb when a
    direct credential is added alongside the OpenRouter one.

    Args:
        model_id (str): The litellm model identifier.

    Returns:
        tuple[str, str, str]: The route slug, vendor slug, and bare model name. The
            first two are empty strings for a bare model id with no prefix.
    """
    parts = [p for p in (model_id or "").split("/") if p]
    if len(parts) >= 3:
        return parts[0], parts[1], "/".join(parts[2:])
    if len(parts) == 2:
        return parts[0], parts[0], parts[1]
    return "", "", model_id or "unknown"


def _hidden(obj: Any, key: str) -> Any:
    """Read `key` out of a litellm response's `_hidden_params`, if present."""
    hidden = getattr(obj, "_hidden_params", None)
    if isinstance(hidden, dict):
        return hidden.get(key)
    return None


def extract_route(model_id: str, response: Any = None) -> str:
    """The API chiron called and is billed by, e.g. `"openrouter"`.

    Prefers litellm's own `custom_llm_provider` (authoritative — it is what litellm
    actually dispatched to) and falls back to the model id's prefix.

    Args:
        model_id (str): The litellm model identifier the call was made with.
        response (Any): The litellm response or stream chunk, when available.

    Returns:
        str: The route slug, or `"unknown"` when it cannot be determined.
    """
    from_litellm = _hidden(response, "custom_llm_provider")
    if isinstance(from_litellm, str) and from_litellm:
        return from_litellm
    route, _, _ = split_model_id(model_id)
    return route or "unknown"


def read_provider_field(obj: Any) -> str | None:
    """The upstream provider a litellm response or stream chunk reports, if any.

    OpenRouter puts the serving provider in a top-level `provider` field on both
    completions and stream chunks; litellm passes unknown fields through, so it turns up
    either as an attribute, in the pydantic extras, or in `_hidden_params`. All three
    are checked.

    Unlike :func:`extract_provider` this does *not* fall back to the model id, so a
    caller can tell "the route named a provider" apart from "we inferred one" — which
    matters when reading a streamed response, where only some chunks carry the field.

    Args:
        obj (Any): A litellm response or stream chunk.

    Returns:
        str | None: The reported provider name, or None when the object doesn't carry one.
    """
    extras = getattr(obj, "model_extra", None)
    for source in (
        getattr(obj, "provider", None),
        extras.get("provider") if isinstance(extras, dict) else None,
        _hidden(obj, "provider"),
    ):
        if isinstance(source, str) and source.strip():
            return source.strip()
    return None


def extract_provider(model_id: str, response: Any = None) -> str | None:
    """The upstream that actually served the request, falling back to the model's vendor.

    When neither the response nor a chunk reports a provider — a direct provider call,
    or an OpenRouter response predating the field — the vendor segment of the model id
    is the right answer anyway, since a direct call is served by the party you called.

    Args:
        model_id (str): The litellm model identifier the call was made with.
        response (Any): The litellm response or stream chunk, when available.

    Returns:
        str | None: The provider name, or None if neither the response nor the model id
            identifies one.
    """
    reported = read_provider_field(response)
    if reported:
        return reported
    _, vendor, _ = split_model_id(model_id)
    return vendor or None


class LLMCallRecord(BaseModel):
    """One LLM call, priced and attributed — the unit the cost dashboard is built on.

    Written once per call to an append-only per-book JSONL ledger. Every field is
    populated at the moment of the call rather than derived later, because the things
    that make a call expensive (which upstream served it, what the live rate was, how
    much of the prompt hit cache) are all knowable then and unrecoverable afterwards.

    Attributes:
        id (str): Unique id for this row.
        run_id (str | None): Groups every call made by one agent run.
        session_id (str | None): The gameplay session this call belongs to, so a
            row can be attributed to an evening of play after the fact. Set by
            whoever installs the sink, through `usage_labels`.
        book_id (str | None): The book being processed, when the run has one.
        agent_id (str | None): The settings-registry agent id (e.g. `ebook_loader`).
        kind (LLMCallKind): Whether this was an agent turn or a compaction summary.
        status (LLMCallStatus): Whether the call succeeded, was retried, or failed.
        turn (int | None): 1-based agent turn number, for `kind="turn"` calls.
        attempt (int): 1-based attempt number within that turn (>1 after a retry).
        started_at (str): ISO-8601 UTC timestamp of when the request was issued.
        duration_ms (int | None): Wall-clock time until the response settled.
        model_id (str): The litellm model id the call was made with.
        model (str): The bare model name, without route/vendor prefixes.
        route (str): The API that bills for this call (`openrouter`, `anthropic`…).
        provider (str | None): The upstream that served it, as reported by the route.
        streamed (bool): Whether a streaming completion was requested.
        finish_reason (str | None): The provider's stop reason.
        error (str | None): The failure message, for non-`ok` rows.
        prompt_tokens (int): Input tokens, cache reads included.
        completion_tokens (int): Generated output tokens.
        total_tokens (int): `prompt_tokens + completion_tokens`.
        cache_read_tokens (int): Input tokens served from the prompt cache.
        cache_write_tokens (int): Tokens written into the prompt cache.
        reasoning_tokens (int): Thinking tokens, when the provider reports them.
        cost_usd (float): The authoritative charge — the route's actual cost when it
            reported one, else the estimate.
        estimated_cost_usd (float): What the rate table predicted, always recorded even
            when an actual cost is known, so estimate drift is measurable.
        actual_cost_usd (float | None): The route's real charge, when it reports one.
        cost_by_type (dict[str, float]): USD split across `prompt` / `completion` /
            `cache_read` / `cache_write` / `reasoning` / `request`, reconciled to
            sum to `cost_usd` whenever an actual cost is known.
        pricing_source (PricingSource): Which rate table produced the figures.
    """

    id: str = Field(default_factory=new_call_id)
    run_id: str | None = None
    session_id: str | None = None
    book_id: str | None = None
    agent_id: str | None = None

    kind: LLMCallKind = "turn"
    status: LLMCallStatus = "ok"
    turn: int | None = None
    attempt: int = 1

    started_at: str = Field(default_factory=utc_now_iso)
    duration_ms: int | None = None

    model_id: str = ""
    model: str = ""
    route: str = "unknown"
    provider: str | None = None
    streamed: bool = False
    finish_reason: str | None = None
    error: str | None = None

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    reasoning_tokens: int = 0

    cost_usd: float = 0.0
    estimated_cost_usd: float = 0.0
    actual_cost_usd: float | None = None
    cost_by_type: dict[str, float] = Field(default_factory=dict)
    pricing_source: PricingSource = "unpriced"

    @property
    def provider_name(self) -> str:
        """The upstream provider as a display string."""
        return provider_label(self.provider)

    @property
    def date(self) -> str:
        """The `YYYY-MM-DD` this call was made on (UTC), for daily rollups."""
        return self.started_at[:10]


__all__ = [
    "LLMCallKind",
    "LLMCallRecord",
    "LLMCallStatus",
    "PricingSource",
    "UsageSink",
    "call_timer",
    "extract_provider",
    "extract_route",
    "new_call_id",
    "new_run_id",
    "provider_label",
    "read_provider_field",
    "split_model_id",
    "utc_now_iso",
]
