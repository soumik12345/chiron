"""litellm-backed model wrapper.

chiron talks to every model through litellm (OpenRouter by default), per the
project's locked architecture. This class is intentionally thin: it issues the
`acompletion` call and keeps cumulative token/cost accounting.

It is also the one place *every* LLM call in the app passes through — the agent
loop's turns and the context compactor's summarisation calls alike — which makes
:meth:`LiteLLMModel.record_usage` the natural (and only) instrumentation point for
per-call cost tracking. When a `usage_sink` is installed, each call emits a fully
priced :class:`~chiron.models.usage.LLMCallRecord` to it. With no sink installed
nothing changes and nothing is stored, so the core framework stays free of any
persistence concern; the backend supplies a sink that appends to a per-book ledger.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Callable

# Model metadata is resolved by Chiron's own refreshable catalogue. LiteLLM's
# import-time remote price-map fetch adds network I/O to otherwise offline paths;
# use its bundled map unless the embedding process made an explicit choice.
os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")

import litellm
import weave
from pydantic import BaseModel, Field

from chiron.models.google_pricing import get_pricing as google_pricing
from chiron.models.pricing import (
    ModelPricing,
    PricingTable,
    cost_model_candidates,
    litellm_pricing,
)
from chiron.models.prompt_cache import apply_prompt_caching, extract_cache_tokens
from chiron.models.providers import GOOGLE, provider_id_for_model
from chiron.models.usage import (
    LLMCallRecord,
    PricingSource,
    extract_provider,
    extract_route,
    split_model_id,
    utc_now_iso,
)

logger = logging.getLogger(__name__)


def _extract_token_counts(usage: Any) -> tuple[int, int]:
    """Return (prompt_tokens, completion_tokens) from a litellm usage object/dict."""
    if usage is None:
        return 0, 0
    if isinstance(usage, dict):
        return int(usage.get("prompt_tokens") or 0), int(
            usage.get("completion_tokens") or 0
        )
    return int(getattr(usage, "prompt_tokens", 0) or 0), int(
        getattr(usage, "completion_tokens", 0) or 0
    )


def _u(usage: Any, key: str) -> Any:
    """Extract a field from a usage object or dict, returning None if absent.

    Args:
        usage (Any): A litellm usage object, dict, or None.
        key (str): The attribute or key name to retrieve.

    Returns:
        Any: The field value, or None if usage is None or the key is missing.
    """
    if usage is None:
        return None
    if isinstance(usage, dict):
        return usage.get(key)
    return getattr(usage, key, None)


def _extract_reasoning_tokens(usage: Any) -> int:
    """Reasoning/thinking tokens, if the provider reports them (else 0)."""
    details = _u(usage, "completion_tokens_details")
    val = _u(details, "reasoning_tokens")
    return int(val or 0)


def extract_reasoning(message_or_delta: Any) -> tuple[str | None, list | None]:
    """Return `(reasoning_content, thinking_blocks)` from a message or stream delta.

    Reasoning models return their thinking alongside the visible answer. litellm
    normalises this to two fields: `reasoning_content` (plain text, output-only)
    and `thinking_blocks` (Anthropic's signed blocks, which **must** be sent back
    verbatim on the next request or the provider rejects the conversation).

    Which of the two you get depends on the route, not just the model: the native
    Anthropic route populates `thinking_blocks`, while the same model reached
    through OpenRouter returns `reasoning_content` only. Some routes tuck the
    blocks into `provider_specific_fields` instead, so that is checked too.

    Args:
        message_or_delta (Any): A litellm message or streaming delta object.

    Returns:
        tuple[str | None, list | None]: The reasoning text and thinking blocks, each
            None when the provider did not report it.
    """
    reasoning = _u(message_or_delta, "reasoning_content")
    blocks = _u(message_or_delta, "thinking_blocks")
    if blocks is None:
        extras = _u(message_or_delta, "provider_specific_fields")
        if isinstance(extras, dict):
            blocks = extras.get("thinking_blocks")
    if blocks is not None:
        blocks = [
            block if isinstance(block, dict) else dict(block)
            for block in blocks
            if block is not None
        ] or None
    return (reasoning or None), blocks


def _extract_actual_cost(usage: Any) -> float | None:
    """OpenRouter's real per-request USD cost when usage accounting is on, else None."""
    for key in ("cost", "total_cost"):
        val = _u(usage, key)
        if val is not None:
            try:
                return float(val)
            except (TypeError, ValueError):
                continue
    return None


class LiteLLMModel(BaseModel):
    """Async chat-completion wrapper with cumulative usage/cost tracking.

    Thin wrapper around litellm's `acompletion` that adds per-call token/cost
    accounting and Anthropic prompt-cache breakpoint injection. All mutable state
    is confined to the `cumulative` dict so the model object is safe to share
    across sub-agents.

    Attributes:
        model_id (str): The litellm model identifier (e.g. `openrouter/openai/gpt-4o-mini`).
        temperature (float): Sampling temperature passed to the provider. Defaults to 0.7.
        max_tokens (int | None): Maximum completion tokens; None lets the provider decide.
        api_base (str | None): Override the provider base URL (e.g. for local inference).
        api_key (str | None): Provider credential. None falls back to litellm's own
            environment lookup (`OPENROUTER_API_KEY` and friends), which is the
            path every call took before the settings page existed. Excluded from
            `repr` so the key doesn't surface in logs or tracebacks.
        timeout (int): Request timeout in seconds. Defaults to 600.
        enable_prompt_caching (bool): Whether to inject Anthropic cache breakpoints. Defaults to True.
        reasoning_effort (str | None): Request extended thinking from reasoning-capable
            models (`"minimal"`, `"low"`, `"medium"`, `"high"`). litellm maps
            this to each provider's native control (e.g. an Anthropic thinking budget).
            None leaves the provider default.
        cumulative (dict[str, float]): Accumulated token and cost counters across all calls.
        usage_sink (Callable | None): Receives one :class:`LLMCallRecord` per call, for
            durable cost tracking. None (the default) records nothing beyond
            `cumulative`, which is exactly how the model behaved before the cost
            dashboard existed. Excluded from serialisation — it is wiring, not config.
        usage_labels (dict[str, Any]): Run-level provenance stamped onto every emitted
            record (`run_id`, `book_id`, `agent_id`). Set once by whoever owns the
            run; per-call fields like the turn number are passed to
            :meth:`record_usage` instead, since they change call to call.
    """

    model_id: str
    temperature: float | None = 0.7
    max_tokens: int | None = None
    api_base: str | None = None
    api_key: str | None = Field(default=None, repr=False, exclude=True)
    timeout: int = 600
    enable_prompt_caching: bool = True
    reasoning_effort: str | None = None

    usage_sink: Callable[[LLMCallRecord], None] | None = Field(
        default=None, repr=False, exclude=True
    )
    usage_labels: dict[str, Any] = Field(default_factory=dict, repr=False, exclude=True)

    cumulative: dict[str, float] = Field(
        default_factory=lambda: {
            "input_tokens": 0.0,
            "output_tokens": 0.0,
            "total_tokens": 0.0,
            "cost_usd": 0.0,
            "cache_read_tokens": 0.0,
            "cache_write_tokens": 0.0,
            "reasoning_tokens": 0.0,
        }
    )

    @weave.op
    async def acompletion(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        stream: bool = False,
        tool_choice: str | dict[str, Any] | None = None,
        response_format: dict[str, Any] | None = None,
        extra_body: dict[str, Any] | None = None,
    ) -> Any:
        """Issue a (possibly streaming) chat completion via litellm.

        Applies Anthropic prompt-cache breakpoints before sending (a no-op for
        non-Anthropic providers). For OpenRouter models, requests real per-request
        cost accounting via `extra_body`.

        Args:
            messages (list[dict[str, Any]]): The conversation history in OpenAI
                Chat Completions format.
            tools (list[dict[str, Any]] | None): Tool schemas in OpenAI function-calling
                format. Defaults to None.
            stream (bool): Whether to request a streaming response. Defaults to False.
            tool_choice (str | dict | None): Provider tool-selection policy. When
                omitted, tool-enabled calls use ``auto``.
            response_format (dict | None): Provider response schema/format request.
            extra_body (dict | None): Provider-specific request options merged with
                Chiron's accounting options.

        Returns:
            Any: A litellm `ModelResponse` (non-streaming) or async generator
                (streaming).
        """
        # Mark cache breakpoints for Anthropic models (no-op otherwise). Operates
        # on copies, so persisted history / the tool router are never mutated.
        messages, tools = apply_prompt_caching(
            messages, tools, self.model_id, enabled=self.enable_prompt_caching
        )
        kwargs: dict[str, Any] = {
            "model": self.model_id,
            "messages": messages,
            "timeout": self.timeout,
            "stream": stream,
        }
        if self.temperature is not None:
            kwargs["temperature"] = self.temperature
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = tool_choice or "auto"
        if self.max_tokens is not None:
            kwargs["max_tokens"] = self.max_tokens
        if self.api_base:
            kwargs["api_base"] = self.api_base
        if self.api_key:
            kwargs["api_key"] = self.api_key
        if self.reasoning_effort:
            kwargs["reasoning_effort"] = self.reasoning_effort
        if response_format is not None:
            kwargs["response_format"] = response_format
        if stream:
            kwargs["stream_options"] = {"include_usage": True}
        provider_body = dict(extra_body or {})
        # Ask OpenRouter to include the real per-request cost in usage so we can
        # reconcile our pricing-table estimate against ground truth (no-op for
        # providers that ignore it).
        if "openrouter/" in self.model_id:
            usage = dict(provider_body.get("usage") or {})
            usage["include"] = True
            provider_body["usage"] = usage
        if provider_body:
            kwargs["extra_body"] = provider_body
        return await litellm.acompletion(**kwargs)

    def cost_for(self, prompt_tokens: int, completion_tokens: int) -> float:
        """Best-effort USD cost for a call; returns 0.0 if litellm can't price it.

        Tries the model id and provider-stripped fallbacks so OpenRouter-prefixed
        models (which litellm doesn't price directly) still report real spend.
        """
        for model in cost_model_candidates(self.model_id):
            try:
                prompt_cost, completion_cost = litellm.cost_per_token(
                    model=model,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                )
            except Exception:
                continue
            total = float(prompt_cost or 0.0) + float(completion_cost or 0.0)
            if total:
                return total
        return 0.0

    def _price_call(
        self,
        prompt_tokens: int,
        completion_tokens: int,
        cache_read: int,
        cache_write: int,
        reasoning: int,
    ) -> tuple[float, dict[str, float], str]:
        """Estimate (total_cost, cost_by_type, pricing_source) for one call.

        Rate tables are tried in descending order of fidelity, and which one answered
        is returned alongside the number rather than thrown away — a dashboard that
        cannot say *how* a figure was arrived at invites the reader to trust a flat
        litellm guess exactly as much as a live per-token-type rate.

        1. chiron's own Gemini table, for a direct `gemini/` call. Google publishes
           no machine-readable price list, so this is the hand-verified override for
           the model families it covers.
        2. The live OpenRouter table, which prices each token type separately (cache
           reads are far cheaper than fresh prompt tokens, cache writes dearer).
        3. litellm's static cost map, read in full (:func:`litellm_pricing`) — token
           types priced separately, same as the two above. Anything the hand table
           hasn't caught up with lands here rather than losing its cache rate: the map
           tracks new releases faster than a hardcoded dict can.
        4. `litellm.cost_per_token()`, flat prompt/completion. Last ditch, for ids
           the map has no entry for but litellm can still price dynamically.

        Each table is keyed on the route the model id names, so a provider added later
        slots in as a step of its own; nothing above this method assumes a single payer.

        Args:
            prompt_tokens (int): Number of non-cached input tokens.
            completion_tokens (int): Number of generated output tokens.
            cache_read (int): Number of cache-read tokens (billed at the cheaper rate).
            cache_write (int): Number of cache-creation tokens.
            reasoning (int): Number of reasoning/thinking tokens.

        Returns:
            tuple[float, dict[str, float], str]: The total estimate, the per-token-type
                cost split, and the :data:`~chiron.models.usage.PricingSource` that
                produced them.
        """
        pricing: ModelPricing | None = None
        source: PricingSource = "unpriced"
        if provider_id_for_model(self.model_id) == GOOGLE:
            pricing, source = google_pricing(self.model_id), "google_static"
        if pricing is None:
            pricing, source = (
                PricingTable.instance().get(self.model_id),
                "openrouter_live",
            )
        if pricing is None:
            pricing, source = litellm_pricing(self.model_id), "litellm_static"
        if pricing is not None:
            breakdown = pricing.cost_breakdown(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                cache_read_tokens=cache_read,
                cache_write_tokens=cache_write,
                reasoning_tokens=reasoning,
            )
            return sum(breakdown.values()), breakdown, source
        # Fallback: litellm flat pricing, attributed to prompt/completion.
        flat = self.cost_for(prompt_tokens, completion_tokens)
        source = "litellm_static" if flat else "unpriced"
        return flat, {"prompt": 0.0, "completion": flat, "request": 0.0}, source

    def record_usage(
        self,
        usage: Any,
        *,
        response: Any = None,
        context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Fold one call's usage into the cumulative totals; return that call's slice.

        Computes a per-token-type estimate from the OpenRouter pricing table and,
        when OpenRouter reports the real cost, reconciles the breakdown to it (the
        components are scaled so they sum to the actual charge). `cost_usd` is
        the authoritative figure (reconciled where known, else estimated).

        When a `usage_sink` is installed, this also emits one
        :class:`~chiron.models.usage.LLMCallRecord` — the durable ledger row behind
        the cost dashboard.

        Args:
            usage (Any): The provider's usage payload (object or dict), or None.
            response (Any): The full litellm response or a representative stream chunk.
                Carries the route and upstream-provider provenance that `usage` alone
                does not, so passing it is what makes per-provider attribution possible.
            context (dict[str, Any] | None): Per-call provenance for the emitted record —
                `kind`, `turn`, `attempt`, `duration_ms`, `streamed`,
                `finish_reason`, and an optional pre-extracted `provider` (needed for
                streaming, where the provider is read off a chunk rather than a response).

        Returns:
            dict[str, Any]: This call's token counts, costs, and cost split.
        """
        prompt_tokens, completion_tokens = _extract_token_counts(usage)
        cache_read, cache_write = extract_cache_tokens(usage)
        reasoning = _extract_reasoning_tokens(usage)
        actual_cost = _extract_actual_cost(usage)

        estimated, cost_by_type, pricing_source = self._price_call(
            prompt_tokens, completion_tokens, cache_read, cache_write, reasoning
        )

        # Reconcile: scale the per-type breakdown so it sums to OpenRouter's actual.
        if actual_cost is not None and estimated > 0:
            factor = actual_cost / estimated
            cost_by_type = {k: v * factor for k, v in cost_by_type.items()}
        elif actual_cost is not None and estimated == 0:
            cost_by_type = {**cost_by_type, "completion": actual_cost}
            pricing_source = "actual"

        cost = actual_cost if actual_cost is not None else estimated

        self.cumulative["input_tokens"] += prompt_tokens
        self.cumulative["output_tokens"] += completion_tokens
        self.cumulative["total_tokens"] += prompt_tokens + completion_tokens
        self.cumulative["cost_usd"] += cost
        self.cumulative["cache_read_tokens"] += cache_read
        self.cumulative["cache_write_tokens"] += cache_write
        self.cumulative["reasoning_tokens"] += reasoning

        slice_ = {
            "input_tokens": prompt_tokens,
            "output_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
            "cost_usd": cost,
            "estimated_cost_usd": estimated,
            "actual_cost_usd": actual_cost,
            "cache_read_tokens": cache_read,
            "cache_write_tokens": cache_write,
            "reasoning_tokens": reasoning,
            "cost_by_type": cost_by_type,
        }

        ctx = dict(context or {})
        self._emit(
            LLMCallRecord(
                **self._record_identity(response, ctx),
                kind=ctx.get("kind", "turn"),
                status=ctx.get("status", "ok"),
                turn=ctx.get("turn"),
                attempt=int(ctx.get("attempt", 1)),
                started_at=ctx.get("started_at") or utc_now_iso(),
                duration_ms=ctx.get("duration_ms"),
                streamed=bool(ctx.get("streamed", False)),
                finish_reason=ctx.get("finish_reason"),
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
                cache_read_tokens=cache_read,
                cache_write_tokens=cache_write,
                reasoning_tokens=reasoning,
                cost_usd=cost,
                estimated_cost_usd=estimated,
                actual_cost_usd=actual_cost,
                cost_by_type=cost_by_type,
                pricing_source=pricing_source,
            )
        )
        return slice_

    def record_failure(
        self, error: BaseException | str, *, context: dict[str, Any] | None = None
    ) -> None:
        """Emit a ledger row for a call that failed, with no usage to account for.

        A failed attempt usually costs nothing, so it never touches `cumulative`. It
        is still recorded, because a call table that silently omits the three attempts
        before a success misrepresents both what the provider saw and how long the run
        really took.

        Args:
            error (BaseException | str): The failure, rendered into the record's
                `error` field.
            context (dict[str, Any] | None): Same per-call provenance as
                :meth:`record_usage`. `status` defaults to `"error"`; pass
                `"retry"` for an attempt that is about to be re-issued.
        """
        ctx = dict(context or {})
        self._emit(
            LLMCallRecord(
                **self._record_identity(ctx.get("response"), ctx),
                kind=ctx.get("kind", "turn"),
                status=ctx.get("status", "error"),
                turn=ctx.get("turn"),
                attempt=int(ctx.get("attempt", 1)),
                started_at=ctx.get("started_at") or utc_now_iso(),
                duration_ms=ctx.get("duration_ms"),
                streamed=bool(ctx.get("streamed", False)),
                error=str(error)[:500] or error.__class__.__name__,
            )
        )

    def _record_identity(self, response: Any, ctx: dict[str, Any]) -> dict[str, Any]:
        """The run-level and model-level fields shared by every emitted record."""
        _, _, bare_model = split_model_id(self.model_id)
        return {
            "run_id": self.usage_labels.get("run_id"),
            "session_id": self.usage_labels.get("session_id"),
            "book_id": self.usage_labels.get("book_id"),
            "agent_id": self.usage_labels.get("agent_id"),
            "model_id": self.model_id,
            "model": bare_model,
            "route": extract_route(self.model_id, response),
            "provider": ctx.get("provider")
            or extract_provider(self.model_id, response),
        }

    def _emit(self, record: LLMCallRecord) -> None:
        """Hand a finished record to the sink, if one is installed.

        Deliberately swallows sink failures: cost tracking is observability, and a
        full disk or an unwritable ledger must never take down a book that is
        otherwise processing fine.
        """
        if self.usage_sink is None:
            return
        try:
            self.usage_sink(record)
        except Exception:  # noqa: BLE001 — never let bookkeeping break a run
            logger.warning("Usage sink rejected a call record", exc_info=True)
