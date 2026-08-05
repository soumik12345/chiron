"""Hand-maintained per-token rates for Google AI Studio (Gemini) models.

Every other rate in chiron is fetched: OpenRouter publishes live per-model pricing
at `/api/v1/models` and :mod:`chiron.models.pricing` reads it. Google has no
equivalent — its model-list API reports token limits, display names and supported
methods, and says nothing at all about money. So a direct `gemini/` call has two
possible rate sources: this table, or litellm's static cost map.

This table is consulted first, for one reason: litellm's map prices prompt and
completion tokens and nothing else, while Gemini's whole economic story for an agent
loop is **cached input**. The ebook loader replays a growing transcript on every
turn; at a 75%-off cached-input rate versus the full prompt rate, treating a cache
read as a fresh prompt token overstates a long run's cost several times over. A
table that knows the cache rate is worth maintaining by hand.

**Maintenance.** Rates below are USD per million tokens on the **paid tier**, for the
**standard context window** (Gemini charges a higher rate above 200k tokens for the
Pro models — a long-context run is therefore under-estimated, never over-). Figures
current as of July 2026; a model absent from this table isn't an error, it simply
falls through to litellm's static pricing, which tracks new releases faster than a
hardcoded dict can.

**Thinking tokens are deliberately not priced separately.** Google bills them at the
output rate, and litellm folds `thoughtsTokenCount` into `completion_tokens`
before chiron ever sees it (`vertex_and_google_ai_studio_gemini.py`), so a
non-zero `reasoning` rate here would bill the same tokens twice.
"""

from __future__ import annotations

from chiron.models.pricing import ModelPricing
from chiron.models.providers import strip_prefix

_M = 1_000_000

#: model family → (input, output, cached input) in USD per million tokens.
#: Longest key wins, so `gemini-2.5-flash-lite-preview-…` can't be priced as
#: `gemini-2.5-flash`. Where Google publishes no cached-input rate, the entry uses
#: its standard 75%-off discount rather than 0.0 — a free cache read is a claim, and
#: the wrong one.
_PER_MILLION: dict[str, tuple[float, float, float]] = {
    "gemini-3.6-flash": (1.50, 7.50, 0.15),
    "gemini-3-pro": (2.00, 12.00, 0.20),
    "gemini-2.5-pro": (1.25, 10.00, 0.31),
    "gemini-2.5-flash-lite": (0.10, 0.40, 0.025),
    "gemini-2.5-flash": (0.30, 2.50, 0.075),
    "gemini-2.0-flash-lite": (0.075, 0.30, 0.019),
    "gemini-2.0-flash": (0.10, 0.40, 0.025),
    "gemini-1.5-pro": (1.25, 5.00, 0.3125),
    "gemini-1.5-flash-8b": (0.0375, 0.15, 0.01),
    "gemini-1.5-flash": (0.075, 0.30, 0.019),
}

_FAMILIES: list[str] = sorted(_PER_MILLION, key=len, reverse=True)


def _family(model_id: str) -> str | None:
    """The table key covering `model_id`, matching the longest family prefix."""
    name = strip_prefix((model_id or "").strip()).removeprefix("models/").lower()
    return next((f for f in _FAMILIES if name.startswith(f)), None)


def get_pricing(model_id: str) -> ModelPricing | None:
    """Per-token rates for a Gemini model, or None if this table doesn't cover it.

    Args:
        model_id (str): A litellm id (`gemini/gemini-2.5-flash`), a bare model name,
            or Google's own `models/…` form.

    Returns:
        ModelPricing | None: Rates in USD per token, or None to fall through to
            litellm's static pricing.
    """
    family = _family(model_id)
    if family is None:
        return None
    prompt, completion, cached = _PER_MILLION[family]
    return ModelPricing(
        prompt=prompt / _M,
        completion=completion / _M,
        cache_read=cached / _M,
        # Google charges explicit-cache *storage* per hour, not per token written,
        # and reports no write count — so there is nothing to price here.
        cache_write=0.0,
        # Billed at the output rate and already inside completion_tokens; see module
        # docstring. Pricing it again would double-count every thinking token.
        reasoning=0.0,
    )


def is_priced(model_id: str) -> bool:
    """Whether this table covers `model_id` (used to label the picker honestly)."""
    return _family(model_id) is not None


__all__ = ["get_pricing", "is_priced"]
