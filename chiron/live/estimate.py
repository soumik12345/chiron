"""What the Live Observer costs, by arithmetic rather than measurement.

Every Responder call goes through litellm, which reports token counts and—on
OpenRouter—the real charge. The Live API reports no equivalent per-checkpoint
invoice; its ``usage_metadata`` is a running context total.

So live-mode rows are **estimates**, and this module is the whole of the estimate:

* frames sent, times the per-frame token cost implied by the selected frame
  detail (the same figure the settings page's eviction estimate uses);
* checkpoint instructions and journal seed/rotation traffic at characters ÷ 4;
* discarded 24 kHz, 16-bit PCM output at roughly 25 audio tokens/second;
* all priced against a small hand-kept table of Live API rates.

Every record it produces carries ``pricing_source="estimated"``, which is what
makes the ``~`` in front of the footer's total honest rather than decorative. The
arithmetic is stated here plainly so a number that looks wrong can be checked
rather than trusted: it is a guide to whether an evening cost cents or dollars,
not a figure to reconcile a bill against.
"""

from __future__ import annotations

from typing import Any

from chiron.models.pricing import ModelPricing
from chiron.models.usage import (
    LLMCallRecord,
    call_timer,
    split_model_id,
    utc_now_iso,
)

#: Tokens one video frame costs at each frame-detail setting. ``low`` is the
#: documented figure for the Live API; the other two are that figure scaled by
#: the tile counts the higher resolutions imply, and are the rougher end of an
#: already-rough estimate.
TOKENS_PER_FRAME: dict[str, int] = {"low": 260, "medium": 560, "high": 1120}

#: Characters per token for the text sides of the estimate.
CHARS_PER_TOKEN = 4

#: Live output is raw mono 24 kHz, 16-bit PCM. Google's Live guidance says audio
#: accumulates at about 25 tokens/second, so byte duration gives us a better
#: estimate than treating an entire audio chunk as a few characters of text.
AUDIO_OUTPUT_BYTES_PER_SECOND = 24_000 * 2
AUDIO_TOKENS_PER_SECOND = 25

_M = 1_000_000

#: Live API families → (input $/M, output $/M). Native-audio models bill audio
#: output at a much higher rate than text, and Chiron reads a transcript of
#: speech it never plays — so the audio rate is the one that applies, however
#: text-shaped the result looks in the overlay. Hand-maintained, like
#: :mod:`chiron.models.google_pricing`, because Google publishes no machine
#: readable price list.
_LIVE_PER_MILLION: dict[str, tuple[float, float]] = {
    "gemini-3.1-flash-live": (0.50, 12.00),
    "gemini-2.5-flash-native-audio": (0.50, 12.00),
    "gemini-2.5-flash-live": (0.50, 12.00),
    "gemini-2.0-flash-live": (0.35, 8.50),
}

_FAMILIES: list[str] = sorted(_LIVE_PER_MILLION, key=len, reverse=True)

#: Used when the selected Live model matches no family above. A new preview id
#: appears every few months and pricing it as the current flash tier is a far
#: better answer than pricing it at zero, which would read as "free".
_FALLBACK = (0.50, 12.00)


def tokens_per_frame(media_resolution: str) -> int:
    """The per-frame token cost implied by a frame-detail setting."""
    return TOKENS_PER_FRAME.get(media_resolution, TOKENS_PER_FRAME["low"])


def live_pricing(model_id: str) -> ModelPricing:
    """Per-token rates for a Live API model, falling back to the flash tier."""
    name = (model_id or "").strip().lower().removeprefix("live/")
    family = next((f for f in _FAMILIES if name.startswith(f)), None)
    prompt, completion = _LIVE_PER_MILLION.get(family or "", _FALLBACK)
    return ModelPricing(prompt=prompt / _M, completion=completion / _M)


def text_tokens(text: str) -> int:
    """Characters ÷ 4, the same crude estimate used everywhere else offline."""
    if not text:
        return 0
    return max(1, (len(text) + CHARS_PER_TOKEN - 1) // CHARS_PER_TOKEN)


def audio_output_tokens(byte_count: int) -> int:
    """Estimate Live audio tokens from raw 24 kHz, 16-bit PCM bytes."""
    size = max(0, int(byte_count))
    if not size:
        return 0
    numerator = size * AUDIO_TOKENS_PER_SECOND
    return max(
        1,
        (numerator + AUDIO_OUTPUT_BYTES_PER_SECOND - 1)
        // AUDIO_OUTPUT_BYTES_PER_SECOND,
    )


def estimate_turn(
    *,
    model_id: str,
    frames: int,
    media_resolution: str,
    prompt_text: str = "",
    output_text: str = "",
    output_tokens: int = 0,
    session_id: str | None = None,
    started_at: str | None = None,
    duration_ms: int | None = None,
    agent_id: str | None = None,
    kind: str = "observer_checkpoint",
) -> LLMCallRecord:
    """Price one Observer checkpoint or context write, as an estimate.

    Args:
        model_id (str): The Live model id, with or without the ``live/`` prefix.
        frames (int): Frames streamed since the previous turn.
        media_resolution (str): The frame-detail setting in force.
        prompt_text (str): Text the player sent since the previous turn.
        output_text (str): Tool/content text produced by the model.
        output_tokens (int): Additional estimated output tokens, notably discarded
            native audio calculated from its PCM byte duration.
        session_id (str | None): The gameplay session to bill this to.
        started_at (str | None): ISO-8601 UTC start; defaults to now.
        duration_ms (int | None): Wall-clock duration, when known.

    Returns:
        LLMCallRecord: A ledger row tagged ``pricing_source="estimated"`` and
            ``kind="observer_checkpoint"``, so no surface can render it as
            measured.
    """
    prompt_tokens = frames * tokens_per_frame(media_resolution) + text_tokens(
        prompt_text
    )
    completion_tokens = text_tokens(output_text) + max(0, int(output_tokens))
    pricing = live_pricing(model_id)
    breakdown = pricing.cost_breakdown(
        prompt_tokens=prompt_tokens, completion_tokens=completion_tokens
    )
    _, _, bare = split_model_id(model_id)
    return LLMCallRecord(
        session_id=session_id,
        agent_id=agent_id,
        kind=kind,
        model_id=model_id,
        model=bare,
        route="live",
        provider="google-ai-studio",
        streamed=True,
        started_at=started_at or utc_now_iso(),
        duration_ms=duration_ms,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=prompt_tokens + completion_tokens,
        cost_usd=sum(breakdown.values()),
        estimated_cost_usd=sum(breakdown.values()),
        cost_by_type=breakdown,
        pricing_source="estimated",
    )


def estimate_context_tokens(
    *, frames: int, media_resolution: str, transcript_chars: int
) -> int:
    """The client-side estimate of how full a live context window is.

    The same arithmetic that already powers the settings page's
    "how far back can it see" figure, in one place so the compaction trigger and
    the settings estimate cannot drift apart. Used when the SDK surfaces no
    ``usage_metadata``, which is most of the time.
    """
    return (
        frames * tokens_per_frame(media_resolution)
        + transcript_chars // CHARS_PER_TOKEN
    )


def read_usage_metadata(message: Any) -> int | None:
    """The server's own context token count, when a message carries one.

    Live server messages have carried ``usage_metadata`` for a while and the
    session loop has always ignored it. Trusting it when present and falling back
    to the estimate otherwise is strictly better than either alone: the server
    knows what it is actually holding, and it does not always say.
    """
    usage = getattr(message, "usage_metadata", None)
    if usage is None:
        return None
    for name in ("total_token_count", "totalTokenCount"):
        value = getattr(usage, name, None)
        if value is None and isinstance(usage, dict):
            value = usage.get(name)
        try:
            if value is not None and int(value) > 0:
                return int(value)
        except (TypeError, ValueError):
            continue
    return None


__all__ = [
    "AUDIO_OUTPUT_BYTES_PER_SECOND",
    "AUDIO_TOKENS_PER_SECOND",
    "CHARS_PER_TOKEN",
    "TOKENS_PER_FRAME",
    "audio_output_tokens",
    "call_timer",
    "estimate_context_tokens",
    "estimate_turn",
    "live_pricing",
    "read_usage_metadata",
    "text_tokens",
    "tokens_per_frame",
]
