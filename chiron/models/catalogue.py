"""The model lists the settings UI's picker offers, one fetcher per provider.

Deliberately separate from :mod:`chiron.models.pricing`, which reads OpenRouter's
same ``/api/v1/models`` endpoint: that module caches only the per-token *rates* keyed
by model id and drops everything else, because pricing a call is all it is for. A
picker needs the parts it throws away — display name, context length, tool support —
so this module keeps its own 24h disk caches of the fuller shape. Two caches rather
than one shared cache means a stale settings page can never invalidate live pricing,
and neither module has to grow a field for the other's benefit.

**The two providers are not symmetrical, and the shape reflects it.** OpenRouter's
catalogue is public, prices every model, and states which accept a ``tools``
parameter. Google's needs an API key, prices nothing (see
:mod:`chiron.models.google_pricing`), and reports capability only as the list of
generation methods a model supports — so tool support there is inferred from the
model family, and a Gemini model's price comes from chiron's own table or is marked
unknown. :class:`ModelInfo` therefore carries ``pricing_known``: a picker that prints
"Free" for a model it simply has no rate for is worse than one that prints nothing.

**One Google fetch populates two providers.** ``supportedGenerationMethods`` is what
separates a Live API model (``bidiGenerateContent``) from an ordinary one
(``generateContent``), and a model may support both — so :func:`list_google_models`
can emit two entries for one model, one per mode, with ids that differ by prefix
(``live/…`` versus ``gemini/…``). That is the whole of Chiron's mode selection: the
picker lists both kinds and picking one decides which session provider runs.

Everything here is best-effort: a failed fetch returns a stale cache if one exists
and a small static list otherwise, so the picker is never empty and the settings UI
still accepts a typed id for anything a catalogue has not caught up with.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path

from chiron.models.google_pricing import get_pricing as google_pricing
from chiron.models.pricing import litellm_pricing
from chiron.models.providers import GOOGLE, LIVE, OPENROUTER, to_litellm_id

logger = logging.getLogger(__name__)

_CACHE_DIR = Path.home() / ".cache" / "chiron"
_OPENROUTER_URL = "https://openrouter.ai/api/v1/models"
_OPENROUTER_CACHE = _CACHE_DIR / "openrouter_catalogue.json"
_GOOGLE_URL = "https://generativelanguage.googleapis.com/v1beta/models"
_GOOGLE_CACHE = _CACHE_DIR / "google_catalogue.json"
_DEFAULT_TTL_SECONDS = 24 * 60 * 60
_FETCH_TIMEOUT = 10.0

#: Google's list endpoint pages; ask for the maximum and follow the token defensively
#: rather than trusting one page to hold every model.
_GOOGLE_PAGE_SIZE = 1000
_GOOGLE_MAX_PAGES = 5

#: Substrings marking a Gemini-family model that cannot call tools. Google's list API
#: has no function-calling flag — only ``supportedGenerationMethods``, which says
#: ``generateContent`` for image, speech and Gemma models alike. Everything that
#: generates text on AI Studio supports function calling *except* these families, so
#: the inference is a denylist. Wrong in the safe direction: a mislabelled model still
#: appears in the picker, just with a "no tools" caveat the user can override.
_NO_TOOL_MARKERS = ("gemma", "embedding", "aqa", "imagen", "veo", "tts", "image")

#: Google's generation methods, by the mode each implies.
_LIVE_METHOD = "bidiGenerateContent"
_CHAT_METHOD = "generateContent"


@dataclass(frozen=True)
class ModelInfo:
    """One entry in the picker.

    Attributes:
        id (str): The litellm model id (``openrouter/openai/gpt-4o``,
            ``gemini/gemini-2.5-flash``) — the value stored in settings and handed to
            the agent, and the thing whose prefix decides which key authenticates it.
        provider (str): Which provider serves it (``openrouter`` / ``google``).
        provider_model_id (str): The bare id at that provider (``openai/gpt-4o``).
        name (str): Display name, e.g. "OpenAI: GPT-4o".
        vendor (str): The lab behind the model, used to group the picker.
        context_length (int | None): Context window in tokens, when reported.
        prompt_price (float): USD per prompt token. Meaningless unless
            ``pricing_known``.
        completion_price (float): USD per completion token.
        pricing_known (bool): Whether the two prices above are real. False for a
            Gemini model absent from chiron's rate table — where 0.0 means "no idea",
            not "free".
        supports_tools (bool): Whether the model can call tools. Reported by
            OpenRouter; inferred from the family for Google. The ebook loader is
            useless without it, so the picker filters on this by default.
        supports_vision (bool): Whether the model accepts images. Read from
            OpenRouter's declared input modalities; **assumed True for Google**,
            whose model list reports no modalities at all. Defaults to True for the
            same reason: this flag is only ever used to *warn* (the literary research
            agent looks at illustrations), and warning on an absence of evidence would
            cry wolf on every model a catalogue happens not to describe.
        is_live (bool): Whether this entry runs over the Live API — a persistent
            websocket — rather than a request/response endpoint. Carried
            explicitly rather than derived from the prefix because it is the one
            fact the rest of the app branches on.
    """

    id: str
    provider: str
    provider_model_id: str
    name: str
    vendor: str
    context_length: int | None
    prompt_price: float
    completion_price: float
    pricing_known: bool
    supports_tools: bool
    supports_vision: bool = True
    is_live: bool = False

    def label(self) -> str:
        """The picker's one-line description of this model.

        ``provider · id · live/non-live · $in/$out per M tokens``, with the price
        omitted rather than guessed when nothing knows the rate: a picker that
        prints "$0.00" for a model it simply has no rate for is worse than one
        that says nothing.
        """
        mode = "live" if self.is_live else "non-live"
        parts = [_PROVIDER_LABELS.get(self.provider, self.provider), self.id, mode]
        if self.pricing_known:
            parts.append(
                f"${self.prompt_price * 1_000_000:.2f}/"
                f"${self.completion_price * 1_000_000:.2f} per M"
            )
        else:
            parts.append("pricing unavailable")
        return "  ·  ".join(parts)


#: Short provider names for :meth:`ModelInfo.label`.
_PROVIDER_LABELS = {
    LIVE: "Live API",
    GOOGLE: "AI Studio",
    OPENROUTER: "OpenRouter",
}


def _f(value: object) -> float:
    try:
        return float(value)  # OpenRouter sends rates as strings
    except (TypeError, ValueError):
        return 0.0


# --------------------------------------------------------------------------- #
# Disk cache (shared by both fetchers, one file each)
# --------------------------------------------------------------------------- #
def _read_cache(cache_path: Path, ttl: int) -> list[ModelInfo] | None:
    """Return cached models if the cache exists and is within ``ttl``, else None."""
    try:
        if not cache_path.exists():
            return None
        data = json.loads(cache_path.read_text(encoding="utf-8"))
        if time.time() - float(data.get("fetched_at") or 0) > ttl:
            return None
        return [ModelInfo(**m) for m in data.get("models") or []]
    except Exception:  # noqa: BLE001 — a corrupt or outdated-shape cache is a miss
        logger.warning("Could not read model catalogue cache", exc_info=True)
        return None


def _write_cache(cache_path: Path, models: list[ModelInfo]) -> None:
    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(
            json.dumps(
                {"fetched_at": time.time(), "models": [asdict(m) for m in models]}
            ),
            encoding="utf-8",
        )
    except Exception:  # noqa: BLE001
        logger.warning("Could not write model catalogue cache", exc_info=True)


# --------------------------------------------------------------------------- #
# OpenRouter
# --------------------------------------------------------------------------- #
def _parse_openrouter(entry: dict) -> ModelInfo | None:
    """Build a :class:`ModelInfo` from one raw ``/api/v1/models`` entry, or None."""
    model_ref = entry.get("id")
    if not isinstance(model_ref, str) or not model_ref:
        return None
    pricing = entry.get("pricing") if isinstance(entry.get("pricing"), dict) else {}
    context = entry.get("context_length")
    params = entry.get("supported_parameters")
    architecture = (
        entry.get("architecture") if isinstance(entry.get("architecture"), dict) else {}
    )
    modalities = architecture.get("input_modalities")
    return ModelInfo(
        id=to_litellm_id(model_ref, OPENROUTER),
        provider=OPENROUTER,
        provider_model_id=model_ref,
        name=entry.get("name") or model_ref,
        vendor=model_ref.split("/", 1)[0] if "/" in model_ref else "other",
        context_length=int(context) if isinstance(context, (int, float)) else None,
        prompt_price=_f(pricing.get("prompt")),
        completion_price=_f(pricing.get("completion")),
        pricing_known=True,  # OpenRouter prices everything, free models included
        # OpenRouter reports capability as the parameters a model accepts; "tools"
        # in that list is the closest thing to a function-calling flag it exposes.
        supports_tools=isinstance(params, list) and "tools" in params,
        # Only a *declared* text-only model is reported as sightless. An entry that
        # lists no modalities says nothing, and is left at the permissive default.
        supports_vision=(
            "image" in modalities
            if isinstance(modalities, list) and modalities
            else True
        ),
    )


def list_models(
    *,
    force: bool = False,
    cache_path: Path = _OPENROUTER_CACHE,
    ttl_seconds: int = _DEFAULT_TTL_SECONDS,
) -> list[ModelInfo]:
    """Every model OpenRouter currently serves, newest cache first.

    Makes a blocking HTTP request on a cache miss, so call it from a thread
    (``run_in_threadpool``) rather than directly on the event loop.

    Args:
        force (bool): Skip the disk cache and re-fetch.
        cache_path (Path): Where the JSON disk cache lives.
        ttl_seconds (int): Seconds before the cache is considered stale.

    Returns:
        list[ModelInfo]: Models sorted by vendor then name; empty if the catalogue
            could not be fetched and no usable cache exists.
    """
    if not force:
        cached = _read_cache(cache_path, ttl_seconds)
        if cached:
            return cached

    try:
        import httpx

        response = httpx.get(_OPENROUTER_URL, timeout=_FETCH_TIMEOUT)
        response.raise_for_status()
        raw = response.json().get("data") or []
    except Exception:  # noqa: BLE001 — offline is a normal state for a local tool
        logger.warning("Could not fetch the OpenRouter model catalogue", exc_info=True)
        # A stale cache still beats an empty picker.
        return _read_cache(cache_path, ttl=2**31) or []

    models = [
        m for m in (_parse_openrouter(e) for e in raw if isinstance(e, dict)) if m
    ]
    if not models:
        return _read_cache(cache_path, ttl=2**31) or []

    models.sort(key=lambda m: (m.vendor.lower(), m.name.lower()))
    _write_cache(cache_path, models)
    return models


# --------------------------------------------------------------------------- #
# Google AI Studio
# --------------------------------------------------------------------------- #
def _parse_google(entry: dict) -> list[ModelInfo]:
    """Build the :class:`ModelInfo` entries one raw ``v1beta/models`` entry implies.

    Usually one, occasionally two: a model that serves both ``generateContent``
    and ``bidiGenerateContent`` is genuinely two choices — the same weights
    reached through a websocket or through a completions call, with different
    costs and a different session provider behind each — so the picker shows
    both rather than silently deciding for the player.

    Returns an empty list for anything that can't hold a conversation at all —
    embedding models, answer-attribution models and the like — since a picker
    choosing what Chiron thinks with should not offer them.
    """
    name = entry.get("name")
    if not isinstance(name, str) or not name:
        return []
    model_ref = name.removeprefix("models/")
    methods = entry.get("supportedGenerationMethods")
    if not isinstance(methods, list):
        return []

    litellm_id = to_litellm_id(model_ref, GOOGLE)
    # Same cascade the ledger prices a call with (see LiteLLMModel._price_call), so the
    # picker's label and the cost page can't disagree about whether a model has a rate.
    pricing = google_pricing(litellm_id) or litellm_pricing(litellm_id)
    context = entry.get("inputTokenLimit")
    lowered = model_ref.lower()
    common = {
        "provider_model_id": model_ref,
        "name": entry.get("displayName") or model_ref,
        "vendor": "google",
        "context_length": int(context) if isinstance(context, (int, float)) else None,
        "prompt_price": pricing.prompt if pricing else 0.0,
        "completion_price": pricing.completion if pricing else 0.0,
        "pricing_known": pricing is not None,
        "supports_tools": not any(marker in lowered for marker in _NO_TOOL_MARKERS),
    }

    models: list[ModelInfo] = []
    if _CHAT_METHOD in methods:
        models.append(ModelInfo(id=litellm_id, provider=GOOGLE, **common))
    if _LIVE_METHOD in methods:
        models.append(
            ModelInfo(
                id=to_litellm_id(model_ref, LIVE),
                provider=LIVE,
                is_live=True,
                **common,
            )
        )
    return models


def list_google_models(
    api_key: str | None,
    *,
    force: bool = False,
    cache_path: Path = _GOOGLE_CACHE,
    ttl_seconds: int = _DEFAULT_TTL_SECONDS,
) -> list[ModelInfo]:
    """Every Gemini model the given key can call, newest cache first.

    Unlike OpenRouter's, this catalogue is behind authentication, so **no key means
    no models** — not even a cached list. That is the honest answer: without a key
    nothing here is callable anyway, and the settings page says so rather than
    offering models that would fail on the first request.

    Blocking HTTP on a cache miss; call it from a thread.

    Args:
        api_key (str | None): The Google AI Studio key to list with.
        force (bool): Skip the disk cache and re-fetch.
        cache_path (Path): Where the JSON disk cache lives.
        ttl_seconds (int): Seconds before the cache is considered stale.

    Returns:
        list[ModelInfo]: Conversational models sorted newest-looking first, with
            live and non-live entries interleaved; empty when there is no key, or
            the fetch failed with no usable cache.
    """
    if not api_key:
        return []
    if not force:
        cached = _read_cache(cache_path, ttl_seconds)
        if cached:
            return cached

    raw: list[dict] = []
    try:
        import httpx

        page_token: str | None = None
        with httpx.Client(timeout=_FETCH_TIMEOUT) as http:
            for _ in range(_GOOGLE_MAX_PAGES):
                params = {"pageSize": _GOOGLE_PAGE_SIZE}
                if page_token:
                    params["pageToken"] = page_token
                response = http.get(
                    _GOOGLE_URL, params=params, headers={"x-goog-api-key": api_key}
                )
                response.raise_for_status()
                payload = response.json()
                raw.extend(
                    m for m in payload.get("models") or [] if isinstance(m, dict)
                )
                page_token = payload.get("nextPageToken") or None
                if not page_token:
                    break
    except Exception:  # noqa: BLE001 — offline, or a key that can't list
        logger.warning("Could not fetch the Google model catalogue", exc_info=True)
        return _read_cache(cache_path, ttl=2**31) or []

    models = [m for entry in raw for m in _parse_google(entry)]
    if not models:
        return _read_cache(cache_path, ttl=2**31) or []

    # Google returns models in no useful order and its ids sort newest-last
    # ("1.5" before "2.5"), so reverse the natural sort to surface current families.
    models.sort(key=lambda m: (m.provider_model_id.lower(), m.provider), reverse=True)
    _write_cache(cache_path, models)
    return models


# --------------------------------------------------------------------------- #
# The picker's combined view
# --------------------------------------------------------------------------- #
#: Enough of a catalogue to choose from with no network and no cache. Deliberately
#: tiny — it exists so a first run offline still offers something sane, not so it
#: can stand in for the real lists.
STATIC_MODELS: list[ModelInfo] = [
    ModelInfo(
        id="live/gemini-3.1-flash-live-preview",
        provider=LIVE,
        provider_model_id="gemini-3.1-flash-live-preview",
        name="Gemini 3.1 Flash (Live)",
        vendor="google",
        context_length=128_000,
        prompt_price=0.0,
        completion_price=0.0,
        pricing_known=False,
        supports_tools=True,
        is_live=True,
    ),
    ModelInfo(
        id="gemini/gemini-3.6-flash",
        provider=GOOGLE,
        provider_model_id="gemini-3.6-flash",
        name="Gemini 3.6 Flash",
        vendor="google",
        context_length=1_000_000,
        prompt_price=0.0,
        completion_price=0.0,
        pricing_known=False,
        supports_tools=True,
    ),
    ModelInfo(
        id="gemini/gemini-2.5-flash",
        provider=GOOGLE,
        provider_model_id="gemini-2.5-flash",
        name="Gemini 2.5 Flash",
        vendor="google",
        context_length=1_000_000,
        prompt_price=0.30 / 1_000_000,
        completion_price=2.50 / 1_000_000,
        pricing_known=True,
        supports_tools=True,
    ),
    ModelInfo(
        id="openrouter/google/gemini-2.5-flash",
        provider=OPENROUTER,
        provider_model_id="google/gemini-2.5-flash",
        name="Google: Gemini 2.5 Flash",
        vendor="google",
        context_length=1_000_000,
        prompt_price=0.30 / 1_000_000,
        completion_price=2.50 / 1_000_000,
        pricing_known=True,
        supports_tools=True,
    ),
]


def available_models(
    *,
    google_key: str = "",
    openrouter_key: str = "",
    force: bool = False,
) -> list[ModelInfo]:
    """Every model the configured keys can actually reach, for the picker.

    **A provider is activated by supplying its key.** No key, no entries — which
    is the honest answer, since without one nothing from that provider is
    callable. Google's key covers both the Live API and AI Studio, so one fetch
    populates both.

    Non-live entries are filtered to image-capable models: a Chiron that cannot
    see the screen is not a Chiron.

    Blocking HTTP on a cache miss; call it from a thread.

    Args:
        google_key (str): Gemini key, for the Live API and AI Studio entries.
        openrouter_key (str): OpenRouter key.
        force (bool): Skip the disk caches and re-fetch.

    Returns:
        list[ModelInfo]: Live entries first (they are what Chiron was built
            around), then non-live ones. Falls back to :data:`STATIC_MODELS`
            when no provider yielded anything at all.
    """
    models: list[ModelInfo] = list(
        list_google_models(google_key, force=force) if google_key else []
    )
    if openrouter_key:
        models.extend(
            m for m in list_models(force=force) if m.supports_vision and not m.is_live
        )
    if not models:
        return list(STATIC_MODELS)
    models.sort(key=lambda m: (not m.is_live, m.provider, m.vendor.lower(), m.id))
    return models


def find_model(models: list[ModelInfo], model_id: str) -> ModelInfo | None:
    """The entry for `model_id`, or None when the catalogue has never heard of it."""
    target = (model_id or "").strip()
    return next((m for m in models if m.id == target), None)


def describe_unknown(model_id: str) -> ModelInfo:
    """A placeholder entry for an id typed in by hand.

    The picker has to be able to show a selection it did not supply — an id from
    an older settings file, or a model released this morning — without pretending
    to know anything about it.
    """
    from chiron.models.providers import provider_for_model

    provider = provider_for_model(model_id)
    return replace(
        STATIC_MODELS[0],
        id=model_id,
        provider=provider.id if provider else "custom",
        provider_model_id=model_id,
        name=model_id,
        vendor="custom",
        context_length=None,
        pricing_known=False,
        is_live=bool(provider and provider.is_live),
    )


__all__ = [
    "STATIC_MODELS",
    "ModelInfo",
    "available_models",
    "describe_unknown",
    "find_model",
    "list_google_models",
    "list_models",
    "to_litellm_id",
]
