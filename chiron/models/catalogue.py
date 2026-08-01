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

Everything here is best-effort: a failed fetch returns a stale cache if one exists
and an empty list otherwise, and the settings UI falls back to free-text entry.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from chiron.models.google_pricing import get_pricing as google_pricing
from chiron.models.pricing import litellm_pricing
from chiron.models.providers import GOOGLE, OPENROUTER, to_litellm_id

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
def _parse_google(entry: dict) -> ModelInfo | None:
    """Build a :class:`ModelInfo` from one raw ``v1beta/models`` entry, or None.

    Returns None for anything that can't hold a conversation — embedding models,
    answer-attribution models and the like — since a picker choosing the agent's
    brain should not offer them.
    """
    name = entry.get("name")
    if not isinstance(name, str) or not name:
        return None
    model_ref = name.removeprefix("models/")
    methods = entry.get("supportedGenerationMethods")
    if not isinstance(methods, list) or "generateContent" not in methods:
        return None

    litellm_id = to_litellm_id(model_ref, GOOGLE)
    # Same cascade the ledger prices a call with (see LiteLLMModel._price_call), so the
    # picker's label and the cost page can't disagree about whether a model has a rate.
    pricing = google_pricing(litellm_id) or litellm_pricing(litellm_id)
    context = entry.get("inputTokenLimit")
    lowered = model_ref.lower()
    return ModelInfo(
        id=litellm_id,
        provider=GOOGLE,
        provider_model_id=model_ref,
        name=entry.get("displayName") or model_ref,
        vendor="google",
        context_length=int(context) if isinstance(context, (int, float)) else None,
        prompt_price=pricing.prompt if pricing else 0.0,
        completion_price=pricing.completion if pricing else 0.0,
        pricing_known=pricing is not None,
        supports_tools=not any(marker in lowered for marker in _NO_TOOL_MARKERS),
    )


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
        list[ModelInfo]: Text-generation models sorted newest-looking first; empty
            when there is no key, or the fetch failed with no usable cache.
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

    models = [m for m in (_parse_google(e) for e in raw) if m]
    if not models:
        return _read_cache(cache_path, ttl=2**31) or []

    # Google returns models in no useful order and its ids sort newest-last
    # ("1.5" before "2.5"), so reverse the natural sort to surface current families.
    models.sort(key=lambda m: m.provider_model_id.lower(), reverse=True)
    _write_cache(cache_path, models)
    return models


__all__ = ["ModelInfo", "list_google_models", "list_models", "to_litellm_id"]
