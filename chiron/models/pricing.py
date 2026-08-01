"""Rate tables: OpenRouter's live pricing, and litellm's static cost map.

Two sources of per-token rates live here, both returning the same
:class:`ModelPricing`:

* :class:`PricingTable` fetches OpenRouter's live per-model pricing from
  `/api/v1/models` (USD per token for `prompt`/`completion`, per request for
  `request`, plus cache read/write and reasoning rates), cached to disk with a TTL
  (default 24h) so normal runs pay no network cost. Pure in-memory lookups never
  block the async hot path; warm it explicitly at startup with
  :meth:`PricingTable.warm`.
* :func:`litellm_pricing` reads litellm's bundled static cost map, which covers
  models OpenRouter never sees — every direct `gemini/` call, for one.

**Read the whole map, not just two fields of it.** `litellm.cost_per_token()`
answers with prompt and completion cost only, so pricing a call through it silently
values every cache-read token at zero. For an agent loop that replays a growing
transcript on each turn, cache reads are most of the tokens — the ebook loader's
long runs were under-billed by exactly that amount. The map itself has carried
`cache_read_input_token_cost` for a while; :func:`litellm_pricing` reads it, along
with the cache-write rate, so the fallback prices each token type at its own rate
like the other two tables do.

Everything here is best-effort: an unavailable table returns `None` and the caller
falls through to the next one.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

_MODELS_URL = "https://openrouter.ai/api/v1/models"
_CACHE_PATH = Path.home() / ".cache" / "chiron" / "openrouter_pricing.json"
_DEFAULT_TTL_SECONDS = 24 * 60 * 60
_FETCH_TIMEOUT = 10.0


@dataclass(frozen=True)
class ModelPricing:
    """Per-unit USD pricing for one model (0.0 for any rate OpenRouter omits).

    Attributes:
        prompt (float): Cost per prompt token in USD.
        completion (float): Cost per completion token in USD.
        request (float): Fixed cost per request in USD.
        cache_read (float): Cost per cache-read input token in USD.
        cache_write (float): Cost per cache-creation (write) token in USD.
        reasoning (float): Cost per reasoning/thinking token in USD.
    """

    prompt: float = 0.0  # $/prompt token
    completion: float = 0.0  # $/completion token
    request: float = 0.0  # $/request
    cache_read: float = 0.0  # $/cached input token (read)
    cache_write: float = 0.0  # $/cache-creation token (write)
    reasoning: float = 0.0  # $/reasoning token

    def cost_breakdown(
        self,
        *,
        prompt_tokens: int,
        completion_tokens: int,
        cache_read_tokens: int = 0,
        cache_write_tokens: int = 0,
        reasoning_tokens: int = 0,
    ) -> dict[str, float]:
        """Estimated USD cost per token type for one call.

        `prompt` is the *non-cached* prompt portion: cache-read tokens are
        treated as a subset of the prompt and billed at the cheaper read rate
        (the common OpenAI/OpenRouter shape). Cache-write and reasoning are
        billed additively. When the caller reconciles against OpenRouter's
        actual cost, any provider-shape skew in these proportions is corrected.
        """
        billable_prompt = max(0, prompt_tokens - cache_read_tokens)
        return {
            "prompt": billable_prompt * self.prompt,
            "cache_read": cache_read_tokens * self.cache_read,
            "cache_write": cache_write_tokens * self.cache_write,
            "completion": completion_tokens * self.completion,
            "reasoning": reasoning_tokens * self.reasoning,
            "request": self.request,
        }


def _f(value: object) -> float:
    """Coerce an OpenRouter pricing value to float, returning 0.0 on failure.

    Args:
        value (object): A raw pricing rate value (may be a string, int, float, or None).

    Returns:
        float: The numeric rate, or 0.0 if conversion fails.
    """
    try:
        return float(value)  # OpenRouter sends rates as strings
    except (TypeError, ValueError):
        return 0.0


def _parse_pricing(pricing: dict) -> ModelPricing:
    """Build a :class:`ModelPricing` from a raw OpenRouter pricing dict.

    Args:
        pricing (dict): The `pricing` object from an OpenRouter `/api/v1/models` entry.

    Returns:
        ModelPricing: The parsed per-token-type pricing for the model.
    """
    return ModelPricing(
        prompt=_f(pricing.get("prompt")),
        completion=_f(pricing.get("completion")),
        request=_f(pricing.get("request")),
        cache_read=_f(pricing.get("input_cache_read")),
        cache_write=_f(pricing.get("input_cache_write")),
        reasoning=_f(pricing.get("internal_reasoning")),
    )


def normalize_model_id(model_id: str) -> str:
    """Map a litellm id to its OpenRouter id (strip the `openrouter/` prefix)."""
    mid = model_id or ""
    if mid.startswith("openrouter/"):
        return mid[len("openrouter/") :]
    return mid


def cost_model_candidates(model_id: str) -> list[str]:
    """Model-id forms to try when looking a model up in litellm's tables.

    litellm doesn't know the `openrouter/` prefix but does know the underlying
    `openai/gpt-4o-mini` / `gpt-4o-mini`, so try the id, then the stripped route,
    then the bare name — in that order, de-duplicated.
    """
    candidates = [model_id]
    if model_id.startswith("openrouter/"):
        candidates.append(model_id[len("openrouter/") :])
    if "/" in model_id:
        candidates.append(model_id.rsplit("/", 1)[1])
    seen: set[str] = set()
    return [c for c in candidates if not (c in seen or seen.add(c))]


def litellm_pricing(model_id: str) -> ModelPricing | None:
    """Per-token rates for `model_id` from litellm's static cost map, or None.

    Unlike `litellm.cost_per_token()` this reads the cache-read and cache-write
    rates too, so a cached agent transcript isn't billed as if every replayed token
    were fresh input (see the module docstring).

    `reasoning` is deliberately left at 0.0. litellm folds `thoughtsTokenCount`
    into `completion_tokens` before chiron sees it, so pricing the map's
    `output_cost_per_reasoning_token` on top would bill every thinking token twice.

    Returns:
        ModelPricing | None: The rates, or None when the map has no usable entry —
            an entry with no input *and* no output rate counts as no entry, since
            "priced at zero" and "not priced" are different claims.
    """
    try:
        import litellm
    except ImportError:  # pragma: no cover — litellm is a hard dependency
        return None

    for candidate in cost_model_candidates(model_id or ""):
        entry = litellm.model_cost.get(candidate)
        if not isinstance(entry, dict):
            continue
        prompt = _f(entry.get("input_cost_per_token"))
        completion = _f(entry.get("output_cost_per_token"))
        if not prompt and not completion:
            continue
        return ModelPricing(
            prompt=prompt,
            completion=completion,
            cache_read=_f(entry.get("cache_read_input_token_cost")),
            cache_write=_f(entry.get("cache_creation_input_token_cost")),
        )
    return None


class PricingTable:
    """Singleton cache of OpenRouter model pricing (disk-backed, TTL-refreshed).

    Fetches per-model pricing from the OpenRouter `/api/v1/models` endpoint and
    caches the result to disk. All lookups after warming are pure in-memory and
    never block the async event loop. Falls back gracefully to litellm pricing when
    the table is unavailable.
    """

    _instance: "PricingTable | None" = None

    def __init__(
        self, cache_path: Path = _CACHE_PATH, ttl_seconds: int = _DEFAULT_TTL_SECONDS
    ):
        """Initialise an empty (not yet loaded) pricing table.

        Args:
            cache_path (Path): Filesystem path for the JSON disk cache.
                Defaults to `~/.cache/chiron/openrouter_pricing.json`.
            ttl_seconds (int): Number of seconds before the disk cache is
                considered stale and re-fetched. Defaults to 86400 (24 hours).
        """
        self._models: dict[str, ModelPricing] = {}
        #: Whether the table actually holds rates. Distinct from `_disk_checked`: a
        #: missing or stale disk cache means the disk has been looked at and found
        #: wanting, which must not be mistaken for "the table is populated".
        self._loaded = False
        self._disk_checked = False
        self._cache_path = cache_path
        self._ttl = ttl_seconds

    @classmethod
    def instance(cls) -> "PricingTable":
        """Return the process-wide singleton, creating it on first call.

        Returns:
            PricingTable: The shared singleton instance.
        """
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    # -- lookup (pure, never blocks) -----------------------------------
    def get(self, model_id: str) -> ModelPricing | None:
        """Return pricing for `model_id` if the table is warmed, else `None`."""
        if not self._loaded and not self._disk_checked:
            # Cheap, non-network: try the disk cache once.
            self._load_from_disk()
        return self._models.get(normalize_model_id(model_id))

    # -- warming (may hit network / disk) ------------------------------
    def warm(self, model_ids: list[str] | None = None, *, force: bool = False) -> bool:
        """Ensure the table is loaded: fresh disk cache, else fetch from OpenRouter.

        Returns True if any pricing is available afterward. Best-effort, never
        raises. `model_ids` is advisory (used only for a debug log).

        A prior :meth:`get` that found no disk cache must not stop this from
        fetching — that is the case warming exists for.
        """
        if self._loaded and not force:
            return bool(self._models)
        if not force and not self._disk_checked and self._load_from_disk():
            return True
        return self._fetch_and_cache()

    def _load_from_disk(self) -> bool:
        """Attempt to populate the table from the on-disk JSON cache.

        Marks the disk as checked regardless of success, so it is not re-read on
        every `get()` call, and sets `_loaded` only when rates were actually found.
        Returns False when the cache is absent, stale, or corrupt.

        Returns:
            bool: True if at least one model's pricing was loaded, False otherwise.
        """
        self._disk_checked = True  # don't retry disk on every get()
        try:
            if not self._cache_path.exists():
                return False
            data = json.loads(self._cache_path.read_text(encoding="utf-8"))
            fetched_at = float(data.get("fetched_at") or 0)
            if time.time() - fetched_at > self._ttl:
                return False  # stale → caller may fetch
            models = data.get("models") or {}
            self._models = {
                mid: _parse_pricing(p)
                for mid, p in models.items()
                if isinstance(p, dict)
            }
            self._loaded = bool(self._models)
            return self._loaded
        except Exception:  # noqa: BLE001
            logger.warning("Could not read pricing cache", exc_info=True)
            return False

    def _fetch_and_cache(self) -> bool:
        """Fetch live pricing from the OpenRouter API and write the disk cache.

        Makes a synchronous HTTP request (intended to be called from a thread via
        `asyncio.to_thread`). Silently returns False if the network is unavailable
        or the response is malformed.

        Returns:
            bool: True if at least one model's pricing was fetched and stored.
        """
        try:
            import httpx

            resp = httpx.get(_MODELS_URL, timeout=_FETCH_TIMEOUT)
            resp.raise_for_status()
            raw = resp.json().get("data") or []
        except Exception:  # noqa: BLE001
            logger.warning("Could not fetch OpenRouter pricing", exc_info=True)
            return False

        models_raw: dict[str, dict] = {}
        parsed: dict[str, ModelPricing] = {}
        for entry in raw:
            mid = entry.get("id")
            pricing = entry.get("pricing")
            if mid and isinstance(pricing, dict):
                models_raw[mid] = pricing
                parsed[mid] = _parse_pricing(pricing)
        if not parsed:
            return False

        self._models = parsed
        self._loaded = True
        try:
            self._cache_path.parent.mkdir(parents=True, exist_ok=True)
            self._cache_path.write_text(
                json.dumps({"fetched_at": time.time(), "models": models_raw}),
                encoding="utf-8",
            )
        except Exception:  # noqa: BLE001
            logger.warning("Could not write pricing cache", exc_info=True)
        return True
