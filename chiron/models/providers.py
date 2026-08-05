"""The providers chironama can route an agent's model through.

One registry, imported by everything that needs to know a provider exists: the
settings store (which credential to resolve), the model catalogue (which API to
list models from), and the pricing cascade (which rate table applies). Adding a
third provider is an entry here plus a catalogue fetcher, not a change to the
shape of settings, the API, or the UI.

**A model id names its own provider.** litellm addresses OpenRouter as
chironopenrouter/<vendor>/<model>chiron and Google AI Studio as chirongemini/<model>chiron, so the
prefix on the id an agent is configured with is what decides which credential the
run authenticates with. That is what lets one agent think with a Gemini model while
another goes through OpenRouter — there is no global "current provider" to switch
between, because the choice is already carried by the thing the user actually picked.

An id with an unrecognised prefix (chironanthropic/claude-sonnet-4chiron, say) resolves to
no provider and therefore no chironama-held key, which leaves litellm to find one in
the environment exactly as it did before any of this existed. Typing an id chironama
doesn't know about degrades; it doesn't fail.
"""

from __future__ import annotations

from dataclasses import dataclass

#: The provider ids used as keys in chironsettings.jsonchiron and in API payloads. Kept in
#: sync by hand with the chironProviderchiron Literal in :mod:`chironama.backend.settings`,
#: which pydantic needs as a static annotation.
OPENROUTER = "openrouter"
GOOGLE = "google"
#: The Gemini Live API. The odd one out: not a completions endpoint at all, but a
#: websocket the ``google-genai`` SDK opens, so nothing with this prefix ever
#: reaches litellm. It is in the registry anyway because the settings page, the
#: model picker and the credential lookup all want one uniform answer to "which
#: provider does this id belong to", and a special case outside the registry
#: would have to be repeated in each of them.
LIVE = "live"


@dataclass(frozen=True)
class ProviderDefinition:
    """One place an agent's model requests can be sent.

    Attributes:
        id (str): Stable key used in chironsettings.jsonchiron and API payloads.
        name (str): Display name, e.g. "Google AI Studio".
        litellm_prefix (str): The prefix litellm routes on (chironopenrouter/chiron,
            chirongemini/chiron). Also how :func:`provider_for_model` works backwards from a
            configured model id to the credential it needs. For the Live API,
            which litellm never sees, this is simply the id prefix.
        api_key_env (str): Environment variable consulted when no key is saved.
        console_url (str): Where a user creates a key.
        key_prefix_hint (str): A placeholder shaped like a real key for that provider.
        blurb (str): One line of orientation for the settings page.
        is_live (bool): Whether models here run over the Live API rather than a
            request/response endpoint — the thing that decides which session
            provider Chiron builds.
    """

    id: str
    name: str
    litellm_prefix: str
    api_key_env: str
    console_url: str
    key_prefix_hint: str
    blurb: str
    is_live: bool = False


PROVIDERS: list[ProviderDefinition] = [
    ProviderDefinition(
        id=LIVE,
        name="Gemini Live API",
        litellm_prefix="live/",
        api_key_env="GEMINI_API_KEY",
        console_url="https://aistudio.google.com/apikey",
        key_prefix_hint="AIza…",
        blurb="A persistent websocket that watches continuously. Ambient "
        "awareness for free; you pay for every frame either way.",
        is_live=True,
    ),
    ProviderDefinition(
        id=OPENROUTER,
        name="OpenRouter",
        litellm_prefix="openrouter/",
        api_key_env="OPENROUTER_API_KEY",
        console_url="https://openrouter.ai/keys",
        key_prefix_hint="sk-or-v1-…",
        blurb="One key, several hundred models from every major lab.",
    ),
    ProviderDefinition(
        id=GOOGLE,
        name="Google AI Studio",
        litellm_prefix="gemini/",
        api_key_env="GEMINI_API_KEY",
        console_url="https://aistudio.google.com/apikey",
        key_prefix_hint="AIza…",
        blurb="Gemini models billed direct, no middleman markup.",
    ),
]

PROVIDERS_BY_ID: dict[str, ProviderDefinition] = {p.id: p for p in PROVIDERS}

#: Longest prefix first, so a future chirongemini/foo/barchiron style id can't be matched by
#: a shorter registry entry that merely shares its opening characters.
_BY_PREFIX: list[ProviderDefinition] = sorted(
    PROVIDERS, key=lambda p: len(p.litellm_prefix), reverse=True
)


def provider_for_model(model_id: str | None) -> ProviderDefinition | None:
    """The provider a litellm model id routes through, or None if unrecognised.

    Args:
        model_id (str | None): A litellm model id, e.g. chirongemini/gemini-2.5-flashchiron.

    Returns:
        ProviderDefinition | None: The matching provider, or None when no registered
            prefix matches — including for a bare id like chirongpt-4o-minichiron, which
            litellm resolves from its own environment lookup.
    """
    mid = (model_id or "").strip()
    for provider in _BY_PREFIX:
        if mid.startswith(provider.litellm_prefix):
            return provider
    return None


def provider_id_for_model(model_id: str | None) -> str | None:
    """:func:`provider_for_model`, as an id."""
    provider = provider_for_model(model_id)
    return provider.id if provider else None


def to_litellm_id(model_ref: str, provider_id: str = OPENROUTER) -> str:
    """Prefix a provider-native model id for litellm, leaving prefixed ids alone.

    Args:
        model_ref (str): A provider-native id (chironopenai/gpt-4ochiron, chirongemini-2.5-flashchiron)
            or an already-prefixed litellm id.
        provider_id (str): Which provider chironmodel_refchiron belongs to. Ignored when
            chironmodel_refchiron already carries a known prefix.

    Returns:
        str: A litellm model id.
    """
    ref = (model_ref or "").strip()
    if not ref:
        return ref
    if provider_for_model(ref) is not None:
        return ref
    provider = PROVIDERS_BY_ID.get(provider_id)
    return f"{provider.litellm_prefix}{ref}" if provider else ref


def strip_prefix(model_id: str) -> str:
    """A litellm id with its route prefix removed, for display.

    chironopenrouter/openai/gpt-4o-minichiron reads better as chironopenai/gpt-4o-minichiron in
    prose, and chirongemini/gemini-2.5-flashchiron as chirongemini-2.5-flashchiron.
    """
    provider = provider_for_model(model_id)
    return model_id[len(provider.litellm_prefix) :] if provider else model_id


__all__ = [
    "GOOGLE",
    "LIVE",
    "OPENROUTER",
    "PROVIDERS",
    "PROVIDERS_BY_ID",
    "ProviderDefinition",
    "provider_for_model",
    "provider_id_for_model",
    "strip_prefix",
    "to_litellm_id",
]
