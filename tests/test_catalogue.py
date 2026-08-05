"""The model catalogue: what the picker is offered, and what it says about it.

No network. Both fetchers are given raw payloads directly, or pointed at a
temporary cache, so these tests describe parsing and fallback rather than
whether a provider happens to be up.
"""

from __future__ import annotations

import json
import time

from chiron.models.catalogue import (
    STATIC_MODELS,
    ModelInfo,
    _parse_google,
    _parse_openrouter,
    available_models,
    describe_unknown,
    find_model,
    list_google_models,
)
from chiron.models.providers import GOOGLE, LIVE, OPENROUTER, provider_for_model


def _google(name: str, methods: list[str]) -> dict:
    return {
        "name": f"models/{name}",
        "displayName": name,
        "supportedGenerationMethods": methods,
        "inputTokenLimit": 1_000_000,
    }


# ------------------------------------------------------------------- parsing


def test_a_live_only_model_becomes_a_live_entry():
    models = _parse_google(
        _google("gemini-3.1-flash-live-preview", ["bidiGenerateContent"])
    )
    assert [(m.id, m.provider, m.is_live) for m in models] == [
        ("live/gemini-3.1-flash-live-preview", LIVE, True)
    ]


def test_a_chat_only_model_becomes_an_ai_studio_entry():
    models = _parse_google(_google("gemini-2.5-flash", ["generateContent"]))
    assert [(m.id, m.provider, m.is_live) for m in models] == [
        ("gemini/gemini-2.5-flash", GOOGLE, False)
    ]


def test_a_model_serving_both_methods_is_offered_as_both():
    """Same weights, two genuinely different products; the player chooses."""
    models = _parse_google(
        _google("gemini-2.5-flash", ["generateContent", "bidiGenerateContent"])
    )
    assert sorted(m.id for m in models) == [
        "gemini/gemini-2.5-flash",
        "live/gemini-2.5-flash",
    ]


def test_models_that_cannot_hold_a_conversation_are_dropped():
    assert _parse_google(_google("text-embedding-004", ["embedContent"])) == []
    assert _parse_google({"supportedGenerationMethods": ["generateContent"]}) == []


def test_openrouter_entries_report_their_modalities():
    seeing = _parse_openrouter(
        {
            "id": "openai/gpt-4o",
            "name": "OpenAI: GPT-4o",
            "pricing": {"prompt": "0.0000025", "completion": "0.00001"},
            "architecture": {"input_modalities": ["text", "image"]},
            "supported_parameters": ["tools"],
        }
    )
    blind = _parse_openrouter(
        {
            "id": "meta/llama-text",
            "pricing": {},
            "architecture": {"input_modalities": ["text"]},
        }
    )
    assert seeing.supports_vision is True and seeing.is_live is False
    assert blind.supports_vision is False


# -------------------------------------------------------------------- labels


def test_a_label_says_provider_mode_and_price():
    label = ModelInfo(
        id="openrouter/openai/gpt-4o",
        provider=OPENROUTER,
        provider_model_id="openai/gpt-4o",
        name="GPT-4o",
        vendor="openai",
        context_length=128_000,
        prompt_price=2.5 / 1_000_000,
        completion_price=10.0 / 1_000_000,
        pricing_known=True,
        supports_tools=True,
    ).label()
    assert "OpenRouter" in label
    assert "openrouter/openai/gpt-4o" in label
    assert "non-live" in label
    assert "$2.50/$10.00 per M" in label


def test_an_unpriced_model_says_so_rather_than_claiming_to_be_free():
    label = STATIC_MODELS[0].label()
    assert "pricing unavailable" in label
    assert "$0.00" not in label


def test_an_unknown_id_is_describable_without_being_known():
    unknown = describe_unknown("openrouter/some/model-from-today")
    assert unknown.provider == OPENROUTER
    assert unknown.is_live is False
    assert unknown.pricing_known is False

    live_unknown = describe_unknown("live/gemini-9-flash-live")
    assert live_unknown.is_live is True


# ------------------------------------------------------------- combined view


def test_a_provider_with_no_key_is_absent():
    """Activation is by key. Offering models nothing can call would be a lie."""
    models = available_models(google_key="", openrouter_key="")
    assert models == [model for model in STATIC_MODELS if model.provider != OPENROUTER]


def test_no_key_means_no_google_models():
    assert list_google_models("") == []


def test_live_models_are_listed_first():
    models = available_models()
    live = [index for index, m in enumerate(models) if m.is_live]
    non_live = [index for index, m in enumerate(models) if not m.is_live]
    assert not live or not non_live or max(live) < min(non_live)


def test_finding_a_model_by_id():
    models = list(STATIC_MODELS)
    assert find_model(models, models[0].id) is models[0]
    assert find_model(models, "nothing/at/all") is None


def test_a_cache_written_by_an_older_build_still_loads(tmp_path):
    """`is_live` is new; a cache without it must be a hit, not a crash."""
    cache = tmp_path / "google_catalogue.json"
    entry = {
        "id": "gemini/gemini-2.5-flash",
        "provider": "google",
        "provider_model_id": "gemini-2.5-flash",
        "name": "Gemini 2.5 Flash",
        "vendor": "google",
        "context_length": 1_000_000,
        "prompt_price": 0.0,
        "completion_price": 0.0,
        "pricing_known": False,
        "supports_tools": True,
        "supports_vision": True,
    }
    cache.write_text(
        json.dumps({"fetched_at": time.time(), "models": [entry]}), encoding="utf-8"
    )

    models = list_google_models("a-key", cache_path=cache)

    assert [m.id for m in models] == ["gemini/gemini-2.5-flash"]
    assert models[0].is_live is False


# ----------------------------------------------------------------- providers


def test_an_id_names_its_own_provider():
    assert provider_for_model("live/gemini-3.1-flash-live-preview").id == LIVE
    assert provider_for_model("gemini/gemini-2.5-flash").id == GOOGLE
    assert provider_for_model("openrouter/openai/gpt-4o").id == OPENROUTER
    assert provider_for_model("anthropic/claude-sonnet-4") is None


def test_only_the_live_provider_is_marked_live():
    assert provider_for_model("live/x").is_live is True
    assert provider_for_model("gemini/x").is_live is False
