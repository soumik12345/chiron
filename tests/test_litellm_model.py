"""Provider request seams kept by the LiteLLM accounting wrapper."""

from __future__ import annotations

from chiron.models import litellm_model
from chiron.models.litellm_model import LiteLLMModel


async def test_response_format_and_openrouter_route_requirements_are_forwarded(
    monkeypatch,
):
    seen = {}

    async def completion(**kwargs):
        seen.update(kwargs)
        return object()

    monkeypatch.setattr(litellm_model.litellm, "acompletion", completion)
    model = LiteLLMModel(
        model_id="openrouter/google/gemini-2.5-flash", api_key="router-key"
    )
    schema = {"type": "json_schema", "json_schema": {"name": "journal"}}
    await model.acompletion(
        messages=[{"role": "user", "content": "review"}],
        response_format=schema,
        extra_body={"provider": {"require_parameters": True}},
    )

    assert seen["response_format"] is schema
    assert seen["extra_body"]["provider"]["require_parameters"] is True
    assert seen["extra_body"]["usage"]["include"] is True
