"""Shared Responder context policy and mixed-agent cost records."""

from __future__ import annotations

from chiron.live import estimate
from chiron.models.usage import LLMCallRecord
from chiron.nonlive import compaction as ctx
from chiron.responder.conversation import ResponderConversation
from chiron.sessions.store import CostRollup


def exchange(number: int) -> list[dict]:
    return [
        {"role": "user", "content": f"question {number}"},
        {"role": "assistant", "content": f"answer {number}"},
    ]


def test_images_are_charged_flat_not_as_base64():
    huge = "A" * 400_000
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "what is this"},
                {"type": "image_url", "image_url": {"url": f"data:{huge}"}},
            ],
        }
    ]
    assert ctx.count_tokens("unknown/model", messages) < 1000


def test_proactive_threshold_is_exactly_eighty_five_percent(monkeypatch):
    monkeypatch.setattr(ctx, "count_tokens", lambda model, messages: 850)
    assert ctx.should_compact("model", [], window=1000) is True
    monkeypatch.setattr(ctx, "count_tokens", lambda model, messages: 849)
    assert ctx.should_compact("model", [], window=1000) is False


def test_context_window_prefers_provider_catalogue_metadata(monkeypatch):
    monkeypatch.setattr(
        "chiron.models.catalogue.cached_context_length_info",
        lambda model_id: (65_536, "OpenRouter catalogue"),
    )
    resolved = ctx.resolve_context_window("openrouter/example/model")
    assert resolved.tokens == 65_536
    assert resolved.source == "OpenRouter catalogue"
    assert resolved.assumed is False


def test_context_window_uses_litellm_then_a_conservative_fallback(monkeypatch):
    import litellm

    monkeypatch.setattr(
        "chiron.models.catalogue.cached_context_length_info", lambda model_id: None
    )
    monkeypatch.setattr(
        litellm, "get_model_info", lambda model_id: {"max_input_tokens": 48_000}
    )
    resolved = ctx.resolve_context_window("custom/known-to-litellm")
    assert (resolved.tokens, resolved.source, resolved.assumed) == (
        48_000,
        "LiteLLM metadata",
        False,
    )

    def unknown(_model_id):
        raise KeyError("unknown")

    monkeypatch.setattr(litellm, "get_model_info", unknown)
    fallback = ctx.resolve_context_window("custom/not-known-anywhere")
    assert fallback.tokens == 32_000
    assert fallback.assumed is True


def test_typed_context_overflow_is_recognised():
    class ContextWindowExceededError(Exception):
        pass

    assert (
        ctx.is_context_overflow(ContextWindowExceededError("context window exceeded"))
        is True
    )
    assert ctx.is_context_overflow(RuntimeError("ordinary failure")) is False


def test_canonical_memory_keeps_user_and_final_answer_only():
    memory = ResponderConversation()
    memory.commit("where next?", "go north")
    assert memory.messages == [
        {"role": "user", "content": "where next?"},
        {"role": "assistant", "content": "go north"},
    ]


def test_images_become_capture_time_placeholders_in_canonical_memory():
    from chiron.capture.frames import Frame

    memory = ResponderConversation()
    memory.commit(
        "what is this?",
        "a door",
        Frame(jpeg=b"pixels", captured_at=1_700_000_000, width=8, height=4),
    )
    assert "data:" not in str(memory.messages)
    assert "[frame " in memory.messages[0]["content"]


def test_compaction_keeps_the_latest_eight_messages_verbatim():
    memory = ResponderConversation()
    for number in range(8):
        memory.messages.extend(exchange(number))
    head, tail, until = memory.compaction_parts()
    assert len(tail) == 8
    assert len(head) == 8
    assert until == 8
    assert tail == memory.messages[-8:]


def test_summary_changes_active_context_without_deleting_canonical_turns():
    memory = ResponderConversation()
    for number in range(6):
        memory.messages.extend(exchange(number))
    head, tail, until = memory.compaction_parts()
    memory.apply_summary("Earlier progress", until)
    assert len(memory.messages) == 12
    assert memory.active_messages()[0]["role"] == "system"
    assert "Earlier progress" in memory.active_messages()[0]["content"]
    assert memory.active_messages()[1:] == tail
    assert len(head) == 4


def test_all_v3_call_kinds_validate():
    for kind in (
        "fixed_answer",
        "react_step",
        "responder_compaction",
        "journal_compaction",
        "observer_checkpoint",
        "observer_context",
    ):
        assert LLMCallRecord(kind=kind).kind == kind


def test_observer_estimate_is_attributed_and_marked_approximate():
    record = estimate.estimate_turn(
        model_id="live/gemini-3.1-flash-live-preview",
        frames=1,
        media_resolution="low",
        prompt_text="checkpoint",
        agent_id="observer",
        kind="observer_checkpoint",
    )
    assert record.agent_id == "observer"
    assert record.pricing_source == "estimated"


def test_discarded_live_audio_is_estimated_from_pcm_duration():
    assert estimate.audio_output_tokens(0) == 0
    assert estimate.audio_output_tokens(48_000) == 25
    record = estimate.estimate_turn(
        model_id="live/gemini-3.1-flash-live-preview",
        frames=0,
        media_resolution="low",
        output_tokens=25,
        kind="observer_checkpoint",
    )
    assert record.completion_tokens == 25


def test_mixed_estimated_and_measured_cost_keeps_the_tilde():
    rollup = CostRollup()
    rollup.add(
        LLMCallRecord(
            agent_id="observer",
            kind="observer_checkpoint",
            cost_usd=0.01,
            pricing_source="estimated",
        ).model_dump()
    )
    rollup.add(
        LLMCallRecord(
            agent_id="responder",
            kind="fixed_answer",
            cost_usd=0.02,
            pricing_source="actual",
        ).model_dump()
    )
    assert rollup.total_usd == 0.03
    assert rollup.render() == "~$0.03"
