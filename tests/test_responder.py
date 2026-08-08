"""Fixed-horizon and ReAct behavior behind one canonical conversation."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from chiron.capture.frames import Frame
from chiron.config.settings import Settings
from chiron.journal.log import JournalLog
from chiron.journal.service import JournalService
from chiron.responder.conversation import ResponderConversation
from chiron.responder.session import ResponderSessionManager
from chiron.session import ObserverStatus


def status(state: str = "live") -> ObserverStatus:
    return ObserverStatus(
        state=state,
        watch_requested=state == "live",
        last_observed_at=100.0 if state == "live" else None,
        detail="test",
    )


def test_live_before_the_first_successful_checkpoint_is_still_stale():
    snapshot = ObserverStatus(state="live", watch_requested=True)
    assert snapshot.stale is True


def test_batched_freshness_distinguishes_sampled_pending_and_processed():
    snapshot = ObserverStatus(
        state="watching",
        watch_requested=True,
        mode="nonlive",
        accepting_frames=True,
        last_sampled_at=110.0,
        last_observed_at=100.0,
        pending_frames=2,
    )
    from chiron.responder.prompts import observer_context

    rendered = observer_context(snapshot)
    assert snapshot.stale is True
    assert "pending Observer frames: 2" in rendered
    assert "newest frame processed" in rendered
    assert "newest frame sampled" in rendered


def frame() -> Frame:
    return Frame(jpeg=b"pixels", captured_at=1_700_000_000, width=8, height=4)


@pytest.fixture
def manager(qapp):
    settings = Settings(api_key="test-key")
    journal = JournalService(JournalLog())
    result = ResponderSessionManager(
        settings, journal.reader(), ResponderConversation()
    )
    result.test_journal_service = journal
    return result


async def test_fixed_horizon_ordinary_path_is_one_completion(manager):
    manager.test_journal_service.record("Found a brass key.", "item")
    calls = []

    async def complete(messages, *, kind, max_tokens=1000):
        calls.append((messages, kind))
        return "Use it on the tower door."

    manager._complete = complete
    answers = []
    manager.responseCompleted.connect(answers.append)
    manager.ask("where does the key go?", None, status())
    await manager._worker

    assert len(calls) == 1
    assert calls[0][1] == "fixed_answer"
    assert "Found a brass key" in str(calls[0][0])
    assert answers == ["Use it on the tower door."]
    assert len(manager.conversation) == 2


async def test_fixed_prompt_carries_status_journal_and_current_image(manager):
    manager.test_journal_service.record("Reached the tower.", "location")
    seen = []

    async def complete(messages, *, kind, max_tokens=1000):
        seen.extend(messages)
        return "That is the tower door."

    manager._complete = complete
    current = frame()
    manager.ask("what is this?", current, status())
    await manager._worker

    prompt = seen[-1]["content"]
    assert any(part["type"] == "image_url" for part in prompt)
    assert "Reached the tower" in str(prompt)
    assert "stale: no" in str(prompt)
    assert "seconds ago" in str(prompt)
    assert "data:" not in str(manager.conversation.messages)


async def test_stale_status_is_explicit_and_never_reuses_an_image(manager):
    seen = []

    async def complete(messages, *, kind, max_tokens=1000):
        seen.extend(messages)
        return "I cannot verify the current screen."

    manager._complete = complete
    manager.ask("what is on screen?", None, status("watch_off"))
    await manager._worker
    assert "stale: yes" in str(seen[-1]["content"])
    assert "No current screenshot" in str(seen[-1]["content"])


async def test_questions_are_committed_fifo(manager):
    release = asyncio.Event()
    started = []

    async def complete(messages, *, kind, max_tokens=1000):
        question = str(messages[-1]["content"])
        started.append(question)
        if len(started) == 1:
            await release.wait()
        return "first answer" if len(started) == 1 else "second answer"

    manager._complete = complete
    manager.ask("first question", None, status())
    manager.ask("second question", None, status())
    await asyncio.sleep(0)
    assert len(started) == 1
    release.set()
    await manager._worker
    assert [m["content"] for m in manager.conversation.messages] == [
        "first question",
        "first answer",
        "second question",
        "second answer",
    ]


async def test_proactive_compaction_keeps_canonical_history(manager, monkeypatch):
    for number in range(6):
        manager.conversation.commit(f"q{number}", f"a{number}")
    kinds = []

    async def complete(messages, *, kind, max_tokens=1000):
        kinds.append(kind)
        return "summary" if kind == "responder_compaction" else "answer"

    manager._complete = complete
    monkeypatch.setattr(
        "chiron.responder.session.ctx.should_compact", lambda model, messages: True
    )
    manager.ask("next?", None, status())
    await manager._worker
    assert kinds == ["responder_compaction", "fixed_answer"]
    assert manager.conversation.summary == "summary"
    assert len(manager.conversation.messages) == 14


async def test_overflow_compacts_and_retries_once(manager):
    class ContextWindowExceededError(Exception):
        pass

    for number in range(6):
        manager.conversation.commit(f"q{number}", f"a{number}")
    kinds = []
    failed = False

    async def complete(messages, *, kind, max_tokens=1000):
        nonlocal failed
        kinds.append(kind)
        if kind == "fixed_answer" and not failed:
            failed = True
            raise ContextWindowExceededError("context window exceeded")
        return "summary" if kind == "responder_compaction" else "recovered"

    manager._complete = complete
    manager.ask("next?", None, status())
    await manager._worker
    assert kinds == ["fixed_answer", "responder_compaction", "fixed_answer"]
    assert manager.conversation.messages[-1]["content"] == "recovered"


async def test_overflow_compacts_journal_when_conversation_has_no_old_head(manager):
    manager.test_journal_service.record("Found the moon key.", "item")

    class JournalCompactorSpy:
        def __init__(self, service):
            self.service = service
            self.calls = []

        async def prepare(self, model, *, force=False, reason="threshold"):
            self.calls.append((force, reason))
            if force:
                self.service.apply_summary("The player found the moon key.", 1)
            return self.service.snapshot()

    compactor = JournalCompactorSpy(manager.test_journal_service)
    manager.journal_compactor = compactor
    failed = False

    async def complete(messages, *, kind, max_tokens=1000):
        nonlocal failed
        if kind == "fixed_answer" and not failed:
            failed = True
            raise RuntimeError("context window exceeded")
        return "recovered"

    manager._complete = complete
    manager.ask("what did I find?", None, status())
    await manager._worker

    assert compactor.calls == [
        (False, "responder_threshold"),
        (True, "responder_overflow"),
    ]
    assert manager.conversation.messages[-1]["content"] == "recovered"


def test_gemini_36_omits_temperature(manager):
    assert manager._model().temperature is None


def test_model_and_mode_changes_keep_conversation(manager):
    manager.conversation.commit("where?", "north")
    edited = manager.settings.copy_deep()
    edited.responder_model = "openrouter/openai/gpt-4o"
    edited.responder_mode = "react"
    manager.apply_settings(edited)
    assert len(manager.conversation) == 2
    assert manager.conversation.messages[-1]["content"] == "north"


class FakeReactModel:
    def __init__(self, responses):
        self.model_id = "gemini/gemini-3.6-flash"
        self.responses = list(responses)
        self.calls = []
        self.cumulative = {"cost_usd": 0.0}

    async def acompletion(
        self, *, messages, tools=None, stream=False, tool_choice=None
    ):
        self.calls.append(
            {"messages": messages, "tools": tools, "tool_choice": tool_choice}
        )
        kind, content = self.responses.pop(0)
        tool_calls = None
        finish_reason = "stop"
        if kind == "tool":
            finish_reason = "tool_calls"
            tool_calls = [
                SimpleNamespace(
                    id="journal-1",
                    function=SimpleNamespace(name="read_journal", arguments="{}"),
                )
            ]
            content = None
        message = SimpleNamespace(content=content, tool_calls=tool_calls)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=message, finish_reason=finish_reason)],
            usage=None,
        )

    def record_usage(self, usage, *, response=None, context=None):
        return {}

    def record_failure(self, error, *, context=None):
        return None


async def test_react_forces_its_only_tool_then_commits_only_the_final(manager):
    manager.settings.responder_mode = "react"
    manager.test_journal_service.record("Boss is weak to fire.", "combat")
    model = FakeReactModel([("tool", None), ("final", "Use fire damage.")])
    manager._model = lambda max_tokens=1000: model
    traces = []
    manager.agentTrace.connect(traces.append)

    manager.ask("any weakness?", None, status())
    await manager._worker

    assert model.calls[0]["tool_choice"] == "required"
    assert [tool["function"]["name"] for tool in model.calls[0]["tools"]] == [
        "read_journal"
    ]
    assert model.calls[1]["tool_choice"] is None
    assert "Boss is weak to fire" in str(model.calls[1]["messages"])
    assert [row["role"] for row in manager.conversation.messages] == [
        "user",
        "assistant",
    ]
    assert manager.conversation.messages[-1]["content"] == "Use fire damage."
    assert traces


async def test_react_forces_a_fresh_read_on_every_request(manager):
    manager.settings.responder_mode = "react"
    model = FakeReactModel(
        [
            ("tool", None),
            ("final", "one"),
            ("tool", None),
            ("final", "two"),
        ]
    )
    manager._model = lambda max_tokens=1000: model
    manager.ask("first", None, status())
    manager.ask("second", None, status())
    await manager._worker
    assert [call["tool_choice"] for call in model.calls] == [
        "required",
        None,
        "required",
        None,
    ]


async def test_react_guard_rejects_an_ungrounded_final(manager):
    manager.settings.responder_mode = "react"
    model = FakeReactModel([("final", "guess"), ("final", "still guessing")])
    manager._model = lambda max_tokens=1000: model
    errors = []
    manager.errorOccurred.connect(errors.append)
    manager.ask("what now?", None, status())
    await manager._worker
    assert errors
    assert len(manager.conversation) == 0


async def test_react_accepts_a_multimodal_initial_question(manager):
    manager.settings.responder_mode = "react"
    model = FakeReactModel([("tool", None), ("final", "a map")])
    manager._model = lambda max_tokens=1000: model
    traces = []
    manager.agentTrace.connect(traces.append)
    manager.ask("what is this?", frame(), status())
    await manager._worker
    current = model.calls[0]["messages"][-1]["content"]
    assert isinstance(current, list)
    assert any(part["type"] == "image_url" for part in current)
    assert "data:" not in str(manager.conversation.messages)
    assert "data:image" not in str(traces)
    assert "current request frame" in str(traces)
