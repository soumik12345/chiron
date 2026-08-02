"""The non-live provider: the observer, the Q&A path, and the seam above both.

No model is ever reached. :meth:`NonLiveSessionManager._model` is the single
place a litellm wrapper is built, so replacing it is enough to drive every path
here — including streaming — with no network and no key.
"""

from __future__ import annotations

import asyncio
import json
import time
from types import SimpleNamespace

import pytest
from PIL import Image

from chiron.capture.frames import encode_frame
from chiron.config.settings import Settings
from chiron.journal.log import JournalLog
from chiron.journal.writers import (
    ObserverJournal,
    ToolCallJournal,
    build_journal_writer,
)
from chiron.nonlive import session as session_module
from chiron.nonlive.prompts import (
    build_observer_instruction,
    build_qa_instruction,
)
from chiron.nonlive.session import NonLiveSessionManager
from chiron.session import build_session_provider


class FakeModel:
    """Records the messages it was given and replays a scripted reply."""

    def __init__(self, model_id: str, reply: str = "") -> None:
        self.model_id = model_id
        self.reply = reply
        self.calls: list[list[dict]] = []
        self.usage_kinds: list[str] = []
        self.fail: Exception | None = None

    async def acompletion(self, *, messages, tools=None, stream=False):
        self.calls.append(messages)
        if self.fail is not None:
            raise self.fail
        if stream:
            return _Chunks(self.reply)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=self.reply))],
            usage=None,
        )

    def record_usage(self, usage, *, response=None, context=None):
        self.usage_kinds.append((context or {}).get("kind", ""))


class _Chunks:
    """An async iterable of streaming deltas, one word at a time."""

    def __init__(self, text: str) -> None:
        self.words = text.split(" ") if text else []
        self.usage = None

    async def __aiter__(self):
        for index, word in enumerate(self.words):
            suffix = "" if index == len(self.words) - 1 else " "
            yield SimpleNamespace(
                choices=[SimpleNamespace(delta=SimpleNamespace(content=word + suffix))],
                usage=None,
            )


def _frame(colour=(20, 30, 40), *, when: float | None = None):
    """A real encoded frame — cheap enough, and carries a real signature."""
    return encode_frame(
        Image.new("RGB", (64, 36), colour), width=64, captured_at=when or time.time()
    )


@pytest.fixture
def manager(qapp, monkeypatch):
    """A non-live manager wired to fake models, never touching the network."""
    monkeypatch.setattr(session_module, "FRESH_FRAME_WAIT_SECONDS", 0.05)
    settings = Settings(api_key="AIza-test", selected_model="gemini/gemini-2.5-flash")
    log = JournalLog()
    manager = NonLiveSessionManager(settings, log, ObserverJournal(log))
    manager.models: dict[str, FakeModel] = {}

    def fake_model(model_id: str, *, max_tokens: int = 600) -> FakeModel:
        model = manager.models.setdefault(model_id, FakeModel(model_id))
        return model

    monkeypatch.setattr(manager, "_model", fake_model)
    return manager


def _reply(manager, model_id: str, text: str) -> FakeModel:
    """Script what `model_id` will say."""
    model = manager.models.setdefault(model_id, FakeModel(model_id))
    model.reply = text
    manager.models[model_id] = model
    return model


def _images(messages: list[dict]) -> int:
    """How many image parts a request carried."""
    total = 0
    for message in messages:
        content = message.get("content")
        if isinstance(content, list):
            total += sum(1 for part in content if part.get("type") == "image_url")
    return total


# --------------------------------------------------------------------- seam


def test_the_model_decides_which_provider_runs(qapp):
    log = JournalLog()
    for selection, expected in (
        ("live/gemini-3.1-flash-live-preview", "LiveSessionManager"),
        ("gemini/gemini-2.5-flash", "NonLiveSessionManager"),
        ("openrouter/openai/gpt-4o-mini", "NonLiveSessionManager"),
    ):
        settings = Settings(selected_model=selection)
        writer = build_journal_writer(settings.journal, log, live=settings.is_live)
        provider = build_session_provider(settings, log, writer)
        assert type(provider).__name__ == expected


def test_non_live_mode_journals_through_the_observer(qapp):
    log = JournalLog()
    live = Settings()
    non_live = Settings(selected_model="gemini/gemini-2.5-flash")
    non_live.journal.strategy = "sidecar"

    assert isinstance(build_journal_writer(live.journal, log), ToolCallJournal)
    assert isinstance(
        build_journal_writer(non_live.journal, log, live=False), ObserverJournal
    ), "the strategy setting is a live-mode setting"


def test_there_is_nothing_to_fold(manager):
    """The journal is in every request already; there is no session to fall behind."""
    manager.journal.append("Entered the crypt.", category="location")
    assert manager.fold_journal() == 0
    assert manager.fold_journal(force=True) == 0


def test_starting_without_a_key_says_so(qapp, monkeypatch):
    for name in ("GEMINI_API_KEY", "GOOGLE_API_KEY", "OPENROUTER_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    log = JournalLog()
    manager = NonLiveSessionManager(
        Settings(selected_model="gemini/gemini-2.5-flash"), log, ObserverJournal(log)
    )
    errors = []
    manager.errorOccurred.connect(errors.append)

    manager.start()

    assert manager.status == "error"
    assert errors and "API key" in errors[0]


def test_frames_are_held_rather_than_sent(manager):
    """Unlike live mode, a frame not sent is not a frame the model never sees."""
    manager.send_frame(_frame())
    manager.send_frame(_frame((90, 10, 10)))

    assert manager.frames_sent == 0, "nothing is paid for until something asks"
    assert len(manager._frames) == 2


# ---------------------------------------------------------------------- Q&A


async def test_an_answer_streams_into_the_overlay(manager):
    _reply(manager, "gemini/gemini-2.5-flash", "A skeleton on the bridge.")
    manager.send_frame(_frame())

    deltas, completed, started = [], [], []
    manager.responseDelta.connect(deltas.append)
    manager.responseCompleted.connect(completed.append)
    manager.responseStarted.connect(lambda: started.append(True))

    manager.send_text("what killed me?")
    await manager._answer_task

    assert started == [True]
    assert "".join(deltas) == "A skeleton on the bridge."
    assert completed == ["A skeleton on the bridge."]


async def test_a_question_carries_the_journal_and_the_frames(manager):
    _reply(manager, "gemini/gemini-2.5-flash", "Yes.")
    manager.journal.append("Died on the bridge.", category="death")
    manager.send_frame(_frame())
    manager.send_frame(_frame((200, 10, 10)))

    manager.send_text("where am I?")
    await manager._answer_task

    messages = manager.models["gemini/gemini-2.5-flash"].calls[0]
    text = json.dumps(messages[-1]["content"])
    assert "Died on the bridge." in text
    assert "where am I?" in text
    assert _images(messages) == 2, "the fresh frame plus a little of the run-up"
    assert manager.frames_sent == 2


async def test_a_question_waits_for_a_frame_newer_than_itself(manager, monkeypatch):
    monkeypatch.setattr(session_module, "FRESH_FRAME_WAIT_SECONDS", 2.0)
    _reply(manager, "gemini/gemini-2.5-flash", "Now.")
    manager.send_frame(_frame(when=time.time()))

    manager.send_text("what is this?")
    await asyncio.sleep(0)
    fresh = _frame((5, 200, 5), when=time.time() + 0.1)
    manager.send_frame(fresh)
    await manager._answer_task

    assert manager._frames[-1] is fresh
    assert manager.models["gemini/gemini-2.5-flash"].calls, "it did not wait forever"


async def test_a_question_with_nothing_watching_answers_from_the_journal(manager):
    """v0's behaviour: still answered, from what was noted rather than from now."""
    _reply(manager, "gemini/gemini-2.5-flash", "From the journal.")
    manager.journal.append("Lit the bonfire.", category="progress")

    started = time.monotonic()
    manager.send_text("what have I done?")
    await manager._answer_task
    elapsed = time.monotonic() - started

    messages = manager.models["gemini/gemini-2.5-flash"].calls[0]
    assert _images(messages) == 0
    assert "No screenshot is available" in json.dumps(messages[-1]["content"])
    assert elapsed < 0.5, "no waiting for a frame that is never coming"


async def test_history_keeps_the_words_and_drops_the_pixels(manager):
    """The whole of history cost control: text grows, frames do not accumulate."""
    _reply(manager, "gemini/gemini-2.5-flash", "First.")
    manager.send_frame(_frame(when=1_700_000_000.0))
    manager.send_text("one?")
    await manager._answer_task

    manager.send_frame(_frame((200, 10, 10)))
    manager.send_text("two?")
    await manager._answer_task

    second = manager.models["gemini/gemini-2.5-flash"].calls[1]
    history = second[1:-1]
    assert _images(history) == 0, "old turns carry no images"
    assert "[frame " in json.dumps(history), "but they still say when they looked"
    assert _images([second[-1]]) >= 1, "only the current turn has pixels"


async def test_history_is_bounded(manager):
    _reply(manager, "gemini/gemini-2.5-flash", "ok")
    for index in range(30):
        manager.send_text(f"question {index}")
        await manager._answer_task
    assert len(manager._history) <= session_module.MAX_HISTORY_MESSAGES


async def test_a_failed_answer_is_reported_not_swallowed(manager):
    model = _reply(manager, "gemini/gemini-2.5-flash", "")
    model.fail = RuntimeError("provider exploded")
    errors = []
    manager.errorOccurred.connect(errors.append)

    manager.send_text("what now?")
    await manager._answer_task

    assert errors == ["provider exploded"]
    assert manager.status != "error", "a transient failure leaves it armed"


async def test_a_permanent_failure_stops_pretending(manager):
    model = _reply(manager, "gemini/gemini-2.5-flash", "")
    model.fail = RuntimeError("API key not valid")

    manager.send_text("hello?")
    await manager._answer_task

    assert manager.status == "error"


def test_blank_questions_are_ignored(manager):
    manager.send_text("   ")
    assert manager._answer_task is None


# ----------------------------------------------------------------- observer


async def test_the_observer_writes_the_journal_and_says_nothing(manager):
    _reply(
        manager,
        "gemini/gemini-2.5-flash",
        '{"entries": [{"category": "location", "note": "Entered Firelink Shrine."}]}',
    )
    spoke = []
    manager.responseStarted.connect(lambda: spoke.append(True))
    manager.responseCompleted.connect(spoke.append)
    manager.send_frame(_frame())
    manager.trigger.pending.append(manager._latest_frame)

    await manager._run_observer("scene change")

    assert [e.note for e in manager.journal.entries] == ["Entered Firelink Shrine."]
    assert spoke == [], "the observer is silent, always"
    assert manager.observer_runs == 1


async def test_a_unified_observer_tick_joins_the_conversation(manager):
    _reply(manager, "gemini/gemini-2.5-flash", '{"entries": []}')
    manager.send_frame(_frame())
    manager.trigger.pending.append(manager._latest_frame)

    await manager._run_observer("heartbeat")

    assert len(manager._history) == 2, "the tick is remembered"
    assert _images(manager._history) == 0, "and its frames are not"
    assert manager.models["gemini/gemini-2.5-flash"].usage_kinds == ["nonlive_observer"]


async def test_a_split_observer_uses_its_own_model_and_leaves_history_alone(manager):
    manager.settings.agent_mode = "split"
    manager.settings.observer_model = "gemini/gemini-2.5-flash-lite"
    _reply(
        manager,
        "gemini/gemini-2.5-flash-lite",
        '{"entries": [{"category": "death", "note": "Died to the boss."}]}',
    )
    manager.journal.append("Entered the arena.", category="location")
    manager.send_frame(_frame())
    manager.trigger.pending.append(manager._latest_frame)

    await manager._run_observer("scene change")

    cheap = manager.models["gemini/gemini-2.5-flash-lite"]
    assert cheap.calls, "the observer model was the one called"
    assert "gemini/gemini-2.5-flash" not in manager.models
    assert manager._history == [], "a split observer keeps no conversation"
    assert "Entered the arena." in json.dumps(cheap.calls[0][-1]["content"])
    assert [e.note for e in manager.journal.entries][-1] == "Died to the boss."


async def test_an_observer_that_finds_nothing_records_nothing(manager):
    _reply(manager, "gemini/gemini-2.5-flash", '{"entries": []}')
    manager.send_frame(_frame())
    manager.trigger.pending.append(manager._latest_frame)

    await manager._run_observer("heartbeat")

    assert len(manager.journal) == 0


async def test_a_failing_observer_does_not_become_a_retry_loop(manager):
    model = _reply(manager, "gemini/gemini-2.5-flash", "")
    model.fail = RuntimeError("rate limited")
    manager.settings.observer.trigger_strategy = "spike_plain_heartbeat"
    manager.trigger.settings = manager.settings.observer
    manager.send_frame(_frame())
    manager.trigger.last_run = 0.0
    manager.trigger._last_heartbeat = 0.0

    errors = []
    manager.errorOccurred.connect(errors.append)
    manager._observer_task = asyncio.ensure_future(manager._observe_loop())
    await asyncio.sleep(1.2)
    manager._stopping = True
    await manager.stop()

    assert errors, "the failure is reported"
    assert manager.trigger.last_run > 0, "and the cooldown still applies to it"


async def test_stopping_ends_the_observer_loop(manager):
    manager.start()
    assert manager.status == "live"
    await manager.stop()
    assert manager.status == "stopped"
    assert manager._observer_task is None


# ------------------------------------------------------------------ prompts


def test_the_qa_prompt_says_where_its_memory_comes_from():
    prompt = build_qa_instruction(Settings(game_name="Elden Ring"))
    assert "Elden Ring" in prompt
    assert "journal" in prompt.lower()


def test_the_observer_prompt_is_told_nobody_is_listening():
    prompt = build_observer_instruction(Settings())
    assert "not talking to anyone" in prompt
    assert "JSON" in prompt


def test_the_observer_does_not_inherit_the_players_style_instructions():
    """ "Never spoil anything" must not stop the journal recording what happened."""
    settings = Settings(extra_system_prompt="Never spoil anything I haven't found.")
    assert "Never spoil" in build_qa_instruction(settings)
    assert "Never spoil" not in build_observer_instruction(settings)


def test_a_detected_game_is_hedged_and_a_named_one_is_not():
    detected = build_qa_instruction(Settings(), detected_game="Hollow Knight")
    named = build_qa_instruction(Settings(game_name="Hollow Knight"))
    assert "appears to be playing" in detected
    assert "appears to be playing" not in named
