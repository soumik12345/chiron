"""Context compaction, in both modes.

Non-live compaction is measurable and so is tested as such: count, split,
summarise, and the two paths into it (a threshold crossed before sending, and a
provider rejecting a request anyway). Live compaction has no in-place rewrite to
assert on, so what is tested is the *policy* — when it fires, when it waits, and
the one thing that makes the rotation a compaction rather than a reconnect: the
resumption handle has to be gone.
"""

from __future__ import annotations

import asyncio
import time
from types import SimpleNamespace

import pytest
from PIL import Image

from chiron.capture.frames import encode_frame
from chiron.config.settings import Settings
from chiron.journal.log import JournalLog
from chiron.journal.writers import ObserverJournal, ToolCallJournal
from chiron.live import estimate
from chiron.live.session import (
    COMPACTION_DEADLINE_TOKENS,
    COMPACTION_TOKENS,
    QUIET_SECONDS,
    LiveSessionManager,
)
from chiron.nonlive import compaction as ctx, session as session_module
from chiron.nonlive.prompts import JOURNAL_HEADER
from chiron.nonlive.session import NonLiveSessionManager


class FakeModel:
    """A stand-in litellm wrapper that records what it was asked."""

    def __init__(self, model_id: str, reply: str = "") -> None:
        self.model_id = model_id
        self.reply = reply
        self.calls: list[list[dict]] = []
        self.kinds: list[str] = []
        self.fail_once: Exception | None = None

    async def acompletion(self, *, messages, tools=None, stream=False):
        self.calls.append(messages)
        if self.fail_once is not None:
            error, self.fail_once = self.fail_once, None
            raise error
        if stream:
            return _Chunks(self.reply)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=self.reply))],
            usage=None,
        )

    def record_usage(self, usage, *, response=None, context=None):
        self.kinds.append((context or {}).get("kind", ""))


class _Chunks:
    def __init__(self, text: str) -> None:
        self.text = text
        self.usage = None

    async def __aiter__(self):
        yield SimpleNamespace(
            choices=[SimpleNamespace(delta=SimpleNamespace(content=self.text))],
            usage=None,
        )


def _frame(when: float | None = None):
    return encode_frame(
        Image.new("RGB", (64, 36), (20, 30, 40)),
        width=64,
        captured_at=when or time.time(),
    )


def _exchange(n: int) -> list[dict]:
    """One remembered question-and-answer pair."""
    return [
        {"role": "user", "content": f"question {n} " + "x" * 200},
        {"role": "assistant", "content": f"answer {n} " + "y" * 200},
    ]


# ------------------------------------------------------------------ counting


def test_images_are_charged_flat_not_as_base64():
    """A data URI is text to a token counter, and enormous."""
    huge = "A" * 400_000
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "what is this"},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{huge}"},
                },
            ],
        }
    ]

    counted = ctx.count_tokens("gemini/gemini-2.5-flash", messages)

    assert counted < 1000, "the base64 payload must not be counted as prose"
    assert counted >= ctx.IMAGE_TOKEN_ALLOWANCE


def test_the_text_projection_keeps_the_words_and_drops_the_pixels():
    projected = ctx.text_projection(
        [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "hello"},
                    {"type": "image_url", "image_url": {"url": "data:..."}},
                ],
            }
        ]
    )
    assert projected[0]["content"] == "hello\n[image]"


def test_counting_survives_a_model_id_nothing_has_heard_of():
    """A pasted OpenRouter id must still yield a number, or the guard stops running."""
    counted = ctx.count_tokens(
        "openrouter/nobody/never-shipped", [{"role": "user", "content": "hi there"}]
    )
    assert counted > 0


def test_an_unknown_window_falls_back_rather_than_failing():
    assert ctx.context_window_for("openrouter/nobody/never-shipped") > 0


def test_the_threshold_is_a_fraction_of_the_window():
    # Real prose rather than a repeated character: a long run of one letter is
    # a handful of BPE tokens however long it is, which would make this pass or
    # fail for reasons that have nothing to do with the threshold.
    messages = [{"role": "user", "content": "where do I go next in this area? " * 600}]
    tokens = ctx.count_tokens("gemini/gemini-2.5-flash", messages)

    roomy = int(tokens / ctx.COMPACTION_TRIGGER_RATIO) + 100
    tight = int(tokens / ctx.COMPACTION_TRIGGER_RATIO) - 100

    assert (
        ctx.should_compact("gemini/gemini-2.5-flash", messages, window=roomy) is False
    )
    assert ctx.should_compact("gemini/gemini-2.5-flash", messages, window=tight) is True


# ----------------------------------------------------------------- splitting


def test_the_kept_tail_never_opens_on_an_orphaned_answer():
    history = [m for n in range(6) for m in _exchange(n)]

    head, tail = ctx.split_history(history, keep_recent=5)

    assert tail[0]["role"] == "user"
    assert head + tail == history


def test_a_short_history_has_nothing_safe_to_drop():
    history = _exchange(1)
    head, tail = ctx.split_history(history, keep_recent=8)

    assert head == []
    assert tail == history


def test_an_earlier_summary_is_updated_rather_than_resummarised():
    history = [ctx.summary_message("earlier"), *_exchange(1)]

    previous, rest = ctx.extract_previous_summary(history)

    assert previous == "earlier"
    assert rest == _exchange(1)
    request = ctx.build_summary_request(rest, previous=previous)
    assert "earlier-summary" in request[1]["content"]


def test_a_first_compaction_asks_for_a_summary_not_an_update():
    request = ctx.build_summary_request(_exchange(1))
    assert "earlier-summary" not in request[1]["content"]
    assert "Summarise" in request[1]["content"]


def test_the_summariser_is_told_not_to_repeat_the_journal():
    """In this mode the journal rides in every request already."""
    request = ctx.build_summary_request(
        _exchange(1), journal="12:00 [death] died on the bridge"
    )
    prompt = request[1]["content"]

    assert "died on the bridge" in prompt
    assert "<journal>" in prompt
    assert "do not" in prompt.lower()


def test_serialising_history_drops_images_and_keeps_roles():
    text = ctx.serialise_history(
        [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "look"},
                    {"type": "image_url", "image_url": {"url": "data:..."}},
                ],
            }
        ]
    )
    assert text == "user: look"


# -------------------------------------------------------------- overflow


def test_the_typed_overflow_error_is_recognised():
    import litellm

    error = litellm.ContextWindowExceededError(
        message="too long", model="m", llm_provider="openai"
    )
    assert ctx.is_context_overflow(error) is True


def test_an_overflow_described_only_in_words_is_still_recognised():
    assert (
        ctx.is_context_overflow(RuntimeError("maximum context length is 8192")) is True
    )


def test_an_ordinary_failure_is_not_an_overflow():
    assert ctx.is_context_overflow(RuntimeError("connection reset")) is False


# --------------------------------------------------------- the non-live path


@pytest.fixture
def manager(qapp, monkeypatch):
    """A non-live manager wired to fake models, never touching the network."""
    monkeypatch.setattr(session_module, "FRESH_FRAME_WAIT_SECONDS", 0.01)
    settings = Settings(api_key="AIza-test", selected_model="gemini/gemini-2.5-flash")
    log = JournalLog()
    manager = NonLiveSessionManager(settings, log, ObserverJournal(log))
    manager.models = {}

    def fake_model(model_id: str, *, max_tokens: int = 600):
        return manager.models.setdefault(model_id, FakeModel(model_id))

    monkeypatch.setattr(manager, "_model", fake_model)
    return manager


def _model(manager, reply: str = "ok") -> FakeModel:
    model = manager.models.setdefault(
        "gemini/gemini-2.5-flash", FakeModel("gemini/gemini-2.5-flash")
    )
    model.reply = reply
    return model


async def test_the_journal_block_rides_in_the_request_and_not_into_history(manager):
    manager.journal.append("Lit the bonfire.", category="progress")
    model = _model(manager, "north")
    manager.send_frame(_frame())

    await manager._answer("where now?", time.time())

    sent = model.calls[0]
    assert JOURNAL_HEADER in sent[-1]["content"][0]["text"]
    remembered = manager._history[0]["content"][0]["text"]
    assert remembered == "where now?"
    assert JOURNAL_HEADER not in remembered


async def test_twenty_exchanges_carry_one_journal_not_twenty(manager):
    """The amplifier that made the wall far nearer than 'forty messages' suggests."""
    manager.journal.append("Lit the bonfire.", category="progress")
    _model(manager, "ok")
    for n in range(10):
        await manager._answer(f"question {n}", time.time())

    copies = sum(
        JOURNAL_HEADER in part.get("text", "")
        for message in manager._history
        if isinstance(message.get("content"), list)
        for part in message["content"]
        if isinstance(part, dict)
    )
    assert copies == 0


async def test_a_full_history_is_compacted_before_the_next_question(
    manager, monkeypatch
):
    monkeypatch.setattr(ctx, "context_window_for", lambda model_id: 1200)
    manager._history = [m for n in range(12) for m in _exchange(n)]
    model = _model(manager, "a summary of what happened")

    await manager._answer("what now?", time.time())

    assert manager.compactions == 1
    assert manager._history[0]["content"].startswith(ctx.COMPACTION_SUMMARY_PREFIX)
    assert "compaction" in model.kinds


async def test_compaction_keeps_the_recent_tail_verbatim(manager, monkeypatch):
    monkeypatch.setattr(ctx, "context_window_for", lambda model_id: 1200)
    manager._history = [m for n in range(12) for m in _exchange(n)]
    _model(manager, "summary")

    await manager._compact_history("threshold")

    assert len(manager._history) == 1 + ctx.KEEP_RECENT_MESSAGES
    assert manager._history[-1]["content"].startswith("answer 11")


async def test_a_rejected_request_is_compacted_and_retried_once(manager, monkeypatch):
    """The wall: without this, every later question re-assembles the same doomed request."""
    import litellm

    monkeypatch.setattr(ctx, "context_window_for", lambda model_id: 10_000_000)
    manager._history = [m for n in range(12) for m in _exchange(n)]
    model = _model(manager, "the answer")
    model.fail_once = litellm.ContextWindowExceededError(
        message="too long", model="m", llm_provider="gemini"
    )

    await manager._answer("what now?", time.time())

    assert manager.compactions == 1
    assert manager._history[0]["content"].startswith(ctx.COMPACTION_SUMMARY_PREFIX)


async def test_an_ordinary_failure_is_not_retried(manager, monkeypatch):
    monkeypatch.setattr(ctx, "context_window_for", lambda model_id: 10_000_000)
    model = _model(manager, "unused")
    model.fail_once = RuntimeError("connection reset")
    errors: list[str] = []
    manager.errorOccurred.connect(errors.append)

    await manager._answer("what now?", time.time())

    assert manager.compactions == 0
    assert errors and "connection reset" in errors[0]


async def test_a_failed_summariser_leaves_history_exactly_as_it_was(
    manager, monkeypatch
):
    monkeypatch.setattr(ctx, "context_window_for", lambda model_id: 1200)
    history = [m for n in range(12) for m in _exchange(n)]
    manager._history = list(history)
    model = _model(manager, "")
    model.fail_once = RuntimeError("summariser is down")

    outcome = await manager._compact_history("threshold")

    assert outcome is None
    assert manager._history == history


async def test_compaction_announces_itself(manager, monkeypatch):
    monkeypatch.setattr(ctx, "context_window_for", lambda model_id: 1200)
    manager._history = [m for n in range(12) for m in _exchange(n)]
    _model(manager, "summary")
    seen: list[dict] = []
    manager.compacted.connect(seen.append)

    await manager._compact_history("threshold")

    assert seen and seen[0]["mode"] == "nonlive"
    assert seen[0]["tokens_after"] < seen[0]["tokens_before"]
    assert seen[0]["kept_tail_count"] == ctx.KEEP_RECENT_MESSAGES


async def test_a_second_compaction_updates_the_first_summary(manager, monkeypatch):
    monkeypatch.setattr(ctx, "context_window_for", lambda model_id: 1200)
    manager._history = [m for n in range(12) for m in _exchange(n)]
    model = _model(manager, "summary one")
    await manager._compact_history("threshold")

    manager._history.extend(m for n in range(20, 32) for m in _exchange(n))
    model.reply = "summary two"
    await manager._compact_history("threshold")

    request = model.calls[-1][1]["content"]
    assert "summary one" in request, (
        "the earlier summary is updated, not summarised again"
    )
    assert manager._history[0]["content"].endswith("summary two")


# ------------------------------------------------------- non-live plumbing


def test_every_model_built_carries_the_usage_sink(manager):
    """The seam was installed and never connected; a throwaway with no sink is
    a call that was made and never booked."""
    manager.session_id = "2026-08-02-hades-a1b2"

    # The fixture replaces `_model`; call the real one to see what it builds.
    model = NonLiveSessionManager._model(manager, "gemini/gemini-2.5-flash")

    assert isinstance(model, session_module.LiteLLMModel)
    assert model.usage_sink == manager._on_usage_record
    assert model.usage_labels == {"session_id": "2026-08-02-hades-a1b2"}


def test_a_booked_call_is_published_and_remembered(manager):
    from chiron.models.usage import LLMCallRecord

    records = []
    manager.llmCall.connect(records.append)

    manager._on_usage_record(LLMCallRecord(id="abc123", model_id="m", cost_usd=0.01))

    assert [r.id for r in records] == ["abc123"]
    assert manager._last_call_id == "abc123"


def test_the_apps_own_call_kinds_actually_validate():
    """They used to raise on construction and die in a silent except."""
    from chiron.models.usage import LLMCallRecord

    for kind in (
        "nonlive_qa",
        "nonlive_observer",
        "journal_sidecar",
        "compaction",
        "live_turn",
    ):
        assert LLMCallRecord(kind=kind).kind == kind


async def test_frames_are_announced_as_they_reach_the_model(manager):
    _model(manager, "ok")
    sent: list[tuple] = []
    manager.frameSent.connect(lambda frame, reason: sent.append((frame, reason)))
    manager.send_frame(_frame())

    await manager._answer("what is this?", time.time())

    assert sent and all(reason == "question" for _, reason in sent)


async def test_an_observer_run_reports_what_it_looked_at_and_found(manager):
    _model(
        manager, '{"entries": [{"category": "location", "note": "Entered Limgrave."}]}'
    )
    runs: list[dict] = []
    manager.observerRan.connect(runs.append)
    manager.trigger.observe(_frame(), time.time())

    await manager._run_observer("scene change")

    assert runs[0]["reason"] == "scene change"
    assert runs[0]["entries"] == 1
    assert runs[0]["frames"], "the frames it was shown, for the record"


async def test_an_observer_tick_with_nothing_to_look_at_is_recorded_as_skipped(manager):
    runs: list[dict] = []
    manager.observerRan.connect(runs.append)

    await manager._run_observer("heartbeat")

    assert runs[0]["skipped"] is True


def test_a_new_gameplay_session_empties_the_conversation(manager):
    manager._history = [m for n in range(4) for m in _exchange(n)]
    manager._conversation.append((time.time(), "player", "hi"))
    manager.send_frame(_frame())

    manager.reset_memory()

    assert manager._history == []
    assert list(manager._conversation) == []
    assert manager._latest_frame is None


# ------------------------------------------------------------ the live path


@pytest.fixture
def live(qapp):
    """A live manager that never opens a socket."""
    log = JournalLog()
    settings = Settings(api_key="AIza-test")
    return LiveSessionManager(settings, log, ToolCallJournal(log))


def test_the_estimate_grows_with_frames_and_talk(live):
    assert live.estimated_context_tokens == 0
    live._context_frames = 100
    live._context_chars = 4000
    assert live.estimated_context_tokens == 100 * 260 + 1000


def test_the_servers_own_count_wins_when_it_sends_one(live):
    live._context_frames = 100
    live._server_tokens = 42_000
    assert live.estimated_context_tokens == 42_000


def test_usage_metadata_is_read_off_a_server_message():
    message = SimpleNamespace(usage_metadata=SimpleNamespace(total_token_count=1234))
    assert estimate.read_usage_metadata(message) == 1234
    assert estimate.read_usage_metadata(SimpleNamespace()) is None


def test_compaction_waits_for_a_quiet_moment(live):
    live.status = "live"
    live._server_tokens = COMPACTION_TOKENS + 1
    now = time.time()
    live._last_spike_at = now

    assert live.compaction_due(now) == "", "mid-fight is the wrong time to blink"
    assert live.compaction_due(now + QUIET_SECONDS + 1) == "quiet"


def test_a_pending_question_is_never_a_quiet_moment(live):
    live._server_tokens = COMPACTION_TOKENS + 1
    live._pending_question = True

    assert live.is_quiet(time.time()) is False
    assert live.compaction_due() == ""


def test_gating_cannot_postpone_into_eviction(live):
    """The hard deadline: rotate anyway, mid-action, rather than silently forget."""
    live._server_tokens = COMPACTION_DEADLINE_TOKENS + 1
    live._pending_question = True
    live._last_spike_at = time.time()

    assert live.compaction_due() == "deadline"


def test_below_the_threshold_nothing_is_due(live):
    live._server_tokens = COMPACTION_TOKENS - 1
    assert live.compaction_due(time.time() + 1000) == ""


async def test_a_compaction_rotate_drops_the_resumption_handle(live):
    """Otherwise resumption faithfully restores the context being shed."""
    live._resumption_handle = "handle-from-the-server"
    live.journal.append("Lit the bonfire.", category="progress")
    seen: list[dict] = []
    live.compacted.connect(seen.append)

    await live._compact_and_rotate("quiet")

    assert live._resumption_handle is None
    assert seen and seen[0]["mode"] == "live_rotate"


def test_an_ordinary_rotate_keeps_the_handle(live):
    live._resumption_handle = "handle"
    live.rotate("server GoAway")
    assert live._resumption_handle == "handle"


async def test_compaction_folds_the_journal_before_rotating(live):
    live.journal.append("Entered Limgrave.", category="location")

    await live._compact_and_rotate("deadline")

    kinds = [live._queue.get_nowait()[0] for _ in range(live._queue.qsize())]
    assert "context" in kinds, (
        "the facts reach the old session too, in case rotation fails"
    )


async def test_a_sidecar_pass_runs_before_the_rotation(qapp):
    """In live mode the journal *is* the running summary, so it must be current."""
    log = JournalLog()

    class Sidecar(ToolCallJournal):
        def __init__(self, log):
            super().__init__(log)
            self.passes = 0

        async def summarise_once(self):
            self.passes += 1
            return []

    writer = Sidecar(log)
    manager = LiveSessionManager(Settings(api_key="AIza-test"), log, writer)

    await manager._compact_and_rotate("quiet")

    assert writer.passes == 1


async def test_a_broken_sidecar_does_not_stop_the_rotation(qapp):
    log = JournalLog()

    class Broken(ToolCallJournal):
        async def summarise_once(self):
            raise RuntimeError("summariser is down")

    manager = LiveSessionManager(Settings(api_key="AIza-test"), log, Broken(log))
    manager._resumption_handle = "handle"

    await manager._compact_and_rotate("deadline")

    assert manager._resumption_handle is None
    assert manager.compactions == 1


def test_a_new_gameplay_session_clears_the_server_side_context(live):
    live._resumption_handle = "handle"
    live._context_frames = 500

    live.reset_memory()

    assert live._resumption_handle is None
    assert live._context_frames == 0


# ------------------------------------------------------ live cost estimation


def test_a_live_turn_is_priced_by_frames_and_transcript():
    record = estimate.estimate_turn(
        model_id="live/gemini-3.1-flash-live-preview",
        frames=10,
        media_resolution="low",
        prompt_text="where do I go?",
        output_text="north past the bridge",
    )

    assert record.prompt_tokens == 10 * 260 + estimate.text_tokens("where do I go?")
    assert record.completion_tokens == estimate.text_tokens("north past the bridge")
    assert record.cost_usd > 0


def test_every_live_row_says_it_is_an_estimate():
    record = estimate.estimate_turn(
        model_id="live/anything", frames=1, media_resolution="low", output_text="hi"
    )
    assert record.pricing_source == "estimated"
    assert record.kind == "live_turn"


def test_frame_detail_changes_what_a_frame_costs():
    low = estimate.tokens_per_frame("low")
    high = estimate.tokens_per_frame("high")
    assert high > low
    assert estimate.tokens_per_frame("nonsense") == low


def test_an_unknown_live_model_is_priced_rather_than_called_free():
    """Zero would read as 'free', which is a claim, and the wrong one."""
    pricing = estimate.live_pricing("live/gemini-9-flash-live-preview")
    assert pricing.prompt > 0 and pricing.completion > 0


def test_nothing_happened_emits_no_row(live):
    records = []
    live.llmCall.connect(records.append)
    live._emit_turn_estimate("")
    assert records == []


def test_frames_with_no_question_are_still_billed_at_the_end(live):
    """A watched evening with no questions asked must not estimate at zero."""
    records = []
    live.llmCall.connect(records.append)
    live._turn_frames = 40

    live._emit_turn_estimate("")

    assert records and records[0].prompt_tokens == 40 * 260


async def test_stopping_flushes_the_last_unbilled_frames(live):
    records = []
    live.llmCall.connect(records.append)
    live._turn_frames = 5

    await live.stop(timeout=0.1)

    assert records and records[0].prompt_tokens == 5 * 260


def test_an_estimated_turn_is_billed_to_the_open_session(live):
    live.session_id = "2026-08-02-hades-a1b2"
    records = []
    live.llmCall.connect(records.append)
    live._turn_frames = 1

    live._emit_turn_estimate("hello")

    assert records[0].session_id == "2026-08-02-hades-a1b2"


def test_the_estimate_and_the_settings_page_share_one_arithmetic():
    assert (
        estimate.estimate_context_tokens(
            frames=10, media_resolution="low", transcript_chars=400
        )
        == 10 * 260 + 100
    )


async def test_asyncio_is_actually_running():
    """Guards the auto asyncio mode the rest of this file leans on."""
    await asyncio.sleep(0)


def test_the_live_sidecar_books_its_own_calls(qapp):
    """The one part of live mode that reaches litellm, and so is measured."""
    from chiron.journal.writers import SidecarJournal, build_journal_writer

    def sink(record):  # pragma: no cover - identity is what is asserted
        pass

    writer = build_journal_writer(
        Settings(journal={"strategy": "sidecar"}).journal,
        JournalLog(),
        usage_sink=sink,
    )

    assert isinstance(writer, SidecarJournal)
    assert writer.model.usage_sink is sink


def test_the_recorder_stamps_the_session_onto_a_call_that_did_not_know_it(
    qapp, tmp_path
):
    """The sidecar's model is built once and outlives several sessions."""
    from chiron.models.usage import LLMCallRecord
    from chiron.sessions.recorder import SessionRecorder

    recorder = SessionRecorder(tmp_path / "sessions")
    session_id = recorder.ensure_session(game="Hades", mode="live")

    recorder.record_llm_call(
        LLMCallRecord(model_id="gemini/flash", kind="journal_sidecar")
    )
    recorder.flush()

    call = next(e for e in recorder.read_session(session_id) if e.type == "llm_call")
    assert call.payload["session_id"] == session_id
