"""The silent, tool-only Gemini Live Observer."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from chiron.capture.frames import Frame
from chiron.config.settings import Settings
from chiron.journal.log import JournalLog
from chiron.journal.service import JournalService
from chiron.observer.prompts import CHECKPOINT_INSTRUCTION, OBSERVER_SYSTEM_PROMPT
from chiron.observer.session import ObserverSessionManager, is_permanent_error


def frame(stamp: float) -> Frame:
    return Frame(jpeg=f"frame-{stamp}".encode(), captured_at=stamp, width=8, height=4)


@pytest.fixture
def observer(qapp):
    return ObserverSessionManager(
        Settings(api_key="test-key"), JournalService(JournalLog())
    )


class FakeSession:
    def __init__(self):
        self.realtime = []
        self.context = []
        self.tool_responses = []

    async def send_realtime_input(self, **kwargs):
        self.realtime.append(kwargs)

    async def send_client_content(self, **kwargs):
        self.context.append(kwargs)

    async def send_tool_response(self, **kwargs):
        self.tool_responses.append(kwargs)


def test_live_config_is_audio_but_has_no_transcription_or_response_surface(observer):
    config = observer._build_config()
    assert [str(item).lower() for item in config.response_modalities] == [
        "modality.audio"
    ]
    assert getattr(config, "output_audio_transcription", None) is None
    declarations = config.tools[0].function_declarations
    assert [declaration.name for declaration in declarations] == ["record_event"]
    assert not hasattr(observer, "responseDelta")
    assert not hasattr(observer, "responseCompleted")


def test_live_prompts_request_vivid_grounded_visual_memory():
    assert "visual memory for Chiron-Responder" in OBSERVER_SYSTEM_PROMPT
    assert "vivid, self-contained paragraph" in OBSERVER_SYSTEM_PROMPT
    assert "setting and spatial context" in OBSERVER_SYSTEM_PROMPT
    assert all(
        qualifier in OBSERVER_SYSTEM_PROMPT
        for qualifier in ('"appears,"', '"seems,"', '"is unclear."')
    )
    assert "Do not record unchanged scenery" in OBSERVER_SYSTEM_PROMPT
    assert "materially changed scene" in CHECKPOINT_INSTRUCTION
    assert "finish silently" in CHECKPOINT_INSTRUCTION


def test_manual_activity_detection_is_enabled(observer):
    config = observer._build_config()
    assert config.realtime_input_config.automatic_activity_detection.disabled is True
    assert config.explicit_vad_signal is None


def test_pending_ticks_coalesce_to_the_newest_frame(observer):
    observer.status = "live"
    observer.observe(frame(1.0))
    observer.observe(frame(2.0))
    assert observer._pending_frame[0].captured_at == 2.0


async def test_checkpoint_order_is_activity_video_text_activity(observer):
    session = FakeSession()
    calls = []
    observer.llmCall.connect(calls.append)
    task = asyncio.create_task(
        observer._send_observation(session, frame(1.0), "scheduled")
    )
    while len(session.realtime) < 4:
        await asyncio.sleep(0)

    assert [next(iter(call)) for call in session.realtime] == [
        "activity_start",
        "video",
        "text",
        "activity_end",
    ]
    observer._finish_observation()
    await task
    assert calls[0].agent_id == "observer"
    assert calls[0].kind == "observer_checkpoint"
    assert observer.last_observed_at == 1.0


async def test_transactions_cannot_interleave(observer):
    session = FakeSession()
    first = asyncio.create_task(
        observer._send_observation(session, frame(1.0), "scheduled")
    )
    second = asyncio.create_task(
        observer._send_observation(session, frame(2.0), "scheduled")
    )
    while len(session.realtime) < 4:
        await asyncio.sleep(0)
    await asyncio.sleep(0)
    assert len(session.realtime) == 4

    observer._finish_observation()
    await first
    while len(session.realtime) < 8:
        await asyncio.sleep(0)
    observer._finish_observation()
    await second
    assert [next(iter(call)) for call in session.realtime] == [
        "activity_start",
        "video",
        "text",
        "activity_end",
    ] * 2


async def test_zero_event_turn_finishes_without_a_journal_write(observer):
    observer._observation_in_flight = True
    observer._inflight_frame = frame(3.0)
    observer._turn_done.clear()
    observer._finish_observation()
    assert len(observer.journal.log) == 0
    assert observer._turn_done.is_set()


def test_discarded_audio_tokens_reach_checkpoint_cost(observer):
    records = []
    observer.llmCall.connect(records.append)
    observer._observation_in_flight = True
    observer._inflight_frame = frame(3.0)
    observer._inflight_output_tokens = 25
    observer._finish_observation()
    assert records[0].completion_tokens == 25


async def test_multiple_tool_calls_write_and_are_acknowledged(observer):
    calls = [
        SimpleNamespace(
            id="1",
            name="record_event",
            args={"note": "Found a key", "category": "item"},
        ),
        SimpleNamespace(
            id="2",
            name="record_event",
            args={"note": "Reached the tower", "category": "location"},
        ),
    ]
    session = FakeSession()
    await observer._handle_tool_call(session, SimpleNamespace(function_calls=calls))
    assert [entry.note for entry in observer.journal.log.entries] == [
        "Found a key",
        "Reached the tower",
    ]
    responses = session.tool_responses[0]["function_responses"]
    assert len(responses) == 2
    assert all(response.response["status"] == "recorded" for response in responses)


async def test_fresh_connection_replays_recent_journal(observer):
    observer.journal.record("Reached Moon Tower.", "location")
    session = FakeSession()

    await observer._seed(session)

    sent = str(session.context[0]["turns"])
    assert "Earlier gameplay journal" in sent
    assert "Reached Moon Tower." in sent


async def test_fresh_connection_uses_automatic_compacted_journal_view(observer):
    observer.journal.record("Reached Moon Tower.", "location")
    observer.journal.apply_summary("The player reached Moon Tower.", 1)
    observer.journal.record("Found the lift.", "progress")

    class CompactorSpy:
        def __init__(self, journal):
            self.journal = journal
            self.calls = []

        async def prepare(self, model, *, force=False, reason="threshold"):
            self.calls.append((model, force, reason))
            return self.journal.snapshot()

    compactor = CompactorSpy(observer.journal)
    observer.journal_compactor = compactor
    session = FakeSession()

    await observer._seed(session)

    sent = str(session.context[0]["turns"])
    assert "The player reached Moon Tower." in sent
    assert "Found the lift." in sent
    assert compactor.calls == [
        (observer.settings.observer_model, False, "observer_seed")
    ]


async def test_compaction_does_not_replay_into_the_context_being_discarded(observer):
    observer.journal.record("Found the lift key.", "item")
    observer._active_session = FakeSession()
    observer._resumption_handle = "old-context"

    await observer._compact_and_rotate()

    assert observer._active_session.context == []
    assert observer._resumption_handle is None
    assert observer._rotate_reason == "context compaction"


async def test_journal_context_is_included_in_estimated_cost(observer):
    session = FakeSession()
    observer.session_id = "session-1"
    records = []
    observer.llmCall.connect(records.append)
    await observer._send_context(session, "journal context")
    assert records[0].kind == "observer_context"
    assert records[0].agent_id == "observer"
    assert records[0].session_id == "session-1"
    assert records[0].pricing_source == "estimated"


async def test_context_falls_back_to_realtime_if_client_history_is_rejected(observer):
    class RejectingSession(FakeSession):
        async def send_client_content(self, **kwargs):
            raise RuntimeError("not accepted")

    session = RejectingSession()
    await observer._send_context(session, "journal context")
    assert session.realtime == [{"text": "journal context"}]


def test_new_session_drops_resumption_and_transient_visual_state(observer):
    observer._resumption_handle = "resume-old-context"
    observer.last_observed_at = 10.0
    observer._pending_frame = (frame(11.0), "scheduled")
    observer.reset_memory()
    assert observer._resumption_handle is None
    assert observer.last_observed_at is None
    assert observer._pending_frame is None


@pytest.mark.parametrize(
    "message",
    [
        "API key not valid",
        "permission_denied",
        "model was not found",
        "response modalities not supported by the model",
    ],
)
def test_configuration_errors_are_permanent(message):
    assert is_permanent_error(RuntimeError(message)) is True


def test_network_errors_remain_retryable():
    assert is_permanent_error(RuntimeError("connection reset")) is False
