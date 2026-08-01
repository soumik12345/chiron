"""The Live session manager's configuration, queueing and prompt assembly."""

from __future__ import annotations

import asyncio
import time
from types import SimpleNamespace

import pytest
from PIL import Image

from chiron.capture.frames import encode_frame
from chiron.config.settings import JournalSettings, Settings
from chiron.journal.log import JournalLog
from chiron.journal.writers import SidecarJournal, ToolCallJournal
from chiron.live.prompts import build_journal_context, build_system_instruction
from chiron.live.session import LiveSessionManager, _text_parts, is_permanent_error


@pytest.fixture
def manager(qapp):
    """A manager wired to a tool-call journal, never connected."""
    log = JournalLog()
    return LiveSessionManager(Settings(), log, ToolCallJournal(log))


# ------------------------------------------------------------------ prompts


def test_system_instruction_mentions_the_game_and_extras():
    settings = Settings(game_name="Hollow Knight", extra_system_prompt="No spoilers.")
    instruction = build_system_instruction(settings)
    assert "Hollow Knight" in instruction
    assert "No spoilers." in instruction
    assert "record_event" in instruction


def test_system_instruction_omits_tool_guidance_for_the_sidecar():
    instruction = build_system_instruction(Settings(), journal_enabled=False)
    assert "record_event" not in instruction
    assert "Speak ONLY when the player asks" in instruction


def test_journal_context_is_empty_when_there_is_nothing_to_say():
    assert build_journal_context("   ") == ""


def test_seed_context_explains_the_amnesia():
    text = build_journal_context("12:01 — [location] Entered the crypt.", is_seed=True)
    assert "do not remember" in text
    assert "Entered the crypt." in text


# ------------------------------------------------------------------- config


def test_config_enables_compression_and_resumption(manager):
    config = manager._build_config()
    assert config.context_window_compression.sliding_window is not None
    assert config.session_resumption is not None


def test_config_asks_for_audio_and_reads_the_transcription(manager):
    """Live models only serve AUDIO; the text in the overlay is their transcript."""
    config = manager._build_config()
    assert [str(m) for m in config.response_modalities] == ["Modality.AUDIO"]
    assert config.output_audio_transcription is not None


def test_config_installs_the_record_event_tool(manager):
    config = manager._build_config()
    declarations = config.tools[0].function_declarations
    assert [d.name for d in declarations] == ["record_event"]


def test_config_has_no_tools_under_the_sidecar_strategy(qapp):
    log = JournalLog()
    settings = Settings()
    settings.journal.strategy = "sidecar"
    writer = SidecarJournal(log, JournalSettings(strategy="sidecar"))
    manager = LiveSessionManager(settings, log, writer)
    assert manager._build_config().tools is None


def test_media_resolution_follows_settings(qapp):
    settings = Settings()
    settings.capture.media_resolution = "medium"
    log = JournalLog()
    manager = LiveSessionManager(settings, log, ToolCallJournal(log))
    assert "MEDIUM" in str(manager._build_config().media_resolution)


# ------------------------------------------------------------------ sending


def test_blank_prompts_are_ignored(manager):
    manager.send_text("   ")
    assert manager._queue.empty()


def test_prompts_are_queued(manager):
    manager.send_text("what is this boss weak to?")
    kind, payload = manager._queue.get_nowait()
    assert (kind, payload) == ("text", "what is this boss weak to?")


def test_only_the_newest_frame_waits_in_the_queue(manager):
    first = encode_frame(Image.new("RGB", (64, 36), (10, 10, 10)))
    second = encode_frame(Image.new("RGB", (64, 36), (200, 10, 10)))

    manager.send_frame(first)
    manager.send_frame(second)

    assert manager._queue.qsize() == 1, "a stale frame is replaced, not stacked"
    kind, _ = manager._queue.get_nowait()
    assert kind == "frame"
    assert manager._latest_frame is second


# ------------------------------------------------------------------ folding


def test_fold_sends_only_new_entries(manager):
    manager.journal.append("Entered Firelink Shrine.", category="location")
    assert manager.fold_journal() == 1

    kind, payload = manager._queue.get_nowait()
    assert kind == "context"
    assert "Entered Firelink Shrine." in payload

    assert manager.fold_journal() == 0, "nothing new to fold"
    assert manager._queue.empty()


def test_forced_fold_replays_the_recent_journal(manager):
    manager.journal.append("Died on the bridge.", category="death")
    manager.fold_journal()
    assert manager.fold_journal(force=True) == 1


def test_fold_respects_the_entry_limit(manager):
    manager.settings.journal.fold_entry_limit = 2
    for index in range(5):
        manager.journal.append(f"event {index}")
    assert manager.fold_journal() == 2


# ------------------------------------------------------------------ parsing


def test_text_parts_extracts_model_text():
    content = SimpleNamespace(
        model_turn=SimpleNamespace(
            parts=[
                SimpleNamespace(text="Fire "),
                SimpleNamespace(text=None),
                SimpleNamespace(text="works."),
            ]
        )
    )
    assert _text_parts(content) == ["Fire ", "works."]


def test_text_parts_tolerates_frames_without_a_turn():
    assert _text_parts(SimpleNamespace(model_turn=None)) == []


# ------------------------------------------------------- talking to a session


def _server_message(**fields) -> SimpleNamespace:
    """A server message with every field absent unless named."""
    defaults = {
        "session_resumption_update": None,
        "go_away": None,
        "tool_call": None,
        "server_content": None,
    }
    return SimpleNamespace(**{**defaults, **fields})


def _model_turn(*texts, turn_complete=False) -> SimpleNamespace:
    """A server_content payload carrying model text parts."""
    return SimpleNamespace(
        model_turn=SimpleNamespace(parts=[SimpleNamespace(text=t) for t in texts]),
        output_transcription=None,
        turn_complete=turn_complete,
    )


def _spoken(text, *, turn_complete=False) -> SimpleNamespace:
    """A server_content payload carrying an output-audio transcription chunk."""
    return SimpleNamespace(
        model_turn=None,
        output_transcription=SimpleNamespace(text=text),
        turn_complete=turn_complete,
    )


class FakeSession:
    """Records what was sent and replays scripted server messages.

    ``receive()`` covers one turn and then stops, exactly as the real SDK does,
    so ``turns`` is a list of per-turn message batches. Passing a flat list is
    treated as a single turn.
    """

    def __init__(self, messages=(), turns=None) -> None:
        self.turns = [list(t) for t in turns] if turns is not None else [list(messages)]
        self.receive_calls = 0
        self.messages = list(messages)
        self.client_content: list[tuple] = []
        self.realtime: list[dict] = []
        self.tool_responses: list[list] = []
        self.client_content_fails = False

    async def send_client_content(self, *, turns=None, turn_complete=True):
        if self.client_content_fails:
            raise RuntimeError("client content is not allowed on this model")
        self.client_content.append((turns, turn_complete))

    async def send_realtime_input(self, **kwargs):
        self.realtime.append(kwargs)

    async def send_tool_response(self, *, function_responses):
        self.tool_responses.append(function_responses)

    async def receive(self):
        batch = (
            self.turns[self.receive_calls]
            if self.receive_calls < len(self.turns)
            else []
        )
        self.receive_calls += 1
        for message in batch:
            yield message


async def test_answers_stream_into_signals(manager):
    deltas, completed = [], []
    manager.responseDelta.connect(deltas.append)
    manager.responseCompleted.connect(completed.append)

    await manager._receive(
        FakeSession(
            [
                _server_message(server_content=_model_turn("Fire ")),
                _server_message(
                    server_content=_model_turn("works.", turn_complete=True)
                ),
            ]
        )
    )

    assert deltas == ["Fire ", "works."]
    assert completed == ["Fire works."]
    assert manager._model_has_spoken is True


async def test_spoken_answers_stream_as_text(manager):
    """The native-audio path: transcription chunks reach the overlay as deltas."""
    deltas, completed = [], []
    manager.responseDelta.connect(deltas.append)
    manager.responseCompleted.connect(completed.append)

    await manager._receive(
        FakeSession(
            [
                _server_message(server_content=_spoken("It is ")),
                _server_message(server_content=_spoken("Paris.", turn_complete=True)),
            ]
        )
    )

    assert deltas == ["It is ", "Paris."]
    assert completed == ["It is Paris."]


def test_audio_payloads_are_ignored():
    """Chiron never plays audio; inline data parts must not reach the transcript."""
    content = SimpleNamespace(
        model_turn=SimpleNamespace(
            parts=[SimpleNamespace(text=None, inline_data=b"\x00\x01")]
        ),
        output_transcription=SimpleNamespace(text="spoken words"),
    )
    assert _text_parts(content) == ["spoken words"]


async def test_a_finished_answer_does_not_end_the_session(manager):
    """The SDK's iterator stops at each turn; the session must outlive that."""
    session = FakeSession(
        turns=[
            [_server_message(server_content=_spoken("Paris.", turn_complete=True))],
            [_server_message(server_content=_spoken("Tokyo.", turn_complete=True))],
            [],  # the socket finally closes
        ]
    )
    completed = []
    manager.responseCompleted.connect(completed.append)

    await manager._receive(session)

    assert completed == ["Paris.", "Tokyo."], "two turns over one connection"
    assert session.receive_calls == 3


async def test_stop_cannot_be_blocked_by_a_session_that_will_not_unwind(manager):
    """Quitting has to finish, even if a socket's cleanup is stuck."""

    async def stubborn():
        try:
            await asyncio.sleep(3600)
        except asyncio.CancelledError:
            await asyncio.sleep(0.3)  # ignores the cancellation for a while

    manager._task = asyncio.ensure_future(stubborn())
    await asyncio.sleep(0)

    started = time.monotonic()
    await manager.stop(timeout=0.05)
    elapsed = time.monotonic() - started

    assert elapsed < 0.25, "shutdown gave up rather than waiting on it"
    assert manager.status == "stopped"
    await asyncio.sleep(0.35)  # let the abandoned task finish, for a clean loop


async def test_stop_waits_for_a_session_that_unwinds_normally(manager):
    unwound = []

    async def polite():
        try:
            await asyncio.sleep(3600)
        except asyncio.CancelledError:
            unwound.append("cleaned up")
            raise

    manager._task = asyncio.ensure_future(polite())
    await asyncio.sleep(0)

    await manager.stop()

    assert unwound == ["cleaned up"], "normal cleanup still runs to completion"
    assert manager.status == "stopped"


def test_permanent_errors_are_recognised():
    assert is_permanent_error(
        RuntimeError(
            "1007 None. The requested combination of response modalities (TEXT) "
            "is not supported by the model."
        )
    )
    assert is_permanent_error(ValueError("API key not valid"))
    assert not is_permanent_error(OSError("connection reset by peer"))


async def test_go_away_triggers_rotation(manager):
    session = FakeSession(
        [
            _server_message(go_away=SimpleNamespace(time_left="5s")),
            _server_message(server_content=_model_turn("never read")),
        ]
    )
    deltas = []
    manager.responseDelta.connect(deltas.append)

    await manager._receive(session)

    assert manager._rotate_reason == "server GoAway"
    assert manager.status == "reconnecting"
    assert deltas == [], "the receive loop stops at GoAway"


async def test_resumption_handles_are_kept(manager):
    await manager._receive(
        FakeSession(
            [
                _server_message(
                    session_resumption_update=SimpleNamespace(
                        resumable=True, new_handle="handle-1"
                    )
                ),
                _server_message(
                    session_resumption_update=SimpleNamespace(
                        resumable=False, new_handle=None
                    )
                ),
            ]
        )
    )
    assert manager._resumption_handle == "handle-1"
    assert manager._build_config().session_resumption.handle == "handle-1"


async def test_tool_calls_reach_the_journal_and_are_answered(manager):
    session = FakeSession()
    tool_call = SimpleNamespace(
        function_calls=[
            SimpleNamespace(
                id="call-1",
                name="record_event",
                args={"note": "Lit the bonfire.", "category": "progress"},
            )
        ]
    )

    await manager._handle_tool_call(session, tool_call)

    assert manager.journal.entries[0].note == "Lit the bonfire."
    assert session.tool_responses[0][0].id == "call-1"
    assert session.tool_responses[0][0].response["status"] == "recorded"


async def test_first_message_seeds_and_later_ones_stream(manager):
    session = FakeSession()

    await manager._send_user_text(session, "where am I?")
    assert session.client_content and not session.realtime

    manager._model_has_spoken = True
    await manager._send_user_text(session, "and now?")
    assert session.realtime == [{"text": "and now?"}]


async def test_context_folds_without_asking_for_a_reply(manager):
    session = FakeSession()
    await manager._send_context(session, "12:01 — [location] Entered the crypt.")
    _, turn_complete = session.client_content[0]
    assert turn_complete is False


async def test_context_falls_back_to_realtime_when_refused(manager):
    session = FakeSession()
    session.client_content_fails = True
    await manager._send_context(session, "journal text")
    assert session.realtime == [{"text": "journal text"}]
