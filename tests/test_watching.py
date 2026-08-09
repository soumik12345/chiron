"""Requested/effective Watch state and both question-frame policies."""

from __future__ import annotations

import asyncio

from chiron.capture.frames import Frame
from chiron.capture.service import CaptureService
from chiron.config.settings import CaptureSettings


def frame(stamp: float = 100.0) -> Frame:
    return Frame(jpeg=b"jpeg", captured_at=stamp, width=8, height=4)


async def settle() -> None:
    await asyncio.sleep(0)


def test_capture_service_starts_idle():
    assert CaptureService(CaptureSettings()).is_watching is False


def test_starting_capture_resets_the_fixed_deadline():
    service = CaptureService(CaptureSettings())
    service.scheduler.note_capture(1000.0)
    service.set_watching(True)
    assert service.scheduler.is_due(1000.1) is True


def test_immediate_request_does_not_move_the_periodic_deadline():
    service = CaptureService(CaptureSettings())
    service.set_watching(True)
    service.scheduler.note_capture(1000.0)
    assert service.request_immediate("question-1") is True
    assert service.scheduler.seconds_until_due(1002.0) == 3.0


def test_watching_is_off_by_default(app):
    assert app.watch_requested is False
    assert app.capture_active is False


def test_capture_waits_for_observer_live(app):
    def stay_connecting():
        app.observer.starts += 1
        app.observer.status = "connecting"

    app.observer.start = stay_connecting
    app.set_watching(True)

    assert app.watch_requested is True
    assert app.capture_active is False

    app.observer.status = "live"
    app.observer.statusChanged.emit("live", "ready")
    assert app.capture_active is True


async def test_toggle_starts_and_stops_observer_and_capture(app):
    app.toggle_watching()
    assert app.observer.starts == 1
    assert app.watch_requested is True
    assert app.capture_active is True

    app.toggle_watching()
    await settle()
    assert app.observer.stops == 1
    assert app.watch_requested is False
    assert app.capture_active is False


def test_repeating_a_watch_state_is_a_no_op(app):
    app.set_watching(True)
    app.set_watching(True)
    assert app.observer.starts == 1


def test_all_watch_bindings_are_registered(app):
    bindings = app._hotkey_bindings()
    assert bindings["toggle_watching"] == "ctrl+alt+w"
    assert "start_watching" in bindings
    assert "stop_watching" in bindings


def test_latest_policy_uses_only_the_active_spans_latest_frame(app):
    app.set_watching(True)
    current = frame()
    app._on_scheduled_frame(current)
    app._on_prompt("what is ahead?")

    question, attached, status = app.responder.questions[-1]
    assert question == "what is ahead?"
    assert attached is current
    assert status.stale is False

    app.set_watching(False)
    app._on_prompt("what was I doing?")
    assert app.responder.questions[-1][1] is None
    assert app.responder.questions[-1][2].stale is True


def test_observer_loss_stops_capture_and_invalidates_the_latest_frame(app):
    app.set_watching(True)
    app._on_scheduled_frame(frame())
    assert app._latest_frame is not None

    app.observer.status = "reconnecting"
    app.observer.statusChanged.emit("reconnecting", "network")

    assert app.watch_requested is True
    assert app.capture_active is False
    assert app._latest_frame is None


def test_reconnect_resumes_with_a_fresh_periodic_deadline(app):
    app.set_watching(True)
    app.capture.scheduler.note_capture(1000.0)
    app.observer.status = "reconnecting"
    app.observer.statusChanged.emit("reconnecting", "network")
    app.observer.status = "live"
    app.observer.statusChanged.emit("live", "ready")

    assert app.capture_active is True
    assert app.capture.scheduler.last_capture is None


def test_immediate_frame_reaches_both_agents_without_waiting_for_observer(app):
    app.settings.capture.question_frame_policy = "immediate"
    app.set_watching(True)
    app._on_prompt("read this menu")
    assert app._waiting_immediate is not None
    token = app._waiting_immediate[0]
    current = frame(101.0)

    app._on_immediate_frame(current, token)

    assert app.observer.frames[-1] == (current, "immediate")
    assert app.responder.questions[-1][1] is current


async def test_flush_before_answering_waits_for_buffered_observer(app):
    app.settings.capture.question_answer_policy = "flush_observer"
    app.observer.mode = "nonlive"
    flushed = []

    async def flush_and_wait():
        flushed.append(True)
        app.journal_service.record("Observed before answering.", "progress")
        return True

    app.observer.flush_and_wait = flush_and_wait
    app.set_watching(True)
    app._on_scheduled_frame(frame(100.0))
    app._on_prompt("what changed?")

    assert app.responder.questions == []
    await settle()

    assert flushed == [True]
    assert len(app.journal) == 1
    assert app.responder.questions[-1][0] == "what changed?"


async def test_failed_flush_keeps_the_question_pending_for_observer_recovery(app):
    app.settings.capture.question_answer_policy = "flush_observer"
    app.observer.mode = "nonlive"

    async def flush_and_wait():
        return False

    app.observer.flush_and_wait = flush_and_wait
    app.set_watching(True)
    app._on_prompt("what changed?")
    await settle()

    assert app.responder.questions == []
    assert app._blocked_flush_question is not None
    assert "waiting for Retry review or Discard" in app.overlay.transcript.toPlainText()


def test_an_immediate_request_degrades_if_observer_disconnects(app):
    app.settings.capture.question_frame_policy = "immediate"
    app.set_watching(True)
    app._on_prompt("what happened?")

    app.observer.status = "reconnecting"
    app.observer.statusChanged.emit("reconnecting", "network")

    assert app._waiting_immediate is None
    assert app.responder.questions[-1][1] is None
    assert app.responder.questions[-1][2].stale is True


def test_questions_while_observer_is_unavailable_warn_once(app):
    app._on_prompt("what is this boss weak to?")
    app._on_prompt("and this one?")
    transcript = app.overlay.transcript.toPlainText()
    assert transcript.count("stale journal") == 1
    assert [row[0] for row in app.responder.questions] == [
        "what is this boss weak to?",
        "and this one?",
    ]


def test_agent_errors_are_attributed_in_the_gameplay_record(app):
    app.ensure_session()
    app.observer.errorOccurred.emit("Live socket failed")
    app.responder.errorOccurred.emit("Responder rejected the request")
    app.recorder.flush()
    statuses = [
        event.payload
        for event in app.recorder.read_session(app.recorder.session_id)
        if event.type == "status" and event.payload["status"] == "error"
    ]
    assert [(row["agent_id"], row["detail"]) for row in statuses] == [
        ("observer", "Live socket failed"),
        ("responder", "Responder rejected the request"),
    ]


def test_width_change_invalidates_cached_frames(app):
    app.set_watching(True)
    app._on_scheduled_frame(frame())
    edited = app.settings.copy_deep()
    edited.capture.frame_width = 1024

    app.apply_settings(edited)

    assert app.capture.settings.frame_width == 1024
    assert app._latest_frame is None


def test_frame_width_and_live_media_resolution_are_independent(app):
    edited = app.settings.copy_deep()
    edited.capture.frame_width = 1920
    edited.capture.media_resolution = "high"
    app.apply_settings(edited)

    assert app.capture.settings.frame_width == 1920
    assert app.settings.capture.media_resolution == "high"


def test_app_accepts_openrouter_only_agent_settings(app):
    edited = app.settings.copy_deep()
    edited.api_key = ""
    edited.openrouter_api_key = "router-key"
    edited.observer_model = "openrouter/google/gemini-2.5-flash"
    edited.responder_model = "openrouter/google/gemini-2.5-flash"
    app.apply_settings(edited)
    assert app.settings.resolved_api_key() == ""
    assert app.settings.key_for_model(app.settings.observer_model) == "router-key"


def test_app_rejects_missing_selected_agent_credentials(app):
    edited = app.settings.copy_deep()
    edited.api_key = ""
    app.apply_settings(edited)
    assert app.settings.api_key == "test-key"
    assert "missing API key" in app.overlay.transcript.toPlainText()


def test_footer_distinguishes_requested_from_effective_watch(app):
    app.observer.start = lambda: None
    app.set_watching(True)
    app._refresh_footer()
    assert "paused" in app.overlay.footer.text()


def test_new_session_clears_both_agents_and_keeps_watch_requested(app):
    app.set_watching(True)
    app.journal_service.record("found a key")
    app.conversation.commit("where?", "in the tower")

    app.new_session()

    assert app.watch_requested is True
    assert app.capture_active is True
    assert len(app.journal) == 0
    assert len(app.conversation) == 0
    assert app.observer.memory_resets == 1
    assert app.responder.memory_resets == 1
