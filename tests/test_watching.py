"""Watching is opt-in: nothing reads the screen until asked.

These tests drive the real :class:`ChironApp` wiring — the capture service, the
overlay and the session are the production objects — but never call `start()`, so
no thread runs and no socket is opened.
"""

from __future__ import annotations

import asyncio

from chiron.capture.service import CaptureService
from chiron.config.settings import CaptureSettings


async def settle() -> None:
    """Let scheduled shutdown tasks run."""
    await asyncio.sleep(0)


# ------------------------------------------------------------------- service


def test_capture_service_starts_idle():
    service = CaptureService(CaptureSettings())
    assert service.is_watching is False


def test_capture_service_toggles():
    service = CaptureService(CaptureSettings())
    service.set_watching(True)
    assert service.is_watching is True
    service.set_watching(False)
    assert service.is_watching is False


def test_starting_clears_stale_scene_state():
    """The screen has moved on during a pause; the first frame back is not news."""
    service = CaptureService(CaptureSettings())
    service._last_signature = b"whatever was on screen before"
    service.scheduler.note_capture(1000.0)

    service.set_watching(True)

    assert service._last_signature == b""
    assert service.scheduler.last_capture is None


def test_starting_captures_immediately():
    service = CaptureService(CaptureSettings())
    service.scheduler.note_capture(1000.0)
    service.set_watching(True)
    assert service.scheduler.is_due(1000.1) is True, "no waiting out the old interval"


# ----------------------------------------------------------------------- app


def test_watching_is_off_by_default(app):
    assert app.watching is False
    assert app.settings.capture.watch_on_launch is False


async def test_toggle_starts_and_stops(app):
    app.toggle_watching()
    assert app.watching is True
    assert app.capture.is_watching is True

    app.toggle_watching()
    await settle()
    assert app.watching is False
    assert app.capture.is_watching is False


async def test_watching_drives_the_live_session(app):
    """Frames with nowhere to go are pure cost; a stopped watch leaves nothing open."""
    app.set_watching(True)
    assert app.session.starts == 1

    app.set_watching(False)
    await settle()
    assert app.session.stops == 1


async def test_repeating_a_state_is_a_no_op(app):
    app.set_watching(False)
    assert app.watching is False
    assert app.session.stops == 0, "not watching already; nothing to tear down"

    app.set_watching(True)
    app.set_watching(True)
    assert app.watching is True
    assert app.session.starts == 1


async def test_hotkeys_drive_the_state(app):
    app._on_hotkey("start_watching")
    assert app.watching is True

    app._on_hotkey("start_watching")
    assert app.watching is True, "start is idempotent"
    assert app.session.starts == 1

    app._on_hotkey("stop_watching")
    await settle()
    assert app.watching is False

    app._on_hotkey("stop_watching")
    await settle()
    assert app.watching is False, "stop is idempotent"
    assert app.session.stops == 1

    app._on_hotkey("toggle_watching")
    assert app.watching is True


def test_all_watch_bindings_are_registered(app):
    bindings = app._hotkey_bindings()
    assert bindings["toggle_watching"] == "ctrl+alt+w"
    assert "start_watching" in bindings
    assert "stop_watching" in bindings


async def test_overlay_reflects_the_state(app):
    app.set_watching(True)
    assert app.overlay.watching is True
    assert "Watching" in app.overlay.watch_button.text()

    app.set_watching(False)
    await settle()
    assert app.overlay.watching is False
    assert "Not watching" in app.overlay.watch_button.text()


async def test_overlay_button_asks_the_app_to_watch(app):
    app.overlay.watch_button.click()
    assert app.watching is True
    app.overlay.watch_button.click()
    await settle()
    assert app.watching is False


def test_footer_says_when_it_is_not_watching(app):
    app._refresh_footer()
    assert "not watching" in app.overlay.footer.text()

    app.set_watching(True)
    app._refresh_footer()
    assert "not watching" not in app.overlay.footer.text()


def test_a_question_while_not_watching_warns_once(app):
    app._on_prompt("what is this boss weak to?")
    transcript = app.overlay.transcript.toPlainText()
    assert "not watching" in transcript
    assert app.session.texts == ["what is this boss weak to?"], "still answered"

    before = transcript.count("not watching")
    app._on_prompt("and this one?")
    after = app.overlay.transcript.toPlainText().count("not watching")
    assert after == before, "the warning is not repeated for every question"


def test_a_question_while_watching_bursts_the_shutter(app):
    app.set_watching(True)
    app._on_prompt("what killed me?")
    assert app.capture.scheduler.burst_reason == "question"


def test_a_question_while_not_watching_does_not_burst(app):
    app._on_prompt("what killed me?")
    assert app.capture.scheduler.burst_reason == ""


async def test_stopping_is_recorded_in_the_transcript(app):
    app.set_watching(True)
    app.set_watching(False)
    await settle()
    transcript = app.overlay.transcript.toPlainText()
    assert "Watching your screen." in transcript
    assert "Stopped watching" in transcript


def test_starting_to_watch_resets_what_the_session_thinks_it_has_seen(app):
    """The screen moved on during the pause; the first frame back is not news."""
    app.set_watching(True)
    assert app.session.resets == 1


# ---------------------------------------------------------------- mode swap


async def test_choosing_a_non_live_model_swaps_the_provider(app):
    from chiron.journal.writers import ObserverJournal, ToolCallJournal
    from chiron.nonlive.session import NonLiveSessionManager

    _real_session(app)
    assert isinstance(app.writer, ToolCallJournal)

    edited = app.settings.copy_deep()
    edited.selected_model = "gemini/gemini-2.5-flash"
    app.apply_settings(edited)
    await settle()
    await settle()

    assert isinstance(app.session, NonLiveSessionManager)
    assert isinstance(app.writer, ObserverJournal), (
        "the two modes journal differently, so the writer goes with the provider"
    )
    assert "Non-live mode." in app.overlay.transcript.toPlainText()


async def test_the_detected_game_survives_a_mode_swap(app):
    app.session = _real_session(app)
    app.session.detected_game = "Hollow Knight"

    edited = app.settings.copy_deep()
    edited.selected_model = "gemini/gemini-2.5-flash"
    app.apply_settings(edited)
    await settle()
    await settle()

    assert app.session.detected_game == "Hollow Knight"


async def test_non_live_capture_width_follows_frame_detail(app):
    edited = app.settings.copy_deep()
    edited.selected_model = "gemini/gemini-2.5-flash"
    edited.capture.frame_width = 1920
    edited.capture.media_resolution = "low"

    app.apply_settings(edited)
    await settle()

    assert app.capture.settings.frame_width == 512
    assert app.settings.capture.frame_width == 1920, "the stored value is untouched"


def _real_session(app):
    """Replace the spy with the production provider the settings imply."""
    from chiron.session import build_session_provider

    session = build_session_provider(app.settings, app.journal, app.writer, app)
    app.session = session
    app._connect_session()
    return session
