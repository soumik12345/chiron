"""The views in one overlay, and the session lifecycle behind them.

The overlay tests drive real widgets on Qt's offscreen backend. The app tests
use the stubbed session provider from `conftest`, so nothing here opens a socket
or reaches a model — what is being checked is the wiring, which is where a
feature made of signals actually lives.
"""

from __future__ import annotations

import time

import pytest
from PIL import Image

from chiron.capture.frames import encode_frame
from chiron.config.settings import OverlaySettings, Settings
from chiron.models.usage import LLMCallRecord
from chiron.sessions.store import SessionRow
from chiron.ui.overlay import (
    PLAY_VIEW,
    REVIEW_MIN_HEIGHT,
    SESSION_VIEW,
    OverlayWindow,
)


@pytest.fixture
def overlay(qapp):
    """A real overlay panel on the offscreen backend."""
    return OverlayWindow(OverlaySettings())


def _row(session_id="s1", **kwargs) -> SessionRow:
    defaults = {
        "title": "Elden Ring — Aug 2",
        "game": "Elden Ring",
        "created_at": time.time(),
        "last_active_at": time.time(),
        "message_count": 4,
        "disk_bytes": 2048,
    }
    return SessionRow(id=session_id, **{**defaults, **kwargs})


def _frame():
    return encode_frame(Image.new("RGB", (64, 36), (10, 20, 30)), width=64)


# ------------------------------------------------------------- the view stack


def test_the_panel_opens_on_the_game(overlay):
    assert overlay.current_view == PLAY_VIEW


def test_the_viewer_is_one_step_from_play(overlay):
    """Choosing a session is a popup, so reading one is the only page transition."""
    overlay.show_session_viewer()
    assert overlay.current_view == SESSION_VIEW


def test_escape_walks_back_before_it_hides(overlay):
    from PySide6.QtCore import Qt
    from PySide6.QtGui import QKeyEvent

    def escape():
        overlay.keyPressEvent(
            QKeyEvent(
                QKeyEvent.Type.KeyPress,
                Qt.Key.Key_Escape,
                Qt.KeyboardModifier.NoModifier,
            )
        )

    overlay.show()
    overlay.show_session_viewer()

    escape()
    assert overlay.current_view == PLAY_VIEW
    assert overlay.isVisible() is True

    escape()
    assert overlay.isVisible() is False, "from play, Escape still hides the panel"


def test_reviewing_grows_the_panel_and_returning_shrinks_it(overlay):
    play = (overlay.width(), overlay.height())

    overlay.show_session_viewer()
    assert overlay.height() >= REVIEW_MIN_HEIGHT

    overlay.show_play()
    assert (overlay.width(), overlay.height()) == play


def test_a_grown_panel_never_becomes_the_remembered_placement(overlay):
    """Opening a session once must not permanently resize what floats over the game."""
    overlay.show_session_viewer()

    _, _, width, height = overlay.current_geometry()

    assert (width, height) == (OverlaySettings().width, OverlaySettings().height)


def test_the_prompt_is_hidden_while_reviewing(overlay):
    """There is nothing to ask a session that finished on Tuesday."""
    overlay.show_session_viewer()
    assert overlay.input.isVisible() is False

    overlay.show_play()
    overlay.show()
    assert overlay.input.isVisible() is True


def test_an_incoming_answer_snaps_back_to_the_game(overlay):
    overlay.show_session_viewer()

    overlay.start_response()

    assert overlay.current_view == PLAY_VIEW


# ---------------------------------------------------------------- the header


def test_no_session_yet_is_a_state_not_an_error(overlay):
    overlay.set_session("", "")
    assert "no session" in overlay.session_button.text()
    assert overlay.session_button.isEnabled() is False


def test_the_open_session_is_named_and_renameable(overlay):
    overlay.set_session("s1", "Elden Ring — Aug 2")
    assert "Elden Ring — Aug 2" in overlay.session_button.text()
    assert overlay.session_button.isEnabled() is True


def test_the_session_controls_sit_at_the_right_edge(overlay):
    """A QToolButton's policy is Fixed, so a stretch *factor* on the title never
    grows it — the layout scatters the slack between the controls instead, and
    the two buttons end up floating in the middle of the row."""
    overlay.show()
    overlay.set_session("s1", "Elden Ring — Aug 2")

    row = overlay.new_session_button.parentWidget()
    right_edge = row.width() - row.layout().contentsMargins().right()
    history = overlay.history_button.geometry()

    assert history.x() + history.width() == pytest.approx(right_edge, abs=2)
    assert overlay.new_session_button.x() > overlay.session_button.geometry().right()


def test_a_long_title_is_elided_rather_than_shoving_the_controls_away(overlay):
    overlay.show()
    overlay.set_session("s1", "Hades")
    settled = overlay.history_button.geometry().x()

    overlay.set_session(
        "s1", "The night I finally beat Margit the Fell Omen after nine hours"
    )

    assert overlay.history_button.geometry().x() == settled
    assert overlay.session_button.text().endswith("…")
    assert "nine hours" in overlay.session_button.toolTip(), "the full title survives"


def test_the_footer_carries_the_cost_beside_the_counters(overlay):
    overlay.set_cost(0.43, True)
    overlay.set_footer("frames sent: 12")

    assert overlay.footer.text() == "frames sent: 12  ·  ~$0.43"


def test_a_measured_cost_carries_no_tilde(overlay):
    overlay.set_cost(1.5, False)
    overlay.set_footer("frames sent: 12")

    assert "~" not in overlay.footer.text()
    assert "$1.50" in overlay.footer.text()


def test_a_refreshed_counter_line_keeps_the_cost(overlay):
    """The two are updated by different things at different rates."""
    overlay.set_cost(0.2, False)
    overlay.set_footer("first")
    overlay.set_footer("second")

    assert overlay.footer.text() == "second  ·  $0.20"


# -------------------------------------------------------- the session picker


def test_the_list_shows_what_the_index_knows(overlay):
    overlay.picker.set_sessions([_row("a"), _row("b")], total_bytes=4096)

    assert overlay.picker.list.count() == 2
    assert "4 KB on disk" in overlay.picker.total.text()


def test_the_pinned_total_is_the_retention_policy(overlay):
    overlay.picker.set_sessions([], total_bytes=0)
    assert "No sessions" in overlay.picker.total.text()


def test_search_filters_by_title_and_game(overlay):
    overlay.picker.set_sessions(
        [
            _row("a", title="Elden Ring — Aug 2", game="Elden Ring"),
            _row("b", title="Hades run", game="Hades"),
        ]
    )

    overlay.picker.search.setText("hades")

    assert overlay.picker.list.count() == 1


def test_tapping_a_row_asks_for_it_and_closes_the_popup(overlay):
    opened: list[str] = []
    overlay.sessionOpened.connect(opened.append)
    overlay.picker.set_sessions([_row("a")])
    overlay.picker.show()

    overlay.picker.list.itemClicked.emit(overlay.picker.list.item(0))

    assert opened == ["a"]
    assert overlay.picker.isVisible() is False


def test_the_picker_never_resizes_the_panel(overlay):
    """The whole point of a popup: choosing costs no geometry."""
    before = (overlay.width(), overlay.height())
    overlay.picker.set_sessions([_row("a")])

    overlay.open_session_picker()

    assert (overlay.width(), overlay.height()) == before
    assert overlay.current_view == PLAY_VIEW


def test_managing_a_session_happens_on_the_one_being_looked_at(overlay):
    """Rename and the two deletes moved off list rows and into the viewer."""
    deleted: list[str] = []
    overlay.sessionDeleted.connect(deleted.append)
    overlay.viewer.show_session(_row("a"), [])

    overlay.viewer.deleteRequested.emit(overlay.viewer.session_id)

    assert deleted == ["a"]


def test_nothing_can_be_managed_when_nothing_is_open(overlay):
    overlay.viewer.clear()
    assert overlay.viewer.manage_button.isEnabled() is False


def test_the_viewer_pins_the_summary_and_renders_the_events(overlay):
    from chiron.sessions.events import SessionEvent

    row = _row("a", journal_count=2, frame_count=3)
    events = [
        SessionEvent(ts=1.0, type="message", payload={"role": "user", "text": "hello"})
    ]

    overlay.viewer.show_session(row, events)

    assert "3 frames" in overlay.viewer.summary.text()
    assert "hello" in overlay.viewer.body.toHtml()
    assert overlay.viewer.title.text() == "Elden Ring — Aug 2"


# ------------------------------------------------------------ the lifecycle


def test_launching_opens_no_session(app):
    assert app.recorder.active is False
    assert app.recorder.sessions() == []


def test_watching_creates_a_session(app):
    app.set_watching(True)

    assert app.recorder.active is True
    assert app.observer.session_id == app.recorder.session_id
    assert app.responder.session_id == app.recorder.session_id


def test_asking_a_question_creates_a_session(app):
    app.overlay.promptSubmitted.emit("where do I go?")

    assert app.recorder.active is True
    app.recorder.flush()
    events = app.recorder.read_session(app.recorder.session_id)
    assert any(
        e.type == "message" and e.payload["text"] == "where do I go?" for e in events
    )


async def test_a_watch_span_is_recorded_at_both_ends(app):
    """Async because stopping schedules the session's own shutdown."""
    app.set_watching(True)
    app.set_watching(False)
    app.recorder.flush()

    types = [e.type for e in app.recorder.read_session(app.recorder.session_id)]
    assert "watch_started" in types and "watch_stopped" in types


def test_journal_entries_reach_the_record(app):
    app.set_watching(True)
    app.journal_service.record("Lit the bonfire.", timestamp=time.time())
    app.recorder.flush()

    events = app.recorder.read_session(app.recorder.session_id)
    assert any(e.type == "journal_entry" for e in events)


def test_an_answer_is_recorded(app):
    app.set_watching(True)
    app._on_answer("Go north.")
    app.recorder.flush()

    events = app.recorder.read_session(app.recorder.session_id)
    assert any(
        e.payload.get("role") == "assistant" for e in events if e.type == "message"
    )


def test_new_session_is_the_one_canonical_reset(app):
    app.set_watching(True)
    app.journal.append("Lit the bonfire.", category="progress")
    first = app.recorder.session_id

    app.new_session()

    assert len(app.journal) == 0
    assert app.observer.memory_resets == 1
    assert app.responder.memory_resets == 1
    assert app.recorder.session_id != first
    assert app.recorder.index.get(first) is not None, "the record is kept, not deleted"


def test_new_session_while_idle_opens_nothing(app):
    """Launch state, reachable again — no session, nothing recorded."""
    app.new_session()

    assert app.recorder.active is False


def test_new_session_does_not_stop_watching(app):
    """A decision about memory, not about whether Chiron may see the screen."""
    app.set_watching(True)

    app.new_session()

    assert app.watching is True
    assert app.recorder.active is True


def test_costs_reach_the_footer(app):
    app.set_watching(True)

    app.recorder.record_llm_call(
        LLMCallRecord(model_id="m", cost_usd=0.25, pricing_source="estimated")
    )

    assert "~$0.25" in app.overlay.footer.text()


def test_a_compaction_is_recorded_and_announced(app):
    app.set_watching(True)

    app._on_compacted(
        {
            "mode": "responder",
            "agent_id": "responder",
            "tokens_before": 110000,
            "tokens_after": 20000,
            "summary": "s",
        }
    )
    app.recorder.flush()

    events = app.recorder.read_session(app.recorder.session_id)
    assert any(e.type == "compaction" for e in events)
    assert "summarised" in app.overlay.transcript.toHtml().lower()


async def test_settings_changes_are_recorded_by_name(app):
    """Async because a changed instruction schedules a session restart."""
    app.set_watching(True)
    changed = Settings(game_name="Elden Ring")

    app.apply_settings(changed)
    app.recorder.flush()

    events = app.recorder.read_session(app.recorder.session_id)
    event = next(e for e in events if e.type == "settings_changed")
    assert "game_name" in event.payload["fields"]


def test_only_the_fields_that_moved_are_recorded():
    from chiron.app import _changed_fields

    before = Settings()
    after = Settings(game_name="Hades")
    after.capture.jpeg_quality = 80

    assert set(_changed_fields(before, after)) == {"game_name", "capture.jpeg_quality"}


def test_opening_a_session_renders_it(app):
    app.set_watching(True)
    app._on_prompt("where do I go?")
    session_id = app.recorder.session_id

    app.open_session(session_id)

    assert app.overlay.current_view == SESSION_VIEW
    assert "where do I go?" in app.overlay.viewer.body.toHtml()


def test_opening_a_session_that_is_gone_says_so(app):
    app.open_session("2026-01-01-nothing-0000")

    assert "no longer on disk" in app.overlay.transcript.toHtml()


def test_renaming_a_session_sticks(app):
    app.set_watching(True)
    session_id = app.recorder.session_id

    app.overlay.sessionRenamed.emit(session_id, "The night I beat Margit")

    assert app.recorder.title == "The night I beat Margit"


def test_deleting_the_session_being_viewed_leaves_the_viewer(app):
    app.set_watching(True)
    session_id = app.recorder.session_id
    app.open_session(session_id)

    app.overlay.sessionDeleted.emit(session_id)

    assert app.recorder.index.get(session_id) is None
    assert app.overlay.current_view == PLAY_VIEW
    assert app.overlay.viewer.session_id == ""


def test_deleting_a_session_you_are_not_reading_leaves_the_view_alone(app):
    app.set_watching(True)
    open_id = app.recorder.session_id
    app.new_session()
    app.open_session(open_id)
    other = app.recorder.session_id

    app.overlay.sessionDeleted.emit(other)

    assert app.overlay.current_view == SESSION_VIEW
    assert app.overlay.viewer.session_id == open_id


def test_deleting_thumbnails_keeps_the_session(app):
    app.set_watching(True)
    app.recorder.record_frame(_frame(), "question")
    session_id = app.recorder.session_id

    app.overlay.sessionThumbnailsDeleted.emit(session_id)

    row = app.recorder.index.get(session_id)
    assert row is not None
    assert row.has_thumbnails is False


def test_the_history_button_refreshes_from_the_index(app):
    app.set_watching(True)

    app.show_history()

    assert app.overlay.picker.isVisible() is True
    assert app.overlay.picker.list.count() == 1


def test_both_agents_share_the_evening_id(app):
    app.set_watching(True)
    session_id = app.recorder.session_id

    app._on_session_changed(session_id, "title")

    assert app.observer.session_id == session_id
    assert app.responder.session_id == session_id


# ----------------------------------------------------------- fresh install


def test_fresh_install_says_play_history_is_untouched(qapp, tmp_path):
    from chiron.app import describe_untouched_sessions
    from chiron.sessions.recorder import SessionRecorder

    root = tmp_path / "sessions"
    recorder = SessionRecorder(root)
    recorder.ensure_session(game="Hades", mode="nonlive")
    recorder.close_session()

    lines = describe_untouched_sessions(root)

    assert len(lines) == 1
    assert lines[0].startswith("keep")
    assert "1 recorded session" in lines[0]


def test_fresh_install_says_nothing_when_there_is_no_history(tmp_path):
    from chiron.app import describe_untouched_sessions

    assert describe_untouched_sessions(tmp_path / "nothing") == []
