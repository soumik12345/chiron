"""Gameplay sessions: the event stream, the index, and the recorder above both.

Nothing here touches the real data directory — every test is handed a `tmp_path`
root — because a test suite that wrote into someone's play history would be a
worse bug than anything it could catch.
"""

from __future__ import annotations

import json
import time

import pytest
from PIL import Image

from chiron.capture.frames import encode_frame
from chiron.journal.log import JournalEntry
from chiron.models.usage import LLMCallRecord
from chiron.sessions import events as ev
from chiron.sessions.recorder import SessionRecorder
from chiron.sessions.render import (
    format_bytes,
    format_duration,
    relative_date,
    render_events,
    summary_line,
)
from chiron.sessions.store import (
    CostRollup,
    SessionIndex,
    SessionStore,
    default_title,
    new_session_id,
    slugify,
    summarise,
)


def _frame(colour=(30, 60, 90), *, when: float | None = None):
    """A real encoded frame, so thumbnailing is exercised rather than mocked."""
    return encode_frame(
        Image.new("RGB", (320, 180), colour), width=320, captured_at=when or time.time()
    )


@pytest.fixture
def recorder(qapp, tmp_path):
    """A recorder writing into a throwaway sessions root."""
    return SessionRecorder(tmp_path / "sessions")


# ------------------------------------------------------------------ ids


def test_session_ids_sort_by_date_and_name_the_game():
    session_id = new_session_id(
        game="Elden Ring", when=time.mktime((2026, 8, 2, 21, 0, 0, 0, 0, -1))
    )
    assert session_id.startswith("2026-08-02-elden-ring-")
    assert len(session_id.rsplit("-", 1)[1]) == 4


def test_two_sessions_of_one_game_on_one_day_do_not_collide():
    when = time.time()
    first = new_session_id(game="Hades", when=when)
    second = new_session_id(game="Hades", when=when)
    assert first != second, "the normal case, not the exception"


def test_a_session_without_a_game_is_still_titled():
    assert "Session" in default_title(game="", when=time.time())
    assert "Hades" in default_title(game="Hades", when=time.time())


def test_slugify_survives_names_a_filesystem_would_not():
    assert slugify("Baldur's Gate 3: Patch 7!") == "baldur-s-gate-3-patch-7"
    assert slugify("???") == ""


# ---------------------------------------------------------------- store


def test_events_round_trip_through_the_file(tmp_path):
    store = SessionStore.create(tmp_path, "s1")
    store.append(
        [
            ev.SessionEvent(ts=1.0, type="session_meta", payload={"id": "s1"}),
            ev.SessionEvent(ts=2.0, type="message", payload={"role": "user"}),
        ]
    )
    store.append([ev.SessionEvent(ts=3.0, type="watch_started", payload={})])

    read = store.read_events()
    assert [e.type for e in read] == ["session_meta", "message", "watch_started"]
    assert [e.ts for e in read] == [1.0, 2.0, 3.0]


def test_a_torn_final_line_costs_one_event_not_the_session(tmp_path):
    """The crash an append-only format exists to survive."""
    store = SessionStore.create(tmp_path, "s1")
    store.append([ev.SessionEvent(ts=1.0, type="message", payload={"role": "user"})])
    with store.events_path.open("a", encoding="utf-8") as handle:
        handle.write('{"ts": 2.0, "type": "mess')

    read = store.read_events()
    assert [e.type for e in read] == ["message"]


def test_a_missing_event_file_reads_as_no_events(tmp_path):
    assert SessionStore.create(tmp_path, "s1").read_events() == []


def test_thumbnails_are_downscaled_and_referenced_by_id(tmp_path):
    store = SessionStore.create(tmp_path, "s1")
    path = store.write_thumbnail("000001", _frame().jpeg)

    assert path is not None and path.is_file()
    assert Image.open(path).width <= 256
    assert store.thumbnail_count() == 1


def test_an_undecodable_frame_costs_a_thumbnail_not_a_crash(tmp_path):
    store = SessionStore.create(tmp_path, "s1")
    assert store.write_thumbnail("000001", b"not a jpeg") is None


def test_deleting_thumbnails_keeps_the_text(tmp_path):
    store = SessionStore.create(tmp_path, "s1")
    store.append([ev.SessionEvent(ts=1.0, type="message", payload={"role": "user"})])
    store.write_thumbnail("000001", _frame().jpeg)

    freed = store.delete_thumbnails()

    assert freed > 0
    assert store.thumbnail_count() == 0
    assert len(store.read_events()) == 1, "the text is the point of the half-measure"


# ---------------------------------------------------------------- index


def test_the_index_is_a_cache_that_can_be_rebuilt(recorder, tmp_path):
    session_id = recorder.ensure_session(game="Hades", mode="nonlive")
    recorder.record_message("user", "how do I beat Meg?")
    recorder.close_session()

    (tmp_path / "sessions" / "index.json").unlink()
    rebuilt = SessionIndex(tmp_path / "sessions")
    rebuilt.rebuild()

    row = rebuilt.get(session_id)
    assert row is not None
    assert row.message_count == 1
    assert row.game == "Hades"


def test_a_corrupt_index_rebuilds_rather_than_losing_the_history(recorder, tmp_path):
    session_id = recorder.ensure_session(game="Hades", mode="nonlive")
    recorder.record_message("user", "hello")
    recorder.close_session()
    (tmp_path / "sessions" / "index.json").write_text("{ not json", encoding="utf-8")

    assert SessionIndex(tmp_path / "sessions").get(session_id) is not None


def test_the_index_is_written_atomically(recorder, tmp_path):
    recorder.ensure_session(game="Hades", mode="nonlive")
    recorder.close_session()

    payload = json.loads((tmp_path / "sessions" / "index.json").read_text())

    assert payload["version"] == 1
    assert len(payload["sessions"]) == 1
    leftovers = list((tmp_path / "sessions").glob(".index.json.*"))
    assert leftovers == [], "the temporary file is renamed, not left behind"


def test_rows_come_back_newest_first(recorder):
    first = recorder.ensure_session(game="A", mode="nonlive")
    recorder.close_session()
    time.sleep(0.01)
    second = recorder.ensure_session(game="B", mode="nonlive")
    recorder.close_session()

    assert [row.id for row in recorder.sessions()] == [second, first]


# ------------------------------------------------------------- recorder


def test_launching_records_nothing(recorder):
    """An overlay left running all day while nobody plays leaves no trace."""
    assert recorder.active is False
    recorder.record_message("user", "this has nowhere to go")
    recorder.record_watch(True)

    assert recorder.sessions() == []


def test_the_first_watch_creates_a_session(recorder):
    session_id = recorder.ensure_session(game="Elden Ring", mode="live")

    assert recorder.active is True
    assert "elden-ring" in session_id
    assert recorder.title.startswith("Elden Ring")


def test_a_second_ensure_returns_the_same_session(recorder):
    first = recorder.ensure_session(game="Elden Ring", mode="live")
    second = recorder.ensure_session(game="Elden Ring", mode="live")
    assert first == second


def test_session_meta_is_always_the_first_line(recorder, tmp_path):
    session_id = recorder.ensure_session(game="Hades", mode="nonlive")
    recorder.record_message("user", "hi")
    recorder.close_session()

    events = SessionStore(tmp_path / "sessions" / session_id).read_events()
    assert events[0].type == "session_meta"
    assert events[0].payload["mode"] == "nonlive"


def test_watch_spans_accumulate_across_one_session(recorder):
    recorder.ensure_session(mode="live")
    recorder.record_watch(True)
    time.sleep(0.02)
    recorder.record_watch(False)
    recorder.record_watch(True)
    time.sleep(0.02)
    recorder.record_watch(False)

    assert recorder.sessions()[0].watched_seconds >= 0.03


def test_repeating_a_watch_state_records_nothing(recorder):
    recorder.ensure_session(mode="live")
    recorder.record_watch(True)
    recorder.record_watch(True)
    recorder.record_watch(False)
    recorder.flush()

    events = recorder.read_session(recorder.session_id)
    assert [e.type for e in events].count("watch_started") == 1


def test_closing_while_watching_closes_the_span(recorder):
    recorder.ensure_session(mode="live")
    recorder.record_watch(True)
    time.sleep(0.02)
    recorder.close_session()

    assert recorder.sessions()[0].watched_seconds > 0


def test_a_frame_thumbnail_is_reused_but_each_agent_delivery_is_attributed(recorder):
    recorder.ensure_session(mode="dual_agent")
    frame = _frame()

    first = recorder.record_frame(frame, "question", agent_id="responder")
    second = recorder.record_frame(frame, "scheduled", agent_id="observer")

    assert first == second == "000001"
    assert recorder.sessions()[0].frame_count == 2
    assert (
        len(list(recorder.store_for(recorder.session_id).frames_directory.iterdir()))
        == 1
    )
    recorder.flush()
    events = [
        event
        for event in recorder.read_session(recorder.session_id)
        if event.type == "frame"
    ]
    assert [event.payload["agent_id"] for event in events] == [
        "responder",
        "observer",
    ]


def test_the_thumbnail_rate_cap_is_the_escape_hatch(qapp, tmp_path):
    """Off by default; on, it drops closely spaced extra thumbnails."""
    recorder = SessionRecorder(tmp_path / "sessions", thumbnail_min_interval=60.0)
    recorder.ensure_session(mode="dual_agent")

    kept = [recorder.record_frame(_frame(), "scheduled") for _ in range(5)]

    assert kept[0] == "000001"
    assert kept[1:] == ["", "", "", ""]


def test_an_old_observer_run_still_rescans_and_renders(tmp_path):
    store = SessionStore.create(tmp_path, "old-v2")
    store.append(
        [
            ev.SessionEvent(
                ts=1.0,
                type="observer_run",
                payload={"reason": "heartbeat", "frames": [], "entries": 1},
            )
        ]
    )
    assert summarise(store).observer_runs == 1
    assert "looked (heartbeat)" in render_events(store.read_events())


def test_costs_roll_up_by_model_and_by_kind(recorder):
    recorder.ensure_session(mode="nonlive")
    recorder.record_llm_call(
        LLMCallRecord(
            model_id="gemini/flash", kind="nonlive_qa", cost_usd=0.01, total_tokens=100
        )
    )
    recorder.record_llm_call(
        LLMCallRecord(
            model_id="gemini/flash", kind="compaction", cost_usd=0.02, total_tokens=900
        )
    )

    rollup = recorder.cost
    assert rollup.total_usd == pytest.approx(0.03)
    assert rollup.total_tokens == 1000
    assert rollup.by_kind == {
        "nonlive_qa": pytest.approx(0.01),
        "compaction": pytest.approx(0.02),
    }
    assert rollup.by_model == {"gemini/flash": pytest.approx(0.03)}


def test_an_estimated_call_marks_the_whole_total(recorder):
    recorder.ensure_session(mode="live")
    recorder.record_llm_call(
        LLMCallRecord(
            model_id="live/x",
            kind="live_turn",
            cost_usd=0.5,
            pricing_source="estimated",
        )
    )

    assert recorder.cost.is_estimated is True
    assert recorder.cost.render() == "~$0.50"


def test_a_measured_total_carries_no_tilde():
    rollup = CostRollup()
    rollup.add(
        {"cost_usd": 1.25, "pricing_source": "actual", "model_id": "m", "kind": "turn"}
    )
    assert rollup.render() == "$1.25"


def test_journal_entries_are_recorded_verbatim(recorder):
    recorder.ensure_session(mode="nonlive")
    entry = JournalEntry(
        timestamp=1000.0,
        note="Lit the bonfire.",
        category="progress",
        source="observer",
    )

    recorder.record_journal_entry(entry)
    recorder.flush()

    event = next(
        e
        for e in recorder.read_session(recorder.session_id)
        if e.type == "journal_entry"
    )
    assert event.payload == {
        "timestamp": 1000.0,
        "note": "Lit the bonfire.",
        "category": "progress",
        "source": "observer",
    }
    assert event.ts == 1000.0, "the entry's own time, not when it was written down"


def test_settings_changes_record_names_and_never_values(recorder):
    recorder.ensure_session(mode="nonlive")
    recorder.record_settings_changed(["api_key", "selected_model"], "live")
    recorder.flush()

    event = next(
        e
        for e in recorder.read_session(recorder.session_id)
        if e.type == "settings_changed"
    )
    assert event.payload["fields"] == ["api_key", "selected_model"]
    assert "AIza" not in json.dumps(event.payload)


def test_renaming_keeps_the_id_and_so_the_directory(recorder):
    session_id = recorder.ensure_session(game="Hades", mode="nonlive")
    recorder.rename("Meg finally died")

    assert recorder.session_id == session_id
    assert recorder.sessions()[0].title == "Meg finally died"


def test_deleting_the_open_session_closes_it_first(recorder):
    session_id = recorder.ensure_session(mode="nonlive")
    recorder.record_message("user", "hi")

    recorder.delete_session(session_id)

    assert recorder.active is False
    assert recorder.sessions() == []


def test_every_recorded_type_is_in_the_vocabulary(recorder):
    recorder.ensure_session(mode="nonlive")
    recorder.record_watch(True)
    recorder.record_game(
        label="Hades", identity="steam:1", described="Hades (Steam)", changed=False
    )
    recorder.record_message("user", "hi")
    recorder.record_frame(_frame(), "question")
    recorder.record_journal_entry(JournalEntry(timestamp=time.time(), note="n"))
    recorder.record_agent_trace({"agent_id": "responder", "event": {"type": "turn"}})
    recorder.record_llm_call(LLMCallRecord(model_id="m"))
    recorder.record_compaction({"mode": "nonlive"})
    recorder.record_status("live", "gemini/flash")
    recorder.record_settings_changed(["game_name"], "nonlive")
    recorder.record_watch(False)
    recorder.flush()

    types = {e.type for e in recorder.read_session(recorder.session_id)}
    assert types <= set(ev.EVENT_TYPES)
    assert len(types) == 12, "every type above, plus the opening session_meta"


def test_two_different_frames_are_two_thumbnails(recorder):
    """A value key, not an address: a freed frame's address gets reused."""
    recorder.ensure_session(mode="live")

    ids = [recorder.record_frame(_frame(), "scheduled") for _ in range(4)]

    assert ids == ["000001", "000002", "000003", "000004"]


def test_the_recorders_counters_match_a_rescan(recorder, tmp_path):
    """The index is a cache of the stream; the two must not be able to disagree."""
    session_id = recorder.ensure_session(game="Hades", mode="nonlive")
    recorder.record_watch(True)
    recorder.record_message("user", "hi")
    recorder.record_message("assistant", "hello")
    recorder.record_frame(_frame(), "question")
    recorder.record_journal_entry(JournalEntry(timestamp=time.time(), note="n"))
    recorder.record_agent_trace({"agent_id": "responder", "event": {"type": "turn"}})
    recorder.record_compaction({"mode": "nonlive"})
    recorder.record_llm_call(
        LLMCallRecord(model_id="m", cost_usd=0.02, total_tokens=10)
    )
    recorder.record_watch(False)
    live = recorder.sessions()[0]
    recorder.close_session()

    rescanned = summarise(SessionStore(tmp_path / "sessions" / session_id))

    for field in (
        "message_count",
        "frame_count",
        "journal_count",
        "observer_runs",
        "compactions",
    ):
        assert getattr(rescanned, field) == getattr(live, field), field
    assert rescanned.rollup.total_usd == pytest.approx(live.rollup.total_usd)


# ----------------------------------------------------------------- render


def test_the_viewer_interleaves_by_timestamp_not_by_arrival():
    events = [
        ev.SessionEvent(
            ts=30.0, type="message", payload={"role": "assistant", "text": "second"}
        ),
        ev.SessionEvent(
            ts=10.0, type="message", payload={"role": "user", "text": "first"}
        ),
    ]
    html = render_events(events)
    assert html.index("first") < html.index("second")


def test_a_removed_thumbnail_renders_as_a_placeholder_not_a_broken_image(tmp_path):
    events = [
        ev.SessionEvent(
            ts=1.0,
            type="frame",
            payload={"id": "000001", "reason": "question", "thumbnail": "000001.jpg"},
        )
    ]
    html = render_events(events, frames_directory=tmp_path / "gone")

    assert "<img" not in html
    assert "thumbnail removed" in html


def test_a_present_thumbnail_renders_inline(tmp_path):
    store = SessionStore.create(tmp_path, "s1")
    store.write_thumbnail("000001", _frame().jpeg)
    events = [
        ev.SessionEvent(
            ts=1.0,
            type="frame",
            payload={"id": "000001", "reason": "question", "thumbnail": "000001.jpg"},
        )
    ]

    html = render_events(events, frames_directory=store.frames_directory)
    assert "<img" in html and "000001.jpg" in html


def test_status_noise_is_filtered_but_errors_survive():
    events = [
        ev.SessionEvent(ts=1.0, type="status", payload={"status": "reconnecting"}),
        ev.SessionEvent(
            ts=2.0, type="status", payload={"status": "error", "detail": "bad key"}
        ),
    ]
    html = render_events(events)

    assert "reconnecting" not in html
    assert "bad key" in html


def test_an_empty_session_says_so_rather_than_rendering_nothing():
    assert "recorded nothing" in render_events([])


def test_model_text_is_escaped_not_interpreted():
    events = [
        ev.SessionEvent(
            ts=1.0, type="message", payload={"role": "assistant", "text": "<b>hi</b>"}
        )
    ]
    assert "&lt;b&gt;hi&lt;/b&gt;" in render_events(events)


def test_durations_and_sizes_read_as_english():
    assert format_duration(45) == "45s"
    assert format_duration(605) == "10m"
    assert format_duration(8100) == "2h 15m"
    assert format_bytes(900) == "900 B"
    assert format_bytes(4096) == "4 KB"
    assert format_bytes(5 * 1024 * 1024) == "5.0 MB"


def test_relative_dates_say_today_and_yesterday():
    now = time.time()
    assert relative_date(now, now=now).startswith("today")
    assert relative_date(now - 86400, now=now).startswith("yesterday")


def test_the_summary_line_carries_the_cost_and_the_counts(recorder):
    recorder.ensure_session(game="Hades", mode="nonlive")
    recorder.record_message("user", "hi")
    recorder.record_llm_call(
        LLMCallRecord(model_id="m", cost_usd=0.42, pricing_source="estimated")
    )

    line = summary_line(recorder.sessions()[0])
    assert "1 messages" in line
    assert "~$0.42" in line


def test_an_open_watch_span_counts_before_it_closes(recorder):
    """A browser opened mid-evening must not report zero seconds watched."""
    recorder.ensure_session(mode="live")
    recorder.record_watch(True)
    time.sleep(0.05)

    assert recorder.sessions()[0].watched_seconds >= 0.05


def test_an_open_span_is_not_counted_twice_when_it_closes(recorder):
    recorder.ensure_session(mode="live")
    recorder.record_watch(True)
    time.sleep(0.05)
    mid = recorder.sessions()[0].watched_seconds
    recorder.record_watch(False)

    closed = recorder.sessions()[0].watched_seconds
    assert closed == pytest.approx(mid, abs=0.05)


def test_the_open_sessions_row_is_current_not_last_saved(recorder):
    """`index.json` is rewritten every few seconds; a viewer must not show zeroes."""
    session_id = recorder.ensure_session(game="Hades", mode="nonlive")
    recorder.record_message("user", "hi")
    recorder.record_frame(_frame(), "question")

    assert recorder.index.get(session_id).message_count == 0, "the saved row is stale"
    row = recorder.row_for(session_id)
    assert row.message_count == 1
    assert row.frame_count == 1


def test_a_closed_sessions_row_comes_from_the_index(recorder):
    session_id = recorder.ensure_session(game="Hades", mode="nonlive")
    recorder.record_message("user", "hi")
    recorder.close_session()

    assert recorder.row_for(session_id).message_count == 1
    assert recorder.row_for("2026-01-01-nothing-0000") is None


def test_a_window_title_masquerading_as_a_game_is_truncated():
    """A Steam name is short; a fallback window title is a whole document path."""
    title = default_title(game="Implement sessions - chiron - Visual Studio Code")
    assert title.split(" — ")[0].endswith("…")
    assert len(title.split(" — ")[0]) <= 40
