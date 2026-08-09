"""The one journal write path and immutable Responder snapshot."""

from __future__ import annotations

from chiron.journal.log import JournalLog
from chiron.journal.service import JournalService


def test_append_and_render():
    log = JournalLog()
    log.append("Entered Firelink Shrine.", category="location", timestamp=1_700_000_000)
    assert "[location]" in log.render()
    assert len(log) == 1


def test_blank_notes_are_dropped():
    log = JournalLog()
    assert log.append("   ") is None
    assert len(log) == 0


def test_raw_journal_is_lossless_in_active_memory():
    log = JournalLog()
    for index in range(600):
        log.append(f"event {index}")
    assert len(log.snapshot()) == 600
    assert log.snapshot()[0].note == "event 0"


def test_clear_removes_all_entries():
    log = JournalLog()
    log.append("a")
    log.clear()
    assert len(log) == 0


def test_service_records_and_notifies_once():
    seen = []
    service = JournalService(JournalLog(), seen.append)
    entry = service.record("Died to the skeleton.", "death")
    assert entry is not None
    assert entry.source == "observer"
    assert [item.note for item in seen] == ["Died to the skeleton."]
    assert len(service.log) == 1


async def test_observer_tool_is_the_only_write_handler():
    service = JournalService(JournalLog())
    declaration = service.function_declarations
    assert [item["name"] for item in declaration] == ["record_event"]

    result = await service.handle_observer_tool(
        "record_event", {"note": "Found the lift key.", "category": "item"}
    )
    rejected = await service.handle_observer_tool("launch_missiles", {})
    assert result["status"] == "recorded"
    assert "error" in rejected
    assert len(service.log) == 1


def test_observer_tool_requests_a_vivid_grounded_scene_paragraph():
    declaration = JournalService(JournalLog()).function_declarations[0]
    note = declaration["parameters"]["properties"]["note"]

    assert "visual memory for the Responder" in declaration["description"]
    assert "vivid, self-contained scene paragraph" in note["description"]
    assert "qualify uncertain details" in note["description"]
    assert set(declaration["parameters"]["properties"]) == {"note", "category"}
    assert declaration["parameters"]["required"] == ["note"]


async def test_empty_observer_tool_note_is_ignored():
    service = JournalService(JournalLog())
    result = await service.handle_observer_tool("record_event", {"note": " "})
    assert result["status"] == "ignored"
    assert len(service.log) == 0


def test_snapshot_is_immutable_and_does_not_follow_later_writes():
    service = JournalService(JournalLog())
    service.record("first")
    snapshot = service.snapshot()
    service.record("second")
    assert len(snapshot) == 1
    assert "first" in snapshot.render()
    assert "second" not in snapshot.render()


def test_summary_changes_model_view_without_deleting_raw_entries():
    service = JournalService(JournalLog())
    for note in ("first", "second", "third"):
        service.record(note)

    service.apply_summary("The first two events happened.", 2)
    snapshot = service.snapshot()

    assert len(snapshot) == 3
    assert snapshot.summarized_entries == 2
    assert [entry.note for entry in snapshot.entries] == ["third"]
    assert "The first two events happened." in snapshot.render()
    assert [entry.note for entry in service.log.snapshot()] == [
        "first",
        "second",
        "third",
    ]


def test_clear_resets_raw_and_summarized_memory():
    service = JournalService(JournalLog())
    service.record("first")
    service.apply_summary("Earlier memory", 1)
    service.clear()
    assert service.snapshot().render() == ""
    assert len(service.log) == 0


def test_summary_from_a_cancelled_old_session_cannot_contaminate_new_memory():
    service = JournalService(JournalLog())
    service.record("old session")
    generation = service.snapshot().generation
    service.clear()

    applied = service.apply_summary("stale result", 1, generation=generation)

    assert applied is False
    assert service.snapshot().render() == ""


def test_responder_reader_exposes_no_write_operation():
    service = JournalService(JournalLog())
    reader = service.reader()
    assert hasattr(reader, "snapshot")
    assert not hasattr(reader, "record")
    assert not hasattr(reader, "handle_observer_tool")


def test_application_entries_take_the_same_notification_path():
    seen = []
    service = JournalService(JournalLog(), seen.append)
    service.record("The player switched to Hades.", source="system")
    assert seen[0].source == "system"
