"""The journal log and both writer strategies."""

from __future__ import annotations

from typing import Any

import pytest

from chiron.config.settings import JournalSettings
from chiron.journal.log import JournalLog
from chiron.journal.writers import (
    SidecarJournal,
    ToolCallJournal,
    build_journal_writer,
    parse_sidecar_entries,
)


class FakeModel:
    """Stands in for LiteLLMModel, recording what it was asked."""

    def __init__(self, reply: str) -> None:
        self.reply = reply
        self.calls: list[list[dict[str, Any]]] = []

    async def acompletion(self, messages, tools=None, stream=False):
        self.calls.append(messages)

        class _Message:
            content = self.reply

        class _Choice:
            message = _Message()

        class _Response:
            choices = [_Choice()]
            usage = None

        return _Response()

    def record_usage(self, usage, response=None, context=None):
        return {}


# ------------------------------------------------------------------- the log


def test_append_and_render():
    log = JournalLog()
    log.append("Entered Firelink Shrine.", category="location", timestamp=1_700_000_000)
    rendered = log.render()
    assert "[location]" in rendered
    assert "Entered Firelink Shrine." in rendered
    assert len(log) == 1


def test_blank_notes_are_dropped():
    log = JournalLog()
    assert log.append("   ") is None
    assert len(log) == 0


def test_trimming_keeps_the_newest():
    log = JournalLog(max_entries=3)
    for index in range(6):
        log.append(f"event {index}")
    assert len(log) == 3
    assert log.entries[0].note == "event 3"


def test_fold_tracking():
    log = JournalLog()
    log.append("first")
    assert [e.note for e in log.unfolded()] == ["first"]

    log.mark_folded()
    assert log.unfolded() == []

    log.append("second")
    assert [e.note for e in log.unfolded()] == ["second"]


def test_trimming_does_not_resurrect_folded_entries():
    log = JournalLog(max_entries=2)
    log.append("a")
    log.append("b")
    log.mark_folded()
    log.append("c")
    assert [e.note for e in log.unfolded()] == ["c"]


def test_clear_resets_fold_state():
    log = JournalLog()
    log.append("a")
    log.mark_folded()
    log.clear()
    assert len(log) == 0
    assert log.unfolded() == []


# --------------------------------------------------------------- tool-call


async def test_tool_call_journal_records_events():
    log = JournalLog()
    seen = []
    writer = ToolCallJournal(log, seen.append)

    declarations = writer.function_declarations()
    assert declarations[0]["name"] == "record_event"

    result = await writer.handle_tool_call(
        "record_event", {"note": "Died to the skeleton.", "category": "death"}
    )
    assert result["status"] == "recorded"
    assert log.entries[0].category == "death"
    assert log.entries[0].source == "tool_call"
    assert len(seen) == 1


async def test_tool_call_journal_rejects_unknown_functions():
    writer = ToolCallJournal(JournalLog())
    result = await writer.handle_tool_call("launch_missiles", {})
    assert "error" in result


async def test_tool_call_journal_ignores_empty_notes():
    log = JournalLog()
    writer = ToolCallJournal(log)
    result = await writer.handle_tool_call("record_event", {"note": ""})
    assert result["status"] == "ignored"
    assert len(log) == 0


# ----------------------------------------------------------------- sidecar


def test_sidecar_declares_no_functions():
    writer = SidecarJournal(JournalLog(), JournalSettings(strategy="sidecar"))
    assert writer.function_declarations() == []


async def test_sidecar_summarises_the_transcript():
    log = JournalLog()
    model = FakeModel(
        '{"entries": [{"category": "location", "note": "Entered the Undead Parish."}]}'
    )
    writer = SidecarJournal(log, JournalSettings(strategy="sidecar"), model=model)
    writer.observe_user_message("where am I?")
    writer.observe_model_message("The Undead Parish, by the look of the architecture.")

    entries = await writer.summarise_once()

    assert [e.note for e in entries] == ["Entered the Undead Parish."]
    assert log.entries[0].source == "sidecar"
    assert "where am I?" in str(model.calls[0][1]["content"])


async def test_sidecar_does_nothing_when_nothing_happened():
    model = FakeModel("{}")
    writer = SidecarJournal(
        JournalLog(), JournalSettings(strategy="sidecar"), model=model
    )
    assert await writer.summarise_once() == []
    assert model.calls == []


async def test_sidecar_sends_kept_frames():
    from PIL import Image

    from chiron.capture.frames import encode_frame

    model = FakeModel('{"entries": []}')
    writer = SidecarJournal(
        JournalLog(),
        JournalSettings(strategy="sidecar", sidecar_frame_count=2),
        model=model,
    )
    for _ in range(3):
        writer.observe_frame(encode_frame(Image.new("RGB", (64, 36))))

    await writer.summarise_once()

    content = model.calls[0][1]["content"]
    images = [part for part in content if part["type"] == "image_url"]
    assert len(images) == 2, "only the most recent frames are kept"
    assert images[0]["image_url"]["url"].startswith("data:image/jpeg;base64,")


@pytest.mark.parametrize(
    "reply,expected",
    [
        ('{"entries": [{"note": "a", "category": "item"}]}', [("a", "item")]),
        ('```json\n{"entries": [{"note": "b"}]}\n```', [("b", "note")]),
        ('Sure!\n{"entries": [{"note": "c"}]}\nHope that helps.', [("c", "note")]),
        ('{"entries": ["d"]}', [("d", "note")]),
        ('{"entries": []}', []),
        ("not json at all", []),
        ("", []),
        ('{"entries": [{"note": "   "}]}', []),
    ],
)
def test_sidecar_reply_parsing(reply, expected):
    assert parse_sidecar_entries(reply) == expected


# ------------------------------------------------------------------ factory


def test_factory_picks_the_configured_strategy():
    log = JournalLog()
    assert isinstance(
        build_journal_writer(JournalSettings(strategy="tool_call"), log),
        ToolCallJournal,
    )
    assert isinstance(
        build_journal_writer(JournalSettings(strategy="sidecar"), log), SidecarJournal
    )
