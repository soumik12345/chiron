"""Automatic journal compaction keeps raw evidence and bounds model context."""

from __future__ import annotations

from chiron.config.settings import Settings
from chiron.journal.compaction import JournalCompactor
from chiron.journal.log import JournalLog
from chiron.journal.service import JournalService
from chiron.nonlive.compaction import ContextWindowInfo


async def test_small_journal_needs_no_summarization(qapp, monkeypatch):
    service = JournalService(JournalLog())
    service.record("Reached the tower.", "location")
    compactor = JournalCompactor(Settings(api_key="test-key"), service)

    async def must_not_run(*args, **kwargs):
        raise AssertionError("small journal should stay verbatim")

    compactor._complete_summary = must_not_run
    snapshot = await compactor.prepare("gemini/gemini-3.6-flash")

    assert snapshot.summary == ""
    assert [entry.note for entry in snapshot.entries] == ["Reached the tower."]


async def test_compaction_is_token_driven_and_raw_entries_remain_lossless(
    qapp, monkeypatch
):
    service = JournalService(JournalLog())
    for index in range(40):
        service.record(f"event {index}: " + "durable detail " * 40, "progress")
    raw = service.log.snapshot()
    compactor = JournalCompactor(Settings(api_key="test-key"), service)
    monkeypatch.setattr(
        "chiron.journal.compaction.ctx.resolve_context_window",
        lambda model_id: ContextWindowInfo(20_000, "test"),
    )
    summaries = []

    async def summarize(previous, entries, target_tokens):
        summaries.append((previous, list(entries), target_tokens))
        return f"summary through {entries[-1].note.split(':', 1)[0]}"

    compactor._complete_summary = summarize
    events = []
    compactor.compacted.connect(events.append)

    snapshot = await compactor.prepare(
        "gemini/gemini-3.6-flash", force=True, reason="test_overflow"
    )

    assert summaries
    assert snapshot.summary.startswith("summary through event")
    assert snapshot.summarized_entries > 0
    assert len(snapshot.entries) >= 8
    assert service.log.snapshot() == raw
    assert len(snapshot) == 40
    assert events[0]["reason"] == "test_overflow"
    assert events[0]["dropped_count"] == snapshot.summarized_entries


async def test_new_entries_remain_verbatim_after_an_existing_summary(qapp):
    service = JournalService(JournalLog())
    service.record("old one")
    service.record("old two")
    service.apply_summary("Earlier progress", 2)
    service.record("new objective", "objective")

    snapshot = service.snapshot()

    assert snapshot.summary == "Earlier progress"
    assert [entry.note for entry in snapshot.entries] == ["new objective"]
    assert len(service.log.snapshot()) == 3
