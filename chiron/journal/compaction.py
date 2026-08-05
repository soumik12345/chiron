"""Token-aware, lossless-source compaction for shared gameplay memory."""

from __future__ import annotations

import asyncio
from typing import Any

from PySide6.QtCore import QObject, Signal

from chiron.config.settings import Settings
from chiron.journal.log import JournalEntry
from chiron.journal.service import JournalService, JournalSnapshot
from chiron.models.litellm_model import LiteLLMModel
from chiron.models.usage import LLMCallRecord, call_timer, utc_now_iso
from chiron.nonlive import compaction as ctx

# Journal memory may use this much of a consumer's context before it is compacted,
# and is reduced to the lower watermark to avoid another call on the next entry.
JOURNAL_TRIGGER_RATIO = 0.20
JOURNAL_TARGET_RATIO = 0.10
JOURNAL_FORCED_TARGET_RATIO = 0.05
JOURNAL_TAIL_SHARE = 0.60
MIN_JOURNAL_TARGET_TOKENS = 1_024
MIN_RECENT_ENTRIES = 8
MAX_SUMMARY_OUTPUT_TOKENS = 1_200
SUMMARY_INPUT_RATIO = 0.60

JOURNAL_SUMMARY_SYSTEM_PROMPT = """\
You maintain compact long-term memory for a gaming assistant. Merge the earlier \
journal summary with the new journal entries. Preserve exact names, objectives, \
locations, items, NPCs, deaths, discoveries, player decisions, and unresolved \
questions. Remove repetition and facts explicitly superseded by later entries. \
Do not invent anything. Return only a compact factual briefing, without preamble.\
"""


class JournalCompactor(QObject):
    """Build one shared summary while retaining every raw entry in `JournalLog`."""

    llmCall = Signal(object)
    compacted = Signal(object)

    def __init__(
        self,
        settings: Settings,
        journal: JournalService,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self.settings = settings
        self.journal = journal
        self.session_id = ""
        self._lock = asyncio.Lock()

    def apply_settings(self, settings: Settings) -> None:
        self.settings = settings

    async def prepare(
        self,
        target_model: str,
        *,
        force: bool = False,
        reason: str = "threshold",
    ) -> JournalSnapshot:
        """Return journal context sized for `target_model`, compacting if needed."""
        window = ctx.resolve_context_window(target_model).tokens
        trigger = max(MIN_JOURNAL_TARGET_TOKENS, int(window * JOURNAL_TRIGGER_RATIO))
        ratio = JOURNAL_FORCED_TARGET_RATIO if force else JOURNAL_TARGET_RATIO
        target = max(MIN_JOURNAL_TARGET_TOKENS, int(window * ratio))
        snapshot = self.journal.snapshot()
        before = _snapshot_tokens(target_model, snapshot)
        if before < (target if force else trigger):
            return snapshot

        async with self._lock:
            snapshot = self.journal.snapshot()
            before = _snapshot_tokens(target_model, snapshot)
            if before < (target if force else trigger):
                return snapshot
            cut = _summary_cut(target_model, snapshot.entries, target)
            if cut <= 0:
                return snapshot

            previous = snapshot.summary or None
            absolute = snapshot.summarized_entries
            generation = snapshot.generation
            pending = list(snapshot.entries[:cut])
            dropped = 0
            while pending:
                chunk = self._next_chunk(previous, pending)
                summary = await self._complete_summary(previous, chunk, target)
                if not summary.strip():
                    raise RuntimeError(
                        "Journal summarization returned an empty result."
                    )
                absolute += len(chunk)
                dropped += len(chunk)
                if not self.journal.apply_summary(
                    summary, absolute, generation=generation
                ):
                    return self.journal.snapshot()
                previous = summary.strip()
                del pending[: len(chunk)]

            compacted = self.journal.snapshot()
            after = _snapshot_tokens(target_model, compacted)
            self.compacted.emit(
                {
                    "agent_id": "journal",
                    "mode": "journal",
                    "summary": compacted.summary,
                    "tokens_before": before,
                    "tokens_after": after,
                    "kept_tail_count": len(compacted.entries),
                    "dropped_count": dropped,
                    "reason": reason,
                }
            )
            return compacted

    def _next_chunk(
        self, previous: str | None, entries: list[JournalEntry]
    ) -> list[JournalEntry]:
        model_id = self.settings.responder_model
        window = ctx.resolve_context_window(model_id).tokens
        limit = max(4_096, int(window * SUMMARY_INPUT_RATIO))
        chunk: list[JournalEntry] = []
        for entry in entries:
            candidate = [*chunk, entry]
            request = _summary_request(previous, candidate)
            if chunk and ctx.count_tokens(model_id, request) > limit:
                break
            chunk = candidate
        return chunk or entries[:1]

    async def _complete_summary(
        self,
        previous: str | None,
        entries: list[JournalEntry],
        target_tokens: int,
    ) -> str:
        model_id = self.settings.responder_model
        api_key = self.settings.key_for_model(model_id)
        if not api_key:
            raise RuntimeError(
                "Journal memory needs compaction, but the selected Responder "
                "model has no API key."
            )
        max_tokens = min(
            MAX_SUMMARY_OUTPUT_TOKENS,
            max(256, target_tokens // 3),
        )
        model = LiteLLMModel(
            model_id=model_id,
            api_key=api_key,
            temperature=None if "gemini-3.6" in model_id.lower() else 0.2,
            max_tokens=max_tokens,
            usage_sink=self._on_usage,
            usage_labels={
                "session_id": self.session_id or None,
                "agent_id": "journal",
            },
        )
        messages = _summary_request(previous, entries)
        started_at = utc_now_iso()
        elapsed = call_timer()
        try:
            response = await model.acompletion(messages=messages)
        except Exception as error:
            model.record_failure(
                error,
                context={
                    "kind": "journal_compaction",
                    "started_at": started_at,
                    "duration_ms": elapsed(),
                },
            )
            raise
        choice = response.choices[0]
        model.record_usage(
            getattr(response, "usage", None),
            response=response,
            context={
                "kind": "journal_compaction",
                "started_at": started_at,
                "duration_ms": elapsed(),
                "finish_reason": getattr(choice, "finish_reason", None),
            },
        )
        content: Any = getattr(choice.message, "content", "") or ""
        if isinstance(content, list):
            return "".join(
                str(part.get("text") or "") if isinstance(part, dict) else str(part)
                for part in content
            )
        return str(content)

    def _on_usage(self, record: LLMCallRecord) -> None:
        self.llmCall.emit(record)


def _snapshot_tokens(model_id: str, snapshot: JournalSnapshot) -> int:
    text = snapshot.render()
    if not text:
        return 0
    return ctx.count_tokens(model_id, [{"role": "user", "content": text}])


def _summary_cut(
    model_id: str,
    entries: tuple[JournalEntry, ...],
    target_tokens: int,
) -> int:
    """Number of oldest unsummarised entries to replace with a summary."""
    if len(entries) <= MIN_RECENT_ENTRIES:
        return 0
    tail_budget = max(512, int(target_tokens * JOURNAL_TAIL_SHARE))
    start = len(entries) - MIN_RECENT_ENTRIES
    while start > 0:
        candidate = "\n".join(entry.render() for entry in entries[start - 1 :])
        if (
            ctx.count_tokens(model_id, [{"role": "user", "content": candidate}])
            > tail_budget
        ):
            break
        start -= 1
    return start


def _summary_request(
    previous: str | None, entries: list[JournalEntry]
) -> list[dict[str, str]]:
    parts: list[str] = []
    if previous:
        parts.append(f"<earlier-summary>\n{previous}\n</earlier-summary>")
    rendered = "\n".join(entry.render() for entry in entries)
    parts.append(f"<new-journal-entries>\n{rendered}\n</new-journal-entries>")
    parts.append("Return the updated compact journal memory.")
    return [
        {"role": "system", "content": JOURNAL_SUMMARY_SYSTEM_PROMPT},
        {"role": "user", "content": "\n\n".join(parts)},
    ]


__all__ = [
    "JOURNAL_FORCED_TARGET_RATIO",
    "JOURNAL_SUMMARY_SYSTEM_PROMPT",
    "JOURNAL_TARGET_RATIO",
    "JOURNAL_TRIGGER_RATIO",
    "JournalCompactor",
]
