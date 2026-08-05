"""Read-only tools exposed to Chiron-Responder ReAct runs."""

from __future__ import annotations

from datetime import datetime

from chiron.core.tool import Tool
from chiron.journal.service import JournalReader
from chiron.responder.prompts import observer_context
from chiron.session import ObserverStatus


class ReadJournalTool(Tool):
    """Return compacted journal memory plus Observer freshness metadata."""

    tool_name: str = "read_journal"
    description: str = (
        "Read the current gameplay memory (older journal summary plus recent "
        "verbatim entries) and Observer freshness state. This is required once "
        "before answering."
    )
    parameters_schema: dict = {
        "type": "object",
        "properties": {},
        "required": [],
        "additionalProperties": False,
    }

    journal: JournalReader
    observer_status: ObserverStatus
    succeeded: bool = False

    model_config = {"arbitrary_types_allowed": True}

    async def forward(self) -> dict:
        snapshot = self.journal.snapshot()
        self.succeeded = True
        observed = self.observer_status.last_observed_at
        return {
            "journal": snapshot.render(),
            "entry_count": len(snapshot),
            "summarized_entry_count": snapshot.summarized_entries,
            "recent_entry_count": len(snapshot.entries),
            "observer": observer_context(self.observer_status),
            "last_observed_at": (
                datetime.fromtimestamp(observed).isoformat() if observed else None
            ),
            "stale": self.observer_status.stale,
        }


__all__ = ["ReadJournalTool"]
