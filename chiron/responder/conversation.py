"""Canonical user/final-answer memory shared by both Responder modes."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from chiron.capture.frames import Frame
from chiron.nonlive.compaction import COMPACTION_SUMMARY_PREFIX, KEEP_RECENT_MESSAGES


@dataclass
class ResponderConversation:
    """Lossless canonical transcript plus a replaceable active-context summary."""

    messages: list[dict[str, Any]] = field(default_factory=list)
    summary: str = ""
    summary_until: int = 0

    def commit(self, question: str, answer: str, frame: Frame | None = None) -> None:
        """Commit only the user's question and final Responder answer."""
        content = question.strip()
        if frame is not None:
            content = f"{content}\n[frame {frame.clock}]"
        self.messages.append({"role": "user", "content": content})
        self.messages.append({"role": "assistant", "content": answer.strip()})

    def active_messages(self) -> list[dict[str, Any]]:
        """Summary plus unsummarised canonical tail for the next request."""
        active: list[dict[str, Any]] = []
        if self.summary:
            active.append(
                {
                    "role": "system",
                    "content": COMPACTION_SUMMARY_PREFIX + self.summary,
                }
            )
        active.extend(dict(message) for message in self.messages[self.summary_until :])
        return active

    def compaction_parts(
        self, keep: int = KEEP_RECENT_MESSAGES
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], int]:
        """New head to summarise, retained tail, and the next summary cursor."""
        keep = max(0, int(keep))
        cut = max(self.summary_until, len(self.messages) - keep)
        # Preserve user/final-answer pairs at the boundary.
        if cut % 2:
            cut -= 1
        head = [dict(message) for message in self.messages[self.summary_until : cut]]
        tail = [dict(message) for message in self.messages[cut:]]
        return head, tail, cut

    def apply_summary(self, summary: str, until: int) -> None:
        """Advance active context without deleting canonical messages."""
        self.summary = summary.strip()
        self.summary_until = max(self.summary_until, min(until, len(self.messages)))

    def clear(self) -> None:
        self.messages.clear()
        self.summary = ""
        self.summary_until = 0

    def __len__(self) -> int:
        return len(self.messages)


__all__ = ["ResponderConversation"]
