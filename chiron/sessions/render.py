"""Turning a recorded session back into something a person reads.

Pure functions over a list of :class:`~chiron.sessions.events.SessionEvent` —
no Qt, no disk beyond the thumbnail paths they point at — so the whole of the
viewer's output can be asserted on in a test without a display.

The shape is deliberately the *same* transcript widget the play view uses:
journal entries, messages and observer runs interleaved by timestamp, with each
frame's thumbnail rendered inline at the point it was sent. In a ~400 px panel an
inline image beats a filmstrip, because the question and the thing the model was
looking at when it answered belong next to each other and nowhere else.

Noise is filtered rather than rendered small: status transitions are skipped
except for errors, since a session that reconnected eleven times is a fact about
the network and not about the evening.
"""

from __future__ import annotations

import html
import time
from pathlib import Path
from typing import Iterable

from chiron.sessions.events import SessionEvent
from chiron.sessions.store import SessionRow
from chiron.ui.theme import PALETTE

#: Width the viewer draws a thumbnail at. Narrower than the panel so a frame
#: never pushes the text into a scroll bar.
INLINE_FRAME_WIDTH = 360


def format_duration(seconds: float) -> str:
    """A watched duration as ``2h 14m`` / ``14m`` / ``40s``."""
    total = max(0, int(seconds))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}h {minutes:02d}m"
    if minutes:
        return f"{minutes}m"
    return f"{secs}s"


def format_bytes(count: int) -> str:
    """A byte count as ``1.4 GB`` / ``312 MB`` / ``18 KB``."""
    size = float(max(0, count))
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{size:.0f} {unit}" if unit in ("B", "KB") else f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} GB"  # pragma: no cover - unreachable, loop always returns


def relative_date(when: float, *, now: float | None = None) -> str:
    """A timestamp as ``today 21:14`` / ``yesterday 02:03`` / ``Jul 28``."""
    current = time.time() if now is None else now
    then = time.localtime(when)
    today = time.localtime(current)
    same_day = (then.tm_year, then.tm_yday) == (today.tm_year, today.tm_yday)
    if same_day:
        return time.strftime("today %H:%M", then)
    if current - when < 48 * 3600:
        yesterday = time.localtime(current - 86400)
        if (then.tm_year, then.tm_yday) == (yesterday.tm_year, yesterday.tm_yday):
            return time.strftime("yesterday %H:%M", then)
    if then.tm_year == today.tm_year:
        return time.strftime("%b %d, %H:%M", then)
    return time.strftime("%b %d %Y", then)


def describe_row(row: SessionRow) -> str:
    """The one-line description a history row shows under its title."""
    parts = [relative_date(row.last_active_at or row.created_at)]
    if row.game:
        parts.append(row.game)
    if row.watched_seconds:
        parts.append(format_duration(row.watched_seconds))
    if row.message_count:
        parts.append(f"{row.message_count} msg")
    parts.append(row.rollup.render())
    parts.append(format_bytes(row.disk_bytes))
    return "  ·  ".join(parts)


def summary_line(row: SessionRow) -> str:
    """The pinned line at the top of the viewer: duration, counts, cost."""
    rollup = row.rollup
    parts = [
        f"watched {format_duration(row.watched_seconds)}",
        f"{row.message_count} messages",
        f"{row.journal_count} journal entries",
        f"{row.frame_count} frames",
    ]
    if row.observer_runs:
        parts.append(f"{row.observer_runs} looks")
    if row.compactions:
        parts.append(f"{row.compactions} compactions")
    parts.append(f"{rollup.render()} over {rollup.calls} calls")
    parts.append(format_bytes(row.disk_bytes))
    return "  ·  ".join(parts)


def cost_breakdown(row: SessionRow) -> list[str]:
    """Per-model and per-kind cost lines, biggest first, for the viewer."""
    rollup = row.rollup
    tilde = "~" if rollup.is_estimated else ""
    lines: list[str] = []
    for label, table in (("model", rollup.by_model), ("kind", rollup.by_kind)):
        for name, cost in sorted(table.items(), key=lambda kv: -kv[1]):
            lines.append(f"{label}: {name} — {tilde}${cost:.4f}")
    return lines


def _escape(text: str) -> str:
    """HTML-escape, keeping line breaks."""
    return html.escape(text or "").replace("\n", "<br>")


def _block(colour: str, label: str, body: str, *, dim: bool = False) -> str:
    """One rendered event: a coloured speaker line and its content."""
    if dim:
        return (
            f'<p style="margin:3px 0;color:{colour}">'
            f"<i>{label}{(' ' + body) if body else ''}</i></p>"
        )
    return (
        f'<p style="margin:6px 0 2px 0;color:{colour}"><b>{label}</b></p>'
        f'<p style="margin:0 0 8px 0">{body}</p>'
    )


def render_events(
    events: Iterable[SessionEvent], *, frames_directory: Path | None = None
) -> str:
    """Render a recorded session as the transcript HTML the viewer shows.

    Args:
        events (Iterable[SessionEvent]): The session's events, in order.
        frames_directory (Path | None): Where the thumbnails live. When None —
            or when a thumbnail has been deleted to reclaim disk — frames render
            as a dim placeholder line rather than a broken image.

    Returns:
        str: HTML for a ``QTextBrowser``.
    """
    blocks: list[str] = []
    for event in sorted(events, key=lambda e: e.ts):
        payload = event.payload
        clock = event.clock

        if event.type == "message":
            role = str(payload.get("role") or "")
            text = _escape(str(payload.get("text") or ""))
            if role == "user":
                blocks.append(_block(PALETTE["user"], f"You · {clock}", text))
            else:
                blocks.append(_block(PALETTE["accent"], f"Chiron · {clock}", text))

        elif event.type == "journal_entry":
            note = _escape(str(payload.get("note") or ""))
            category = html.escape(str(payload.get("category") or "note"))
            blocks.append(
                _block(
                    PALETTE["text_faint"],
                    f"✎ {clock} [{category}]",
                    note,
                    dim=True,
                )
            )

        elif event.type == "frame":
            blocks.append(_render_frame(payload, clock, frames_directory))

        elif event.type == "observer_run":
            if payload.get("skipped"):
                continue
            entries = int(
                payload.get("journal_entry_count") or payload.get("entries") or 0
            )
            reason = html.escape(
                str(payload.get("seal_reason") or payload.get("reason") or "look")
            )
            noted = f"{entries} entries" if entries else "nothing new"
            blocks.append(
                _block(
                    PALETTE["text_faint"],
                    f"◎ {clock} looked ({reason})",
                    noted,
                    dim=True,
                )
            )

        elif event.type == "compaction":
            mode = html.escape(str(payload.get("mode") or ""))
            before = int(payload.get("tokens_before") or 0)
            after = int(payload.get("tokens_after") or 0)
            detail = (
                f"{before:,} → {after:,} tokens" if before else "memory consolidated"
            )
            blocks.append(
                _block(
                    PALETTE["warn"], f"✂ {clock} compacted ({mode})", detail, dim=True
                )
            )
            summary = str(payload.get("summary") or "").strip()
            if summary:
                blocks.append(
                    f'<p style="margin:0 0 8px 12px;color:{PALETTE["text_dim"]}">'
                    f"{_escape(summary)}</p>"
                )

        elif event.type in ("watch_started", "watch_stopped"):
            started = event.type == "watch_started"
            blocks.append(
                _block(
                    PALETTE["live"] if started else PALETTE["text_faint"],
                    f"{'●' if started else '○'} {clock}",
                    "started watching" if started else "stopped watching",
                    dim=True,
                )
            )

        elif event.type in ("game_detected", "game_changed"):
            label = html.escape(str(payload.get("label") or ""))
            blocks.append(
                _block(
                    PALETTE["text_faint"], f"▸ {clock}", f"playing {label}", dim=True
                )
            )

        elif event.type == "settings_changed":
            fields = ", ".join(html.escape(f) for f in payload.get("fields") or [])
            blocks.append(
                _block(
                    PALETTE["text_faint"], f"⚙ {clock}", f"changed {fields}", dim=True
                )
            )

        elif event.type == "status" and payload.get("status") == "error":
            detail = _escape(str(payload.get("detail") or "session error"))
            blocks.append(_block(PALETTE["error"], f"⚠ {clock}", detail, dim=True))

    if not blocks:
        return (
            f'<p style="color:{PALETTE["text_faint"]}">'
            "This session recorded nothing.</p>"
        )
    return "".join(blocks)


def _render_frame(payload: dict, clock: str, frames_directory: Path | None) -> str:
    """One frame: the thumbnail inline, or a placeholder when it is gone."""
    reason = html.escape(str(payload.get("reason") or "frame"))
    name = str(payload.get("thumbnail") or "")
    path = (frames_directory / name) if frames_directory and name else None
    if path is not None and path.is_file():
        return (
            f'<p style="margin:4px 0 2px 0;color:{PALETTE["text_faint"]}">'
            f"<i>▣ {clock} · {reason}</i></p>"
            f'<p style="margin:0 0 8px 0">'
            f'<img src="{html.escape(str(path))}" width="{INLINE_FRAME_WIDTH}"></p>'
        )
    return _block(
        PALETTE["text_faint"],
        f"▣ {clock}",
        f"{reason} frame (thumbnail removed)",
        dim=True,
    )


__all__ = [
    "INLINE_FRAME_WIDTH",
    "cost_breakdown",
    "describe_row",
    "format_bytes",
    "format_duration",
    "relative_date",
    "render_events",
    "summary_line",
]
