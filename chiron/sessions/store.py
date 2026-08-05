"""Where sessions live on disk, and the index the browser reads instead.

::

    $XDG_DATA_HOME/chiron/sessions/            # ~/.local/share/chiron/sessions/
      index.json                               # atomic rewrite; the browser reads only this
      2026-08-02-elden-ring-a3f9/
        events.jsonl                           # typed, append-only, torn-tail-tolerant
        frames/000001.jpg 000002.jpg …         # ~256 px thumbnails, referenced by id

**The data directory, not the config directory.** ``--fresh-install``'s
:func:`~chiron.config.settings.plan_removal` only ever considers the config
directory, so a season of play survives a key reset by construction rather than
by anyone remembering to special-case it. Wiping an API key should not bundle in
wiping the history of what you played.

Two durability ideas are lifted from the dormant :class:`~chiron.core.session.JsonlSessionStore`
and nothing else is: appends are a single line written to an open file, and reads
skip a line that will not parse. A process killed mid-append leaves a truncated
final line, which is precisely the crash durable storage exists to survive — so
the reader shrugs at it rather than making the whole session unopenable. What is
deliberately *not* taken is the branching tree, the OpenAI-message replay model
and the CWD-relative directory: a wall-clock recording of an evening is a list,
and it belongs in the user's data directory.

``index.json`` holds one row per session — everything the list UI needs, so
browsing never opens an event file — and is rewritten with the same
``NamedTemporaryFile`` + :func:`os.replace` dance
:func:`~chiron.config.settings.save_settings` uses. It is a cache, not a source
of truth: :meth:`SessionIndex.rebuild` reconstructs it from the session
directories themselves, so losing it costs a rescan rather than a history.
"""

from __future__ import annotations

import io
import json
import logging
import os
import re
import shutil
import tempfile
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

from pydantic import BaseModel, Field, ValidationError

from chiron.sessions.events import SessionEvent

logger = logging.getLogger(__name__)

#: Longest edge of a stored thumbnail. Big enough to recognise a boss arena at a
#: glance in a ~400 px overlay, small enough that an evening of them is tens of
#: megabytes rather than gigabytes.
THUMBNAIL_WIDTH = 256

#: JPEG quality for thumbnails. They are already a lossy re-encode of a lossy
#: frame; pushing quality higher buys artefacts, not detail.
THUMBNAIL_QUALITY = 70

_EVENTS_FILE = "events.jsonl"
_FRAMES_DIR = "frames"
_INDEX_FILE = "index.json"

_SLUG_STRIP = re.compile(r"[^a-z0-9]+")


# --------------------------------------------------------------------- paths


def data_root() -> Path:
    """Chiron's data directory, honouring ``XDG_DATA_HOME``."""
    base = os.environ.get("XDG_DATA_HOME", "").strip()
    root = Path(base) if base else Path.home() / ".local" / "share"
    return root / "chiron"


def sessions_root() -> Path:
    """Where every session directory and the index file live."""
    return data_root() / "sessions"


def slugify(text: str, *, limit: int = 32) -> str:
    """A filesystem-safe fragment of `text`, or empty when nothing survives."""
    slug = _SLUG_STRIP.sub("-", (text or "").strip().lower()).strip("-")
    return slug[:limit].strip("-")


def new_session_id(*, game: str = "", when: float | None = None) -> str:
    """A directory name like ``2026-08-02-elden-ring-a3f9``.

    Date first so the directory listing sorts chronologically, game in the
    middle so a human can find one by eye, and four random characters last
    because two sessions of the same game on the same day is the normal case,
    not the exception.
    """
    stamp = datetime.fromtimestamp(time.time() if when is None else when)
    parts = [stamp.strftime("%Y-%m-%d"), slugify(game), uuid.uuid4().hex[:4]]
    return "-".join(p for p in parts if p)


#: Longest game name an auto-title will carry. A Steam name is a handful of
#: words; a fallback window title is a whole document path, and the header the
#: title lands in is a strip of a 420 px panel.
TITLE_GAME_LIMIT = 40


def default_title(*, game: str = "", when: float | None = None) -> str:
    """The auto-title a new session gets: game and date, or just the date.

    The floor, not the ceiling — the title is renameable, and a one-line
    model-written title at close is a candidate for later.
    """
    stamp = datetime.fromtimestamp(time.time() if when is None else when)
    date = stamp.strftime("%b %-d") if os.name != "nt" else stamp.strftime("%b %d")
    name = (game or "").strip()
    if len(name) > TITLE_GAME_LIMIT:
        name = name[: TITLE_GAME_LIMIT - 1].rstrip() + "…"
    return f"{name} — {date}" if name else f"Session — {date}"


# --------------------------------------------------------------------- store


class SessionStore:
    """One session's directory: its event stream and its thumbnails.

    Attributes:
        directory (Path): The session directory.
        session_id (str): Its name, which is also the session's id.
    """

    def __init__(self, directory: str | Path) -> None:
        """Open (or create) a session directory."""
        self.directory = Path(directory)
        self.session_id = self.directory.name
        self.directory.mkdir(parents=True, exist_ok=True)

    @classmethod
    def create(cls, root: str | Path, session_id: str) -> SessionStore:
        """Create the directory for `session_id` under `root`."""
        return cls(Path(root) / session_id)

    # ------------------------------------------------------------- events

    @property
    def events_path(self) -> Path:
        """The append-only event file."""
        return self.directory / _EVENTS_FILE

    @property
    def frames_directory(self) -> Path:
        """Where thumbnails are written."""
        return self.directory / _FRAMES_DIR

    def append(self, events: Iterable[SessionEvent]) -> int:
        """Append events as JSON lines, returning how many were written.

        One open, one write, one close per flush rather than per event: the
        recorder buffers, so a burst of a dozen events costs a single syscall
        round trip and a crash loses at most the buffer.
        """
        rows = [event.model_dump_json() for event in events]
        if not rows:
            return 0
        with self.events_path.open("a", encoding="utf-8") as handle:
            handle.write("\n".join(rows) + "\n")
        return len(rows)

    def read_events(self) -> list[SessionEvent]:
        """Every event in the file, in write order, skipping unparseable lines.

        A truncated final line means the process died mid-flush, which is the
        crash this format exists to survive. It is dropped with a debug note; a
        line that will not parse from the *middle* of a file is logged louder,
        because that is genuine corruption rather than an interrupted write.
        """
        path = self.events_path
        if not path.exists():
            return []
        events: list[SessionEvent] = []
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        for number, line in enumerate(lines, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                events.append(SessionEvent.model_validate(json.loads(text)))
            except (json.JSONDecodeError, ValidationError) as error:
                if number == len(lines):
                    logger.debug("Ignoring torn final line in %s", path)
                else:
                    logger.warning(
                        "Skipping bad event at %s:%d: %s", path, number, error
                    )
        return events

    # --------------------------------------------------------- thumbnails

    def frame_path(self, frame_id: str) -> Path:
        """Where the thumbnail for `frame_id` lives (whether or not it exists)."""
        return self.frames_directory / f"{frame_id}.jpg"

    def write_thumbnail(self, frame_id: str, jpeg: bytes) -> Path | None:
        """Downscale an encoded frame to a thumbnail and store it.

        Returns None rather than raising when the image cannot be decoded or the
        disk refuses the write: a recorder is an observer of gameplay, and a full
        disk must cost a thumbnail, not a session.
        """
        try:
            from PIL import Image

            self.frames_directory.mkdir(parents=True, exist_ok=True)
            image = Image.open(io.BytesIO(jpeg))
            if image.width > THUMBNAIL_WIDTH:
                height = max(1, round(image.height * (THUMBNAIL_WIDTH / image.width)))
                image = image.resize(
                    (THUMBNAIL_WIDTH, height), Image.Resampling.LANCZOS
                )
            path = self.frame_path(frame_id)
            image.convert("RGB").save(
                path, format="JPEG", quality=THUMBNAIL_QUALITY, optimize=True
            )
            return path
        except Exception:  # noqa: BLE001 — a thumbnail is never worth a crash
            logger.debug("Could not write thumbnail %s", frame_id, exc_info=True)
            return None

    # -------------------------------------------------------------- disk

    def disk_bytes(self) -> int:
        """Total size of everything in this session's directory."""
        total = 0
        for path in self.directory.rglob("*"):
            try:
                if path.is_file():
                    total += path.stat().st_size
            except OSError:  # pragma: no cover - racing with a delete
                continue
        return total

    def thumbnail_count(self) -> int:
        """How many thumbnails are currently stored."""
        directory = self.frames_directory
        return len(list(directory.glob("*.jpg"))) if directory.is_dir() else 0

    def delete_thumbnails(self) -> int:
        """Remove the frames directory, keeping the text. Returns bytes freed.

        The half-measure that makes retention bearable: thumbnails are almost all
        of a session's size and the least of its meaning, so the browser offers
        this beside a full delete rather than making the choice all-or-nothing.
        """
        directory = self.frames_directory
        if not directory.is_dir():
            return 0
        freed = sum(p.stat().st_size for p in directory.glob("*") if p.is_file())
        shutil.rmtree(directory, ignore_errors=True)
        return freed

    def delete(self) -> None:
        """Remove the whole session directory."""
        shutil.rmtree(self.directory, ignore_errors=True)


# --------------------------------------------------------------------- index


@dataclass
class CostRollup:
    """What a session spent, split the ways anyone would want to ask about it.

    Attributes:
        total_usd (float): Everything, measured and estimated together.
        estimated_usd (float): The part of `total_usd` that is arithmetic rather
            than a provider's report — live mode, which never touches litellm.
        total_tokens (int): Prompt plus completion across every call.
        calls (int): How many calls were made.
        by_model (dict[str, float]): USD per model id.
        by_kind (dict[str, float]): USD per call kind (``nonlive_qa``,
            ``nonlive_observer``, ``compaction``, …).
    """

    total_usd: float = 0.0
    estimated_usd: float = 0.0
    total_tokens: int = 0
    calls: int = 0
    by_model: dict[str, float] = field(default_factory=dict)
    by_kind: dict[str, float] = field(default_factory=dict)

    @property
    def is_estimated(self) -> bool:
        """Whether any of this figure is an estimate, and so needs a ``~``."""
        return self.estimated_usd > 0.0

    def add(self, record: dict[str, Any]) -> None:
        """Fold one serialised :class:`~chiron.models.usage.LLMCallRecord` in."""
        cost = float(record.get("cost_usd") or 0.0)
        self.total_usd += cost
        self.total_tokens += int(record.get("total_tokens") or 0)
        self.calls += 1
        if record.get("pricing_source") == "estimated":
            self.estimated_usd += cost
        model = str(record.get("model_id") or "unknown")
        kind = str(record.get("kind") or "turn")
        self.by_model[model] = self.by_model.get(model, 0.0) + cost
        self.by_kind[kind] = self.by_kind.get(kind, 0.0) + cost

    def render(self) -> str:
        """The rollup as one short line, marked with ``~`` when estimated."""
        tilde = "~" if self.is_estimated else ""
        return f"{tilde}${self.total_usd:.2f}"


class SessionRow(BaseModel):
    """One line of ``index.json`` — everything the history list shows.

    Attributes:
        id (str): The session's directory name.
        title (str): The display name, renameable.
        game (str): The game as detected or named when the session started.
        mode (str): ``live`` or ``nonlive`` at creation.
        created_at (float): When the session was opened.
        last_active_at (float): The timestamp of its most recent event.
        watched_seconds (float): Total time spent actually watching, summed
            across watch spans — which is not the session's wall-clock length.
        message_count (int): Player and assistant turns.
        frame_count (int): Frames that reached a model.
        journal_count (int): Journal entries recorded.
        observer_runs (int): Observer calls made.
        compactions (int): Times memory was consolidated.
        disk_bytes (int): Size of the session directory.
        has_thumbnails (bool): Whether the frames directory still exists.
        cost (dict): A serialised :class:`CostRollup`.
    """

    id: str
    title: str = ""
    game: str = ""
    mode: str = "live"
    created_at: float = 0.0
    last_active_at: float = 0.0
    watched_seconds: float = 0.0
    message_count: int = 0
    frame_count: int = 0
    journal_count: int = 0
    observer_runs: int = 0
    compactions: int = 0
    disk_bytes: int = 0
    has_thumbnails: bool = True
    cost: dict[str, Any] = Field(default_factory=lambda: asdict(CostRollup()))

    @property
    def rollup(self) -> CostRollup:
        """The cost fields as a :class:`CostRollup`."""
        data = dict(self.cost)
        known = {f for f in CostRollup.__dataclass_fields__}
        return CostRollup(**{k: v for k, v in data.items() if k in known})

    def matches(self, query: str) -> bool:
        """Whether this row should survive the history view's search box."""
        text = (query or "").strip().lower()
        if not text:
            return True
        return text in f"{self.title} {self.game} {self.id}".lower()


class SessionIndex:
    """The browsable list of every recorded session.

    Attributes:
        root (Path): The sessions directory this index describes.
    """

    def __init__(self, root: str | Path | None = None) -> None:
        """Load the index for `root`, defaulting to :func:`sessions_root`."""
        self.root = Path(root) if root is not None else sessions_root()
        self._rows: dict[str, SessionRow] = {}
        self.load()

    @property
    def path(self) -> Path:
        """The index file."""
        return self.root / _INDEX_FILE

    # -------------------------------------------------------------- reading

    def load(self) -> None:
        """Read ``index.json``, falling back to a rescan when it is unusable."""
        path = self.path
        if not path.exists():
            self._rows = {}
            return
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            rows = [SessionRow.model_validate(r) for r in raw.get("sessions") or []]
            self._rows = {row.id: row for row in rows}
        except (OSError, json.JSONDecodeError, ValidationError, AttributeError) as e:
            logger.warning("Could not read %s (%s); rebuilding from disk", path, e)
            self.rebuild()

    def rows(self) -> list[SessionRow]:
        """Every row, newest activity first — the order the browser shows."""
        return sorted(
            self._rows.values(),
            key=lambda r: r.last_active_at or r.created_at,
            reverse=True,
        )

    def get(self, session_id: str) -> SessionRow | None:
        """The row for `session_id`, or None."""
        return self._rows.get(session_id)

    def total_bytes(self) -> int:
        """Disk used by every session — the line that *is* the retention policy."""
        return sum(row.disk_bytes for row in self._rows.values())

    # -------------------------------------------------------------- writing

    def put(self, row: SessionRow) -> None:
        """Insert or replace a row in memory. Call :meth:`save` to persist."""
        self._rows[row.id] = row

    def remove(self, session_id: str) -> None:
        """Drop a row from memory. Call :meth:`save` to persist."""
        self._rows.pop(session_id, None)

    def save(self) -> Path | None:
        """Rewrite ``index.json`` atomically.

        Returns None rather than raising when the write fails: an index is a
        cache of what the session directories already say, and
        :meth:`rebuild` can always reconstruct it.
        """
        payload = {
            "version": 1,
            "sessions": [row.model_dump() for row in self.rows()],
        }
        try:
            self.root.mkdir(parents=True, exist_ok=True)
            handle = tempfile.NamedTemporaryFile(
                "w",
                encoding="utf-8",
                dir=str(self.root),
                prefix=f".{_INDEX_FILE}.",
                suffix=".tmp",
                delete=False,
            )
            try:
                with handle:
                    handle.write(json.dumps(payload, indent=2) + "\n")
                os.replace(handle.name, self.path)
            except OSError:
                Path(handle.name).unlink(missing_ok=True)
                raise
        except OSError as error:
            logger.warning("Could not write the session index: %s", error)
            return None
        return self.path

    # ------------------------------------------------------------ rebuilding

    def rebuild(self) -> None:
        """Reconstruct every row by reading the session directories.

        Slow and rarely needed, which is the trade the index exists to make: the
        browser reads one small file, and the expensive path only runs when that
        file is missing or corrupt.
        """
        self._rows = {}
        if not self.root.is_dir():
            return
        for directory in sorted(self.root.iterdir()):
            if not directory.is_dir():
                continue
            store = SessionStore(directory)
            events = store.read_events()
            if not events:
                continue
            self._rows[store.session_id] = summarise(store, events)

    def delete_session(self, session_id: str) -> None:
        """Delete a session's directory and its row."""
        row = self._rows.get(session_id)
        if row is not None:
            SessionStore(self.root / session_id).delete()
        self.remove(session_id)
        self.save()

    def delete_thumbnails(self, session_id: str) -> int:
        """Delete a session's thumbnails, keeping its text. Returns bytes freed."""
        row = self._rows.get(session_id)
        if row is None:
            return 0
        store = SessionStore(self.root / session_id)
        freed = store.delete_thumbnails()
        row.has_thumbnails = False
        row.disk_bytes = store.disk_bytes()
        self.put(row)
        self.save()
        return freed

    def rename(self, session_id: str, title: str) -> SessionRow | None:
        """Retitle a session. The id — and so the directory — never changes."""
        row = self._rows.get(session_id)
        if row is None:
            return None
        row.title = title.strip() or row.title
        self.put(row)
        self.save()
        return row


def summarise(
    store: SessionStore, events: list[SessionEvent] | None = None
) -> SessionRow:
    """Fold a session's event stream into the one row the browser shows.

    This is the definition of every counter in the index: the recorder keeps the
    same tallies incrementally while a session is open, and this is what a
    rebuild — or a test — measures them against.

    Args:
        store (SessionStore): The session to summarise.
        events (list[SessionEvent] | None): Its events, if already read.

    Returns:
        SessionRow: The summary, with disk size measured from the directory.
    """
    rows = store.read_events() if events is None else events
    row = SessionRow(id=store.session_id)
    rollup = CostRollup()
    watch_started: float | None = None

    for event in rows:
        row.last_active_at = max(row.last_active_at, event.ts)
        payload = event.payload
        if event.type == "session_meta":
            row.title = str(payload.get("title") or "")
            row.game = str(payload.get("game") or "")
            row.mode = str(payload.get("mode") or "live")
            row.created_at = float(payload.get("created_at") or event.ts)
        elif event.type == "watch_started":
            watch_started = event.ts
        elif event.type == "watch_stopped":
            if watch_started is not None:
                row.watched_seconds += max(0.0, event.ts - watch_started)
                watch_started = None
        elif event.type == "message":
            row.message_count += 1
        elif event.type == "frame":
            row.frame_count += 1
        elif event.type == "journal_entry":
            row.journal_count += 1
        elif event.type == "observer_run":
            if not payload.get("skipped"):
                row.observer_runs += 1
        elif event.type == "compaction":
            row.compactions += 1
        elif event.type == "llm_call":
            rollup.add(payload)
        elif event.type in ("game_detected", "game_changed") and not row.game:
            row.game = str(payload.get("label") or "")

    if watch_started is not None:
        # A session whose last line is a watch start was never closed cleanly —
        # a crash, or a kill. Count up to the last thing that happened rather
        # than to now, which would grow every time the browser is opened.
        row.watched_seconds += max(0.0, row.last_active_at - watch_started)

    if not row.created_at:
        row.created_at = rows[0].ts if rows else time.time()
    row.cost = asdict(rollup)
    row.disk_bytes = store.disk_bytes()
    row.has_thumbnails = store.thumbnail_count() > 0
    return row


__all__ = [
    "THUMBNAIL_QUALITY",
    "THUMBNAIL_WIDTH",
    "CostRollup",
    "SessionIndex",
    "SessionRow",
    "SessionStore",
    "data_root",
    "default_title",
    "new_session_id",
    "sessions_root",
    "slugify",
    "summarise",
]
