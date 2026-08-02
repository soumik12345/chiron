"""Which window the player is actually in front of, kept current from launch.

The overlay can be switched on at any moment, so the question "what game is
this?" has to already have an answer by the time watching starts — asking X11
*at* that moment is too late, because clicking the overlay's eye button can make
the active window Chiron itself. So a small tracker polls the display from app
launch onwards and remembers the most recent focused window that is *not* one of
Chiron's own. Whatever it holds when watching begins is what the session is told
the player is playing.

Identity comes from three places, best first:

* the ``STEAM_GAME`` window property Steam stamps on game windows (an appid,
  resolved to the game's real name from the local ``appmanifest_*.acf`` — no
  network involved);
* the ``SteamAppId`` environment variable of the window's process, for windows
  Steam launched but did not stamp;
* the window's title, ``WM_CLASS`` and process name, for everything else.

Everything X11 lives behind :class:`X11WindowResolver`; the Steam and ``/proc``
helpers are pure functions over paths so the tests never need a display or a
Steam install. On Wayland (or any display the resolver cannot open) the tracker
simply stays empty and Chiron behaves exactly as before this feature existed.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

from PySide6.QtCore import QObject, QTimer, Signal

logger = logging.getLogger(__name__)

#: How often the focused window is re-checked. Cheap — a handful of X round
#: trips — so this is about journal granularity, not cost.
POLL_INTERVAL_MS = 2000

#: Consecutive poll failures after which the tracker gives up. A display that
#: errors this many times in a row is gone, not busy.
_MAX_FAILURES = 5

#: Steam roots that exist across native and flatpak installs. ``~/.steam/steam``
#: and ``~/.steam/root`` are usually symlinks into ``~/.local/share/Steam``;
#: duplicates are resolved away.
_STEAM_ROOTS = (
    "~/.local/share/Steam",
    "~/.steam/steam",
    "~/.steam/root",
    "~/.var/app/com.valvesoftware.Steam/.local/share/Steam",
)

#: EWMH window types that can never be the thing the player is playing.
_IGNORED_WINDOW_TYPES = (
    "_NET_WM_WINDOW_TYPE_DESKTOP",
    "_NET_WM_WINDOW_TYPE_DOCK",
    "_NET_WM_WINDOW_TYPE_NOTIFICATION",
    "_NET_WM_WINDOW_TYPE_TOOLTIP",
    "_NET_WM_WINDOW_TYPE_SPLASH",
)

_ACF_NAME = re.compile(r'^\s*"name"\s+"(?P<name>(?:\\.|[^"\\])*)"', re.MULTILINE)
_VDF_PATH = re.compile(r'^\s*"path"\s+"(?P<path>(?:\\.|[^"\\])*)"', re.MULTILINE)

#: Environment variables carrying the appid, in order of trust. ``SteamAppId``
#: is the real thing; ``STEAM_COMPAT_APP_ID`` covers Proton titles.
_APPID_ENV_KEYS = (b"SteamAppId=", b"STEAM_COMPAT_APP_ID=")


@dataclass(frozen=True)
class WindowInfo:
    """What is known about one top-level window.

    Attributes:
        window_id (int): X11 window id.
        title (str): The window's title, or empty.
        wm_class (str): The ``WM_CLASS`` class name, or empty.
        pid (int | None): Owning process id, when the window advertises one.
        process (str): The process's ``comm`` name, or empty.
        steam_app_id (int | None): Steam appid, when the window is a Steam game.
        steam_name (str): The game's name from its appmanifest, or empty.
    """

    window_id: int
    title: str = ""
    wm_class: str = ""
    pid: int | None = None
    process: str = ""
    steam_app_id: int | None = None
    steam_name: str = ""

    @property
    def label(self) -> str:
        """The best short human-readable name for this window."""
        return (
            self.steam_name
            or self.title
            or self.wm_class
            or self.process
            or f"window 0x{self.window_id:x}"
        )

    @property
    def identity(self) -> tuple[str, object]:
        """A key that survives title churn, for change detection.

        Titles change constantly (browsers, chapter names in games), so a
        "switched windows" event keys on the appid or application class instead.
        """
        if self.steam_app_id:
            return ("steam", self.steam_app_id)
        if self.wm_class:
            return ("class", self.wm_class.lower())
        return ("window", self.window_id)

    def describe(self) -> str:
        """The window as a phrase fit for the system instruction."""
        if self.steam_name:
            return f"{self.steam_name} (Steam)"
        if self.steam_app_id:
            return f"Steam app {self.steam_app_id}"
        application = self.wm_class or self.process
        if self.title and application and self.title.lower() != application.lower():
            return f"'{self.title}' ({application})"
        return self.label


# ------------------------------------------------------------- Steam, on disk


def parse_appmanifest_name(text: str) -> str:
    """Extract the game name from an ``appmanifest_*.acf`` file's text.

    Args:
        text (str): The ACF file contents.

    Returns:
        str: The ``name`` value, or an empty string when there is none.
    """
    match = _ACF_NAME.search(text)
    if match is None:
        return ""
    return match.group("name").replace('\\"', '"').replace("\\\\", "\\")


def steam_library_steamapps(roots: Iterable[Path] | None = None) -> list[Path]:
    """Every ``steamapps`` directory Steam is using, main install and libraries.

    Extra library folders live in ``steamapps/libraryfolders.vdf`` under the
    main install; each contributes its own ``steamapps``.

    Args:
        roots (Iterable[Path] | None): Steam install roots to inspect. Defaults
            to the usual native and flatpak locations.

    Returns:
        list[Path]: Existing ``steamapps`` directories, deduplicated.
    """
    if roots is None:
        roots = [Path(p).expanduser() for p in _STEAM_ROOTS]
    found: list[Path] = []
    seen: set[Path] = set()

    def add(steamapps: Path) -> None:
        if not steamapps.is_dir():
            return
        key = steamapps.resolve()
        if key not in seen:
            seen.add(key)
            found.append(steamapps)

    for root in roots:
        steamapps = root / "steamapps"
        add(steamapps)
        manifest = steamapps / "libraryfolders.vdf"
        if manifest.is_file():
            try:
                text = manifest.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            for match in _VDF_PATH.finditer(text):
                library = match.group("path").replace("\\\\", "\\")
                add(Path(library) / "steamapps")
    return found


def steam_game_name(app_id: int, steamapps_dirs: Iterable[Path]) -> str:
    """The installed name of a Steam appid, from its local appmanifest.

    Args:
        app_id (int): The Steam appid.
        steamapps_dirs (Iterable[Path]): ``steamapps`` directories to search.

    Returns:
        str: The game's name, or an empty string when it is not installed here.
    """
    for steamapps in steamapps_dirs:
        manifest = steamapps / f"appmanifest_{app_id}.acf"
        if not manifest.is_file():
            continue
        try:
            name = parse_appmanifest_name(
                manifest.read_text(encoding="utf-8", errors="replace")
            )
        except OSError:
            continue
        if name:
            return name
    return ""


# ------------------------------------------------------------------- /proc


def steam_app_id_from_environ(pid: int, proc_root: str | Path = "/proc") -> int | None:
    """The Steam appid a process was launched with, if any.

    Steam exports ``SteamAppId`` into every game it starts, which identifies
    games whose windows carry no ``STEAM_GAME`` property.

    Args:
        pid (int): The process to inspect.
        proc_root (str | Path): The proc filesystem root, swappable for tests.

    Returns:
        int | None: The appid, or None when the process is not a Steam game.
    """
    try:
        raw = (Path(proc_root) / str(pid) / "environ").read_bytes()
    except OSError:
        return None
    for chunk in raw.split(b"\0"):
        for key in _APPID_ENV_KEYS:
            if chunk.startswith(key):
                value = chunk[len(key) :]
                if value.isdigit() and int(value) > 0:
                    return int(value)
    return None


def process_name(pid: int, proc_root: str | Path = "/proc") -> str:
    """A process's short ``comm`` name, or an empty string."""
    try:
        return (Path(proc_root) / str(pid) / "comm").read_text().strip()
    except OSError:
        return ""


def ordered_candidates(active: int | None, stacking: list[int]) -> list[int]:
    """Window ids in the order they should be considered.

    The active window is the player's answer when it is usable; after that the
    stacking order top-down — ``_NET_CLIENT_LIST_STACKING`` is bottom-to-top, so
    it is walked in reverse. With the overlay always-on-top, the first eligible
    window below it is exactly what the player is looking at.

    Args:
        active (int | None): The ``_NET_ACTIVE_WINDOW`` id, if any.
        stacking (list[int]): ``_NET_CLIENT_LIST_STACKING``, bottom-to-top.

    Returns:
        list[int]: Candidate ids, best first, without duplicates.
    """
    ordered = [active] if active else []
    ordered.extend(wid for wid in reversed(stacking) if wid and wid != active)
    return ordered


# ---------------------------------------------------------------------- X11


class X11WindowResolver:
    """Answers "which window is the player in?" by asking the X server.

    One persistent display connection, reopened on the next call if an error
    forces it closed. Steam appid → name lookups are cached, since the manifest
    on disk does not change mid-session.
    """

    def __init__(self, steamapps_dirs: list[Path] | None = None) -> None:
        self._display = None
        self._steamapps = (
            steamapps_dirs if steamapps_dirs is not None else steam_library_steamapps()
        )
        self._steam_names: dict[int, str] = {}

    @staticmethod
    def available() -> bool:
        """Whether python-xlib can open the current display."""
        try:
            from Xlib import display

            display.Display().close()
            return True
        except Exception:
            return False

    def close(self) -> None:
        """Drop the display connection; the next resolve reopens it."""
        if self._display is not None:
            try:
                self._display.close()
            except Exception:  # pragma: no cover - teardown noise
                logger.debug("Closing X display failed", exc_info=True)
            self._display = None

    def resolve(self, excluded: set[int]) -> WindowInfo | None:
        """The window the player is in, or None when there is no candidate.

        Args:
            excluded (set[int]): Window ids that can never be the answer —
                Chiron's own windows.

        Raises:
            Exception: Any X11 failure; the connection is dropped first so the
                caller can simply try again next poll.
        """
        try:
            return self._resolve(excluded)
        except Exception:
            self.close()
            raise

    # ------------------------------------------------------------- internals

    def _conn(self):
        if self._display is None:
            from Xlib import display

            self._display = display.Display()
        return self._display

    def _atom(self, name: str) -> int:
        return self._conn().intern_atom(name)

    def _property(self, window, name: str):
        from Xlib import X

        prop = window.get_full_property(self._atom(name), X.AnyPropertyType)
        return prop.value if prop is not None else None

    def _resolve(self, excluded: set[int]) -> WindowInfo | None:
        conn = self._conn()
        root = conn.screen().root

        active_value = self._property(root, "_NET_ACTIVE_WINDOW")
        active = int(active_value[0]) if active_value else None
        stacking_value = self._property(root, "_NET_CLIENT_LIST_STACKING")
        stacking = [int(wid) for wid in stacking_value] if stacking_value else []

        ignored_types = {self._atom(name) for name in _IGNORED_WINDOW_TYPES}
        for wid in ordered_candidates(active, stacking):
            if wid in excluded:
                continue
            window = conn.create_resource_object("window", wid)
            types = self._property(window, "_NET_WM_WINDOW_TYPE")
            if types is not None and any(int(t) in ignored_types for t in types):
                continue
            info = self._window_info(window, wid)
            # Chiron's windows are excluded by id, but a window of ours created
            # after the caller built its exclusion set is caught by class.
            if info.wm_class.lower().startswith("chiron"):
                continue
            return info
        return None

    def _window_info(self, window, wid: int) -> WindowInfo:
        title = self._text(self._property(window, "_NET_WM_NAME"))
        if not title:
            title = self._text(self._property(window, "WM_NAME"))

        wm_class = ""
        try:
            pair = window.get_wm_class()
        except Exception:
            pair = None
        if pair:
            wm_class = pair[1] or pair[0] or ""

        pid_value = self._property(window, "_NET_WM_PID")
        pid = int(pid_value[0]) if pid_value else None

        steam_value = self._property(window, "STEAM_GAME")
        steam_app_id = int(steam_value[0]) if steam_value else None
        if steam_app_id is None and pid is not None:
            steam_app_id = steam_app_id_from_environ(pid)

        return WindowInfo(
            window_id=wid,
            title=title,
            wm_class=wm_class,
            pid=pid,
            process=process_name(pid) if pid is not None else "",
            steam_app_id=steam_app_id,
            steam_name=self._steam_name(steam_app_id) if steam_app_id else "",
        )

    def _steam_name(self, app_id: int) -> str:
        if app_id not in self._steam_names:
            self._steam_names[app_id] = steam_game_name(app_id, self._steamapps)
        return self._steam_names[app_id]

    @staticmethod
    def _text(value) -> str:
        if value is None:
            return ""
        if isinstance(value, bytes):
            return value.decode("utf-8", errors="replace").strip()
        return str(value).strip()


class ActiveWindowTracker(QObject):
    """Remembers the player's window so the answer predates the question.

    Polls from application start — deliberately *before* watching begins, so
    that clicking the overlay (which can steal focus to Chiron) never erases
    the answer: Chiron's own windows are excluded, and ``current`` simply keeps
    pointing at the last real window.

    Signals:
        windowChanged (object): The player moved to a different application;
            carries the new :class:`WindowInfo`.

    Attributes:
        current (WindowInfo | None): The last non-Chiron focused window, or None
            when nothing has been seen (or no display is available).
    """

    windowChanged = Signal(object)

    def __init__(
        self,
        exclude_ids: Callable[[], set[int]],
        parent: QObject | None = None,
        interval_ms: int = POLL_INTERVAL_MS,
        resolver: X11WindowResolver | None = None,
    ) -> None:
        """Create a tracker that is inert until :meth:`start`.

        Args:
            exclude_ids (Callable[[], set[int]]): Supplies Chiron's own window
                ids, called fresh on every poll since windows come and go.
            parent (QObject | None): Qt parent.
            interval_ms (int): Poll cadence.
            resolver (X11WindowResolver | None): Injected resolver, for tests.
        """
        super().__init__(parent)
        self._exclude_ids = exclude_ids
        self._resolver = resolver
        self.current: WindowInfo | None = None
        self._failures = 0
        self._timer = QTimer(self)
        self._timer.setInterval(interval_ms)
        self._timer.timeout.connect(self.poll)

    def start(self) -> None:
        """Begin polling; a no-op when no display can be opened."""
        if self._resolver is None:
            if not X11WindowResolver.available():
                logger.info("No X11 display available; active-window detection is off")
                return
            self._resolver = X11WindowResolver()
        self.poll()
        self._timer.start()

    def stop(self) -> None:
        """Stop polling and release the display connection."""
        self._timer.stop()
        if self._resolver is not None:
            self._resolver.close()

    def poll(self) -> None:
        """Check the focused window once, emitting on an application change."""
        if self._resolver is None:
            return
        try:
            info = self._resolver.resolve(set(self._exclude_ids()))
        except Exception as error:
            self._failures += 1
            logger.debug("Active-window poll failed: %s", error)
            if self._failures >= _MAX_FAILURES:
                logger.warning(
                    "Active-window detection disabled after %d failures: %s",
                    self._failures,
                    error,
                )
                self._timer.stop()
            return
        self._failures = 0
        if info is None:
            return
        previous = self.current
        self.current = info
        if previous is None or previous.identity != info.identity:
            logger.info("Player's window is now: %s", info.describe())
            self.windowChanged.emit(info)


__all__ = [
    "POLL_INTERVAL_MS",
    "ActiveWindowTracker",
    "WindowInfo",
    "X11WindowResolver",
    "ordered_candidates",
    "parse_appmanifest_name",
    "process_name",
    "steam_app_id_from_environ",
    "steam_game_name",
    "steam_library_steamapps",
]
