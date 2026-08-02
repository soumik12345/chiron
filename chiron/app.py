"""Chiron's entry point: the wiring that makes the parts one application.

Everything runs on a single event loop. ``qasync`` drives asyncio *inside* Qt's
loop, so the Live session's coroutines and the overlay's widgets share a thread
and neither has to marshal calls to the other. The one genuinely concurrent piece
is screen capture, which lives on its own thread and comes back as Qt signals.

The flow through the app is small enough to state in full:

* the capture thread produces frames, which go to the session (newest wins) and
  to the journal writer;
* the player's question bursts the shutter and is sent as text;
* the model's answer streams back into the overlay;
* notable events become journal entries — by the model calling ``record_event``
  or by the sidecar summariser, depending on the setting;
* every couple of minutes the new journal lines are folded back into the session,
  so they outlive the frames that produced them;
* when the session dies, it is reopened and re-seeded from the journal.

Saving settings goes through here too, because only this object knows which
changes can be applied in place and which need the session rotated.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import signal
import sys
import time
from pathlib import Path
from typing import Callable, TextIO

import qasync
from PySide6.QtCore import QObject, QTimer
from PySide6.QtWidgets import QApplication

from chiron.capture.active_window import ActiveWindowTracker, WindowInfo
from chiron.capture.frames import Frame
from chiron.capture.service import CaptureService
from chiron.config.settings import (
    Settings,
    default_settings_path,
    load_settings,
    plan_removal,
    remove_configuration,
    save_settings,
)
from chiron.journal.log import JournalEntry, JournalLog
from chiron.journal.writers import JournalWriter, build_journal_writer
from chiron.live.session import LiveSessionManager
from chiron.ui.hotkeys import GlobalHotkeyManager
from chiron.ui.overlay import OverlayWindow
from chiron.ui.settings_window import SettingsWindow

logger = logging.getLogger(__name__)

#: How often the overlay's footer line is refreshed.
_FOOTER_INTERVAL_MS = 1000


class ChironApp(QObject):
    """Owns every component and the connections between them.

    Attributes:
        settings (Settings): The configuration currently in force.
        settings_path (Path): Where settings are persisted.
        journal (JournalLog): The shared journal.
        writer (JournalWriter): The active journal strategy.
        session (LiveSessionManager): The Live API connection.
        capture (CaptureService): The screen capture thread.
        overlay (OverlayWindow): The floating chat panel.
    """

    def __init__(
        self,
        settings: Settings,
        settings_path: Path,
        parent: QObject | None = None,
    ) -> None:
        """Build every component and wire them together."""
        super().__init__(parent)
        self.settings = settings
        self.settings_path = settings_path
        self.quit_requested = asyncio.Event()
        self._warned_not_watching = False

        self.journal = JournalLog(max_entries=settings.journal.max_entries)
        self.writer: JournalWriter = self._build_writer()
        self.session = LiveSessionManager(settings, self.journal, self.writer, self)
        self.capture = CaptureService(settings.capture, self)
        self.overlay = OverlayWindow(settings.overlay)
        self.settings_window: SettingsWindow | None = None
        self.hotkeys = GlobalHotkeyManager(self)
        self.window_tracker = ActiveWindowTracker(self._chiron_window_ids, self)

        self.fold_timer = QTimer(self)
        self.fold_timer.setInterval(int(settings.journal.fold_interval_seconds * 1000))
        self.fold_timer.timeout.connect(lambda: self.session.fold_journal())

        self.footer_timer = QTimer(self)
        self.footer_timer.setInterval(_FOOTER_INTERVAL_MS)
        self.footer_timer.timeout.connect(self._refresh_footer)

        self._connect()

    # --------------------------------------------------------------- wiring

    def _build_writer(self) -> JournalWriter:
        """Create the journal writer named by the current settings."""
        return build_journal_writer(
            self.settings.journal,
            self.journal,
            api_key=self.settings.resolved_api_key(),
            on_entry=self._on_journal_entry,
        )

    def _connect(self) -> None:
        """Connect every signal to its handler."""
        self.overlay.promptSubmitted.connect(self._on_prompt)
        self.overlay.settingsRequested.connect(self.show_settings)
        self.overlay.panelHidden.connect(self._remember_geometry)
        self.overlay.quitRequested.connect(self.request_quit)
        self.overlay.watchToggled.connect(self.set_watching)

        self.capture.frameCaptured.connect(self._on_frame)
        self.capture.sceneChanged.connect(self._on_scene_change)
        self.capture.errorOccurred.connect(self._on_capture_error)

        self.session.statusChanged.connect(self.overlay.set_status)
        self.session.responseStarted.connect(self.overlay.start_response)
        self.session.responseDelta.connect(self.overlay.append_delta)
        self.session.responseCompleted.connect(self.overlay.end_response)
        self.session.errorOccurred.connect(self._on_session_error)

        self.window_tracker.windowChanged.connect(self._on_active_window)

        self.hotkeys.activated.connect(self._on_hotkey)
        self.hotkeys.failed.connect(
            lambda message: self.overlay.append_system(f"⚠ {message}")
        )

    # ------------------------------------------------------------ lifecycle

    def start(self) -> None:
        """Show the overlay and arm everything — without looking at the screen.

        Launching Chiron deliberately does not start watching. The capture thread
        comes up idle and waits to be told, so nothing is read until the player
        asks for it.
        """
        self.overlay.set_status("idle", "")
        self.overlay.set_watching(False, self.settings.hotkeys.toggle_watching)
        self.overlay.set_hide_hint(self.settings.hotkeys.toggle_overlay)
        if not self.settings.overlay.start_hidden:
            self.overlay.show_and_focus()
        self.overlay.append_system(
            f"Chiron is not watching yet. Press "
            f"{self._watch_hotkey_hint()} to start, "
            f"{self.settings.hotkeys.toggle_overlay} to show or hide this panel. "
            f"Run your game borderless-windowed."
        )

        self.hotkeys.set_bindings(self._hotkey_bindings())
        self.hotkeys.start()
        self.window_tracker.start()
        self.capture.start()
        self.fold_timer.start()
        self.footer_timer.start()

        if not self.settings.resolved_api_key():
            self.overlay.append_system(
                "No Gemini API key configured — open Settings (⚙) to add one."
            )
        if self.settings.capture.watch_on_launch:
            self.set_watching(True)

    def _hotkey_bindings(self) -> dict[str, str]:
        """The name → combination map handed to the hotkey manager."""
        hotkeys = self.settings.hotkeys
        return {
            "toggle_overlay": hotkeys.toggle_overlay,
            "open_settings": hotkeys.open_settings,
            "toggle_watching": hotkeys.toggle_watching,
            "start_watching": hotkeys.start_watching,
            "stop_watching": hotkeys.stop_watching,
        }

    def _watch_hotkey_hint(self) -> str:
        """However the player has chosen to start watching, in words."""
        hotkeys = self.settings.hotkeys
        if hotkeys.toggle_watching.strip():
            return hotkeys.toggle_watching
        if hotkeys.start_watching.strip():
            return hotkeys.start_watching
        return "the eye button above"

    async def shutdown(self) -> None:
        """Stop everything and persist the overlay's placement."""
        logger.info("Shutting down")
        self.fold_timer.stop()
        self.footer_timer.stop()
        self.hotkeys.stop()
        self.window_tracker.stop()
        self.capture.stop()
        self._remember_geometry()
        try:
            save_settings(self.settings, self.settings_path)
        except OSError as error:
            logger.warning("Could not save settings on exit: %s", error)
        await self.session.stop()

    def request_quit(self) -> None:
        """Ask the main loop to shut the application down."""
        self.quit_requested.set()

    # -------------------------------------------------------------- watching

    @property
    def watching(self) -> bool:
        """Whether Chiron is currently reading the screen."""
        return self.capture.is_watching

    def set_watching(self, watching: bool) -> None:
        """Start or stop watching the screen.

        Starting also opens the live session, since frames with nowhere to go are
        pure cost. Stopping closes it: "stop watching" should leave nothing
        running that could still see anything, and the journal means the next
        session picks up knowing what happened rather than starting blank.

        Args:
            watching (bool): The state to move to. Repeating the current state
                does nothing.
        """
        if watching == self.watching:
            return

        if watching:
            # A fresh look before anything else: while `self.watching` is still
            # False this cannot double-journal through _on_active_window.
            self.window_tracker.poll()

        self.capture.set_watching(watching)
        self.overlay.set_watching(watching, self.settings.hotkeys.toggle_watching)

        if watching:
            info = self.window_tracker.current
            if info is not None and not self.settings.game_name.strip():
                self.session.detected_game = info.describe()
                self.journal.append(
                    f"Watching started; the player is in {info.describe()}.",
                    source="system",
                )
                self.overlay.append_system(
                    f"● Watching your screen — looks like {info.label}."
                )
            else:
                self.overlay.append_system("● Watching your screen.")
            if self.session.status in ("idle", "stopped", "error"):
                self.session.start()
        else:
            self.overlay.append_system(
                "○ Stopped watching. Chiron can still answer from what it "
                "already noted."
            )
            asyncio.ensure_future(self.session.stop())

    def toggle_watching(self) -> None:
        """Flip the watching state."""
        self.set_watching(not self.watching)

    # -------------------------------------------------------------- handlers

    def _on_prompt(self, text: str) -> None:
        """Send a question, bursting the shutter so the answer is about *now*.

        A question asked while not watching is still sent — the model can answer
        from the journal and the conversation — but the player is told once that
        nothing on screen is being seen, because "I can't see your screen" from a
        screen-watching assistant is otherwise baffling.
        """
        self.overlay.append_user(text)
        if self.watching:
            self.capture.request_burst("question")
        elif not self._warned_not_watching:
            self._warned_not_watching = True
            self.overlay.append_system(
                f"Chiron is not watching, so it cannot see your screen right now "
                f"— press {self._watch_hotkey_hint()} to let it look."
            )
        if self.session.status in ("idle", "stopped"):
            self.session.start()
        self.session.send_text(text)

    def _on_frame(self, frame: Frame) -> None:
        """Forward a captured frame to the session."""
        self.session.send_frame(frame)

    def _on_scene_change(self, distance: float) -> None:
        """Log a hard scene change; the capture service has already burst."""
        logger.debug("Scene change detected (distance %.3f)", distance)

    def _on_capture_error(self, message: str) -> None:
        """Report a capture failure in the overlay."""
        self.overlay.append_system(f"⚠ {message}")

    def _on_session_error(self, message: str) -> None:
        """Report a session failure in the overlay."""
        self.overlay.append_system(f"⚠ {message}")

    def _on_journal_entry(self, entry: JournalEntry) -> None:
        """Show a new journal entry inline in the transcript."""
        self.overlay.append_journal(entry)

    def _chiron_window_ids(self) -> set[int]:
        """Chiron's own window ids, which can never be "the player's window"."""
        ids = {int(self.overlay.winId())}
        if self.settings_window is not None:
            ids.add(int(self.settings_window.winId()))
        return ids

    def _on_active_window(self, info: WindowInfo) -> None:
        """Note that the player moved to a different application.

        The session's ``detected_game`` is kept current so the next connection's
        instruction names the right game, and while watching, the switch is
        journaled — the running session's instruction is fixed, so the fold is
        how it learns mid-session.
        """
        if not self.settings.game_name.strip():
            self.session.detected_game = info.describe()
        if self.watching:
            self.journal.append(
                f"The player switched to {info.describe()}.", source="system"
            )

    def _on_hotkey(self, name: str) -> None:
        """Act on a global hotkey."""
        if name == "toggle_overlay":
            self.overlay.toggle()
        elif name == "open_settings":
            self.show_settings()
        elif name == "toggle_watching":
            self.toggle_watching()
        elif name == "start_watching":
            self.set_watching(True)
        elif name == "stop_watching":
            self.set_watching(False)

    def _refresh_footer(self) -> None:
        """Update the overlay's small status line."""
        if not self.watching:
            shutter = "not watching"
        else:
            mode = self.capture.scheduler.mode(time.time())
            reason = self.capture.scheduler.burst_reason
            shutter = f"{mode} ({reason})" if reason else mode
        self.overlay.set_footer(
            f"shutter: {shutter}  ·  frames sent: {self.session.frames_sent}"
            f"  ·  journal: {len(self.journal)}"
        )

    # -------------------------------------------------------------- settings

    def show_settings(self) -> None:
        """Open (or raise) the settings window."""
        if self.settings_window is None:
            self.settings_window = SettingsWindow(self.settings)
            self.settings_window.settingsSaved.connect(self.apply_settings)
            self.settings_window.appearanceChanged.connect(self.overlay.apply_settings)
            self.settings_window.quitRequested.connect(self.request_quit)
        self.settings_window.set_hotkey_backend(self.hotkeys.backend_name)
        current = self.window_tracker.current
        self.settings_window.set_detected_game(current.label if current else "")
        self.settings_window.load(self.settings)
        self.settings_window.show()
        self.settings_window.raise_()
        self.settings_window.activateWindow()

    def apply_settings(self, new_settings: Settings) -> None:
        """Persist and apply edited settings, restarting the session if needed.

        Args:
            new_settings (Settings): The configuration to adopt.
        """
        previous = self.settings
        # Placement is owned by the overlay, not the settings form.
        self._remember_geometry()
        new_settings.overlay.position_x = previous.overlay.position_x
        new_settings.overlay.position_y = previous.overlay.position_y
        self.settings = new_settings

        try:
            save_settings(new_settings, self.settings_path)
        except OSError as error:
            self.overlay.append_system(f"⚠ Could not save settings: {error}")

        self.capture.apply_settings(new_settings.capture)
        self.overlay.apply_settings(new_settings.overlay)
        self.journal.max_entries = new_settings.journal.max_entries
        self.fold_timer.setInterval(
            int(new_settings.journal.fold_interval_seconds * 1000)
        )
        self.hotkeys.set_bindings(self._hotkey_bindings())
        self.overlay.set_watching(self.watching, new_settings.hotkeys.toggle_watching)
        self.overlay.set_hide_hint(new_settings.hotkeys.toggle_overlay)
        self.session.apply_settings(new_settings)

        strategy_changed = previous.journal.strategy != new_settings.journal.strategy
        if previous.requires_session_restart(new_settings) or strategy_changed:
            asyncio.ensure_future(self._restart_session(strategy_changed))
        self.overlay.append_system("Settings saved.")

    async def _restart_session(self, rebuild_writer: bool) -> None:
        """Reconnect with the new configuration, keeping the journal.

        Args:
            rebuild_writer (bool): True when the journal strategy changed, which
                means the old writer has to be stopped and replaced.
        """
        await self.session.stop()
        if rebuild_writer:
            await self.writer.stop()
            self.writer = self._build_writer()
            self.session.writer = self.writer
        # Reconnect only if the screen is still being watched. Editing settings
        # must never be a back door into watching.
        if self.watching:
            self.session.start()

    def _remember_geometry(self) -> None:
        """Record the overlay's position and size into settings."""
        x, y, width, height = self.overlay.current_geometry()
        self.settings.overlay.position_x = x
        self.settings.overlay.position_y = y
        self.settings.overlay.width = width
        self.settings.overlay.height = height


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Args:
        argv (list[str] | None): Arguments to parse; defaults to ``sys.argv``.

    Returns:
        argparse.Namespace: Parsed options.
    """
    parser = argparse.ArgumentParser(
        prog="chiron", description="An AI gaming assistant that watches your screen."
    )
    parser.add_argument(
        "--settings",
        type=Path,
        default=None,
        help="Path to the settings file (default: ~/.config/chiron/settings.json)",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity (default: INFO)",
    )
    parser.add_argument(
        "--fresh-install",
        action="store_true",
        help=(
            "Delete Chiron's saved configuration — including any saved API key — "
            "and exit, so the next run starts as if newly installed. Asks first."
        ),
    )
    parser.add_argument(
        "--yes",
        "-y",
        action="store_true",
        help="Answer yes to the --fresh-install confirmation (for scripts).",
    )
    return parser.parse_args(argv)


def fresh_install(
    settings_path: Path,
    *,
    assume_yes: bool = False,
    stream: TextIO | None = None,
    confirm: Callable[[str], str] | None = None,
) -> int:
    """Delete the saved configuration after showing exactly what goes.

    Deletion is irreversible and takes the API key with it, so the plan is
    printed first and confirmation is required. When the answer cannot be asked
    for — a pipe, a cron job, no terminal — nothing is deleted and the caller is
    told to pass ``--yes``. Silence is not consent.

    Args:
        settings_path (Path): The settings file to remove.
        assume_yes (bool): Skip the prompt.
        stream (TextIO | None): Where to write. Defaults to stdout.
        confirm (Callable[[str], str] | None): Prompt function. Defaults to
            :func:`input`, and is only called on an interactive terminal.

    Returns:
        int: Process exit code — 0 for done or nothing to do, 1 for cancelled or
            unable to ask.
    """
    out = stream or sys.stdout
    plan = plan_removal(settings_path)

    if plan.is_empty:
        print(f"Nothing to remove — no configuration at {settings_path}.", file=out)
        return 0

    print("This will:", file=out)
    for line in plan.describe():
        print(f"  {line}", file=out)
    if plan.holds_api_key:
        print(
            "\nYour saved Gemini API key is in that file and will be gone. "
            "No backup is kept.",
            file=out,
        )

    if not assume_yes:
        ask = confirm
        if ask is None:
            # Only the real prompt needs a terminal to read from.
            if not sys.stdin.isatty():
                print(
                    "\nNot a terminal, so nothing was removed. "
                    "Re-run with --yes if you meant it.",
                    file=out,
                )
                return 1
            ask = input
        answer = ask("\nRemove it? [y/N] ").strip().lower()
        if answer not in ("y", "yes"):
            print("Cancelled; nothing was removed.", file=out)
            return 1

    removed = remove_configuration(settings_path)
    for line in removed.describe():
        if line.startswith("delete"):
            print(f"Removed {line.split(maxsplit=1)[1]}", file=out)
    print("Chiron is back to a fresh install. Run `chiron` to start again.", file=out)
    return 0


def main(argv: list[str] | None = None) -> int:
    """Run Chiron.

    Args:
        argv (list[str] | None): Command-line arguments.

    Returns:
        int: Process exit code.
    """
    args = parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
    )

    settings_path = Path(args.settings or default_settings_path())
    if args.fresh_install:
        # Handled before any Qt object exists: this command never wants a window,
        # and a GUI that fails to start must not stop someone wiping their config.
        return fresh_install(settings_path, assume_yes=args.yes)

    app = QApplication(sys.argv)
    app.setApplicationName("Chiron")
    app.setApplicationDisplayName("Chiron")
    # Hiding the overlay must not end the process — the hotkey has to be able to
    # bring it back.
    app.setQuitOnLastWindowClosed(False)

    loop = qasync.QEventLoop(app)
    asyncio.set_event_loop(loop)

    chiron = ChironApp(load_settings(settings_path), settings_path)

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, chiron.request_quit)
        except (NotImplementedError, RuntimeError):  # pragma: no cover - platform
            signal.signal(sig, lambda *_: chiron.request_quit())

    app.aboutToQuit.connect(chiron.request_quit)

    with loop:
        chiron.start()
        loop.run_until_complete(chiron.quit_requested.wait())
        loop.run_until_complete(chiron.shutdown())
    return 0


if __name__ == "__main__":  # pragma: no cover - manual entry point
    raise SystemExit(main())
