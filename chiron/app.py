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
* notable events become journal entries — by the model calling ``record_event``,
  by the sidecar summariser, or by the non-live observer, depending on the mode
  and the setting;
* every couple of minutes the new journal lines are folded back into the session,
  so they outlive the frames that produced them;
* when the session dies, it is reopened and re-seeded from the journal.

Which *kind* of session that is depends entirely on the selected model, and this
object is where the choice is made concrete: :func:`~chiron.session.build_session_provider`
returns a live or a non-live manager, and everything else here is written against
the surface they share. Switching models across that boundary tears one down and
builds the other — the same path as a session restart, one step longer.

Saving settings goes through here too, because only this object knows which
changes can be applied in place and which need the session rotated.

v2 adds a second, slower clock to all of this: the **gameplay session**. Launch
opens none. The first watch-start or first message creates one, and everything
above is written down as it happens by
:class:`~chiron.sessions.recorder.SessionRecorder`, which observes this object
through the signals it was already wiring. **New Session** closes that record and
performs the full reset — journal cleared, conversation cleared, observer
baseline reset — which has a unifying effect on the code: "start a new session"
becomes the one canonical reset path, and a fresh launch is simply *no session
open yet* rather than a separate implicit reset nobody wrote down.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
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
from chiron.session import SessionProvider, build_session_provider
from chiron.sessions.recorder import SessionRecorder
from chiron.ui.hotkeys import GlobalHotkeyManager
from chiron.ui.overlay import SESSION_VIEW, OverlayWindow
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
        session (SessionProvider): The live or non-live session, whichever the
            selected model implies.
        capture (CaptureService): The screen capture thread.
        overlay (OverlayWindow): The floating chat panel.
        recorder (SessionRecorder): The gameplay session being written to disk,
            which outlives `session` across a live/non-live swap.
    """

    def __init__(
        self,
        settings: Settings,
        settings_path: Path,
        parent: QObject | None = None,
        *,
        sessions_root: Path | None = None,
    ) -> None:
        """Build every component and wire them together.

        Args:
            settings (Settings): The configuration to run with.
            settings_path (Path): Where to persist it.
            parent (QObject | None): Qt parent.
            sessions_root (Path | None): Where gameplay sessions are recorded.
                Defaults to the XDG data directory — deliberately *not* the
                config directory, so ``--fresh-install`` cannot take a season of
                play with it.
        """
        super().__init__(parent)
        self.settings = settings
        self.settings_path = settings_path
        self.quit_requested = asyncio.Event()
        self._warned_not_watching = False

        self.recorder = SessionRecorder(sessions_root, self)
        self.journal = JournalLog(max_entries=settings.journal.max_entries)
        self.writer: JournalWriter = self._build_writer()
        self.session: SessionProvider = build_session_provider(
            settings, self.journal, self.writer, self
        )
        self.capture = CaptureService(settings.effective_capture(), self)
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
        """Create the journal writer the current mode and settings call for."""
        return build_journal_writer(
            self.settings.journal,
            self.journal,
            api_key=self.settings.resolved_api_key(),
            on_entry=self._on_journal_entry,
            live=self.settings.is_live,
            usage_sink=self.recorder.record_llm_call,
        )

    def _connect(self) -> None:
        """Connect every signal to its handler.

        Both session providers emit the same five signals, so this method has no
        idea which one it is wiring — and neither does the overlay.
        """
        self.overlay.promptSubmitted.connect(self._on_prompt)
        self.overlay.settingsRequested.connect(self.show_settings)
        self.overlay.panelHidden.connect(self._remember_geometry)
        self.overlay.quitRequested.connect(self.request_quit)
        self.overlay.watchToggled.connect(self.set_watching)

        self.overlay.newSessionRequested.connect(self.new_session)
        self.overlay.historyRequested.connect(self.show_history)
        self.overlay.sessionOpened.connect(self.open_session)
        self.overlay.sessionRenamed.connect(self._on_session_renamed)
        self.overlay.sessionDeleted.connect(self._on_session_deleted)
        self.overlay.sessionThumbnailsDeleted.connect(self._on_thumbnails_deleted)

        self.recorder.sessionChanged.connect(self.overlay.set_session)
        self.recorder.sessionChanged.connect(self._on_session_changed)
        self.recorder.costChanged.connect(self.overlay.set_cost)

        self.capture.frameCaptured.connect(self._on_frame)
        self.capture.sceneChanged.connect(self._on_scene_change)
        self.capture.errorOccurred.connect(self._on_capture_error)

        self._connect_session()

        self.window_tracker.windowChanged.connect(self._on_active_window)

        self.hotkeys.activated.connect(self._on_hotkey)
        self.hotkeys.failed.connect(
            lambda message: self.overlay.append_system(f"⚠ {message}")
        )

    def _connect_session(self) -> None:
        """Wire the current session provider's signals to the overlay and record.

        Separate from :meth:`_connect` because the provider is replaced whenever
        the selected model crosses the live/non-live boundary, and a new object
        arrives with none of the old one's connections. The recorder is *not*
        replaced with it — a gameplay session spans whatever the player does
        with the model picker mid-evening — which is why its connections are
        made here, to the new object, rather than once at construction.
        """
        self.session.statusChanged.connect(self.overlay.set_status)
        self.session.responseStarted.connect(self.overlay.start_response)
        self.session.responseDelta.connect(self.overlay.append_delta)
        self.session.responseCompleted.connect(self.overlay.end_response)
        self.session.errorOccurred.connect(self._on_session_error)

        self.session.statusChanged.connect(self.recorder.record_status)
        self.session.responseCompleted.connect(self._on_answer)
        self.session.frameSent.connect(self.recorder.record_frame)
        self.session.observerRan.connect(self.recorder.record_observer_run)
        self.session.llmCall.connect(self.recorder.record_llm_call)
        self.session.compacted.connect(self._on_compacted)

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

        if not self.settings.key_for_model(self.settings.selected_model):
            self.overlay.append_system(
                "No API key for the selected model. Open Settings (⚙) to add one."
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
            "toggle_journal": hotkeys.toggle_journal,
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
        # Stopping the session emits its last events — the live provider's
        # closing cost estimate among them — so the record is closed after it,
        # not before.
        await self.session.stop()
        self.recorder.shutdown()

    def request_quit(self) -> None:
        """Ask the main loop to shut the application down."""
        self.quit_requested.set()

    # ------------------------------------------------------- gameplay sessions

    def ensure_session(self) -> str:
        """Open a gameplay session if one is not already open.

        The two callers are the two things that count as starting to play:
        watching the screen, and asking a question. Launching and sitting idle
        deliberately records nothing.
        """
        game = self.settings.game_name.strip() or self._detected_game_label()
        session_id = self.recorder.ensure_session(
            game=game, mode="live" if self.settings.is_live else "nonlive"
        )
        self.session.session_id = session_id
        return session_id

    def new_session(self) -> None:
        """Close the current record and forget everything.

        The session boundary *is* the memory boundary: one session, one memory
        state. So this is the canonical reset — journal, conversation, observer
        baseline and transcript, in one place — and every other "start clean"
        in the app is either this or the absence of a session at all.

        Watching is deliberately untouched. Starting a new session is a decision
        about memory, not about whether Chiron is allowed to see the screen, and
        silently stopping the capture would be a surprising way to answer a
        question nobody asked.
        """
        self.recorder.close_session()
        self.journal.clear()
        self.session.reset_memory()
        self.session.session_id = ""
        self.overlay.clear_transcript()
        self.overlay.clear_journal()
        self.overlay.show_play()
        self.overlay.append_system(
            "New session. Chiron has forgotten the journal and the conversation; "
            "the previous session is kept in history."
        )
        if self.watching:
            # Watching without a record is a hole in the history, so re-open one
            # immediately rather than waiting for the next question.
            self.ensure_session()
            self.recorder.record_watch(True)

    def show_history(self) -> None:
        """Drop the session picker open, refreshed from the index."""
        self.overlay.picker.set_sessions(
            self.recorder.sessions(), total_bytes=self.recorder.total_bytes()
        )
        self.overlay.open_session_picker()

    def open_session(self, session_id: str) -> None:
        """Render one recorded session in the viewer."""
        row = self.recorder.row_for(session_id)
        events = self.recorder.read_session(session_id)
        if row is None:
            self.overlay.append_system("That session is no longer on disk.")
            return
        store = self.recorder.store_for(session_id)
        self.overlay.viewer.show_session(
            row,
            events,
            frames_directory=store.frames_directory if store is not None else None,
        )
        self.overlay.show_session_viewer()

    def _on_session_changed(self, session_id: str, title: str) -> None:
        """Keep the provider's billing label in step with the open record."""
        self.session.session_id = session_id

    def _on_session_renamed(self, session_id: str, title: str) -> None:
        """Retitle a session, re-rendering the viewer if it is showing it."""
        self.recorder.rename_session(session_id, title)
        if self._viewing(session_id):
            self.open_session(session_id)

    def _on_session_deleted(self, session_id: str) -> None:
        """Delete a session outright, leaving the viewer if it was showing it."""
        viewing = self._viewing(session_id)
        self.recorder.delete_session(session_id)
        if viewing:
            self.overlay.viewer.clear()
            self.overlay.show_play()

    def _on_thumbnails_deleted(self, session_id: str) -> None:
        """Delete a session's thumbnails, keeping its text, then re-render."""
        freed = self.recorder.delete_thumbnails(session_id)
        if self._viewing(session_id):
            # The rendered page is full of <img> tags pointing at files that
            # have just gone; re-rendering is what turns them into the dim
            # placeholders the renderer has for exactly this.
            self.open_session(session_id)
        if freed:
            from chiron.sessions.render import format_bytes

            self.overlay.append_system(f"Freed {format_bytes(freed)} of thumbnails.")

    def _viewing(self, session_id: str) -> bool:
        """Whether the viewer is currently showing `session_id`."""
        return (
            self.overlay.current_view == SESSION_VIEW
            and self.overlay.viewer.session_id == session_id
        )

    def _detected_game_label(self) -> str:
        """The focused window's short name, or empty."""
        current = self.window_tracker.current
        return current.label if current is not None else ""

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
            # The game is known now, so the auto-title can name it.
            self.ensure_session()
            # The screen moved on while nobody was looking, so whatever the
            # session last saw is not a baseline to compare the next frame
            # against. The capture service resets its own; this resets the
            # non-live observer's novelty detector.
            self.session.reset_observation()

        self.capture.set_watching(watching)
        self.overlay.set_watching(watching, self.settings.hotkeys.toggle_watching)
        self.recorder.record_watch(watching)

        if watching:
            info = self.window_tracker.current
            if info is not None:
                self._record_game(info, changed=False)
            if info is not None and not self.settings.game_name.strip():
                self.session.detected_game = info.describe()
                self.journal.append(
                    f"Watching started; the player is in {info.describe()}.",
                    source="system",
                )
                self.overlay.append_system(
                    f"● Watching your screen. Looks like {info.label}."
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
        self.ensure_session()
        self.recorder.record_message("user", text)
        if self.watching:
            self.capture.request_burst("question")
        elif not self._warned_not_watching:
            self._warned_not_watching = True
            self.overlay.append_system(
                f"Chiron is not watching, so it cannot see your screen right now. "
                f"Press {self._watch_hotkey_hint()} to let it look."
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

    def _on_answer(self, text: str) -> None:
        """Record a completed answer."""
        self.recorder.record_message("assistant", text)

    def _on_compacted(self, payload: dict) -> None:
        """Record a consolidation and say so, briefly, in the transcript.

        Visible on purpose. Compaction is the moment Chiron's memory of the last
        hour changes shape, and a player who is told it happened can tell the
        difference between "it forgot" and "it summarised" when a later answer
        is thinner than they expected.
        """
        self.recorder.record_compaction(payload)
        if payload.get("mode") == "live_rotate":
            self.overlay.append_system(
                "✂ Consolidating memory into the journal and reconnecting. "
                "Chiron will blink for a moment."
            )
        else:
            before = int(payload.get("tokens_before") or 0)
            after = int(payload.get("tokens_after") or 0)
            detail = f" ({before:,} → {after:,} tokens)" if before else ""
            self.overlay.append_system(
                f"✂ Summarised the earlier conversation{detail}."
            )

    def _on_journal_entry(self, entry: JournalEntry) -> None:
        """Show a new journal entry in the drawer, and record it."""
        self.overlay.append_journal(entry)
        self.recorder.record_journal_entry(entry)

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
            entry = self.journal.append(
                f"The player switched to {info.describe()}.", source="system"
            )
            # Straight into the log rather than through the writer, so this is
            # the one entry `on_entry` never sees — and the drawer would show a
            # count one short of the footer's if it were not shown by hand.
            if entry is not None:
                self.overlay.append_journal(entry)
            self._record_game(info, changed=True)

    def _record_game(self, info: WindowInfo, *, changed: bool) -> None:
        """Note the focused application in the session record."""
        self.recorder.record_game(
            label=info.label,
            identity=":".join(str(part) for part in info.identity),
            described=info.describe(),
            changed=changed,
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
        elif name == "toggle_journal":
            if self.overlay.isVisible():
                self.overlay.toggle_journal()
            else:
                # Asking for the journal while the panel is hidden is asking to
                # see it. Toggling here would show the panel and, if the drawer
                # was already open, close the one thing that was wanted.
                self.overlay.show_and_focus()
                self.overlay.set_journal_open(True)

    def _refresh_footer(self) -> None:
        """Update the overlay's small status line."""
        if not self.watching:
            shutter = "not watching"
        else:
            mode = self.capture.scheduler.mode(time.time())
            reason = self.capture.scheduler.burst_reason
            shutter = f"{mode} ({reason})" if reason else mode
        # Observer runs only exist in non-live mode, and there they are the
        # number that actually explains the bill.
        observed = getattr(self.session, "observer_runs", None)
        looks = f"  ·  looks: {observed}" if observed is not None else ""
        self.overlay.set_footer(
            f"shutter: {shutter}  ·  frames sent: {self.session.frames_sent}"
            f"{looks}  ·  journal: {len(self.journal)}"
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
        # Which providers exist depends on which keys are set, so the model list
        # is only knowable once the current settings are in hand — here, not at
        # build time. It arrives asynchronously; the form works meanwhile.
        self.settings_window.refresh_catalogue()
        self.settings_window.show()
        self.settings_window.raise_()
        self.settings_window.activateWindow()

    def apply_settings(self, new_settings: Settings) -> None:
        """Persist and apply edited settings, restarting the session if needed.

        Args:
            new_settings (Settings): The configuration to adopt.
        """
        previous = self.settings
        # Placement is owned by the overlay, not the settings form — and so is
        # the drawer, which is a thing you open mid-fight rather than a
        # preference you set. Both are read back off the live panel here, or
        # saving settings would quietly close it.
        self._remember_geometry()
        new_settings.overlay.position_x = previous.overlay.position_x
        new_settings.overlay.position_y = previous.overlay.position_y
        new_settings.overlay.journal_open = previous.overlay.journal_open
        new_settings.overlay.journal_width = previous.overlay.journal_width
        self.settings = new_settings

        try:
            save_settings(new_settings, self.settings_path)
        except OSError as error:
            self.overlay.append_system(f"⚠ Could not save settings: {error}")

        self.capture.apply_settings(new_settings.effective_capture())
        self.overlay.apply_settings(new_settings.overlay)
        self.journal.max_entries = new_settings.journal.max_entries
        self.fold_timer.setInterval(
            int(new_settings.journal.fold_interval_seconds * 1000)
        )
        self.hotkeys.set_bindings(self._hotkey_bindings())
        self.overlay.set_watching(self.watching, new_settings.hotkeys.toggle_watching)
        self.overlay.set_hide_hint(new_settings.hotkeys.toggle_overlay)
        self.session.apply_settings(new_settings)
        if previous.effective_frame_width() != new_settings.effective_frame_width():
            # New thumbnail statistics; what the detector learned no longer
            # describes the frames it is about to be given.
            self.session.reset_observation()

        # Without this the cost data is uninterpretable later: an evening whose
        # per-call price triples halfway through is a mystery unless the record
        # says the model changed. Field names only — the values include keys.
        self.recorder.record_settings_changed(
            _changed_fields(previous, new_settings),
            "live" if new_settings.is_live else "nonlive",
        )

        swap = previous.requires_provider_swap(new_settings)
        strategy_changed = (
            swap or previous.journal.strategy != new_settings.journal.strategy
        )
        if swap or previous.requires_session_restart(new_settings):
            asyncio.ensure_future(self._restart_session(strategy_changed, swap))
        self.overlay.append_system("Settings saved.")

    async def _restart_session(
        self, rebuild_writer: bool, swap_provider: bool = False
    ) -> None:
        """Restart with the new configuration, keeping the journal.

        Args:
            rebuild_writer (bool): True when the journal strategy changed — or
                when the mode did, since the two modes journal differently —
                which means the old writer has to be stopped and replaced.
            swap_provider (bool): True when the selection crossed the
                live/non-live boundary, so the session object itself is
                replaced rather than reopened.
        """
        await self.session.stop()
        if rebuild_writer:
            await self.writer.stop()
            self.writer = self._build_writer()
        if swap_provider:
            detected = self.session.detected_game
            session_id = self.session.session_id
            self.session = build_session_provider(
                self.settings, self.journal, self.writer, self
            )
            # Runtime state, not settings, so nothing reloads either — but the
            # game is still the game (re-detecting it would need another window
            # switch first), and the evening is still the same evening, which is
            # the whole reason the record outlives the provider.
            self.session.detected_game = detected
            self.session.session_id = session_id
            self._connect_session()
            self.overlay.append_system(
                "Live mode." if self.settings.is_live else "Non-live mode."
            )
        elif rebuild_writer:
            self.session.writer = self.writer
        # Restart only if the screen is still being watched. Editing settings
        # must never be a back door into watching.
        if self.watching:
            self.session.reset_observation()
            self.session.start()

    def _remember_geometry(self) -> None:
        """Record the overlay's placement and drawer state into settings.

        `current_geometry()` reports the width *without* the journal column, so
        the two are independent: an evening reopens the size it was and with the
        drawer the way it was left.
        """
        x, y, width, height = self.overlay.current_geometry()
        self.settings.overlay.position_x = x
        self.settings.overlay.position_y = y
        self.settings.overlay.width = width
        self.settings.overlay.height = height
        self.settings.overlay.journal_open = self.overlay.journal_open


def describe_untouched_sessions(root: Path | None = None) -> list[str]:
    """Lines for the ``--fresh-install`` plan saying play history is safe.

    Args:
        root (Path | None): Sessions directory. Defaults to the real one.

    Returns:
        list[str]: One "keep" line when sessions exist, empty otherwise — there
            is no reassurance to give about a history nobody has.
    """
    from chiron.sessions.render import format_bytes
    from chiron.sessions.store import SessionIndex, sessions_root

    directory = root if root is not None else sessions_root()
    if not directory.is_dir():
        return []
    index = SessionIndex(directory)
    rows = index.rows()
    if not rows:
        return []
    noun = "session" if len(rows) == 1 else "sessions"
    return [
        f"keep    {directory}{os.sep} "
        f"({len(rows)} recorded {noun}, {format_bytes(index.total_bytes())}; "
        "not Chiron's configuration)"
    ]


def _changed_fields(previous: Settings, current: Settings) -> list[str]:
    """Top-level setting names that differ between two configurations.

    Nested sections are reported as ``capture.frame_width`` rather than as
    ``capture``, since "capture changed" is not a sentence that explains a cost
    curve. Credentials are named but never valued.
    """
    changed: list[str] = []
    before, after = previous.model_dump(), current.model_dump()
    for name, old in before.items():
        new = after.get(name)
        if isinstance(old, dict) and isinstance(new, dict):
            changed.extend(
                f"{name}.{key}" for key, value in old.items() if new.get(key) != value
            )
        elif old != new:
            changed.append(name)
    return changed


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
            "Delete Chiron's saved configuration, including any saved API key, "
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
        print(f"Nothing to remove: no configuration at {settings_path}.", file=out)
        return 0

    print("This will:", file=out)
    for line in plan.describe():
        print(f"  {line}", file=out)

    # Sessions live in the data directory precisely so this command cannot reach
    # them, and saying so is the difference between a guarantee and a hope:
    # wiping an API key should not bundle in wiping a season of play.
    for line in describe_untouched_sessions():
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
