"""Application wiring for Chiron's permanent Observer and Responder agents."""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import signal
import sys
import time
from collections import deque
from dataclasses import dataclass
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
from chiron.journal.compaction import JournalCompactor
from chiron.journal.log import JournalEntry, JournalLog
from chiron.journal.service import JournalService
from chiron.observer.factory import build_observer
from chiron.responder.conversation import ResponderConversation
from chiron.responder.session import ResponderSessionManager
from chiron.session import ObserverStatus
from chiron.sessions.recorder import SessionRecorder
from chiron.ui.hotkeys import GlobalHotkeyManager
from chiron.ui.overlay import SESSION_VIEW, OverlayWindow
from chiron.ui.settings_window import SettingsWindow

logger = logging.getLogger(__name__)

_FOOTER_INTERVAL_MS = 1000


@dataclass(frozen=True)
class _PendingQuestion:
    text: str
    immediate: bool
    flush_observer: bool


class ChironApp(QObject):
    """Own both agents and every cross-component connection."""

    def __init__(
        self,
        settings: Settings,
        settings_path: Path,
        parent: QObject | None = None,
        *,
        sessions_root: Path | None = None,
    ) -> None:
        super().__init__(parent)
        self.settings = settings
        self.settings_path = settings_path
        self.quit_requested = asyncio.Event()
        self._watch_requested = False
        self._warned_not_watching = False
        self._latest_frame: Frame | None = None
        self._questions: deque[_PendingQuestion] = deque()
        self._waiting_immediate: tuple[int, _PendingQuestion] | None = None
        self._flushing_question: tuple[_PendingQuestion, Frame | None] | None = None
        self._blocked_flush_question: tuple[_PendingQuestion, Frame | None] | None = (
            None
        )
        self._flush_task: asyncio.Task[None] | None = None
        self._question_token = 0
        self._observer_drain_action: str | None = None

        self.recorder = SessionRecorder(sessions_root, self)
        self.journal = JournalLog()
        self.journal_service = JournalService(self.journal, self._on_journal_entry)
        self.journal_compactor = JournalCompactor(
            settings, self.journal_service, parent=self
        )
        self.conversation = ResponderConversation()
        self.observer = build_observer(
            settings, self.journal_service, self.journal_compactor, parent=self
        )
        self.responder = ResponderSessionManager(
            settings,
            self.journal_service.reader(),
            self.conversation,
            self.journal_compactor,
            parent=self,
        )
        self.capture = CaptureService(settings.effective_capture(), self)
        self.overlay = OverlayWindow(settings.overlay)
        self.settings_window: SettingsWindow | None = None
        self.hotkeys = GlobalHotkeyManager(self)
        self.window_tracker = ActiveWindowTracker(self._chiron_window_ids, self)

        self.footer_timer = QTimer(self)
        self.footer_timer.setInterval(_FOOTER_INTERVAL_MS)
        self.footer_timer.timeout.connect(self._refresh_footer)
        self._connect()

    # --------------------------------------------------------------- wiring

    def _connect(self) -> None:
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
        self.overlay.observerRetryRequested.connect(self.retry_observer_batch)
        self.overlay.observerDiscardRequested.connect(self.discard_observer_batch)

        self.recorder.sessionChanged.connect(self.overlay.set_session)
        self.recorder.sessionChanged.connect(self._on_session_changed)
        self.recorder.costChanged.connect(self.overlay.set_cost)
        self.journal_compactor.llmCall.connect(self.recorder.record_llm_call)
        self.journal_compactor.compacted.connect(self._on_compacted)

        self.capture.frameCaptured.connect(self._on_scheduled_frame)
        self.capture.immediateFrameCaptured.connect(self._on_immediate_frame)
        self.capture.errorOccurred.connect(self._on_capture_error)

        self._connect_observer()
        self._connect_responder()
        self.window_tracker.windowChanged.connect(self._on_active_window)
        self.hotkeys.activated.connect(self._on_hotkey)
        self.hotkeys.failed.connect(
            lambda message: self.overlay.append_system(f"⚠ {message}")
        )

    def _connect_observer(self) -> None:
        self.observer.statusChanged.connect(self._on_observer_status)
        self.observer.errorOccurred.connect(self._on_observer_error)
        self.observer.errorOccurred.connect(
            lambda detail: self.recorder.record_status(
                "error", detail, agent_id="observer"
            )
        )
        self.observer.statusChanged.connect(
            lambda state, detail: self.recorder.record_status(
                state, detail, agent_id="observer"
            )
        )
        self.observer.frameSent.connect(
            lambda frame, reason: self.recorder.record_frame(
                frame, reason, agent_id="observer"
            )
        )
        self.observer.llmCall.connect(self.recorder.record_llm_call)
        self.observer.compacted.connect(self._on_compacted)
        if hasattr(self.observer, "observerRan"):
            self.observer.observerRan.connect(self.recorder.record_observer_run)

    def _connect_responder(self) -> None:
        self.responder.responseStarted.connect(self.overlay.start_response)
        self.responder.responseDelta.connect(self.overlay.append_delta)
        self.responder.responseCompleted.connect(self.overlay.end_response)
        self.responder.responseCompleted.connect(self._on_answer)
        self.responder.errorOccurred.connect(self._on_responder_error)
        self.responder.errorOccurred.connect(
            lambda detail: self.recorder.record_status(
                "error", detail, agent_id="responder"
            )
        )
        self.responder.llmCall.connect(self.recorder.record_llm_call)
        self.responder.compacted.connect(self._on_compacted)
        self.responder.agentTrace.connect(self.recorder.record_agent_trace)

    # ------------------------------------------------------------ lifecycle

    def start(self) -> None:
        self.overlay.set_status("idle", "Observer is off")
        self.overlay.set_watching(False, self.settings.hotkeys.toggle_watching)
        self.overlay.set_hide_hint(self.settings.hotkeys.toggle_overlay)
        if not self.settings.overlay.start_hidden:
            self.overlay.show_and_focus()
        self.overlay.append_system(
            f"Chiron is not watching yet. Press {self._watch_hotkey_hint()} to "
            f"start, {self.settings.hotkeys.toggle_overlay} to show or hide this "
            "panel. Run your game borderless-windowed."
        )
        self.hotkeys.set_bindings(self._hotkey_bindings())
        self.hotkeys.start()
        self.window_tracker.start()
        self.capture.start()
        self.footer_timer.start()
        if not self.settings.key_for_model(self.settings.observer_model):
            self.overlay.append_system(
                "No API key for the selected Observer model. Open Settings."
            )
        if not self.settings.key_for_model(self.settings.responder_model):
            self.overlay.append_system(
                "No API key for the selected Responder model. Open Settings."
            )
        if self.settings.capture.watch_on_launch:
            self.set_watching(True)

    async def shutdown(self) -> None:
        self.footer_timer.stop()
        self.hotkeys.stop()
        self.window_tracker.stop()
        self.capture.set_watching(False)
        self.capture.stop()
        self._remember_geometry()
        try:
            save_settings(self.settings, self.settings_path)
        except OSError as error:
            logger.warning("Could not save settings on exit: %s", error)
        await asyncio.gather(
            self.observer.stop(timeout=90.0),
            self.responder.stop(),
            return_exceptions=True,
        )
        self.recorder.shutdown()

    def request_quit(self) -> None:
        self.quit_requested.set()

    # ------------------------------------------------------- gameplay session

    def ensure_session(self) -> str:
        game = self.settings.game_name.strip() or self._detected_game_label()
        session_id = self.recorder.ensure_session(game=game, mode="dual_agent")
        self.observer.session_id = session_id
        self.responder.session_id = session_id
        self.journal_compactor.session_id = session_id
        return session_id

    def new_session(self) -> None:
        self._clear_flush_question()
        if getattr(self.observer, "pending_frames", 0):
            self._observer_drain_action = "new_session"
            self.capture.set_watching(False)
            self._invalidate_frames()
            self.overlay.append_system(
                "Draining captured Observer frames before starting the new session."
            )
            begin_drain = getattr(self.observer, "begin_drain", None)
            if begin_drain is not None:
                begin_drain()
            self._schedule(self._drain_then_new_session())
            return
        self._finish_new_session()

    async def _drain_then_new_session(self) -> None:
        await self.observer.stop(timeout=90.0)
        if getattr(self.observer, "pending_frames", 0):
            self.overlay.append_system(
                "⚠ New Session is waiting: fix or explicitly discard the retained "
                "Observer batch first."
            )
            return
        self._observer_drain_action = None
        self._finish_new_session()

    def _finish_new_session(self) -> None:
        self._observer_drain_action = None
        self.recorder.close_session()
        if self.watch_requested:
            self.capture.set_watching(False)
        self.journal_service.clear()
        self.conversation.clear()
        self.responder.reset_memory()
        self.observer.reset_memory()
        self.observer.session_id = ""
        self.responder.session_id = ""
        self.journal_compactor.session_id = ""
        self._invalidate_frames()
        self._questions.clear()
        self._waiting_immediate = None
        self._clear_flush_question()
        self.overlay.clear_transcript()
        self.overlay.clear_journal()
        self.overlay.show_play()
        self.overlay.append_system(
            "New session. Chiron has forgotten the journal and conversation; "
            "the previous session remains in history."
        )
        if self.watch_requested:
            self.ensure_session()
            self.recorder.record_watch(True)
            self.observer.start()

    # --------------------------------------------------------------- history

    def show_history(self) -> None:
        self.overlay.picker.set_sessions(
            self.recorder.sessions(), total_bytes=self.recorder.total_bytes()
        )
        self.overlay.open_session_picker()

    def open_session(self, session_id: str) -> None:
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

    def _on_session_changed(self, session_id: str, _title: str) -> None:
        self.observer.session_id = session_id
        self.responder.session_id = session_id
        self.journal_compactor.session_id = session_id

    def _on_session_renamed(self, session_id: str, title: str) -> None:
        self.recorder.rename_session(session_id, title)
        if self._viewing(session_id):
            self.open_session(session_id)

    def _on_session_deleted(self, session_id: str) -> None:
        viewing = self._viewing(session_id)
        self.recorder.delete_session(session_id)
        if viewing:
            self.overlay.viewer.clear()
            self.overlay.show_play()

    def _on_thumbnails_deleted(self, session_id: str) -> None:
        freed = self.recorder.delete_thumbnails(session_id)
        if self._viewing(session_id):
            self.open_session(session_id)
        if freed:
            from chiron.sessions.render import format_bytes

            self.overlay.append_system(f"Freed {format_bytes(freed)} of thumbnails.")

    def _viewing(self, session_id: str) -> bool:
        return (
            self.overlay.current_view == SESSION_VIEW
            and self.overlay.viewer.session_id == session_id
        )

    # -------------------------------------------------------------- watching

    @property
    def watch_requested(self) -> bool:
        return self._watch_requested

    @property
    def watching(self) -> bool:
        """User-requested Watch state, independent of temporary Observer outage."""
        return self._watch_requested

    @property
    def capture_active(self) -> bool:
        return self.capture.is_watching

    def set_watching(self, watching: bool) -> None:
        if watching == self._watch_requested:
            return
        if watching:
            self.window_tracker.poll()
            self.ensure_session()
            self._watch_requested = True
            self._warned_not_watching = False
            self._invalidate_frames()
            self.overlay.set_watching(True, self.settings.hotkeys.toggle_watching)
            self.recorder.record_watch(True)
            info = self.window_tracker.current
            if info is not None:
                self._record_game(info, changed=False)
                if not self.settings.game_name.strip():
                    described = info.describe()
                    self.observer.detected_game = described
                    self.responder.detected_game = described
                    self.journal_service.record(
                        f"Watching started; the player is in {described}.",
                        source="system",
                    )
            self.overlay.append_system(
                "● Watch requested. Connecting Chiron-Observer before capture starts."
            )
            if self.observer.accepting_frames:
                self._activate_capture()
            else:
                self.observer.start()
            return

        # Privacy boundary: stop and invalidate before closing the socket.
        self._watch_requested = False
        self.capture.set_watching(False)
        self._invalidate_frames()
        self._flush_waiting_immediate()
        self.overlay.set_watching(False, self.settings.hotkeys.toggle_watching)
        self.recorder.record_watch(False)
        self.overlay.append_system(
            "○ Stopped capturing. Chiron-Responder can still answer from the "
            "existing journal and conversation; already buffered frames may finish "
            "their final review."
        )
        begin_drain = getattr(self.observer, "begin_drain", None)
        if begin_drain is not None:
            begin_drain()
        self._schedule(self.observer.stop())

    def toggle_watching(self) -> None:
        self.set_watching(not self.watch_requested)

    def _activate_capture(self) -> None:
        if not self.watch_requested or not self.observer.accepting_frames:
            return
        if self.capture_active:
            return
        self._invalidate_frames()
        self.capture.set_watching(True)
        self.overlay.append_system("● Observer connected; capture is active.")

    def _on_observer_status(self, status: str, detail: str) -> None:
        self.overlay.set_status(status, f"Observer: {detail}" if detail else "Observer")
        pending = getattr(self.observer, "pending_frames", 0)
        self.overlay.set_observer_recovery(
            pending if status in {"error", "incompatible"} else 0
        )
        if self.observer.accepting_frames and self.watch_requested:
            self._activate_capture()
            return
        if not self.observer.accepting_frames:
            self.capture.set_watching(False)
            self._invalidate_frames()
            self._flush_waiting_immediate()

    # ------------------------------------------------------------- questions

    def _on_prompt(self, text: str) -> None:
        self.overlay.append_user(text)
        self.ensure_session()
        self.recorder.record_message("user", text, agent_id="responder")
        immediate = self.settings.capture.question_frame_policy == "immediate"
        flush_observer = (
            self.settings.capture.question_answer_policy == "flush_observer"
        )
        self._questions.append(_PendingQuestion(text, immediate, flush_observer))
        if not self.capture_active and not self._warned_not_watching:
            self._warned_not_watching = True
            self.overlay.append_system(
                "Chiron-Observer is unavailable, so this answer uses stale journal "
                "and conversation context without a current screenshot."
            )
        self._resolve_questions()

    def _resolve_questions(self) -> None:
        if (
            self._waiting_immediate is not None
            or self._flushing_question is not None
            or self._blocked_flush_question is not None
        ):
            return
        while self._questions:
            question = self._questions.popleft()
            if question.immediate and self.capture_active:
                self._question_token += 1
                token = self._question_token
                self._waiting_immediate = (token, question)
                if self.capture.request_immediate(token):
                    return
                self._waiting_immediate = None
            frame = self._latest_frame if self.capture_active else None
            if self._begin_flush_or_submit(question, frame):
                return

    def _submit_question(self, question: _PendingQuestion, frame: Frame | None) -> None:
        if frame is not None:
            self.recorder.record_frame(
                frame,
                "immediate" if question.immediate else "question",
                agent_id="responder",
            )
        self.responder.ask(question.text, frame, self._observer_snapshot())

    def _begin_flush_or_submit(
        self, question: _PendingQuestion, frame: Frame | None
    ) -> bool:
        flush = getattr(self.observer, "flush_and_wait", None)
        if (
            not question.flush_observer
            or not self.capture_active
            or getattr(self.observer, "mode", "live") != "nonlive"
            or flush is None
        ):
            self._submit_question(question, frame)
            return False
        self._flushing_question = (question, frame)
        self.overlay.append_system("Reviewing recent gameplay before answering.")
        task = self._schedule(self._flush_observer_then_submit(flush, question, frame))
        if task is None:
            self._flushing_question = None
            self._submit_question(question, frame)
            return False
        self._flush_task = task
        return True

    async def _flush_observer_then_submit(
        self,
        flush: Callable[[], object],
        question: _PendingQuestion,
        frame: Frame | None,
    ) -> None:
        try:
            completed = await flush()
        except asyncio.CancelledError:
            raise
        except Exception as error:  # the normal retained-batch path returns false
            logger.warning("Observer question flush failed: %s", error)
            completed = False
        finally:
            if self._flushing_question == (question, frame):
                self._flushing_question = None
                self._flush_task = None
        if not completed:
            self._blocked_flush_question = (question, frame)
            self.overlay.append_system(
                "⚠ Observer review did not finish; this question is waiting for "
                "Retry review or Discard."
            )
            return
        self._submit_question(question, frame)
        self._resolve_questions()

    def _clear_flush_question(self) -> None:
        if self._flush_task is not None:
            self._flush_task.cancel()
        self._flushing_question = None
        self._blocked_flush_question = None
        self._flush_task = None

    def _resume_blocked_flush_question(self, *, discarded: bool = False) -> None:
        pending, self._blocked_flush_question = self._blocked_flush_question, None
        if pending is None:
            return
        if discarded:
            self.overlay.append_system(
                "Observer frames were discarded; answering with the existing journal."
            )
        self._submit_question(*pending)
        self._resolve_questions()

    def _on_immediate_frame(self, frame: Frame, token: object) -> None:
        waiting = self._waiting_immediate
        if waiting is None or token != waiting[0]:
            return
        self._waiting_immediate = None
        question = waiting[1]
        if self.capture_active and self.observer.accepting_frames:
            self.observer.observe(frame, "immediate")
            self._begin_flush_or_submit(question, frame)
        else:
            self._submit_question(question, None)
        self._resolve_questions()

    def _flush_waiting_immediate(self) -> None:
        waiting, self._waiting_immediate = self._waiting_immediate, None
        self.capture.clear_pending()
        if waiting is not None:
            self._submit_question(waiting[1], None)
        self._resolve_questions()

    def _observer_snapshot(self) -> ObserverStatus:
        state = self.observer.status
        if not self.watch_requested:
            state = "watch_off"
        return ObserverStatus(
            state=state,
            watch_requested=self.watch_requested,
            mode=getattr(self.observer, "mode", "live"),
            accepting_frames=self.observer.accepting_frames,
            last_sampled_at=getattr(self.observer, "last_sampled_at", None),
            last_observed_at=self.observer.last_observed_at,
            pending_frames=getattr(self.observer, "pending_frames", 0),
            next_process_at=getattr(self.observer, "next_process_at", None),
            detail=self.observer.status_detail,
        )

    # --------------------------------------------------------------- frames

    def _on_scheduled_frame(self, frame: Frame) -> None:
        if not self.capture_active or not self.observer.accepting_frames:
            return
        self._latest_frame = frame
        self.observer.observe(frame, "scheduled")

    def _invalidate_frames(self) -> None:
        self._latest_frame = None
        self.capture.clear_pending()

    # -------------------------------------------------------------- handlers

    def _on_capture_error(self, message: str) -> None:
        self.capture.set_watching(False)
        self._invalidate_frames()
        self._flush_waiting_immediate()
        self.overlay.append_system(f"⚠ {message}")

    def _on_observer_error(self, message: str) -> None:
        self.overlay.append_system(f"⚠ Observer: {message}")

    def _on_responder_error(self, message: str) -> None:
        self.overlay.append_system(f"⚠ Responder: {message}")

    def _on_answer(self, text: str) -> None:
        self.recorder.record_message("assistant", text, agent_id="responder")

    def _on_compacted(self, payload: dict) -> None:
        self.recorder.record_compaction(payload)
        if payload.get("agent_id") == "observer":
            self.overlay.append_system(
                "✂ Observer rotated its Live context and will reseed from journal "
                "memory."
            )
        elif payload.get("agent_id") == "journal":
            before = int(payload.get("tokens_before") or 0)
            after = int(payload.get("tokens_after") or 0)
            self.overlay.append_system(
                "✂ Journal compacted older entries into durable model memory "
                f"({before:,} → {after:,} tokens)."
            )
        else:
            before = int(payload.get("tokens_before") or 0)
            after = int(payload.get("tokens_after") or 0)
            detail = f" ({before:,} → {after:,} tokens)" if before else ""
            self.overlay.append_system(
                f"✂ Responder summarised earlier conversation{detail}."
            )

    def _on_journal_entry(self, entry: JournalEntry) -> None:
        self.overlay.append_journal(entry)
        self.recorder.record_journal_entry(entry)

    # ---------------------------------------------------------- active window

    def _detected_game_label(self) -> str:
        current = self.window_tracker.current
        return current.label if current is not None else ""

    def _chiron_window_ids(self) -> set[int]:
        ids = {int(self.overlay.winId())}
        if self.settings_window is not None:
            ids.add(int(self.settings_window.winId()))
        return ids

    def _on_active_window(self, info: WindowInfo) -> None:
        if not self.settings.game_name.strip():
            described = info.describe()
            self.observer.detected_game = described
            self.responder.detected_game = described
        if self.watch_requested:
            self.journal_service.record(
                f"The player switched to {info.describe()}.", source="system"
            )
            self._record_game(info, changed=True)

    def _record_game(self, info: WindowInfo, *, changed: bool) -> None:
        self.recorder.record_game(
            label=info.label,
            identity=":".join(str(part) for part in info.identity),
            described=info.describe(),
            changed=changed,
        )

    # --------------------------------------------------------------- hotkeys

    def _hotkey_bindings(self) -> dict[str, str]:
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
        hotkeys = self.settings.hotkeys
        return (
            hotkeys.toggle_watching.strip()
            or hotkeys.start_watching.strip()
            or ("the eye button above")
        )

    def _on_hotkey(self, name: str) -> None:
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
                self.overlay.show_and_focus()
                self.overlay.set_journal_open(True)

    # --------------------------------------------------------------- footer

    def _refresh_footer(self) -> None:
        if not self.watch_requested:
            shutter = "not watching"
        elif not self.capture_active:
            shutter = f"paused ({self.observer.status})"
        else:
            shutter = f"fixed {self.settings.capture.interval_seconds:g}s"
        review = ""
        pending = getattr(self.observer, "pending_frames", 0)
        next_process = getattr(self.observer, "next_process_at", None)
        if pending:
            review += f"  ·  {pending} frames pending"
        if next_process is not None:
            remaining = max(0, int(next_process - time.time()))
            minutes, seconds = divmod(remaining, 60)
            countdown = f"{minutes}m {seconds:02d}s" if minutes else f"{seconds}s"
            review += f"  ·  next review in {countdown}"
        self.overlay.set_footer(
            f"capture: {shutter}  ·  Observer frames: {self.observer.frames_sent}"
            f"{review}  ·  journal: {len(self.journal)}"
        )

    # -------------------------------------------------------------- settings

    def show_settings(self) -> None:
        if self.settings_window is None:
            self.settings_window = SettingsWindow(self.settings)
            self.settings_window.settingsSaved.connect(self.apply_settings)
            self.settings_window.appearanceChanged.connect(self.overlay.apply_settings)
            self.settings_window.quitRequested.connect(self.request_quit)
        self.settings_window.set_hotkey_backend(self.hotkeys.backend_name)
        current = self.window_tracker.current
        self.settings_window.set_detected_game(current.label if current else "")
        self.settings_window.load(self.settings)
        self.settings_window.refresh_catalogue()
        self.settings_window.show()
        self.settings_window.raise_()
        self.settings_window.activateWindow()

    def apply_settings(self, new_settings: Settings) -> None:
        try:
            new_settings = Settings.model_validate(new_settings.model_dump())
        except ValueError as error:
            self.overlay.append_system(f"⚠ Settings were not applied: {error}")
            return
        missing = [
            role
            for role, model_id in (
                ("Observer", new_settings.observer_model),
                ("Responder", new_settings.responder_model),
            )
            if not new_settings.key_for_model(model_id)
        ]
        if missing:
            self.overlay.append_system(
                "⚠ Settings were not applied: missing API key for "
                + " and ".join(missing)
                + "."
            )
            return
        previous = self.settings
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

        width_changed = previous.capture.frame_width != new_settings.capture.frame_width
        self.capture.apply_settings(new_settings.effective_capture())
        if width_changed:
            self._invalidate_frames()
        self.overlay.apply_settings(new_settings.overlay)
        self.hotkeys.set_bindings(self._hotkey_bindings())
        self.overlay.set_watching(
            self.watch_requested, new_settings.hotkeys.toggle_watching
        )
        self.overlay.set_hide_hint(new_settings.hotkeys.toggle_overlay)
        replace_observer = previous.observer_model != new_settings.observer_model
        if not replace_observer:
            self.observer.apply_settings(new_settings)
        self.responder.apply_settings(new_settings)
        self.journal_compactor.apply_settings(new_settings)
        self.recorder.record_settings_changed(
            _changed_fields(previous, new_settings), "dual_agent"
        )

        if previous.requires_observer_reconnect(new_settings):
            if replace_observer:
                self._schedule(self._replace_observer())
            else:
                self._schedule(self._reconnect_observer())
        if previous.requires_responder_rebuild(new_settings):
            self._schedule(self._rebuild_responder())
        self.overlay.append_system("Settings saved.")

    async def _reconnect_observer(self) -> None:
        self.capture.set_watching(False)
        self._invalidate_frames()
        self._flush_waiting_immediate()
        await self.observer.stop()
        if self.watch_requested:
            self.observer.start()

    async def _replace_observer(self) -> None:
        """Drain the old transport/model before constructing its replacement."""
        self.capture.set_watching(False)
        self._invalidate_frames()
        self._flush_waiting_immediate()
        old = self.observer
        await old.stop(timeout=90.0)
        if getattr(old, "pending_frames", 0):
            self._observer_drain_action = "replace_observer"
            self.overlay.append_system(
                "⚠ Observer change is waiting: the old captured batch must be "
                "retried or explicitly discarded first."
            )
            return
        self._install_observer_replacement(old)

    def _install_observer_replacement(self, old) -> None:
        """Install the already-selected Observer after its predecessor drained."""
        self._observer_drain_action = None
        detected, session_id = old.detected_game, old.session_id
        self.observer = build_observer(
            self.settings,
            self.journal_service,
            self.journal_compactor,
            parent=self,
        )
        self.observer.detected_game = detected
        self.observer.session_id = session_id
        self._connect_observer()
        if self.watch_requested:
            self.observer.start()

    def retry_observer_batch(self) -> None:
        self._schedule(self._retry_observer_batch())

    async def _retry_observer_batch(self) -> None:
        retry = getattr(self.observer, "retry_retained_and_wait", None)
        if retry is None:
            return
        try:
            completed = await retry(self.settings, timeout=90.0)
        except Exception as error:  # invalid retry transport remains visible
            self.overlay.append_system(f"⚠ Observer batch was not retried: {error}")
            return
        if not completed:
            return
        self.overlay.set_observer_recovery(0)
        self._resume_blocked_flush_question()
        await self._finish_observer_drain_action()

    def discard_observer_batch(self) -> None:
        discard = getattr(self.observer, "discard_retained", None)
        if discard is None:
            return
        count = discard()
        self.overlay.set_observer_recovery(0)
        self.overlay.append_system(
            f"Discarded {count} captured Observer frames; they cannot be recovered."
        )
        self.recorder.record_status(
            "discarded",
            f"{count} captured frames were not observed",
            agent_id="observer",
        )
        self._resume_blocked_flush_question(discarded=True)
        self._schedule(self._finish_observer_drain_action())

    async def _finish_observer_drain_action(self) -> None:
        action, self._observer_drain_action = self._observer_drain_action, None
        if action == "new_session":
            self._finish_new_session()
        elif action == "replace_observer":
            self._install_observer_replacement(self.observer)
        elif self.watch_requested and not self.observer.accepting_frames:
            self.observer.start()

    async def _rebuild_responder(self) -> None:
        old = self.responder
        await old.stop()
        detected, session_id = old.detected_game, old.session_id
        self.responder = ResponderSessionManager(
            self.settings,
            self.journal_service.reader(),
            self.conversation,
            parent=self,
        )
        self.responder.detected_game = detected
        self.responder.session_id = session_id
        self._connect_responder()

    def _remember_geometry(self) -> None:
        x, y, width, height = self.overlay.current_geometry()
        self.settings.overlay.position_x = x
        self.settings.overlay.position_y = y
        self.settings.overlay.width = width
        self.settings.overlay.height = height
        self.settings.overlay.journal_open = self.overlay.journal_open

    @staticmethod
    def _schedule(coroutine) -> asyncio.Task | None:
        """Schedule app work only when qasync is actually running."""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            coroutine.close()
            return None
        return loop.create_task(coroutine)


def describe_untouched_sessions(root: Path | None = None) -> list[str]:
    """Lines for the ``--fresh-install`` plan saying play history is safe."""
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
        "--yes", "-y", action="store_true", help="Confirm --fresh-install."
    )
    return parser.parse_args(argv)


def fresh_install(
    settings_path: Path,
    *,
    assume_yes: bool = False,
    stream: TextIO | None = None,
    confirm: Callable[[str], str] | None = None,
) -> int:
    out = stream or sys.stdout
    plan = plan_removal(settings_path)
    if plan.is_empty:
        print(f"Nothing to remove: no configuration at {settings_path}.", file=out)
        return 0
    print("This will:", file=out)
    for line in plan.describe():
        print(f"  {line}", file=out)
    for line in describe_untouched_sessions():
        print(f"  {line}", file=out)
    if plan.holds_api_key:
        print(
            "\nYour saved API key is in that file and will be gone. No backup is kept.",
            file=out,
        )
    if not assume_yes:
        ask = confirm
        if ask is None:
            if not sys.stdin.isatty():
                print(
                    "\nNot a terminal, so nothing was removed. Re-run with --yes.",
                    file=out,
                )
                return 1
            ask = input
        if ask("\nRemove it? [y/N] ").strip().lower() not in ("y", "yes"):
            print("Cancelled; nothing was removed.", file=out)
            return 1
    removed = remove_configuration(settings_path)
    for line in removed.describe():
        if line.startswith("delete"):
            print(f"Removed {line.split(maxsplit=1)[1]}", file=out)
    print("Chiron is back to a fresh install. Run `chiron` to start again.", file=out)
    return 0


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
    )
    settings_path = Path(args.settings or default_settings_path())
    if args.fresh_install:
        return fresh_install(settings_path, assume_yes=args.yes)
    app = QApplication(sys.argv)
    app.setApplicationName("Chiron")
    app.setApplicationDisplayName("Chiron")
    app.setQuitOnLastWindowClosed(False)
    loop = qasync.QEventLoop(app)
    asyncio.set_event_loop(loop)
    chiron = ChironApp(load_settings(settings_path), settings_path)
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, chiron.request_quit)
        except (NotImplementedError, RuntimeError):  # pragma: no cover
            signal.signal(sig, lambda *_: chiron.request_quit())
    app.aboutToQuit.connect(chiron.request_quit)
    with loop:
        chiron.start()
        loop.run_until_complete(chiron.quit_requested.wait())
        loop.run_until_complete(chiron.shutdown())
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
