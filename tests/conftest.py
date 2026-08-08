"""Shared fixtures.

Qt widgets need a `QApplication`, and CI has no display, so the platform is
forced to Qt's offscreen backend before PySide6 is imported anywhere.
"""

from __future__ import annotations

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


@pytest.fixture(scope="session")
def qapp():
    """A single `QApplication` shared by every widget test."""
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([])
    yield app


class ObserverSpy:
    """No-network Observer with the production signal and state surface."""

    frames_sent = 0

    def __init__(self) -> None:
        from PySide6.QtCore import QObject, Signal

        class Signals(QObject):
            statusChanged = Signal(str, str)
            errorOccurred = Signal(str)
            frameSent = Signal(object, str)
            compacted = Signal(object)
            llmCall = Signal(object)
            observerRan = Signal(object)

        self.signals = Signals()
        for name in (
            "statusChanged",
            "errorOccurred",
            "frameSent",
            "compacted",
            "llmCall",
            "observerRan",
        ):
            setattr(self, name, getattr(self.signals, name))
        self.status = "idle"
        self.status_detail = ""
        self.mode = "live"
        self.starts = 0
        self.stops = 0
        self.memory_resets = 0
        self.detected_game = ""
        self.session_id = ""
        self.last_observed_at = None
        self.last_sampled_at = None
        self.pending_frames = 0
        self.next_process_at = None
        self.frames: list[tuple[object, str]] = []

    def start(self) -> None:
        self.starts += 1
        self.status = "live"
        self.status_detail = "test Observer"
        self.statusChanged.emit(self.status, self.status_detail)

    async def stop(self) -> None:
        self.stops += 1
        self.status = "stopped"
        self.status_detail = ""
        self.statusChanged.emit(self.status, self.status_detail)

    def observe(self, frame, reason: str = "scheduled") -> None:
        self.frames.append((frame, reason))
        self.frames_sent += 1
        self.last_sampled_at = frame.captured_at
        self.last_observed_at = frame.captured_at
        self.frameSent.emit(frame, reason)

    def reset_memory(self) -> None:
        self.memory_resets += 1
        self.last_observed_at = None
        self.last_sampled_at = None

    def apply_settings(self, settings) -> None: ...

    @property
    def accepting_frames(self) -> bool:
        return self.status == "live"


class ResponderSpy:
    """No-network Responder that records FIFO submissions."""

    def __init__(self) -> None:
        from PySide6.QtCore import QObject, Signal

        class Signals(QObject):
            responseStarted = Signal()
            responseDelta = Signal(str)
            responseCompleted = Signal(str)
            errorOccurred = Signal(str)
            llmCall = Signal(object)
            compacted = Signal(object)
            agentTrace = Signal(object)

        self.signals = Signals()
        for name in (
            "responseStarted",
            "responseDelta",
            "responseCompleted",
            "errorOccurred",
            "llmCall",
            "compacted",
            "agentTrace",
        ):
            setattr(self, name, getattr(self.signals, name))
        self.session_id = ""
        self.detected_game = ""
        self.questions: list[tuple[str, object, object]] = []
        self.memory_resets = 0
        self.stops = 0

    def ask(self, text, frame, observer_status) -> None:
        self.questions.append((text, frame, observer_status))

    def answer(self, text: str) -> None:
        self.responseStarted.emit()
        self.responseDelta.emit(text)
        self.responseCompleted.emit(text)

    def reset_memory(self) -> None:
        self.memory_resets += 1

    def apply_settings(self, settings) -> None: ...

    async def stop(self) -> None:
        self.stops += 1


@pytest.fixture
def app(qapp, tmp_path, monkeypatch):
    """A fully wired `ChironApp`, not started, with a stubbed live session.

    Sessions are recorded under `tmp_path`, never the real data directory: a
    test suite that writes into the user's play history would be a worse bug
    than anything it could catch.
    """
    from chiron.app import ChironApp
    from chiron.config.settings import Settings

    for name in ("GEMINI_API_KEY", "GOOGLE_API_KEY", "OPENROUTER_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    instance = ChironApp(
        Settings(api_key="test-key"),
        tmp_path / "settings.json",
        sessions_root=tmp_path / "sessions",
    )
    instance.observer = ObserverSpy()
    instance.responder = ResponderSpy()
    instance._connect_observer()
    instance._connect_responder()
    return instance
