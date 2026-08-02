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


class SessionSpy:
    """Stands in for a session provider, so no socket and no model is reached."""

    frames_sent = 0

    def __init__(self) -> None:
        self.status = "idle"
        self.starts = 0
        self.stops = 0
        self.resets = 0
        self.memory_resets = 0
        self.detected_game = ""
        self.session_id = ""
        self.texts: list[str] = []

    def start(self) -> None:
        self.starts += 1
        self.status = "live"

    async def stop(self) -> None:
        self.stops += 1
        self.status = "stopped"

    def send_text(self, text: str) -> None:
        self.texts.append(text)

    def send_frame(self, frame) -> None: ...

    def fold_journal(self, *, force: bool = False) -> int:
        return 0

    def reset_observation(self) -> None:
        self.resets += 1

    def reset_memory(self) -> None:
        self.memory_resets += 1

    def apply_settings(self, settings) -> None: ...


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
        Settings(), tmp_path / "settings.json", sessions_root=tmp_path / "sessions"
    )
    instance.session = SessionSpy()
    return instance
