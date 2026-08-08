"""Observer-only transport selection."""

from __future__ import annotations

from PySide6.QtCore import QObject

from chiron.config.settings import Settings, is_live_selection
from chiron.journal.compaction import JournalCompactor
from chiron.journal.service import JournalService
from chiron.observer.nonlive import NonLiveObserverSessionManager
from chiron.observer.session import LiveObserverSessionManager
from chiron.session import ObserverAgent


def build_observer(
    settings: Settings,
    journal: JournalService,
    journal_compactor: JournalCompactor | None = None,
    parent: QObject | None = None,
) -> ObserverAgent:
    """Construct only an Observer; no answer surface is shared or selectable."""
    cls = (
        LiveObserverSessionManager
        if is_live_selection(settings.observer_model)
        else NonLiveObserverSessionManager
    )
    return cls(settings, journal, journal_compactor, parent=parent)


__all__ = ["build_observer"]
