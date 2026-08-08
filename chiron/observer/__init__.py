"""Chiron-Observer transports: Live checkpoints or buffered video reviews."""

from __future__ import annotations


def __getattr__(name: str):
    """Keep pure media imports from eagerly loading provider dependencies."""
    if name == "build_observer":
        from chiron.observer.factory import build_observer

        return build_observer
    if name == "NonLiveObserverSessionManager":
        from chiron.observer.nonlive import NonLiveObserverSessionManager

        return NonLiveObserverSessionManager
    if name in {"LiveObserverSessionManager", "ObserverSessionManager"}:
        from chiron.observer.session import (
            LiveObserverSessionManager,
            ObserverSessionManager,
        )

        return {
            "LiveObserverSessionManager": LiveObserverSessionManager,
            "ObserverSessionManager": ObserverSessionManager,
        }[name]
    raise AttributeError(name)


__all__ = [
    "LiveObserverSessionManager",
    "NonLiveObserverSessionManager",
    "ObserverSessionManager",
    "build_observer",
]
