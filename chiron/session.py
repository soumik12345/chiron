"""The seam between "how Chiron thinks" and everything else it does.

:class:`~chiron.app.ChironApp` already talked to the Live session through a
narrow surface — start, stop, send a question, send a frame, fold the journal,
apply settings, plus five signals. v1 turns that surface into an interface with
two implementations behind it:

* :class:`~chiron.live.session.LiveSessionManager` — a persistent websocket that
  frames stream into, unchanged from v0.
* :class:`~chiron.nonlive.session.NonLiveSessionManager` — a request/response
  endpoint called on demand, with an observer keeping the journal.

**The selected model decides which one runs**, and nothing else does. There is no
mode toggle to contradict the model choice, so the two cannot disagree; picking a
``live/…`` model runs live, picking anything else runs non-live. Crossing that
boundary tears one provider down and builds the other, which is a longer version
of the session restart the settings page already knew how to do.

The status vocabulary is shared deliberately (``idle``, ``connecting``, ``live``,
``reconnecting``, ``error``, ``stopped``). In non-live mode ``live`` means
*armed* rather than *connected* — there is no socket being held open — but the
overlay needs no changes and no branch, which is the whole return on defining the
seam rather than teaching the app about two kinds of session.

:class:`SessionProvider` is a :class:`typing.Protocol`, not a base class. Both
implementations are ``QObject`` subclasses with Qt signals, and Qt's metaclass
does not take kindly to sharing a hierarchy with an ABC; a structural type also
states the honest requirement, which is that the app depends on the surface and
not on the lineage.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from PySide6.QtCore import QObject

from chiron.capture.frames import Frame
from chiron.config.settings import Settings
from chiron.journal.log import JournalLog
from chiron.journal.writers import JournalWriter


@runtime_checkable
class SessionProvider(Protocol):
    """What :class:`~chiron.app.ChironApp` requires of a session.

    Attributes:
        status (str): Current status, from the shared vocabulary.
        detected_game (str): The focused window, phrased for the instruction.
            Runtime state, deliberately not a setting.
        frames_sent (int): Frames handed to the model so far.
    """

    status: str
    detected_game: str

    @property
    def frames_sent(self) -> int:
        """Frames handed to the model so far."""

    def start(self) -> None:
        """Begin a session, or arm the provider where there is nothing to open."""

    async def stop(self, timeout: float = ...) -> None:
        """End the session and stop any background work."""

    def send_text(self, text: str) -> None:
        """Send a player question."""

    def send_frame(self, frame: Frame) -> None:
        """Offer a captured frame."""

    def fold_journal(self, *, force: bool = ...) -> int:
        """Push new journal entries into the session, where that means anything."""

    def reset_observation(self) -> None:
        """Forget what the screen looked like before now."""

    def apply_settings(self, settings: Settings) -> None:
        """Adopt new settings for subsequent requests."""


def build_session_provider(
    settings: Settings,
    journal: JournalLog,
    writer: JournalWriter,
    parent: QObject | None = None,
) -> SessionProvider:
    """Build the provider the selected model implies.

    Args:
        settings (Settings): Current configuration; its ``selected_model`` is
            what decides.
        journal (JournalLog): The shared journal.
        writer (JournalWriter): The journal strategy in force.
        parent (QObject | None): Qt parent for the manager.

    Returns:
        SessionProvider: A live or non-live session manager.
    """
    if settings.is_live:
        from chiron.live.session import LiveSessionManager

        return LiveSessionManager(settings, journal, writer, parent)

    from chiron.nonlive.session import NonLiveSessionManager

    return NonLiveSessionManager(settings, journal, writer, parent)


__all__ = ["SessionProvider", "build_session_provider"]
