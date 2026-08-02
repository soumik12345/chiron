"""When to pay for a look: the observer's trigger policy.

Three mechanisms, in order of how much they are trusted:

* **Spikes.** :class:`~chiron.capture.novelty.NoveltyDetector` says a frame
  changed differently from how the screen has been changing. This is the
  responsive channel and the one that catches real events.
* **The heartbeat.** A periodic tick, so a session that somehow never spikes
  still journals something. Under ``spike_gated_heartbeat`` the tick consults
  drift first and skips the call outright when nothing has moved since the last
  run; under ``spike_plain_heartbeat`` it always fires. The gated form is the
  better cost profile; the plain form is the simpler guarantee.
* **The cooldown.** A hard floor on the gap between calls, applied to everything
  above. It is what makes the worst case bounded: however noisy a game's
  triggers are, observer spend cannot exceed one call per cooldown.

Like :class:`~chiron.capture.scheduler.AdaptiveScheduler`, this is clock-free —
every method takes the current time — so the policy can be tested by handing it
a sequence of timestamps and signatures, with no sleeping and no model.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from chiron.capture.frames import Frame
from chiron.capture.novelty import NoveltyDetector
from chiron.config.settings import ObserverSettings


@dataclass
class ObserverTrigger:
    """Decides when the observer should run, and why.

    Attributes:
        settings (ObserverSettings): Cadence and sensitivity, re-read on every
            call so edits apply in place without a restart.
        detector (NoveltyDetector): The novelty measurement.
        last_run (float): When the observer last actually ran.
        pending (list[Frame]): Frames captured at trigger moments since then —
            what the observer is shown alongside the latest frame, so it sees
            the moment that fired rather than only the aftermath.
        reason (str): Why the next run is due, for the status line.
    """

    settings: ObserverSettings = field(default_factory=ObserverSettings)
    detector: NoveltyDetector = field(default_factory=NoveltyDetector)
    last_run: float = 0.0
    pending: list[Frame] = field(default_factory=list)
    reason: str = ""
    _dirty: bool = field(default=False, repr=False)
    _last_heartbeat: float = field(default=0.0, repr=False)
    _latest: Frame | None = field(default=None, repr=False)

    #: Trigger frames kept between runs. More than a handful is pointless — the
    #: observer is only ever shown `max_frames_per_call` of them.
    max_pending: int = 8

    def reset(self, now: float) -> None:
        """Start again: no history, no pending frames, warm-up from `now`.

        Called when watching starts and when capture settings change. Both cases
        would otherwise have the detector comparing frames across a gap it knows
        nothing about, which reads as an event every time.
        """
        self.detector.reset(now)
        self.detector.sensitivity = self.settings.spike_sensitivity
        self.pending.clear()
        self.reason = ""
        self._dirty = False
        self._latest = None
        self.last_run = now
        self._last_heartbeat = now

    def observe(self, frame: Frame, now: float) -> bool:
        """Fold one captured frame in.

        Args:
            frame (Frame): The frame that just arrived.
            now (float): Current time.

        Returns:
            bool: True when this frame was a novelty spike. The caller does not
                need it — :meth:`due` is the decision — but the status line and
                the tests do.
        """
        self.detector.sensitivity = self.settings.spike_sensitivity
        self._latest = frame
        reading = self.detector.observe(frame.signature, now)
        if reading.spike:
            self._dirty = True
            self.reason = self.reason or "scene change"
            self.pending.append(frame)
            del self.pending[: max(0, len(self.pending) - self.max_pending)]
        return reading.spike

    def due(self, now: float) -> str | None:
        """Whether the observer should run now, and why.

        Returns:
            str | None: A short reason ("scene change", "heartbeat"), or None.
        """
        if now - self.last_run < self.settings.cooldown_seconds:
            return None
        if self._dirty:
            return self.reason or "scene change"
        if now - self._last_heartbeat < self.settings.heartbeat_interval_seconds:
            return None
        # The heartbeat has come round. Advance it either way: a gated tick that
        # decides against calling must still wait a full interval before asking
        # again, or a drifting-but-not-yet-drifted screen would be re-examined
        # on every pass.
        self._last_heartbeat = now
        if self.settings.trigger_strategy == "spike_plain_heartbeat":
            return "heartbeat"
        signature = self._latest.signature if self._latest is not None else b""
        if signature and self.detector.has_drifted(signature):
            return "drift"
        return None

    def frames_for_call(self, limit: int | None = None) -> list[Frame]:
        """The frames to show the observer: trigger moments, newest last.

        The latest frame is always included and always last, because "what is
        happening now" is the thing the observer is being asked about; the
        trigger frames before it are what made the question worth asking.
        """
        cap = self.settings.max_frames_per_call if limit is None else limit
        frames = list(self.pending)
        if self._latest is not None and (not frames or frames[-1] is not self._latest):
            frames.append(self._latest)
        return frames[-max(1, cap) :]

    def note_run(self, now: float) -> None:
        """Record that the observer has just run.

        Resets drift to the screen as it is right now, so the next gated
        heartbeat asks "has anything changed since I last looked" rather than
        since some older reference.
        """
        self.last_run = now
        self._last_heartbeat = now
        self._dirty = False
        self.reason = ""
        self.pending.clear()
        if self._latest is not None:
            self.detector.note_observation(self._latest.signature)


__all__ = ["ObserverTrigger"]
