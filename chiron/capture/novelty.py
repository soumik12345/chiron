"""Deciding when the screen has changed in a way worth paying to look at.

The v0 scene-change detector — mean absolute difference between consecutive
32x32 signatures, fire above 0.12 — is fine for what it does, which is *bursting*
the shutter. A false burst costs a few extra frames into a session that is
already open. An observer trigger costs a whole LLM call per firing, and at that
price the detector's structural false positive stops being tolerable:
**persistent, legitimate pixel motion with no semantic content**. Swaying grass,
water shaders and animated menu backgrounds hold the consecutive-frame distance
permanently elevated. No fixed threshold separates "shimmering menu" from
"entered a new area": below the ambient level it fires on every frame, and the
cooldown then degrades the observer into a fixed-interval poller burning money
while the player is AFK in a menu; above it, real events go unseen. Being
frame-to-consecutive-frame, it is also blind to gradual change — a slow fade, a
long walk into a new biome — which never crosses any threshold between adjacent
frames.

The fix is to ask a different question. Idle animations are *stationary*: the
same cells change by roughly the same amount, frame after frame. So instead of
"did pixels change?" this asks **"did pixels change differently than they have
been changing?"**

1. A per-cell **ambient motion field** — an exponential moving average of each
   cell's absolute frame-to-frame difference. Grass and fire cells develop high
   ambient values; HUD and sky stay near zero.
2. **Weighted novelty**: the mean per-cell difference, each cell down-weighted by
   its own ambient motion. A static player amid looping animation scores near
   zero, because the moving cells are exactly the discounted ones. When the scene
   really changes, the *stable* full-weight cells all move at once and novelty
   spikes.
3. An **adaptive threshold** — novelty measured against its own rolling mean and
   deviation rather than a constant — so a busy shooter and a static visual novel
   self-calibrate without a per-game knob.
4. A **drift** measure against the signature at the observer's last run, which
   catches the slow change consecutive differencing misses, and which is also
   what lets a gated heartbeat skip its call entirely.
5. **Warm-up**: for the first half-minute the ambient field is still learning, so
   spikes are suppressed and the heartbeat covers the window.

The detector is independent of frame detail by construction: it reads
:attr:`~chiron.capture.frames.Frame.signature`, which ``encode_frame`` always
produces at 32x32 whatever the capture width, and never touches the API-side
``media_resolution`` knob at all. Raising detail only makes each cell average
more source pixels — marginally *more* stable, never less.

Honest residual limits, accepted: the onset of a new animation reads as an event
(usually defensible — something did start); after a real transition the ambient
field re-learns over a few frames (harmless, the transition already fired); and
it detects statistically unusual visual change, not meaning. The observer model
remains the final arbiter of significance. Detection only decides when it is
worth paying to ask.

Pure arithmetic, no clock of its own and no Qt: every method takes the current
time, so the whole policy is testable by handing it a sequence of signatures.
"""

from __future__ import annotations

from dataclasses import dataclass, field

#: Frames over which a cell's ambient motion estimate decays by half. Roughly a
#: minute at the 4 s baseline, which is long enough to average out a burst and
#: short enough to re-learn after a real transition.
DEFAULT_AMBIENT_HALF_LIFE = 12.0

#: Added to every cell's ambient motion before inverting it into a weight. Sets
#: how hard a busy cell can be discounted: without it a perfectly still cell
#: would carry infinite weight and a single stray pixel would read as an event.
DEFAULT_EPSILON = 6.0

#: Seconds after a reset during which spikes are suppressed. The ambient field
#: has seen nothing yet, so *everything* looks unusual; the heartbeat covers it.
DEFAULT_WARMUP_SECONDS = 30.0

#: Novelty below this never counts as a spike however quiet the rolling stats
#: get. On a frozen screen the deviation collapses towards zero and any
#: rounding wobble would clear a purely relative threshold.
DEFAULT_NOVELTY_FLOOR = 0.02

#: Weighted drift above which a gated heartbeat decides something has actually
#: happened since the observer last looked.
DEFAULT_DRIFT_THRESHOLD = 0.05


@dataclass(frozen=True)
class NoveltyReading:
    """What one frame looked like to the detector.

    Attributes:
        novelty (float): Ambient-weighted difference from the previous frame,
            normalised to 0-1.
        threshold (float): What `novelty` had to beat to count as a spike.
        drift (float): Ambient-weighted difference from the signature at the
            last :meth:`NoveltyDetector.note_observation`.
        spike (bool): Whether this frame is an event worth an observer call.
        warming_up (bool): Whether the ambient field is still learning, in which
            case `spike` is suppressed regardless of `novelty`.
    """

    novelty: float = 0.0
    threshold: float = 0.0
    drift: float = 0.0
    spike: bool = False
    warming_up: bool = True


@dataclass
class NoveltyDetector:
    """Ambient-weighted novelty over a stream of frame signatures.

    Attributes:
        sensitivity (float): Deviations above the rolling mean that count as a
            spike. Higher is more conservative.
        ambient_half_life (float): Frames over which the ambient field decays by
            half.
        epsilon (float): Floor on a cell's ambient motion when weighting.
        warmup_seconds (float): How long after a reset spikes are suppressed.
        drift_threshold (float): Weighted drift that counts as "something has
            happened since the observer last ran".
    """

    sensitivity: float = 3.0
    ambient_half_life: float = DEFAULT_AMBIENT_HALF_LIFE
    epsilon: float = DEFAULT_EPSILON
    warmup_seconds: float = DEFAULT_WARMUP_SECONDS
    novelty_floor: float = DEFAULT_NOVELTY_FLOOR
    drift_threshold: float = DEFAULT_DRIFT_THRESHOLD

    _ambient: list[float] = field(default_factory=list, repr=False)
    _previous: bytes = field(default=b"", repr=False)
    _baseline: bytes = field(default=b"", repr=False)
    _mean: float = field(default=0.0, repr=False)
    _deviation: float = field(default=0.0, repr=False)
    _started_at: float | None = field(default=None, repr=False)
    _frames: int = field(default=0, repr=False)

    # ------------------------------------------------------------------ state

    def reset(self, now: float | None = None) -> None:
        """Forget everything and start learning again.

        Called whenever watching starts — the screen has moved on during the
        pause, so the first frame back is not news — and whenever capture
        settings change, since a different frame width shifts the statistics the
        thumbnail is drawn from.

        Args:
            now (float | None): Current time, which starts the warm-up window.
                None leaves the window to start with the next frame.
        """
        self._ambient = []
        self._previous = b""
        self._baseline = b""
        self._mean = 0.0
        self._deviation = 0.0
        self._frames = 0
        self._started_at = now

    @property
    def warm(self) -> bool:
        """Whether the ambient field has learned enough to be trusted."""
        return self._frames >= 3

    def note_observation(self, signature: bytes) -> None:
        """Record the screen as the observer last saw it, resetting drift."""
        self._baseline = signature

    # -------------------------------------------------------------- measuring

    def observe(self, signature: bytes, now: float) -> NoveltyReading:
        """Fold one frame in and say whether it is an event.

        Args:
            signature (bytes): The frame's 32x32 greyscale signature.
            now (float): Capture time, used only for the warm-up window.

        Returns:
            NoveltyReading: The measurement, and the verdict.
        """
        if not signature:
            return NoveltyReading()
        if self._started_at is None:
            self._started_at = now

        previous, self._previous = self._previous, signature
        if not self._baseline:
            self._baseline = signature
        if len(previous) != len(signature):
            # First frame, or the signature size changed under us. Either way
            # there is nothing to compare against; start the field from here.
            self._ambient = [0.0] * len(signature)
            return NoveltyReading(warming_up=True)

        differences = [abs(a - b) for a, b in zip(previous, signature)]
        weights = self._weights()
        novelty = _weighted_mean(differences, weights) / 255.0
        drift = self.drift(signature)

        self._update_ambient(differences)
        self._frames += 1

        warming_up = not self.warm or (
            self._started_at is not None
            and now - self._started_at < self.warmup_seconds
        )
        threshold = max(
            self.novelty_floor, self._mean + self.sensitivity * self._deviation
        )
        # The rolling statistics are updated *after* the comparison, so a frame
        # is never measured against a threshold it helped set.
        self._update_statistics(novelty)

        return NoveltyReading(
            novelty=novelty,
            threshold=threshold,
            drift=drift,
            spike=novelty > threshold and not warming_up,
            warming_up=warming_up,
        )

    def drift(self, signature: bytes) -> float:
        """Weighted distance from the screen the observer last saw.

        This is the slow-change channel: a long walk into a new biome moves no
        two adjacent frames far apart, but moves a long way from where it
        started. It is also what a gated heartbeat consults before deciding a
        tick is worth paying for.

        Args:
            signature (bytes): The signature to compare against the baseline.

        Returns:
            float: Normalised 0-1 weighted difference, 0.0 when there is no
                baseline yet or the lengths disagree.
        """
        if not signature or len(self._baseline) != len(signature):
            return 0.0
        differences = [abs(a - b) for a, b in zip(self._baseline, signature)]
        return _weighted_mean(differences, self._weights()) / 255.0

    def has_drifted(self, signature: bytes) -> bool:
        """Whether the screen has moved on since the observer last looked."""
        return self.drift(signature) >= self.drift_threshold

    # ------------------------------------------------------------- internals

    def _weights(self) -> list[float]:
        """One weight per cell, inversely proportional to its ambient motion."""
        if not self._ambient:
            return []
        return [1.0 / (self.epsilon + value) for value in self._ambient]

    def _update_ambient(self, differences: list[float]) -> None:
        """Fold this frame's per-cell motion into the ambient field."""
        if len(self._ambient) != len(differences):
            self._ambient = list(differences)
            return
        decay = 0.5 ** (1.0 / max(1.0, self.ambient_half_life))
        self._ambient = [
            decay * ambient + (1.0 - decay) * difference
            for ambient, difference in zip(self._ambient, differences)
        ]

    def _update_statistics(self, novelty: float) -> None:
        """Track novelty's own rolling mean and mean absolute deviation.

        Mean absolute deviation rather than standard deviation, because the
        distribution this watches is spiky by construction and squaring the
        outliers would let one real event raise the bar against the next.
        """
        decay = 0.5 ** (1.0 / max(1.0, self.ambient_half_life))
        deviation = abs(novelty - self._mean)
        self._mean = decay * self._mean + (1.0 - decay) * novelty
        self._deviation = decay * self._deviation + (1.0 - decay) * deviation


def _weighted_mean(values: list[float], weights: list[float]) -> float:
    """The weighted mean of `values`, or their plain mean if weights are absent."""
    if not values:
        return 0.0
    if len(weights) != len(values):
        return sum(values) / len(values)
    total_weight = sum(weights)
    if total_weight <= 0.0:
        return sum(values) / len(values)
    return sum(v * w for v, w in zip(values, weights)) / total_weight


__all__ = [
    "DEFAULT_AMBIENT_HALF_LIFE",
    "DEFAULT_DRIFT_THRESHOLD",
    "DEFAULT_EPSILON",
    "DEFAULT_NOVELTY_FLOOR",
    "DEFAULT_WARMUP_SECONDS",
    "NoveltyDetector",
    "NoveltyReading",
]
