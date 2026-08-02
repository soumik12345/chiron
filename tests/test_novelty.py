"""The ambient-weighted novelty detector, and the observer trigger over it.

The central claim these tests exist to hold down: a screen full of *looping*
motion — grass, water, an animated menu — must not read as an event, while a
real transition must, and both with the same settings. That is the entire reason
this detector exists rather than the plain pixel diff the capture service uses.
"""

from __future__ import annotations

import random

from chiron.capture.frames import Frame, signature_distance
from chiron.capture.novelty import NoveltyDetector
from chiron.config.settings import ObserverSettings
from chiron.nonlive.observer import ObserverTrigger

CELLS = 32 * 32
#: Rows at or below this index shimmer every frame; the rest of the screen is
#: still. Deliberately violent enough — over half the frame, full range — to
#: clear the v0 detector's threshold on its own, which is the whole problem.
ANIMATED_FROM = 14


def _screen(base: int, *, rng: random.Random, animated: bool = True) -> bytes:
    """A signature: a flat field of `base`, with an optional shimmering band."""
    cells = []
    for index in range(CELLS):
        if animated and index // 32 >= ANIMATED_FROM:
            cells.append(rng.randint(0, 255))
        else:
            cells.append(base)
    return bytes(cells)


def _settle(detector: NoveltyDetector, rng: random.Random, frames: int = 40) -> float:
    """Feed `frames` of pure idle animation; return the time reached."""
    now = 0.0
    detector.reset(now)
    for _ in range(frames):
        now += 4.0
        detector.observe(_screen(100, rng=rng), now)
    return now


def _frame(signature: bytes, when: float) -> Frame:
    """A Frame carrying nothing but a signature and a capture time."""
    return Frame(jpeg=b"", captured_at=when, width=32, height=32, signature=signature)


# ------------------------------------------------------------------ detector


def test_idle_animation_is_not_an_event():
    """The failure the v0 detector has: permanent motion, no meaning."""
    rng = random.Random(7)
    detector = NoveltyDetector()
    now = detector_time = 0.0
    detector.reset(now)

    spikes = 0
    for index in range(60):
        detector_time += 4.0
        reading = detector.observe(_screen(100, rng=rng), detector_time)
        if index >= 10:
            spikes += int(reading.spike)

    assert spikes == 0, "swaying grass is not news, however much of it there is"


def test_the_plain_pixel_diff_would_have_fired_on_that_same_animation():
    """Why the weighting is needed at all, stated as a test rather than a claim."""
    rng = random.Random(7)
    first = _screen(100, rng=rng)
    second = _screen(100, rng=rng)
    assert signature_distance(first, second) >= 0.12, (
        "the v0 threshold is cleared by idle animation alone"
    )


def test_a_real_transition_spikes_through_the_animation():
    rng = random.Random(7)
    detector = NoveltyDetector()
    now = _settle(detector, rng)

    # The still part of the screen changes wholesale; the band keeps shimmering.
    reading = detector.observe(_screen(10, rng=rng), now + 4.0)

    assert reading.spike is True
    assert reading.novelty > reading.threshold


def test_spikes_are_suppressed_while_the_field_is_still_learning():
    """Everything looks unusual to a detector that has seen nothing."""
    rng = random.Random(3)
    detector = NoveltyDetector()
    detector.reset(0.0)

    readings = [
        detector.observe(_screen(100 if step else 10, rng=rng), 1.0 + step)
        for step in range(4)
    ]

    assert all(r.warming_up for r in readings)
    assert not any(r.spike for r in readings)


def test_warm_up_ends_after_the_configured_window():
    rng = random.Random(5)
    detector = NoveltyDetector(warmup_seconds=10.0)
    detector.reset(0.0)
    for step in range(5):
        reading = detector.observe(_screen(100, rng=rng), 2.0 * step)

    reading = detector.observe(_screen(100, rng=rng), 60.0)
    assert reading.warming_up is False


def test_drift_catches_the_slow_change_a_frame_diff_misses():
    """A long walk into a new biome moves no two adjacent frames far apart."""
    rng = random.Random(11)
    detector = NoveltyDetector()
    now = _settle(detector, rng)
    detector.note_observation(_screen(100, rng=rng))

    spiked = False
    for step in range(1, 41):
        now += 4.0
        # One grey level per frame: never a spike, eventually a different place.
        spiked |= detector.observe(_screen(100 + step, rng=rng), now).spike

    assert spiked is False, "no single frame is a scene change"
    assert detector.has_drifted(_screen(140, rng=rng)) is True


def test_observation_resets_the_drift_reference():
    rng = random.Random(13)
    detector = NoveltyDetector()
    _settle(detector, rng)

    moved = _screen(160, rng=rng)
    assert detector.has_drifted(moved) is True

    detector.note_observation(moved)
    assert detector.has_drifted(moved) is False, "already looked at this"


def test_reset_forgets_everything():
    rng = random.Random(17)
    detector = NoveltyDetector()
    _settle(detector, rng)
    assert detector.warm is True

    detector.reset(0.0)

    assert detector.warm is False
    assert detector.drift(_screen(10, rng=rng)) == 0.0


def test_a_signature_of_a_different_size_is_not_an_event():
    """Only reachable if the thumbnail size changed, and then nothing is comparable."""
    detector = NoveltyDetector()
    detector.reset(0.0)
    detector.observe(bytes(CELLS), 1.0)
    assert detector.observe(bytes(16), 2.0).spike is False


def test_an_empty_signature_is_ignored():
    detector = NoveltyDetector()
    assert detector.observe(b"", 1.0).novelty == 0.0


# ------------------------------------------------------------------- trigger


def _trigger(**overrides) -> ObserverTrigger:
    settings = ObserverSettings(**overrides)
    return ObserverTrigger(settings=settings)


def test_the_cooldown_is_the_ceiling_on_observer_spend():
    rng = random.Random(19)
    trigger = _trigger(cooldown_seconds=30.0)
    trigger.reset(0.0)
    # Warm the detector up past its window, then hand it a real transition.
    now = 100.0
    for _ in range(10):
        now += 4.0
        trigger.observe(_frame(_screen(100, rng=rng), now), now)
    trigger.observe(_frame(_screen(10, rng=rng), now + 4.0), now + 4.0)

    assert trigger.due(now + 5.0) is not None
    trigger.note_run(now + 5.0)
    assert trigger.due(now + 6.0) is None, "however noisy, not twice inside a cooldown"


def test_a_plain_heartbeat_fires_on_a_still_screen():
    trigger = _trigger(
        trigger_strategy="spike_plain_heartbeat",
        heartbeat_interval_seconds=60.0,
        cooldown_seconds=10.0,
    )
    trigger.reset(0.0)
    still = _screen(100, rng=random.Random(1), animated=False)
    trigger.observe(_frame(still, 1.0), 1.0)

    assert trigger.due(30.0) is None, "not yet due"
    assert trigger.due(61.0) == "heartbeat"


def test_a_gated_heartbeat_skips_a_screen_that_has_not_moved():
    trigger = _trigger(
        trigger_strategy="spike_gated_heartbeat",
        heartbeat_interval_seconds=60.0,
        cooldown_seconds=10.0,
    )
    trigger.reset(0.0)
    still = _screen(100, rng=random.Random(1), animated=False)
    trigger.observe(_frame(still, 1.0), 1.0)

    assert trigger.due(61.0) is None, "nothing has happened; the tick costs nothing"


def test_a_gated_heartbeat_fires_once_the_screen_has_drifted():
    rng = random.Random(23)
    trigger = _trigger(heartbeat_interval_seconds=60.0, cooldown_seconds=10.0)
    trigger.reset(0.0)
    trigger.observe(_frame(_screen(100, rng=rng, animated=False), 1.0), 1.0)
    trigger.note_run(2.0)

    trigger.observe(_frame(_screen(200, rng=rng, animated=False), 70.0), 70.0)

    assert trigger.due(71.0) == "drift"


def test_a_skipped_heartbeat_waits_a_full_interval_before_asking_again():
    trigger = _trigger(heartbeat_interval_seconds=60.0, cooldown_seconds=10.0)
    trigger.reset(0.0)
    still = _screen(100, rng=random.Random(1), animated=False)
    trigger.observe(_frame(still, 1.0), 1.0)

    assert trigger.due(61.0) is None
    assert trigger.due(62.0) is None, "not re-examined on every pass"


def test_the_observer_is_shown_the_moment_that_fired_and_the_present():
    rng = random.Random(29)
    trigger = _trigger(max_frames_per_call=3)
    trigger.reset(0.0)
    now = 100.0
    for _ in range(10):
        now += 4.0
        trigger.observe(_frame(_screen(100, rng=rng), now), now)

    now += 4.0
    trigger.observe(_frame(_screen(10, rng=rng), now), now)
    now += 4.0
    latest = _frame(_screen(10, rng=rng), now)
    trigger.observe(latest, now)

    frames = trigger.frames_for_call()
    assert frames[-1] is latest, "the present is always last"
    assert len(frames) <= 3


def test_a_run_clears_what_was_pending():
    rng = random.Random(31)
    trigger = _trigger()
    trigger.reset(0.0)
    now = 100.0
    for _ in range(10):
        now += 4.0
        trigger.observe(_frame(_screen(100, rng=rng), now), now)
    trigger.observe(_frame(_screen(10, rng=rng), now + 4.0), now + 4.0)
    assert trigger.pending

    trigger.note_run(now + 5.0)

    assert trigger.pending == []
    assert trigger.reason == ""


def test_reset_starts_the_warm_up_again():
    rng = random.Random(37)
    trigger = _trigger()
    trigger.reset(0.0)
    now = 100.0
    for _ in range(10):
        now += 4.0
        trigger.observe(_frame(_screen(100, rng=rng), now), now)

    trigger.reset(now)

    assert trigger.observe(_frame(_screen(10, rng=rng), now + 4.0), now + 4.0) is False
