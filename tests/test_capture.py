"""Frame encoding, scene-change signatures and the adaptive shutter."""

from __future__ import annotations

import io

from PIL import Image

from chiron.capture.frames import (
    Frame,
    downscale,
    encode_frame,
    frame_signature,
    signature_distance,
    stamp_timestamp,
)
from chiron.capture.scheduler import AdaptiveScheduler


def _image(colour=(30, 60, 90), size=(1920, 1080)) -> Image.Image:
    return Image.new("RGB", size, colour)


# --------------------------------------------------------------------- frames


def test_downscale_preserves_aspect_ratio():
    scaled = downscale(_image(size=(1920, 1080)), 768)
    assert scaled.width == 768
    assert scaled.height == 432


def test_downscale_never_upscales():
    original = _image(size=(400, 300))
    assert downscale(original, 768) is original


def test_encode_produces_a_jpeg():
    frame = encode_frame(_image(), width=768, quality=60)
    assert frame.jpeg[:2] == b"\xff\xd8"  # JPEG SOI marker
    assert Image.open(io.BytesIO(frame.jpeg)).size == (768, 432)
    assert frame.width == 768


def test_stamp_marks_the_corner_without_touching_the_rest():
    plain = _image(colour=(30, 60, 90), size=(320, 180))
    stamped = stamp_timestamp(plain, 1_700_000_000.0)
    assert stamped.getpixel((2, 2)) != (30, 60, 90), "corner is stamped"
    assert stamped.getpixel((300, 170)) == (30, 60, 90), "rest is untouched"


def test_signature_is_computed_before_stamping():
    """The clock ticks every second; that must not read as a scene change."""
    image = _image(size=(640, 360))
    first = encode_frame(image, captured_at=1_700_000_000.0)
    second = encode_frame(image, captured_at=1_700_000_060.0)
    assert first.signature == second.signature
    assert signature_distance(first.signature, second.signature) == 0.0


def test_signature_distance_bounds():
    black = frame_signature(Image.new("RGB", (64, 64), (0, 0, 0)))
    white = frame_signature(Image.new("RGB", (64, 64), (255, 255, 255)))
    assert signature_distance(black, black) == 0.0
    assert signature_distance(black, white) == 1.0
    assert signature_distance(b"", white) == 0.0


def test_scene_change_crosses_default_threshold():
    game = encode_frame(_image(colour=(40, 70, 30), size=(640, 360)))
    loading = encode_frame(_image(colour=(5, 5, 5), size=(640, 360)))
    assert signature_distance(game.signature, loading.signature) > 0.12


def test_frame_clock_formats_local_time():
    frame = Frame(jpeg=b"", captured_at=1_700_000_000.0, width=1, height=1)
    assert len(frame.clock.split(":")) == 3


# ------------------------------------------------------------------ scheduler


def test_first_frame_is_due_immediately():
    scheduler = AdaptiveScheduler()
    assert scheduler.is_due(1000.0) is True


def test_baseline_spacing():
    scheduler = AdaptiveScheduler(baseline_interval=4.0)
    scheduler.note_capture(1000.0)
    assert scheduler.is_due(1002.0) is False
    assert scheduler.seconds_until_due(1002.0) == 2.0
    assert scheduler.is_due(1004.0) is True


def test_burst_raises_the_rate_and_fires_at_once():
    scheduler = AdaptiveScheduler(baseline_interval=4.0, burst_interval=1.0)
    scheduler.note_capture(1000.0)
    assert scheduler.is_due(1001.0) is False

    scheduler.request_burst(1001.0, "question")
    assert scheduler.mode(1001.0) == "burst"
    assert scheduler.is_due(1001.0) is True, "a question cannot wait out the gap"

    scheduler.note_capture(1001.0)
    assert scheduler.is_due(1002.0) is True, "1 fps while bursting"


def test_burst_expires_back_to_baseline():
    scheduler = AdaptiveScheduler(burst_duration=15.0)
    scheduler.request_burst(1000.0, "scene change")
    assert scheduler.mode(1010.0) == "burst"
    assert scheduler.mode(1020.0) == "baseline"

    assert scheduler.expire_burst(1020.0) is True
    assert scheduler.expire_burst(1020.0) is False, "reported exactly once"
    assert scheduler.burst_reason == ""


def test_burst_extends_rather_than_restarts():
    scheduler = AdaptiveScheduler(burst_duration=15.0)
    scheduler.request_burst(1000.0, "question")
    scheduler.request_burst(1005.0, "scene change")
    assert scheduler.burst_until == 1020.0
    assert scheduler.mode(1019.0) == "burst"


def test_interval_depends_on_mode():
    scheduler = AdaptiveScheduler(baseline_interval=4.0, burst_interval=1.0)
    assert scheduler.interval(1000.0) == 4.0
    scheduler.request_burst(1000.0)
    assert scheduler.interval(1000.0) == 1.0
