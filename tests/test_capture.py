"""Frame encoding and deterministic fixed scheduling."""

from __future__ import annotations

import io

import pytest
from PIL import Image

from chiron.capture.frames import Frame, downscale, encode_frame, stamp_timestamp
from chiron.capture.scheduler import FixedIntervalScheduler


def image(colour=(30, 60, 90), size=(1920, 1080)) -> Image.Image:
    return Image.new("RGB", size, colour)


def test_downscale_preserves_aspect_ratio():
    scaled = downscale(image(size=(1920, 1080)), 768)
    assert scaled.size == (768, 432)


def test_downscale_never_upscales():
    original = image(size=(400, 300))
    assert downscale(original, 768) is original


def test_encode_produces_a_jpeg_without_detection_metadata():
    frame = encode_frame(image(), width=768, quality=60)
    assert frame.jpeg[:2] == b"\xff\xd8"
    assert Image.open(io.BytesIO(frame.jpeg)).size == (768, 432)
    assert not hasattr(frame, "signature")


def test_stamp_marks_the_corner_without_touching_the_rest():
    plain = image(colour=(30, 60, 90), size=(320, 180))
    stamped = stamp_timestamp(plain, 1_700_000_000.0)
    assert stamped.getpixel((2, 2)) != (30, 60, 90)
    assert stamped.getpixel((300, 170)) == (30, 60, 90)


def test_frame_clock_formats_local_time():
    frame = Frame(jpeg=b"", captured_at=1_700_000_000.0, width=1, height=1)
    assert len(frame.clock.split(":")) == 3


def test_first_fixed_frame_is_due_immediately():
    assert FixedIntervalScheduler().is_due(1000.0) is True


def test_fixed_deadline_spacing():
    scheduler = FixedIntervalScheduler(interval_seconds=5.0)
    scheduler.note_capture(1000.0)
    assert scheduler.seconds_until_due(1002.0) == 3.0
    assert scheduler.is_due(1004.999) is False
    assert scheduler.is_due(1005.0) is True


def test_missed_ticks_do_not_catch_up():
    scheduler = FixedIntervalScheduler(interval_seconds=5.0)
    scheduler.note_capture(1000.0)
    assert scheduler.is_due(1030.0) is True
    scheduler.note_capture(1030.0)
    assert scheduler.is_due(1030.1) is False
    assert scheduler.seconds_until_due(1030.1) == pytest.approx(4.9)


def test_reset_makes_one_frame_due_now():
    scheduler = FixedIntervalScheduler(interval_seconds=5.0)
    scheduler.note_capture(1000.0)
    scheduler.reset()
    assert scheduler.is_due(1001.0) is True
    scheduler.note_capture(1001.0)
    assert scheduler.is_due(1001.1) is False
