"""Turning a screenshot into something worth sending.

Everything here is pure image work — no threads, no screen — so the interesting
parts are testable without a display. A captured screen becomes a :class:`Frame`
in three steps: downscale to the width the API is going to look at anyway, stamp
the capture time into a corner, and JPEG-encode.

The timestamp stamp is not decoration. At one frame every few seconds the model
sees a slideshow with no inherent clock, and "the chest you saw earlier" is only
groundable if each frame says when it was. Burning the time into the pixels gives
the model an explicit *now* that survives into its context alongside the image.

Each frame also carries a :attr:`Frame.signature`: a 32x32 greyscale thumbnail
flattened to bytes. Comparing two signatures is a few hundred integer subtractions
— cheap enough to run on every capture — and is what lets the scheduler notice a
loading screen, a new area or a death screen and burst the shutter for the moments
actually worth journaling.
"""

from __future__ import annotations

import io
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime

from PIL import Image, ImageDraw

logger = logging.getLogger(__name__)

#: Edge length of the greyscale thumbnail used for scene-change comparison.
SIGNATURE_SIZE = 32


@dataclass(frozen=True)
class Frame:
    """One encoded screen capture, ready to send.

    Attributes:
        jpeg (bytes): The encoded image.
        captured_at (float): Unix timestamp of the grab.
        width (int): Encoded width in pixels.
        height (int): Encoded height in pixels.
        signature (bytes): Greyscale thumbnail used for scene-change detection.
    """

    jpeg: bytes
    captured_at: float
    width: int
    height: int
    signature: bytes = field(default=b"", repr=False)

    @property
    def clock(self) -> str:
        """The capture time as ``HH:MM:SS`` local time."""
        return datetime.fromtimestamp(self.captured_at).strftime("%H:%M:%S")


def downscale(image: Image.Image, width: int) -> Image.Image:
    """Return `image` scaled to `width`, preserving aspect ratio.

    Images already narrower than `width` are returned untouched — upscaling costs
    bytes and adds no detail the model can use.
    """
    if width <= 0 or image.width <= width:
        return image
    height = max(1, round(image.height * (width / image.width)))
    return image.resize((width, height), Image.Resampling.LANCZOS)


def stamp_timestamp(image: Image.Image, when: float) -> Image.Image:
    """Draw the capture time into the top-left corner of a copy of `image`.

    The text is drawn white-on-black in a small filled box so it stays legible
    over any game art, and is deliberately placed in a corner where HUD elements
    are least likely to matter.

    Args:
        image (Image.Image): The frame to stamp.
        when (float): Unix timestamp to render.

    Returns:
        Image.Image: A new stamped image.
    """
    stamped = image.convert("RGB")
    draw = ImageDraw.Draw(stamped)
    text = datetime.fromtimestamp(when).strftime("%H:%M:%S")
    left, top, right, bottom = draw.textbbox((0, 0), text)
    pad = 3
    draw.rectangle(
        (0, 0, right - left + 2 * pad, bottom - top + 2 * pad), fill=(0, 0, 0)
    )
    draw.text((pad - left, pad - top), text, fill=(255, 255, 255))
    return stamped


def frame_signature(image: Image.Image, size: int = SIGNATURE_SIZE) -> bytes:
    """Return a tiny greyscale thumbnail of `image` as raw bytes."""
    thumb = image.convert("L").resize((size, size), Image.Resampling.BILINEAR)
    return thumb.tobytes()


def signature_distance(left: bytes, right: bytes) -> float:
    """Normalised 0-1 difference between two signatures.

    Args:
        left (bytes): A signature from :func:`frame_signature`.
        right (bytes): Another signature of the same length.

    Returns:
        float: Mean absolute per-pixel difference divided by 255. Returns 0.0
            when either side is empty and 1.0 when the lengths disagree (which
            can only mean the signature size changed, i.e. everything is new).
    """
    if not left or not right:
        return 0.0
    if len(left) != len(right):
        return 1.0
    total = sum(abs(a - b) for a, b in zip(left, right))
    return total / (len(left) * 255.0)


def encode_frame(
    image: Image.Image,
    *,
    width: int = 768,
    quality: int = 60,
    stamp: bool = True,
    captured_at: float | None = None,
) -> Frame:
    """Downscale, stamp and encode `image` into a :class:`Frame`.

    Args:
        image (Image.Image): The raw screen grab.
        width (int): Target width before encoding.
        quality (int): JPEG quality.
        stamp (bool): Whether to burn the capture time into the frame.
        captured_at (float | None): Capture time; defaults to now.

    Returns:
        Frame: The encoded frame, with its scene-change signature computed from
            the downscaled (but unstamped) image so the stamp's own ticking
            digits never register as a scene change.
    """
    when = time.time() if captured_at is None else captured_at
    scaled = downscale(image, width)
    signature = frame_signature(scaled)
    final = stamp_timestamp(scaled, when) if stamp else scaled.convert("RGB")

    buffer = io.BytesIO()
    final.save(buffer, format="JPEG", quality=quality, optimize=True)
    return Frame(
        jpeg=buffer.getvalue(),
        captured_at=when,
        width=final.width,
        height=final.height,
        signature=signature,
    )


def _mss_factory():
    """The ``mss`` screenshot class, under whichever name this version uses.

    ``mss.mss`` was renamed to ``mss.MSS`` in version 10.2 and the old name warns;
    both spellings are still around in the wild, so the lookup is done here once
    rather than pinning a version.
    """
    import mss

    return getattr(mss, "MSS", None) or mss.mss


class ScreenGrabber:
    """A thin, thread-confined wrapper around ``mss``.

    ``mss`` keeps a per-instance connection to the X server and is explicitly not
    safe to share between threads, so the instance is created lazily on first use
    and therefore belongs to whichever thread called :meth:`grab` first — the
    capture worker.

    Attributes:
        monitor_index (int): Which ``mss`` monitor to grab. 0 is the virtual
            all-monitors screen; 1 is the primary display.
    """

    def __init__(self, monitor_index: int = 1) -> None:
        self.monitor_index = monitor_index
        self._sct = None

    def _instance(self):
        """Return the thread's ``mss`` instance, creating it on first use."""
        if self._sct is None:
            self._sct = _mss_factory()()
        return self._sct

    def grab(self, monitor_index: int | None = None) -> Image.Image:
        """Capture a monitor and return it as a PIL image.

        Args:
            monitor_index (int | None): Override the configured monitor for this
                grab.

        Returns:
            Image.Image: The captured screen in RGB.

        Raises:
            IndexError: If the requested monitor does not exist.
        """
        sct = self._instance()
        index = self.monitor_index if monitor_index is None else monitor_index
        monitors = sct.monitors
        if index >= len(monitors):
            raise IndexError(
                f"Monitor {index} not available; {len(monitors) - 1} attached"
            )
        shot = sct.grab(monitors[index])
        return Image.frombytes("RGB", shot.size, shot.bgra, "raw", "BGRX")

    def close(self) -> None:
        """Release the ``mss`` connection, if one was opened."""
        if self._sct is not None:
            try:
                self._sct.close()
            except Exception:  # pragma: no cover - platform teardown noise
                logger.debug("Ignoring error while closing mss", exc_info=True)
            self._sct = None


def describe_monitors() -> list[str]:
    """Human-readable descriptions of the attached monitors, for the settings UI.

    Returns:
        list[str]: One label per ``mss`` monitor index, starting with the virtual
            all-monitors entry. Returns an empty list when ``mss`` cannot open a
            display (headless CI, for instance), which callers should treat as
            "offer the index as a plain number".
    """
    try:
        with _mss_factory()() as sct:
            labels = []
            for index, monitor in enumerate(sct.monitors):
                scope = "All monitors" if index == 0 else f"Monitor {index}"
                labels.append(
                    f"{scope}: {monitor['width']}x{monitor['height']} "
                    f"at ({monitor['left']}, {monitor['top']})"
                )
            return labels
    except Exception as error:
        logger.warning("Could not enumerate monitors: %s", error)
        return []


__all__ = [
    "SIGNATURE_SIZE",
    "Frame",
    "ScreenGrabber",
    "describe_monitors",
    "downscale",
    "encode_frame",
    "frame_signature",
    "signature_distance",
    "stamp_timestamp",
]
