"""Pure in-memory time-lapse encoding for the buffered Observer."""

from __future__ import annotations

import io
from dataclasses import dataclass
from fractions import Fraction
from typing import Sequence

from PIL import Image, ImageOps

from chiron.capture.frames import Frame

INLINE_MP4_LIMIT = 10 * 1024 * 1024
DETAIL_WIDTHS = {"low": 512, "medium": 768, "high": 1152}


class VideoEncodingError(RuntimeError):
    """A retained JPEG batch could not be encoded within the media contract."""


@dataclass(frozen=True)
class EncodedVideo:
    """One contiguous encoded part and the source frames it represents."""

    data: bytes
    frames: tuple[Frame, ...]
    width: int
    height: int


def target_dimensions(frames: Sequence[Frame], detail: str) -> tuple[int, int]:
    """Return even, aspect-preserving dimensions without upscaling."""
    if not frames:
        raise ValueError("Cannot size an empty Observer batch")
    first = frames[0]
    target_width = min(
        min(frame.width for frame in frames),
        DETAIL_WIDTHS.get(detail, DETAIL_WIDTHS["low"]),
    )
    width = max(2, int(target_width) // 2 * 2)
    height = max(2, round(first.height * width / max(1, first.width)) // 2 * 2)
    return width, height


def reduced_detail(detail: str) -> str:
    """The one-step lower encoder target used by the local retry."""
    return {"high": "medium", "medium": "low", "low": "reduced"}.get(detail, "reduced")


def _dimensions(frames: Sequence[Frame], detail: str) -> tuple[int, int]:
    if detail != "reduced":
        return target_dimensions(frames, detail)
    first = frames[0]
    width = max(2, min(min(frame.width for frame in frames), 256) // 2 * 2)
    height = max(2, round(first.height * width / max(1, first.width)) // 2 * 2)
    return width, height


def encode_mp4(frames: Sequence[Frame], detail: str = "low") -> EncodedVideo:
    """Encode one silent H.264/yuv420p MP4 with one source frame per second."""
    if not frames:
        raise ValueError("Cannot encode an empty Observer batch")
    try:
        import av
    except ImportError as error:  # pragma: no cover - packaging failure
        raise VideoEncodingError("PyAV is required for buffered observation") from error

    source = tuple(frames)
    width, height = _dimensions(source, detail)
    output = io.BytesIO()
    try:
        container = av.open(output, mode="w", format="mp4")
        stream = container.add_stream("libx264", rate=1)
        stream.width = width
        stream.height = height
        stream.pix_fmt = "yuv420p"
        stream.options = {"preset": "veryfast", "crf": "28"}
        stream.time_base = Fraction(1, 1)
        for index, frame in enumerate(source):
            with Image.open(io.BytesIO(frame.jpeg)) as decoded:
                image = ImageOps.pad(
                    decoded.convert("RGB"),
                    (width, height),
                    method=Image.Resampling.LANCZOS,
                    color=(0, 0, 0),
                )
                video_frame = av.VideoFrame.from_image(image)
            video_frame.pts = index
            video_frame.time_base = Fraction(1, 1)
            for packet in stream.encode(video_frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
        container.close()
    except Exception as error:
        raise VideoEncodingError(f"Could not encode Observer video: {error}") from error
    return EncodedVideo(output.getvalue(), source, width, height)


def encode_split_mp4(
    frames: Sequence[Frame],
    detail: str = "low",
    *,
    max_bytes: int = INLINE_MP4_LIMIT,
) -> list[EncodedVideo]:
    """Encode and recursively split oversized media into contiguous parts."""
    encoded = encode_mp4(frames, detail)
    if len(encoded.data) <= max_bytes:
        return [encoded]
    if len(frames) == 1:
        raise VideoEncodingError(
            f"One encoded frame exceeds the {max_bytes}-byte inline video limit"
        )
    middle = len(frames) // 2
    return [
        *encode_split_mp4(frames[:middle], detail, max_bytes=max_bytes),
        *encode_split_mp4(frames[middle:], detail, max_bytes=max_bytes),
    ]


__all__ = [
    "DETAIL_WIDTHS",
    "INLINE_MP4_LIMIT",
    "EncodedVideo",
    "VideoEncodingError",
    "encode_mp4",
    "encode_split_mp4",
    "reduced_detail",
    "target_dimensions",
]
