"""Buffered Observer media construction and structured result validation."""

from __future__ import annotations

import io
import json

import av
import pytest
from PIL import Image

from chiron.capture.frames import encode_frame
from chiron.config.settings import Settings
from chiron.journal.service import JournalSnapshot
from chiron.observer.media import encode_mp4, encode_split_mp4
from chiron.observer.structured import (
    BATCHED_OBSERVER_SYSTEM_PROMPT,
    StructuredObserverError,
    build_batch_messages,
    journal_result_schema,
    validate_result,
    video_part,
)


def frames(count: int = 3, *, width: int = 66, height: int = 34):
    return [
        encode_frame(
            Image.new("RGB", (width, height), (index * 50, 10, 20)),
            width=width,
            stamp=False,
            captured_at=1_700_000_000.125 + index * 5,
        )
        for index in range(count)
    ]


def test_mp4_is_silent_one_fps_even_and_in_source_order():
    source = frames()
    encoded = encode_mp4(source, "low")
    container = av.open(io.BytesIO(encoded.data))
    decoded = list(container.decode(video=0))

    assert (encoded.width, encoded.height) == (66, 34)
    assert len(container.streams.audio) == 0
    assert len(decoded) == 3
    assert [float(frame.pts * frame.time_base) for frame in decoded] == [0, 1, 2]


def test_detail_never_upscales_and_produces_even_dimensions():
    encoded = encode_mp4(frames(1, width=65, height=33), "high")
    assert encoded.width == 64
    assert encoded.height == 32


def test_oversized_media_splits_contiguously():
    source = frames(4, width=128, height=64)
    single_size = max(len(encode_mp4([frame]).data) for frame in source)
    parts = encode_split_mp4(source, max_bytes=single_size + 100)
    assert len(parts) > 1
    assert [frame for part in parts for frame in part.frames] == source
    assert all(len(part.data) <= single_size + 100 for part in parts)


def test_provider_video_parts_use_the_documented_shapes():
    google = video_part("gemini/gemini-2.5-flash-lite", b"mp4", "low")
    router = video_part("openrouter/google/gemini-2.5-flash", b"mp4", "low")
    assert google["type"] == "file"
    assert google["file"]["file_data"].startswith("data:video/mp4;base64,")
    assert google["video_metadata"] == {"fps": 1}
    assert router["type"] == "video_url"
    assert router["video_url"]["url"].startswith("data:video/mp4;base64,")


def test_schema_and_messages_carry_exact_second_mapping():
    encoded = encode_mp4(frames(2), "low")
    messages = build_batch_messages(Settings(), encoded, JournalSnapshot(()))
    text = messages[-1]["content"][0]["text"]
    schema = journal_result_schema(2)
    assert "video_second=0" in text
    assert "video_second=1" in text
    assert "captured_at=" in text
    assert (
        schema["properties"]["entries"]["items"]["properties"]["video_second"][
            "maximum"
        ]
        == 1
    )


def test_buffered_prompt_and_schema_request_vivid_grounded_visual_memory():
    schema = journal_result_schema(2)
    entry = schema["properties"]["entries"]["items"]
    note = entry["properties"]["note"]

    assert "visual memory for Chiron-Responder" in BATCHED_OBSERVER_SYSTEM_PROMPT
    assert "vivid, self-contained paragraph" in BATCHED_OBSERVER_SYSTEM_PROMPT
    assert "sequence supported by the sampled frames" in BATCHED_OBSERVER_SYSTEM_PROMPT
    assert "explicitly qualified" in BATCHED_OBSERVER_SYSTEM_PROMPT
    assert "empty entries array" in BATCHED_OBSERVER_SYSTEM_PROMPT
    assert "vivid, self-contained scene paragraph" in note["description"]
    assert set(entry["properties"]) == {"video_second", "category", "note"}
    assert entry["required"] == ["video_second", "category", "note"]


@pytest.mark.parametrize(
    "payload,match",
    [
        ("not json", "one JSON"),
        ('{"entries": []} prose', "prose outside"),
        ('{"entries": [], "extra": 1}', "top level"),
        ('{"entries": {}}', "array"),
        (
            json.dumps(
                {"entries": [{"video_second": 2, "category": "progress", "note": "x"}]}
            ),
            "outside",
        ),
        (
            json.dumps(
                {
                    "entries": [
                        {"video_second": 1, "category": "note", "note": "later"},
                        {"video_second": 0, "category": "note", "note": "earlier"},
                    ]
                }
            ),
            "chronological",
        ),
    ],
)
def test_structured_result_rejects_the_whole_invalid_response(payload, match):
    with pytest.raises(StructuredObserverError, match=match):
        validate_result(payload, 2)


def test_valid_empty_and_unknown_category_results_are_accepted():
    assert validate_result('{"entries": []}', 2) == ()
    result = validate_result(
        '{"entries":[{"video_second":1,"category":"mystery","note":"Found it."}]}',
        2,
    )
    assert result[0].category == "mystery"
    assert result[0].video_second == 1
