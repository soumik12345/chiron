"""Provider-neutral request construction and strict Observer result validation."""

from __future__ import annotations

import base64
import json
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from chiron.config.settings import Settings
from chiron.journal.service import JournalSnapshot
from chiron.observer.media import EncodedVideo

BATCHED_OBSERVER_SYSTEM_PROMPT = """\
You are Chiron-Observer, the visual memory for Chiron-Responder. Inspect a silent \
one-frame-per-second gameplay time-lapse and return only structured journal \
entries. Never answer or coach the player, and never emit player-facing narration.

For each distinct meaningful event or materially changed scene, write a vivid, \
self-contained paragraph that lets a reader reconstruct what visibly happened. \
Weave together the relevant setting and spatial context, the player's visible \
activity, identifiable characters, enemies, objects or hazards, legible and \
relevant HUD state, the sequence supported by the sampled frames, and the \
immediate outcome. Keep entries chronological and do not combine unrelated events.

Be concrete, not poetic. Treat names, motives, causes, off-screen state, and action \
hidden between sampled frames as unknown unless the video or existing journal \
supports them. Keep useful but uncertain details only when explicitly qualified \
with language such as "appears," "seems," or "is unclear."

For example, avoid a thin note such as "Defeated an enemy." Prefer a grounded \
account such as: "In a rain-dark courtyard bordered by broken stone arches, the \
player closes on a heavily armored enemy near the central steps. Across the next \
sampled frames the enemy disappears and a reward notification appears, strongly \
suggesting the encounter ended in the player's favor, although the finishing blow \
is not visible."

Do not record unchanged scenery, routine motion, repeated HUD state, transient UI, \
or information already captured in the journal. Return an empty entries array when \
nothing meaningfully changed. Every entry must cite its evidence using the \
zero-based video_second from the supplied mapping.\
"""


class StructuredObserverError(ValueError):
    """The model response failed all-or-nothing local validation."""


@dataclass(frozen=True)
class StructuredJournalEntry:
    video_second: int
    category: str
    note: str


def journal_result_schema(frame_count: int) -> dict[str, Any]:
    """Strict JSON Schema for one encoded sub-batch."""
    maximum = max(0, frame_count - 1)
    return {
        "type": "object",
        "properties": {
            "entries": {
                "type": "array",
                "description": (
                    "Chronological, distinct meaningful events or materially "
                    "changed scenes; empty when nothing meaningfully changed."
                ),
                "items": {
                    "type": "object",
                    "properties": {
                        "video_second": {
                            "type": "integer",
                            "minimum": 0,
                            "maximum": maximum,
                            "description": (
                                "Video second containing the clearest evidence "
                                "for this entry."
                            ),
                        },
                        "category": {
                            "type": "string",
                            "minLength": 1,
                            "description": (
                                "A concise, loose gameplay category such as "
                                "location, combat, objective, item, or progress."
                            ),
                        },
                        "note": {
                            "type": "string",
                            "minLength": 1,
                            "description": (
                                "One vivid, self-contained scene paragraph using "
                                "concrete visual evidence and explicitly qualified "
                                "uncertainty."
                            ),
                        },
                    },
                    "required": ["video_second", "category", "note"],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["entries"],
        "additionalProperties": False,
    }


def response_format(frame_count: int) -> dict[str, Any]:
    """OpenAI-compatible strict response formatting passed through LiteLLM."""
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "chiron_observer_journal",
            "strict": True,
            "schema": journal_result_schema(frame_count),
        },
    }


def video_part(model_id: str, video: bytes, detail: str) -> dict[str, Any]:
    """Build the provider-specific inline MP4 content part."""
    data_uri = "data:video/mp4;base64," + base64.b64encode(video).decode("ascii")
    if model_id.startswith("openrouter/"):
        return {"type": "video_url", "video_url": {"url": data_uri}}
    return {
        "type": "file",
        "file": {"file_data": data_uri, "mime_type": "video/mp4"},
        "media_resolution": detail,
        "video_metadata": {"fps": 1},
    }


def timestamp_mapping(video: EncodedVideo) -> str:
    """Authoritative video-second to local captured-at mapping."""
    return "\n".join(
        "video_second="
        f"{second}  captured_at="
        f"{datetime.fromtimestamp(frame.captured_at).astimezone().isoformat(timespec='milliseconds')}"
        for second, frame in enumerate(video.frames)
    )


def build_batch_messages(
    settings: Settings,
    video: EncodedVideo,
    journal: JournalSnapshot,
    *,
    detected_game: str = "",
    repair: str = "",
) -> list[dict[str, Any]]:
    """Assemble one independent video review request."""
    instruction = BATCHED_OBSERVER_SYSTEM_PROMPT
    game = settings.game_name.strip()
    if game:
        instruction += f"\nThe player is playing: {game}.\n"
    elif detected_game.strip():
        instruction += (
            "\nThe focused window suggests: "
            f"{detected_game.strip()}. Trust the video if it disagrees.\n"
        )
    if settings.observer_system_prompt.strip():
        instruction += (
            "\nAdditional Observer instructions:\n"
            + settings.observer_system_prompt.strip()
            + "\n"
        )
    journal_text = journal.render() or "(empty)"
    text = (
        "<journal>\n"
        f"{journal_text}\n"
        "</journal>\n\n"
        "The mapping below is authoritative. Select a video_second; do not return "
        "an ISO timestamp.\n"
        f"{timestamp_mapping(video)}\n\n"
        "Return exactly the JSON object required by this schema:\n"
        f"{json.dumps(journal_result_schema(len(video.frames)), separators=(',', ':'))}"
    )
    if repair:
        text += f"\n\nYour previous result was invalid. Repair it once: {repair}"
    return [
        {"role": "system", "content": instruction},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": text},
                video_part(
                    settings.observer_model,
                    video.data,
                    settings.capture.media_resolution,
                ),
            ],
        },
    ]


def response_text(response: Any) -> str:
    """Extract completion text without accepting model prose as application output."""
    try:
        content = response.choices[0].message.content
    except (AttributeError, IndexError, TypeError) as error:
        raise StructuredObserverError("response has no first message") from error
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            str(part.get("text") or "") if isinstance(part, dict) else str(part)
            for part in content
        )
    return str(content or "")


def validate_result(text: str, frame_count: int) -> tuple[StructuredJournalEntry, ...]:
    """Parse and validate the entire response before any journal mutation."""
    source = (text or "").strip()
    try:
        value, end = json.JSONDecoder().raw_decode(source)
    except (json.JSONDecodeError, TypeError) as error:
        raise StructuredObserverError("response is not one JSON value") from error
    if source[end:].strip():
        raise StructuredObserverError("prose outside the JSON object is not allowed")
    if not isinstance(value, dict) or set(value) != {"entries"}:
        raise StructuredObserverError("top level must contain only entries")
    entries = value["entries"]
    if not isinstance(entries, list):
        raise StructuredObserverError("entries must be an array")

    validated: list[StructuredJournalEntry] = []
    previous = -1
    for item in entries:
        if not isinstance(item, dict) or set(item) != {
            "video_second",
            "category",
            "note",
        }:
            raise StructuredObserverError("each entry has an invalid shape")
        second = item["video_second"]
        category = item["category"]
        note = item["note"]
        if isinstance(second, bool) or not isinstance(second, int):
            raise StructuredObserverError("video_second must be an integer")
        if second < 0 or second >= frame_count:
            raise StructuredObserverError("video_second is outside this video")
        if second < previous:
            raise StructuredObserverError("entries must be chronological")
        if not isinstance(category, str) or not category.strip():
            raise StructuredObserverError("category must be a non-empty string")
        if not isinstance(note, str) or not note.strip():
            raise StructuredObserverError("note must be a non-empty string")
        previous = second
        validated.append(StructuredJournalEntry(second, category.strip(), note.strip()))
    return tuple(validated)


__all__ = [
    "BATCHED_OBSERVER_SYSTEM_PROMPT",
    "StructuredJournalEntry",
    "StructuredObserverError",
    "build_batch_messages",
    "journal_result_schema",
    "response_format",
    "response_text",
    "timestamp_mapping",
    "validate_result",
    "video_part",
]
