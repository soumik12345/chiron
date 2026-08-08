"""Opt-in real-provider checks for assumptions offline fakes cannot prove.

Run with ``CHIRON_RUN_PROVIDER_SMOKE=1 GEMINI_API_KEY=... uv run pytest
tests/test_provider_smoke.py``. OpenRouter ReAct additionally needs
``OPENROUTER_API_KEY`` and ``CHIRON_OPENROUTER_SMOKE_MODEL``.
"""

from __future__ import annotations

import asyncio
import os

import pytest
from PIL import Image, ImageDraw

from chiron.capture.frames import encode_frame
from chiron.config.settings import Settings, live_model_id
from chiron.journal.log import JournalLog
from chiron.journal.service import JournalService
from chiron.observer.nonlive import NonLiveObserverSessionManager
from chiron.observer.prompts import CHECKPOINT_INSTRUCTION
from chiron.observer.session import ObserverSessionManager
from chiron.responder.session import ResponderSessionManager
from chiron.session import ObserverStatus

RUN_SMOKE = os.environ.get("CHIRON_RUN_PROVIDER_SMOKE") == "1"
GOOGLE_KEY = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")

pytestmark = pytest.mark.skipif(
    not RUN_SMOKE,
    reason="set CHIRON_RUN_PROVIDER_SMOKE=1 to run provider smoke tests",
)
google_smoke = pytest.mark.skipif(
    not GOOGLE_KEY, reason="set GEMINI_API_KEY for Google provider smoke tests"
)


def screenshot(text: str):
    image = Image.new("RGB", (640, 360), (20, 24, 32))
    ImageDraw.Draw(image).text((24, 24), text, fill=(255, 255, 255))
    return encode_frame(image, width=640, stamp=False)


def live_status() -> ObserverStatus:
    return ObserverStatus(
        state="live", watch_requested=True, last_observed_at=1.0, detail="smoke"
    )


async def responder_answer(settings: Settings, question: str):
    journal = JournalService(JournalLog())
    journal.record("The player reached a locked stone gate.", "progress")
    manager = ResponderSessionManager(settings, journal.reader())
    answers, errors = [], []
    manager.responseCompleted.connect(answers.append)
    manager.errorOccurred.connect(errors.append)
    manager.ask(question, screenshot("LOCKED STONE GATE"), live_status())
    await asyncio.wait_for(manager._worker, timeout=90)
    assert not errors, errors
    assert answers and answers[0].strip()
    return answers[0]


@google_smoke
async def test_gemini_36_fixed_horizon_with_screenshot():
    settings = Settings(api_key=GOOGLE_KEY or "")
    answer = await responder_answer(settings, "Briefly describe what I reached.")
    assert isinstance(answer, str)


@google_smoke
async def test_gemini_36_react_forces_read_journal():
    settings = Settings(api_key=GOOGLE_KEY or "", responder_mode="react")
    answer = await responder_answer(settings, "What did I reach?")
    assert isinstance(answer, str)


async def test_openrouter_react_with_one_known_tool_model():
    model = os.environ.get("CHIRON_OPENROUTER_SMOKE_MODEL", "")
    key = os.environ.get("OPENROUTER_API_KEY", "")
    if not model or not key:
        pytest.skip("set OPENROUTER_API_KEY and CHIRON_OPENROUTER_SMOKE_MODEL")
    settings = Settings(
        api_key=GOOGLE_KEY or "",
        openrouter_api_key=key,
        responder_model=(
            model if model.startswith("openrouter/") else f"openrouter/{model}"
        ),
        responder_mode="react",
    )
    answer = await responder_answer(settings, "What did I reach?")
    assert isinstance(answer, str)


async def _live_checkpoint(text: str) -> tuple[int, int]:
    from google import genai
    from google.genai import types

    settings = Settings(
        api_key=GOOGLE_KEY or "",
        observer_model="live/gemini-3.1-flash-live-preview",
    )
    journal = JournalService(JournalLog())
    observer = ObserverSessionManager(settings, journal)
    client = genai.Client(api_key=GOOGLE_KEY)
    calls = 0
    completed = 0
    async with client.aio.live.connect(
        model=live_model_id(settings.observer_model),
        config=observer._build_config(),
    ) as session:
        current = screenshot(text)
        await session.send_realtime_input(activity_start=types.ActivityStart())
        await session.send_realtime_input(
            video=types.Blob(data=current.jpeg, mime_type="image/jpeg")
        )
        await session.send_realtime_input(text=CHECKPOINT_INSTRUCTION)
        await session.send_realtime_input(activity_end=types.ActivityEnd())

        async def receive():
            nonlocal calls, completed
            for _ in range(3):
                async for message in session.receive():
                    tool_call = getattr(message, "tool_call", None)
                    if tool_call is not None:
                        calls += len(tool_call.function_calls or [])
                        await observer._handle_tool_call(session, tool_call)
                    content = getattr(message, "server_content", None)
                    if content is not None and getattr(content, "turn_complete", False):
                        completed += 1
                        return

        await asyncio.wait_for(receive(), timeout=45)
    return calls, completed


@google_smoke
async def test_live_manual_video_checkpoint_can_call_record_event():
    calls, completed = await _live_checkpoint(
        "QUEST COMPLETE: Restored power to the named Moon Tower"
    )
    assert completed == 1
    assert calls >= 1


@google_smoke
async def test_live_no_event_checkpoint_finishes_without_visible_output():
    calls, completed = await _live_checkpoint("ordinary unchanged pause menu")
    assert completed == 1
    assert calls == 0


@google_smoke
async def test_live_fresh_connection_accepts_a_journal_seed():
    from google import genai

    settings = Settings(
        api_key=GOOGLE_KEY or "",
        observer_model="live/gemini-3.1-flash-live-preview",
    )
    journal = JournalService(JournalLog())
    journal.record("Reached Moon Tower.", "location")
    observer = ObserverSessionManager(settings, journal)
    client = genai.Client(api_key=GOOGLE_KEY)
    async with client.aio.live.connect(
        model=live_model_id(settings.observer_model),
        config=observer._build_config(),
    ) as session:
        await asyncio.wait_for(observer._seed(session), timeout=30)
    observer._resumption_handle = "discard-me"
    observer.rotate("smoke fresh rotation", fresh=True)
    assert observer._resumption_handle is None


async def _batched_observer_smoke(settings: Settings, qapp) -> None:
    journal = JournalService(JournalLog())
    observer = NonLiveObserverSessionManager(settings, journal)
    calls, runs, errors = [], [], []
    observer.llmCall.connect(calls.append)
    observer.observerRan.connect(runs.append)
    observer.errorOccurred.connect(errors.append)
    first = screenshot("ENTERED NAMED LOCATION: Moon Tower")
    second = screenshot("QUEST COMPLETE: Restored power to Moon Tower")
    observer.start()
    observer.observe(first)
    observer.observe(second)
    await observer.stop(timeout=90)
    assert not errors, errors
    assert observer.last_observed_at == second.captured_at
    assert runs and runs[0]["source_frame_count"] == 2
    assert calls and all(call.kind == "nonlive_observer" for call in calls)
    assert all(
        entry.timestamp in {first.captured_at, second.captured_at}
        for entry in journal.snapshot().entries
    )


@google_smoke
async def test_google_batched_video_schema_timestamp_and_usage(qapp):
    await _batched_observer_smoke(Settings(api_key=GOOGLE_KEY or ""), qapp)


async def test_openrouter_batched_video_schema_timestamp_and_usage(qapp):
    model = os.environ.get("CHIRON_OPENROUTER_VIDEO_SMOKE_MODEL", "")
    key = os.environ.get("OPENROUTER_API_KEY", "")
    if not model or not key:
        pytest.skip("set OPENROUTER_API_KEY and CHIRON_OPENROUTER_VIDEO_SMOKE_MODEL")
    await _batched_observer_smoke(
        Settings(
            openrouter_api_key=key,
            observer_model=(
                model if model.startswith("openrouter/") else f"openrouter/{model}"
            ),
            responder_model="openrouter/google/gemini-2.5-flash",
        ),
        qapp,
    )
