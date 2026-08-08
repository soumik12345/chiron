"""Buffered Observer scheduling, validation, draining, and journal ownership."""

from __future__ import annotations

import asyncio
import inspect
import json
from collections import deque
from types import SimpleNamespace

from chiron.capture.frames import Frame
from chiron.config.settings import Settings
from chiron.journal.log import JournalLog
from chiron.journal.service import JournalService
from chiron.models.usage import LLMCallRecord
from chiron.observer.factory import build_observer
from chiron.observer.media import EncodedVideo
from chiron.observer.nonlive import NonLiveObserverSessionManager
from chiron.observer.session import LiveObserverSessionManager


def frame(stamp: float) -> Frame:
    return Frame(jpeg=f"jpeg-{stamp}".encode(), captured_at=stamp, width=64, height=32)


def response(payload: dict):
    message = SimpleNamespace(content=json.dumps(payload))
    choice = SimpleNamespace(message=message, finish_reason="stop")
    return SimpleNamespace(choices=[choice], usage=None)


class FakeProvider:
    def __init__(self, outputs):
        self.outputs = deque(outputs)
        self.calls = []
        self.failures = []
        self.max_in_flight = 0
        self.in_flight = 0

    def factory(self, **kwargs):
        provider = self

        class Model:
            async def acompletion(self, **call):
                provider.calls.append(call)
                provider.in_flight += 1
                provider.max_in_flight = max(provider.max_in_flight, provider.in_flight)
                await asyncio.sleep(0)
                provider.in_flight -= 1
                result = provider.outputs.popleft()
                if isinstance(result, BaseException):
                    raise result
                if inspect.isawaitable(result):
                    result = await result
                return result

            def record_usage(self, _usage, *, response=None, context=None):
                kwargs["usage_sink"](
                    LLMCallRecord(
                        model_id=kwargs["model_id"],
                        kind=context["kind"],
                        session_id=kwargs["usage_labels"].get("session_id"),
                        agent_id="observer",
                        run_id=kwargs["usage_labels"].get("run_id"),
                    )
                )

            def record_failure(self, error, *, context=None):
                provider.failures.append((error, context))

        return Model()


def fake_encoder(source, _detail):
    source = tuple(source)
    return [EncodedVideo(b"mp4", source, source[0].width, source[0].height)]


def manager(provider: FakeProvider, *, settings: Settings | None = None):
    journal = JournalService(JournalLog())
    selected = settings or Settings(api_key="google-key")
    observer = NonLiveObserverSessionManager(
        selected,
        journal,
        model_factory=provider.factory,
        encoder=fake_encoder,
    )
    return observer, journal


def test_factory_selects_only_the_observer_transport(qapp):
    journal = JournalService(JournalLog())
    assert isinstance(
        build_observer(Settings(), journal), NonLiveObserverSessionManager
    )
    live = Settings(observer_model="live/gemini-3.1-flash-live-preview")
    assert isinstance(build_observer(live, journal), LiveObserverSessionManager)


async def test_final_flush_maps_entries_to_source_capture_time(qapp):
    provider = FakeProvider(
        [
            response(
                {
                    "entries": [
                        {
                            "video_second": 1,
                            "category": "progress",
                            "note": "Defeated the guardian.",
                        }
                    ]
                }
            )
        ]
    )
    observer, journal = manager(provider)
    runs = []
    delivered = []
    observer.observerRan.connect(runs.append)
    observer.frameSent.connect(lambda item, reason: delivered.append((item, reason)))
    observer.start()
    source = [frame(100.25), frame(105.5)]
    observer.observe(source[0], "scheduled")
    observer.observe(source[1], "immediate")

    await observer.stop()

    assert journal.snapshot().entries[0].timestamp == 105.5
    assert journal.snapshot().entries[0].note == "Defeated the guardian."
    assert delivered == [(source[0], "scheduled"), (source[1], "immediate")]
    assert observer.last_observed_at == 105.5
    assert observer.pending_frames == 0
    assert runs[0]["seal_reason"] == "final_flush"
    assert runs[0]["source_frame_count"] == 2
    assert runs[0]["successful_call_ids"]


async def test_empty_success_advances_freshness_without_journal_write(qapp):
    observer, journal = manager(FakeProvider([response({"entries": []})]))
    observer.start()
    observer.observe(frame(200.0))
    await observer.stop()
    assert observer.last_observed_at == 200.0
    assert len(journal.snapshot().entries) == 0


async def test_malformed_output_is_billed_and_repaired_exactly_once(qapp):
    invalid = response(
        {"entries": [{"video_second": 99, "category": "x", "note": "x"}]}
    )
    valid = response(
        {"entries": [{"video_second": 0, "category": "note", "note": "Valid."}]}
    )
    provider = FakeProvider([invalid, valid])
    observer, journal = manager(provider)
    usage = []
    observer.llmCall.connect(usage.append)
    observer.start()
    observer.observe(frame(300.0))
    await observer.stop()
    assert len(provider.calls) == 2
    assert len(usage) == 2
    assert [entry.note for entry in journal.snapshot().entries] == ["Valid."]
    assert "previous result was invalid" in str(provider.calls[1]["messages"])


async def test_two_invalid_results_retain_batch_and_write_nothing(qapp):
    bad = response({"wrong": []})
    provider = FakeProvider([bad, bad])
    observer, journal = manager(provider)
    errors = []
    observer.errorOccurred.connect(errors.append)
    observer.start()
    observer.observe(frame(400.0))
    await observer.stop()
    assert observer.pending_frames == 1
    assert observer.status == "incompatible"
    assert len(journal.snapshot().entries) == 0
    assert "invalid structured output twice" in errors[-1]
    observer.start()
    await asyncio.sleep(0)
    assert len(provider.calls) == 2, "a retained failure needs an explicit retry"


async def test_local_commit_failure_never_triggers_another_model_completion(qapp):
    provider = FakeProvider(
        [
            response(
                {"entries": [{"video_second": 0, "category": "note", "note": "Once."}]}
            )
        ]
    )
    journal = JournalService(
        JournalLog(), on_entry=lambda _entry: (_ for _ in ()).throw(OSError("disk"))
    )
    observer = NonLiveObserverSessionManager(
        Settings(api_key="google-key"),
        journal,
        model_factory=provider.factory,
        encoder=fake_encoder,
    )
    observer.start()
    observer.observe(frame(450.0))
    await observer.stop()
    assert observer.pending_frames == 1
    assert len(provider.calls) == 1

    journal.on_entry = None
    observer.retry_retained()
    await observer._processing_task
    assert len(provider.calls) == 1
    assert len(journal.snapshot().entries) == 1


async def test_sample_cap_seals_early_and_requests_remain_sequential(qapp):
    provider = FakeProvider([response({"entries": []}), response({"entries": []})])
    observer, _journal = manager(provider)
    observer.start()
    for index in range(300):
        observer.observe(frame(float(index)))
    assert len(observer._sealed) == 1
    assert observer._sealed[0].seal_reason == "sample_cap"
    observer.observe(frame(500.0))
    observer._seal("interval")
    await observer._processing_task
    assert len(provider.calls) == 2
    assert provider.max_in_flight == 1


async def test_detected_game_change_seals_old_prompt_context(qapp):
    provider = FakeProvider([response({"entries": []}), response({"entries": []})])
    observer, _journal = manager(provider)
    observer.detected_game = "Old Game"
    observer.start()
    observer.observe(frame(510.0))
    observer.detected_game = "New Game"
    observer.observe(frame(515.0))
    await observer.stop()
    assert "Old Game" in str(provider.calls[0]["messages"])
    assert "New Game" not in str(provider.calls[0]["messages"])
    assert "New Game" in str(provider.calls[1]["messages"])


async def test_new_watch_during_final_drain_waits_then_restarts_acceptance(qapp):
    release = asyncio.Event()

    async def delayed_response():
        await release.wait()
        return response({"entries": []})

    observer, _journal = manager(FakeProvider([delayed_response()]))
    observer.start()
    observer.observe(frame(600.0))
    observer.begin_drain()
    stopping = asyncio.create_task(observer.stop())
    observer.start()
    assert observer.accepting_frames is False
    assert observer.status == "draining"

    release.set()
    await stopping
    assert observer.accepting_frames is True
    assert observer.status == "watching"
    await observer.stop()


def test_missing_selected_provider_key_blocks_frame_acceptance(qapp):
    observer, _journal = manager(FakeProvider([]), settings=Settings(api_key=""))
    observer.start()
    assert observer.accepting_frames is False
    assert observer.status == "error"
