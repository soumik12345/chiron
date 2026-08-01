# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Chiron is a desktop AI gaming assistant: a frameless, always-on-top PySide6 overlay that watches the player's screen via the Google Gemini Live API and answers questions about the game in a small text panel. See [docs/00_v0_architecture.md](docs/00_v0_architecture.md) for the original design doc and its build-time amendment (the amendment records two premises that turned out to be wrong once real API calls were made — read it before trusting the "Decisions" table above it).

## Commands

```bash
uv sync                                    # install
uv run chiron                              # run (also: uv run python -m chiron)
uv run chiron --fresh-install              # wipe saved config + API key, with confirmation
uv run pytest                              # full suite — no network or display needed
uv run pytest tests/test_capture.py -q     # one file
uv run pytest tests/test_ui.py::test_watch_button_shows_state_and_asks_for_the_other_one  # one test
uv run ruff check chiron/ tests/
uv run ruff format chiron/ tests/
```

Qt tests need `QT_QPA_PLATFORM=offscreen` when there is no real display (CI, this sandbox); `tests/conftest.py` sets it automatically before PySide6 is imported. A shared `qapp` fixture backs every widget test.

`pyproject.toml`'s `[tool.ruff.lint]` only selects `I` (isort) and `F401` (unused imports) — it is not a general linter here, just import hygiene.

## Architecture

### Two things live in this repo, one of them dormant

`chiron/core/` and `chiron/models/` are a **pre-existing, general-purpose ReAct agent harness** (durable JSONL sessions with branching, a typed event stream, litellm-backed model wrapper with per-call cost accounting, tool router with deferred tools) carried over from a prior project. It is not wired into the overlay's control flow. The only load-bearing connection is `chiron/journal/writers.py`'s `SidecarJournal`, which reuses `chiron.models.litellm_model.LiteLLMModel` for its periodic summarisation calls — that's why journal usage lands in the same litellm plumbing as everything else. Don't assume `core/react.py`'s `ReactAgent` drives anything at runtime; it currently doesn't.

### Everything else runs on one event loop

`chiron/app.py:main()` wires a single `qasync.QEventLoop` shared by Qt and asyncio — no thread-bridging signals between them. The one real thread is `chiron/capture/service.py`'s `CaptureService`, which grabs and encodes screenshots off the Qt thread and reports back via Qt signals (automatically queued across the thread boundary). Everything else — the Live session, the journal, the settings window — is coroutines and widgets on the one loop.

`ChironApp` (`chiron/app.py`) is the wiring hub: it owns every component (`journal`, `writer`, `session`, `capture`, `overlay`, `hotkeys`) and every cross-component signal connection lives in `_connect()`. When adding a new interaction, that's the method to extend, not the individual components.

### Watching is a state, not a lifecycle

Chiron launches with the capture thread *running* but not *watching* (`CaptureService._watching` is an `Event`, separate from thread liveness). Nothing is captured, and no Live session opens, until the user starts watching — by hotkey (`ctrl+alt+w` toggles; separate start/stop bindings are also configurable) or the overlay's eye button. `ChironApp.set_watching()` is the single place that starts/stops both capture and the Live session together; anything that flips watching state should go through it rather than poking `capture` or `session` directly. Starting watching also resets the scene-change baseline (`_last_signature`, `scheduler.last_capture`) so resuming after a pause never reads as a false scene change.

### The Live API has retired the model class this was designed around

`docs/00_v0_architecture.md`'s original plan assumed a text-out ("half-cascade") Live model with a 32k context. Those models (`gemini-live-2.5-flash-preview`, `gemini-2.0-flash-live-001`) now 404 — every served Live model is native-audio and only accepts `response_modalities=[AUDIO]`. Two consequences baked into `chiron/live/session.py`:

- Connections request `AUDIO` + `output_audio_transcription`; the overlay renders the model's own transcript of its speech. `_text_parts()` reads both `output_transcription` chunks and (for future text-capable models) `model_turn` text parts — never `inline_data` audio bytes, which are discarded.
- `session.receive()` in the `google-genai` SDK covers **exactly one model turn** and stops when it completes. `_receive()` wraps `_receive_turn()` in a loop so a long-lived session survives past its first answer; treating the iterator's end as a disconnect (the natural first instinct) reconnects after every single reply.

Context window is 128k (native-audio), not the 32k the doc assumed — `LIVE_CONTEXT_TOKENS` in `chiron/config/settings.py` is the source of truth, referenced by the settings page's token-burn estimate.

Session errors are split into permanent vs. transient (`is_permanent_error()` in `chiron/live/session.py`) — a bad key or unsupported modality stops the reconnect loop instead of retrying forever with the same doomed config.

### The journal is the memory layer, in two interchangeable strategies

`chiron/journal/log.py`'s `JournalLog` is a plain timestamped append-only list. `chiron/journal/writers.py` provides two `JournalWriter` implementations behind one interface, selected by `settings.journal.strategy`: `ToolCallJournal` (the live model gets a `record_event` function) and `SidecarJournal` (a separate litellm call summarises the transcript + kept frames on a timer). `LiveSessionManager.fold_journal()` periodically pushes new entries back into the live session as plain text — this is what lets facts outlive the frames the sliding-window compression evicts, and it's also the reconnect-seed mechanism (`_seed()` replays the recent journal into a freshly opened session).

Changing `journal.strategy` at runtime requires tearing down and rebuilding the writer (`ChironApp._restart_session(rebuild_writer=True)`), since the two strategies install different things into the Live session config (function declarations vs. nothing).

### Settings: edit-a-copy, whole-object apply

`chiron/ui/settings_window.py`'s `SettingsWindow` never mutates the live `Settings` in place. Widgets are populated from a snapshot (`load()`), `collect()` reads the whole form back into a fresh `Settings`, and only `Save` hands that object to `ChironApp.apply_settings()`. The one exception is overlay appearance (opacity/size/font), which previews live via a separate `appearanceChanged` signal, independent of Save. `Settings.requires_session_restart()` decides whether an edit needs the Live session torn down and reopened (model, key, instructions, journal strategy, media resolution) versus applying for free (everything else) — extend that method, don't special-case call sites, when adding a new session-affecting setting.

`chiron/config/settings.py` is also where `--fresh-install` lives (`plan_removal()` / `remove_configuration()`): it only ever considers deleting a directory literally named `chiron`, so pointing `--settings` at a file in some other shared directory can't put that directory up for deletion.

### Hotkeys: X11-native, not pynput

`chiron/ui/hotkeys.py` grabs global hotkeys via `python-xlib` (`XGrabKey`) rather than `pynput` — `pynput` pulls in `evdev` on Linux, which has no prebuilt wheels and needs a C toolchain to compile. `pynput` is used automatically as a fallback backend if it happens to be importable, but is not a project dependency. Both backends live behind `GlobalHotkeyManager`, which is the only thing `app.py` talks to.
