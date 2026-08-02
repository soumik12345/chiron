# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Chiron is a desktop AI gaming assistant: a frameless, always-on-top PySide6 overlay that watches the player's screen and answers questions about the game in a small text panel. It runs against either the Gemini Live API (a persistent websocket frames stream into) or an ordinary request/response multimodal endpoint via Google AI Studio or OpenRouter.

It also **records what it saw**: every evening of play is a persistent session under `~/.local/share/chiron/sessions/` — journal, transcript, agent traces, frame thumbnails and per-call pricing — browsable and re-readable inside the overlay.

Linux/X11 only (GNOME on Ubuntu is the tested setup) — screen capture (`mss`), global hotkeys (`python-xlib`) and active-window detection are all X11-specific, and degrade to inert rather than crashing on Wayland or a missing display. Requires Python 3.10+.

Design docs, each with a build-time amendment worth reading before trusting the "Decisions" table above it: [docs/00_v0_architecture.md](docs/00_v0_architecture.md) (v0 — the amendment records two premises that turned out to be wrong once real API calls were made), [docs/01_non_live_provider.md](docs/01_non_live_provider.md) (v1 — the non-live provider; its amendment records the resolved deferrals and two premises still untested against real games) and [docs/02_sessions_and_compaction.md](docs/02_sessions_and_compaction.md) (v2 — gameplay sessions and context compaction; its amendment records one design detail that inverted at build time and three premises that need a real long evening to confirm).

## Commands

```bash
uv sync                                    # install
uv run chiron                              # run (also: uv run python -m chiron)
uv run chiron --fresh-install              # wipe saved config + API key, with confirmation
uv run pytest                              # full suite — no network or display needed
uv run pytest tests/test_capture.py -q     # one file
uv run pytest tests/test_novelty.py -q     # observer trigger + novelty detector
uv run pytest tests/test_sessions.py -q    # session store, index, recorder, viewer render
uv run pytest tests/test_compaction.py -q  # both modes' compaction + live cost estimation
uv run pytest tests/test_journal_drawer.py -q  # journal column, unread count, drawer geometry
uv run pytest tests/test_ui.py::test_watch_button_shows_state_and_asks_for_the_other_one  # one test
uv run ruff check chiron/ tests/
uv run ruff format chiron/ tests/
```

Qt tests need `QT_QPA_PLATFORM=offscreen` when there is no real display (CI, this sandbox); `tests/conftest.py` sets it automatically before PySide6 is imported. A shared `qapp` fixture backs every widget test. `asyncio_mode = "auto"` in `pyproject.toml` means `async def test_...` runs directly — no `@pytest.mark.asyncio` needed.

`pyproject.toml`'s `[tool.ruff.lint]` only selects `I` (isort) and `F401` (unused imports) — it is not a general linter here, just import hygiene.

## Architecture

### Two things live in this repo, one of them dormant

`chiron/core/` and `chiron/models/` are a **pre-existing, general-purpose ReAct agent harness** (durable JSONL sessions with branching, a typed event stream, litellm-backed model wrapper with per-call cost accounting, tool router with deferred tools) carried over from a prior project. `chiron/core/` is not wired into the overlay's control flow — don't assume `core/react.py`'s `ReactAgent` drives anything at runtime; it currently doesn't. `chiron/models/` very much is: `LiteLLMModel` is what every non-live call goes through, and its `usage_sink` seam is what feeds v2's cost record.

Note `chiron/core/session.py` (the dormant branching store) versus `chiron/session.py` (the provider seam) versus `chiron/sessions/` (v2's gameplay sessions). Three different things called "session", and only the last two run.

That harness is also why `weave` is a dependency: `LiteLLMModel.acompletion()` (and several `core/` functions) carry a `@weave.op` trace decorator, but `weave.init()` is only ever called from `core/react.py`'s startup path. Since nothing in the overlay's runtime calls into `core/`, those decorators are inert no-ops in the app as it actually runs — don't read `@weave.op`'s presence as evidence that a call is traced anywhere.

### The selected model decides the mode, and nothing else does

`Settings.selected_model` is provider-qualified — `live/<id>`, `gemini/<id>`, `openrouter/<vendor>/<id>` — and `chiron/session.py`'s `build_session_provider()` turns it into either a `LiveSessionManager` or a `NonLiveSessionManager`. There is deliberately **no mode toggle**, so the two can never disagree. v0's `live_model` field is migrated to a `live/…` selection by a `model_validator(mode="before")` on load; `Settings.live_model` survives as a read-only property.

Both providers expose the same surface (`start`, `stop`, `send_text`, `send_frame`, `fold_journal`, `reset_observation`, `apply_settings`, `detected_game`, `frames_sent`) and the same five signals, described as a `Protocol` in `chiron/session.py`. The overlay and `_connect_session()` therefore have no idea which one is running. Status vocabulary is shared too — in non-live mode `live` means *armed*, since there is no socket being held open.

Crossing the boundary replaces the object rather than reopening it, which is why `Settings` has both `requires_session_restart()` and `requires_provider_swap()`. `ChironApp._restart_session(rebuild_writer, swap_provider)` handles both, rebuilds the journal writer (the two modes journal differently), reconnects signals, and carries `detected_game` across.

### The non-live provider inverts the memory architecture

In live mode the model watches and the journal remembers. In non-live mode **the journal watches**: every call starts blank, so context is assembled per request from journal + history + selected frames, and the model only ever sees what the observer distilled plus the frames of the current moment. `fold_journal()` is therefore a no-op there — there is no stateful session to fall behind — and the journal strategy setting (`tool_call`/`sidecar`) applies to live mode only. Non-live always uses `ObserverJournal`, a writer that does nothing on its own; the observer in `chiron/nonlive/session.py` produces the entries and pushes them through `JournalWriter.record()`.

Frames arriving via `send_frame` are *held*, not sent — a bounded ring buffer — because unlike live mode a frame not sent is not a frame the model never sees. Images ride in the current request only; once a turn ages, its image parts become `[frame HH:MM:SS]` placeholders (the frame's own capture time, not "an image was here"), keeping history cost linear in text.

`agent_mode` picks the topology: `unified` puts observer ticks into the same conversation questions use, `split` gives the observer its own near-stateless call against `observer_model`. Both paths share one `asyncio.Lock` when they touch the conversation.

### The observer trigger is where the cost story lives

The v0 scene-change detector (mean absolute consecutive-frame difference ≥ 0.12) is fine for bursting a shutter and structurally wrong for an observer, which pays a full LLM call per firing: swaying grass, water shaders and animated menus hold the plain diff permanently above any usable threshold. `chiron/capture/novelty.py`'s `NoveltyDetector` asks "did pixels change *differently* than they have been changing?" via a per-cell ambient EWMA, weights each cell by `1/(ε + ambient)`, and compares against novelty's own rolling mean/deviation. It reads `Frame.signature`, which is always 32x32 whatever the capture width, so it is independent of frame detail by construction.

`chiron/nonlive/observer.py`'s `ObserverTrigger` layers spikes, a heartbeat (`spike_gated_heartbeat` consults drift and can skip the call entirely; `spike_plain_heartbeat` always fires) and a hard cooldown that is the ceiling on observer spend. Both classes are clock-free — every method takes the current time — like `AdaptiveScheduler`, so the whole policy is testable with a list of timestamps.

The detector resets when watching starts (`ChironApp.set_watching` → `session.reset_observation()`) and when the effective capture width changes, since new thumbnail statistics invalidate what it learned.

### Frame detail means one thing and does two

`capture.media_resolution` is the single stored "Frame detail" value in both modes. In live mode it is the API's `media_resolution` — same pixels, different server-side token budget. In non-live mode there is no such knob, so it resolves to a capture width through `DETAIL_CAPTURE_WIDTH` (512/768/1152) and **supersedes `frame_width`**, which the settings UI hides rather than leaving as a second dial on the same pixels. Always feed the capture thread `Settings.effective_capture()`, never `settings.capture` directly.

### Everything else runs on one event loop

`chiron/app.py:main()` wires a single `qasync.QEventLoop` shared by Qt and asyncio — no thread-bridging signals between them. The one real thread is `chiron/capture/service.py`'s `CaptureService`, which grabs and encodes screenshots off the Qt thread and reports back via Qt signals (automatically queued across the thread boundary). Everything else — the Live session, the journal, the settings window — is coroutines and widgets on the one loop.

`ChironApp` (`chiron/app.py`) is the wiring hub: it owns every component (`journal`, `writer`, `session`, `capture`, `overlay`, `hotkeys`, `recorder`) and every cross-component signal connection lives in `_connect()`. When adding a new interaction, that's the method to extend, not the individual components. The session's own connections are split into `_connect_session()`, because that object is replaced whenever the model selection crosses the live/non-live boundary — the recorder is *not*, and its connections to the provider are remade there too.

### Watching is a state, not a lifecycle — and so is the gameplay session

Chiron launches with the capture thread *running* but not *watching* (`CaptureService._watching` is an `Event`, separate from thread liveness). Nothing is captured, and no Live session opens, until the user starts watching — by hotkey (`ctrl+alt+w` toggles; separate start/stop bindings are also configurable) or the overlay's eye button. `ChironApp.set_watching()` is the single place that starts/stops both capture and the Live session together; anything that flips watching state should go through it rather than poking `capture` or `session` directly. Starting watching also resets the scene-change baseline (`_last_signature`, `scheduler.last_capture`) so resuming after a pause never reads as a false scene change.

A **gameplay session** is a second, slower axis on the same idea. Launch opens none; `ChironApp.ensure_session()` creates one lazily on the first watch-start or first message, so an idle overlay records nothing. `ChironApp.new_session()` is the one canonical reset in the codebase — it closes the record, clears the journal, calls `session.reset_memory()` and empties the transcript — and a fresh launch is simply "no session open yet" rather than a separate implicit reset. Watching and sessions are independent: one session spans many watch spans, and `new_session()` deliberately does not stop watching.

### The session record is an observer, and outlives the provider

`chiron/sessions/` (plural — not `chiron/session.py`) persists an evening of play under `$XDG_DATA_HOME/chiron/sessions/`: one directory per session holding an append-only `events.jsonl` and ~256 px JPEG thumbnails of every frame that reached a model, plus one `index.json` the browser reads so listing never opens an event file. **The data dir, not the config dir**, so `--fresh-install`'s `plan_removal()` cannot reach it by construction; `describe_untouched_sessions()` says so in the plan output.

`SessionRecorder` (`chiron/sessions/recorder.py`) is owned by `ChironApp` and observes it through signals — it never participates, swallows its own failures, buffers appends on a 2 s timer, and survives a live/non-live swap exactly as the journal does. Four provider signals exist for it (`frameSent`, `observerRan`, `llmCall`, `compacted`) alongside the five the overlay already used; `SessionProvider` describes all nine, and each mode simply never emits the ones it has no opinion about.

Two invariants to preserve when touching this: the recorder's incremental counters must match what `chiron/sessions/store.py`'s `summarise()` computes by rescan (the index is a *cache* of the stream, and `SessionIndex.rebuild()` proves it), and thumbnails are memoised by capture-time-plus-shape rather than by `id(frame)` — a freed frame's address gets reused, which silently deduplicates distinct captures.

### Cost is measured in non-live mode and estimated in live mode

`LiteLLMModel.usage_sink` is the seam, and it is now actually installed: `NonLiveSessionManager._model()` attaches it to every throwaway wrapper it builds, and `build_journal_writer(..., usage_sink=…)` attaches it to the live sidecar's long-lived one. Every call lands as an `llm_call` event and rolls up per model and per kind.

The Live API reports nothing billable, so `chiron/live/estimate.py` produces `pricing_source="estimated"` rows from frames × per-frame token cost + transcript length against a hand-kept rate table. **Every surface that renders a total checks that flag and prefixes `~`** (`CostRollup.render()`, the overlay footer, the viewer's summary line). Adding a new place that shows money means honouring it too.

Adding a call kind means adding it to the `LLMCallKind` Literal in `chiron/models/usage.py`. That Literal used to hold only `turn` and `compaction` while the app passed `nonlive_qa` and `journal_sidecar`, so every record raised on construction and died inside the try/except that wraps usage accounting — nothing was recorded and nothing said so.

### Compaction is deliberate in both modes, and asymmetric

Non-live (`chiron/nonlive/compaction.py`, orchestrated by `NonLiveSessionManager._call_with_history`): deduplicate, then measure, then retry. The journal block is stripped as a turn enters history (`_remember(..., history_text=…)`) because it is prepended to every question and twenty exchanges used to carry twenty copies. Then `should_compact()` prices the assembled request against the model's window before each call and summarises above 85%; `is_context_overflow()` catches litellm's typed `ContextWindowExceededError` as a backstop, compacts, and retries once. Token counting projects image parts out first — a data URI is text to a token counter, and enormous.

Live (`LiveSessionManager._compact_and_rotate`): the context is server-side and immutable, so the equivalent is a sidecar pass + forced journal fold, then `rotate("compaction", fresh=True)`. **The `fresh=True` is the whole point** — it drops the resumption handle, which would otherwise faithfully restore the context being shed. Every other caller of `rotate()` wants the handle kept. Gated on a quiet moment (no pending question, no novelty spike for 4 s) with a hard deadline so gating can never postpone into eviction; sliding-window compression stays on underneath as the backstop.

### The game is detected before watching starts, not when

`chiron/capture/active_window.py`'s `ActiveWindowTracker` polls X11 from app launch for the focused window, always excluding Chiron's own window ids — so when the eye button steals focus to the overlay, `tracker.current` still points at the game. Identity resolution prefers Steam (the `STEAM_GAME` window property or the process's `SteamAppId` env var, appid → name via local `appmanifest_*.acf`) and falls back to title/`WM_CLASS`/process name. The result feeds the session provider's `detected_game` (runtime state, deliberately not a `Settings` field, and carried across a mode swap by hand), which `build_system_instruction()` — and its non-live counterpart in `chiron/nonlive/prompts.py` — uses **only when** the user's `game_name` is empty — a typed Game always wins. Mid-watch application switches are journaled (the running session's instruction is fixed; the fold is how it learns). `WindowInfo.identity` keys on appid/class, not title, so title churn never emits a change. On Wayland or no display, the tracker stays inert and everything behaves as before.

### The Live API has retired the model class this was designed around

`docs/00_v0_architecture.md`'s original plan assumed a text-out ("half-cascade") Live model with a 32k context. Those models (`gemini-live-2.5-flash-preview`, `gemini-2.0-flash-live-001`) now 404 — every served Live model is native-audio and only accepts `response_modalities=[AUDIO]`. Two consequences baked into `chiron/live/session.py`:

- Connections request `AUDIO` + `output_audio_transcription`; the overlay renders the model's own transcript of its speech. `_text_parts()` reads both `output_transcription` chunks and (for future text-capable models) `model_turn` text parts — never `inline_data` audio bytes, which are discarded.
- `session.receive()` in the `google-genai` SDK covers **exactly one model turn** and stops when it completes. `_receive()` wraps `_receive_turn()` in a loop so a long-lived session survives past its first answer; treating the iterator's end as a disconnect (the natural first instinct) reconnects after every single reply.

Context window is 128k (native-audio), not the 32k the doc assumed — `LIVE_CONTEXT_TOKENS` in `chiron/config/settings.py` is the source of truth, referenced by the settings page's token-burn estimate.

Session errors are split into permanent vs. transient (`is_permanent_error()` in `chiron/live/session.py`) — a bad key or unsupported modality stops the reconnect loop instead of retrying forever with the same doomed config.

### The journal is the memory layer, in interchangeable strategies

`chiron/journal/log.py`'s `JournalLog` is a plain timestamped append-only list. `chiron/journal/writers.py` provides three `JournalWriter` implementations behind one interface. Two are live-mode strategies selected by `settings.journal.strategy`: `ToolCallJournal` (the live model gets a `record_event` function) and `SidecarJournal` (a separate litellm call summarises the transcript + kept frames on a timer). The third, `ObserverJournal`, is what `build_journal_writer(..., live=False)` returns — it does nothing on its own, because in non-live mode the observer already is the journal writer; what survives is `record()`, the shared route to the log and the overlay callback.

`LiveSessionManager.fold_journal()` periodically pushes new entries back into the live session as plain text — this is what lets facts outlive the frames the sliding-window compression evicts, and it's also the reconnect-seed mechanism (`_seed()` replays the recent journal into a freshly opened session). The non-live implementation returns 0; see the memory-inversion section above.

Changing `journal.strategy` at runtime requires tearing down and rebuilding the writer (`ChironApp._restart_session(rebuild_writer=True)`), since the two live strategies install different things into the Live session config (function declarations vs. nothing). A mode swap forces the same rebuild.

### The overlay is a two-view stack plus a drawer, not a transcript

`chiron/ui/overlay.py` holds a `QStackedWidget` with two pages — `PLAY_VIEW` (today's transcript) and `SESSION_VIEW` (one recorded session read back, in `chiron/ui/history_view.py`). Getting *to* a session is not a page: `SessionPicker` is a `Qt.Popup` that drops under the 🕘 button, so choosing an evening costs no panel resize. Navigation is a small back-stack: `overlay.back()` walks one step, `Escape` calls it and only hides the panel once already on play, and `start_response()` snaps to play so an answer arriving mid-browse is never missed. Managing a session (rename, delete, delete-thumbnails) lives in the viewer's `⋯` menu — on the thing you are looking at, not on a row you are skimming — and the three signals still surface on `OverlayWindow` unchanged, so `app.py`'s wiring did not move.

**Two independent things resize the window, and neither may reach the saved width.** Entering the viewer temporarily enlarges the panel and stores the pre-review size in `_play_size`; opening the journal drawer widens it by `journal_width + BODY_SPACING`. `current_geometry()` undoes both — it reports `_play_size` when set, minus `_journal_offset()` — so `OverlaySettings.width` always means the un-drawered play width. Anything that changes the drawer while `_play_size` is set must move `_play_size` by the same delta, or closing the drawer mid-review reopens a gap on the way back to play.

### The journal is a column, and a closed column still counts

Journal entries used to be dimmed `✎` lines inline in the transcript. They now live in `chiron/ui/journal_drawer.py`'s `JournalDrawer`, a collapsible right-hand column, newest first, category-coloured via `JOURNAL_CATEGORY_COLOURS` (an unrecognised category falls back to dim text rather than being dropped — categories are a hint to the model, not a schema). Rendering is pure like `chiron/sessions/render.py`: `render_entries()` takes entries and returns HTML, assertable with no display.

The column is a *surface* — raised background, border, radius — not a `border-left` hairline; the first version was the hairline, and it read as leftover space with text floating in it. Two QTextDocument quirks are load-bearing in the render and will look like sloppy spacing if undone: a paragraph's bottom margin is dropped before a following `<table>`, so entries are separated by a blank `GAP_POINTS` paragraph rather than a margin, and `font-size` on a `<p>` is unreliable (which is why the empty state carries no scaled-up watermark glyph). A plain `QWidget` also paints no stylesheet background or border without `WA_StyledBackground`.

The drawer sits *outside* the view stack, which is what lets it stay open while a recorded session is read. Moving the entries out of the transcript removes the only signal that the journal was being written, so the toggle carries an unread count (`✎ 3`) that clears on open — `_refresh_journal_button()` also has to `unpolish`/`polish` the button, since Qt does not re-evaluate the `[unread="true"]` property selector on its own.

Two things are easy to get wrong here. `ChironApp._on_active_window` appends to the log *directly* rather than through `JournalWriter.record()`, so it is the one entry `on_entry` never sees — it is handed to the drawer by hand, or the drawer's count and the footer's `journal: N` disagree. And the drawer's open state is runtime state the *overlay* owns, persisted in `OverlaySettings.journal_open`: `SettingsWindow.collect()` builds a fresh `Settings` with no field for it, so `ChironApp.apply_settings()` copies it (and `journal_width`) off the previous object exactly as it does `position_x`/`position_y`, or saving settings would quietly close the drawer.

Rendering a recorded session is pure: `chiron/sessions/render.py` turns events into HTML with thumbnails inline (`<img src="…">` at absolute paths), so the viewer's whole output is assertable without a display. A thumbnail that has been deleted to reclaim disk renders as a dim placeholder, never a broken image.

### Settings: edit-a-copy, whole-object apply

`chiron/ui/settings_window.py`'s `SettingsWindow` never mutates the live `Settings` in place. Widgets are populated from a snapshot (`load()`), `collect()` reads the whole form back into a fresh `Settings`, and only `Save` hands that object to `ChironApp.apply_settings()`. The one exception is overlay appearance (opacity/size/font), which previews live via a separate `appearanceChanged` signal, independent of Save. `Settings.requires_session_restart()` decides whether an edit needs the session torn down and reopened (model, either key, agent mode, observer model, instructions, journal strategy, media resolution) versus applying for free (everything else — including every observer cadence number, which is re-read per tick) — extend that method, don't special-case call sites, when adding a new session-affecting setting.

Because one dropdown decides the mode, `_refresh_derived()` also does mode-dependent enabling: observer settings are disabled with a reason under a live model, agent mode likewise, `frame_width` is hidden under a non-live one, and the token-burn estimate changes meaning entirely (context-eviction minutes in live mode; per-look cost and a cooldown-capped ceiling in non-live). The model list comes from `chiron/models/catalogue.py`'s `available_models()`, fetched off-thread when the window opens — a provider with no key is simply absent, a failed fetch falls back to a 24h disk cache and then to `STATIC_MODELS`. `chiron/ui/model_picker.py`'s `ModelPicker` is a purpose-built control rather than a combo box, because several hundred entries with a provider, an id, a mode and two prices each will not fit on one line of item text. The field is a card (name and id left, provider and prices right) and the popup is search + provider chips + a delegate-painted list. Its value is always a plain model id: a search term containing `/` that matches nothing becomes an offered row, so an id no catalogue lists can still be pasted in. `set_selection()` is the silent programmatic path used by `load()`; `choose()` is the user path and emits `selectionChanged`.

`chiron/config/settings.py` is also where `--fresh-install` lives (`plan_removal()` / `remove_configuration()`): it only ever considers deleting a directory literally named `chiron`, so pointing `--settings` at a file in some other shared directory can't put that directory up for deletion.

### Hotkeys: X11-native, not pynput

`chiron/ui/hotkeys.py` grabs global hotkeys via `python-xlib` (`XGrabKey`) rather than `pynput` — `pynput` pulls in `evdev` on Linux, which has no prebuilt wheels and needs a C toolchain to compile. `pynput` is used automatically as a fallback backend if it happens to be importable, but is not a project dependency. Both backends live behind `GlobalHotkeyManager`, which is the only thing `app.py` talks to.
