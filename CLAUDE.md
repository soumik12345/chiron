# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Chiron is a desktop AI gaming assistant: a frameless, always-on-top PySide6 overlay with two permanent agents. **Chiron-Observer** is a tool-only Gemini Live connection that receives scheduled screenshots and may only write durable facts to the journal. **Chiron-Responder** is an ordinary request/response multimodal agent that produces every user-visible answer through Google AI Studio or, optionally, OpenRouter. A Google API key is therefore required even when the Responder uses OpenRouter.

It also **records what it saw**: every evening of play is a persistent session under `~/.local/share/chiron/sessions/` — journal, transcript, agent traces, frame thumbnails and per-call pricing — browsable and re-readable inside the overlay.

Linux/X11 only (GNOME on Ubuntu is the tested setup) — screen capture (`mss`), global hotkeys (`python-xlib`) and active-window detection are all X11-specific, and degrade to inert rather than crashing on Wayland or a missing display. Requires Python 3.10+.

Design docs, each with a build-time amendment worth reading before trusting the "Decisions" table above it: [docs/00_v0_architecture.md](docs/00_v0_architecture.md) (v0 Live overlay), [docs/01_non_live_provider.md](docs/01_non_live_provider.md) (v1 alternative provider), [docs/02_sessions_and_compaction.md](docs/02_sessions_and_compaction.md) (v2 gameplay sessions and compaction), and [docs/03_dual_agent_architecture.md](docs/03_dual_agent_architecture.md) (v3 current architecture). The v3 amendment records the completed ownership split and the real-provider assumptions that remain unverified without credentials.

## Commands

```bash
uv sync                                    # install
uv run chiron                              # run (also: uv run python -m chiron)
uv run chiron --fresh-install              # wipe saved config + API key, with confirmation
uv run pytest                              # full suite — no network or display needed
uv run pytest tests/test_capture.py -q     # fixed scheduler + immediate capture
uv run pytest tests/test_observer.py -q    # tool-only Live Observer
uv run pytest tests/test_responder.py -q   # fixed-horizon + ReAct Responder
uv run pytest tests/test_journal_compaction.py -q  # automatic journal summary/tail
uv run pytest tests/test_sessions.py -q    # session store, index, recorder, viewer render
uv run pytest tests/test_compaction.py -q  # Responder context measurement + compaction
uv run pytest tests/test_journal_drawer.py -q  # journal column, unread count, drawer geometry
uv run pytest tests/test_ui.py::test_watch_button_shows_state_and_asks_for_the_other_one  # one test
uv run ruff check chiron/ tests/
uv run ruff format --check chiron/ tests/
```

Qt tests need `QT_QPA_PLATFORM=offscreen` when there is no real display (CI, this sandbox); `tests/conftest.py` sets it automatically before PySide6 is imported. A shared `qapp` fixture backs every widget test. `asyncio_mode = "auto"` in `pyproject.toml` means `async def test_...` runs directly — no `@pytest.mark.asyncio` needed.

`pyproject.toml`'s `[tool.ruff.lint]` only selects `I` (isort) and `F401` (unused imports) — it is not a general linter here, just import hygiene.

## Architecture

### The carried-over ReAct harness is now partly live

`chiron/core/` and `chiron/models/` came from a general-purpose ReAct harness. v3 now uses `core/react.py`'s loop, typed events, router and tool base when `Settings.responder_mode == "react"`; `chiron/responder/session.py` adapts it with multimodal input, a forced first tool call and a completion guard. The branching `chiron/core/session.py` store is still dormant and is **not** gameplay memory. `LiteLLMModel` backs every Responder and journal-compaction call, and its `usage_sink` feeds the gameplay cost ledger.

Three similarly named paths still mean different things: `chiron/core/session.py` is the unused branching harness store, `chiron/session.py` contains the narrow Observer/Responder contracts plus `ObserverStatus`, and `chiron/sessions/` owns durable gameplay recordings. `chiron/nonlive/compaction.py` also survives only as a pure context-measurement helper; there is no longer a non-live runtime provider.

`weave` remains a harness dependency and several functions carry `@weave.op`, but the overlay does not call the harness startup path that runs `weave.init()`. Do not treat those decorators as evidence that gameplay calls are externally traced; v3 ReAct diagnostics instead flow through `ResponderSessionManager.agentTrace` into the gameplay recorder.

### Both agents always exist, with deliberately different contracts

`ChironApp` constructs `ObserverSessionManager` and `ResponderSessionManager` together and keeps both for the application's lifetime. The old `SessionProvider` protocol, `build_session_provider()` factory and `selected_model` mode switch are gone. `chiron/session.py` defines two narrow protocols so an Observer cannot accidentally acquire an answer surface and a Responder cannot acquire the journal write handler.

`Settings.observer_model` must be `live/<id>` and always uses the required Google key. `Settings.responder_model` must be `gemini/<id>` or `openrouter/<vendor>/<id>`; OpenRouter is optional and never replaces the Live Observer. `Settings.responder_mode` selects `fixed_horizon` or `react`, not a provider topology. v0-v2 settings are migrated one way in `Settings._migrate_dual_agents()` and only the v3 shape is saved.

Only the Observer owns a persistent socket; only the Responder emits `responseStarted`, `responseDelta` and `responseCompleted`. `ChironApp._connect_observer()` and `_connect_responder()` wire those distinct signals explicitly. A Responder rebuild preserves the application-owned conversation and never stops or swaps the Observer.

### The Observer writes; the Responder answers

`chiron/observer/session.py` is a Gemini Live client with exactly one declared function, `record_event`. It requests the native-audio model's required `AUDIO` modality but exposes no transcription or response signal; all model audio/content is discarded. Each checkpoint is one send-locked transaction: `activity_start`, one JPEG video frame, `CHECKPOINT_INSTRUCTION`, then `activity_end`. Tool calls go through `JournalService.handle_observer_tool()` and are acknowledged so the turn can finish. If another frame arrives mid-turn, `_pending_frame` retains only the newest one instead of building an unbounded queue.

`chiron/responder/session.py` serializes questions FIFO and produces every visible answer through `LiteLLMModel`. Fixed-horizon mode injects Observer freshness, the current journal view, shared conversation and optional current frame into one normal completion. ReAct creates an ephemeral `ReactAgent` whose only tool is read-only `read_journal`; the first tool call is forced and a completion guard rejects a final answer until the read succeeds. Intermediate tool protocol and traces are recorded but never committed to the transcript.

Both modes share `ResponderConversation`: its canonical record contains only user messages and final answers. Current images become `[frame HH:MM:SS]` placeholders at commit time and never enter durable conversation history. Switching mode or model preserves canonical messages and the active summary; only New Session clears them.

### Capture is fixed, and frame detail is split by consumer

Client-side scene-change, novelty, spike, drift, heartbeat, cooldown and burst logic has been deleted. `FixedIntervalScheduler` has one deadline, defaults to five seconds, and advances from the actual capture time without catching up missed ticks. Every scheduled frame is an Observer checkpoint; semantic filtering belongs to the Observer prompt.

`capture.frame_width` always controls the JPEG dimensions delivered to both agents. `capture.media_resolution` independently controls the Live Observer's server-side visual token budget. Feed the capture thread `Settings.effective_capture()`; a width change invalidates cached frames, while a media-resolution change requires an Observer reconnect.

`capture.question_frame_policy` is `latest` or `immediate`. `latest` attaches the most recent frame from the current effective Watch span. `immediate` captures exactly one extra frame, sends the same object to both agents, and does not move the periodic deadline. Neither policy may reuse a frame after Watch stops or the Observer disconnects.

### Everything else runs on one event loop

`chiron/app.py:main()` wires a single `qasync.QEventLoop` shared by Qt and asyncio. The one worker thread is `CaptureService`, which grabs and encodes screenshots and reports back through queued Qt signals. The Observer socket, Responder queue, compaction calls, recorder and UI all run on the shared Qt/asyncio loop.

`ChironApp` is the wiring hub and owns `journal`, `journal_service`, `journal_compactor`, `conversation`, `observer`, `responder`, `capture`, `overlay`, `hotkeys` and `recorder`. Cross-component connections belong in `_connect()`, `_connect_observer()` or `_connect_responder()`. The Observer object reconnects in place; a Responder-affecting settings change replaces only the Responder and reconnects its signals while reusing the application-owned conversation and journal reader.

### Requested Watch and effective capture are different states

The capture thread starts at launch but is inert. `watch_requested` is the user's desired state; `capture_active` is true only while Watch is requested **and** `observer.status == "live"`. `ChironApp.set_watching()` is the single user-facing transition: Watch-on opens the Observer first, and `_on_observer_status()` activates capture only after connection. Watch-off stops capture, invalidates cached/pending frames, then closes the socket. A transient or permanent Observer failure performs the same capture stop and invalidation while leaving Watch requested for bounded reconnects.

The Responder remains usable with Watch off or during an Observer outage, but receives an `ObserverStatus` snapshot marked stale and no screenshot. Do not weaken this privacy/freshness boundary by reading `capture.is_running`, reusing `_latest_frame`, or starting capture before the Observer is live.

### Gameplay sessions are independent of Watch spans

A **gameplay session** is a second, slower axis on the same idea. Launch opens none; `ChironApp.ensure_session()` creates one lazily on the first Watch request or first message, so an idle overlay records nothing. A fresh launch is simply "no session open yet" rather than a separate implicit reset. Watching and sessions are independent: one session spans many watch spans, and `new_session()` deliberately keeps Watch requested.

In v3 the reset is explicitly dual-agent: it clears `JournalService`, `ResponderConversation`, both agents' transient memory, queued questions, transcript and cached frames. If Watch remains requested, Observer reset forces a fresh connection without a resumption handle and capture resumes only when it becomes live again. The previous durable record is closed, not deleted.

### The session record observes both agents and outlives them

`chiron/sessions/` (plural — not `chiron/session.py`) persists an evening of play under `$XDG_DATA_HOME/chiron/sessions/`: one directory per session holding an append-only `events.jsonl` and ~256 px JPEG thumbnails of every frame that reached a model, plus one `index.json` the browser reads so listing never opens an event file. **The data dir, not the config dir**, so `--fresh-install`'s `plan_removal()` cannot reach it by construction; `describe_untouched_sessions()` says so in the plan output.

`SessionRecorder` observes signals and never participates in agent decisions. v3 events attribute statuses, frames, calls, compactions, messages and ReAct traces with stable `observer`, `responder` or `journal` agent IDs. `observer_run` remains in reader-side vocabularies for v1/v2 compatibility but no v3 path emits it. One immediate image delivered to both agents produces two attributed frame-delivery events while thumbnail memoization stores the JPEG once.

Two invariants to preserve when touching this: the recorder's incremental counters must match what `chiron/sessions/store.py`'s `summarise()` computes by rescan (the index is a *cache* of the stream, and `SessionIndex.rebuild()` proves it), and thumbnails are memoised by capture-time-plus-shape rather than by `id(frame)` — a freed frame's address gets reused, which silently deduplicates distinct captures.

### Responder cost is measured; Observer cost is estimated

`LiteLLMModel.usage_sink` records fixed answers, ReAct steps, Responder compaction and journal compaction. Those request/response calls use provider-reported usage where available. Adding a runtime call kind requires adding it to `LLMCallKind` in `chiron/models/usage.py`; the closed vocabulary prevents silent cost-category drift.

Gemini Live does not expose an equivalent billable per-checkpoint seam, so `chiron/live/estimate.py` emits `pricing_source="estimated"` rows for Observer checkpoints and context writes using frame detail, checkpoint/seed text and discarded native-audio duration. **Every surface that renders a total checks that flag and prefixes `~`** (`CostRollup.render()`, the overlay footer, viewer summaries and breakdowns). A mixed session remains estimated even when all Responder rows are exact.

### Context is compacted from model limits, never a wall clock

Raw journal entries and canonical conversation turns are lossless sources. Only their **model-facing derived views** are summarized. `chiron/nonlive/compaction.py`'s `resolve_context_window()` consults the provider catalogue's disk cache/bundled metadata, then LiteLLM metadata, and finally uses a conservative 32k assumption for unknown pasted IDs; it never performs network I/O on the question path.

`JournalCompactor.prepare()` measures the shared summary-plus-tail against the consuming model. It triggers at 20% of that model's window, targets 10% (5% on forced overflow recovery), summarizes through the selected Responder model and advances a cursor without deleting `JournalLog`. The old retention counts, replay counts, sidecar cadence and 120-second fold no longer exist.

The Responder measures the fully assembled request and compacts conversation above 85%, keeping the latest eight messages verbatim. Image data URIs are projected out before text counting and charged a flat allowance. A typed context-overflow error forces journal/conversation compaction and exactly one retry; a second overflow is surfaced with a larger-context recommendation.

The Observer uses the selected Live model's resolved window and rotates at 78%. It waits for the current checkpoint for at most ten seconds, then `rotate(..., fresh=True)` drops the resumption handle; the next fresh connection seeds from the shared journal summary/tail. Normal reconnects keep the handle. Server sliding-window compression remains a backstop, and no periodic journal replay duplicates tool calls already present in Live context.

### The game is detected before watching starts, not when

`ActiveWindowTracker` polls X11 from app launch and excludes Chiron's own window IDs, so clicking the overlay does not replace the remembered game. Identity resolution prefers Steam metadata and falls back to title/`WM_CLASS`/process name. The result updates both agents' runtime `detected_game`, but `settings.game_name` always wins when typed. Responder prompts read the current label per request; an already-open Observer's system instruction changes only on reconnect. Mid-watch switches also go through `JournalService.record()` so the Responder and next Observer seed see them immediately/durably. `WindowInfo.identity` excludes title churn; Wayland/no-display failures remain inert.

### Native-audio Live constraints are contained inside the Observer

Served Live models require `response_modalities=[AUDIO]`, so the Observer requests audio while discarding it and omitting output transcription. `_receive()` still loops because `google-genai`'s `session.receive()` covers one model turn. It handles tool calls, resumption handles, `go_away`, reported context usage and discarded PCM token estimation; it never turns model content into a visible answer.

Automatic activity detection is disabled because Chiron supplies manual checkpoint boundaries. Do not add `explicit_vad_signal`: that asks Enterprise Agent Platform to report server VAD events, is unsupported by the Developer API client Chiron uses, and is unrelated to sending `activity_start`/`activity_end`.

`is_permanent_error()` stops retries for a bad key, missing model or unsupported modality; other failures use bounded backoff. `DEFAULT_OBSERVER_MODEL` is the static fresh-install selection, while the picker and context resolver use refreshable catalogue metadata with bundled fallback rather than a global context constant. The exact manual video/checkpoint transaction is covered by opt-in `tests/test_provider_smoke.py` because offline fakes cannot prove preview-model behavior.

### The journal has one write path and a read-only Responder capability

`JournalLog` is the raw timestamped append-only source. `JournalService.record()` is the single path for Observer tool calls and application-generated facts; it appends once and notifies the drawer and recorder once. `JournalService.reader()` grants the Responder only an immutable `JournalSnapshot`, structurally preventing it from writing journal facts. `JournalCompactor` owns the replaceable older summary plus recent verbatim tail.

Fresh Observer connections seed from the token-budgeted journal view; active connections already contain their own `record_event` tool calls, so new entries are not replayed on a timer. The deleted `ToolCallJournal`, `SidecarJournal` and `ObserverJournal` strategies must not be resurrected as compatibility branches.

### The overlay is a two-view stack plus a drawer, not a transcript

`chiron/ui/overlay.py` holds a `QStackedWidget` with two pages — `PLAY_VIEW` (today's transcript) and `SESSION_VIEW` (one recorded session read back, in `chiron/ui/history_view.py`). Getting *to* a session is not a page: `SessionPicker` is a `Qt.Popup` that drops under the 🕘 button, so choosing an evening costs no panel resize. Navigation is a small back-stack: `overlay.back()` walks one step, `Escape` calls it and only hides the panel once already on play, and `start_response()` snaps to play so an answer arriving mid-browse is never missed. Managing a session (rename, delete, delete-thumbnails) lives in the viewer's `⋯` menu — on the thing you are looking at, not on a row you are skimming — and the three signals still surface on `OverlayWindow` unchanged, so `app.py`'s wiring did not move.

**Two independent things resize the window, and neither may reach the saved width.** Entering the viewer temporarily enlarges the panel and stores the pre-review size in `_play_size`; opening the journal drawer widens it by `journal_width + BODY_SPACING`. `current_geometry()` undoes both — it reports `_play_size` when set, minus `_journal_offset()` — so `OverlaySettings.width` always means the un-drawered play width. Anything that changes the drawer while `_play_size` is set must move `_play_size` by the same delta, or closing the drawer mid-review reopens a gap on the way back to play.

### The journal is a column, and a closed column still counts

Journal entries used to be dimmed `✎` lines inline in the transcript. They now live in `chiron/ui/journal_drawer.py`'s `JournalDrawer`, a collapsible right-hand column, newest first, category-coloured via `JOURNAL_CATEGORY_COLOURS` (an unrecognised category falls back to dim text rather than being dropped — categories are a hint to the model, not a schema). Rendering is pure like `chiron/sessions/render.py`: `render_entries()` takes entries and returns HTML, assertable with no display.

The column is a *surface* — raised background, border, radius — not a `border-left` hairline; the first version was the hairline, and it read as leftover space with text floating in it. Two QTextDocument quirks are load-bearing in the render and will look like sloppy spacing if undone: a paragraph's bottom margin is dropped before a following `<table>`, so entries are separated by a blank `GAP_POINTS` paragraph rather than a margin, and `font-size` on a `<p>` is unreliable (which is why the empty state carries no scaled-up watermark glyph). A plain `QWidget` also paints no stylesheet background or border without `WA_StyledBackground`.

The drawer sits *outside* the view stack, which is what lets it stay open while a recorded session is read. Moving the entries out of the transcript removes the only signal that the journal was being written, so the toggle carries an unread count (`✎ 3`) that clears on open — `_refresh_journal_button()` also has to `unpolish`/`polish` the button, since Qt does not re-evaluate the `[unread="true"]` property selector on its own.

All entry producers, including `ChironApp._on_active_window`, must use `JournalService.record()` rather than append to `JournalLog` directly; bypassing the service skips the drawer and recorder callbacks. Separately, the drawer's open state is runtime state the *overlay* owns, persisted in `OverlaySettings.journal_open`: `SettingsWindow.collect()` builds a fresh `Settings` from form fields, so `ChironApp.apply_settings()` copies `journal_open` and `journal_width` from the previous object exactly as it does `position_x`/`position_y`, or saving settings would quietly reset the drawer.

Rendering a recorded session is pure: `chiron/sessions/render.py` turns events into HTML with thumbnails inline (`<img src="…">` at absolute paths), so the viewer's whole output is assertable without a display. A thumbnail that has been deleted to reclaim disk renders as a dim placeholder, never a broken image.

### Settings: edit-a-copy, whole-object apply

`SettingsWindow` never mutates live `Settings` in place. `load()` populates widgets from a snapshot, `collect()` builds a fresh whole object, and Save hands it to `ChironApp.apply_settings()`. Overlay appearance previews separately through `appearanceChanged`. The page exposes distinct Observer and Responder model cards, `responder_mode`, fixed capture interval, question-frame policy, frame width and Live media resolution; the old unified/split, trigger cadence, journal strategy and replay/retention fields do not exist.

`Settings.requires_observer_reconnect()` covers the Google key, Observer model/instruction, game and media resolution. `requires_responder_rebuild()` covers the selected Responder model/mode/key, instruction and game. Capture cadence and question policy apply in place; a width change also invalidates frames. Extend these predicates rather than special-casing new agent configuration at call sites. A Responder rebuild must preserve `ResponderConversation`; an Observer reconnect while Watch is requested must keep capture paused until `live`.

`available_models()` is fetched off-thread when Settings opens. Google `supportedGenerationMethods` produces separate Live Observer and ordinary Responder entries; OpenRouter contributes only Responder entries. `observer_models()` and `responder_models()` enforce the picker split, and ReAct excludes known tool-incompatible models. Failed fetches fall back to a 24-hour disk cache then `STATIC_MODELS`; unknown pasted IDs remain selectable with warnings, and unknown context windows use the conservative compaction fallback. `ModelPicker.set_selection()` is the silent load path; `choose()` is the user path and emits `selectionChanged`.

`chiron/config/settings.py` is also where `--fresh-install` lives (`plan_removal()` / `remove_configuration()`): it only ever considers deleting a directory literally named `chiron`, so pointing `--settings` at a file in some other shared directory can't put that directory up for deletion.

### Hotkeys: X11-native, not pynput

`chiron/ui/hotkeys.py` grabs global hotkeys via `python-xlib` (`XGrabKey`) rather than `pynput` — `pynput` pulls in `evdev` on Linux, which has no prebuilt wheels and needs a C toolchain to compile. `pynput` is used automatically as a fallback backend if it happens to be importable, but is not a project dependency. Both backends live behind `GlobalHotkeyManager`, which is the only thing `app.py` talks to.

## Behavioral Guidelines

Behavioral guidelines to reduce common LLM coding mistakes. Merge with project-specific instructions as needed.

**Tradeoff:** These guidelines bias toward caution over speed. For trivial tasks, use judgment.

**These guidelines are working if:** fewer unnecessary changes in diffs, fewer rewrites due to overcomplication, and clarifying questions come before implementation rather than after mistakes.

### 1. Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

Before implementing:
- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them - don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

### 2. Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

### 3. Surgical Changes

**Touch only what you must. Clean up only your own mess.**

When editing existing code:
- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it - don't delete it.

When your changes create orphans:
- Remove imports/variables/functions that YOUR changes made unused.
- Don't remove pre-existing dead code unless asked.

The test: Every changed line should trace directly to the user's request.

### 4. Goal-Driven Execution

**Define success criteria. Loop until verified.**

Transform tasks into verifiable goals:
- "Add validation" → "Write tests for invalid inputs, then make them pass"
- "Fix the bug" → "Write a test that reproduces it, then make it pass"
- "Refactor X" → "Ensure tests pass before and after"

For multi-step tasks, state a brief plan:
```
1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]
```

Strong success criteria let you loop independently. Weak criteria ("make it work") require constant clarification.
