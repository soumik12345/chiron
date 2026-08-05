# Chiron

Chiron is a small AI companion that sits over your game, watches when you ask it
to, and helps you make sense of what is happening on screen.

Stuck in an unfamiliar room? Wondering what killed you, where you were headed,
or what that item description means? Type a question into the overlay and Chiron
answers with the current scene and the recent history of your play session in
mind. It is meant to feel less like a chatbot beside the game and more like a
friend looking over your shoulder—one who stays quiet until invited in.

Chiron is currently an early Linux/X11 project with two deliberately separate
agents: a Gemini Live Observer that watches and journals, and a request/response
Responder that produces every answer.

## What it does

- Lives in a frameless, always-on-top panel that can be moved and resized.
- Captures at a fixed interval while the Live Observer is connected.
- Answers typed questions through a separately selected Gemini or OpenRouter
  Responder, using the current frame, journal, and conversation.
- Keeps a lightweight journal of notable events so useful context can outlive
  old screenshots and connection changes.
- Starts idle. Chiron does not capture or send screenshots until you explicitly
  turn watching on.

It works best as a guide for exploration, puzzles, builds, objectives, and
post-mortems. With a maximum capture rate of one frame per second, it is not a
twitch-game copilot and will not tell you to dodge in time.

## Getting started

You will need:

- Linux running X11 (GNOME on Ubuntu is the currently tested setup)
- Python 3.10 or newer
- [uv](https://docs.astral.sh/uv/)
- A Gemini API key (required for the Live Observer and Gemini Responders)
- Optionally, an OpenRouter key for OpenRouter Responder models

Install the dependencies from the repository root:

```bash
uv sync
```

Then either export your key and launch Chiron:

```bash
uv run chiron
```

or run `uv run chiron` and paste the key into **Settings**. Chiron also checks
`GOOGLE_API_KEY` if `GEMINI_API_KEY` is not set. You can test the key from the
settings window before saving it.

Run the game in **borderless-windowed mode**. A true-fullscreen X11 game owns the
display and prevents overlays such as Chiron from appearing above it.

## Using Chiron

When Chiron opens, it is visible but not watching. The default shortcuts are:

| Shortcut | Action |
|---|---|
| `Ctrl+Alt+W` | Start or stop watching |
| `Ctrl+Alt+C` | Show or hide the overlay |
| `Ctrl+Alt+S` | Open settings |
| `Esc` | Hide the panel |
| `Ctrl+Q` | Quit |

The eye button also toggles watching, and the header always shows the current
state. You can bind separate start and stop shortcuts if you want an unambiguous
way to make sure capture has ended. All global shortcuts can be changed under
**Settings → Hotkeys**.

The rest of the header is `⚙` for settings, `–` to hide the panel, and `✕` to
quit Chiron. Hiding leaves it running in the background — the show/hide shortcut
brings it back — while `✕` shuts it down for real.

Drag the header to move the panel and use the corner grip to resize it. Type a
question in the input field whenever you want help.

Turning Watch on first connects Chiron-Observer; capture does not begin until it
reports `live`. If the socket drops, capture stops and cached frames are cleared
while bounded reconnects continue. Chiron-Responder can still answer from the
existing journal and conversation, and is told explicitly that those observations
may be stale.

Stopping Watch closes the Observer socket but leaves the current gameplay
session available for journal-only questions. If you prefer it to begin watching
immediately, enable **Start Watch on launch** under **Settings → Capture**.

## A note on privacy

The boundary is deliberately simple: no screenshot is taken unless Watch is
requested and Chiron-Observer is connected. Chiron launches with Watch off.

With an OpenRouter Responder, a screenshot attached to a question is sent to
both Google Live (Observer) and OpenRouter (Responder). The Settings privacy page
calls this out before you choose that topology.

Settings, including saved API keys, are stored with owner-only permissions in
`~/.config/chiron/settings.json`. Gameplay sessions—including the journal,
transcript, traces, costs, and model-frame thumbnails—are stored under
`~/.local/share/chiron/sessions/` and can be browsed in the overlay.

To remove Chiron's saved configuration and API key:

```bash
uv run chiron --fresh-install
```

Chiron prints what it plans to remove and asks for confirmation. For a
non-interactive invocation, add `--yes`. Recorded gameplay sessions are in the
data directory and are deliberately not removed by `--fresh-install`.

## How the memory works

Raw screenshots make good short-term evidence but poor durable memory. Every five
seconds by default, Chiron gives the Live Observer one frame inside an explicit
activity boundary. The Observer can call only `record_event`; its audio and
content output are discarded. Durable progress, objectives, locations, items,
and decisions enter one shared journal write path.

The Responder has read-only journal access. Fixed-horizon mode injects journal
memory and normally uses one completion. ReAct mode exposes only `read_journal`
and structurally forces a successful read before a final answer. Both modes share
one canonical user/final-answer conversation. When either journal or conversation
context grows, older material is summarized for the active prompt while every raw
journal entry and canonical conversation turn remains intact.

Questions use the latest frame by default. The optional immediate policy takes
exactly one extra frame, sends the same image to both agents, and does not shift
the periodic deadline. There is no client-side scene detection, novelty trigger,
heartbeat, or burst mode.

## Current limits

- X11 is supported; Wayland capture and global shortcuts are not yet wired up.
- There is no game-wiki retrieval or cross-session memory yet.
- Chiron responds only when asked and does not offer proactive coaching.
- Generated answers can be mistaken. Treat them as guidance, especially when a
  game has hidden information or the relevant moment was not captured.

## Configuration and command-line options

Most configuration lives in the settings window: separate Observer and Responder
models, Responder mode, credentials, fixed capture behavior, overlay appearance,
and hotkeys. Journal memory is token-budgeted automatically from the selected
models' reported context windows. Appearance and cadence changes apply directly;
Observer configuration reconnects only the Observer, while Responder changes
rebuild only the Responder and preserve its conversation.

Useful launch options:

```bash
uv run chiron --settings /path/to/settings.json
uv run chiron --log-level DEBUG
uv run python -m chiron
```

Use `uv run chiron --help` for the complete list.

## Development

The test suite does not require a network connection or a display:

```bash
uv run pytest
uv run ruff check chiron/ tests/
uv run ruff format chiron/ tests/
```

The application uses one `qasync` event loop for Qt and asyncio. Screen capture
is the only worker thread; it hands encoded frames back through Qt signals.

```text
chiron/
├── app.py          # application lifecycle and component wiring
├── capture/        # screenshots, encoding, and fixed scheduling
├── config/         # settings model and persistence
├── journal/        # one journal write path and read-only snapshots
├── observer/       # silent Gemini Live Observer
├── responder/      # fixed-horizon and ReAct answers
├── live/           # Observer cost estimation
├── nonlive/        # shared request-context compaction helpers
├── sessions/       # durable gameplay events, thumbnails, and index
├── ui/             # overlay, settings, hotkeys, and theme
├── core/           # ReAct harness adapted by Chiron-Responder
└── models/         # model catalogue, calls, pricing, and usage
```
