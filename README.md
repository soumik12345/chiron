# Chiron

Chiron is a small AI companion that sits over your game, watches when you ask it
to, and helps you make sense of what is happening on screen.

Stuck in an unfamiliar room? Wondering what killed you, where you were headed,
or what that item description means? Type a question into the overlay and Chiron
answers with the current scene and the recent history of your play session in
mind. It is meant to feel less like a chatbot beside the game and more like a
friend looking over your shoulder—one who stays quiet until invited in.

Chiron is currently an early Linux/X11 project powered by the
[Google Gemini Live API](https://ai.google.dev/gemini-api/docs/live-api).

## What it does

- Lives in a frameless, always-on-top panel that can be moved and resized.
- Captures your screen adaptively instead of streaming it continuously.
- Answers typed questions using the newest frames and your recent conversation.
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
- A Gemini API key

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

Stopping watching closes the Live session as well as pausing capture. The
in-memory journal remains available, so you can still ask about earlier events
and Chiron can use them when watching resumes. If you prefer it to begin watching
immediately, enable **Start watching as soon as Chiron opens** under
**Settings → Capture**.

## A note on privacy

The boundary is deliberately simple: “not watching” means no screenshots are
taken and no Live session is open. Chiron launches in that state by default.

Settings, including a saved API key, are stored with owner-only permissions in
`~/.config/chiron/settings.json`. The journal is currently held only in memory
and disappears when the application exits.

To remove Chiron's saved configuration and API key:

```bash
uv run chiron --fresh-install
```

Chiron prints what it plans to remove and asks for confirmation. For a
non-interactive invocation, add `--yes`. Files it did not create are left alone.

## How the memory works

Raw screenshots make good short-term memory but poor long-term memory. They are
large, and the Live model eventually has to discard older frames. Chiron turns
important moments into compact timestamped journal entries and periodically
feeds those notes back into the conversation.

There are two journal strategies in Settings:

- **In-session tool call:** the Live model records notable events itself. This
  needs no extra model call, but journaling shares the model's attention with the
  conversation.
- **Sidecar summarizer:** a separate, cheaper model periodically distills recent
  frames and conversation. It keeps the jobs separate but makes additional API
  calls.

Capture follows the same “use only what matters” approach. Chiron takes a frame
roughly every four seconds while the scene is calm, then briefly increases to
one frame per second after a question or a substantial scene change. The timing,
image quality, scene detection, and monitor can all be adjusted in Settings.

Although the interface is text-in and text-out, currently available Gemini Live
models produce native audio. Chiron requests the model's output transcription,
shows that text, and discards the audio without playing it. This means responses
are billed by Gemini as audio output tokens.

## Current limits

- X11 is supported; Wayland capture and global shortcuts are not yet wired up.
- The journal does not persist between application runs.
- There is no game-wiki retrieval or cross-session memory yet.
- Chiron responds only when asked and does not offer proactive coaching.
- Generated answers can be mistaken. Treat them as guidance, especially when a
  game has hidden information or the relevant moment was not captured.

## Configuration and command-line options

Most configuration lives in the settings window: model and API key, capture
behavior, journal strategy, overlay appearance, and hotkeys. Appearance changes
preview immediately; changes that affect the Live session reconnect it after
you save.

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
├── capture/        # screenshots, encoding, and adaptive scheduling
├── config/         # settings model and persistence
├── journal/        # in-memory log and journal strategies
├── live/           # Gemini Live session and prompts
├── ui/             # overlay, settings, hotkeys, and theme
├── core/           # dormant general-purpose ReAct agent harness
└── models/         # model plumbing used by the sidecar summarizer
```
