# Chiron

Chiron is a small AI companion that sits over your game, watches when you ask it
to, and helps you make sense of what is happening on screen.

Stuck in an unfamiliar room? Wondering what killed you, where you were headed,
or what that item description means? Type a question into the overlay and Chiron
answers with the current scene and the recent history of your play session in
mind. It is meant to feel less like a chatbot beside the game and more like a
friend looking over your shoulder—one who stays quiet until invited in.

Chiron is currently an early Linux/X11 project built around two deliberately
separate agents. **Chiron-Observer** watches and remembers; **Chiron-Responder**
answers. Keeping those jobs apart means the agent looking at a steady stream of
screenshots never gets to speak for you, while the agent speaking to you cannot
rewrite what was observed.

## What it does

- Lives in a frameless, always-on-top panel that can be moved and resized.
- Captures at a fixed interval while the Live Observer is connected.
- Answers typed questions through a separately selected Gemini or OpenRouter
  Responder, using the current frame, journal, and conversation.
- Keeps a lightweight journal of notable events so useful context can outlive
  old screenshots and connection changes.
- Records each evening of play—journal, chat, model activity, frame thumbnails,
  and costs—as a session you can revisit from the overlay.
- Starts idle. Chiron does not capture or send screenshots until you explicitly
  turn watching on.

It works best as a guide for exploration, puzzles, builds, objectives, and
post-mortems. It sees periodic snapshots rather than video, so it is not a
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
export GEMINI_API_KEY="your-key-here"
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
| `Ctrl+Alt+J` | Open or close the journal |
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
question in the input field whenever you want help. The journal opens as a
right-hand column; when it is closed, its button keeps an unread count so you
can tell when the Observer has written something down.

Turning Watch on first connects Chiron-Observer; capture does not begin until it
reports `live`. If the socket drops, capture stops and cached frames are cleared
while bounded reconnects continue. Chiron-Responder can still answer from the
existing journal and conversation, and is told explicitly that those observations
may be stale.

Stopping Watch closes the Observer socket but leaves the current gameplay
session available for journal-only questions. If you prefer it to begin watching
immediately, enable **Start Watch on launch** under **Settings → Capture**.

A gameplay session begins lazily with your first message or Watch request and
can span any number of Watch-on and Watch-off periods. Use **New Session** when
you want a clean journal and conversation. The previous session is closed, not
deleted, and remains available from the history button. Recorded sessions can be
searched, renamed, reviewed with their frame thumbnails, or deleted from the
session viewer.

## A note on privacy

The boundary is deliberately simple: no screenshot is taken unless Watch is
requested and Chiron-Observer is connected. Chiron launches with Watch off.

The Observer always uses Google Gemini Live, so a Google API key is required even
when the Responder uses OpenRouter. With an OpenRouter Responder, a screenshot
attached to a question is sent to OpenRouter as well as to the Google Observer.

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

## Two agents, one memory

Raw screenshots make good short-term evidence but poor durable memory. Every five
seconds by default, Chiron gives the Live Observer one frame. The Observer is a
persistent, tool-only Gemini Live connection: it can record a journal event, but
it has no path to the chat panel. The native-audio output required by the Live
API is discarded and never played or transcribed.

The Responder is an ordinary multimodal request/response model, selected
independently from Gemini or OpenRouter. It receives the question, Observer
freshness, relevant journal memory, conversation, and—when available—a current
frame. It alone produces user-visible answers and has read-only access to the
journal.

There are two Responder modes:

- **Fixed horizon** assembles the available context into a normal model request.
- **ReAct** gives the Responder one read-only `read_journal` tool and requires it
  to consult that tool before it may answer.

Both modes share the same clean conversation history: only your messages and
final answers are kept, not internal tool chatter. When the journal or
conversation approaches the selected model's context limit, Chiron summarizes
older material for future prompts without deleting the raw journal or recorded
conversation.

Questions use the latest frame by default. The optional immediate policy takes
exactly one extra frame, sends the same image to both agents, and does not shift
the periodic deadline. There is no client-side scene detection, novelty trigger,
heartbeat, or burst mode.

Chiron also watches the active X11 window to infer the game name, preferring
Steam metadata when available. You can always override the detected name in
Settings.

## Recorded sessions and costs

Chiron keeps an append-only record of each gameplay session under
`~/.local/share/chiron/sessions/`. Alongside the conversation and journal, it
records which agent received a frame, model calls, compaction events, ReAct
traces, and small thumbnails of frames that reached a model. This is why the
session viewer can reconstruct an evening without relying on the agents' current
memory.

Responder and compaction costs use provider-reported token usage when available.
Gemini Live does not expose comparable per-checkpoint billing data, so Observer
costs are estimates. Any total containing estimated activity is prefixed with
`~` in the overlay and session viewer.

## Current limits

- X11 is supported; Wayland capture and global shortcuts are not yet wired up.
- Recorded sessions persist, but they are not automatically used as memory in a
  new gameplay session.
- There is no game-wiki retrieval yet.
- Chiron responds only when asked and does not offer proactive coaching.
- Generated answers can be mistaken. Treat them as guidance, especially when a
  game has hidden information or the relevant moment was not captured.

## Configuration and command-line options

Most configuration lives in the settings window: separate Observer and Responder
models and instructions, Responder mode, credentials, capture interval, question
frame policy, image size and detail, overlay appearance, and hotkeys. Journal
memory is token-budgeted automatically from the selected models' context windows.
Appearance and cadence changes apply directly; Observer configuration reconnects
only the Observer, while Responder changes rebuild only the Responder and
preserve its conversation.

Useful launch options:

```bash
uv run chiron --settings /path/to/settings.json
uv run chiron --log-level DEBUG
uv run python -m chiron
```

Use `uv run chiron --help` for the complete list.
