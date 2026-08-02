"""User-editable configuration for the Chiron overlay.

Everything the settings page can change lives in one nested :class:`Settings`
model, persisted as JSON under the XDG config directory (``~/.config/chiron/
settings.json`` by default). One file, one schema, one load/save pair — the
settings window edits a copy and hands back a whole new :class:`Settings`, so
nothing in the app has to reason about partially-applied configuration.

The Gemini credential is resolved lazily rather than at load time: a key saved
here wins, and an empty field falls back to ``GEMINI_API_KEY`` (then
``GOOGLE_API_KEY``) in the environment. That keeps the "I already export the key
in my shell" path working without ever writing the secret to disk, while still
letting someone paste a key into the UI. When the file *does* hold a key it is
written with mode ``0600``.

Unknown keys in an existing file are ignored and missing ones fall back to
defaults, so a settings file written by an older build still opens; a file that
is outright unparseable is reported and replaced by defaults rather than being
allowed to stop the app from starting.

**The selected model decides the mode.** ``selected_model`` is provider-qualified
— ``live/<id>`` for the Gemini Live API, ``gemini/<id>`` for Google AI Studio,
``openrouter/<vendor>/<id>`` for OpenRouter — and picking one is the only way to
choose between live and non-live operation. There is no separate mode toggle to
contradict it. v0's ``live_model`` field is migrated on load, so an existing
settings file keeps running the model it was already running.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, ValidationError, model_validator

logger = logging.getLogger(__name__)

#: Prefix marking a Live API selection. Not a litellm route — the Live API is a
#: websocket the ``google-genai`` SDK opens, not a completions endpoint — so this
#: is the one selection id that never reaches litellm.
LIVE_PREFIX = "live/"

#: The Live API model the overlay talks to. Every Live model still served is a
#: native-audio one — the text-out half-cascade models were retired — so Chiron
#: takes the model's speech and renders its own transcription as text.
DEFAULT_LIVE_MODEL = "gemini-3.1-flash-live-preview"

#: What a fresh install talks to: the Live API, as in v0.
DEFAULT_MODEL = LIVE_PREFIX + DEFAULT_LIVE_MODEL

#: Known Live API model ids, offered in the settings page combo box. The field is
#: editable, so a newer id can always be typed in.
LIVE_MODEL_CHOICES: list[str] = [
    "gemini-3.1-flash-live-preview",
    "gemini-2.5-flash-native-audio-latest",
    "gemini-2.5-flash-native-audio-preview-12-2025",
]

#: Context window of the native-audio Live models, used for the "how far back can
#: it see" estimate on the capture page.
LIVE_CONTEXT_TOKENS = 128_000

#: litellm id for the sidecar summariser — a regular (non-Live) chat model.
DEFAULT_SIDECAR_MODEL = "gemini/gemini-3.6-flash"

SIDECAR_MODEL_CHOICES: list[str] = [
    "gemini/gemini-3.6-flash",
    "gemini/gemini-2.5-flash",
    "gemini/gemini-2.5-flash-lite",
]

#: Environment variables consulted, in order, when no key is saved.
API_KEY_ENV_VARS = ("GEMINI_API_KEY", "GOOGLE_API_KEY")

#: Environment variable consulted when no OpenRouter key is saved.
OPENROUTER_API_KEY_ENV_VARS = ("OPENROUTER_API_KEY",)

JournalStrategy = Literal["tool_call", "sidecar"]
MediaResolution = Literal["low", "medium", "high"]
AgentMode = Literal["unified", "split"]
TriggerStrategy = Literal["spike_gated_heartbeat", "spike_plain_heartbeat"]

#: "Frame detail" resolved to a capture width, for non-live providers.
#:
#: The setting means the same thing in both modes — how closely the model reads
#: each frame, at what token cost — but the mechanism differs. The Live API takes
#: the same pixels and spends a different server-side token budget on them
#: (``media_resolution``); a request/response endpoint has no such knob, so the
#: lever is the pixels themselves. 512 px is deliberate: a 16:9 frame at 512x288
#: lands in Gemini's flat sub-384px tier at ~258 tokens, near enough the ~260
#: tokens a live ``low`` frame costs that the tiers mean the same spend either way.
DETAIL_CAPTURE_WIDTH: dict[str, int] = {"low": 512, "medium": 768, "high": 1152}


def is_live_selection(selection: str) -> bool:
    """Whether a provider-qualified selection names a Live API model."""
    return (selection or "").strip().startswith(LIVE_PREFIX)


def live_model_id(selection: str) -> str:
    """The bare Live API model id inside a ``live/…`` selection.

    A non-live selection has no live model in it, so the default is returned
    rather than a nonsense id — the caller is asking what to connect with, and
    "nothing" is not an answer the websocket accepts.
    """
    text = (selection or "").strip()
    if text.startswith(LIVE_PREFIX):
        return text[len(LIVE_PREFIX) :] or DEFAULT_LIVE_MODEL
    return DEFAULT_LIVE_MODEL


def litellm_model_id(selection: str) -> str:
    """The litellm id for a non-live selection (already provider-qualified)."""
    return (selection or "").strip()


class CaptureSettings(BaseModel):
    """Screen capture and the adaptive shutter.

    Attributes:
        monitor_index (int): ``mss`` monitor number. 0 is the virtual "all
            monitors" screen; 1 is the primary display.
        baseline_interval_seconds (float): Seconds between keepalive frames when
            nothing in particular is happening.
        burst_interval_seconds (float): Seconds between frames while bursting.
            1.0 is the API ceiling of 1 fps.
        burst_duration_seconds (float): How long a burst lasts once triggered.
        frame_width (int): Frames are downscaled to this width before encoding.
        jpeg_quality (int): JPEG quality (1-95) for encoded frames.
        stamp_timestamp (bool): Draw the capture time into the frame's corner so
            the model has an explicit "now" to reason about.
        scene_change_enabled (bool): Trigger a burst when a cheap pixel diff sees
            the screen change hard (loading screen, new area, death screen).
        scene_change_threshold (float): Normalised 0-1 difference between two
            frame signatures above which a scene change is declared.
        media_resolution (MediaResolution): "Frame detail" — how closely the
            model reads each frame. In live mode this is the API's token budget
            per frame (``low`` is roughly 260 tokens); in non-live mode there is
            no such server-side knob, so it resolves to a capture width through
            :data:`DETAIL_CAPTURE_WIDTH` and supersedes ``frame_width``.
        watch_on_launch (bool): Begin watching the moment Chiron starts. Off by
            default: a screen recorder that switches itself on when you log in is
            not something anyone should have to opt out of.
    """

    monitor_index: int = 1
    baseline_interval_seconds: float = Field(default=4.0, ge=0.5, le=60.0)
    burst_interval_seconds: float = Field(default=1.0, ge=1.0, le=10.0)
    burst_duration_seconds: float = Field(default=15.0, ge=1.0, le=120.0)
    frame_width: int = Field(default=768, ge=256, le=1920)
    jpeg_quality: int = Field(default=60, ge=10, le=95)
    stamp_timestamp: bool = True
    scene_change_enabled: bool = True
    scene_change_threshold: float = Field(default=0.12, ge=0.01, le=1.0)
    media_resolution: MediaResolution = "low"
    watch_on_launch: bool = False

    def tokens_per_minute(self, *, tokens_per_frame: int = 260) -> float:
        """Rough visual token burn per minute at the baseline rate.

        Args:
            tokens_per_frame (int): Cost of one frame at the configured media
                resolution. The default is the documented figure for ``low``.

        Returns:
            float: Estimated tokens per minute of ambient capture.
        """
        if self.baseline_interval_seconds <= 0:
            return 0.0
        return (60.0 / self.baseline_interval_seconds) * tokens_per_frame


class JournalSettings(BaseModel):
    """How meaning is moved out of the frames before they are evicted.

    Attributes:
        strategy (JournalStrategy): ``tool_call`` gives the live model a
            ``record_event`` function to call; ``sidecar`` summarises the recent
            transcript with a separate cheap model on a timer.
        sidecar_model (str): litellm model id used by the sidecar summariser.
        sidecar_interval_seconds (float): How often the sidecar runs.
        sidecar_frame_count (int): How many recent frames the sidecar is shown.
        fold_interval_seconds (float): How often the recent journal is folded
            back into the live session as text.
        fold_entry_limit (int): Maximum journal entries included in one fold.
        max_entries (int): Entries kept in memory before the oldest are dropped.
    """

    strategy: JournalStrategy = "tool_call"
    sidecar_model: str = DEFAULT_SIDECAR_MODEL
    sidecar_interval_seconds: float = Field(default=180.0, ge=30.0, le=1800.0)
    sidecar_frame_count: int = Field(default=3, ge=0, le=8)
    fold_interval_seconds: float = Field(default=120.0, ge=30.0, le=900.0)
    fold_entry_limit: int = Field(default=40, ge=1, le=200)
    max_entries: int = Field(default=500, ge=10, le=5000)


class ObserverSettings(BaseModel):
    """When the non-live observer is worth paying for.

    A live session gets ambient awareness free — frames stream in and simply
    *are* in the model's context. A request/response endpoint gives nothing
    away: every observation is a billed call. So the observer fires on evidence
    (an ambient-weighted novelty spike) plus a heartbeat, with a cooldown that
    caps the damage however noisy the triggers get.

    Attributes:
        trigger_strategy (TriggerStrategy): ``spike_gated_heartbeat`` skips the
            periodic tick when nothing has drifted since the last run — best
            cost profile. ``spike_plain_heartbeat`` always fires on the
            interval, for a simpler liveness guarantee at a small idle cost.
        heartbeat_interval_seconds (float): How often the periodic tick comes
            round, whether or not it ends up firing.
        cooldown_seconds (float): Minimum gap between observer calls. This is
            the ceiling on observer spend: no sequence of triggers can beat it.
        spike_sensitivity (float): How many deviations above its own rolling
            mean a frame's novelty must reach to count as an event. Higher is
            more conservative.
        max_frames_per_call (int): Frames shown to the observer in one call.
    """

    trigger_strategy: TriggerStrategy = "spike_gated_heartbeat"
    heartbeat_interval_seconds: float = Field(default=90.0, ge=15.0, le=900.0)
    cooldown_seconds: float = Field(default=25.0, ge=5.0, le=600.0)
    spike_sensitivity: float = Field(default=3.0, ge=1.0, le=10.0)
    max_frames_per_call: int = Field(default=3, ge=1, le=8)


class OverlaySettings(BaseModel):
    """Look and placement of the always-on-top chat panel.

    Attributes:
        width (int): Overlay width in pixels.
        height (int): Overlay height in pixels.
        position_x (int | None): Last window x, or None to place automatically.
        position_y (int | None): Last window y, or None to place automatically.
        opacity (float): Window opacity, 0.2-1.0.
        font_size (int): Base point size for the transcript.
        always_on_top (bool): Keep the overlay above other windows.
        start_hidden (bool): Launch to the hotkey rather than to a visible panel.
        journal_open (bool): Whether the journal drawer is showing. Runtime state
            the *overlay* owns rather than the settings form — it is persisted
            here so an evening opens the way the last one closed.
        journal_width (int): Width of the drawer column in pixels. `width` is
            measured without it, so opening the drawer widens the window rather
            than narrowing the transcript.
    """

    width: int = Field(default=420, ge=280, le=1600)
    height: int = Field(default=560, ge=240, le=1400)
    position_x: int | None = None
    position_y: int | None = None
    opacity: float = Field(default=0.92, ge=0.2, le=1.0)
    font_size: int = Field(default=11, ge=7, le=24)
    always_on_top: bool = True
    start_hidden: bool = False
    journal_open: bool = False
    journal_width: int = Field(default=240, ge=140, le=600)


class HotkeySettings(BaseModel):
    """Global (system-wide) shortcuts, in ``ctrl+alt+c`` form.

    Every binding is optional: an empty string means "do not grab this one".

    Watching has three bindings rather than one because both habits are
    reasonable. One key you press twice is fewer things to remember; two
    unambiguous keys mean you never have to know the current state to get the
    state you want — which matters for the binding whose job is *stop looking at
    my screen*. The toggle is bound by default and the other two are not.

    Attributes:
        toggle_overlay (str): Show/hide the overlay and focus its input.
        open_settings (str): Open the settings window.
        toggle_watching (str): Start watching if stopped, stop it if watching.
        start_watching (str): Start watching; does nothing if already watching.
        stop_watching (str): Stop watching; does nothing if already stopped.
        toggle_journal (str): Open or close the journal drawer. Global rather
            than a plain shortcut because the panel is usually not focused —
            the point of the drawer is to check what Chiron has written down
            without leaving the game.
    """

    toggle_overlay: str = "ctrl+alt+c"
    open_settings: str = "ctrl+alt+s"
    toggle_watching: str = "ctrl+alt+w"
    start_watching: str = ""
    stop_watching: str = ""
    toggle_journal: str = "ctrl+alt+j"


class Settings(BaseModel):
    """The whole of Chiron's user-editable configuration.

    Attributes:
        api_key (str): Gemini API key, shared by the Live API and Google AI
            Studio. Empty means "look in the environment".
        openrouter_api_key (str): OpenRouter key. Empty means the environment,
            and no key at all means OpenRouter is simply absent from the picker.
        selected_model (str): The provider-qualified model Chiron thinks with —
            ``live/…``, ``gemini/…`` or ``openrouter/…``. Whether this names a
            live model is what decides which session provider runs.
        agent_mode (AgentMode): Non-live only. ``unified`` runs one model and
            one conversation for both observing and answering; ``split`` gives
            the observer its own (typically cheaper) model.
        observer_model (str): Non-live litellm id for the split-mode observer.
            Empty falls back to the selected model.
        game_name (str): Optional name of the game being played, folded into the
            system instruction so the model knows what it is looking at.
        extra_system_prompt (str): Free-form additions to the system instruction.
        capture (CaptureSettings): Screen capture configuration.
        journal (JournalSettings): Journal strategy and cadence. The strategy
            applies to live mode only — in non-live mode the observer is the
            journal writer, always.
        observer (ObserverSettings): Non-live observer cadence and triggers.
        overlay (OverlaySettings): Overlay appearance.
        hotkeys (HotkeySettings): Global shortcuts.
    """

    api_key: str = ""
    openrouter_api_key: str = ""
    selected_model: str = DEFAULT_MODEL
    agent_mode: AgentMode = "unified"
    observer_model: str = ""
    game_name: str = ""
    extra_system_prompt: str = ""

    capture: CaptureSettings = Field(default_factory=CaptureSettings)
    journal: JournalSettings = Field(default_factory=JournalSettings)
    observer: ObserverSettings = Field(default_factory=ObserverSettings)
    overlay: OverlaySettings = Field(default_factory=OverlaySettings)
    hotkeys: HotkeySettings = Field(default_factory=HotkeySettings)

    @model_validator(mode="before")
    @classmethod
    def _migrate_live_model(cls, data: Any) -> Any:
        """Read a v0 file's ``live_model`` as a ``live/…`` selection.

        v0 had one model field and it was always a Live API id. Someone
        upgrading has a settings file saying so, and the honest reading of it is
        "keep talking to that model" — not "fall back to the new default".
        """
        if not isinstance(data, dict) or data.get("selected_model"):
            return data
        legacy = str(data.get("live_model") or "").strip()
        if legacy:
            data = dict(data)
            data["selected_model"] = (
                legacy if is_live_selection(legacy) else LIVE_PREFIX + legacy
            )
        return data

    # ------------------------------------------------------------ credentials

    def resolved_api_key(self) -> str:
        """The Google key to authenticate with: the saved one, else the env."""
        if self.api_key.strip():
            return self.api_key.strip()
        for name in API_KEY_ENV_VARS:
            value = os.environ.get(name, "").strip()
            if value:
                return value
        return ""

    def api_key_source(self) -> str:
        """Where :meth:`resolved_api_key` found a key: ``settings``, an env var
        name, or ``none``."""
        if self.api_key.strip():
            return "settings"
        for name in API_KEY_ENV_VARS:
            if os.environ.get(name, "").strip():
                return name
        return "none"

    def resolved_openrouter_key(self) -> str:
        """The OpenRouter key: the saved one, else ``OPENROUTER_API_KEY``."""
        if self.openrouter_api_key.strip():
            return self.openrouter_api_key.strip()
        for name in OPENROUTER_API_KEY_ENV_VARS:
            value = os.environ.get(name, "").strip()
            if value:
                return value
        return ""

    def key_for_model(self, model_id: str) -> str:
        """The credential a given model id authenticates with.

        Args:
            model_id (str): A selection or litellm id.

        Returns:
            str: The matching key, or empty — which leaves litellm to find one
                in the environment, exactly as it did before this existed.
        """
        if (model_id or "").startswith("openrouter/"):
            return self.resolved_openrouter_key()
        return self.resolved_api_key()

    # ----------------------------------------------------------------- model

    @property
    def is_live(self) -> bool:
        """Whether the selected model runs over the Live API."""
        return is_live_selection(self.selected_model)

    @property
    def live_model(self) -> str:
        """The Live API model id implied by the current selection."""
        return live_model_id(self.selected_model)

    def observer_model_id(self) -> str:
        """The litellm id the observer runs on.

        Split mode's whole point is a cheap observer and a smart answerer, but
        an empty field must not mean "no observer" — it means "the same model as
        everything else", which is exactly unified mode's behaviour.
        """
        if self.agent_mode == "split" and self.observer_model.strip():
            return self.observer_model.strip()
        return litellm_model_id(self.selected_model)

    # --------------------------------------------------------------- capture

    def effective_frame_width(self) -> int:
        """The capture width in force, honouring the mode's meaning of detail.

        In live mode the raw ``frame_width`` is the width, and frame detail is a
        separate API-side budget. In non-live mode detail *is* the width — two
        dials on the same pixels would only let them contradict each other — so
        ``frame_width`` is superseded (and hidden in the UI).
        """
        if self.is_live:
            return self.capture.frame_width
        return DETAIL_CAPTURE_WIDTH.get(self.capture.media_resolution, 768)

    def effective_capture(self) -> CaptureSettings:
        """Capture settings as the capture thread should actually run them."""
        capture = self.capture.model_copy()
        capture.frame_width = self.effective_frame_width()
        return capture

    # ---------------------------------------------------------------- change

    def copy_deep(self) -> Settings:
        """An independent copy, for editing in the settings window."""
        return Settings.model_validate(self.model_dump())

    def requires_session_restart(self, other: Settings) -> bool:
        """Whether moving from `self` to `other` invalidates the session.

        For a live session, model id, credential, system instruction and journal
        strategy are baked into the websocket's setup message, so changing any
        of them means rotating rather than reconfiguring. For a non-live one
        there is no socket to rotate, but the same edits change what every
        request is built from — and crossing the live/non-live boundary replaces
        the provider outright.

        Observer cadence numbers are deliberately absent: they are read per tick
        and apply in place.

        Args:
            other (Settings): The settings about to be applied.

        Returns:
            bool: True when the session must be restarted for the change to take
                effect.
        """
        return (
            self.selected_model != other.selected_model
            or self.resolved_api_key() != other.resolved_api_key()
            or self.resolved_openrouter_key() != other.resolved_openrouter_key()
            or self.agent_mode != other.agent_mode
            or self.observer_model_id() != other.observer_model_id()
            or self.game_name != other.game_name
            or self.extra_system_prompt != other.extra_system_prompt
            or self.journal.strategy != other.journal.strategy
            or self.capture.media_resolution != other.capture.media_resolution
        )

    def requires_provider_swap(self, other: Settings) -> bool:
        """Whether the change moves across the live/non-live boundary.

        A restart reopens the same kind of session; this asks the sharper
        question of whether the object itself has to be replaced.
        """
        return self.is_live != other.is_live


def default_settings_path() -> Path:
    """The settings file path, honouring ``XDG_CONFIG_HOME``."""
    base = os.environ.get("XDG_CONFIG_HOME", "").strip()
    root = Path(base) if base else Path.home() / ".config"
    return root / "chiron" / "settings.json"


def load_settings(path: str | Path | None = None) -> Settings:
    """Read settings from disk, falling back to defaults.

    A missing file is not an error — it is a first run. A file that exists but
    cannot be parsed is logged and ignored, because refusing to start over a
    malformed config is a worse failure than starting with defaults.

    Args:
        path (str | Path | None): File to read. Defaults to
            :func:`default_settings_path`.

    Returns:
        Settings: The loaded configuration.
    """
    target = Path(path) if path is not None else default_settings_path()
    if not target.exists():
        return Settings()
    try:
        raw: Any = json.loads(target.read_text(encoding="utf-8"))
        return Settings.model_validate(raw)
    except (OSError, json.JSONDecodeError, ValidationError) as error:
        logger.warning(
            "Could not read settings from %s (%s); using defaults", target, error
        )
        return Settings()


def save_settings(settings: Settings, path: str | Path | None = None) -> Path:
    """Write settings to disk atomically with restrictive permissions.

    The file is written to a temporary sibling and renamed, so an interrupted
    write can never leave a half-written config behind.

    Args:
        settings (Settings): The configuration to persist.
        path (str | Path | None): Destination. Defaults to
            :func:`default_settings_path`.

    Returns:
        Path: The path written to.
    """
    target = Path(path) if path is not None else default_settings_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(settings.model_dump(), indent=2, sort_keys=False)

    handle = tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=str(target.parent),
        prefix=f".{target.name}.",
        suffix=".tmp",
        delete=False,
    )
    try:
        with handle:
            handle.write(payload + "\n")
        os.chmod(handle.name, 0o600)
        os.replace(handle.name, target)
    except OSError:
        Path(handle.name).unlink(missing_ok=True)
        raise
    return target


@dataclass
class RemovalPlan:
    """What resetting to a fresh install would delete.

    Built before anything is touched, so the user can be shown the exact list and
    told what it costs them — the settings file holds the API key, and deleting
    it means pasting the key in again.

    Attributes:
        settings_file (Path | None): The settings file, if it exists.
        directory (Path | None): The config directory, if it exists and would be
            left empty.
        strangers (list[Path]): Anything else in the config directory. Chiron did
            not put these there and will not remove them.
        holds_api_key (bool): Whether the settings file has a key saved in it.
    """

    settings_file: Path | None = None
    directory: Path | None = None
    strangers: list[Path] = field(default_factory=list)
    holds_api_key: bool = False

    @property
    def is_empty(self) -> bool:
        """Whether there is nothing to remove."""
        return self.settings_file is None and self.directory is None

    def describe(self) -> list[str]:
        """Human-readable lines describing the plan."""
        lines: list[str] = []
        if self.settings_file is not None:
            lines.append(f"delete  {self.settings_file}")
        if self.directory is not None:
            lines.append(f"delete  {self.directory}{os.sep} (directory)")
        for stranger in self.strangers:
            lines.append(f"keep    {stranger} (not Chiron's; left alone)")
        return lines


def plan_removal(path: str | Path | None = None) -> RemovalPlan:
    """Work out what a fresh install would delete, without deleting anything.

    Args:
        path (str | Path | None): Settings file to target. Defaults to
            :func:`default_settings_path`.

    Returns:
        RemovalPlan: The paths involved.
    """
    target = Path(path) if path is not None else default_settings_path()
    plan = RemovalPlan()

    if target.is_file():
        plan.settings_file = target
        try:
            raw = json.loads(target.read_text(encoding="utf-8"))
            plan.holds_api_key = any(
                str(raw.get(name, "")).strip()
                for name in ("api_key", "openrouter_api_key")
            )
        except (OSError, json.JSONDecodeError, AttributeError):
            plan.holds_api_key = False

    directory = target.parent
    # Only ever consider Chiron's own directory. Pointing --settings at a file in
    # a shared directory must not put that directory up for deletion.
    if directory.is_dir() and directory.name == "chiron":
        remaining = [p for p in sorted(directory.iterdir()) if p != target]
        if remaining:
            plan.strangers = remaining
        else:
            plan.directory = directory
    return plan


def remove_configuration(path: str | Path | None = None) -> RemovalPlan:
    """Delete Chiron's configuration, returning what was removed.

    No backup is left behind. A saved API key is a secret, and quietly keeping a
    copy of it under another name would defeat the point of the person asking for
    a clean slate.

    Args:
        path (str | Path | None): Settings file to remove. Defaults to
            :func:`default_settings_path`.

    Returns:
        RemovalPlan: What was actually removed.

    Raises:
        OSError: If a file exists but cannot be deleted.
    """
    plan = plan_removal(path)
    if plan.settings_file is not None:
        plan.settings_file.unlink()
        logger.info("Removed settings file %s", plan.settings_file)
    if plan.directory is not None:
        try:
            plan.directory.rmdir()
            logger.info("Removed config directory %s", plan.directory)
        except OSError as error:
            # Something appeared in the directory between planning and removing.
            logger.info("Left %s in place: %s", plan.directory, error)
            plan.directory = None
    return plan


__all__ = [
    "API_KEY_ENV_VARS",
    "DEFAULT_LIVE_MODEL",
    "DEFAULT_MODEL",
    "DEFAULT_SIDECAR_MODEL",
    "DETAIL_CAPTURE_WIDTH",
    "LIVE_CONTEXT_TOKENS",
    "LIVE_MODEL_CHOICES",
    "LIVE_PREFIX",
    "OPENROUTER_API_KEY_ENV_VARS",
    "SIDECAR_MODEL_CHOICES",
    "AgentMode",
    "CaptureSettings",
    "HotkeySettings",
    "JournalSettings",
    "JournalStrategy",
    "MediaResolution",
    "ObserverSettings",
    "OverlaySettings",
    "RemovalPlan",
    "Settings",
    "TriggerStrategy",
    "default_settings_path",
    "is_live_selection",
    "litellm_model_id",
    "live_model_id",
    "load_settings",
    "plan_removal",
    "remove_configuration",
    "save_settings",
]
