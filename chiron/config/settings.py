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
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, ValidationError

logger = logging.getLogger(__name__)

#: The Live API model the overlay talks to. Every Live model still served is a
#: native-audio one — the text-out half-cascade models were retired — so Chiron
#: takes the model's speech and renders its own transcription as text.
DEFAULT_LIVE_MODEL = "gemini-3.1-flash-live-preview"

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

JournalStrategy = Literal["tool_call", "sidecar"]
MediaResolution = Literal["low", "medium", "high"]


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
        media_resolution (MediaResolution): Token budget the API spends per
            frame. ``low`` is roughly 260 tokens per frame.
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
    """

    width: int = Field(default=420, ge=280, le=1600)
    height: int = Field(default=560, ge=240, le=1400)
    position_x: int | None = None
    position_y: int | None = None
    opacity: float = Field(default=0.92, ge=0.2, le=1.0)
    font_size: int = Field(default=11, ge=7, le=24)
    always_on_top: bool = True
    start_hidden: bool = False


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
    """

    toggle_overlay: str = "ctrl+alt+c"
    open_settings: str = "ctrl+alt+s"
    toggle_watching: str = "ctrl+alt+w"
    start_watching: str = ""
    stop_watching: str = ""


class Settings(BaseModel):
    """The whole of Chiron's user-editable configuration.

    Attributes:
        api_key (str): Gemini API key. Empty means "look in the environment".
        live_model (str): Live API model id.
        game_name (str): Optional name of the game being played, folded into the
            system instruction so the model knows what it is looking at.
        extra_system_prompt (str): Free-form additions to the system instruction.
        capture (CaptureSettings): Screen capture configuration.
        journal (JournalSettings): Journal strategy and cadence.
        overlay (OverlaySettings): Overlay appearance.
        hotkeys (HotkeySettings): Global shortcuts.
    """

    api_key: str = ""
    live_model: str = DEFAULT_LIVE_MODEL
    game_name: str = ""
    extra_system_prompt: str = ""

    capture: CaptureSettings = Field(default_factory=CaptureSettings)
    journal: JournalSettings = Field(default_factory=JournalSettings)
    overlay: OverlaySettings = Field(default_factory=OverlaySettings)
    hotkeys: HotkeySettings = Field(default_factory=HotkeySettings)

    def resolved_api_key(self) -> str:
        """The key to authenticate with: the saved one, else the environment."""
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

    def copy_deep(self) -> Settings:
        """An independent copy, for editing in the settings window."""
        return Settings.model_validate(self.model_dump())

    def requires_session_restart(self, other: Settings) -> bool:
        """Whether moving from `self` to `other` invalidates the live session.

        Model id, credential, system instruction and journal strategy are all
        baked into the websocket's setup message, so changing any of them means
        the current session has to be rotated rather than merely reconfigured.

        Args:
            other (Settings): The settings about to be applied.

        Returns:
            bool: True when the session must be restarted for the change to take
                effect.
        """
        return (
            self.live_model != other.live_model
            or self.resolved_api_key() != other.resolved_api_key()
            or self.game_name != other.game_name
            or self.extra_system_prompt != other.extra_system_prompt
            or self.journal.strategy != other.journal.strategy
            or self.capture.media_resolution != other.capture.media_resolution
        )


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
            plan.holds_api_key = bool(str(raw.get("api_key", "")).strip())
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
    "DEFAULT_SIDECAR_MODEL",
    "LIVE_MODEL_CHOICES",
    "SIDECAR_MODEL_CHOICES",
    "CaptureSettings",
    "HotkeySettings",
    "JournalSettings",
    "JournalStrategy",
    "RemovalPlan",
    "plan_removal",
    "remove_configuration",
    "MediaResolution",
    "OverlaySettings",
    "Settings",
    "default_settings_path",
    "load_settings",
    "save_settings",
]
