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

Chiron v3 always has two agents. ``observer_model`` names the required Gemini
Live observer and ``responder_model`` names an ordinary Google AI Studio or
OpenRouter completion model. Older selected-model settings are accepted by the
pre-validator, but only the dual-agent shape is written back to disk.
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

#: The Live API model Chiron-Observer talks to. It is native-audio, but Observer
#: audio and content are discarded; only journal tool calls cross the boundary.
DEFAULT_LIVE_MODEL = "gemini-3.1-flash-live-preview"

#: Provider-qualified defaults for the two v3 agents.
DEFAULT_OBSERVER_MODEL = LIVE_PREFIX + DEFAULT_LIVE_MODEL
DEFAULT_RESPONDER_MODEL = "gemini/gemini-3.6-flash"

#: Known Live API model ids, offered in the settings page combo box. The field is
#: editable, so a newer id can always be typed in.
LIVE_MODEL_CHOICES: list[str] = [
    "gemini-3.1-flash-live-preview",
    "gemini-2.5-flash-native-audio-latest",
    "gemini-2.5-flash-native-audio-preview-12-2025",
]

#: Environment variables consulted, in order, when no key is saved.
API_KEY_ENV_VARS = ("GEMINI_API_KEY", "GOOGLE_API_KEY")

#: Environment variable consulted when no OpenRouter key is saved.
OPENROUTER_API_KEY_ENV_VARS = ("OPENROUTER_API_KEY",)

MediaResolution = Literal["low", "medium", "high"]
ResponderMode = Literal["fixed_horizon", "react"]
QuestionFramePolicy = Literal["latest", "immediate"]


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


class CaptureSettings(BaseModel):
    """Fixed-interval screen capture shared by both agents.

    Attributes:
        monitor_index (int): ``mss`` monitor number. 0 is the virtual "all
            monitors" screen; 1 is the primary display.
        interval_seconds (float): Seconds between scheduled Observer frames.
            Gemini Live accepts at most one video frame per second.
        question_frame_policy (QuestionFramePolicy): Reuse the latest scheduled
            frame, or capture exactly one new frame for a question.
        frame_width (int): Frames are downscaled to this width before encoding.
        jpeg_quality (int): JPEG quality (1-95) for encoded frames.
        stamp_timestamp (bool): Draw the capture time into the frame's corner so
            the model has an explicit "now" to reason about.
        media_resolution (MediaResolution): Gemini Live's Observer-side visual
            token budget. It is independent of ``frame_width`` in v3.
        watch_on_launch (bool): Begin watching the moment Chiron starts. Off by
            default: a screen recorder that switches itself on when you log in is
            not something anyone should have to opt out of.
    """

    monitor_index: int = 1
    interval_seconds: float = Field(default=5.0, ge=1.0, le=60.0)
    question_frame_policy: QuestionFramePolicy = "latest"
    frame_width: int = Field(default=768, ge=256, le=1920)
    jpeg_quality: int = Field(default=60, ge=10, le=95)
    stamp_timestamp: bool = True
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
        if self.interval_seconds <= 0:
            return 0.0
        return (60.0 / self.interval_seconds) * tokens_per_frame


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
        observer_model (str): Provider-qualified Gemini Live model used only by
            Chiron-Observer.
        responder_model (str): Non-live Google or OpenRouter model used only by
            Chiron-Responder.
        responder_mode (ResponderMode): Fixed-horizon or ReAct execution.
        game_name (str): Optional name of the game being played, folded into the
            system instruction so the model knows what it is looking at.
        observer_system_prompt (str): Optional additions to Observer behavior.
        responder_system_prompt (str): Optional additions to Responder behavior.
        capture (CaptureSettings): Screen capture configuration.
        overlay (OverlaySettings): Overlay appearance.
        hotkeys (HotkeySettings): Global shortcuts.
    """

    api_key: str = ""
    openrouter_api_key: str = ""
    observer_model: str = DEFAULT_OBSERVER_MODEL
    responder_model: str = DEFAULT_RESPONDER_MODEL
    responder_mode: ResponderMode = "fixed_horizon"
    game_name: str = ""
    observer_system_prompt: str = ""
    responder_system_prompt: str = ""

    capture: CaptureSettings = Field(default_factory=CaptureSettings)
    overlay: OverlaySettings = Field(default_factory=OverlaySettings)
    hotkeys: HotkeySettings = Field(default_factory=HotkeySettings)

    @model_validator(mode="before")
    @classmethod
    def _migrate_dual_agents(cls, data: Any) -> Any:
        """Accept v0-v2 settings and emit one unambiguous in-memory v3 shape."""
        if not isinstance(data, dict):
            return data
        migrated = dict(data)

        selected = str(migrated.get("selected_model") or "").strip()
        legacy_live = str(migrated.get("live_model") or "").strip()
        is_legacy = bool(selected or legacy_live)
        if not selected and legacy_live:
            selected = (
                legacy_live
                if is_live_selection(legacy_live)
                else LIVE_PREFIX + legacy_live
            )
        if selected:
            if is_live_selection(selected):
                migrated["observer_model"] = selected
                migrated.setdefault("responder_model", DEFAULT_RESPONDER_MODEL)
            else:
                migrated["observer_model"] = DEFAULT_OBSERVER_MODEL
                migrated["responder_model"] = selected
        else:
            observer = str(migrated.get("observer_model") or "").strip()
            responder = str(migrated.get("responder_model") or "").strip()
            if observer and "/" not in observer:
                observer = LIVE_PREFIX + observer
            migrated["observer_model"] = observer or DEFAULT_OBSERVER_MODEL
            migrated["responder_model"] = responder or DEFAULT_RESPONDER_MODEL

        # v2 used ``observer_model`` for a split request/response observer. The
        # presence of the old selected-model switch disambiguates that shape;
        # the value was intentionally ignored above unless it was the selection.
        if is_legacy and not is_live_selection(selected):
            migrated["observer_model"] = DEFAULT_OBSERVER_MODEL

        capture = migrated.get("capture")
        if isinstance(capture, dict):
            capture = dict(capture)
            if "interval_seconds" not in capture:
                baseline = capture.get("baseline_interval_seconds")
                try:
                    value = float(baseline)
                except (TypeError, ValueError):
                    value = 5.0
                capture["interval_seconds"] = value if 1.0 <= value <= 60.0 else 5.0
            for name in (
                "baseline_interval_seconds",
                "burst_interval_seconds",
                "burst_duration_seconds",
                "scene_change_enabled",
                "scene_change_threshold",
            ):
                capture.pop(name, None)
            migrated["capture"] = capture

        migrated.pop("journal", None)
        migrated.pop("agent_mode", None)
        migrated.pop("observer", None)
        migrated.pop("selected_model", None)
        migrated.pop("live_model", None)

        old_prompt = str(migrated.get("extra_system_prompt") or "")
        if old_prompt:
            migrated.setdefault("observer_system_prompt", old_prompt)
            migrated.setdefault("responder_system_prompt", old_prompt)
        return migrated

    @model_validator(mode="after")
    def _validate_agent_models(self) -> Settings:
        """Keep the two provider roles structurally disjoint."""
        if not is_live_selection(self.observer_model):
            raise ValueError("observer_model must be a live/<gemini-model> id")
        responder = self.responder_model.strip()
        if not responder.startswith(("gemini/", "openrouter/")):
            raise ValueError(
                "responder_model must use the gemini/ or openrouter/ provider"
            )
        return self

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

    # --------------------------------------------------------------- capture

    def effective_frame_width(self) -> int:
        """The one capture width delivered to both v3 agents."""
        return self.capture.frame_width

    def effective_capture(self) -> CaptureSettings:
        """Capture settings as the capture thread should actually run them."""
        capture = self.capture.model_copy()
        capture.frame_width = self.effective_frame_width()
        return capture

    # ---------------------------------------------------------------- change

    def copy_deep(self) -> Settings:
        """An independent copy, for editing in the settings window."""
        return Settings.model_validate(self.model_dump())

    def requires_observer_reconnect(self, other: Settings) -> bool:
        """Whether Observer websocket configuration changed."""
        return (
            self.observer_model != other.observer_model
            or self.resolved_api_key() != other.resolved_api_key()
            or self.game_name != other.game_name
            or self.observer_system_prompt != other.observer_system_prompt
            or self.capture.media_resolution != other.capture.media_resolution
        )

    def requires_responder_rebuild(self, other: Settings) -> bool:
        """Whether the Responder adapter must be rebuilt, preserving memory."""
        return (
            self.responder_model != other.responder_model
            or self.responder_mode != other.responder_mode
            or self.key_for_model(self.responder_model)
            != other.key_for_model(other.responder_model)
            or self.game_name != other.game_name
            or self.responder_system_prompt != other.responder_system_prompt
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
    "DEFAULT_OBSERVER_MODEL",
    "DEFAULT_RESPONDER_MODEL",
    "LIVE_MODEL_CHOICES",
    "LIVE_PREFIX",
    "OPENROUTER_API_KEY_ENV_VARS",
    "CaptureSettings",
    "HotkeySettings",
    "MediaResolution",
    "OverlaySettings",
    "QuestionFramePolicy",
    "RemovalPlan",
    "ResponderMode",
    "Settings",
    "default_settings_path",
    "is_live_selection",
    "live_model_id",
    "load_settings",
    "plan_removal",
    "remove_configuration",
    "save_settings",
]
