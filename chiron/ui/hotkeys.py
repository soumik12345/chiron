"""System-wide hotkeys, so the overlay can be summoned from inside a game.

A game has keyboard focus; Qt shortcuts only fire when the overlay does, which
would make the panel unreachable exactly when it is wanted. So the binding has to
be grabbed from the X server itself.

The default backend is `XGrabKey` through python-xlib: pure Python, no build
step, and precisely the right tool on the X11/GNOME target this version aims at.
`pynput` is used instead when it happens to be installed, since it also covers
Wayland-ish and non-Linux setups; it is not a dependency because on Linux it
drags in `evdev`, which needs a C toolchain to install.

Grabs can fail — a desktop environment or another app may already own
`ctrl+alt+c`. That is reported through :attr:`GlobalHotkeyManager.failed` rather
than raised, because a hotkey clash should cost you a hotkey, not the app.
"""

from __future__ import annotations

import logging
import threading
from typing import Callable

from PySide6.QtCore import QObject, Signal

logger = logging.getLogger(__name__)

#: Modifier token -> X11 mask bit. Populated lazily to keep Xlib import optional.
_MODIFIER_TOKENS = {
    "ctrl": "control",
    "control": "control",
    "alt": "alt",
    "shift": "shift",
    "super": "super",
    "win": "super",
    "meta": "super",
    "cmd": "super",
}

#: Friendly key names -> X keysym names.
_KEY_ALIASES = {
    "esc": "Escape",
    "escape": "Escape",
    "space": "space",
    "enter": "Return",
    "return": "Return",
    "tab": "Tab",
    "backspace": "BackSpace",
    "delete": "Delete",
    "insert": "Insert",
    "home": "Home",
    "end": "End",
    "pgup": "Prior",
    "pgdn": "Next",
    "up": "Up",
    "down": "Down",
    "left": "Left",
    "right": "Right",
    "`": "grave",
    "-": "minus",
    "=": "equal",
    "[": "bracketleft",
    "]": "bracketright",
    ";": "semicolon",
    "'": "apostrophe",
    ",": "comma",
    ".": "period",
    "/": "slash",
    "\\": "backslash",
}


class HotkeyError(ValueError):
    """Raised when a hotkey string cannot be understood."""


def parse_hotkey(spec: str) -> tuple[set[str], str]:
    """Split ``"ctrl+alt+c"`` into its modifiers and its key.

    Args:
        spec (str): A hotkey in ``modifier+modifier+key`` form, case-insensitive.

    Returns:
        tuple[set[str], str]: Normalised modifier names (``ctrl``, ``alt``,
            ``shift``, ``super``) and the X keysym name of the key.

    Raises:
        HotkeyError: If the string is empty, has no key, or names two keys.
    """
    tokens = [part.strip().lower() for part in spec.split("+") if part.strip()]
    if not tokens:
        raise HotkeyError("Hotkey is empty")

    modifiers: set[str] = set()
    key: str | None = None
    for token in tokens:
        canonical = _MODIFIER_TOKENS.get(token)
        if canonical is not None:
            modifiers.add({"control": "ctrl"}.get(canonical, canonical))
            continue
        if key is not None:
            raise HotkeyError(f"Hotkey '{spec}' names more than one key")
        if token in _KEY_ALIASES:
            key = _KEY_ALIASES[token]
        elif len(token) == 1:
            key = token
        elif token.startswith("f") and token[1:].isdigit():
            key = token.upper()
        else:
            raise HotkeyError(f"Unknown key '{token}' in hotkey '{spec}'")
    if key is None:
        raise HotkeyError(f"Hotkey '{spec}' has modifiers but no key")
    return modifiers, key


def to_pynput(spec: str) -> str:
    """Render a hotkey in pynput's ``<ctrl>+<alt>+c`` notation."""
    modifiers, key = parse_hotkey(spec)
    order = [m for m in ("ctrl", "alt", "shift", "super") if m in modifiers]
    rendered_key = key if len(key) == 1 else f"<{key.lower()}>"
    return "+".join([f"<{m}>" for m in order] + [rendered_key])


class _XlibBackend:
    """Grabs hotkeys directly from the X server on a listener thread."""

    def __init__(
        self, on_activated: Callable[[str], None], on_failed: Callable[[str], None]
    ) -> None:
        self._on_activated = on_activated
        self._on_failed = on_failed
        self._bindings: dict[str, str] = {}
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()

    @staticmethod
    def available() -> bool:
        """Whether python-xlib can open the current display."""
        try:
            from Xlib import display

            display.Display().close()
            return True
        except Exception:
            return False

    def set_bindings(self, bindings: dict[str, str]) -> None:
        """Replace the bindings, restarting the listener if it is running."""
        self._bindings = dict(bindings)
        if self._thread is not None and self._thread.is_alive():
            self.stop()
            self.start()

    def start(self) -> None:
        """Begin listening for the configured hotkeys."""
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run, name="chiron-hotkeys", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        """Stop listening and release the grabs."""
        self._stop.set()
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=1.0)
        self._thread = None

    def _run(self) -> None:
        """Grab each binding, then poll for key presses until stopped."""
        from Xlib import XK, X, display, error

        try:
            conn = display.Display()
        except Exception as failure:
            self._on_failed(f"Cannot open X display for hotkeys: {failure}")
            return

        root = conn.screen().root
        mask_bits = {
            "ctrl": X.ControlMask,
            "alt": X.Mod1Mask,
            "shift": X.ShiftMask,
            "super": X.Mod4Mask,
        }
        # NumLock and CapsLock show up in the event state, so every combination
        # of them has to be grabbed separately or the hotkey silently stops
        # working the moment someone hits CapsLock.
        noise_masks = (0, X.LockMask, X.Mod2Mask, X.LockMask | X.Mod2Mask)
        relevant = X.ControlMask | X.Mod1Mask | X.ShiftMask | X.Mod4Mask

        grabbed: list[tuple[int, int, str]] = []
        for name, spec in self._bindings.items():
            if not spec.strip():
                continue
            try:
                modifiers, key = parse_hotkey(spec)
            except HotkeyError as failure:
                self._on_failed(str(failure))
                continue
            keysym = XK.string_to_keysym(key)
            keycode = conn.keysym_to_keycode(keysym) if keysym else 0
            if not keycode:
                self._on_failed(f"Unknown key in hotkey '{spec}'")
                continue
            mask = 0
            for modifier in modifiers:
                mask |= mask_bits[modifier]

            catcher = error.CatchError(error.BadAccess)
            for noise in noise_masks:
                root.grab_key(
                    keycode,
                    mask | noise,
                    1,
                    X.GrabModeAsync,
                    X.GrabModeAsync,
                    onerror=catcher,
                )
            conn.sync()
            if catcher.get_error():
                self._on_failed(
                    f"Hotkey '{spec}' is already taken by another application"
                )
                continue
            grabbed.append((keycode, mask, name))
            logger.info("Bound global hotkey %s -> %s", spec, name)

        root.change_attributes(event_mask=X.KeyPressMask)
        try:
            while not self._stop.is_set():
                for _ in range(conn.pending_events()):
                    event = conn.next_event()
                    if event.type != X.KeyPress:
                        continue
                    state = event.state & relevant
                    for keycode, mask, name in grabbed:
                        if event.detail == keycode and state == mask:
                            self._on_activated(name)
                            break
                self._stop.wait(0.02)
        finally:
            for keycode, mask, _ in grabbed:
                for noise in noise_masks:
                    try:
                        root.ungrab_key(keycode, mask | noise)
                    except Exception:  # pragma: no cover - teardown noise
                        logger.debug("Ungrab failed", exc_info=True)
            conn.sync()
            conn.close()


class _PynputBackend:
    """Uses pynput's `GlobalHotKeys` when the package is installed."""

    def __init__(
        self, on_activated: Callable[[str], None], on_failed: Callable[[str], None]
    ) -> None:
        self._on_activated = on_activated
        self._on_failed = on_failed
        self._bindings: dict[str, str] = {}
        self._listener = None

    @staticmethod
    def available() -> bool:
        """Whether pynput can be imported."""
        try:
            import pynput  # noqa: F401

            return True
        except Exception:
            return False

    def set_bindings(self, bindings: dict[str, str]) -> None:
        """Replace the bindings, restarting the listener if it is running."""
        self._bindings = dict(bindings)
        if self._listener is not None:
            self.stop()
            self.start()

    def start(self) -> None:
        """Begin listening for the configured hotkeys."""
        from pynput import keyboard

        mapping = {}
        for name, spec in self._bindings.items():
            if not spec.strip():
                continue
            try:
                mapping[to_pynput(spec)] = lambda bound=name: self._on_activated(bound)
            except HotkeyError as failure:
                self._on_failed(str(failure))
        if not mapping:
            return
        self._listener = keyboard.GlobalHotKeys(mapping)
        self._listener.start()

    def stop(self) -> None:
        """Stop listening."""
        if self._listener is not None:
            self._listener.stop()
            self._listener = None


class GlobalHotkeyManager(QObject):
    """Binds global shortcuts and reports when they fire.

    Signals:
        activated (str): The name of a binding that was pressed.
        failed (str): A binding could not be registered, with the reason.

    Attributes:
        backend_name (str): Which backend is in use — ``xlib``, ``pynput`` or
            ``none`` when no global hotkey mechanism is available.
    """

    activated = Signal(str)
    failed = Signal(str)

    def __init__(self, parent: QObject | None = None) -> None:
        """Choose a backend for the current environment."""
        super().__init__(parent)
        self._backend = None
        self.backend_name = "none"
        if _XlibBackend.available():
            self._backend = _XlibBackend(self.activated.emit, self.failed.emit)
            self.backend_name = "xlib"
        elif _PynputBackend.available():
            self._backend = _PynputBackend(self.activated.emit, self.failed.emit)
            self.backend_name = "pynput"
        else:
            logger.warning("No global hotkey backend available")

    def set_bindings(self, bindings: dict[str, str]) -> None:
        """Install `bindings` as ``{name: hotkey}``, replacing any existing set."""
        if self._backend is None:
            if any(spec.strip() for spec in bindings.values()):
                self.failed.emit(
                    "Global hotkeys are unavailable on this display server"
                )
            return
        self._backend.set_bindings(bindings)

    def start(self) -> None:
        """Begin listening."""
        if self._backend is not None:
            self._backend.start()

    def stop(self) -> None:
        """Stop listening and release the grabs."""
        if self._backend is not None:
            self._backend.stop()


def normalise_hotkey(spec: str) -> str:
    """Return a canonical ``ctrl+alt+c`` rendering of `spec`.

    Args:
        spec (str): Any accepted hotkey spelling.

    Returns:
        str: The canonical form, ordered ctrl, alt, shift, super, then the key.

    Raises:
        HotkeyError: If the string cannot be parsed.
    """
    modifiers, key = parse_hotkey(spec)
    order = [m for m in ("ctrl", "alt", "shift", "super") if m in modifiers]
    return "+".join(order + [key if len(key) == 1 else key.lower()])


__all__ = [
    "GlobalHotkeyManager",
    "HotkeyError",
    "normalise_hotkey",
    "parse_hotkey",
    "to_pynput",
]
