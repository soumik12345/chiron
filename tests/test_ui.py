"""Widget construction, transcript rendering and the settings form round-trip."""

from __future__ import annotations

import pytest

from chiron.config.settings import Settings
from chiron.journal.log import JournalEntry
from chiron.ui.hotkeys import HotkeyError, normalise_hotkey, parse_hotkey, to_pynput
from chiron.ui.overlay import OverlayWindow, format_message_text
from chiron.ui.settings_window import SettingsWindow


@pytest.fixture
def overlay(qapp):
    return OverlayWindow(Settings().overlay)


@pytest.fixture
def settings_window(qapp):
    return SettingsWindow(Settings())


# ------------------------------------------------------------------ overlay


def test_overlay_starts_empty(overlay):
    assert overlay.transcript.toPlainText().strip() == ""


def test_streaming_builds_one_message(overlay):
    overlay.append_user("what killed me?")
    overlay.start_response()
    overlay.append_delta("A skeleton ")
    overlay.append_delta("on the bridge.")
    overlay.end_response("A skeleton on the bridge.")

    text = overlay.transcript.toPlainText()
    assert "what killed me?" in text
    assert text.count("A skeleton on the bridge.") == 1


def test_journal_entries_appear_inline(overlay):
    overlay.append_journal(
        JournalEntry(
            timestamp=1_700_000_000.0, note="Lit the bonfire.", category="progress"
        )
    )
    assert "Lit the bonfire." in overlay.transcript.toPlainText()


def test_transcript_is_bounded(overlay):
    for index in range(200):
        overlay.append_system(f"line {index}")
    assert len(overlay._messages) <= 80
    assert "line 199" in overlay.transcript.toPlainText()


def test_prompt_submission_clears_the_input(overlay):
    seen = []
    overlay.promptSubmitted.connect(seen.append)
    overlay.input.setText("  where now?  ")
    overlay._submit()
    assert seen == ["where now?"]
    assert overlay.input.text() == ""


def test_empty_prompts_are_not_submitted(overlay):
    seen = []
    overlay.promptSubmitted.connect(seen.append)
    overlay.input.setText("   ")
    overlay._submit()
    assert seen == []


def test_watch_button_shows_state_and_asks_for_the_other_one(overlay):
    requested = []
    overlay.watchToggled.connect(requested.append)

    assert overlay.watching is False
    assert "Not watching" in overlay.watch_button.text()

    overlay.watch_button.click()
    assert requested == [True], "clicking asks the app; the app decides"

    overlay.set_watching(True, "ctrl+alt+w")
    assert "Watching" in overlay.watch_button.text()
    assert "ctrl+alt+w" in overlay.watch_button.toolTip()

    overlay.watch_button.click()
    assert requested == [True, False]


def test_status_updates_the_header(overlay):
    overlay.set_status("live", "gemini-3.1-flash-live-preview")
    assert "live" in overlay.status_text.text()


def test_appearance_applies(overlay):
    settings = Settings().overlay
    settings.opacity = 0.5
    settings.width, settings.height = 500, 400
    overlay.apply_settings(settings)
    assert overlay.windowOpacity() == pytest.approx(0.5, abs=0.01)
    assert overlay.width() == 500


def test_message_formatting_escapes_html():
    assert "&lt;script&gt;" in format_message_text("<script>")
    assert format_message_text("**bold**") == "<b>bold</b>"
    assert "<br>" in format_message_text("one\ntwo")


# ------------------------------------------------------------------ hotkeys


@pytest.mark.parametrize(
    "spec,modifiers,key",
    [
        ("ctrl+alt+c", {"ctrl", "alt"}, "c"),
        ("Ctrl+Shift+F5", {"ctrl", "shift"}, "F5"),
        ("super+space", {"super"}, "space"),
        ("ctrl+alt+esc", {"ctrl", "alt"}, "Escape"),
    ],
)
def test_hotkey_parsing(spec, modifiers, key):
    assert parse_hotkey(spec) == (modifiers, key)


@pytest.mark.parametrize("spec", ["", "ctrl+", "ctrl+a+b", "ctrl+nonsense"])
def test_bad_hotkeys_are_rejected(spec):
    with pytest.raises(HotkeyError):
        parse_hotkey(spec)


def test_hotkey_normalisation_orders_modifiers():
    assert normalise_hotkey("ALT+Ctrl+c") == "ctrl+alt+c"


def test_pynput_rendering():
    assert to_pynput("ctrl+alt+c") == "<ctrl>+<alt>+c"
    assert to_pynput("ctrl+space") == "<ctrl>+<space>"


# ----------------------------------------------------------------- settings


def test_construction_raises_nothing_from_signal_handlers(qapp, capsys):
    """Populating a combo fires its change signal mid-build; nothing may react yet."""
    SettingsWindow(Settings())
    assert "Traceback" not in capsys.readouterr().err


def test_settings_window_round_trips_defaults(settings_window):
    assert settings_window.collect().model_dump() == Settings().model_dump()


def test_editing_a_field_shows_up_in_collect(settings_window):
    settings_window.game_name_edit.setText("Elden Ring")
    settings_window.baseline_spin.setValue(8.0)
    settings_window.opacity_slider.setValue(60)

    collected = settings_window.collect()
    assert collected.game_name == "Elden Ring"
    assert collected.capture.baseline_interval_seconds == 8.0
    assert collected.overlay.opacity == pytest.approx(0.6)


def test_saving_emits_the_new_settings(settings_window):
    received = []
    settings_window.settingsSaved.connect(received.append)
    settings_window.game_name_edit.setText("Hades")
    settings_window._save()
    assert received[0].game_name == "Hades"


def test_load_repopulates_the_form(settings_window):
    settings = Settings()
    settings.journal.strategy = "sidecar"
    settings.capture.frame_width = 1024
    settings.hotkeys.toggle_overlay = "ctrl+shift+g"

    settings_window.load(settings)

    assert settings_window.sidecar_radio.isChecked()
    assert settings_window.frame_width_spin.value() == 1024
    assert settings_window.toggle_hotkey_edit.text() == "ctrl+shift+g"
    assert settings_window.collect().journal.strategy == "sidecar"


def test_watch_settings_round_trip(settings_window):
    settings = Settings()
    settings.capture.watch_on_launch = True
    settings.hotkeys.toggle_watching = "ctrl+alt+g"
    settings.hotkeys.start_watching = "ctrl+alt+shift+g"
    settings.hotkeys.stop_watching = ""

    settings_window.load(settings)

    assert settings_window.watch_on_launch_check.isChecked() is True
    assert settings_window.watch_toggle_hotkey_edit.text() == "ctrl+alt+g"
    assert settings_window.start_watch_hotkey_edit.text() == "ctrl+alt+shift+g"

    collected = settings_window.collect()
    assert collected.capture.watch_on_launch is True
    assert collected.hotkeys.toggle_watching == "ctrl+alt+g"
    assert collected.hotkeys.start_watching == "ctrl+alt+shift+g"
    assert collected.hotkeys.stop_watching == "", "an unbound key stays unbound"


def test_watching_is_not_on_by_default_in_the_form(settings_window):
    assert settings_window.watch_on_launch_check.isChecked() is False
    assert settings_window.collect().capture.watch_on_launch is False


def test_sidecar_fields_track_the_selected_strategy(settings_window):
    settings_window.tool_call_radio.setChecked(True)
    assert settings_window.sidecar_model_combo.isEnabled() is False

    settings_window.sidecar_radio.setChecked(True)
    assert settings_window.sidecar_model_combo.isEnabled() is True


def test_restart_badge_appears_only_for_session_level_changes(settings_window):
    settings_window.opacity_slider.setValue(45)
    assert settings_window.restart_badge.text() == ""

    settings_window.live_model_combo.setCurrentText("gemini-live-2.5-flash-preview")
    assert "reconnect" in settings_window.restart_badge.text()


def test_token_estimate_reacts_to_the_shutter(settings_window):
    settings_window.baseline_spin.setValue(4.0)
    fast = settings_window.burn_label.text()
    settings_window.baseline_spin.setValue(20.0)
    slow = settings_window.burn_label.text()
    assert fast != slow
    assert "tokens/min" in slow


def test_key_source_is_reported(settings_window, monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    settings_window.api_key_edit.setText("")
    settings_window._refresh_derived()
    assert "No key" in settings_window.key_source_label.text()

    settings_window.api_key_edit.setText("AIzaXXX")
    settings_window._refresh_derived()
    assert "saved here" in settings_window.key_source_label.text()


def test_every_page_is_reachable(settings_window):
    assert settings_window.nav.count() == settings_window.pages.count() == 6
    for row in range(settings_window.nav.count()):
        settings_window.nav.setCurrentRow(row)
        assert settings_window.pages.currentIndex() == row
