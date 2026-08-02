"""Widget construction, transcript rendering and the settings form round-trip."""

from __future__ import annotations

import pytest
from PySide6.QtCore import Qt

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

    settings_window.model_picker.choose("live/gemini-live-2.5-flash-preview")
    assert "restart" in settings_window.restart_badge.text()


def test_the_badge_says_when_a_save_changes_modes(settings_window):
    settings_window.model_picker.choose("gemini/gemini-2.5-flash")
    assert "switch modes" in settings_window.restart_badge.text()


# ------------------------------------------------------------ model picking


def test_the_picker_returns_an_id_for_a_listed_model(settings_window):
    picker = settings_window.model_picker
    picker.choose("gemini/gemini-2.5-flash")

    assert picker.selection() == "gemini/gemini-2.5-flash"
    assert settings_window.collect().selected_model == "gemini/gemini-2.5-flash"
    assert "AI STUDIO" in picker.provider_label.text(), "the card says where it runs"
    assert picker.id_label.text() == "gemini/gemini-2.5-flash"


def test_the_picker_accepts_a_model_it_has_never_heard_of(settings_window):
    """A catalogue fetched over the network cannot be the only way to name a model."""
    settings_window.model_picker.choose("openrouter/some/model-from-today")

    assert (
        settings_window.collect().selected_model == "openrouter/some/model-from-today"
    )
    assert "not in any catalogue" in settings_window.model_note_label.text()


def test_refreshing_the_catalogue_keeps_the_current_selection(settings_window):
    from chiron.models.catalogue import STATIC_MODELS

    settings_window.model_picker.set_selection("openrouter/typed/by-hand")
    settings_window.set_models(list(STATIC_MODELS))

    assert settings_window.collect().selected_model == "openrouter/typed/by-hand"


def test_the_observer_picker_offers_no_live_models(settings_window):
    picker = settings_window.observer_model_picker
    ids = picker.model_ids()
    assert ids, "the static list is not empty"
    assert not any(str(i).startswith("live/") for i in ids)


def test_the_card_shows_the_inherit_state_rather_than_looking_empty(settings_window):
    picker = settings_window.observer_model_picker
    picker.set_selection("")
    assert "model chosen above" in picker.name_label.text()
    assert picker.id_label.text() == ""


# ----------------------------------------------------------- picker search


@pytest.fixture
def picker(qapp):
    """A picker holding a small, known catalogue, with its popup built."""
    from chiron.models.catalogue import STATIC_MODELS
    from chiron.ui.model_picker import ModelPicker

    widget = ModelPicker()
    widget.set_models(list(STATIC_MODELS))
    widget.open_popup()
    widget._popup.hide()  # built and filled, but not on screen for a test
    return widget


def _rows(picker):
    """The model ids currently listed in the popup."""
    listing = picker._popup.list
    return [
        listing.item(row).data(Qt.ItemDataRole.UserRole).id
        for row in range(listing.count())
    ]


def test_search_matches_name_id_and_vendor(picker):
    popup = picker._popup
    popup.search.setText("flash")
    assert _rows(picker), "several models are called flash"
    assert all("flash" in model_id for model_id in _rows(picker))

    popup.search.setText("openrouter/")
    assert all(model_id.startswith("openrouter/") for model_id in _rows(picker))


def test_search_words_all_have_to_match(picker):
    popup = picker._popup
    popup.search.setText("gemini 2.5")
    assert _rows(picker)
    assert all("2.5" in model_id for model_id in _rows(picker))

    popup.search.setText("gemini nonsense")
    assert _rows(picker) == []


def test_a_provider_chip_filters_the_list(picker):
    from chiron.models.providers import GOOGLE

    picker._popup._set_provider(GOOGLE)
    assert _rows(picker)
    assert all(model_id.startswith("gemini/") for model_id in _rows(picker))

    picker._popup._set_provider("all")
    assert any(model_id.startswith("live/") for model_id in _rows(picker))


def test_chips_are_offered_only_for_providers_that_have_models(picker):
    labels = [chip.text() for chip in picker._popup._chips]
    assert labels[0].startswith("All")
    assert any(label.startswith("Live API") for label in labels)
    assert all("0" != label.split()[-1] for label in labels), "no empty filters"


def test_a_pasted_id_nothing_matches_is_offered_anyway(picker):
    """The escape hatch stays one keystroke deep instead of becoming a dead end."""
    picker._popup.search.setText("openrouter/brand/new-model")
    assert _rows(picker) == ["openrouter/brand/new-model"]

    picker._popup._choose(picker._popup.list.item(0))
    assert picker.selection() == "openrouter/brand/new-model"


def test_a_half_typed_word_is_not_offered_as_an_id(picker):
    """ "gemi" is a search in progress, not a model nobody has heard of."""
    picker._popup.search.setText("zzzz")
    assert _rows(picker) == []


def test_choosing_a_row_announces_the_change(picker):
    seen = []
    picker.selectionChanged.connect(seen.append)

    picker._popup.search.setText("gemini/gemini-2.5-flash")
    picker._popup._choose(picker._popup.list.item(0))

    assert seen == ["gemini/gemini-2.5-flash"]
    assert picker.selection() == "gemini/gemini-2.5-flash"


def test_arrows_and_enter_work_without_leaving_the_search_box(picker):
    """Typing and steering are one gesture; tabbing between them would be wrong."""
    from PySide6.QtGui import QKeyEvent
    from PySide6.QtWidgets import QApplication

    popup = picker._popup
    popup.search.setText("gemini")
    first = _rows(picker)[0]

    def press(key):
        QApplication.sendEvent(
            popup.search,
            QKeyEvent(QKeyEvent.Type.KeyPress, key, Qt.KeyboardModifier.NoModifier),
        )

    press(Qt.Key.Key_Down)
    assert popup.list.currentRow() == 1, "the list moved, not the caret"

    press(Qt.Key.Key_Up)
    press(Qt.Key.Key_Return)
    assert picker.selection() == first


def test_setting_a_selection_programmatically_is_silent(picker):
    """load() populates the form; it is not the user changing their mind."""
    seen = []
    picker.selectionChanged.connect(seen.append)
    picker.set_selection("gemini/gemini-2.5-flash")
    assert seen == []


def test_an_empty_choice_is_offered_only_when_it_is_allowed(qapp):
    from chiron.models.catalogue import STATIC_MODELS
    from chiron.ui.model_picker import ModelPicker

    optional = ModelPicker(allow_empty=True, empty_label="Use the main model")
    optional.set_models(list(STATIC_MODELS))
    optional.open_popup()
    optional._popup.hide()

    assert _rows(optional)[0] == "", "the first row clears the selection"

    optional._popup.search.setText("flash")
    assert "" not in _rows(optional), "searching is asking for a model"


def test_prices_are_rendered_per_million_tokens():
    from chiron.models.catalogue import STATIC_MODELS
    from chiron.ui.model_picker import format_context, format_prices

    priced = next(m for m in STATIC_MODELS if m.pricing_known)
    assert format_prices(priced) == "$0.30 / $2.50"
    assert format_prices(next(m for m in STATIC_MODELS if not m.pricing_known)) == (
        "no price"
    )
    assert format_context(1_000_000) == "1.0M ctx"
    assert format_context(128_000) == "128K ctx"
    assert format_context(None) == ""


# ------------------------------------------------------------- mode effects


def test_choosing_a_non_live_model_hides_frame_width(settings_window):
    """Detail *is* the width there; two dials on the same pixels could disagree."""
    assert settings_window.frame_width_spin.isHidden() is False

    settings_window.model_picker.choose("gemini/gemini-2.5-flash")

    assert settings_window.frame_width_spin.isHidden() is True
    assert "512 px" in settings_window.frame_width_note_label.text()


def test_observer_settings_are_inert_under_a_live_model(settings_window):
    assert settings_window.cooldown_spin.isEnabled() is False
    assert "no observer" in settings_window.observer_mode_label.text()

    settings_window.model_picker.choose("gemini/gemini-2.5-flash")

    assert settings_window.cooldown_spin.isEnabled() is True
    assert settings_window.gated_radio.isEnabled() is True


def test_the_observer_model_only_matters_in_split_mode(settings_window):
    settings_window.model_picker.choose("gemini/gemini-2.5-flash")
    assert settings_window.observer_model_picker.isEnabled() is False

    settings_window.split_radio.setChecked(True)
    assert settings_window.observer_model_picker.isEnabled() is True
    assert settings_window.collect().agent_mode == "split"


def test_observer_settings_round_trip(settings_window):
    settings = Settings(selected_model="gemini/gemini-2.5-flash")
    settings.observer.trigger_strategy = "spike_plain_heartbeat"
    settings.observer.cooldown_seconds = 45.0
    settings.observer.spike_sensitivity = 4.5
    settings.observer.max_frames_per_call = 5

    settings_window.load(settings)
    collected = settings_window.collect().observer

    assert settings_window.plain_radio.isChecked()
    assert collected.trigger_strategy == "spike_plain_heartbeat"
    assert collected.cooldown_seconds == 45.0
    assert collected.spike_sensitivity == pytest.approx(4.5)
    assert collected.max_frames_per_call == 5


def test_the_burn_estimate_changes_meaning_with_the_mode(settings_window):
    live = settings_window.burn_label.text()
    assert "evicted" in live

    settings_window.model_picker.choose("gemini/gemini-2.5-flash")
    non_live = settings_window.burn_label.text()

    assert "buffered, not streamed" in non_live
    assert non_live != live


def test_the_openrouter_key_round_trips(settings_window):
    settings_window.openrouter_key_edit.setText("sk-or-v1-abc")
    assert settings_window.collect().openrouter_api_key == "sk-or-v1-abc"


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
    assert settings_window.nav.count() == settings_window.pages.count() == 7
    for row in range(settings_window.nav.count()):
        settings_window.nav.setCurrentRow(row)
        assert settings_window.pages.currentIndex() == row
