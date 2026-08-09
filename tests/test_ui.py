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


def test_retained_observer_actions_are_visible_only_while_needed(overlay):
    assert overlay.observer_retry_button.isHidden() is True
    assert overlay.observer_discard_button.isHidden() is True
    overlay.set_observer_recovery(12)
    assert overlay.observer_retry_button.isHidden() is False
    assert overlay.observer_discard_button.isHidden() is False
    assert "12" in overlay.observer_discard_button.toolTip()
    overlay.set_observer_recovery(0)
    assert overlay.observer_retry_button.isHidden() is True


def test_streaming_builds_one_message(overlay):
    overlay.append_user("what killed me?")
    overlay.start_response()
    overlay.append_delta("A skeleton ")
    overlay.append_delta("on the bridge.")
    overlay.end_response("A skeleton on the bridge.")

    text = overlay.transcript.toPlainText()
    assert "what killed me?" in text
    assert text.count("A skeleton on the bridge.") == 1


def test_journal_entries_stay_out_of_the_conversation(overlay):
    """They live in the drawer now — see tests/test_journal_drawer.py."""
    overlay.append_journal(
        JournalEntry(
            timestamp=1_700_000_000.0, note="Lit the bonfire.", category="progress"
        )
    )
    assert "Lit the bonfire." not in overlay.transcript.toPlainText()


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
    settings_window.interval_spin.setValue(8.0)
    settings_window.opacity_slider.setValue(60)

    collected = settings_window.collect()
    assert collected.game_name == "Elden Ring"
    assert collected.capture.interval_seconds == 8.0
    assert collected.overlay.opacity == pytest.approx(0.6)


def test_saving_emits_the_new_settings(settings_window):
    received = []
    settings_window.settingsSaved.connect(received.append)
    settings_window.game_name_edit.setText("Hades")
    settings_window.api_key_edit.setText("google-key")
    settings_window._save()
    assert received[0].game_name == "Hades"


def test_save_blocks_only_keys_required_by_selected_agents(settings_window):
    received = []
    settings_window.settingsSaved.connect(received.append)
    settings_window._save()
    assert received == []
    assert "missing API key for Observer and Responder" in (
        settings_window.restart_badge.text()
    )

    settings_window.openrouter_key_edit.setText("router-key")
    settings_window.observer_model_picker.choose("openrouter/google/gemini-2.5-flash")
    settings_window.responder_model_picker.choose("openrouter/google/gemini-2.5-flash")
    settings_window._save()
    assert received[-1].api_key == ""
    assert received[-1].observer_model.startswith("openrouter/")


def test_process_cadence_is_only_enabled_for_batched_observer(settings_window):
    assert settings_window.process_interval_spin.isEnabled() is True
    live = Settings(observer_model="live/gemini-3.1-flash-live-preview")
    settings_window.load(live)
    assert settings_window.process_interval_spin.isEnabled() is False


def test_a_live_responder_id_is_rejected_on_save(settings_window):
    received = []
    settings_window.settingsSaved.connect(received.append)
    settings_window.responder_model_picker.choose("live/gemini-3.1-flash-live-preview")
    settings_window._save()
    assert received == []
    assert "Cannot save" in settings_window.restart_badge.text()


def test_load_repopulates_the_form(settings_window):
    settings = Settings()
    settings.capture.frame_width = 1024
    settings.capture.question_frame_policy = "immediate"
    settings.hotkeys.toggle_overlay = "ctrl+shift+g"

    settings_window.load(settings)

    assert settings_window.frame_width_spin.value() == 1024
    assert settings_window.question_policy_combo.currentData() == "immediate"
    assert settings_window.toggle_hotkey_edit.text() == "ctrl+shift+g"


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


def test_restart_badge_names_only_the_affected_agent(settings_window):
    settings_window.opacity_slider.setValue(45)
    assert settings_window.restart_badge.text() == ""

    settings_window.observer_model_picker.choose("live/another-live-model")
    assert "reconnect Observer" in settings_window.restart_badge.text()
    assert "Responder" not in settings_window.restart_badge.text()

    settings_window.load(Settings())
    settings_window.responder_mode_combo.setCurrentIndex(1)
    assert "rebuild Responder" in settings_window.restart_badge.text()
    assert "Observer" not in settings_window.restart_badge.text()


# ------------------------------------------------------------ model picking


def test_the_picker_returns_an_id_for_a_listed_model(settings_window):
    picker = settings_window.responder_model_picker
    picker.choose("gemini/gemini-2.5-flash")

    assert picker.selection() == "gemini/gemini-2.5-flash"
    assert settings_window.collect().responder_model == "gemini/gemini-2.5-flash"
    assert "AI STUDIO" in picker.provider_label.text(), "the card says where it runs"
    assert picker.id_label.text() == "gemini/gemini-2.5-flash"


def test_the_picker_accepts_a_model_it_has_never_heard_of(settings_window):
    """A catalogue fetched over the network cannot be the only way to name a model."""
    settings_window.responder_model_picker.choose("openrouter/some/model-from-today")

    assert settings_window.collect().responder_model == (
        "openrouter/some/model-from-today"
    )
    assert "not in a catalogue" in settings_window.responder_note_label.text()


def test_refreshing_the_catalogue_keeps_the_current_selection(settings_window):
    from chiron.models.catalogue import STATIC_MODELS

    settings_window.responder_model_picker.set_selection("openrouter/typed/by-hand")
    settings_window.set_models(list(STATIC_MODELS))

    assert settings_window.collect().responder_model == "openrouter/typed/by-hand"


def test_agent_pickers_reflect_each_agents_transport_contract(settings_window):
    observer_ids = settings_window.observer_model_picker.model_ids()
    responder_ids = settings_window.responder_model_picker.model_ids()
    assert observer_ids and responder_ids
    assert any(model_id.startswith("live/") for model_id in observer_ids)
    assert any(model_id.startswith("gemini/") for model_id in observer_ids)
    assert all(not model_id.startswith("live/") for model_id in responder_ids)


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
    assert format_prices(priced) == "$1.50 / $7.50"
    assert format_prices(next(m for m in STATIC_MODELS if not m.pricing_known)) == (
        "no price"
    )
    assert format_context(1_000_000) == "1.0M ctx"
    assert format_context(128_000) == "128K ctx"
    assert format_context(None) == ""


# --------------------------------------------------------- dual-agent effects


def test_frame_width_and_live_media_resolution_are_both_visible(settings_window):
    assert settings_window.frame_width_spin.isHidden() is False
    assert settings_window.media_res_combo.isHidden() is False


def test_journal_memory_is_automatic_not_user_counted(settings_window):
    assert not hasattr(settings_window, "fold_limit_spin")
    assert not hasattr(settings_window, "max_entries_spin")
    assert "ctx" in settings_window.journal_observer_context_label.text()
    assert "ctx" in settings_window.journal_responder_context_label.text()


def test_dual_agent_fields_round_trip(settings_window):
    settings = Settings(
        observer_model="live/custom-observer",
        responder_model="openrouter/openai/gpt-4o",
        responder_mode="react",
        responder_reasoning_effort="low",
        observer_system_prompt="record objectives",
        responder_system_prompt="avoid spoilers",
    )
    settings.capture.question_frame_policy = "immediate"
    settings.capture.media_resolution = "high"
    settings_window.load(settings)

    collected = settings_window.collect()
    assert collected.observer_model == "live/custom-observer"
    assert collected.responder_model == "openrouter/openai/gpt-4o"
    assert collected.responder_mode == "react"
    assert collected.responder_reasoning_effort == "low"
    assert collected.observer_system_prompt == "record objectives"
    assert collected.responder_system_prompt == "avoid spoilers"
    assert collected.capture.question_frame_policy == "immediate"
    assert collected.capture.media_resolution == "high"


def test_react_filters_known_models_without_tools(settings_window):
    from chiron.models.catalogue import ModelInfo

    no_tools = ModelInfo(
        id="gemini/vision-no-tools",
        provider="google",
        provider_model_id="vision-no-tools",
        name="No tools",
        vendor="vendor",
        context_length=100_000,
        prompt_price=0.0,
        completion_price=0.0,
        pricing_known=True,
        supports_tools=False,
        supports_vision=True,
    )
    settings_window.set_models([*settings_window._models, no_tools])
    assert no_tools.id in settings_window.responder_model_picker.model_ids()

    settings_window.responder_mode_combo.setCurrentIndex(1)
    assert no_tools.id not in settings_window.responder_model_picker.model_ids()


def test_the_openrouter_key_round_trips(settings_window):
    settings_window.openrouter_key_edit.setText("sk-or-v1-abc")
    assert settings_window.collect().openrouter_api_key == "sk-or-v1-abc"


def test_token_estimate_reacts_to_the_shutter(settings_window):
    settings_window.interval_spin.setValue(4.0)
    fast = settings_window.burn_label.text()
    settings_window.interval_spin.setValue(20.0)
    slow = settings_window.burn_label.text()
    assert fast != slow
    assert "visual tokens/minute" in slow


def test_privacy_page_warns_about_two_provider_screenshot_delivery(settings_window):
    settings_window.nav.setCurrentRow(5)
    text = " ".join(
        label.text()
        for label in settings_window.pages.currentWidget().findChildren(
            type(settings_window.key_source_label)
        )
    )
    assert "Google Live and OpenRouter" in text


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
