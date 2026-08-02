"""Settings model, persistence and credential resolution."""

from __future__ import annotations

import json
import os
import stat

from chiron.config.settings import (
    Settings,
    default_settings_path,
    load_settings,
    save_settings,
)


def test_defaults_are_usable():
    settings = Settings()
    assert settings.live_model.startswith("gemini")
    assert settings.is_live is True, "a fresh install runs the Live API, as in v0"
    assert settings.agent_mode == "unified"
    assert settings.journal.strategy == "tool_call"
    assert settings.capture.burst_interval_seconds >= 1.0
    assert settings.overlay.always_on_top is True


def test_round_trip(tmp_path):
    settings = Settings()
    settings.game_name = "Elden Ring"
    settings.capture.baseline_interval_seconds = 6.5
    settings.journal.strategy = "sidecar"
    path = tmp_path / "settings.json"

    save_settings(settings, path)
    loaded = load_settings(path)

    assert loaded.game_name == "Elden Ring"
    assert loaded.capture.baseline_interval_seconds == 6.5
    assert loaded.journal.strategy == "sidecar"


def test_saved_file_is_owner_only(tmp_path):
    path = tmp_path / "settings.json"
    save_settings(Settings(api_key="AIzaSECRET"), path)
    mode = stat.S_IMODE(path.stat().st_mode)
    assert mode == 0o600


def test_missing_file_yields_defaults(tmp_path):
    assert load_settings(tmp_path / "nope.json").live_model == Settings().live_model


def test_corrupt_file_yields_defaults(tmp_path):
    path = tmp_path / "settings.json"
    path.write_text("{not json", encoding="utf-8")
    assert load_settings(path).game_name == ""


def test_unknown_keys_are_ignored(tmp_path):
    path = tmp_path / "settings.json"
    payload = json.loads(Settings().model_dump_json())
    payload["a_setting_from_the_future"] = True
    path.write_text(json.dumps(payload), encoding="utf-8")
    assert load_settings(path).live_model == Settings().live_model


def test_api_key_prefers_settings_over_environment(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "from-env")
    settings = Settings(api_key="  from-settings  ")
    assert settings.resolved_api_key() == "from-settings"
    assert settings.api_key_source() == "settings"


def test_api_key_falls_back_to_environment(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "from-env")
    settings = Settings()
    assert settings.resolved_api_key() == "from-env"
    assert settings.api_key_source() == "GEMINI_API_KEY"


def test_api_key_absent(monkeypatch):
    for name in ("GEMINI_API_KEY", "GOOGLE_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    settings = Settings()
    assert settings.resolved_api_key() == ""
    assert settings.api_key_source() == "none"


def test_default_path_honours_xdg(monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    assert default_settings_path() == tmp_path / "chiron" / "settings.json"


def test_restart_detection():
    current = Settings()
    same = current.copy_deep()
    assert current.requires_session_restart(same) is False

    same.overlay.opacity = 0.5
    same.capture.baseline_interval_seconds = 9.0
    assert current.requires_session_restart(same) is False, "cosmetic edits are live"

    same.selected_model = "live/gemini-live-2.5-flash-preview"
    assert current.requires_session_restart(same) is True


def test_observer_cadence_applies_without_a_restart():
    """Cost knobs are read per tick; reconnecting to change one would be absurd."""
    current = Settings(selected_model="gemini/gemini-2.5-flash")
    edited = current.copy_deep()
    edited.observer.cooldown_seconds = 60.0
    edited.observer.spike_sensitivity = 5.0
    assert current.requires_session_restart(edited) is False


def test_agent_mode_and_observer_model_need_a_restart():
    current = Settings(selected_model="gemini/gemini-2.5-flash")
    edited = current.copy_deep()
    edited.agent_mode = "split"
    edited.observer_model = "gemini/gemini-2.5-flash-lite"
    assert current.requires_session_restart(edited) is True


# ----------------------------------------------------------- model selection


def test_mode_follows_the_model():
    assert Settings(selected_model="live/gemini-3.1-flash-live-preview").is_live
    assert not Settings(selected_model="gemini/gemini-2.5-flash").is_live
    assert not Settings(selected_model="openrouter/openai/gpt-4o-mini").is_live


def test_crossing_the_mode_boundary_swaps_the_provider():
    live = Settings()
    non_live = live.copy_deep()
    non_live.selected_model = "gemini/gemini-2.5-flash"

    assert live.requires_provider_swap(non_live) is True
    assert live.requires_provider_swap(live.copy_deep()) is False


def test_a_v0_settings_file_keeps_its_model(tmp_path):
    """Upgrading must not silently move someone onto a different model."""
    path = tmp_path / "settings.json"
    path.write_text(
        json.dumps({"live_model": "gemini-2.5-flash-native-audio-latest"}),
        encoding="utf-8",
    )

    loaded = load_settings(path)

    assert loaded.selected_model == "live/gemini-2.5-flash-native-audio-latest"
    assert loaded.live_model == "gemini-2.5-flash-native-audio-latest"
    assert loaded.is_live is True


def test_a_v1_file_is_not_re_migrated(tmp_path):
    path = tmp_path / "settings.json"
    path.write_text(
        json.dumps(
            {"selected_model": "gemini/gemini-2.5-flash", "live_model": "stale"}
        ),
        encoding="utf-8",
    )
    assert load_settings(path).selected_model == "gemini/gemini-2.5-flash"


def test_the_observer_falls_back_to_the_main_model():
    settings = Settings(selected_model="gemini/gemini-2.5-flash", agent_mode="split")
    assert settings.observer_model_id() == "gemini/gemini-2.5-flash"

    settings.observer_model = "gemini/gemini-2.5-flash-lite"
    assert settings.observer_model_id() == "gemini/gemini-2.5-flash-lite"

    settings.agent_mode = "unified"
    assert settings.observer_model_id() == "gemini/gemini-2.5-flash", (
        "unified mode ignores the observer model rather than half-honouring it"
    )


# --------------------------------------------------------------- credentials


def test_each_provider_resolves_its_own_key(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    settings = Settings(api_key="AIza-google", openrouter_api_key="sk-or-router")

    assert settings.key_for_model("gemini/gemini-2.5-flash") == "AIza-google"
    assert settings.key_for_model("live/gemini-3.1-flash-live-preview") == "AIza-google"
    assert settings.key_for_model("openrouter/openai/gpt-4o") == "sk-or-router"


def test_the_openrouter_key_falls_back_to_the_environment(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-env")
    assert Settings().resolved_openrouter_key() == "sk-or-env"


# ------------------------------------------------------------- frame detail


def test_detail_sets_the_capture_width_only_in_non_live_mode():
    live = Settings()
    live.capture.frame_width = 1024
    live.capture.media_resolution = "low"
    assert live.effective_frame_width() == 1024, "live detail is an API-side budget"

    non_live = live.copy_deep()
    non_live.selected_model = "gemini/gemini-2.5-flash"
    assert non_live.effective_frame_width() == 512
    assert non_live.effective_capture().frame_width == 512
    assert non_live.capture.frame_width == 1024, "the stored value is left alone"


def test_detail_tiers_map_to_widths():
    settings = Settings(selected_model="gemini/gemini-2.5-flash")
    widths = []
    for detail in ("low", "medium", "high"):
        settings.capture.media_resolution = detail
        widths.append(settings.effective_frame_width())
    assert widths == sorted(widths) and len(set(widths)) == 3


def test_journal_strategy_change_requires_restart():
    current = Settings()
    edited = current.copy_deep()
    edited.journal.strategy = "sidecar"
    assert current.requires_session_restart(edited) is True


def test_token_estimate_tracks_baseline():
    settings = Settings()
    settings.capture.baseline_interval_seconds = 4.0
    assert settings.capture.tokens_per_minute() == 15 * 260


def test_copy_deep_is_independent():
    original = Settings()
    clone = original.copy_deep()
    clone.capture.frame_width = 1024
    assert original.capture.frame_width != 1024


def test_save_creates_parent_directories(tmp_path):
    path = tmp_path / "deep" / "nested" / "settings.json"
    save_settings(Settings(), path)
    assert path.exists()
    assert os.access(path, os.R_OK)
