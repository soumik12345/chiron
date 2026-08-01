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

    same.live_model = "gemini-live-2.5-flash-preview"
    assert current.requires_session_restart(same) is True


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
