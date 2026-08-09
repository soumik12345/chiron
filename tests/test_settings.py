"""Dual-agent settings, one-way migration, and restart boundaries."""

from __future__ import annotations

import json
import os
import stat

import pytest
from pydantic import ValidationError

from chiron.config.settings import (
    DEFAULT_RESPONDER_MODEL,
    Settings,
    default_settings_path,
    load_settings,
    save_settings,
)


def test_fresh_install_defaults_to_two_explicit_agents():
    settings = Settings()
    assert settings.observer_model == "gemini/gemini-2.5-flash-lite"
    assert settings.responder_model == "gemini/gemini-3.6-flash"
    assert settings.responder_mode == "fixed_horizon"
    assert settings.responder_reasoning_effort is None
    assert settings.capture.interval_seconds == 5.0
    assert settings.capture.process_interval_seconds == 300.0
    assert settings.capture.question_frame_policy == "latest"


def test_round_trip_writes_only_the_v3_shape(tmp_path):
    settings = Settings(game_name="Elden Ring")
    settings.capture.interval_seconds = 6.5
    settings.responder_mode = "react"
    settings.responder_reasoning_effort = "low"
    path = tmp_path / "settings.json"

    save_settings(settings, path)
    raw = json.loads(path.read_text(encoding="utf-8"))
    loaded = load_settings(path)

    assert loaded.game_name == "Elden Ring"
    assert loaded.capture.interval_seconds == 6.5
    assert loaded.responder_mode == "react"
    assert loaded.responder_reasoning_effort == "low"
    assert "selected_model" not in raw
    assert "agent_mode" not in raw
    assert "observer" not in raw
    assert "journal" not in raw
    assert "burst_interval_seconds" not in raw["capture"]
    assert "baseline_interval_seconds" not in raw["capture"]


def test_saved_file_is_owner_only(tmp_path):
    path = tmp_path / "settings.json"
    save_settings(Settings(api_key="AIzaSECRET"), path)
    assert stat.S_IMODE(path.stat().st_mode) == 0o600


def test_missing_file_yields_defaults(tmp_path):
    assert load_settings(tmp_path / "nope.json") == Settings()


def test_corrupt_file_yields_defaults(tmp_path):
    path = tmp_path / "settings.json"
    path.write_text("{not json", encoding="utf-8")
    assert load_settings(path) == Settings()


def test_unknown_keys_are_ignored(tmp_path):
    path = tmp_path / "settings.json"
    payload = Settings().model_dump()
    payload["a_setting_from_the_future"] = True
    path.write_text(json.dumps(payload), encoding="utf-8")
    assert load_settings(path) == Settings()


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
    assert Settings().resolved_api_key() == ""


def test_each_provider_resolves_its_own_key(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    settings = Settings(api_key="AIza-google", openrouter_api_key="sk-or-router")
    assert settings.key_for_model(settings.observer_model) == "AIza-google"
    assert settings.key_for_model(settings.responder_model) == "AIza-google"
    assert settings.key_for_model("openrouter/openai/gpt-4o") == "sk-or-router"


def test_openrouter_only_agents_need_no_google_key(monkeypatch):
    for name in ("GEMINI_API_KEY", "GOOGLE_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    settings = Settings(
        openrouter_api_key="router",
        observer_model="openrouter/google/gemini-2.5-flash",
        responder_model="openrouter/google/gemini-2.5-flash",
    )
    assert settings.resolved_api_key() == ""
    assert settings.key_for_model(settings.observer_model) == "router"
    assert settings.key_for_model(settings.responder_model) == "router"


def test_openrouter_key_falls_back_to_environment(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-env")
    assert Settings().resolved_openrouter_key() == "sk-or-env"


def test_default_path_honours_xdg(monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    assert default_settings_path() == tmp_path / "chiron" / "settings.json"


def test_observer_supports_live_and_batched_but_responder_stays_nonlive():
    assert Settings(observer_model="gemini/gemini-2.5-flash").observer_model.startswith(
        "gemini/"
    )
    assert Settings(
        observer_model="openrouter/google/gemini-2.5-flash"
    ).observer_model.startswith("openrouter/")
    with pytest.raises(ValidationError, match="observer_model"):
        Settings(observer_model="anthropic/claude-sonnet-4")
    with pytest.raises(ValidationError, match="responder_model"):
        Settings(responder_model="live/gemini-3.1-flash-live-preview")
    with pytest.raises(ValidationError, match="gemini/ or openrouter/"):
        Settings(responder_model="anthropic/claude-sonnet-4")


def test_observer_restart_boundary_is_explicit():
    current = Settings(api_key="one")
    cosmetic = current.copy_deep()
    cosmetic.capture.interval_seconds = 9.0
    cosmetic.capture.frame_width = 1024
    cosmetic.overlay.opacity = 0.5
    assert current.requires_observer_reconnect(cosmetic) is False

    for edit in (
        lambda s: setattr(s, "observer_model", "live/another-live-model"),
        lambda s: setattr(s, "api_key", "two"),
        lambda s: setattr(s, "observer_system_prompt", "watch carefully"),
        lambda s: setattr(s.capture, "media_resolution", "high"),
        lambda s: setattr(s.capture, "process_interval_seconds", 600),
    ):
        changed = current.copy_deep()
        edit(changed)
        assert current.requires_observer_reconnect(changed) is True


def test_observer_swap_is_only_needed_when_transport_changes():
    current = Settings()
    same_transport = current.copy_deep()
    same_transport.observer_model = "gemini/gemini-2.5-flash"
    assert current.requires_observer_swap(same_transport) is False

    live = current.copy_deep()
    live.observer_model = "live/gemini-3.1-flash-live-preview"
    assert current.requires_observer_swap(live) is True

    live_cadence = live.copy_deep()
    live_cadence.capture.process_interval_seconds = 600
    assert live.requires_observer_reconnect(live_cadence) is False


def test_responder_rebuild_boundary_preserves_unrelated_agent_changes():
    current = Settings(api_key="one", openrouter_api_key="router-one")
    observer_only = current.copy_deep()
    observer_only.observer_model = "live/another-live-model"
    assert current.requires_responder_rebuild(observer_only) is False

    for edit in (
        lambda s: setattr(s, "responder_model", "gemini/gemini-2.5-flash"),
        lambda s: setattr(s, "responder_mode", "react"),
        lambda s: setattr(s, "responder_reasoning_effort", "low"),
        lambda s: setattr(s, "responder_system_prompt", "be terse"),
    ):
        changed = current.copy_deep()
        edit(changed)
        assert current.requires_responder_rebuild(changed) is True


def test_frame_width_and_media_resolution_no_longer_alias_each_other():
    settings = Settings()
    settings.capture.frame_width = 1024
    settings.capture.media_resolution = "high"
    assert settings.effective_capture().frame_width == 1024
    assert settings.capture.media_resolution == "high"


def test_token_estimate_tracks_fixed_interval():
    settings = Settings()
    settings.capture.interval_seconds = 4.0
    assert settings.capture.tokens_per_minute() == 15 * 260


def test_v0_live_model_migrates_to_observer(tmp_path):
    path = tmp_path / "settings.json"
    path.write_text(
        json.dumps({"live_model": "gemini-2.5-flash-native-audio-latest"}),
        encoding="utf-8",
    )
    loaded = load_settings(path)
    assert loaded.observer_model == "live/gemini-2.5-flash-native-audio-latest"
    assert loaded.responder_model == DEFAULT_RESPONDER_MODEL


def test_old_live_selection_migrates_only_to_observer():
    loaded = Settings.model_validate(
        {"selected_model": "live/gemini-3.1-flash-live-preview"}
    )
    assert loaded.observer_model == "live/gemini-3.1-flash-live-preview"
    assert loaded.responder_model == DEFAULT_RESPONDER_MODEL


def test_old_nonlive_selection_migrates_only_to_responder():
    loaded = Settings.model_validate(
        {
            "selected_model": "openrouter/anthropic/claude-sonnet-4",
            "agent_mode": "split",
            "observer_model": "gemini/gemini-2.5-flash-lite",
        }
    )
    assert loaded.observer_model == "live/gemini-3.1-flash-live-preview"
    assert loaded.responder_model == "openrouter/anthropic/claude-sonnet-4"


def test_old_baseline_and_prompt_migrate_without_obsolete_fields():
    loaded = Settings.model_validate(
        {
            "capture": {"baseline_interval_seconds": 7.5, "burst_duration_seconds": 30},
            "journal": {"fold_entry_limit": 40, "max_entries": 500},
            "extra_system_prompt": "Remember my preferred build.",
            "overlay": {"journal_open": True, "journal_width": 300},
        }
    )
    assert loaded.capture.interval_seconds == 7.5
    assert loaded.observer_system_prompt == "Remember my preferred build."
    assert loaded.responder_system_prompt == "Remember my preferred build."
    assert loaded.overlay.journal_open is True
    assert not hasattr(loaded, "journal")


def test_invalid_old_baseline_uses_the_five_second_default():
    loaded = Settings.model_validate(
        {"capture": {"baseline_interval_seconds": "hand-edited nonsense"}}
    )
    assert loaded.capture.interval_seconds == 5.0


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
