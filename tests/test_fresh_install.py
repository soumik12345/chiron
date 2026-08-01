"""`chiron --fresh-install`: what it deletes, what it refuses to delete."""

from __future__ import annotations

import io

from chiron.app import fresh_install, parse_args
from chiron.config.settings import (
    Settings,
    plan_removal,
    remove_configuration,
    save_settings,
)


def _config(tmp_path, **kwargs):
    """Write a settings file into a realistic `chiron/` config directory."""
    path = tmp_path / "chiron" / "settings.json"
    save_settings(Settings(**kwargs), path)
    return path


# -------------------------------------------------------------------- plan


def test_plan_lists_the_file_and_its_directory(tmp_path):
    path = _config(tmp_path)
    plan = plan_removal(path)
    assert plan.settings_file == path
    assert plan.directory == path.parent
    assert plan.is_empty is False


def test_plan_is_empty_when_there_is_no_config(tmp_path):
    plan = plan_removal(tmp_path / "chiron" / "settings.json")
    assert plan.is_empty is True
    assert plan.describe() == []


def test_plan_notices_a_saved_api_key(tmp_path):
    assert plan_removal(_config(tmp_path, api_key="AIzaSECRET")).holds_api_key is True
    assert plan_removal(_config(tmp_path)).holds_api_key is False


def test_plan_keeps_files_chiron_did_not_write(tmp_path):
    path = _config(tmp_path)
    stranger = path.parent / "notes.txt"
    stranger.write_text("mine", encoding="utf-8")

    plan = plan_removal(path)

    assert plan.strangers == [stranger]
    assert plan.directory is None, "a directory with other files is not removed"
    assert any("keep" in line for line in plan.describe())


def test_a_shared_directory_is_never_a_target(tmp_path):
    """--settings can point anywhere; only Chiron's own directory is removable."""
    path = tmp_path / "settings.json"
    save_settings(Settings(), path)
    plan = plan_removal(path)
    assert plan.settings_file == path
    assert plan.directory is None


# ------------------------------------------------------------------ removal


def test_removal_deletes_the_file_and_the_directory(tmp_path):
    path = _config(tmp_path)
    removed = remove_configuration(path)
    assert removed.settings_file == path
    assert not path.exists()
    assert not path.parent.exists()


def test_removal_leaves_no_backup(tmp_path):
    """A saved key is a secret; keeping a copy under another name betrays the ask."""
    path = _config(tmp_path, api_key="AIzaSECRET")
    directory = path.parent
    remove_configuration(path)
    assert not directory.exists()


def test_removal_spares_other_files(tmp_path):
    path = _config(tmp_path)
    stranger = path.parent / "notes.txt"
    stranger.write_text("mine", encoding="utf-8")

    remove_configuration(path)

    assert not path.exists()
    assert stranger.exists(), "someone else's file is not collateral"


def test_removing_nothing_is_not_an_error(tmp_path):
    plan = remove_configuration(tmp_path / "chiron" / "settings.json")
    assert plan.is_empty is True


# ---------------------------------------------------------------------- cli


def test_flag_parses():
    args = parse_args(["--fresh-install"])
    assert args.fresh_install is True
    assert args.yes is False
    assert parse_args(["--fresh-install", "--yes"]).yes is True
    assert parse_args([]).fresh_install is False


def test_confirmed_removal(tmp_path):
    path = _config(tmp_path, api_key="AIzaSECRET")
    out = io.StringIO()

    code = fresh_install(path, stream=out, confirm=lambda _: "y")

    assert code == 0
    assert not path.exists()
    text = out.getvalue()
    assert str(path) in text, "the user is shown exactly what goes"
    assert "API key" in text, "and warned about the key"


def test_declining_removes_nothing(tmp_path):
    path = _config(tmp_path)
    out = io.StringIO()

    code = fresh_install(path, stream=out, confirm=lambda _: "n")

    assert code == 1
    assert path.exists()
    assert "Cancelled" in out.getvalue()


def test_enter_alone_declines(tmp_path):
    path = _config(tmp_path)
    assert fresh_install(path, stream=io.StringIO(), confirm=lambda _: "") == 1
    assert path.exists()


def test_yes_flag_skips_the_prompt(tmp_path):
    path = _config(tmp_path)

    def refuse_to_ask(_):
        raise AssertionError("should not prompt when --yes was passed")

    assert (
        fresh_install(
            path, assume_yes=True, stream=io.StringIO(), confirm=refuse_to_ask
        )
        == 0
    )
    assert not path.exists()


def test_non_interactive_without_yes_removes_nothing(tmp_path, monkeypatch):
    """Silence is not consent: a pipe or a cron job must not wipe a config."""
    path = _config(tmp_path)
    out = io.StringIO()
    # A pipe rather than a terminal, with a "y" sitting in it that must be ignored.
    monkeypatch.setattr("sys.stdin", io.StringIO("y\n"))

    code = fresh_install(path, stream=out)

    assert code == 1
    assert path.exists()
    assert "--yes" in out.getvalue()


def test_nothing_to_remove_is_success(tmp_path):
    out = io.StringIO()
    code = fresh_install(tmp_path / "chiron" / "settings.json", stream=out)
    assert code == 0
    assert "Nothing to remove" in out.getvalue()


def test_next_run_gets_defaults(tmp_path):
    """The point of the command: what comes back is a fresh install."""
    from chiron.config.settings import load_settings

    path = _config(tmp_path, api_key="AIzaSECRET", game_name="Elden Ring")
    fresh_install(path, assume_yes=True, stream=io.StringIO())

    restored = load_settings(path)
    assert restored.api_key == ""
    assert restored.game_name == ""
    assert restored.capture.watch_on_launch is False
