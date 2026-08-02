"""The player's window is known before watching starts, and named to the model.

The X11 resolver itself needs a display, so it is not driven here; everything
around it — Steam manifest parsing, /proc reading, candidate ordering, the
tracker's change detection, and the app wiring that turns a detected window into
instruction text and journal entries — is exercised with fakes and tmp paths.
"""

from __future__ import annotations

from pathlib import Path

from chiron.capture.active_window import (
    ActiveWindowTracker,
    WindowInfo,
    ordered_candidates,
    parse_appmanifest_name,
    process_name,
    steam_app_id_from_environ,
    steam_game_name,
    steam_library_steamapps,
)
from chiron.config.settings import Settings
from chiron.live.prompts import build_system_instruction

ACF = """\
"AppState"
{
\t"appid"\t\t"1363080"
\t"Universe"\t\t"1"
\t"name"\t\t"Manor Lords"
\t"StateFlags"\t\t"4"
}
"""


# ------------------------------------------------------------------ manifests


def test_appmanifest_name_is_extracted():
    assert parse_appmanifest_name(ACF) == "Manor Lords"


def test_appmanifest_without_name_yields_empty():
    assert parse_appmanifest_name('"AppState"\n{\n\t"appid"\t"1"\n}\n') == ""


def test_appmanifest_name_unescapes_quotes():
    text = '"AppState"\n{\n\t"name"\t\t"The \\"Best\\" Game"\n}\n'
    assert parse_appmanifest_name(text) == 'The "Best" Game'


def _steam_install(root: Path, app_id: int, name: str) -> Path:
    steamapps = root / "steamapps"
    steamapps.mkdir(parents=True)
    (steamapps / f"appmanifest_{app_id}.acf").write_text(
        ACF.replace("Manor Lords", name).replace("1363080", str(app_id))
    )
    return steamapps


def test_steam_game_name_reads_the_manifest(tmp_path):
    steamapps = _steam_install(tmp_path / "Steam", 620, "Portal 2")
    assert steam_game_name(620, [steamapps]) == "Portal 2"
    assert steam_game_name(999, [steamapps]) == "", "not installed here"


def test_library_folders_are_discovered_and_deduplicated(tmp_path):
    main = tmp_path / "Steam"
    extra = tmp_path / "mnt" / "games" / "SteamLibrary"
    steamapps = _steam_install(main, 620, "Portal 2")
    _steam_install(extra, 990080, "Hogwarts Legacy")
    (steamapps / "libraryfolders.vdf").write_text(
        '"libraryfolders"\n{\n\t"0"\n\t{\n\t\t"path"\t\t"%s"\n\t}\n'
        '\t"1"\n\t{\n\t\t"path"\t\t"%s"\n\t}\n}\n' % (main, extra)
    )

    found = steam_library_steamapps([main, main])
    assert found == [main / "steamapps", extra / "steamapps"]
    assert steam_game_name(990080, found) == "Hogwarts Legacy"


# ---------------------------------------------------------------------- /proc


def _proc(tmp_path: Path, pid: int, environ: bytes, comm: str = "game") -> Path:
    proc = tmp_path / "proc" / str(pid)
    proc.mkdir(parents=True)
    (proc / "environ").write_bytes(environ)
    (proc / "comm").write_text(comm + "\n")
    return tmp_path / "proc"


def test_steam_app_id_is_read_from_environ(tmp_path):
    proc = _proc(tmp_path, 42, b"HOME=/home/p\0SteamAppId=1363080\0LANG=C\0")
    assert steam_app_id_from_environ(42, proc) == 1363080


def test_compat_app_id_is_the_fallback(tmp_path):
    proc = _proc(tmp_path, 42, b"STEAM_COMPAT_APP_ID=620\0")
    assert steam_app_id_from_environ(42, proc) == 620


def test_zero_or_absent_app_id_is_none(tmp_path):
    proc = _proc(tmp_path, 42, b"SteamAppId=0\0PATH=/bin\0")
    assert steam_app_id_from_environ(42, proc) is None
    assert steam_app_id_from_environ(7777, proc) is None, "no such process"


def test_process_name_comes_from_comm(tmp_path):
    proc = _proc(tmp_path, 42, b"", comm="eldenring.exe")
    assert process_name(42, proc) == "eldenring.exe"
    assert process_name(7777, proc) == ""


# ----------------------------------------------------------------- WindowInfo


def test_label_prefers_the_steam_name():
    info = WindowInfo(
        window_id=1,
        title="ManorLords ",
        wm_class="steam_app_1363080",
        steam_app_id=1363080,
        steam_name="Manor Lords",
    )
    assert info.label == "Manor Lords"
    assert info.describe() == "Manor Lords (Steam)"


def test_label_falls_back_through_title_class_and_process():
    assert (
        WindowInfo(window_id=1, title="Celeste", wm_class="celeste").label == "Celeste"
    )
    assert WindowInfo(window_id=1, wm_class="celeste").label == "celeste"
    assert WindowInfo(window_id=1, process="celeste.bin").label == "celeste.bin"
    assert WindowInfo(window_id=0xAB).label == "window 0xab"


def test_describe_pairs_title_with_application():
    info = WindowInfo(window_id=1, title="Dwarf Fortress", wm_class="dwarffortress")
    assert info.describe() == "'Dwarf Fortress' (dwarffortress)"


def test_identity_ignores_title_churn():
    a = WindowInfo(window_id=1, title="Chapter 1", wm_class="Celeste")
    b = WindowInfo(window_id=1, title="Chapter 2", wm_class="celeste")
    assert a.identity == b.identity

    steam_a = WindowInfo(window_id=1, wm_class="x", steam_app_id=620)
    steam_b = WindowInfo(window_id=9, wm_class="y", steam_app_id=620)
    assert steam_a.identity == steam_b.identity, "same game, new window"


# ----------------------------------------------------------------- candidates


def test_active_window_is_considered_first():
    assert ordered_candidates(30, [10, 20, 30]) == [30, 20, 10]


def test_without_an_active_window_stacking_wins_top_down():
    assert ordered_candidates(None, [10, 20, 30]) == [30, 20, 10]
    assert ordered_candidates(0, [10, 20]) == [20, 10], "0 means none"


# -------------------------------------------------------------------- tracker


class FakeResolver:
    """Serves scripted WindowInfo answers and records the exclusions given."""

    def __init__(self, *answers: WindowInfo | None) -> None:
        self.answers = list(answers)
        self.excluded_seen: list[set[int]] = []

    def resolve(self, excluded: set[int]) -> WindowInfo | None:
        self.excluded_seen.append(excluded)
        return self.answers.pop(0) if self.answers else None

    def close(self) -> None: ...


def _tracker(qapp, *answers, exclude=frozenset()):
    return ActiveWindowTracker(lambda: set(exclude), resolver=FakeResolver(*answers))


def test_tracker_emits_on_application_change(qapp):
    game = WindowInfo(window_id=1, wm_class="Celeste")
    tracker = _tracker(qapp, game)
    seen: list[WindowInfo] = []
    tracker.windowChanged.connect(seen.append)

    tracker.poll()
    assert tracker.current == game
    assert seen == [game]


def test_tracker_stays_quiet_across_title_churn(qapp):
    tracker = _tracker(
        qapp,
        WindowInfo(window_id=1, title="Chapter 1", wm_class="celeste"),
        WindowInfo(window_id=1, title="Chapter 2", wm_class="celeste"),
        WindowInfo(window_id=2, wm_class="firefox"),
    )
    seen: list[WindowInfo] = []
    tracker.windowChanged.connect(seen.append)

    tracker.poll()
    tracker.poll()
    assert len(seen) == 1, "same application; only the title moved"
    assert tracker.current.title == "Chapter 2", "still refreshed"

    tracker.poll()
    assert len(seen) == 2, "a different application is a change"


def test_tracker_keeps_the_last_window_when_nothing_qualifies(qapp):
    game = WindowInfo(window_id=1, wm_class="celeste")
    tracker = _tracker(qapp, game, None, None)
    tracker.poll()
    tracker.poll()
    assert tracker.current == game, "a poll with no candidate erases nothing"


def test_tracker_passes_the_exclusions(qapp):
    tracker = _tracker(qapp, None, exclude={7, 8})
    tracker.poll()
    assert tracker._resolver.excluded_seen == [{7, 8}]


def test_tracker_without_a_resolver_is_inert(qapp):
    tracker = ActiveWindowTracker(lambda: set())
    tracker.poll()
    assert tracker.current is None


# ---------------------------------------------------------------- instruction


def test_detected_game_reaches_the_instruction():
    text = build_system_instruction(Settings(), detected_game="Manor Lords (Steam)")
    assert "appears to be playing: Manor Lords (Steam)" in text
    assert "Trust the frames" in text, "a focused window is evidence, not certainty"


def test_the_players_own_game_name_wins():
    settings = Settings(game_name="Elden Ring")
    text = build_system_instruction(settings, detected_game="Manor Lords (Steam)")
    assert "The player is playing: Elden Ring." in text
    assert "Manor Lords" not in text


def test_no_game_and_no_detection_says_nothing():
    text = build_system_instruction(Settings())
    assert "playing" not in text.split("How you see the world")[0]
    assert "appears to be playing" not in text


# ------------------------------------------------------------------------ app


def _detected(app, **kwargs) -> WindowInfo:
    info = WindowInfo(
        **{
            "window_id": 5,
            "wm_class": "steam_app_1363080",
            "steam_app_id": 1363080,
            "steam_name": "Manor Lords",
            **kwargs,
        }
    )
    app.window_tracker.current = info
    return info


def test_watching_registers_the_detected_game(app):
    _detected(app)
    app.set_watching(True)

    assert app.session.detected_game == "Manor Lords (Steam)"
    assert "Manor Lords" in app.journal.render()
    assert "looks like Manor Lords" in app.overlay.transcript.toPlainText()


def test_a_manually_named_game_silences_detection(app):
    _detected(app)
    app.settings.game_name = "Elden Ring"
    app.set_watching(True)

    assert getattr(app.session, "detected_game", "") == ""
    assert "Manor Lords" not in app.journal.render()
    assert "Watching your screen." in app.overlay.transcript.toPlainText()


def test_switching_games_mid_watch_is_journaled(app):
    app.set_watching(True)
    app._on_active_window(WindowInfo(window_id=9, title="Hades", wm_class="supergiant"))

    assert app.session.detected_game == "'Hades' (supergiant)"
    assert "switched to 'Hades' (supergiant)" in app.journal.render()


def test_switching_windows_while_not_watching_is_not_journaled(app):
    app._on_active_window(WindowInfo(window_id=9, wm_class="firefox"))
    assert len(app.journal) == 0, "browsing before watching is nobody's business"
    assert app.session.detected_game == "firefox", "still remembered for later"


def test_settings_window_shows_the_detection_as_a_placeholder(app):
    _detected(app)
    app.show_settings()
    assert app.settings_window.game_name_edit.placeholderText() == (
        "auto-detected: Manor Lords"
    )
    assert app.settings_window.game_name_edit.text() == "", "a guess is never saved"
