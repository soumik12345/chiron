"""The journal drawer: the column, the count, and the width it costs.

Three things are worth pinning down here, and they are the three that were easy
to get wrong. The drawer must be the *only* place journal entries appear, or
moving them out of the transcript achieved nothing. A closed drawer must still
say that something was written, or the honesty the inline lines provided is
simply gone. And the width it adds must never reach the saved settings, or an
evening with the drawer open reopens permanently wider — the same bug the review
views already had to avoid, arriving down a second path.
"""

from __future__ import annotations

import time

import pytest

from chiron.config.settings import OverlaySettings, Settings
from chiron.journal.log import JournalEntry
from chiron.ui.journal_drawer import (
    GAP_POINTS,
    MAX_DRAWER_ENTRIES,
    category_colour,
    render_entries,
)
from chiron.ui.overlay import PLAY_VIEW, OverlayWindow
from chiron.ui.theme import PALETTE


@pytest.fixture
def overlay(qapp):
    """A real overlay panel on the offscreen backend, drawer closed."""
    return OverlayWindow(OverlaySettings())


def _entry(note="Lit the bonfire.", category="progress", at=1_700_000_000.0):
    return JournalEntry(timestamp=at, note=note, category=category)


# ------------------------------------------------------------- pure rendering


def test_the_newest_entry_needs_no_scrolling():
    """The drawer is a memory being audited, not a chronology being followed."""
    html = render_entries([_entry("first"), _entry("second"), _entry("third")])

    assert html.index("third") < html.index("first")


def test_an_empty_journal_says_so_rather_than_showing_nothing():
    assert "Nothing written down yet" in render_entries([])


def test_a_category_nobody_defined_still_renders():
    """Categories are a hint to the model, not a schema to reject entries with."""
    assert category_colour("boss-fight") == PALETTE["text_dim"]
    assert category_colour("death") == PALETTE["error"]
    assert "BOSS-FIGHT" in render_entries([_entry(category="boss-fight")])


def test_entries_are_told_apart_by_the_space_between_them():
    """The gap is a blank paragraph: Qt drops a margin before a table."""
    html = render_entries([_entry("first"), _entry("second")])

    assert html.count(f"font-size:{GAP_POINTS}pt") == 2


def test_notes_are_escaped_not_interpreted():
    html = render_entries([_entry("picked up <b>Estus</b>")])

    assert "&lt;b&gt;" in html


def test_the_drawer_is_bounded(overlay):
    for index in range(MAX_DRAWER_ENTRIES + 40):
        overlay.append_journal(_entry(f"note {index}"))

    assert len(overlay.journal_drawer.entries) == MAX_DRAWER_ENTRIES
    assert "note 239" in overlay.journal_drawer.body.toPlainText()


# ------------------------------------------------------------------ the column


def test_journal_entries_go_to_the_drawer_and_not_the_transcript(overlay):
    overlay.append_journal(_entry())

    assert "Lit the bonfire." in overlay.journal_drawer.body.toPlainText()
    assert "Lit the bonfire." not in overlay.transcript.toPlainText()


def test_a_new_session_empties_the_drawer(overlay):
    overlay.append_journal(_entry())

    overlay.clear_journal()

    assert overlay.journal_drawer.entries == []


def test_the_drawer_is_hidden_until_it_is_asked_for(overlay):
    overlay.show()

    assert overlay.journal_open is False
    assert overlay.journal_drawer.isVisible() is False


def test_opening_shows_the_column(overlay):
    overlay.show()

    overlay.toggle_journal()

    assert overlay.journal_open is True
    assert overlay.journal_drawer.isVisible() is True


# ------------------------------------------------------------- the unread count


def test_a_closed_drawer_still_says_something_was_written(overlay):
    """Moving the journal out of the transcript removed the only signal there was."""
    overlay.append_journal(_entry())
    overlay.append_journal(_entry("Died to the skeleton.", "death"))

    assert overlay.journal_button.text() == "✎ 2"
    assert "2 new entries" in overlay.journal_button.toolTip()


def test_opening_the_drawer_is_reading_it(overlay):
    overlay.append_journal(_entry())

    overlay.set_journal_open(True)

    assert overlay.journal_button.text() == "✎"


def test_an_open_drawer_never_accrues_a_count(overlay):
    overlay.set_journal_open(True)

    overlay.append_journal(_entry())

    assert overlay.journal_button.text() == "✎"


def test_one_entry_is_singular(overlay):
    overlay.append_journal(_entry())
    assert "1 new entry" in overlay.journal_button.toolTip()


# ----------------------------------------------------------------- the geometry


def test_opening_widens_the_window_rather_than_the_transcript(overlay):
    """A 420px panel split two ways is two columns of nothing."""
    before = overlay.width()

    overlay.set_journal_open(True)

    assert overlay.width() > before
    assert overlay.height() == overlay.settings.height


def test_closing_gives_the_width_back(overlay):
    before = (overlay.width(), overlay.height())

    overlay.set_journal_open(True)
    overlay.set_journal_open(False)

    assert (overlay.width(), overlay.height()) == before


def test_the_drawer_width_never_becomes_the_remembered_width(overlay):
    """Otherwise an evening with the drawer open reopens permanently wider."""
    overlay.set_journal_open(True)

    _, _, width, height = overlay.current_geometry()

    assert (width, height) == (OverlaySettings().width, OverlaySettings().height)


def test_the_drawer_survives_reading_a_session_back(overlay):
    """Two different things: the journal being written now, and last Tuesday."""
    overlay.set_journal_open(True)

    overlay.show_session_viewer()
    assert overlay.journal_open is True

    overlay.show_play()
    assert overlay.journal_open is True


def test_toggling_while_reviewing_still_restores_the_play_size(overlay):
    """The size play is restored to has to move with the drawer, or a gap opens."""
    play = (overlay.width(), overlay.height())

    overlay.show_session_viewer()
    overlay.set_journal_open(True)
    overlay.set_journal_open(False)
    overlay.show_play()

    assert (overlay.width(), overlay.height()) == play


def test_reopening_restores_both_the_size_and_the_drawer(overlay):
    """What `apply_settings` does at launch, with a remembered open drawer."""
    reopened = OverlayWindow(OverlaySettings(journal_open=True, width=500))

    assert reopened.journal_open is True
    assert reopened.width() > 500
    assert reopened.current_geometry()[2] == 500


# -------------------------------------------------------------- the app wiring


def test_a_journal_entry_reaches_the_drawer_and_the_record(app):
    app.set_watching(True)

    app.journal_service.record("Lit the bonfire.", timestamp=time.time())
    app.recorder.flush()

    assert "Lit the bonfire." in app.overlay.journal_drawer.body.toPlainText()
    assert any(
        e.type == "journal_entry"
        for e in app.recorder.read_session(app.recorder.session_id)
    )


def test_a_new_session_forgets_the_drawer_too(app):
    app.set_watching(True)
    app.journal_service.record("Lit the bonfire.", timestamp=time.time())

    app.new_session()

    assert app.overlay.journal_drawer.entries == []
    assert app.overlay.journal_button.text() == "✎"


def test_the_drawer_and_the_footer_never_disagree_on_the_count(app):
    """The window-switch entry goes straight into the log, past `on_entry`."""
    from chiron.capture.active_window import WindowInfo

    app.set_watching(True)
    app._on_active_window(WindowInfo(window_id=1, title="Elden Ring", wm_class="game"))

    assert len(app.overlay.journal_drawer.entries) == len(app.journal)


def test_the_hotkey_toggles_the_drawer(app):
    app.overlay.show()

    app._on_hotkey("toggle_journal")

    assert app.overlay.journal_open is True


def test_the_hotkey_on_a_hidden_panel_shows_the_journal(app):
    """Toggling here would reveal the panel and close the thing that was asked for."""
    app.overlay.hide()
    app.overlay.set_journal_open(True)

    app._on_hotkey("toggle_journal")

    assert app.overlay.isVisible() is True
    assert app.overlay.journal_open is True


async def test_saving_settings_does_not_close_the_drawer(app):
    """The form has no field for it, so a fresh Settings would default it shut.

    Async because a changed instruction schedules a session restart.
    """
    app.overlay.set_journal_open(True)

    app.apply_settings(Settings(game_name="Elden Ring"))

    assert app.overlay.journal_open is True
    assert app.settings.overlay.journal_open is True


def test_the_open_drawer_is_remembered_without_its_width(app):
    app.overlay.set_journal_open(True)

    app._remember_geometry()

    assert app.settings.overlay.journal_open is True
    assert app.settings.overlay.width == OverlaySettings().width


def test_the_picker_is_not_a_view(app):
    """Choosing an evening leaves the panel on the game."""
    app.set_watching(True)

    app.show_history()

    assert app.overlay.current_view == PLAY_VIEW
