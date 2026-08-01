"""Application-level wiring: leaving, and the difference between hiding and quitting."""

from __future__ import annotations


def test_quit_is_not_requested_by_default(app):
    assert app.quit_requested.is_set() is False


def test_the_cross_button_quits(app):
    """The complaint this exists to prevent: a ✕ that only hides the window."""
    app.overlay.close_button.click()
    assert app.quit_requested.is_set() is True


def test_closing_the_window_quits(app):
    """Alt+F4 and anything else the desktop counts as closing."""
    app.overlay.close()
    assert app.quit_requested.is_set() is True


def test_the_hide_button_hides_without_quitting(app):
    app.overlay.show()
    app.overlay.hide_button.click()

    assert app.overlay.isVisible() is False
    assert app.quit_requested.is_set() is False, "hiding is not leaving"


def test_escape_hides_without_quitting(app):
    from PySide6.QtCore import Qt
    from PySide6.QtGui import QKeyEvent

    app.overlay.show()
    app.overlay.keyPressEvent(
        QKeyEvent(
            QKeyEvent.Type.KeyPress, Qt.Key.Key_Escape, Qt.KeyboardModifier.NoModifier
        )
    )

    assert app.overlay.isVisible() is False
    assert app.quit_requested.is_set() is False


def test_hiding_persists_where_the_panel_was(app):
    app.overlay.show()
    app.overlay.move(321, 123)
    app.overlay.hide_button.click()

    assert app.settings.overlay.position_x == 321
    assert app.settings.overlay.position_y == 123


def test_the_hide_button_says_how_to_get_the_panel_back(app):
    app.overlay.set_hide_hint("ctrl+alt+c")
    assert "ctrl+alt+c" in app.overlay.hide_button.toolTip()
    assert "Esc" in app.overlay.hide_button.toolTip()


def test_the_two_buttons_are_not_the_same_control(app):
    """Distinct objects with distinct jobs, so styling and wiring cannot drift."""
    assert app.overlay.close_button is not app.overlay.hide_button
    assert app.overlay.close_button.text() == "✕"
    assert "Quit" in app.overlay.close_button.toolTip()
