"""Shared look for the overlay and the settings window.

The overlay sits on top of a game, which constrains the palette more than taste
does: it has to stay readable over arbitrary bright art without stealing
attention from the thing the player is actually looking at. Hence a near-black
translucent panel, one warm accent used sparingly, and text at slightly reduced
contrast so the panel reads as an instrument rather than a spotlight.

The settings window is an ordinary window and can afford to be plainer and more
opaque; it shares the palette so the two do not look like different programs.
"""

from __future__ import annotations

#: Core palette. Kept as a dict so both stylesheets and status colours read from
#: one source.
PALETTE = {
    "bg": "#12151c",
    "bg_raised": "#1a1f29",
    "bg_input": "#0d1015",
    "border": "#2a3240",
    "text": "#dfe4ee",
    "text_dim": "#8c96a8",
    "text_faint": "#5f6878",
    "accent": "#d9a441",
    "accent_dim": "#8a6a2c",
    "user": "#7fb3ff",
    "live": "#5bd68a",
    "warn": "#e0b34a",
    "error": "#e5736a",
}

#: Journal category -> colour, for the drawer. Categories are a hint rather than
#: a schema, so anything absent here falls back to dim text rather than being
#: coloured at random — see :func:`chiron.ui.journal_drawer.category_colour`.
JOURNAL_CATEGORY_COLOURS = {
    "location": PALETTE["user"],
    "objective": PALETTE["accent"],
    "combat": PALETTE["warn"],
    "death": PALETTE["error"],
    "item": PALETTE["live"],
    "npc": PALETTE["user"],
    "progress": PALETTE["live"],
    "note": PALETTE["text_dim"],
}

#: Status name -> dot colour, for the overlay header.
STATUS_COLOURS = {
    "idle": PALETTE["text_faint"],
    "connecting": PALETTE["warn"],
    "live": PALETTE["live"],
    "watching": PALETTE["live"],
    "reviewing": PALETTE["accent"],
    "draining": PALETTE["warn"],
    "retrying": PALETTE["warn"],
    "backpressure": PALETTE["error"],
    "incompatible": PALETTE["error"],
    "preparing": PALETTE["warn"],
    "reconnecting": PALETTE["warn"],
    "error": PALETTE["error"],
    "stopped": PALETTE["text_faint"],
}


def overlay_stylesheet(font_size: int = 11) -> str:
    """Qt stylesheet for the overlay panel.

    Args:
        font_size (int): Base point size for the transcript and input.

    Returns:
        str: The stylesheet.
    """
    p = PALETTE
    return f"""
    QFrame#panel {{
        background-color: {p["bg"]};
        border: 1px solid {p["border"]};
        border-radius: 10px;
    }}
    QWidget {{
        color: {p["text"]};
        font-size: {font_size}pt;
        font-family: "Inter", "Cantarell", "DejaVu Sans", sans-serif;
    }}
    QLabel#brand {{
        color: {p["accent"]};
        font-size: {font_size}pt;
        font-weight: 700;
        letter-spacing: 2px;
    }}
    QLabel#statusText {{
        color: {p["text_dim"]};
        font-size: {max(7, font_size - 2)}pt;
    }}
    QLabel#footer {{
        color: {p["text_faint"]};
        font-size: {max(7, font_size - 3)}pt;
        padding: 0px 4px;
    }}
    QTextBrowser#transcript {{
        background-color: transparent;
        border: none;
        selection-background-color: {p["accent_dim"]};
    }}
    /* The drawer is a surface, not a hairline. A bare `border-left` left the
       column reading as leftover space with text floating in it — a faintly
       raised, rounded panel says "this is a different instrument" with no rule
       needed, and gives the entries something to sit on. */
    QWidget#journalDrawer {{
        background-color: {p["bg_raised"]};
        border: 1px solid {p["border"]};
        border-radius: 8px;
    }}
    QLabel#drawerTitle {{
        color: {p["accent"]};
        font-size: {max(7, font_size - 3)}pt;
        font-weight: 700;
        letter-spacing: 1.5px;
    }}
    /* The count is a fact about the column, not a notification: a quiet chip
       rather than a badge. */
    QLabel#drawerCount {{
        color: {p["text_dim"]};
        background-color: {p["bg_input"]};
        border-radius: 7px;
        font-size: {max(7, font_size - 3)}pt;
        padding: 1px 6px;
    }}
    QFrame#drawerRule {{
        background-color: {p["border"]};
        border: none;
    }}
    QToolButton#drawerClose {{
        color: {p["text_faint"]};
        font-size: {font_size + 2}pt;
        padding: 0px 5px;
    }}
    QTextBrowser#journalBody {{
        background-color: transparent;
        border: none;
        font-size: {max(7, font_size - 2)}pt;
        selection-background-color: {p["accent_dim"]};
    }}
    /* An unread count is information, not an alert: it takes the accent so it
       is findable, and nothing that moves. */
    QToolButton#journalToggle[unread="true"] {{
        color: {p["accent"]};
    }}
    /* The session title is a button that must not read as one: it is a label
       you can click, and a button-shaped control here would compete with the
       three in the header that actually change what Chiron is doing. */
    QToolButton#sessionTitle {{
        color: {p["text_dim"]};
        font-size: {max(7, font_size - 2)}pt;
        text-align: left;
        padding: 1px 4px;
    }}
    QToolButton#sessionTitle:disabled {{
        color: {p["text_faint"]};
    }}
    QListWidget#sessions {{
        background-color: transparent;
        border: none;
        outline: none;
    }}
    QListWidget#sessions::item {{
        padding: 7px 6px;
        border-bottom: 1px solid {p["border"]};
        color: {p["text"]};
    }}
    QListWidget#sessions::item:selected {{
        background-color: {p["bg_raised"]};
        color: {p["accent"]};
    }}
    QListWidget#sessions::item:hover:!selected {{
        background-color: {p["bg_raised"]};
    }}
    QLineEdit#input {{
        background-color: {p["bg_input"]};
        border: 1px solid {p["border"]};
        border-radius: 6px;
        padding: 6px 8px;
        selection-background-color: {p["accent_dim"]};
    }}
    QLineEdit#input:focus {{
        border: 1px solid {p["accent_dim"]};
    }}
    QToolButton {{
        background: transparent;
        border: none;
        color: {p["text_dim"]};
        padding: 2px 6px;
        border-radius: 4px;
    }}
    QToolButton:hover {{
        background-color: {p["bg_raised"]};
        color: {p["text"]};
    }}
    /* The one button that ends the session should not look like the one that
       merely tidies it away. */
    QToolButton#close:hover {{
        background-color: {p["error"]};
        color: {p["bg"]};
    }}
    QPushButton#send {{
        background-color: {p["bg_raised"]};
        border: 1px solid {p["border"]};
        border-radius: 6px;
        padding: 6px 12px;
        color: {p["text"]};
    }}
    QPushButton#send:hover {{
        border-color: {p["accent_dim"]};
        color: {p["accent"]};
    }}
    QScrollBar:vertical {{
        background: transparent;
        width: 8px;
        margin: 2px;
    }}
    QScrollBar::handle:vertical {{
        background: {p["border"]};
        border-radius: 4px;
        min-height: 24px;
    }}
    QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
        height: 0px;
    }}
    """


def settings_stylesheet() -> str:
    """Qt stylesheet for the settings window."""
    p = PALETTE
    return f"""
    QWidget {{
        background-color: {p["bg"]};
        color: {p["text"]};
        font-family: "Inter", "Cantarell", "DejaVu Sans", sans-serif;
        font-size: 10pt;
    }}
    QListWidget#nav {{
        background-color: {p["bg_input"]};
        border: none;
        border-right: 1px solid {p["border"]};
        outline: none;
        padding: 8px 0px;
    }}
    QListWidget#nav::item {{
        padding: 9px 18px;
        color: {p["text_dim"]};
        border-left: 2px solid transparent;
    }}
    QListWidget#nav::item:selected {{
        background-color: {p["bg_raised"]};
        color: {p["text"]};
        border-left: 2px solid {p["accent"]};
    }}
    QListWidget#nav::item:hover:!selected {{
        color: {p["text"]};
    }}
    QLabel#pageTitle {{
        font-size: 15pt;
        font-weight: 600;
        color: {p["text"]};
    }}
    QLabel#pageBlurb, QLabel#hint {{
        color: {p["text_dim"]};
        font-size: 9pt;
    }}
    QLabel#sectionTitle {{
        color: {p["accent"]};
        font-size: 9pt;
        font-weight: 700;
        letter-spacing: 1px;
        margin-top: 6px;
    }}
    QLabel#restartBadge {{
        color: {p["warn"]};
        font-size: 8pt;
    }}
    QLineEdit, QPlainTextEdit, QSpinBox, QDoubleSpinBox, QComboBox {{
        background-color: {p["bg_input"]};
        border: 1px solid {p["border"]};
        border-radius: 5px;
        padding: 5px 7px;
        selection-background-color: {p["accent_dim"]};
    }}
    QLineEdit:focus, QPlainTextEdit:focus, QSpinBox:focus,
    QDoubleSpinBox:focus, QComboBox:focus {{
        border-color: {p["accent_dim"]};
    }}
    QComboBox QAbstractItemView {{
        background-color: {p["bg_raised"]};
        selection-background-color: {p["accent_dim"]};
        border: 1px solid {p["border"]};
    }}
    QCheckBox, QRadioButton {{
        spacing: 8px;
    }}
    QRadioButton#strategy {{
        font-weight: 600;
    }}
    QPushButton {{
        background-color: {p["bg_raised"]};
        border: 1px solid {p["border"]};
        border-radius: 5px;
        padding: 6px 16px;
    }}
    QPushButton:hover {{
        border-color: {p["accent_dim"]};
    }}
    QPushButton#primary {{
        background-color: {p["accent"]};
        color: #17130a;
        border: none;
        font-weight: 600;
    }}
    QPushButton#primary:hover {{
        background-color: #e8b45a;
    }}
    QPushButton:disabled {{
        color: {p["text_faint"]};
        border-color: {p["border"]};
        background-color: {p["bg"]};
    }}
    QFrame#separator {{
        background-color: {p["border"]};
        max-height: 1px;
        border: none;
    }}
    QScrollArea {{
        border: none;
    }}
    QSlider::groove:horizontal {{
        height: 4px;
        background: {p["border"]};
        border-radius: 2px;
    }}
    QSlider::handle:horizontal {{
        background: {p["accent"]};
        width: 14px;
        margin: -6px 0;
        border-radius: 7px;
    }}
    """


__all__ = [
    "JOURNAL_CATEGORY_COLOURS",
    "PALETTE",
    "STATUS_COLOURS",
    "overlay_stylesheet",
    "settings_stylesheet",
]
