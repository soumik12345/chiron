"""A searchable model card used by both agent-specific pickers.

The settings page filters the same control into a Live-only Observer list and a
non-live Responder list. Each instance has three jobs that pull against each
other. It has to show enough to choose with (provider, capabilities, and cost), it has
to stay usable at several hundred entries, and it has to accept an id it has
never heard of, because a catalogue fetched over the network cannot be the only
way to name a model released this morning or held in a settings file from last
year.

A combo box can do the third and fails the first two. Packing provider, id, mode
and both prices into one line of item text produces a row too long to read and a
list too long to scroll, and the field then shows that same crowded line back as
the current value. So this is a small purpose-built control instead:

* **The field** is a card, not a line. Two axes rather than one run-on sentence:
  the model's name and id on the left, its provider and prices on the right.
* **The popup** is a search box, a row of provider filters with counts, and a
  list whose rows use the same two-axis layout. Typing filters on name, id and
  vendor at once, so "gemini", "flash" and "google/" all lead somewhere.
* **A typed id that matches nothing** becomes an offered row of its own rather
  than an error. That is the escape hatch, and it stays one keystroke deep.

Rows are painted by a delegate rather than built from widgets: a few hundred
models is a few hundred rows, and instantiating six labels for each of them to
show ten is the kind of thing that makes a settings window feel slow to open.
"""

from __future__ import annotations

from PySide6.QtCore import QEvent, QObject, QPoint, QRect, QSize, Qt, Signal
from PySide6.QtGui import QColor, QFont, QFontMetrics, QKeyEvent, QPainter
from PySide6.QtWidgets import (
    QAbstractItemView,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QStyle,
    QStyledItemDelegate,
    QStyleOptionViewItem,
    QVBoxLayout,
    QWidget,
)

from chiron.models.catalogue import ModelInfo, describe_unknown
from chiron.models.providers import GOOGLE, LIVE, OPENROUTER
from chiron.ui.theme import PALETTE

#: Full provider names for the filter chips, in the order they are offered.
PROVIDER_ORDER = [LIVE, GOOGLE, OPENROUTER]
PROVIDER_NAMES = {
    LIVE: "Live API",
    GOOGLE: "Google AI Studio",
    OPENROUTER: "OpenRouter",
}

#: Height of one list row: two lines of text plus breathing room.
ROW_HEIGHT = 46

#: The popup is at least this wide however narrow the field is. Below it the
#: right-hand price column starts colliding with long model names.
MIN_POPUP_WIDTH = 560
POPUP_HEIGHT = 380

#: Preferred monospace families for model ids, best first.
_MONO_FAMILIES = [
    "JetBrains Mono",
    "DejaVu Sans Mono",
    "Menlo",
    "Consolas",
    "monospace",
]


def format_price(per_token: float) -> str:
    """A per-token rate as dollars per million tokens."""
    return f"${per_token * 1_000_000:.2f}"


def format_prices(model: ModelInfo) -> str:
    """Both rates, or an honest blank when nothing knows them.

    A picker that prints "$0.00" for a model it simply has no rate for is worse
    than one that prints nothing, so an unpriced model says so in words.
    """
    if not model.pricing_known:
        return "no price"
    return (
        f"{format_price(model.prompt_price)} / {format_price(model.completion_price)}"
    )


def format_context(length: int | None) -> str:
    """A context window in the units people quote it in."""
    if not length:
        return ""
    if length >= 1_000_000:
        return f"{length / 1_000_000:.1f}M ctx"
    if length >= 1_000:
        return f"{length // 1_000}K ctx"
    return f"{length} ctx"


def _mono_font(base: QFont, *, size_delta: int = -1) -> QFont:
    """A monospace font matched to `base`, for model ids."""
    font = QFont(base)
    font.setFamilies(_MONO_FAMILIES)
    font.setPointSize(max(7, base.pointSize() + size_delta))
    return font


class _RowDelegate(QStyledItemDelegate):
    """Paints one model as two columns of two lines.

    The left column is what the model *is* (name, then id in monospace); the
    right is what choosing it costs (prices, then provider and context window).
    Reading down either column compares like with like, which a single line of
    dot-separated fields does not let you do.
    """

    def sizeHint(self, option: QStyleOptionViewItem, index) -> QSize:
        """Every row is the same height; the content is fixed in shape."""
        return QSize(option.rect.width(), ROW_HEIGHT)

    def paint(self, painter: QPainter, option: QStyleOptionViewItem, index) -> None:
        """Draw one row, eliding whatever does not fit."""
        model: ModelInfo | None = index.data(Qt.ItemDataRole.UserRole)
        painter.save()
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)

        rect = option.rect
        selected = bool(option.state & QStyle.StateFlag.State_Selected)
        hovered = bool(option.state & QStyle.StateFlag.State_MouseOver)
        if selected or hovered:
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QColor(PALETTE["bg_raised"] if selected else "#161b24"))
            painter.drawRoundedRect(rect.adjusted(4, 2, -4, -2), 6, 6)

        if model is None:
            painter.restore()
            return

        base = QFont(option.font)
        mono = _mono_font(base)
        left = rect.left() + 14
        right = rect.right() - 14
        top = rect.top() + 7
        line_two = rect.top() + 25

        # Right column first: it is fixed-width, and what is left over is what
        # the name and id get to use.
        prices = format_prices(model)
        meta = "  ·  ".join(
            part
            for part in (
                PROVIDER_NAMES.get(model.provider, model.provider),
                format_context(model.context_length),
            )
            if part
        )
        price_font = QFont(base)
        price_font.setPointSize(max(7, base.pointSize() - 1))
        meta_font = QFont(price_font)

        price_width = QFontMetrics(price_font).horizontalAdvance(prices)
        meta_width = QFontMetrics(meta_font).horizontalAdvance(meta)
        column = max(price_width, meta_width)

        painter.setFont(price_font)
        painter.setPen(
            QColor(PALETTE["text"] if model.pricing_known else PALETTE["text_faint"])
        )
        painter.drawText(
            QRect(right - column, top, column, 16),
            int(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter),
            prices,
        )
        painter.setFont(meta_font)
        painter.setPen(QColor(PALETTE["text_faint"]))
        painter.drawText(
            QRect(right - column, line_two, column, 16),
            int(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter),
            meta,
        )

        available = max(40, (right - column - 18) - left)

        # A live model is a different kind of thing, not merely a different
        # vendor, so it gets a mark rather than a word buried in the meta line.
        name_left = left
        if model.is_live:
            tag = "LIVE"
            tag_font = QFont(base)
            tag_font.setPointSize(max(6, base.pointSize() - 3))
            tag_font.setBold(True)
            tag_width = QFontMetrics(tag_font).horizontalAdvance(tag) + 12
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QColor(PALETTE["accent"]))
            painter.drawRoundedRect(QRect(left, top + 1, tag_width, 15), 7, 7)
            painter.setFont(tag_font)
            painter.setPen(QColor("#17130a"))
            painter.drawText(
                QRect(left, top + 1, tag_width, 15),
                int(Qt.AlignmentFlag.AlignCenter),
                tag,
            )
            name_left = left + tag_width + 8
            available = max(40, (right - column - 18) - name_left)

        painter.setFont(base)
        painter.setPen(QColor(PALETTE["text"]))
        painter.drawText(
            QRect(name_left, top, available, 16),
            int(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter),
            QFontMetrics(base).elidedText(
                model.name, Qt.TextElideMode.ElideRight, available
            ),
        )

        painter.setFont(mono)
        painter.setPen(QColor(PALETTE["text_faint"]))
        id_width = max(40, (right - column - 18) - left)
        painter.drawText(
            QRect(left, line_two, id_width, 16),
            int(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter),
            QFontMetrics(mono).elidedText(
                model.id, Qt.TextElideMode.ElideMiddle, id_width
            ),
        )
        painter.restore()


class _Popup(QWidget):
    """The search-and-filter panel a picker opens.

    A ``Qt.Popup`` rather than a dialog: it closes on a click anywhere else and
    never takes the settings window out of focus, which is what a dropdown is
    expected to do.
    """

    chosen = Signal(str)

    def __init__(self, picker: ModelPicker) -> None:
        super().__init__(picker, Qt.WindowType.Popup)
        self._picker = picker
        self._models: list[ModelInfo] = []
        self._provider = "all"
        self.setObjectName("modelPopup")
        self.setStyleSheet(_popup_stylesheet())
        self._build()

    def _build(self) -> None:
        """Search row, filter chips, list, footnote."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(1, 1, 1, 1)
        layout.setSpacing(0)

        search_row = QWidget(self)
        search_row.setObjectName("searchRow")
        row = QHBoxLayout(search_row)
        row.setContentsMargins(12, 10, 12, 10)
        row.setSpacing(8)
        glyph = QLabel("⌕", search_row)
        glyph.setObjectName("searchGlyph")
        row.addWidget(glyph)
        self.search = QLineEdit(search_row)
        self.search.setObjectName("modelSearch")
        self.search.setPlaceholderText("Search models, or paste a model id")
        self.search.setFrame(False)
        self.search.textChanged.connect(lambda _: self._refill())
        row.addWidget(self.search, stretch=1)
        layout.addWidget(search_row)

        self.chip_row = QWidget(self)
        self.chip_row.setObjectName("chipRow")
        chips = QHBoxLayout(self.chip_row)
        chips.setContentsMargins(12, 8, 12, 8)
        chips.setSpacing(6)
        chips.addStretch(1)
        layout.addWidget(self.chip_row)
        self._chips: list[QPushButton] = []

        self.list = QListWidget(self)
        self.list.setObjectName("modelList")
        self.list.setItemDelegate(_RowDelegate(self.list))
        self.list.setMouseTracking(True)
        self.list.setUniformItemSizes(True)
        self.list.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.list.setVerticalScrollMode(QAbstractItemView.ScrollMode.ScrollPerPixel)
        self.list.itemActivated.connect(self._choose)
        self.list.itemClicked.connect(self._choose)
        layout.addWidget(self.list, stretch=1)

        self.footnote = QLabel("Prices are USD per million tokens, input / output.")
        self.footnote.setObjectName("modelFootnote")
        self.footnote.setContentsMargins(12, 8, 12, 8)
        layout.addWidget(self.footnote)

        self.search.installEventFilter(self)

    # ------------------------------------------------------------- filtering

    def set_models(self, models: list[ModelInfo]) -> None:
        """Adopt the catalogue and rebuild the chips for what is in it."""
        self._models = list(models)
        if self._provider != "all" and not any(
            m.provider == self._provider for m in self._models
        ):
            self._provider = "all"
        self._rebuild_chips()
        self._refill()

    def _rebuild_chips(self) -> None:
        """One chip per provider actually present, plus All.

        Built from the catalogue rather than from the provider registry: a chip
        for a provider with no key would filter to an empty list and tell the
        user nothing they can act on from here.
        """
        layout = self.chip_row.layout()
        for chip in self._chips:
            layout.removeWidget(chip)
            chip.deleteLater()
        self._chips = []

        counts = {"all": len(self._models)}
        for model in self._models:
            counts[model.provider] = counts.get(model.provider, 0) + 1

        offered = [("all", "All")] + [
            (provider, PROVIDER_NAMES.get(provider, provider))
            for provider in PROVIDER_ORDER
            if counts.get(provider)
        ]
        for index, (key, name) in enumerate(offered):
            chip = QPushButton(f"{name}  {counts.get(key, 0)}", self.chip_row)
            chip.setObjectName("chip")
            chip.setCheckable(True)
            chip.setCursor(Qt.CursorShape.PointingHandCursor)
            chip.setChecked(key == self._provider)
            chip.clicked.connect(lambda _=False, k=key: self._set_provider(k))
            layout.insertWidget(index, chip)
            self._chips.append(chip)

    def _set_provider(self, provider: str) -> None:
        """Apply a provider filter and reflect it in the chips."""
        self._provider = provider
        for chip in self._chips:
            chip.setChecked(chip.text().startswith(_chip_name(provider)))
        self._refill()

    def _matches(self, model: ModelInfo, needle: str) -> bool:
        """Whether a model answers to `needle` on any of the fields shown."""
        if self._provider != "all" and model.provider != self._provider:
            return False
        if not needle:
            return True
        haystack = f"{model.name} {model.id} {model.vendor}".lower()
        return all(word in haystack for word in needle.split())

    def _refill(self) -> None:
        """Rebuild the visible rows for the current search and filter."""
        needle = self.search.text().strip().lower()
        self.list.clear()

        if self._picker.allow_empty and not needle:
            self._add_row(
                ModelInfo(
                    id="",
                    provider="inherit",
                    provider_model_id="",
                    name=self._picker.empty_label,
                    vendor="",
                    context_length=None,
                    prompt_price=0.0,
                    completion_price=0.0,
                    pricing_known=False,
                    supports_tools=True,
                )
            )

        shown = [m for m in self._models if self._matches(m, needle)]
        for model in shown:
            self._add_row(model)

        # A typed id nothing matches is an offer, not a dead end. Only when it
        # looks like an id — a bare word is far more likely to be a search that
        # has not finished than a model nobody has heard of.
        typed = self.search.text().strip()
        if typed and "/" in typed and not any(m.id == typed for m in shown):
            offer = describe_unknown(typed)
            self._add_row(
                ModelInfo(**{**offer.__dict__, "name": f"Use “{typed}” as typed"})
            )

        if self.list.count():
            self.list.setCurrentRow(0)
        self.footnote.setText(
            "Prices are USD per million tokens, input / output."
            if self.list.count()
            else "No model matches that. Paste a full model id to use it anyway."
        )

    def _add_row(self, model: ModelInfo) -> None:
        """Append one model row."""
        item = QListWidgetItem(self.list)
        item.setData(Qt.ItemDataRole.UserRole, model)
        item.setSizeHint(QSize(0, ROW_HEIGHT))

    # -------------------------------------------------------------- choosing

    def _choose(self, item: QListWidgetItem) -> None:
        """Emit the chosen id and close."""
        model: ModelInfo = item.data(Qt.ItemDataRole.UserRole)
        self.chosen.emit(model.id)
        self.close()

    def open_under(self, field: QWidget, current: str) -> None:
        """Show the popup under `field`, scrolled to the current selection."""
        self.search.clear()
        self._refill()
        self._select(current)

        width = max(field.width(), MIN_POPUP_WIDTH)
        self.resize(width, POPUP_HEIGHT)
        point = field.mapToGlobal(QPoint(0, field.height() + 4))
        screen = field.screen().availableGeometry() if field.screen() else None
        if screen is not None:
            if point.y() + POPUP_HEIGHT > screen.bottom():
                point.setY(field.mapToGlobal(QPoint(0, 0)).y() - POPUP_HEIGHT - 4)
            point.setX(min(point.x(), screen.right() - width))
            point.setX(max(point.x(), screen.left()))
        self.move(point)
        self.show()
        self.search.setFocus(Qt.FocusReason.PopupFocusReason)

    def _select(self, model_id: str) -> None:
        """Highlight the row for `model_id`, if it is showing."""
        for row in range(self.list.count()):
            model = self.list.item(row).data(Qt.ItemDataRole.UserRole)
            if model is not None and model.id == model_id:
                self.list.setCurrentRow(row)
                self.list.scrollToItem(
                    self.list.item(row), QAbstractItemView.ScrollHint.PositionAtCenter
                )
                return

    def eventFilter(self, watched: QObject, event: QEvent) -> bool:
        """Let the search field drive the list without losing the caret.

        Arrows and Enter belong to the list, everything else to the search box.
        Typing and steering are the same gesture here, so making the user tab
        between them would be the wrong kind of correct.
        """
        if watched is self.search and event.type() == QEvent.Type.KeyPress:
            key = event.key()
            if key in (
                Qt.Key.Key_Down,
                Qt.Key.Key_Up,
                Qt.Key.Key_PageDown,
                Qt.Key.Key_PageUp,
            ):
                # Handed on rather than copied: this filter consumes the event
                # either way, so the list is free to take it as its own.
                self.list.keyPressEvent(event)
                return True
            if key in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
                item = self.list.currentItem()
                if item is not None:
                    self._choose(item)
                return True
            if key == Qt.Key.Key_Escape:
                self.close()
                return True
        return super().eventFilter(watched, event)


class ModelPicker(QWidget):
    """A model chooser whose value is a model id however it was chosen.

    Signals:
        selectionChanged (str): A different model id was chosen.

    Attributes:
        allow_empty (bool): Whether "no model" is a valid choice, as it is for
            the observer model, where blank means "the same one as everything
            else".
        empty_label (str): What the empty choice is called.
    """

    selectionChanged = Signal(str)

    def __init__(
        self,
        parent: QWidget | None = None,
        *,
        allow_empty: bool = False,
        empty_label: str = "Use the model chosen above",
    ) -> None:
        """Create an empty picker."""
        super().__init__(parent)
        self.allow_empty = allow_empty
        self.empty_label = empty_label
        self._models: list[ModelInfo] = []
        self._selection = ""
        self._popup: _Popup | None = None

        self.setObjectName("modelPicker")
        self.setStyleSheet(_field_stylesheet())
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self._build()
        self._render()

    def _build(self) -> None:
        """The card: name and id on the left, provider and prices on the right."""
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)

        self.card = QFrame(self)
        self.card.setObjectName("modelCard")
        outer.addWidget(self.card)

        row = QHBoxLayout(self.card)
        row.setContentsMargins(12, 8, 10, 8)
        row.setSpacing(10)

        left = QVBoxLayout()
        left.setContentsMargins(0, 0, 0, 0)
        left.setSpacing(1)
        name_row = QHBoxLayout()
        name_row.setContentsMargins(0, 0, 0, 0)
        name_row.setSpacing(7)
        # The same pill the list rows carry. Which mode a model runs in decides
        # more than anything else on this page, so it should look identical
        # wherever it is stated.
        self.live_tag = QLabel("LIVE", self.card)
        self.live_tag.setObjectName("modelLiveTag")
        self.live_tag.hide()
        name_row.addWidget(self.live_tag)
        self.name_label = QLabel(self.card)
        self.name_label.setObjectName("modelName")
        name_row.addWidget(self.name_label)
        name_row.addStretch(1)
        left.addLayout(name_row)
        self.id_label = QLabel(self.card)
        self.id_label.setObjectName("modelId")
        self.id_label.setFont(_mono_font(self.font()))
        left.addWidget(self.id_label)
        row.addLayout(left, stretch=1)

        right = QVBoxLayout()
        right.setContentsMargins(0, 0, 0, 0)
        right.setSpacing(1)
        self.provider_label = QLabel(self.card)
        self.provider_label.setObjectName("modelProvider")
        self.provider_label.setAlignment(Qt.AlignmentFlag.AlignRight)
        right.addWidget(self.provider_label)
        self.price_label = QLabel(self.card)
        self.price_label.setObjectName("modelPrice")
        self.price_label.setAlignment(Qt.AlignmentFlag.AlignRight)
        right.addWidget(self.price_label)
        row.addLayout(right)

        self.chevron = QLabel("⌄", self.card)
        self.chevron.setObjectName("modelChevron")
        row.addWidget(self.chevron)

    # ------------------------------------------------------------------ API

    def set_models(self, models: list[ModelInfo]) -> None:
        """Replace the offered models, keeping whatever is currently selected.

        Catalogues arrive asynchronously, well after the form was populated, so
        preserving the selection across a repopulate is not a nicety — without
        it, opening settings and waiting a second would silently change which
        model the player is about to save.

        Args:
            models (list[ModelInfo]): The entries to offer, in display order.
        """
        self._models = list(models)
        if self._popup is not None:
            self._popup.set_models(self._models)
        self._render()

    def models(self) -> list[ModelInfo]:
        """The models currently offered."""
        return list(self._models)

    def model_ids(self) -> list[str]:
        """The ids currently offered, in display order."""
        return [model.id for model in self._models]

    def selection(self) -> str:
        """The chosen model id, or empty when nothing is chosen."""
        return self._selection

    def set_selection(self, model_id: str) -> None:
        """Select `model_id`, listed or not, without emitting a change."""
        self._selection = (model_id or "").strip()
        self._render()

    def choose(self, model_id: str) -> None:
        """Select `model_id` as if the user had, announcing the change."""
        text = (model_id or "").strip()
        if text == self._selection:
            return
        self._selection = text
        self._render()
        self.selectionChanged.emit(text)

    # ------------------------------------------------------------- rendering

    def _current(self) -> ModelInfo | None:
        """The catalogue entry for the selection, or None when it is unlisted."""
        return next((m for m in self._models if m.id == self._selection), None)

    def _render(self) -> None:
        """Redraw the card for the current selection."""
        if not self._selection:
            self.live_tag.hide()
            self.name_label.setText(
                self.empty_label if self.allow_empty else "Choose a model"
            )
            self.id_label.setText("")
            self.provider_label.setText("")
            self.price_label.setText("")
            return

        known = self._current()
        model = known or describe_unknown(self._selection)
        self.live_tag.setVisible(model.is_live)
        self.name_label.setText(model.name if known else self._selection)
        self.id_label.setText(model.id)
        self.provider_label.setText(
            PROVIDER_NAMES.get(model.provider, "Typed in").upper()
        )
        self.price_label.setText(format_prices(model) if known else "not in catalogue")

    # ------------------------------------------------------------ interaction

    def open_popup(self) -> None:
        """Show the search panel, building it on first use."""
        if not self.isEnabled():
            return
        if self._popup is None:
            self._popup = _Popup(self)
            self._popup.chosen.connect(self.choose)
            self._popup.set_models(self._models)
        self._popup.open_under(self, self._selection)

    def mousePressEvent(self, event) -> None:
        """Clicking anywhere on the card opens the panel."""
        if event.button() == Qt.MouseButton.LeftButton:
            self.open_popup()
            event.accept()
            return
        super().mousePressEvent(event)

    def keyPressEvent(self, event: QKeyEvent) -> None:
        """Space, Enter or Down opens the panel, as a combo box would."""
        if event.key() in (
            Qt.Key.Key_Space,
            Qt.Key.Key_Return,
            Qt.Key.Key_Enter,
            Qt.Key.Key_Down,
        ):
            self.open_popup()
            event.accept()
            return
        super().keyPressEvent(event)


def _chip_name(provider: str) -> str:
    """The chip label for a provider key, without its count."""
    return "All" if provider == "all" else PROVIDER_NAMES.get(provider, provider)


def _field_stylesheet() -> str:
    """Styling for the collapsed card."""
    p = PALETTE
    return f"""
    QFrame#modelCard {{
        background-color: {p["bg_input"]};
        border: 1px solid {p["border"]};
        border-radius: 6px;
    }}
    QWidget#modelPicker:focus QFrame#modelCard,
    QFrame#modelCard:hover {{
        border-color: {p["accent_dim"]};
    }}
    QLabel {{ background: transparent; }}
    QLabel#modelName {{ color: {p["text"]}; }}
    QLabel#modelId {{ color: {p["text_faint"]}; }}
    QLabel#modelLiveTag {{
        background-color: {p["accent"]};
        color: #17130a;
        border-radius: 8px;
        padding: 2px 8px;
        font-size: 8pt;
        font-weight: 700;
    }}
    QLabel#modelProvider {{
        color: {p["text_dim"]};
        font-size: 8pt;
        letter-spacing: 1px;
    }}
    QLabel#modelPrice {{ color: {p["text_dim"]}; font-size: 8pt; }}
    QLabel#modelChevron {{ color: {p["text_faint"]}; padding-left: 2px; }}
    QWidget#modelPicker:disabled QFrame#modelCard {{
        background-color: {p["bg"]};
        border-color: {p["border"]};
    }}
    QWidget#modelPicker:disabled QLabel {{ color: {p["text_faint"]}; }}
    """


def _popup_stylesheet() -> str:
    """Styling for the search panel."""
    p = PALETTE
    return f"""
    QWidget#modelPopup {{
        background-color: {p["bg_raised"]};
        border: 1px solid {p["border"]};
        border-radius: 8px;
    }}
    QWidget#searchRow {{
        background-color: {p["bg_input"]};
        border-bottom: 1px solid {p["border"]};
    }}
    QWidget#chipRow {{ background-color: {p["bg_raised"]}; }}
    QLabel#searchGlyph {{ color: {p["text_faint"]}; font-size: 13pt; }}
    QLineEdit#modelSearch {{
        background: transparent;
        border: none;
        padding: 0px;
        color: {p["text"]};
        selection-background-color: {p["accent_dim"]};
    }}
    QPushButton#chip {{
        background-color: {p["bg"]};
        border: 1px solid {p["border"]};
        border-radius: 11px;
        padding: 3px 11px;
        font-size: 9pt;
        color: {p["text_dim"]};
    }}
    QPushButton#chip:hover {{ color: {p["text"]}; }}
    QPushButton#chip:checked {{
        background-color: {p["bg_input"]};
        border-color: {p["accent"]};
        color: {p["text"]};
    }}
    QListWidget#modelList {{
        background-color: {p["bg_raised"]};
        border: none;
        border-top: 1px solid {p["border"]};
        outline: none;
    }}
    QLabel#modelFootnote {{
        background-color: {p["bg_raised"]};
        border-top: 1px solid {p["border"]};
        color: {p["text_faint"]};
        font-size: 8pt;
    }}
    QScrollBar:vertical {{
        background: transparent;
        width: 10px;
        margin: 2px;
    }}
    QScrollBar::handle:vertical {{
        background: {p["border"]};
        border-radius: 5px;
        min-height: 28px;
    }}
    QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0px; }}
    """


__all__ = [
    "MIN_POPUP_WIDTH",
    "PROVIDER_NAMES",
    "ROW_HEIGHT",
    "ModelPicker",
    "format_context",
    "format_price",
    "format_prices",
]
