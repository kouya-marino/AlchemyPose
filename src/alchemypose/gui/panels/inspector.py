"""Two small readouts: what the cursor is over, and what the colours mean."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QLabel, QWidget

from alchemypose.gui import theme

_RICH = Qt.TextFormat.RichText


class Inspector(QLabel):
    """Name, confidence and group of the keypoint under the cursor.

    Rich text, so the dot can carry the group's colour; ``text()`` returns the
    markup we set, which is what the tests read.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setTextFormat(_RICH)
        self.setStyleSheet(f"color: {theme.TEXT_DIM}; padding: 4px 0;")
        self.setMinimumHeight(22)
        self.clear()

    def show_keypoint(self, name: str, score: float, group: str) -> None:
        color = theme.GROUP_COLORS.get(group, theme.TEXT_DIM)
        pretty_group = group.replace("_", " ")
        self.setText(
            f"<span style='color:{color}'>&#9679;</span> "
            f"<b style='color:{theme.TEXT}'>{name}</b> &nbsp; {score:.2f}"
            f" &nbsp; <span style='color:{theme.TEXT_DIM}'>{pretty_group}</span>"
        )

    def clear(self) -> None:
        self.setText("")


class LegendWidget(QLabel):
    """The side-colour key. It must say *person's* left, not just left.

    Image-left is the opposite of the person's left in a face-on photo, and
    conflating them is the single most common misreading of a pose overlay.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setTextFormat(_RICH)
        chips = " &nbsp; ".join(
            f"<span style='color:{color}'>&#9632;</span> {label}" for label, color in theme.SIDE_LEGEND
        )
        self.setText(chips)
        self.setWordWrap(True)
        self.setStyleSheet(f"color: {theme.TEXT_DIM}; font-size: 11px;")

