"""One-click visibility presets.

A chip is not a mode. Clicking one sets the layer tree's check states and
nothing else, so the tree stays the single source of truth. A chip lights up
whenever the tree happens to equal it — including when the user got there by
hand.
"""

from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QGridLayout, QToolButton, QWidget

from alchemypose.core.presets import PRESETS

LABELS: dict[str, str] = {
    "full_body": "Full",
    "upper": "Upper",
    "lower": "Lower",
    "left_side": "Left",
    "right_side": "Right",
    "hands": "Hands",
    "face": "Face",
    "all": "All",
    "none": "None",
}


class PresetBar(QWidget):
    """A small grid of chips, three per row."""

    preset_chosen = Signal(str)

    def __init__(self) -> None:
        super().__init__()
        layout = QGridLayout(self)
        layout.setContentsMargins(0, 0, 0, 6)
        layout.setSpacing(4)

        self.chips: dict[str, QToolButton] = {}
        for position, name in enumerate(PRESETS):
            chip = QToolButton()
            chip.setText(LABELS[name])
            chip.setCheckable(True)
            chip.clicked.connect(lambda _checked=False, n=name: self.preset_chosen.emit(n))
            layout.addWidget(chip, position // 3, position % 3)
            self.chips[name] = chip

    def highlight(self, name: str | None) -> None:
        """Light the chip matching the current state, or none of them."""
        for chip_name, chip in self.chips.items():
            chip.setChecked(chip_name == name)
