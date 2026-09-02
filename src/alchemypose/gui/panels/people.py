"""Which person the part-selection applies to.

With two or more people on screen the user must choose a subject before
choosing a limb; every multi-instance tool surveyed (SLEAP, CVAT, FiftyOne)
provides exactly this. No chip is checked until the user picks one, because
"everyone" is the honest default for a fresh result; clicking the checked chip
again returns to it.
"""

from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QHBoxLayout, QToolButton, QWidget


class PersonBar(QWidget):
    """A row of numbered chips, one per detected person."""

    person_chosen = Signal(int)
    everyone_chosen = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._layout = QHBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._layout.setSpacing(4)
        self.chips: list[QToolButton] = []

    @property
    def current(self) -> int | None:
        """The chosen person's index, or ``None`` when everyone is shown."""
        for index, chip in enumerate(self.chips):
            if chip.isChecked():
                return index
        return None

    def set_people(self, count: int) -> None:
        """Rebuild the chips. Cheap enough that diffing is not worth it."""
        for chip in self.chips:
            self._layout.removeWidget(chip)
            chip.deleteLater()
        self.chips.clear()

        for index in range(count):
            chip = QToolButton(self)
            chip.setText(f"P{index + 1}")
            chip.setToolTip(f"Show only person {index + 1}")
            chip.setCheckable(True)
            chip.clicked.connect(lambda _checked=False, i=index: self._choose(i))
            self._layout.addWidget(chip)
            self.chips.append(chip)

    def _choose(self, index: int) -> None:
        # ``clicked`` fires after Qt has already toggled the chip, so a chip that
        # is now unchecked was the chosen one a moment ago: that click means "everyone".
        if not self.chips[index].isChecked():
            self.everyone_chosen.emit()
            return
        for position, chip in enumerate(self.chips):
            chip.setChecked(position == index)
        self.person_chosen.emit(index)
