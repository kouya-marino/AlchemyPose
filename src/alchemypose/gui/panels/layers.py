"""The visibility tree — the one place that knows what is drawn.

``Qt.ItemIsAutoTristate`` gives parent rows their checked / partial / unchecked
behaviour natively, which is why the hierarchy is exactly two levels deep.

Render groups deliberately overlap on anchor joints (a shoulder belongs to both
``torso_head`` and ``left_arm``, so a limb selected alone still renders connected
to the body). ``set_selection`` decides a row is checked with a subset test —
``set(RENDER_GROUPS[key]) <= selection.visible`` — rather than an exact-equality
test, because a broader selection (e.g. "upper body") legitimately contains a
narrower group's indices as a subset without them being a literal `==` match.
This is safe under the overlap: every render group has at least one index that
belongs to no other group (elbow/wrist for an arm, the head points for
torso_head, toes for a foot, ...), so the only way its full index set can be a
subset of ``selection.visible`` is if that selection was actually built to
include this group. A neighbour that merely shares the anchor joint can never
satisfy the subset test, because it is missing that unique index.
"""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QBrush, QColor, QPixmap
from PySide6.QtWidgets import QTreeWidget, QTreeWidgetItem, QVBoxLayout, QWidget

from alchemypose.core.schema import KeypointSelection
from alchemypose.core.topology import RENDER_GROUPS
from alchemypose.gui import theme

#: parent label -> (child group key, child label), in display order
STRUCTURE: tuple[tuple[str, tuple[tuple[str, str], ...]], ...] = (
    ("Body", (
        ("torso_head", "Torso / Head"),
        ("left_arm", "Left arm"),
        ("right_arm", "Right arm"),
        ("left_leg", "Left leg"),
        ("right_leg", "Right leg"),
    )),
    ("Feet", (("left_foot", "Left foot"), ("right_foot", "Right foot"))),
    ("Face", (
        ("face_jaw", "Oval / Jaw"),
        ("face_brows", "Brows"),
        ("face_eyes", "Eyes"),
        ("face_nose", "Nose"),
        ("face_lips", "Lips"),
    )),
    ("Hands", (("left_hand", "Left"), ("right_hand", "Right"))),
)

_GROUP = Qt.ItemDataRole.UserRole + 1


def _swatch(color: QColor) -> QPixmap:
    pixmap = QPixmap(10, 10)
    pixmap.fill(color)
    return pixmap


class LayerPanel(QWidget):
    """Grouped, tri-state show/hide for every render group."""

    selection_changed = Signal(object)     # KeypointSelection
    group_hovered = Signal(str)

    def __init__(self) -> None:
        super().__init__()
        self.tree = QTreeWidget()
        self.tree.setHeaderHidden(True)
        self.tree.setMouseTracking(True)
        self.tree.setIndentation(14)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.tree)

        self.parent_rows: dict[str, QTreeWidgetItem] = {}
        self.group_rows: dict[str, QTreeWidgetItem] = {}
        self._suppress = False

        for parent_label, children in STRUCTURE:
            parent = QTreeWidgetItem(self.tree, [parent_label])
            parent.setFlags(
                parent.flags() | Qt.ItemFlag.ItemIsUserCheckable | Qt.ItemFlag.ItemIsAutoTristate
            )
            parent.setCheckState(0, Qt.CheckState.Checked)
            parent.setExpanded(True)
            self.parent_rows[parent_label] = parent

            for key, label in children:
                child = QTreeWidgetItem(parent, [label])
                child.setFlags(child.flags() | Qt.ItemFlag.ItemIsUserCheckable)
                child.setCheckState(0, Qt.CheckState.Checked)
                child.setData(0, _GROUP, key)
                child.setIcon(0, _swatch(theme.color_for(key)))
                child.setForeground(0, QBrush(QColor(theme.TEXT)))
                self.group_rows[key] = child

        self.tree.itemChanged.connect(self._on_item_changed)
        self.tree.itemEntered.connect(self._on_item_entered)

    # ---------------------------------------------------------------- reading

    def current_selection(self) -> KeypointSelection:
        """Derive the selection from the tree's actual check states.

        Never from a cached shadow copy — the tree is the source of truth, and
        this is what lets ``set_group_checked``/direct ``setCheckState`` calls
        (including a user checking a parent row) be reflected correctly.
        """
        indices: set[int] = set()
        for key, row in self.group_rows.items():
            if row.checkState(0) == Qt.CheckState.Checked:
                indices.update(RENDER_GROUPS[key])
        return KeypointSelection(visible=frozenset(indices))

    # ---------------------------------------------------------------- writing

    def set_selection(self, selection: KeypointSelection) -> None:
        """Apply a selection in one shot, emitting a single change signal."""
        self._suppress = True
        try:
            for key, row in self.group_rows.items():
                wanted = bool(RENDER_GROUPS[key]) and set(RENDER_GROUPS[key]) <= selection.visible
                row.setCheckState(0, Qt.CheckState.Checked if wanted else Qt.CheckState.Unchecked)
        finally:
            self._suppress = False
        self.selection_changed.emit(self.current_selection())

    def set_group_checked(self, group: str, checked: bool) -> None:
        state = Qt.CheckState.Checked if checked else Qt.CheckState.Unchecked
        self.group_rows[group].setCheckState(0, state)

    # ----------------------------------------------------------------- events

    def _on_item_changed(self, item: QTreeWidgetItem, column: int) -> None:
        if self._suppress:
            return
        self.selection_changed.emit(self.current_selection())

    def _on_item_entered(self, item: QTreeWidgetItem, column: int) -> None:
        group = item.data(0, _GROUP)
        if isinstance(group, str):
            self.group_hovered.emit(group)
