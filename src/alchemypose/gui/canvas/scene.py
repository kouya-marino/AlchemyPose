"""Turns a PoseResult plus a KeypointSelection into graphics items.

Rebuilding from scratch on every change is deliberate: a few thousand items is
well inside Graphics View's range, and SLEAP — a PySide pose GUI — clears and
rebuilds its scene every video frame. Simplicity beats a diffing algorithm here.
"""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import QObject, QPointF, QRectF, Signal

from alchemypose.core.schema import KeypointSelection, PoseResult
from alchemypose.gui import theme
from alchemypose.gui.canvas.items import BoneItem, KeypointItem
from alchemypose.gui.canvas.view import PoseCanvas

DASH_BELOW = 0.5
"""A bone whose endpoint confidence falls under this is drawn dashed."""


class PoseOverlay(QObject):
    """Owns every skeleton item in the canvas's scene.

    Inherits ``QObject`` (rather than holding a separate signals object) purely
    so hover can be surfaced as ordinary Qt signals — ``KeypointItem`` itself is
    not a ``QObject`` (``QGraphicsItem`` does not support multiple inheritance
    with it cleanly), so its hover events are routed here through a plain
    callback and re-emitted as ``keypoint_hovered`` / ``keypoint_left``. Task 20
    connects these to the inspector panel.
    """

    keypoint_hovered = Signal(int, int, str)   # person_index, keypoint_index, group
    keypoint_left = Signal()

    def __init__(self, canvas: PoseCanvas) -> None:
        super().__init__(canvas)
        self._canvas = canvas
        self._result: PoseResult | None = None
        self._selection = KeypointSelection(visible=frozenset())
        self._threshold = 0.3
        self.keypoint_items: list[KeypointItem] = []
        self.bone_items: list[BoneItem] = []

    # ----------------------------------------------------------------- counts

    @property
    def keypoint_item_count(self) -> int:
        return len(self.keypoint_items)

    @property
    def bone_item_count(self) -> int:
        return len(self.bone_items)

    @property
    def visible_item_count(self) -> int:
        return self.keypoint_item_count + self.bone_item_count

    # ------------------------------------------------------------------ state

    def set_result(
        self, result: PoseResult, selection: KeypointSelection, confidence_threshold: float
    ) -> None:
        self._result = result
        self._selection = selection
        self._threshold = confidence_threshold
        self._rebuild()

    def set_selection(self, selection: KeypointSelection) -> None:
        self._selection = selection
        self._rebuild()

    def set_confidence_threshold(self, value: float) -> None:
        self._threshold = value
        self._rebuild()

    # ------------------------------------------------------------------ build

    def _clear(self) -> None:
        scene = self._canvas.scene()
        for item in (*self.keypoint_items, *self.bone_items):
            scene.removeItem(item)
        self.keypoint_items.clear()
        self.bone_items.clear()

    def _rebuild(self) -> None:
        self._clear()
        if self._result is None:
            return

        scene = self._canvas.scene()
        topology = self._result.topology
        frame = self._result.frames[0]

        for person_index, person in enumerate(frame.people):
            if self._selection.persons is not None and person_index not in self._selection.persons:
                continue

            def drawable(i: int) -> bool:
                return self._selection.is_visible(i) and float(person.scores[i]) >= self._threshold

            for a, b in topology.edges:
                if not (drawable(a) and drawable(b)):
                    continue
                pa = QPointF(float(person.keypoints[a][0]), float(person.keypoints[a][1]))
                pb = QPointF(float(person.keypoints[b][0]), float(person.keypoints[b][1]))
                bone = BoneItem(
                    pa,
                    pb,
                    theme.color_for(topology.group_of(a) or ""),
                    theme.color_for(topology.group_of(b) or ""),
                    confident=min(float(person.scores[a]), float(person.scores[b])) >= DASH_BELOW,
                )
                scene.addItem(bone)
                self.bone_items.append(bone)

            for i in range(len(person)):
                if not drawable(i):
                    continue
                group_name = topology.group_of(i) or ""
                item = KeypointItem(
                    person_index=person_index,
                    keypoint_index=i,
                    group_name=group_name,
                    name=topology.keypoint_names[i],
                    color=theme.color_for(group_name),
                    score=float(person.scores[i]),
                    on_hover_enter=self._on_keypoint_hover_enter,
                    on_hover_leave=self._on_keypoint_hover_leave,
                )
                item.setPos(float(person.keypoints[i][0]), float(person.keypoints[i][1]))
                scene.addItem(item)
                self.keypoint_items.append(item)

    # ----------------------------------------------------------------- hover

    def _on_keypoint_hover_enter(self, item: KeypointItem) -> None:
        self.keypoint_hovered.emit(item.person_index, item.keypoint_index, item.group_name)

    def _on_keypoint_hover_leave(self) -> None:
        self.keypoint_left.emit()

    # ----------------------------------------------------------------- bounds

    def group_bounds(self, person_index: int, group: str) -> QRectF:
        """The scene rectangle covering one group of one person — used to zoom to it."""
        if self._result is None:
            return QRectF()
        person = self._result.frames[0].people[person_index]
        indices = self._result.topology.groups.get(group, ())
        if not indices:
            return QRectF()
        xs = [float(person.keypoints[i][0]) for i in indices]
        ys = [float(person.keypoints[i][1]) for i in indices]
        return QRectF(min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys))

    def person_bounds(self, person_index: int) -> QRectF:
        if self._result is None:
            return QRectF()
        person = self._result.frames[0].people[person_index]
        if person.bbox is not None:
            x0, y0, x1, y1 = person.bbox
            return QRectF(x0, y0, x1 - x0, y1 - y0)
        xs, ys = person.keypoints[:, 0], person.keypoints[:, 1]
        return QRectF(float(xs.min()), float(ys.min()), float(np.ptp(xs)), float(np.ptp(ys)))
