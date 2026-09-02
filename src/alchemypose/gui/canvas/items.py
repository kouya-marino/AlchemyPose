"""The two drawing primitives.

Both are positioned in image-pixel scene coordinates. Markers set
``ItemIgnoresTransformations`` so a joint stays 6 px at any zoom, and bones use
cosmetic pens so a limb never becomes a slab when the user zooms in.
"""

from __future__ import annotations

from collections.abc import Callable

from PySide6.QtCore import QLineF, QPointF, QRectF, Qt
from PySide6.QtGui import QBrush, QColor, QLinearGradient, QPainterPath, QPen
from PySide6.QtWidgets import (
    QGraphicsEllipseItem,
    QGraphicsItem,
    QGraphicsLineItem,
    QGraphicsSceneHoverEvent,
)

MARKER_RADIUS = 3.0          # screen pixels; the dot is 6 px across
HIT_RADIUS = 11.0            # screen pixels; draw small, hit big
BONE_WIDTH = 3.0
GLOW_WIDTH = 9.0
GLOW_ALPHA = 64              # 25% of 255


class KeypointItem(QGraphicsEllipseItem):
    """One joint. Knows who and what it is, so hit-testing needs no lookup table.

    Carries ``person_index``, ``keypoint_index`` and ``group_name`` (its render
    group, e.g. ``"left_arm"``). The attribute is ``group_name`` rather than the
    more obvious ``group`` because ``QGraphicsItem`` already defines a real
    ``group()`` method (the legacy ``QGraphicsItemGroup`` API); shadowing it
    would silently turn `item.group()` into a `TypeError` for any code — ours,
    a future contributor's, or a library — that expects the standard Qt method.
    """

    def __init__(
        self,
        person_index: int,
        keypoint_index: int,
        group_name: str,
        name: str,
        color: QColor,
        score: float,
        on_hover_enter: Callable[[KeypointItem], None] | None = None,
        on_hover_leave: Callable[[], None] | None = None,
    ) -> None:
        super().__init__(-MARKER_RADIUS, -MARKER_RADIUS, MARKER_RADIUS * 2, MARKER_RADIUS * 2)
        self.person_index = person_index
        self.keypoint_index = keypoint_index
        self.group_name = group_name
        self._on_hover_enter = on_hover_enter
        self._on_hover_leave = on_hover_leave

        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIgnoresTransformations, True)
        self.setAcceptHoverEvents(True)
        self.setZValue(20)
        self.setBrush(QBrush(color))
        self.setPen(QPen(QColor(255, 255, 255, 200), 1.0))
        self.setToolTip(f"{name} · {score:.2f}")

    def shape(self) -> QPainterPath:
        """A hit area far larger than the dot — 3 px targets are miserable to click."""
        path = QPainterPath()
        path.addEllipse(QPointF(0, 0), HIT_RADIUS, HIT_RADIUS)
        return path

    def boundingRect(self) -> QRectF:
        # Must cover the hit shape, or Qt leaves artefacts when the item redraws.
        return QRectF(-HIT_RADIUS, -HIT_RADIUS, HIT_RADIUS * 2, HIT_RADIUS * 2)

    def hoverEnterEvent(self, event: QGraphicsSceneHoverEvent) -> None:
        if self._on_hover_enter is not None:
            self._on_hover_enter(self)
        super().hoverEnterEvent(event)

    def hoverLeaveEvent(self, event: QGraphicsSceneHoverEvent) -> None:
        if self._on_hover_leave is not None:
            self._on_hover_leave()
        super().hoverLeaveEvent(event)


class BoneItem(QGraphicsLineItem):
    """A limb segment: a bright cosmetic stroke over a wide translucent one.

    Two-pass stroking is how the glow is done. A real blur effect is a per-item
    raster pass, which is unaffordable with hundreds of bones on screen.
    """

    def __init__(
        self, a: QPointF, b: QPointF, color_a: QColor, color_b: QColor, *, confident: bool
    ) -> None:
        super().__init__(QLineF(a, b))
        self.setZValue(10)

        gradient = QLinearGradient(a, b)
        gradient.setColorAt(0.0, color_a)
        gradient.setColorAt(1.0, color_b)

        pen = QPen(QBrush(gradient), BONE_WIDTH)
        pen.setCosmetic(True)
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        if not confident:
            pen.setStyle(Qt.PenStyle.DashLine)
        self.setPen(pen)

        glow_color = QColor(color_a)
        glow_color.setAlpha(GLOW_ALPHA)
        self._glow = QGraphicsLineItem(QLineF(a, b), self)
        glow_pen = QPen(glow_color, GLOW_WIDTH)
        glow_pen.setCosmetic(True)
        glow_pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        self._glow.setPen(glow_pen)
        self._glow.setZValue(-1)
