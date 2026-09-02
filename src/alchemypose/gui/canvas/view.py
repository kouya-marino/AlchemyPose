"""The image viewport: zoom under the cursor, middle-drag to pan.

Keypoints live in the same scene in *image pixel* coordinates, so the overlay is
registered to the photograph by construction — no manual rescaling anywhere.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QImage, QMouseEvent, QPainter, QPixmap, QWheelEvent
from PySide6.QtWidgets import QGraphicsPixmapItem, QGraphicsScene, QGraphicsView

from alchemypose.gui import theme

ZOOM_STEP = 1.15
MIN_ZOOM = 0.05
MAX_ZOOM = 40.0


def bgr_to_pixmap(bgr: NDArray[np.uint8]) -> QPixmap:
    """OpenCV BGR to a QPixmap, copying because QImage does not own the buffer."""
    height, width = bgr.shape[:2]
    rgb = np.ascontiguousarray(bgr[:, :, ::-1])
    image = QImage(rgb.data, width, height, 3 * width, QImage.Format.Format_RGB888)
    return QPixmap.fromImage(image.copy())


class PoseCanvas(QGraphicsView):
    """A zoomable, pannable image with room for an overlay above it."""

    def __init__(self) -> None:
        super().__init__()
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)

        self.setRenderHints(
            QPainter.RenderHint.Antialiasing | QPainter.RenderHint.SmoothPixmapTransform
        )
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setBackgroundBrush(Qt.GlobalColor.transparent)
        self.setStyleSheet(f"QGraphicsView {{ background: {theme.CANVAS_BG}; border: none; }}")
        self.setMouseTracking(True)

        self._pixmap_item: QGraphicsPixmapItem | None = None
        self._panning_from: QPointF | None = None

    # ------------------------------------------------------------------ image

    def set_image(self, bgr: NDArray[np.uint8]) -> None:
        """Show a new image, discarding whatever was there."""
        self._scene.clear()
        self._pixmap_item = None

        pixmap = bgr_to_pixmap(bgr)
        self._pixmap_item = self._scene.addPixmap(pixmap)
        self._pixmap_item.setZValue(-100)
        self._scene.setSceneRect(QRectF(pixmap.rect()))
        self.resetTransform()

    @property
    def image_item(self) -> QGraphicsPixmapItem | None:
        return self._pixmap_item

    # ------------------------------------------------------------------- zoom

    @property
    def zoom_percent(self) -> float:
        return float(self.transform().m11()) * 100.0

    def zoom_by(self, factor: float) -> None:
        """Scale, refusing steps that would leave the allowed range."""
        target = self.transform().m11() * factor
        if not (MIN_ZOOM <= target <= MAX_ZOOM):
            return
        self.scale(factor, factor)

    def fit_to_window(self) -> None:
        if self._pixmap_item is not None:
            self.fitInView(self._pixmap_item, Qt.AspectRatioMode.KeepAspectRatio)

    def zoom_to_rect(self, rect: QRectF, margin: float = 0.15) -> None:
        """Frame a scene rectangle — used by the Face preset to magnify the head."""
        if rect.isEmpty():
            return
        padded = rect.adjusted(
            -rect.width() * margin,
            -rect.height() * margin,
            rect.width() * margin,
            rect.height() * margin,
        )
        self.fitInView(padded, Qt.AspectRatioMode.KeepAspectRatio)

    def wheelEvent(self, event: QWheelEvent) -> None:
        if event.modifiers() & Qt.KeyboardModifier.ControlModifier:
            self.zoom_by(ZOOM_STEP if event.angleDelta().y() > 0 else 1.0 / ZOOM_STEP)
            event.accept()
            return
        super().wheelEvent(event)

    # -------------------------------------------------------------------- pan

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.MiddleButton:
            self._panning_from = event.position()
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if self._panning_from is not None:
            delta = event.position() - self._panning_from
            self._panning_from = event.position()
            h, v = self.horizontalScrollBar(), self.verticalScrollBar()
            h.setValue(h.value() - int(delta.x()))
            v.setValue(v.value() - int(delta.y()))
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.MiddleButton:
            self._panning_from = None
            self.unsetCursor()
            event.accept()
            return
        super().mouseReleaseEvent(event)
