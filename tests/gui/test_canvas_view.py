import numpy as np
import pytest
from PySide6.QtCore import QPoint, Qt
from PySide6.QtGui import QWheelEvent

pytestmark = pytest.mark.gui


def _image(w: int = 640, h: int = 480) -> np.ndarray:
    return np.full((h, w, 3), 40, dtype=np.uint8)


def test_set_image_sizes_the_scene_to_the_image(qtbot) -> None:  # type: ignore[no-untyped-def]
    from alchemypose.gui.canvas.view import PoseCanvas

    canvas = PoseCanvas()
    qtbot.addWidget(canvas)
    canvas.set_image(_image(800, 600))
    assert canvas.scene().sceneRect().width() == 800
    assert canvas.scene().sceneRect().height() == 600


def test_zoom_percent_starts_at_one_hundred(qtbot) -> None:  # type: ignore[no-untyped-def]
    from alchemypose.gui.canvas.view import PoseCanvas

    canvas = PoseCanvas()
    qtbot.addWidget(canvas)
    canvas.set_image(_image())
    assert canvas.zoom_percent == pytest.approx(100.0)


def test_ctrl_wheel_zooms_in_by_the_house_step(qtbot) -> None:  # type: ignore[no-untyped-def]
    from alchemypose.gui.canvas.view import ZOOM_STEP, PoseCanvas

    canvas = PoseCanvas()
    qtbot.addWidget(canvas)
    canvas.set_image(_image())
    canvas.wheelEvent(
        QWheelEvent(
            QPoint(100, 100), canvas.mapToGlobal(QPoint(100, 100)), QPoint(0, 0), QPoint(0, 120),
            Qt.MouseButton.NoButton, Qt.KeyboardModifier.ControlModifier,
            Qt.ScrollPhase.NoScrollPhase, False,
        )
    )
    assert canvas.zoom_percent == pytest.approx(100.0 * ZOOM_STEP, rel=1e-3)


def test_zoom_is_clamped_at_both_ends(qtbot) -> None:  # type: ignore[no-untyped-def]
    from alchemypose.gui.canvas.view import MAX_ZOOM, MIN_ZOOM, PoseCanvas

    canvas = PoseCanvas()
    qtbot.addWidget(canvas)
    canvas.set_image(_image())
    for _ in range(200):
        canvas.zoom_by(2.0)
    assert canvas.zoom_percent <= MAX_ZOOM * 100 + 1
    for _ in range(400):
        canvas.zoom_by(0.5)
    assert canvas.zoom_percent >= MIN_ZOOM * 100 - 1


def test_loading_a_second_image_replaces_the_first(qtbot) -> None:  # type: ignore[no-untyped-def]
    from alchemypose.gui.canvas.view import PoseCanvas

    canvas = PoseCanvas()
    qtbot.addWidget(canvas)
    canvas.set_image(_image(640, 480))
    canvas.set_image(_image(320, 240))
    assert canvas.scene().sceneRect().width() == 320
