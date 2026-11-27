"""Webcam mode, with a camera that is a list of frames."""

import threading
from dataclasses import replace

import numpy as np
import pytest

from alchemypose.core.schema import PoseFrame, PoseResult
from alchemypose.core.topology import COCO_WHOLEBODY_133
from tests.support.fakes import walker

pytestmark = pytest.mark.gui


class _Camera:
    def __init__(self, frames: int = 5, opens: bool = True) -> None:
        self._frames = [np.full((48, 64, 3), i, dtype=np.uint8) for i in range(frames)]
        self._opens = opens
        self.released = False

    def isOpened(self) -> bool:  # noqa: N802
        return self._opens

    def read(self) -> tuple[bool, np.ndarray | None]:
        if self._frames:
            return True, self._frames.pop(0)
        threading.Event().wait(0.01)
        return False, None

    def release(self) -> None:
        self.released = True


def _result_for(source: str) -> PoseResult:
    frame = PoseFrame(0, None, (64, 48), (walker(0),))
    return PoseResult(topology=COCO_WHOLEBODY_133, frames=(frame,), source=source, model="m")


def test_each_frame_is_shown_with_its_own_pose(qtbot) -> None:  # type: ignore[no-untyped-def]
    from alchemypose.gui.live import LiveController

    submitted: list[tuple[np.ndarray, str]] = []
    live = LiveController(lambda bgr, source: submitted.append((bgr, source)), camera_factory=lambda _d: _Camera(5))
    live.start(0)
    qtbot.waitUntil(lambda: len(submitted) == 5, timeout=2000)

    shown: list[tuple[object, object]] = []
    live.frame_ready.connect(lambda bgr, view: shown.append((bgr, view)))
    bgr, source = submitted[3]
    live.on_result(_result_for(source))
    live.stop()

    assert len(shown) == 1
    frame_shown, view = shown[0]
    assert np.array_equal(frame_shown, bgr), "the pose is drawn on the frame it was computed from"
    assert isinstance(view, PoseResult)
    assert view.frames[0].people[0].track_id == 0


def test_a_result_for_a_superseded_frame_is_dropped(qtbot) -> None:  # type: ignore[no-untyped-def]
    from alchemypose.gui.live import LiveController

    submitted: list[str] = []
    live = LiveController(lambda _bgr, source: submitted.append(source), camera_factory=lambda _d: _Camera(4))
    live.start(0)
    qtbot.waitUntil(lambda: len(submitted) == 4, timeout=2000)
    shown: list[object] = []
    live.frame_ready.connect(lambda bgr, view: shown.append(view))
    live.on_result(_result_for(submitted[3]))          # the newest arrives first
    live.on_result(_result_for(submitted[1]))          # an older straggler: nothing to draw it on
    live.stop()
    assert len(shown) == 1


def test_a_camera_that_will_not_open_raises_and_is_released() -> None:
    from alchemypose.gui.live import LiveController

    camera = _Camera(opens=False)
    live = LiveController(lambda *_: None, camera_factory=lambda _d: camera)
    with pytest.raises(RuntimeError):
        live.start(0)
    assert camera.released
    assert not live.active


def test_live_owns_only_its_own_sources() -> None:
    from alchemypose.gui.live import LiveController

    live = LiveController(lambda *_: None)
    assert live.owns("webcam:3:0.1000")
    assert not live.owns("/home/me/photo.jpg")


def test_the_window_routes_webcam_results_to_the_live_view(qtbot, reporter) -> None:  # type: ignore[no-untyped-def]
    from alchemypose.gui import live as live_module
    from alchemypose.gui.main_window import MainWindow

    window = MainWindow(reporter=reporter, engine_factory=lambda: None)  # type: ignore[arg-type,return-value]
    qtbot.addWidget(window)
    window.live._camera_factory = lambda _d: _Camera(3)
    submitted: list[str] = []
    window.live._submit = lambda _bgr, source: submitted.append(source)

    window.webcam_action.setChecked(True)
    assert window.live.active
    qtbot.waitUntil(lambda: len(submitted) == 3, timeout=2000)
    window._on_result(replace(_result_for(submitted[2])))

    assert window.canvas.image_item is not None
    assert window.people.count == 1
    assert window.canvas_overlay.keypoint_item_count > 0
    assert "Webcam" in window.status
    window.webcam_action.setChecked(False)
    assert not window.live.active
    assert live_module.SOURCE_PREFIX == "webcam:"
