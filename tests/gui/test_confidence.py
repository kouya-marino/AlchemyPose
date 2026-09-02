import numpy as np
import pytest

from alchemypose.core.schema import PersonPose, PoseFrame, PoseResult
from alchemypose.core.topology import COCO_WHOLEBODY_133

pytestmark = pytest.mark.gui


class IdleEngine:
    model_name = "idle"

    def infer(self, image: np.ndarray, source: str) -> PoseResult:  # type: ignore[type-arg]
        raise AssertionError("these tests never run inference")


def _result(score: float) -> PoseResult:
    person = PersonPose(
        keypoints=np.full((133, 2), 100.0, dtype=np.float32),
        scores=np.full((133,), score, dtype=np.float32),
    )
    frame = PoseFrame(frame_index=0, timestamp_s=None, image_size=(640, 480), people=(person,))
    return PoseResult(topology=COCO_WHOLEBODY_133, frames=(frame,), source="x", model="m")


@pytest.fixture
def window(qtbot, reporter):  # type: ignore[no-untyped-def]
    from alchemypose.gui.main_window import MainWindow

    window = MainWindow(reporter=reporter, engine_factory=IdleEngine)
    qtbot.addWidget(window)
    return window


def test_the_default_threshold_is_moderate(window) -> None:  # type: ignore[no-untyped-def]
    assert window.confidence_threshold == 0.3
    assert window.confidence_slider.value() == 30


def test_raising_the_slider_hides_low_confidence_points(window) -> None:  # type: ignore[no-untyped-def]
    window._on_result(_result(score=0.4))
    assert window.canvas_overlay.keypoint_item_count > 0
    window.set_confidence_threshold(0.8)
    assert window.canvas_overlay.keypoint_item_count == 0


def test_lowering_the_slider_brings_them_back(window) -> None:  # type: ignore[no-untyped-def]
    window._on_result(_result(score=0.4))
    window.set_confidence_threshold(0.8)
    window.set_confidence_threshold(0.1)
    assert window.canvas_overlay.keypoint_item_count > 0


def test_the_slider_widget_drives_the_threshold_and_the_readout(window) -> None:  # type: ignore[no-untyped-def]
    window._on_result(_result(score=0.4))
    window.confidence_slider.setValue(75)
    assert window.confidence_threshold == 0.75
    assert window._confidence_label.text() == "0.75"
    assert window.canvas_overlay.keypoint_item_count == 0
    # And the other way round: setting the value moves the slider, once, without bouncing.
    window.set_confidence_threshold(0.2)
    assert window.confidence_slider.value() == 20
    assert window.confidence_threshold == 0.2


def test_the_threshold_never_mutates_the_stored_result(window) -> None:  # type: ignore[no-untyped-def]
    result = _result(score=0.4)
    before = result.frames[0].people[0].scores.copy()
    window._on_result(result)
    window.set_confidence_threshold(0.9)
    window.set_confidence_threshold(0.0)

    assert window._result is result
    assert len(result.frames[0].people) == 1
    np.testing.assert_array_equal(result.frames[0].people[0].scores, before)


def test_the_legend_names_the_persons_left(qtbot) -> None:  # type: ignore[no-untyped-def]
    from alchemypose.gui.panels.inspector import LegendWidget

    legend = LegendWidget()
    qtbot.addWidget(legend)
    assert "person's left" in legend.text().lower()
    assert "person's right" in legend.text().lower()


def test_inspector_shows_a_hovered_keypoint(qtbot) -> None:  # type: ignore[no-untyped-def]
    from alchemypose.gui.panels.inspector import Inspector

    inspector = Inspector()
    qtbot.addWidget(inspector)
    inspector.show_keypoint("left_wrist", 0.93, "left_arm")
    assert "left_wrist" in inspector.text()
    assert "0.93" in inspector.text()
    inspector.clear()
    assert inspector.text().strip() == ""


def test_hovering_a_keypoint_fills_the_inspector_from_the_result(window) -> None:  # type: ignore[no-untyped-def]
    window._on_result(_result(score=0.93))
    left_wrist = COCO_WHOLEBODY_133.keypoint_names.index("left_wrist")

    window.canvas_overlay.keypoint_hovered.emit(0, left_wrist, "left_arm")
    assert "left_wrist" in window.inspector.text()
    assert "0.93" in window.inspector.text()

    window.canvas_overlay.keypoint_left.emit()
    assert window.inspector.text() == ""


def test_the_window_shows_the_legend_and_inspector(window) -> None:  # type: ignore[no-untyped-def]
    assert "person's left" in window.legend.text().lower()
    assert window.inspector.text() == ""
