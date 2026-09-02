import numpy as np
import pytest

from alchemypose.core.presets import selection_for
from alchemypose.core.schema import KeypointSelection, PersonPose, PoseFrame, PoseResult
from alchemypose.core.topology import COCO_WHOLEBODY_133

pytestmark = pytest.mark.gui

rng = np.random.default_rng(1)


def _result(n: int = 2, score: float = 0.9) -> PoseResult:
    people = tuple(
        PersonPose(
            keypoints=rng.random((133, 2), dtype=np.float32) * 400 + 20,
            scores=np.full((133,), score, dtype=np.float32),
            bbox=(0.0, 0.0, 440.0, 440.0),
        )
        for _ in range(n)
    )
    frame = PoseFrame(frame_index=0, timestamp_s=None, image_size=(640, 480), people=people)
    return PoseResult(topology=COCO_WHOLEBODY_133, frames=(frame,), source="x", model="m")


@pytest.fixture
def overlay(qtbot):  # type: ignore[no-untyped-def]
    from alchemypose.gui.canvas.scene import PoseOverlay
    from alchemypose.gui.canvas.view import PoseCanvas

    canvas = PoseCanvas()
    qtbot.addWidget(canvas)
    canvas.set_image(np.zeros((480, 640, 3), dtype=np.uint8))
    return PoseOverlay(canvas)


def test_all_preset_draws_every_keypoint_for_every_person(overlay) -> None:  # type: ignore[no-untyped-def]
    overlay.set_result(_result(n=2), selection_for("all"), confidence_threshold=0.0)
    assert overlay.keypoint_item_count == 2 * 133


def test_hands_preset_draws_only_hand_keypoints(overlay) -> None:  # type: ignore[no-untyped-def]
    overlay.set_result(_result(n=1), selection_for("hands"), confidence_threshold=0.0)
    assert overlay.keypoint_item_count == 42


def test_changing_selection_does_not_rebuild_the_result(overlay) -> None:  # type: ignore[no-untyped-def]
    overlay.set_result(_result(n=1), selection_for("all"), confidence_threshold=0.0)
    overlay.set_selection(selection_for("hands"))
    assert overlay.keypoint_item_count == 42
    overlay.set_selection(selection_for("all"))
    assert overlay.keypoint_item_count == 133


def test_a_bone_needs_both_endpoints_visible(overlay) -> None:  # type: ignore[no-untyped-def]
    # left shoulder (5) and left elbow (7) — the bone between them exists in the topology
    overlay.set_result(_result(n=1), KeypointSelection(frozenset({5, 7})), confidence_threshold=0.0)
    assert overlay.bone_item_count == 1
    overlay.set_selection(KeypointSelection(frozenset({5})))
    assert overlay.bone_item_count == 0


def test_low_confidence_keypoints_are_dropped_by_the_threshold(overlay) -> None:  # type: ignore[no-untyped-def]
    overlay.set_result(_result(n=1, score=0.2), selection_for("all"), confidence_threshold=0.5)
    assert overlay.keypoint_item_count == 0


def test_raising_the_threshold_removes_items_without_touching_the_data(overlay) -> None:  # type: ignore[no-untyped-def]
    result = _result(n=1, score=0.4)
    overlay.set_result(result, selection_for("all"), confidence_threshold=0.0)
    overlay.set_confidence_threshold(0.9)
    assert overlay.keypoint_item_count == 0
    assert len(result.frames[0].people[0]) == 133, "data must be untouched"


def test_items_know_which_person_and_keypoint_they_are(overlay) -> None:  # type: ignore[no-untyped-def]
    overlay.set_result(_result(n=2), selection_for("hands"), confidence_threshold=0.0)
    pairs = {(i.person_index, i.keypoint_index) for i in overlay.keypoint_items}
    assert (0, 91) in pairs
    assert (1, 132) in pairs


def test_group_bounds_covers_only_that_group(overlay) -> None:  # type: ignore[no-untyped-def]
    result = _result(n=1)
    overlay.set_result(result, selection_for("all"), confidence_threshold=0.0)
    rect = overlay.group_bounds(0, "left_hand")
    kp = result.frames[0].people[0].keypoints[91:112]
    assert rect.left() <= float(kp[:, 0].min()) + 1
    assert rect.right() >= float(kp[:, 0].max()) - 1
