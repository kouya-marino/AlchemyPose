import numpy as np
import pytest

from alchemypose.core.schema import Camera, KeypointSelection, PersonPose, PoseFrame, PoseResult, Space
from alchemypose.core.topology import COCO_WHOLEBODY_133


def _person(k: int = 133) -> PersonPose:
    return PersonPose(
        keypoints=np.zeros((k, 2), dtype=np.float32),
        scores=np.ones((k,), dtype=np.float32),
    )


def test_person_defaults_leave_the_future_slots_empty() -> None:
    p = _person()
    assert p.track_id is None
    assert p.keypoints_3d is None
    assert p.space_3d is None
    assert p.space_2d is Space.IMAGE_PX


def test_person_rejects_mismatched_shapes() -> None:
    with pytest.raises(ValueError, match="scores"):
        PersonPose(
            keypoints=np.zeros((133, 2), dtype=np.float32),
            scores=np.ones((17,), dtype=np.float32),
        )


def test_person_rejects_wrong_keypoint_dimensionality() -> None:
    with pytest.raises(ValueError, match="shape"):
        PersonPose(keypoints=np.zeros((133, 3), dtype=np.float32), scores=np.ones(133, dtype=np.float32))


def test_person_accepts_3d_and_validates_it() -> None:
    p = PersonPose(
        keypoints=np.zeros((133, 2), dtype=np.float32),
        scores=np.ones(133, dtype=np.float32),
        keypoints_3d=np.zeros((133, 3), dtype=np.float32),
        space_3d=Space.ROOT_REL,
    )
    assert p.keypoints_3d is not None and p.keypoints_3d.shape == (133, 3)


def test_default_camera_uses_max_dimension_as_focal_length() -> None:
    cam = Camera.default_for(1280, 850)
    assert cam.fx == cam.fy == 1280.0
    assert (cam.cx, cam.cy) == (640.0, 425.0)
    assert cam.source == "default"


def test_result_is_single_frame_for_an_image() -> None:
    frame = PoseFrame(frame_index=0, timestamp_s=None, image_size=(640, 480), people=(_person(),))
    result = PoseResult(
        topology=COCO_WHOLEBODY_133, frames=(frame,), source="a.jpg", model="rtmw-l_384x288"
    )
    assert result.schema_version == 1
    assert len(result.frames) == 1
    assert result.camera is None


def test_selection_defaults_to_all_persons() -> None:
    sel = KeypointSelection(visible=frozenset({1, 2}))
    assert sel.persons is None
    assert sel.is_visible(1) and not sel.is_visible(3)


def test_selection_edge_visibility_requires_both_endpoints() -> None:
    sel = KeypointSelection(visible=frozenset({5, 7}))
    assert sel.is_edge_visible((5, 7))
    assert not sel.is_edge_visible((5, 9))
