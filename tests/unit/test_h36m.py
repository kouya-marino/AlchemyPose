import numpy as np
import pytest

from alchemypose.core.h36m import H36M_NAMES, to_h36m17
from alchemypose.core.schema import PersonPose


def _person_with(**positions: tuple[float, float]) -> PersonPose:
    """A person whose named COCO keypoints sit at given coordinates, rest at origin."""
    from alchemypose.core.topology import KEYPOINT_NAMES

    kp = np.zeros((133, 2), dtype=np.float32)
    scores = np.ones((133,), dtype=np.float32)
    for name, xy in positions.items():
        kp[KEYPOINT_NAMES.index(name)] = xy
    return PersonPose(keypoints=kp, scores=scores)


def test_returns_seventeen_joints() -> None:
    kp, scores = to_h36m17(_person_with())
    assert kp.shape == (17, 2)
    assert scores.shape == (17,)
    assert len(H36M_NAMES) == 17


def test_pelvis_is_the_midpoint_of_the_hips() -> None:
    person = _person_with(left_hip=(100.0, 200.0), right_hip=(60.0, 220.0))
    kp, _ = to_h36m17(person)
    np.testing.assert_allclose(kp[H36M_NAMES.index("pelvis")], [80.0, 210.0])


def test_thorax_is_the_midpoint_of_the_shoulders() -> None:
    person = _person_with(left_shoulder=(120.0, 100.0), right_shoulder=(80.0, 110.0))
    kp, _ = to_h36m17(person)
    np.testing.assert_allclose(kp[H36M_NAMES.index("thorax")], [100.0, 105.0])


def test_spine_is_midway_between_pelvis_and_thorax() -> None:
    person = _person_with(
        left_hip=(100.0, 200.0), right_hip=(100.0, 200.0),
        left_shoulder=(100.0, 100.0), right_shoulder=(100.0, 100.0),
    )
    kp, _ = to_h36m17(person)
    np.testing.assert_allclose(kp[H36M_NAMES.index("spine")], [100.0, 150.0])


def test_synthesised_joint_confidence_is_the_minimum_of_its_parents() -> None:
    from alchemypose.core.topology import KEYPOINT_NAMES

    person = _person_with(left_hip=(0.0, 0.0), right_hip=(0.0, 0.0))
    scores = person.scores.copy()
    scores[KEYPOINT_NAMES.index("left_hip")] = 0.3
    scores[KEYPOINT_NAMES.index("right_hip")] = 0.9
    person = PersonPose(keypoints=person.keypoints, scores=scores)
    _, out_scores = to_h36m17(person)
    assert out_scores[H36M_NAMES.index("pelvis")] == np.float32(0.3)


def test_direct_joints_are_copied_not_recomputed() -> None:
    person = _person_with(left_ankle=(11.0, 22.0))
    kp, _ = to_h36m17(person)
    np.testing.assert_allclose(kp[H36M_NAMES.index("left_ankle")], [11.0, 22.0])


def test_h36m_joint_order_matches_the_canonical_skeleton() -> None:
    """A lifting model indexes this positionally. A reorder produces plausible-looking
    garbage rather than an error, so the sequence is pinned here explicitly."""
    assert H36M_NAMES == (
        "pelvis", "right_hip", "right_knee", "right_ankle",
        "left_hip", "left_knee", "left_ankle",
        "spine", "thorax", "neck", "head",
        "left_shoulder", "left_elbow", "left_wrist",
        "right_shoulder", "right_elbow", "right_wrist",
    )


def test_neck_sits_near_the_thorax_not_halfway_to_the_head() -> None:
    person = _person_with(
        left_shoulder=(100.0, 100.0), right_shoulder=(100.0, 100.0), nose=(100.0, 0.0)
    )
    kp, _ = to_h36m17(person)
    neck_y = kp[H36M_NAMES.index("neck")][1]
    thorax_y = kp[H36M_NAMES.index("thorax")][1]
    assert neck_y == pytest.approx(75.0), "quarter of the way from thorax (100) to nose (0)"
    assert abs(neck_y - thorax_y) < abs(neck_y - 0.0), "neck must be nearer the thorax than the head"


def test_derived_confidence_propagates_transitively() -> None:
    """spine and neck derive from derived joints; the minimum must chain through."""
    from alchemypose.core.topology import KEYPOINT_NAMES

    person = _person_with(
        left_hip=(0.0, 0.0), right_hip=(0.0, 0.0),
        left_shoulder=(0.0, 0.0), right_shoulder=(0.0, 0.0), nose=(0.0, 0.0),
    )
    scores = person.scores.copy()
    scores[KEYPOINT_NAMES.index("left_hip")] = 0.2      # weakest ancestor of spine
    scores[KEYPOINT_NAMES.index("right_shoulder")] = 0.6
    scores[KEYPOINT_NAMES.index("nose")] = 0.4
    person = PersonPose(keypoints=person.keypoints, scores=scores)
    _, out = to_h36m17(person)
    assert out[H36M_NAMES.index("spine")] == pytest.approx(0.2), "min over all four ancestors"
    assert out[H36M_NAMES.index("neck")] == pytest.approx(0.4), "min(thorax=0.6, nose=0.4)"
