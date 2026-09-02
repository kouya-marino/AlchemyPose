import numpy as np

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
