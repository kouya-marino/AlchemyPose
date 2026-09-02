import pytest

from alchemypose.core.topology import COCO_WHOLEBODY_133

T = COCO_WHOLEBODY_133


def test_has_133_named_keypoints() -> None:
    assert len(T.keypoint_names) == 133
    assert len(set(T.keypoint_names)) == 133, "keypoint names must be unique"


def test_index_ranges_match_the_spec() -> None:
    assert T.groups["body"] == tuple(range(0, 17))
    assert T.groups["feet"] == tuple(range(17, 23))
    assert T.groups["face"] == tuple(range(23, 91))
    assert T.groups["left_hand"] == tuple(range(91, 112))
    assert T.groups["right_hand"] == tuple(range(112, 133))


def test_render_subgroups_partition_their_parent() -> None:
    body_parts = ("torso_head", "left_arm", "right_arm", "left_leg", "right_leg")
    covered = {i for name in body_parts for i in T.groups[name]}
    assert covered == set(T.groups["body"])


def test_face_subgroups_follow_dlib_order_offset_by_23() -> None:
    assert T.groups["face_jaw"] == tuple(range(23, 40))       # dlib 0-16
    assert T.groups["face_brows"] == tuple(range(40, 50))     # dlib 17-26
    assert T.groups["face_nose"] == tuple(range(50, 59))      # dlib 27-35
    assert T.groups["face_eyes"] == tuple(range(59, 71))      # dlib 36-47
    assert T.groups["face_lips"] == tuple(range(71, 91))      # dlib 48-67


def test_every_edge_references_a_real_keypoint() -> None:
    for a, b in T.edges:
        assert 0 <= a < 133 and 0 <= b < 133
        assert a != b


def test_no_duplicate_edges() -> None:
    normalised = {tuple(sorted(e)) for e in T.edges}
    assert len(normalised) == len(T.edges)


def test_group_of_returns_the_narrowest_group() -> None:
    assert T.group_of(0) == "torso_head"
    assert T.group_of(91) == "left_hand"
    assert T.group_of(133) is None


@pytest.mark.parametrize("index", range(133))
def test_every_keypoint_belongs_to_some_render_group(index: int) -> None:
    assert T.group_of(index) is not None
