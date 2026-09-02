import pytest

from alchemypose.core.presets import PRESETS, matching_preset, selection_for, selection_from_groups
from alchemypose.core.topology import COCO_WHOLEBODY_133


def test_all_expected_presets_exist() -> None:
    assert set(PRESETS) == {
        "full_body", "upper", "lower", "left_side", "right_side", "hands", "face", "all", "none"
    }


def test_all_shows_every_keypoint() -> None:
    assert selection_for("all").visible == frozenset(range(133))


def test_none_shows_nothing() -> None:
    assert selection_for("none").visible == frozenset()


def test_hands_shows_only_hand_indices() -> None:
    assert selection_for("hands").visible == frozenset(range(91, 133))


def test_face_shows_only_face_indices() -> None:
    assert selection_for("face").visible == frozenset(range(23, 91))


def test_lower_excludes_arms() -> None:
    lower = selection_for("lower").visible
    assert 15 in lower                      # left ankle
    assert 9 not in lower                   # left wrist


def test_unknown_preset_raises() -> None:
    with pytest.raises(KeyError):
        selection_for("elbows_only")


def test_matching_preset_round_trips_every_preset() -> None:
    for name in PRESETS:
        assert matching_preset(selection_for(name)) == name


def test_matching_preset_returns_none_for_a_hand_made_selection() -> None:
    custom = selection_from_groups(["right_hand", "left_leg"])
    assert matching_preset(custom) is None


def test_selection_from_groups_unions_the_indices() -> None:
    sel = selection_from_groups(["left_hand", "right_leg"])
    expected = set(COCO_WHOLEBODY_133.groups["left_hand"]) | set(COCO_WHOLEBODY_133.groups["right_leg"])
    assert sel.visible == frozenset(expected)
