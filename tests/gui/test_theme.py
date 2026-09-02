import pytest

pytestmark = pytest.mark.gui


def test_every_render_group_has_a_colour() -> None:
    from alchemypose.core.topology import RENDER_GROUPS
    from alchemypose.gui import theme

    for group in RENDER_GROUPS:
        assert group in theme.GROUP_COLORS, f"no colour for {group}"


def test_left_groups_are_green_and_right_groups_orange() -> None:
    from alchemypose.gui import theme

    for group in ("left_arm", "left_leg", "left_hand", "left_foot"):
        assert theme.GROUP_COLORS[group] == theme.LEFT
    for group in ("right_arm", "right_leg", "right_hand", "right_foot"):
        assert theme.GROUP_COLORS[group] == theme.RIGHT


def test_the_legend_says_persons_left_not_just_left() -> None:
    from alchemypose.gui import theme

    labels = [label for label, _ in theme.SIDE_LEGEND]
    assert any("person's left" in label.lower() for label in labels)


def test_color_for_returns_a_valid_qcolor() -> None:
    from alchemypose.gui import theme

    assert theme.color_for("left_arm").isValid()
    assert theme.color_for("no_such_group").isValid()   # falls back, never crashes
