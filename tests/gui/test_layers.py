import pytest
from PySide6.QtCore import Qt

from alchemypose.core.presets import selection_for
from alchemypose.core.schema import KeypointSelection
from alchemypose.core.topology import RENDER_GROUPS

pytestmark = pytest.mark.gui


@pytest.fixture
def panel(qtbot):  # type: ignore[no-untyped-def]
    from alchemypose.gui.panels.layers import LayerPanel

    widget = LayerPanel()
    qtbot.addWidget(widget)
    return widget


def test_every_render_group_gets_a_row(panel) -> None:  # type: ignore[no-untyped-def]
    assert set(panel.group_rows) == set(RENDER_GROUPS)


def test_rows_are_nested_under_four_parents(panel) -> None:  # type: ignore[no-untyped-def]
    assert [panel.tree.topLevelItem(i).text(0) for i in range(panel.tree.topLevelItemCount())] == [
        "Body", "Feet", "Face", "Hands"
    ]


def test_setting_a_selection_checks_the_matching_rows(panel) -> None:  # type: ignore[no-untyped-def]
    panel.set_selection(selection_for("hands"))
    assert panel.group_rows["left_hand"].checkState(0) == Qt.CheckState.Checked
    assert panel.group_rows["left_arm"].checkState(0) == Qt.CheckState.Unchecked


def test_current_selection_round_trips(panel) -> None:  # type: ignore[no-untyped-def]
    for name in ("hands", "face", "lower", "all", "none"):
        panel.set_selection(selection_for(name))
        assert panel.current_selection().visible == selection_for(name).visible


def test_a_parent_goes_partial_when_only_some_children_are_checked(panel) -> None:  # type: ignore[no-untyped-def]
    panel.set_selection(selection_for("none"))
    panel.set_group_checked("left_hand", True)
    hands_parent = panel.parent_rows["Hands"]
    assert hands_parent.checkState(0) == Qt.CheckState.PartiallyChecked


def test_checking_a_parent_checks_all_its_children(panel, qtbot) -> None:  # type: ignore[no-untyped-def]
    panel.set_selection(selection_for("none"))
    panel.parent_rows["Hands"].setCheckState(0, Qt.CheckState.Checked)
    assert panel.current_selection().visible == selection_for("hands").visible


def test_toggling_emits_selection_changed_once(panel, qtbot) -> None:  # type: ignore[no-untyped-def]
    panel.set_selection(selection_for("none"))
    with qtbot.waitSignal(panel.selection_changed, timeout=1000) as blocker:
        panel.set_group_checked("left_arm", True)
    assert blocker.args[0].visible == set(RENDER_GROUPS["left_arm"])


def test_programmatic_set_selection_emits_exactly_one_signal(panel, qtbot) -> None:  # type: ignore[no-untyped-def]
    received: list[object] = []
    panel.selection_changed.connect(received.append)
    panel.set_selection(selection_for("all"))
    assert len(received) == 1, "bulk updates must not emit per-row"


def test_overlapping_groups_do_not_cause_false_positive_checks(panel) -> None:  # type: ignore[no-untyped-def]
    """A group selected alone must not check a neighbour it merely shares an anchor with.

    left_arm=(5,7,9) shares index 5 with torso_head; left_leg=(11,13,15) shares
    11 with torso_head and 15 with left_foot. The subset test used by
    ``set_selection`` must reject these neighbours because they are missing
    indices unique to themselves (torso_head needs 0-4 too; left_foot needs
    17-19 too).
    """
    panel.set_selection(selection_for("hands"))  # unrelated selection first, to reset state
    panel.set_selection(KeypointSelection(visible=frozenset(RENDER_GROUPS["left_arm"])))
    assert panel.group_rows["left_arm"].checkState(0) == Qt.CheckState.Checked
    assert panel.group_rows["torso_head"].checkState(0) == Qt.CheckState.Unchecked
    assert panel.group_rows["right_arm"].checkState(0) == Qt.CheckState.Unchecked

    panel.set_selection(KeypointSelection(visible=frozenset(RENDER_GROUPS["left_leg"])))
    assert panel.group_rows["left_leg"].checkState(0) == Qt.CheckState.Checked
    assert panel.group_rows["torso_head"].checkState(0) == Qt.CheckState.Unchecked
    assert panel.group_rows["left_foot"].checkState(0) == Qt.CheckState.Unchecked
