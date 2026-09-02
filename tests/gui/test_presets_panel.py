import pytest

from alchemypose.core.presets import PRESETS, selection_for

pytestmark = pytest.mark.gui


@pytest.fixture
def bar(qtbot):  # type: ignore[no-untyped-def]
    from alchemypose.gui.panels.presets import PresetBar

    widget = PresetBar()
    qtbot.addWidget(widget)
    return widget


def test_there_is_a_chip_for_every_preset(bar) -> None:  # type: ignore[no-untyped-def]
    assert set(bar.chips) == set(PRESETS)


def test_clicking_a_chip_emits_its_name(bar, qtbot) -> None:  # type: ignore[no-untyped-def]
    with qtbot.waitSignal(bar.preset_chosen, timeout=1000) as blocker:
        bar.chips["hands"].click()
    assert blocker.args[0] == "hands"


def test_highlight_checks_exactly_one_chip(bar) -> None:  # type: ignore[no-untyped-def]
    bar.highlight("face")
    assert bar.chips["face"].isChecked()
    assert not bar.chips["hands"].isChecked()


def test_highlight_none_clears_every_chip(bar) -> None:  # type: ignore[no-untyped-def]
    bar.highlight("face")
    bar.highlight(None)
    assert not any(chip.isChecked() for chip in bar.chips.values())


def test_setting_checked_state_does_not_emit_clicked(bar, qtbot) -> None:  # type: ignore[no-untyped-def]
    """Guards the exact mechanism ``highlight()`` relies on to avoid a feedback loop.

    ``highlight()`` calls ``setChecked`` on every chip. If that ever emitted
    ``clicked`` (it must not, in Qt, unlike ``toggled``), ``PresetBar`` would
    re-emit ``preset_chosen`` for every ``highlight()`` call, and a window
    wiring ``preset_chosen`` back into the tree (as ``MainWindow`` does) would
    loop forever.
    """
    seen: list[str] = []
    bar.preset_chosen.connect(seen.append)
    bar.highlight("face")
    bar.highlight("hands")
    bar.highlight(None)
    assert seen == []


def test_window_applying_a_preset_updates_the_tree_and_lights_the_chip(qtbot, reporter) -> None:  # type: ignore[no-untyped-def]
    from alchemypose.gui.main_window import MainWindow

    window = MainWindow(reporter=reporter)
    qtbot.addWidget(window)
    window.presets.chips["hands"].click()
    assert window.layers.current_selection().visible == selection_for("hands").visible
    assert window.presets.chips["hands"].isChecked()


def test_hand_toggling_into_a_preset_lights_that_chip(qtbot, reporter) -> None:  # type: ignore[no-untyped-def]
    from alchemypose.gui.main_window import MainWindow

    window = MainWindow(reporter=reporter)
    qtbot.addWidget(window)
    window.layers.set_selection(selection_for("none"))
    window.layers.set_group_checked("left_hand", True)
    window.layers.set_group_checked("right_hand", True)
    assert window.presets.chips["hands"].isChecked()


def test_hand_toggling_off_a_preset_clears_every_chip(qtbot, reporter) -> None:  # type: ignore[no-untyped-def]
    from alchemypose.gui.main_window import MainWindow

    window = MainWindow(reporter=reporter)
    qtbot.addWidget(window)
    window.presets.chips["hands"].click()
    window.layers.set_group_checked("left_leg", True)
    assert not any(chip.isChecked() for chip in window.presets.chips.values())
