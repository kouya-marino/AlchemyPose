import pytest

pytestmark = pytest.mark.gui


def test_window_opens_with_a_title_and_ready_status(qtbot, reporter) -> None:  # type: ignore[no-untyped-def]
    from alchemypose.gui.main_window import MainWindow

    window = MainWindow(reporter=reporter)
    qtbot.addWidget(window)
    assert "AlchemyPose" in window.windowTitle()
    assert window.status == "Ready."


def test_window_has_the_canvas_and_both_panels(qtbot, reporter) -> None:  # type: ignore[no-untyped-def]
    from alchemypose.gui.main_window import MainWindow

    window = MainWindow(reporter=reporter)
    qtbot.addWidget(window)
    assert window.canvas is not None
    assert window.layers is not None
    assert window.presets is not None


def test_set_status_updates_the_status_bar(qtbot, reporter) -> None:  # type: ignore[no-untyped-def]
    from alchemypose.gui.main_window import MainWindow

    window = MainWindow(reporter=reporter)
    qtbot.addWidget(window)
    window.set_status("Working ...")
    assert window.status == "Working ..."
