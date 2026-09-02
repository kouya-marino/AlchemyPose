import threading
from pathlib import Path

import numpy as np
import pytest

from alchemypose.core.schema import PersonPose, PoseFrame, PoseResult
from alchemypose.core.topology import COCO_WHOLEBODY_133

pytestmark = pytest.mark.gui


def _person(score: float = 0.9) -> PersonPose:
    return PersonPose(
        keypoints=np.full((133, 2), 60.0, dtype=np.float32),
        scores=np.full((133,), score, dtype=np.float32),
    )


def _result(source: str, people: int = 1) -> PoseResult:
    frame = PoseFrame(
        frame_index=0, timestamp_s=None, image_size=(160, 120), people=tuple(_person() for _ in range(people))
    )
    return PoseResult(topology=COCO_WHOLEBODY_133, frames=(frame,), source=source, model="fake")


class OnePersonEngine:
    """Returns one confident person for whatever it is shown."""

    model_name = "fake"

    def infer(self, image: np.ndarray, source: str) -> PoseResult:  # type: ignore[type-arg]
        return _result(source)


def _write_png(path: Path, width: int = 160, height: int = 120) -> Path:
    import cv2

    cv2.imwrite(str(path), np.zeros((height, width, 3), dtype=np.uint8))
    return path


def test_opening_an_image_shows_it_and_reports_progress(qtbot, reporter, tmp_path) -> None:  # type: ignore[no-untyped-def]
    from alchemypose.gui.main_window import MainWindow

    path = _write_png(tmp_path / "x.png")
    window = MainWindow(reporter=reporter, engine_factory=OnePersonEngine)
    qtbot.addWidget(window)
    window.open_image(path)

    assert window.canvas.scene().sceneRect().width() == 160
    assert "x.png" in window.status and "detecting" in window.status.lower()


def test_a_result_from_the_worker_is_drawn_on_the_gui_thread(qtbot, reporter, tmp_path) -> None:  # type: ignore[no-untyped-def]
    from alchemypose.gui.main_window import MainWindow

    path = _write_png(tmp_path / "x.png")
    window = MainWindow(reporter=reporter, engine_factory=OnePersonEngine)
    qtbot.addWidget(window)
    window.open_image(path)

    qtbot.waitUntil(lambda: window.canvas_overlay.keypoint_item_count > 0, timeout=5000)
    assert window._result is not None and window._result.source == str(path)
    assert len(window.people.chips) == 1
    assert "1 person" in window.status
    assert reporter.errors == []


def test_opening_an_unreadable_file_reports_without_crashing(qtbot, reporter, tmp_path) -> None:  # type: ignore[no-untyped-def]
    from alchemypose.gui.main_window import MainWindow

    bad = tmp_path / "not-an-image.png"
    bad.write_bytes(b"nonsense")

    window = MainWindow(reporter=reporter, engine_factory=OnePersonEngine)
    qtbot.addWidget(window)
    window.open_image(bad)   # the point is: no exception escapes

    assert reporter.errors and reporter.errors[0][0] == "Could not open image"
    assert "could not" in window.status.lower()


def test_an_engine_failure_is_reported_not_raised(qtbot, reporter, tmp_path) -> None:  # type: ignore[no-untyped-def]
    from alchemypose.gui.main_window import MainWindow

    class BoomEngine:
        model_name = "boom"

        def infer(self, image: np.ndarray, source: str) -> PoseResult:  # type: ignore[type-arg]
            raise RuntimeError("no weights")

    window = MainWindow(reporter=reporter, engine_factory=BoomEngine)
    qtbot.addWidget(window)
    window.open_image(_write_png(tmp_path / "x.png"))

    qtbot.waitUntil(lambda: bool(reporter.errors), timeout=5000)
    assert "no weights" in reporter.errors[0][1]
    assert window.status == "Detection failed."


def test_opening_a_second_image_then_changing_selection_does_not_raise(qtbot, reporter, tmp_path) -> None:  # type: ignore[no-untyped-def]
    """``PoseCanvas.set_image`` wipes the scene under the overlay's feet.

    Without the ``scene_cleared`` seam the overlay would call ``removeItem`` on
    C++ objects the scene already deleted: ``RuntimeError: Internal C++ object
    already deleted``. This is the regression test for that.
    """
    from alchemypose.core.presets import selection_for
    from alchemypose.gui.main_window import MainWindow

    first = _write_png(tmp_path / "a.png")
    second = _write_png(tmp_path / "b.png", width=200, height=100)

    window = MainWindow(reporter=reporter, engine_factory=OnePersonEngine)
    qtbot.addWidget(window)

    window.open_image(first)
    window._on_result(_result(str(first)))
    assert window.canvas_overlay.keypoint_item_count > 0

    window.open_image(second)
    window.layers.set_selection(selection_for("upper"))   # would raise before the fix
    window._apply_preset("face")
    window.set_confidence_threshold(0.5)
    assert window.canvas.scene().sceneRect().width() == 200


def test_a_selection_change_before_the_new_result_arrives_draws_nothing(qtbot, reporter, tmp_path) -> None:  # type: ignore[no-untyped-def]
    """Photo A's skeleton must never float over photo B while B is still being detected."""
    from alchemypose.core.presets import selection_for
    from alchemypose.gui.main_window import MainWindow

    first = _write_png(tmp_path / "a.png")
    second = _write_png(tmp_path / "b.png")
    window = MainWindow(reporter=reporter, engine_factory=OnePersonEngine)
    qtbot.addWidget(window)

    window.open_image(first)
    window._on_result(_result(str(first)))
    window.open_image(second)
    assert window.canvas_overlay.keypoint_item_count == 0

    window.layers.set_selection(selection_for("upper"))
    assert window.canvas_overlay.keypoint_item_count == 0, "photo A's skeleton must not float over photo B"


def test_a_stale_result_for_a_replaced_image_is_ignored(qtbot, reporter, tmp_path) -> None:  # type: ignore[no-untyped-def]
    from alchemypose.gui.main_window import MainWindow

    first = _write_png(tmp_path / "a.png")
    second = _write_png(tmp_path / "b.png")
    window = MainWindow(reporter=reporter, engine_factory=OnePersonEngine)
    qtbot.addWidget(window)

    window.open_image(first)
    window.open_image(second)
    window._on_result(_result(str(first), people=3))   # A's answer arriving after B was opened

    assert window._result is None
    assert window.people.chips == []
    assert window.canvas_overlay.keypoint_item_count == 0


def test_constructing_the_window_does_not_build_the_real_engine(qtbot, reporter, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """The default factory must only run on the worker, at first use — never in ``__init__``."""
    from alchemypose.gui.main_window import MainWindow
    from alchemypose.inference import rtmw

    def boom(self: object, preset: str = "performance") -> None:
        raise AssertionError("RTMWEngine was constructed eagerly")

    monkeypatch.setattr(rtmw.RTMWEngine, "__init__", boom)
    window = MainWindow(reporter=reporter)
    qtbot.addWidget(window)
    assert window.status == "Ready."


def test_closing_the_window_stops_the_worker(qtbot, reporter) -> None:  # type: ignore[no-untyped-def]
    from alchemypose.gui.main_window import MainWindow

    window = MainWindow(reporter=reporter, engine_factory=OnePersonEngine)
    qtbot.addWidget(window)
    assert window._bridge.running is True
    assert any(t.name == "alchemypose-inference" for t in threading.enumerate())

    window.close()

    assert window._bridge.running is False
    assert not any(t.name == "alchemypose-inference" for t in threading.enumerate())


def test_the_toolbar_has_an_open_action_with_the_standard_shortcut(qtbot, reporter) -> None:  # type: ignore[no-untyped-def]
    from PySide6.QtGui import QKeySequence

    from alchemypose.gui.main_window import MainWindow

    window = MainWindow(reporter=reporter, engine_factory=OnePersonEngine)
    qtbot.addWidget(window)
    assert window.open_action.text() == "Open"
    assert window.open_action.shortcut() == QKeySequence(QKeySequence.StandardKey.Open)


def test_person_bar_creates_one_chip_per_person(qtbot) -> None:  # type: ignore[no-untyped-def]
    from alchemypose.gui.panels.people import PersonBar

    bar = PersonBar()
    qtbot.addWidget(bar)
    bar.set_people(3)
    assert len(bar.chips) == 3
    assert bar.current is None
    bar.set_people(1)
    assert len(bar.chips) == 1


def test_choosing_a_person_emits_its_index(qtbot) -> None:  # type: ignore[no-untyped-def]
    from alchemypose.gui.panels.people import PersonBar

    bar = PersonBar()
    qtbot.addWidget(bar)
    bar.set_people(2)
    with qtbot.waitSignal(bar.person_chosen, timeout=1000) as blocker:
        bar.chips[1].click()
    assert blocker.args[0] == 1
    assert bar.current == 1
    assert bar.chips[0].isChecked() is False


def test_clicking_the_chosen_chip_again_means_everyone(qtbot) -> None:  # type: ignore[no-untyped-def]
    from alchemypose.gui.panels.people import PersonBar

    bar = PersonBar()
    qtbot.addWidget(bar)
    bar.set_people(2)
    bar.chips[1].click()
    with qtbot.waitSignal(bar.everyone_chosen, timeout=1000):
        bar.chips[1].click()
    assert bar.current is None


def test_choosing_a_person_narrows_the_overlay_to_them(qtbot, reporter) -> None:  # type: ignore[no-untyped-def]
    from alchemypose.core.presets import selection_for
    from alchemypose.gui.main_window import MainWindow

    window = MainWindow(reporter=reporter, engine_factory=OnePersonEngine)
    qtbot.addWidget(window)
    window._apply_preset("all")
    window._on_result(_result("x", people=2))
    everyone = window.canvas_overlay.keypoint_item_count
    assert everyone == 2 * 133

    window.people.chips[1].click()
    assert window.canvas_overlay.keypoint_item_count == 133
    # A tree change must not silently forget the chosen person.
    window.layers.set_selection(selection_for("upper"))
    assert 0 < window.canvas_overlay.keypoint_item_count < 133

    window.people.chips[1].click()   # back to everyone
    window._apply_preset("all")
    assert window.canvas_overlay.keypoint_item_count == everyone
