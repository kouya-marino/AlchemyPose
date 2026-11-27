"""Export: the three writers, and the menu the window puts on top of them.

No test here opens a save dialog. The writers take a path, and the window takes
a ``SavePrompt`` — the same injection ``Reporter`` gets, for the same reason.
"""

import json
from pathlib import Path

import numpy as np
import pytest
from PySide6.QtGui import QImage

from alchemypose.core.presets import selection_for
from alchemypose.core.schema import PersonPose, PoseFrame, PoseResult
from alchemypose.core.topology import COCO_WHOLEBODY_133
from alchemypose.gui.export import (
    ExportError,
    FixedSavePrompt,
    export_csv,
    export_json,
    export_png,
)

pytestmark = pytest.mark.gui

IMAGE_WIDTH, IMAGE_HEIGHT = 64, 48


def _result(source: str = "x") -> PoseResult:
    person = PersonPose(
        keypoints=np.full((133, 2), 5.0, dtype=np.float32),
        scores=np.full((133,), 0.7, dtype=np.float32),
    )
    frame = PoseFrame(
        frame_index=0, timestamp_s=None, image_size=(IMAGE_WIDTH, IMAGE_HEIGHT), people=(person,)
    )
    return PoseResult(topology=COCO_WHOLEBODY_133, frames=(frame,), source=source, model="m")


def _spread_result() -> PoseResult:
    """A person whose joints are scattered across the whole photograph.

    ``_result`` puts every keypoint on one coordinate, which is all the data
    formats need; a picture of it would be a single dot, so the PNG tests use
    something that actually covers the frame.
    """
    rng = np.random.default_rng(0)
    keypoints = rng.random((133, 2)) * np.array([IMAGE_WIDTH - 4, IMAGE_HEIGHT - 4]) + 2.0
    person = PersonPose(
        keypoints=keypoints.astype(np.float32),
        scores=np.full((133,), 0.9, dtype=np.float32),
    )
    frame = PoseFrame(
        frame_index=0, timestamp_s=None, image_size=(IMAGE_WIDTH, IMAGE_HEIGHT), people=(person,)
    )
    return PoseResult(topology=COCO_WHOLEBODY_133, frames=(frame,), source="x", model="m")


def _image() -> np.ndarray:  # type: ignore[type-arg]
    return np.full((IMAGE_HEIGHT, IMAGE_WIDTH, 3), 30, dtype=np.uint8)


def _canvas(qtbot, with_skeleton: bool = False):  # type: ignore[no-untyped-def]
    from alchemypose.gui.canvas.scene import PoseOverlay
    from alchemypose.gui.canvas.view import PoseCanvas

    canvas = PoseCanvas()
    qtbot.addWidget(canvas)
    canvas.set_image(_image())
    if with_skeleton:
        # Parented to the canvas, so it lives as long as the canvas does.
        overlay = PoseOverlay(canvas)
        overlay.set_result(_spread_result(), selection_for("all"), confidence_threshold=0.0)
    return canvas


class _StubEngine:
    """Answers whatever it is shown with one person, so nothing blocks on a model."""

    model_name = "fake"

    def infer(self, image: np.ndarray, source: str) -> PoseResult:  # type: ignore[type-arg]
        return _result(source)


def _window(qtbot, reporter, prompt):  # type: ignore[no-untyped-def]
    from alchemypose.gui.main_window import MainWindow

    window = MainWindow(reporter=reporter, engine_factory=_StubEngine, save_prompt=prompt)
    qtbot.addWidget(window)
    return window


def _loaded_window(qtbot, reporter, prompt):  # type: ignore[no-untyped-def]
    """A window with a photograph on the canvas and a result over it."""
    window = _window(qtbot, reporter, prompt)
    window.canvas.set_image(_image())
    window._on_result(_result())
    return window


# ------------------------------------------------------------- the writers


def test_json_carries_every_keypoint(tmp_path) -> None:  # type: ignore[no-untyped-def]
    path = tmp_path / "out.json"
    export_json(_result(), path)
    payload = json.loads(path.read_text())
    assert len(payload["frames"][0]["people"][0]["keypoints"]) == 133
    assert payload["topology"]["name"] == COCO_WHOLEBODY_133.name


def test_csv_has_a_header_and_a_row_per_keypoint(tmp_path) -> None:  # type: ignore[no-untyped-def]
    path = tmp_path / "out.csv"
    export_csv(_result(), path)
    lines = path.read_text().strip().splitlines()
    assert len(lines) == 1 + 133
    assert lines[0].startswith("frame,person,keypoint_index")


def test_png_writes_a_non_empty_image(qtbot, tmp_path) -> None:  # type: ignore[no-untyped-def]
    path = tmp_path / "out.png"
    export_png(_canvas(qtbot), path)
    assert path.stat().st_size > 0


def test_png_is_the_size_of_the_source_image_not_the_widget(qtbot, tmp_path) -> None:  # type: ignore[no-untyped-def]
    canvas = _canvas(qtbot)
    canvas.resize(500, 500)                 # nothing like the image's shape
    path = tmp_path / "out.png"
    export_png(canvas, path)

    written = QImage(str(path))
    assert (written.width(), written.height()) == (IMAGE_WIDTH, IMAGE_HEIGHT)


def test_png_is_the_same_file_whatever_the_zoom(qtbot, tmp_path) -> None:  # type: ignore[no-untyped-def]
    """The property the whole design of ``export_png`` exists for: it renders
    the scene, so what the user happens to be looking at cannot get into the
    file. Rendering the widget instead would fail exactly here."""
    bare = tmp_path / "bare.png"
    export_png(_canvas(qtbot), bare)

    canvas = _canvas(qtbot, with_skeleton=True)
    fitted = tmp_path / "fitted.png"
    canvas.fit_to_window()
    fitted_zoom = canvas.zoom_percent
    export_png(canvas, fitted)

    zoomed = tmp_path / "zoomed.png"
    canvas.zoom_by(4.0)
    zoomed_zoom = canvas.zoom_percent
    export_png(canvas, zoomed)

    assert fitted.read_bytes() != bare.read_bytes(), "the skeleton is not in the export"
    # The two zooms, against each other — not against a fixed 100%. ``zoom_by``
    # silently declines a step that would leave [MIN_ZOOM, MAX_ZOOM], and two
    # exports taken at the *same* transform would then agree for the one reason
    # this test is not about, proving nothing and staying green while doing it.
    assert zoomed_zoom != pytest.approx(fitted_zoom), "the two exports must differ in zoom"
    assert fitted.stat().st_size == zoomed.stat().st_size
    assert fitted.read_bytes() == zoomed.read_bytes()


# ---------------------------------------------------------------- the guards


def test_exporting_a_png_before_any_image_is_an_error_not_a_traceback(qtbot, tmp_path) -> None:  # type: ignore[no-untyped-def]
    from alchemypose.gui.canvas.view import PoseCanvas

    canvas = PoseCanvas()
    qtbot.addWidget(canvas)
    with pytest.raises(ExportError, match="nothing on the canvas"):
        export_png(canvas, tmp_path / "out.png")


@pytest.mark.parametrize("suffix", ["json", "csv"])
def test_writing_data_into_a_missing_folder_is_an_error_not_a_traceback(tmp_path, suffix) -> None:  # type: ignore[no-untyped-def]
    export = {"json": export_json, "csv": export_csv}[suffix]
    with pytest.raises(ExportError, match="Could not write"):
        export(_result(), tmp_path / "no-such-folder" / f"out.{suffix}")


def test_writing_a_png_into_a_missing_folder_is_an_error_not_a_traceback(qtbot, tmp_path) -> None:  # type: ignore[no-untyped-def]
    with pytest.raises(ExportError, match="Could not write"):
        export_png(_canvas(qtbot), tmp_path / "no-such-folder" / "out.png")


# ----------------------------------------------------------------- the menu


def test_the_menu_offers_every_format(qtbot, reporter) -> None:  # type: ignore[no-untyped-def]
    window = _window(qtbot, reporter, FixedSavePrompt())
    assert list(window.export_actions) == ["json", "csv", "png", "glb", "g1.npz", "g1.csv"]
    assert "JSON" in window.export_actions["json"].text()
    assert "glTF" in window.export_actions["glb"].text()


@pytest.mark.parametrize("suffix", ["json", "csv", "png"])
def test_choosing_a_format_writes_the_file_and_says_so(qtbot, reporter, tmp_path, suffix) -> None:  # type: ignore[no-untyped-def]
    target = tmp_path / f"chosen.{suffix}"
    window = _loaded_window(qtbot, reporter, FixedSavePrompt(target))

    window.export_actions[suffix].trigger()

    assert target.stat().st_size > 0
    assert window.status == f"Wrote chosen.{suffix}"
    assert reporter.errors == []


def test_cancelling_the_dialog_writes_nothing(qtbot, reporter, tmp_path) -> None:  # type: ignore[no-untyped-def]
    window = _loaded_window(qtbot, reporter, FixedSavePrompt(None))
    window.set_status("Ready.")

    window.export_actions["json"].trigger()

    assert list(tmp_path.iterdir()) == []
    assert window.status == "Ready.", "a cancelled export is not an event worth reporting"


def test_exporting_data_before_a_result_says_so_rather_than_asking_where(qtbot, reporter, tmp_path) -> None:  # type: ignore[no-untyped-def]
    prompt = FixedSavePrompt(tmp_path / "never.json")
    window = _window(qtbot, reporter, prompt)

    window.export_actions["json"].trigger()

    assert prompt.asked == [], "the user was asked where to put a file that does not exist"
    assert "Nothing to export" in window.status
    assert reporter.errors == []


def test_a_png_needs_only_a_photograph_not_a_result(qtbot, reporter, tmp_path) -> None:  # type: ignore[no-untyped-def]
    """A picture of the photograph is still a picture; the data formats have
    nothing to say until inference has returned."""
    target = tmp_path / "plain.png"
    prompt = FixedSavePrompt(target)
    window = _window(qtbot, reporter, prompt)
    window.canvas.set_image(_image())

    window.export_actions["png"].trigger()
    assert target.stat().st_size > 0

    window.export_actions["csv"].trigger()
    assert "Nothing to export" in window.status


def test_a_folder_that_does_not_exist_is_reported_not_raised(qtbot, reporter, tmp_path) -> None:  # type: ignore[no-untyped-def]
    missing = tmp_path / "no-such-folder" / "out.json"
    window = _loaded_window(qtbot, reporter, FixedSavePrompt(missing))

    window.export_actions["json"].trigger()      # the point is: no exception escapes

    assert not missing.exists()
    assert reporter.errors and reporter.errors[0][0] == "Export failed"
    assert window.status == "Export failed."


def test_the_suggested_name_follows_the_photograph(qtbot, reporter, tmp_path) -> None:  # type: ignore[no-untyped-def]
    import cv2

    photo = tmp_path / "portrait.png"
    cv2.imwrite(str(photo), _image())
    prompt = FixedSavePrompt(tmp_path / "out.png")
    window = _window(qtbot, reporter, prompt)
    window.open_image(photo)

    window.export_actions["png"].trigger()

    _caption, suggested, file_filter = prompt.asked[0]
    assert suggested == "portrait.png"
    assert "*.png" in file_filter


def test_the_suggested_name_falls_back_when_nothing_was_opened(qtbot, reporter, tmp_path) -> None:  # type: ignore[no-untyped-def]
    prompt = FixedSavePrompt(tmp_path / "out.csv")
    window = _loaded_window(qtbot, reporter, prompt)

    window.export_actions["csv"].trigger()

    assert prompt.asked[0][1] == "pose.csv"


def test_the_exported_json_reads_back_as_the_same_result(qtbot, reporter, tmp_path) -> None:  # type: ignore[no-untyped-def]
    """End to end through the menu, because a file nobody can load is not an export."""
    from alchemypose.core.serialize import from_json

    target = tmp_path / "round-trip.json"
    window = _loaded_window(qtbot, reporter, FixedSavePrompt(target))

    window.export_actions["json"].trigger()

    restored = from_json(target.read_text())
    original = window._result
    assert original is not None
    np.testing.assert_allclose(
        restored.frames[0].people[0].keypoints, original.frames[0].people[0].keypoints
    )


def test_the_path_is_taken_from_the_prompt_verbatim(qtbot, reporter, tmp_path) -> None:  # type: ignore[no-untyped-def]
    """No suffix is appended and none is corrected: what the user typed is where it goes."""
    target = tmp_path / "no-extension-at-all"
    window = _loaded_window(qtbot, reporter, FixedSavePrompt(target))

    window.export_actions["csv"].trigger()

    assert isinstance(target, Path) and target.read_text().startswith("frame,person")
