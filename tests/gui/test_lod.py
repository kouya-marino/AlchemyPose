"""Face level of detail: the thresholds, what the overlay does with them, and
the canvas signal that tells it the zoom moved.

Wherever a size or a zoom is chosen to put the face in a particular band, it is
derived from ``REGIONS_ABOVE`` and ``POINTS_ABOVE`` rather than written out as a
copy of whatever they currently are, so retuning one number cannot silently
invalidate a test that is not about the retune. Three plain numbers survive
below, and none of them is a threshold in disguise: a zero extent, a zoom
applied before any result exists, and the face the Face chip is pointed at —
that one reframes the head to fill the viewport, so what it ends up measuring is
the same whatever size the face was in the photograph.

The overlay tests fix the face's size in *scene* pixels and move the zoom, which
is the same arithmetic the real thing does.
"""

import numpy as np
import pytest
from PySide6.QtCore import QRectF

from alchemypose.core.presets import selection_for, selection_from_groups
from alchemypose.core.schema import PersonPose, PoseFrame, PoseResult
from alchemypose.core.topology import COCO_WHOLEBODY_133, RENDER_GROUPS
from alchemypose.gui import theme
from alchemypose.gui.canvas.lod import (
    POINTS_ABOVE,
    REGION_COLORS,
    REGIONS_ABOVE,
    FaceDetail,
    face_detail,
    is_face_group,
)

pytestmark = pytest.mark.gui

FACE_POINTS = 68
"""Face keypoints in COCO-WholeBody: indices 23-90, in dlib-68 order."""

FACE_ORIGIN = 50.0
"""Where the synthetic face's top-left corner sits, in scene pixels."""

CONTOURS_EXTENT = REGIONS_ABOVE / 2.0
"""A face this many scene pixels across is below ``REGIONS_ABOVE`` at zoom 1.0."""

REGIONS_EXTENT = (REGIONS_ABOVE + POINTS_ABOVE) / 2.0
"""A face this many scene pixels across sits inside the middle band at zoom 1.0."""

POINTS_EXTENT = POINTS_ABOVE * 2.0
"""A face this many scene pixels across is above ``POINTS_ABOVE`` at zoom 1.0."""


def _zoom_into_points(extent: float) -> float:
    """A zoom at which ``extent`` scene pixels clears ``POINTS_ABOVE`` on screen."""
    return 2.0 * POINTS_ABOVE / extent


def _zoom_into_contours(extent: float) -> float:
    """A zoom at which ``extent`` scene pixels falls below ``REGIONS_ABOVE`` on screen."""
    return REGIONS_ABOVE / (2.0 * extent)


def _face_result(extent: float) -> PoseResult:
    """One person whose face occupies exactly ``extent`` scene pixels square.

    Laid out on a diagonal rather than at random: the tests below sit on either
    side of a threshold, so the face's measured size has to be the number asked
    for and not a sample that happened to fall short of it.
    """
    keypoints = np.zeros((133, 2), dtype=np.float32)
    span = np.linspace(FACE_ORIGIN, FACE_ORIGIN + extent, FACE_POINTS, dtype=np.float32)
    keypoints[23:91, 0] = span
    keypoints[23:91, 1] = span
    person = PersonPose(keypoints=keypoints, scores=np.full((133,), 0.9, dtype=np.float32))
    frame = PoseFrame(frame_index=0, timestamp_s=None, image_size=(640, 640), people=(person,))
    return PoseResult(topology=COCO_WHOLEBODY_133, frames=(frame,), source="x", model="m")


def _overlay(qtbot, extent: float, groups: str = "face"):  # type: ignore[no-untyped-def]
    """An overlay showing a face ``extent`` scene pixels across, at zoom 1.0."""
    from alchemypose.gui.canvas.scene import PoseOverlay
    from alchemypose.gui.canvas.view import PoseCanvas

    canvas = PoseCanvas()
    qtbot.addWidget(canvas)
    canvas.set_image(np.zeros((640, 640, 3), dtype=np.uint8))
    overlay = PoseOverlay(canvas)
    overlay.set_result(_face_result(extent), selection_for(groups), confidence_threshold=0.0)
    return canvas, overlay


def _stop_colors(bone) -> set[str]:  # type: ignore[no-untyped-def]
    """The colours a bone's gradient pen is painted with, as hex names."""
    gradient = bone.pen().brush().gradient()
    return {color.name() for _position, color in gradient.stops()}


class _IdleEngine:
    """Never asked for anything: these tests feed results in by hand."""

    model_name = "idle"

    def infer(self, image: np.ndarray, source: str) -> PoseResult:  # type: ignore[type-arg]
        raise AssertionError("these tests never run inference")


# ------------------------------------------------------------- the thresholds


@pytest.mark.parametrize(
    ("size", "expected"),
    [
        (1.0, FaceDetail.CONTOURS),
        (REGIONS_ABOVE - 1.0, FaceDetail.CONTOURS),
        (REGIONS_ABOVE, FaceDetail.REGIONS),
        (POINTS_ABOVE - 1.0, FaceDetail.REGIONS),
        (POINTS_ABOVE, FaceDetail.POINTS),
        (POINTS_ABOVE * 10.0, FaceDetail.POINTS),
    ],
)
def test_each_band_gets_its_detail_level(size: float, expected: FaceDetail) -> None:
    assert face_detail(size) == expected


def test_the_bands_meet_at_their_thresholds() -> None:
    """No gap and no overlap: the constants are the only boundaries."""
    assert REGIONS_ABOVE < POINTS_ABOVE
    assert face_detail(REGIONS_ABOVE - 1e-9) is FaceDetail.CONTOURS
    assert face_detail(POINTS_ABOVE - 1e-9) is FaceDetail.REGIONS


def test_zero_or_negative_size_is_treated_as_the_smallest() -> None:
    assert face_detail(0.0) is FaceDetail.CONTOURS
    assert face_detail(-5.0) is FaceDetail.CONTOURS


def test_only_the_largest_level_is_worth_drawing_points_for() -> None:
    assert [d for d in FaceDetail if d.draws_points] == [FaceDetail.POINTS]


def test_colour_by_region_starts_one_level_below_the_points() -> None:
    assert not FaceDetail.CONTOURS.colours_by_region
    assert FaceDetail.REGIONS.colours_by_region
    assert FaceDetail.POINTS.colours_by_region


def test_the_face_is_recognised_by_group_name_not_by_index() -> None:
    assert is_face_group("face_lips")
    assert not is_face_group("left_hand")
    assert not is_face_group("torso_head")


def test_every_face_render_group_has_a_region_colour_and_nothing_else_does() -> None:
    """A new face group added to the topology must not fall back to grey unnoticed."""
    assert set(REGION_COLORS) == {name for name in RENDER_GROUPS if is_face_group(name)}


# --------------------------------------------------------- what gets built


def test_points_are_only_created_once_the_face_is_large(qtbot) -> None:  # type: ignore[no-untyped-def]
    """Under ``REGIONS_ABOVE`` on screen it is contours; zoomed past
    ``POINTS_ABOVE`` the same face earns its individual points."""
    _, overlay = _overlay(qtbot, extent=CONTOURS_EXTENT)
    assert overlay.face_detail is FaceDetail.CONTOURS
    assert overlay.keypoint_item_count == 0

    overlay.on_zoom_changed(_zoom_into_points(CONTOURS_EXTENT))

    assert overlay.face_detail is FaceDetail.POINTS
    assert overlay.keypoint_item_count == FACE_POINTS


def test_the_contours_are_drawn_at_every_level(qtbot) -> None:  # type: ignore[no-untyped-def]
    """The points are what the lower levels spend; the outlines are not."""
    _, overlay = _overlay(qtbot, extent=CONTOURS_EXTENT)
    contour_bones = overlay.bone_item_count
    assert contour_bones > 0

    overlay.on_zoom_changed(_zoom_into_points(CONTOURS_EXTENT))

    assert overlay.bone_item_count == contour_bones


def test_the_middle_band_colours_the_contours_by_region(qtbot) -> None:  # type: ignore[no-untyped-def]
    """Only the lips are selected, so every bone in the scene is a lip contour."""
    from alchemypose.gui.canvas.scene import PoseOverlay
    from alchemypose.gui.canvas.view import PoseCanvas

    canvas = PoseCanvas()
    qtbot.addWidget(canvas)
    canvas.set_image(np.zeros((640, 640, 3), dtype=np.uint8))
    overlay = PoseOverlay(canvas)
    overlay.set_result(
        _face_result(extent=REGIONS_EXTENT),
        selection_from_groups(["face_lips"]),
        confidence_threshold=0.0,
    )
    assert overlay.face_detail is FaceDetail.REGIONS
    assert _stop_colors(overlay.bone_items[0]) == {REGION_COLORS["face_lips"]}

    overlay.on_zoom_changed(_zoom_into_contours(REGIONS_EXTENT))   # back under it

    assert overlay.face_detail is FaceDetail.CONTOURS
    assert _stop_colors(overlay.bone_items[0]) == {theme.color_for("face_lips").name()}


def test_a_face_with_no_extent_keeps_its_points(qtbot) -> None:  # type: ignore[no-untyped-def]
    """Every point on one coordinate is an unmeasured face, not a small one.

    Dropping the points there would leave nothing at all: the contours through
    coincident points are zero-length strokes. So the trade is not made.
    """
    _, overlay = _overlay(qtbot, extent=0.0)
    assert overlay.face_detail is FaceDetail.POINTS
    assert overlay.keypoint_item_count == FACE_POINTS


def test_the_body_is_never_thinned_out(qtbot) -> None:  # type: ignore[no-untyped-def]
    """Only the face has a detail level; a tiny body still draws every joint."""
    _, overlay = _overlay(qtbot, extent=CONTOURS_EXTENT, groups="all")
    assert overlay.face_detail is FaceDetail.CONTOURS
    assert overlay.keypoint_item_count == 133 - FACE_POINTS


# --------------------------------------------------------- when it rebuilds


def test_a_zoom_within_one_band_does_not_rebuild(qtbot) -> None:  # type: ignore[no-untyped-def]
    """The reason the level is checked before rebuilding: a wheel event arrives
    dozens of times a second, and a rebuild is every item in the scene."""
    _, overlay = _overlay(qtbot, extent=POINTS_EXTENT)
    before = list(overlay.keypoint_items)
    assert before, "the test is only meaningful if there is something to rebuild"

    for zoom in (1.05, 1.1, 1.2, 0.95):
        overlay.on_zoom_changed(zoom)

    assert overlay.keypoint_items == before, "items were recreated inside one band"


def test_crossing_a_threshold_does_rebuild(qtbot) -> None:  # type: ignore[no-untyped-def]
    _, overlay = _overlay(qtbot, extent=POINTS_EXTENT)
    before = list(overlay.keypoint_items)

    overlay.on_zoom_changed(_zoom_into_contours(POINTS_EXTENT))

    assert overlay.keypoint_items != before
    assert overlay.keypoint_item_count == 0


def test_a_zoom_before_any_result_is_harmless(qtbot) -> None:  # type: ignore[no-untyped-def]
    from alchemypose.gui.canvas.scene import PoseOverlay
    from alchemypose.gui.canvas.view import PoseCanvas

    canvas = PoseCanvas()
    qtbot.addWidget(canvas)
    overlay = PoseOverlay(canvas)
    overlay.on_zoom_changed(12.0)
    assert overlay.visible_item_count == 0


# ------------------------------------------------------- the canvas's signal


def test_the_canvas_announces_every_change_of_scale(qtbot) -> None:  # type: ignore[no-untyped-def]
    from alchemypose.gui.canvas.view import PoseCanvas

    canvas = PoseCanvas()
    qtbot.addWidget(canvas)
    seen: list[float] = []
    canvas.zoom_changed.connect(seen.append)

    canvas.set_image(np.zeros((480, 640, 3), dtype=np.uint8))    # resets the transform
    canvas.zoom_by(2.0)
    canvas.fit_to_window()
    canvas.zoom_to_rect(QRectF(0.0, 0.0, 100.0, 100.0))

    assert len(seen) == 4
    assert seen[-1] == pytest.approx(canvas.zoom_percent / 100.0)


def test_a_refused_zoom_step_announces_nothing(qtbot) -> None:  # type: ignore[no-untyped-def]
    """``zoom_by`` outside the allowed range leaves the transform alone, so
    there is nothing to announce — a receiver may trust the signal completely."""
    from alchemypose.gui.canvas.view import MAX_ZOOM, PoseCanvas

    canvas = PoseCanvas()
    qtbot.addWidget(canvas)
    canvas.set_image(np.zeros((480, 640, 3), dtype=np.uint8))
    seen: list[float] = []
    canvas.zoom_changed.connect(seen.append)

    canvas.zoom_by(MAX_ZOOM * 2.0)

    assert seen == []


def test_a_fit_with_no_image_announces_nothing(qtbot) -> None:  # type: ignore[no-untyped-def]
    from alchemypose.gui.canvas.view import PoseCanvas

    canvas = PoseCanvas()
    qtbot.addWidget(canvas)
    seen: list[float] = []
    canvas.zoom_changed.connect(seen.append)

    canvas.fit_to_window()

    assert seen == []


# ------------------------------------------------------------- the Face chip


def test_the_face_chip_magnifies_the_face(qtbot, reporter) -> None:  # type: ignore[no-untyped-def]
    """What a separate face tab would have given, delivered on the same canvas:
    choosing Face frames the head, and the head is then large enough on screen
    to earn its individual points."""
    from alchemypose.gui.main_window import MainWindow

    window = MainWindow(reporter=reporter, engine_factory=_IdleEngine)
    qtbot.addWidget(window)
    with qtbot.waitExposed(window):
        window.show()
    window.canvas.set_image(np.zeros((640, 640, 3), dtype=np.uint8))
    window.canvas.fit_to_window()
    window._on_result(_face_result(extent=64.0))
    fitted = window.canvas.zoom_percent

    window._apply_preset("face")

    assert window.canvas.zoom_percent > fitted
    assert window.canvas_overlay.face_detail is FaceDetail.POINTS
    assert window.canvas_overlay.keypoint_item_count == FACE_POINTS


def test_the_face_chip_before_any_result_leaves_the_view_alone(qtbot, reporter) -> None:  # type: ignore[no-untyped-def]
    """Clickable in the seconds between opening a photograph and detecting in it."""
    from alchemypose.gui.main_window import MainWindow

    window = MainWindow(reporter=reporter, engine_factory=_IdleEngine)
    qtbot.addWidget(window)
    window.canvas.set_image(np.zeros((640, 640, 3), dtype=np.uint8))
    window.canvas.fit_to_window()
    before = window.canvas.zoom_percent

    window._apply_preset("face")

    assert window.canvas.zoom_percent == pytest.approx(before)
    assert window.layers.current_selection().visible == selection_for("face").visible


def test_the_face_chip_with_nobody_detected_leaves_the_view_alone(qtbot, reporter) -> None:  # type: ignore[no-untyped-def]
    from alchemypose.gui.main_window import MainWindow

    empty = PoseResult(
        topology=COCO_WHOLEBODY_133,
        frames=(PoseFrame(frame_index=0, timestamp_s=None, image_size=(640, 640), people=()),),
        source="x",
        model="m",
    )
    window = MainWindow(reporter=reporter, engine_factory=_IdleEngine)
    qtbot.addWidget(window)
    window.canvas.set_image(np.zeros((640, 640, 3), dtype=np.uint8))
    window.canvas.fit_to_window()
    window._on_result(empty)
    before = window.canvas.zoom_percent

    window._apply_preset("face")

    assert window.canvas.zoom_percent == pytest.approx(before)
