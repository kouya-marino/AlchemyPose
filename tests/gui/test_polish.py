"""Confidence opacity, image dimming, staggered draw-in — and the
canvas/overlay lifetime seam that a second ``set_image`` exposes."""

import numpy as np
import pytest

from alchemypose.core.presets import selection_for
from alchemypose.core.schema import PersonPose, PoseFrame, PoseResult
from alchemypose.core.topology import COCO_WHOLEBODY_133
from alchemypose.gui.canvas.scene import (
    DASH_BELOW,
    DIM_FACTOR,
    FULL_OPACITY_ABOVE,
    MIN_OPACITY,
    confidence_opacity,
)

pytestmark = pytest.mark.gui

# Scores are derived from the calibration constants, never hard-coded, so a
# retune of the constants cannot break a test that is not about the retune.
THRESHOLD = 0.3
CONFIDENT = (1.0 + FULL_OPACITY_ABOVE) / 2.0     # strictly above the full-opacity knee
MARGINAL = FULL_OPACITY_ABOVE / 2.0              # strictly between a 0.0 threshold and the knee
assert 0.0 < MARGINAL < FULL_OPACITY_ABOVE < CONFIDENT < 1.0


def _result(score: float) -> PoseResult:
    person = PersonPose(
        keypoints=np.full((133, 2), 100.0, dtype=np.float32),
        scores=np.full((133,), score, dtype=np.float32),
    )
    frame = PoseFrame(frame_index=0, timestamp_s=None, image_size=(640, 480), people=(person,))
    return PoseResult(topology=COCO_WHOLEBODY_133, frames=(frame,), source="x", model="m")


def _overlay(qtbot, score: float, preset: str = "full_body"):  # type: ignore[no-untyped-def]
    from alchemypose.gui.canvas.scene import PoseOverlay
    from alchemypose.gui.canvas.view import PoseCanvas

    canvas = PoseCanvas()
    qtbot.addWidget(canvas)
    canvas.set_image(np.zeros((480, 640, 3), dtype=np.uint8))
    overlay = PoseOverlay(canvas)
    overlay.set_result(_result(score), selection_for(preset), confidence_threshold=0.0)
    return canvas, overlay


# ------------------------------------------------------------- the lifetime seam


def test_loading_a_second_image_does_not_leave_dangling_items(qtbot) -> None:  # type: ignore[no-untyped-def]
    """Regression: scene.clear() destroys the overlay's items at the C++ level.
    Without the scene_cleared signal, the next set_selection crashes with
    'Internal C++ object already deleted'."""
    canvas, overlay = _overlay(qtbot, score=CONFIDENT, preset="all")
    assert overlay.keypoint_item_count == 133

    canvas.set_image(np.zeros((300, 200, 3), dtype=np.uint8))

    assert overlay.keypoint_item_count == 0
    assert overlay.bone_item_count == 0
    # The layer tree fires this on every toggle; it must not touch dead items.
    # (It also draws nothing — the old result went with the old image; see the
    # next test — so the point here is only that it returns rather than raises.)
    overlay.set_selection(selection_for("hands"))
    assert overlay.keypoint_item_count == 0


def test_a_layer_toggle_after_reload_does_not_redraw_the_previous_skeleton(qtbot) -> None:  # type: ignore[no-untyped-def]
    """The overlay's result belongs to the image that was just replaced. Until
    inference for the new image arrives, a toggle must draw nothing — not image
    A's skeleton over photo B."""
    canvas, overlay = _overlay(qtbot, score=CONFIDENT, preset="all")
    canvas.set_image(np.zeros((300, 200, 3), dtype=np.uint8))

    overlay.set_selection(selection_for("hands"))
    assert overlay.keypoint_item_count == 0
    assert overlay.bone_item_count == 0
    assert overlay.person_bounds(0).isNull()

    overlay.set_result(_result(CONFIDENT), selection_for("hands"), confidence_threshold=0.0)
    assert overlay.keypoint_item_count == 42


def test_scene_cleared_fires_before_the_scene_is_wiped(qtbot) -> None:  # type: ignore[no-untyped-def]
    from alchemypose.gui.canvas.view import PoseCanvas

    canvas = PoseCanvas()
    qtbot.addWidget(canvas)
    canvas.set_image(np.zeros((480, 640, 3), dtype=np.uint8))
    item_counts: list[int] = []
    canvas.scene_cleared.connect(lambda: item_counts.append(len(canvas.scene().items())))

    canvas.set_image(np.zeros((480, 640, 3), dtype=np.uint8))

    assert item_counts == [1], "the old pixmap must still be alive when the signal fires"


def test_scene_cleared_with_nothing_to_forget_is_harmless(qtbot) -> None:  # type: ignore[no-untyped-def]
    from alchemypose.gui.canvas.scene import PoseOverlay
    from alchemypose.gui.canvas.view import PoseCanvas

    canvas = PoseCanvas()
    qtbot.addWidget(canvas)
    overlay = PoseOverlay(canvas)          # never given a result
    canvas.set_image(np.zeros((480, 640, 3), dtype=np.uint8))
    canvas.set_image(np.zeros((480, 640, 3), dtype=np.uint8))
    assert overlay.visible_item_count == 0


# ------------------------------------------------------------ confidence opacity


def test_confidence_opacity_is_full_above_the_knee() -> None:
    assert confidence_opacity(FULL_OPACITY_ABOVE, THRESHOLD) == 1.0
    assert confidence_opacity(CONFIDENT, THRESHOLD) == 1.0
    assert confidence_opacity(1.0, THRESHOLD) == 1.0


def test_confidence_opacity_is_minimal_at_the_threshold() -> None:
    assert confidence_opacity(THRESHOLD, THRESHOLD) == pytest.approx(MIN_OPACITY)


def test_confidence_opacity_is_monotonic_and_clamped() -> None:
    knee = FULL_OPACITY_ABOVE
    scores = (0.0, THRESHOLD / 2, THRESHOLD, (THRESHOLD + knee) / 2, knee - 0.01, knee, (knee + 1.0) / 2, 1.0)
    values = [confidence_opacity(s, THRESHOLD) for s in scores]
    assert values == sorted(values)
    assert values[0] == pytest.approx(MIN_OPACITY)   # below threshold: clamped, not negative
    assert MIN_OPACITY < values[3] < 1.0             # genuinely between floor and full
    assert values[-1] == 1.0


def test_confidence_opacity_survives_a_threshold_at_or_above_the_knee() -> None:
    # span would be zero or negative; must not divide by zero or exceed 1.0
    high = FULL_OPACITY_ABOVE + 0.1
    assert 0.0 < confidence_opacity(high - 0.05, high) <= 1.0
    assert confidence_opacity(FULL_OPACITY_ABOVE - 0.05, FULL_OPACITY_ABOVE) == pytest.approx(MIN_OPACITY)


def test_a_bone_is_dashed_only_when_its_weaker_joint_is_below_dash_below(qtbot) -> None:  # type: ignore[no-untyped-def]
    from PySide6.QtCore import Qt

    _, solid = _overlay(qtbot, score=DASH_BELOW)              # at the line: solid
    assert solid.bone_items[0].pen().style() == Qt.PenStyle.SolidLine
    _, dashed = _overlay(qtbot, score=DASH_BELOW - 0.01)      # just under: dashed
    assert dashed.bone_items[0].pen().style() == Qt.PenStyle.DashLine


def test_a_confident_keypoint_is_fully_opaque(qtbot) -> None:  # type: ignore[no-untyped-def]
    _, overlay = _overlay(qtbot, score=CONFIDENT)
    assert overlay.keypoint_items[0].opacity() == pytest.approx(1.0)


def test_a_marginal_keypoint_is_faded(qtbot) -> None:  # type: ignore[no-untyped-def]
    _, overlay = _overlay(qtbot, score=MARGINAL)
    opacity = overlay.keypoint_items[0].opacity()
    assert MIN_OPACITY <= opacity < 1.0
    assert opacity == pytest.approx(confidence_opacity(MARGINAL, 0.0))


def test_a_bone_between_marginal_joints_is_faded_too(qtbot) -> None:  # type: ignore[no-untyped-def]
    _, overlay = _overlay(qtbot, score=MARGINAL)
    assert overlay.bone_item_count > 0
    assert overlay.bone_items[0].opacity() < 1.0
    assert overlay.bone_items[0].opacity() == pytest.approx(confidence_opacity(MARGINAL, 0.0))


def test_zero_score_keypoints_are_never_drawn_even_at_threshold_zero(qtbot) -> None:  # type: ignore[no-untyped-def]
    """A score of exactly 0.0 is decode_simcc's sentinel for 'no peak': the
    position was zeroed too and un-transforms to the crop corner. The slider
    can reach 0.0; those joints must still not appear."""
    from alchemypose.gui.canvas.scene import PoseOverlay
    from alchemypose.gui.canvas.view import PoseCanvas

    scores = np.full((133,), CONFIDENT, dtype=np.float32)
    scores[[0, 5, 6]] = 0.0                                   # nose, both shoulders: no peak
    person = PersonPose(keypoints=np.full((133, 2), 100.0, dtype=np.float32), scores=scores)
    frame = PoseFrame(frame_index=0, timestamp_s=None, image_size=(640, 480), people=(person,))
    result = PoseResult(topology=COCO_WHOLEBODY_133, frames=(frame,), source="x", model="m")

    canvas = PoseCanvas()
    qtbot.addWidget(canvas)
    canvas.set_image(np.zeros((480, 640, 3), dtype=np.uint8))
    overlay = PoseOverlay(canvas)
    overlay.set_result(result, selection_for("all"), confidence_threshold=0.0)

    drawn = {item.keypoint_index for item in overlay.keypoint_items}
    assert drawn.isdisjoint({0, 5, 6})
    assert len(drawn) == 130
    # ...and a tiny-but-positive score is still honoured at threshold 0.0.
    scores[0] = 0.01
    overlay.set_result(result, selection_for("all"), confidence_threshold=0.0)
    assert 0 in {item.keypoint_index for item in overlay.keypoint_items}


# ---------------------------------------------------------------- image dimming


def test_dimming_the_image_lowers_only_the_pixmap_opacity(qtbot) -> None:  # type: ignore[no-untyped-def]
    from alchemypose.gui import theme

    canvas, overlay = _overlay(qtbot, score=CONFIDENT)
    overlay.set_image_dimmed(True)
    assert canvas.image_item is not None
    assert canvas.image_item.opacity() == pytest.approx(theme.DIMMED_IMAGE_OPACITY)
    assert overlay.keypoint_items[0].opacity() == pytest.approx(1.0)
    assert overlay.bone_items[0].opacity() == pytest.approx(1.0)


def test_undimming_restores_the_image(qtbot) -> None:  # type: ignore[no-untyped-def]
    canvas, overlay = _overlay(qtbot, score=CONFIDENT)
    overlay.set_image_dimmed(True)
    overlay.set_image_dimmed(False)
    assert canvas.image_item is not None
    assert canvas.image_item.opacity() == pytest.approx(1.0)


def test_dimming_with_no_image_loaded_is_a_no_op(qtbot) -> None:  # type: ignore[no-untyped-def]
    from alchemypose.gui.canvas.scene import PoseOverlay
    from alchemypose.gui.canvas.view import PoseCanvas

    canvas = PoseCanvas()
    qtbot.addWidget(canvas)
    PoseOverlay(canvas).set_image_dimmed(True)
    assert canvas.image_item is None


# -------------------------------------------------------------------- emphasis


def test_set_emphasis_dims_everything_outside_the_group(qtbot) -> None:  # type: ignore[no-untyped-def]
    _, overlay = _overlay(qtbot, score=CONFIDENT, preset="all")
    overlay.set_emphasis("left_hand")
    assert overlay.emphasis == "left_hand"
    for item in overlay.keypoint_items:
        expected = 1.0 if item.group_name == "left_hand" else DIM_FACTOR
        assert item.opacity() == pytest.approx(expected)
    for bone in overlay.bone_items:
        assert bone.opacity() == pytest.approx(DIM_FACTOR)


def test_set_emphasis_none_restores_confidence_opacity(qtbot) -> None:  # type: ignore[no-untyped-def]
    _, overlay = _overlay(qtbot, score=MARGINAL)
    overlay.set_emphasis("torso_head")
    overlay.set_emphasis(None)
    assert overlay.emphasis is None
    assert overlay.keypoint_items[0].opacity() == pytest.approx(confidence_opacity(MARGINAL, 0.0))


def test_set_emphasis_does_not_rebuild(qtbot) -> None:  # type: ignore[no-untyped-def]
    _, overlay = _overlay(qtbot, score=CONFIDENT)
    before = list(overlay.keypoint_items)
    overlay.set_emphasis("left_arm")
    assert overlay.keypoint_items == before, "emphasis must re-shade in place, not recreate items"


def test_emphasis_survives_a_rebuild(qtbot) -> None:  # type: ignore[no-untyped-def]
    """The collision with SelectionController: a rebuild used to wipe the dim."""
    _, overlay = _overlay(qtbot, score=CONFIDENT, preset="all")
    overlay.set_emphasis("left_hand")
    overlay.set_selection(selection_for("all"))      # what every layer toggle does
    for item in overlay.keypoint_items:
        expected = 1.0 if item.group_name == "left_hand" else DIM_FACTOR
        assert item.opacity() == pytest.approx(expected)


def test_emphasis_composes_with_confidence(qtbot) -> None:  # type: ignore[no-untyped-def]
    _, overlay = _overlay(qtbot, score=MARGINAL, preset="all")
    overlay.set_emphasis("left_hand")
    faded = confidence_opacity(MARGINAL, 0.0)
    assert faded < 1.0
    for item in overlay.keypoint_items:
        expected = faded if item.group_name == "left_hand" else faded * DIM_FACTOR
        assert item.opacity() == pytest.approx(expected)


# ------------------------------------------------------------------- animation


def test_animate_in_starts_from_invisible(qtbot) -> None:  # type: ignore[no-untyped-def]
    _, overlay = _overlay(qtbot, score=CONFIDENT)
    overlay.animate_in(duration_ms=10_000)
    assert overlay.keypoint_items[0].opacity() == pytest.approx(0.0)
    assert overlay.bone_items[0].opacity() == pytest.approx(0.0)


def test_animate_in_finishes_at_full_opacity(qtbot) -> None:  # type: ignore[no-untyped-def]
    _, overlay = _overlay(qtbot, score=CONFIDENT)
    overlay.animate_in(duration_ms=40)
    qtbot.waitUntil(lambda: overlay._animation is None, timeout=2000)
    assert all(i.opacity() == pytest.approx(1.0) for i in overlay.keypoint_items)
    assert all(b.opacity() == pytest.approx(1.0) for b in overlay.bone_items)


def test_animate_in_finishes_at_each_items_intended_opacity(qtbot) -> None:  # type: ignore[no-untyped-def]
    """A marginal keypoint must land on its faded value, not on 1.0."""
    _, overlay = _overlay(qtbot, score=MARGINAL, preset="all")
    overlay.set_emphasis("left_hand")
    overlay.animate_in(duration_ms=40)
    qtbot.waitUntil(lambda: overlay._animation is None, timeout=2000)
    faded = confidence_opacity(MARGINAL, 0.0)
    assert faded < 1.0, "the test is only meaningful if the intended opacity is not 1.0"
    for item in overlay.keypoint_items:
        expected = faded if item.group_name == "left_hand" else faded * DIM_FACTOR
        assert item.opacity() == pytest.approx(expected)


def test_animate_in_keeps_the_animation_alive(qtbot) -> None:  # type: ignore[no-untyped-def]
    _, overlay = _overlay(qtbot, score=CONFIDENT)
    overlay.animate_in(duration_ms=10_000)
    assert overlay._animation is not None
    assert overlay._animation.state() == overlay._animation.State.Running


def test_rebuilding_mid_animation_does_not_crash_or_leave_items_faded(qtbot) -> None:  # type: ignore[no-untyped-def]
    _, overlay = _overlay(qtbot, score=CONFIDENT)
    overlay.animate_in(duration_ms=10_000)
    overlay.set_selection(selection_for("hands"))     # abandons the draw-in
    assert overlay._animation is None
    assert all(i.opacity() == pytest.approx(1.0) for i in overlay.keypoint_items)


def test_second_image_mid_animation_does_not_crash(qtbot) -> None:  # type: ignore[no-untyped-def]
    canvas, overlay = _overlay(qtbot, score=CONFIDENT)
    overlay.animate_in(duration_ms=10_000)
    canvas.set_image(np.zeros((300, 200, 3), dtype=np.uint8))
    qtbot.wait(20)                                    # let any stray tick fire
    assert overlay.visible_item_count == 0
