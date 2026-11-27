"""Select and solo: the controller, the gestures that drive it, and the window
that owns the result.

Two fixtures, deliberately different. ``rig`` is the controller, a layer tree,
and one wire from the tree to the canvas — a cut-down stand-in that exists to
prove ``SelectionController`` has exactly one way out, and nothing more.
``window`` is the real ``MainWindow``. Anything about how the tree's selection
and the person chips combine belongs in the ``window`` tests, because that
combining is precisely what the rig leaves out; see the fixture below.
"""

import numpy as np
import pytest
from PySide6.QtCore import QPoint, Qt

from alchemypose.core.presets import selection_for
from alchemypose.core.schema import KeypointSelection, PersonPose, PoseFrame, PoseResult
from alchemypose.core.topology import COCO_WHOLEBODY_133, RENDER_GROUPS
from alchemypose.gui.canvas.items import HIT_RADIUS, MARKER_RADIUS
from alchemypose.gui.canvas.scene import DIM_FACTOR

pytestmark = pytest.mark.gui

rng = np.random.default_rng(2)

GRID_SPACING = 40.0
"""Distance between neighbouring joints in ``_grid_result``, in scene pixels."""

NEAR_MISS = HIT_RADIUS - 3.0
"""How far off centre the 'draw small, hit big' test clicks.

Derived from the constants it is testing, not chosen: outside the drawn dot,
inside the hit shape. If either radius is retuned this follows it, and the
assertion below fails loudly if the two ever stop leaving room for a click.
"""

BETWEEN_JOINTS = GRID_SPACING / 2.0
"""Offset to the midpoint of a grid cell — the nearest joint from there is
``hypot(20, 20)`` ≈ 28 px away, comfortably outside every hit shape."""

assert MARKER_RADIUS < NEAR_MISS < HIT_RADIUS < BETWEEN_JOINTS


def _result() -> PoseResult:
    person = PersonPose(
        keypoints=rng.random((133, 2), dtype=np.float32) * 400 + 20,
        scores=np.full((133,), 0.9, dtype=np.float32),
        bbox=(0.0, 0.0, 440.0, 440.0),
    )
    frame = PoseFrame(frame_index=0, timestamp_s=None, image_size=(640, 480), people=(person,))
    return PoseResult(topology=COCO_WHOLEBODY_133, frames=(frame,), source="x", model="m")


def _grid_result() -> PoseResult:
    """Keypoints laid out on a coarse grid, far enough apart that no two
    11 px hit-radius circles can overlap — needed for click tests, which must
    resolve to one unambiguous item."""
    xs = GRID_SPACING * (np.arange(133) % 12) + 30.0
    ys = GRID_SPACING * (np.arange(133) // 12) + 30.0
    keypoints = np.stack([xs, ys], axis=1).astype(np.float32)
    person = PersonPose(
        keypoints=keypoints,
        scores=np.full((133,), 0.9, dtype=np.float32),
        bbox=(0.0, 0.0, 500.0, 500.0),
    )
    frame = PoseFrame(frame_index=0, timestamp_s=None, image_size=(500, 500), people=(person,))
    return PoseResult(topology=COCO_WHOLEBODY_133, frames=(frame,), source="x", model="m")


def _crowd_result(people: int) -> PoseResult:
    """Several people, all stacked on one spot: the tests that use this count
    items and read ``person_index``, and never click."""
    person = PersonPose(
        keypoints=np.full((133, 2), 60.0, dtype=np.float32),
        scores=np.full((133,), 0.9, dtype=np.float32),
    )
    frame = PoseFrame(
        frame_index=0, timestamp_s=None, image_size=(500, 500), people=tuple([person] * people)
    )
    return PoseResult(topology=COCO_WHOLEBODY_133, frames=(frame,), source="x", model="m")


@pytest.fixture
def rig(qtbot):  # type: ignore[no-untyped-def]
    """A controller and the one wire it is allowed to reach the canvas through.

    That wire is the whole point: ``SelectionController`` writes only to the
    tree, so a solo reaches the canvas only because the tree announced it, and
    cutting the wire (as one test below does) must leave the canvas untouched.

    It is not how ``MainWindow`` connects the same two objects. The rig joins
    ``selection_changed`` straight to ``PoseOverlay.set_selection``, which
    carries the tree's *raw* selection; the window routes it through
    ``_on_selection_changed`` -> ``_push_selection`` ->
    ``set_selection(_effective_selection())``, narrowing it to the chosen
    person on the way. So no test using this fixture can catch a regression in
    that narrowing, or in anything else the window adds — the person-chip
    tests at the end of this file use the real window for exactly that reason.
    """
    from alchemypose.gui.canvas.interaction import SelectionController
    from alchemypose.gui.canvas.scene import PoseOverlay
    from alchemypose.gui.canvas.view import PoseCanvas
    from alchemypose.gui.panels.layers import LayerPanel

    canvas = PoseCanvas()
    qtbot.addWidget(canvas)
    canvas.set_image(np.zeros((480, 640, 3), dtype=np.uint8))
    overlay = PoseOverlay(canvas)
    layers = LayerPanel()
    qtbot.addWidget(layers)
    layers.selection_changed.connect(overlay.set_selection)
    overlay.set_result(_result(), selection_for("all"), confidence_threshold=0.0)
    return SelectionController(overlay, layers), overlay, layers


class _StubEngine:
    """Never actually asked for anything: the window tests feed results in by hand."""

    model_name = "fake"

    def infer(self, image: np.ndarray, source: str) -> PoseResult:  # type: ignore[type-arg]
        return _grid_result()


@pytest.fixture
def window(qtbot, reporter):  # type: ignore[no-untyped-def]
    """A real window, shown: ``QShortcut`` defaults to window context, so the
    Shift+H / Alt+H / Esc tests need one that can be active."""
    from alchemypose.gui.main_window import MainWindow

    window = MainWindow(reporter=reporter, engine_factory=_StubEngine)
    qtbot.addWidget(window)
    with qtbot.waitExposed(window):
        window.show()
    return window


def _show(window, result: PoseResult) -> None:  # type: ignore[no-untyped-def]
    """Put a result on the canvas with everything visible and no zoom applied.

    The transform stays at identity so a view-pixel offset in a click test is
    also a scene-pixel offset, and the hit radius means what it says.
    """
    window._apply_preset("all")
    window.canvas.set_image(np.zeros((500, 500, 3), dtype=np.uint8))
    window._on_result(result)


def _point_on(canvas, item, dx: float = 0.0, dy: float = 0.0) -> QPoint:  # type: ignore[no-untyped-def]
    return canvas.mapFromScene(item.scenePos()) + QPoint(round(dx), round(dy))


def _item_in(overlay, group: str):  # type: ignore[no-untyped-def]
    return next(item for item in overlay.keypoint_items if item.group_name == group)


# ------------------------------------------------------------------ emphasis


def test_select_emphasises_without_hiding_anything(rig) -> None:  # type: ignore[no-untyped-def]
    controller, overlay, _ = rig
    before = overlay.keypoint_item_count
    controller.select_group("left_hand")
    assert controller.emphasised == "left_hand"
    assert overlay.keypoint_item_count == before, "select must not remove items"


def test_select_dims_items_outside_the_group(rig) -> None:  # type: ignore[no-untyped-def]
    controller, overlay, _ = rig
    controller.select_group("left_hand")
    for item in overlay.keypoint_items:
        expected = 1.0 if item.group_name == "left_hand" else pytest.approx(DIM_FACTOR)
        assert item.opacity() == expected


def test_emphasis_survives_a_rebuild_triggered_by_the_layer_tree(rig) -> None:  # type: ignore[no-untyped-def]
    """The property the overlay-side composition exists to guarantee: a layer
    toggle (which rebuilds every item) must not silently drop the user's dim."""
    controller, overlay, _ = rig
    controller.select_group("left_hand")

    overlay.set_selection(selection_for("all"))            # what LayerPanel fires on a toggle
    overlay.set_confidence_threshold(0.5)                  # and what the threshold slider fires

    assert overlay.keypoint_item_count == 133, "rebuild must have actually happened"
    for item in overlay.keypoint_items:
        expected = 1.0 if item.group_name == "left_hand" else pytest.approx(DIM_FACTOR)
        assert item.opacity() == expected
    for bone in overlay.bone_items:
        assert bone.opacity() == pytest.approx(DIM_FACTOR)


def test_clear_after_a_rebuild_restores_full_opacity(rig) -> None:  # type: ignore[no-untyped-def]
    controller, overlay, _ = rig
    controller.select_group("left_hand")
    overlay.set_selection(selection_for("all"))
    controller.clear()
    assert all(i.opacity() == pytest.approx(1.0) for i in overlay.keypoint_items)


# ---------------------------------------------------------------------- solo


def test_solo_hides_every_other_group(rig) -> None:  # type: ignore[no-untyped-def]
    """What the canvas ends up drawing, arrived at the way the app arrives at
    it: the controller writes the tree, the tree announces, the canvas obeys."""
    controller, overlay, layers = rig
    controller.solo_group("left_hand")

    assert controller.soloed == "left_hand"
    assert layers.current_selection().visible == frozenset(RENDER_GROUPS["left_hand"])
    assert overlay.keypoint_item_count == len(RENDER_GROUPS["left_hand"])


def test_solo_reaches_the_canvas_only_through_the_tree(rig) -> None:  # type: ignore[no-untyped-def]
    """Cut the tree's wire to the canvas and a solo must not reach it anyway.

    A controller that also pushed to the overlay itself would pass every other
    test in this file and still be wrong: its push carries the tree's raw
    selection, with no idea that the window may have narrowed the view to one
    person. Only the window knows that, so only the window may write.
    """
    controller, overlay, layers = rig
    layers.selection_changed.disconnect()
    before = overlay.keypoint_item_count

    controller.solo_group("left_hand")

    assert overlay.keypoint_item_count == before, "the controller wrote to the canvas directly"


def test_solo_writes_through_to_the_layer_tree(rig) -> None:  # type: ignore[no-untyped-def]
    """Not just 'the tree ends up narrow' — the tree is what was *written*, and
    it said so once, so anything downstream of it hears about the solo."""
    controller, _, layers = rig
    announced: list[KeypointSelection] = []
    layers.selection_changed.connect(announced.append)

    controller.solo_group("right_leg")

    assert [s.visible for s in announced] == [frozenset(RENDER_GROUPS["right_leg"])]
    assert layers.group_rows["right_leg"].checkState(0) == Qt.CheckState.Checked
    assert all(
        row.checkState(0) == Qt.CheckState.Unchecked
        for key, row in layers.group_rows.items()
        if key != "right_leg"
    )


def test_clear_restores_what_was_visible_before_the_solo(rig) -> None:  # type: ignore[no-untyped-def]
    controller, overlay, layers = rig
    layers.set_selection(selection_for("upper"))
    before = layers.current_selection().visible
    controller.solo_group("left_hand")
    controller.clear()
    assert layers.current_selection().visible == before
    assert controller.soloed is None
    assert overlay.keypoint_item_count == len(before)


def test_clear_also_drops_the_emphasis(rig) -> None:  # type: ignore[no-untyped-def]
    controller, _, _ = rig
    controller.select_group("left_arm")
    controller.clear()
    assert controller.emphasised is None


def test_soloing_twice_still_restores_the_original(rig) -> None:  # type: ignore[no-untyped-def]
    controller, _, layers = rig
    layers.set_selection(selection_for("full_body"))
    original = layers.current_selection().visible
    controller.solo_group("left_arm")
    controller.solo_group("right_leg")
    controller.clear()
    assert layers.current_selection().visible == original


# ------------------------------------------------------------- the hit shape


def test_clicking_a_keypoint_selects_its_group(qtbot) -> None:  # type: ignore[no-untyped-def]
    from alchemypose.gui.canvas.scene import PoseOverlay
    from alchemypose.gui.canvas.view import PoseCanvas

    canvas = PoseCanvas()
    qtbot.addWidget(canvas)
    canvas.set_image(np.zeros((500, 500, 3), dtype=np.uint8))
    overlay = PoseOverlay(canvas)
    overlay.set_result(_grid_result(), selection_for("all"), confidence_threshold=0.0)

    item = overlay.keypoint_items[0]
    received: list[tuple[int, int, str]] = []
    canvas.keypoint_clicked.connect(lambda p, i, g: received.append((p, i, g)))

    qtbot.mouseClick(canvas.viewport(), Qt.MouseButton.LeftButton, pos=_point_on(canvas, item))

    assert received == [(item.person_index, item.keypoint_index, item.group_name)]


def test_double_clicking_a_keypoint_emits_double_click(qtbot) -> None:  # type: ignore[no-untyped-def]
    from alchemypose.gui.canvas.scene import PoseOverlay
    from alchemypose.gui.canvas.view import PoseCanvas

    canvas = PoseCanvas()
    qtbot.addWidget(canvas)
    canvas.set_image(np.zeros((500, 500, 3), dtype=np.uint8))
    overlay = PoseOverlay(canvas)
    overlay.set_result(_grid_result(), selection_for("all"), confidence_threshold=0.0)

    item = overlay.keypoint_items[0]
    received: list[tuple[int, int, str]] = []
    canvas.keypoint_double_clicked.connect(lambda p, i, g: received.append((p, i, g)))

    qtbot.mouseDClick(canvas.viewport(), Qt.MouseButton.LeftButton, pos=_point_on(canvas, item))

    assert received == [(item.person_index, item.keypoint_index, item.group_name)]


def test_clicking_just_off_a_joint_still_hits_it(qtbot) -> None:  # type: ignore[no-untyped-def]
    """'Draw small, hit big': the dot is 6 px across but ``KeypointItem.shape``
    claims 11 px, and this is the only test that would notice if that override
    were dropped — every other click lands dead centre, where the drawn dot
    would answer on its own."""
    from alchemypose.gui.canvas.scene import PoseOverlay
    from alchemypose.gui.canvas.view import PoseCanvas

    canvas = PoseCanvas()
    qtbot.addWidget(canvas)
    canvas.set_image(np.zeros((500, 500, 3), dtype=np.uint8))
    overlay = PoseOverlay(canvas)
    overlay.set_result(_grid_result(), selection_for("all"), confidence_threshold=0.0)

    item = overlay.keypoint_items[0]
    received: list[tuple[int, int, str]] = []
    canvas.keypoint_clicked.connect(lambda p, i, g: received.append((p, i, g)))

    qtbot.mouseClick(
        canvas.viewport(), Qt.MouseButton.LeftButton, pos=_point_on(canvas, item, dx=NEAR_MISS)
    )
    qtbot.mouseClick(
        canvas.viewport(), Qt.MouseButton.LeftButton, pos=_point_on(canvas, item, dy=NEAR_MISS)
    )

    hit = (item.person_index, item.keypoint_index, item.group_name)
    assert received == [hit, hit]


def test_clicking_between_joints_hits_nothing(qtbot) -> None:  # type: ignore[no-untyped-def]
    """The pad is generous, not unbounded — a click in the gap selects nothing
    rather than grabbing the nearest joint."""
    from alchemypose.gui.canvas.scene import PoseOverlay
    from alchemypose.gui.canvas.view import PoseCanvas

    canvas = PoseCanvas()
    qtbot.addWidget(canvas)
    canvas.set_image(np.zeros((500, 500, 3), dtype=np.uint8))
    overlay = PoseOverlay(canvas)
    overlay.set_result(_grid_result(), selection_for("all"), confidence_threshold=0.0)

    item = overlay.keypoint_items[0]
    received: list[tuple[int, int, str]] = []
    canvas.keypoint_clicked.connect(lambda p, i, g: received.append((p, i, g)))

    pos = _point_on(canvas, item, dx=BETWEEN_JOINTS, dy=BETWEEN_JOINTS)
    qtbot.mouseClick(canvas.viewport(), Qt.MouseButton.LeftButton, pos=pos)

    assert received == []


# ------------------------------------------------------- gestures, end to end


def test_clicking_a_joint_in_the_window_emphasises_its_group(window, qtbot) -> None:  # type: ignore[no-untyped-def]
    _show(window, _grid_result())
    overlay = window.canvas_overlay
    item = _item_in(overlay, "left_hand")

    qtbot.mouseClick(
        window.canvas.viewport(), Qt.MouseButton.LeftButton, pos=_point_on(window.canvas, item)
    )

    assert window.selection.emphasised == "left_hand"
    assert overlay.emphasis == "left_hand"
    assert overlay.keypoint_item_count == 133, "select hides nothing"
    assert "left hand" in window.status


def test_double_clicking_a_joint_in_the_window_solos_its_group(window, qtbot) -> None:  # type: ignore[no-untyped-def]
    _show(window, _grid_result())
    overlay = window.canvas_overlay
    item = _item_in(overlay, "left_hand")

    qtbot.mouseDClick(
        window.canvas.viewport(), Qt.MouseButton.LeftButton, pos=_point_on(window.canvas, item)
    )

    assert window.selection.soloed == "left_hand"
    assert overlay.keypoint_item_count == len(RENDER_GROUPS["left_hand"])
    assert window.layers.current_selection().visible == frozenset(RENDER_GROUPS["left_hand"])
    assert "Solo" in window.status


def test_shift_h_solos_the_emphasised_group(window, qtbot) -> None:  # type: ignore[no-untyped-def]
    _show(window, _grid_result())
    window.canvas.keypoint_clicked.emit(0, 0, "left_hand")     # the click that came first

    qtbot.keyClick(window, Qt.Key.Key_H, Qt.KeyboardModifier.ShiftModifier)

    assert window.selection.soloed == "left_hand"
    assert window.canvas_overlay.keypoint_item_count == len(RENDER_GROUPS["left_hand"])


def test_shift_h_with_nothing_emphasised_does_nothing(window, qtbot) -> None:  # type: ignore[no-untyped-def]
    _show(window, _grid_result())

    qtbot.keyClick(window, Qt.Key.Key_H, Qt.KeyboardModifier.ShiftModifier)

    assert window.selection.soloed is None
    assert window.canvas_overlay.keypoint_item_count == 133


def test_alt_h_undoes_a_solo(window, qtbot) -> None:  # type: ignore[no-untyped-def]
    _show(window, _grid_result())
    window.canvas.keypoint_double_clicked.emit(0, 0, "left_hand")
    assert window.canvas_overlay.keypoint_item_count == len(RENDER_GROUPS["left_hand"])

    qtbot.keyClick(window, Qt.Key.Key_H, Qt.KeyboardModifier.AltModifier)

    assert window.selection.soloed is None
    assert window.selection.emphasised is None
    assert window.canvas_overlay.keypoint_item_count == 133


def test_escape_undoes_a_solo_and_the_emphasis(window, qtbot) -> None:  # type: ignore[no-untyped-def]
    _show(window, _grid_result())
    window.canvas.keypoint_double_clicked.emit(0, 0, "left_hand")

    qtbot.keyClick(window, Qt.Key.Key_Escape)

    assert window.selection.soloed is None
    assert window.selection.emphasised is None
    assert window.canvas_overlay.emphasis is None
    assert window.canvas_overlay.keypoint_item_count == 133


# ------------------------------------------------ solo versus the person chips


def test_soloing_keeps_the_overlay_narrowed_to_the_chosen_person(window) -> None:  # type: ignore[no-untyped-def]
    """A solo used to hand the canvas the tree's raw selection, which says
    nothing about people — so every person came back while the chip still
    claimed one. The chip and the canvas must agree."""
    _show(window, _crowd_result(3))
    window.people.chips[1].click()
    assert window.canvas_overlay.keypoint_item_count == 133

    window.canvas.keypoint_double_clicked.emit(1, 0, "left_hand")

    assert window.canvas_overlay.keypoint_item_count == len(RENDER_GROUPS["left_hand"])
    assert {i.person_index for i in window.canvas_overlay.keypoint_items} == {1}
    assert window.people.current == 1


def test_undoing_a_solo_returns_to_the_chosen_person_not_to_everyone(window, qtbot) -> None:  # type: ignore[no-untyped-def]
    _show(window, _crowd_result(3))
    window.people.chips[1].click()
    window.canvas.keypoint_double_clicked.emit(1, 0, "left_hand")

    qtbot.keyClick(window, Qt.Key.Key_Escape)

    assert window.canvas_overlay.keypoint_item_count == 133
    assert {i.person_index for i in window.canvas_overlay.keypoint_items} == {1}
    assert window.people.current == 1


def test_choosing_everyone_after_a_solo_shows_everyone(window) -> None:  # type: ignore[no-untyped-def]
    """The other direction: the person filter must not outlive the chip either."""
    _show(window, _crowd_result(3))
    window.people.chips[1].click()
    window.canvas.keypoint_double_clicked.emit(1, 0, "left_hand")

    window.people.chips[1].click()      # clicking the chosen chip again means everyone

    assert window.canvas_overlay.keypoint_item_count == 3 * len(RENDER_GROUPS["left_hand"])
    assert {i.person_index for i in window.canvas_overlay.keypoint_items} == {0, 1, 2}


# ------------------------------------------------------------ photograph dimming
#
# ``set_image_dimmed`` existed and was tested in isolation, but nothing called
# it. These pin the call sites: a selection pushes the photograph back so the
# skeleton reads as the subject, and clearing it brings the photograph forward.


def test_selecting_a_group_dims_the_photograph(rig) -> None:  # type: ignore[no-untyped-def]
    from alchemypose.gui import theme

    controller, overlay, _layers = rig
    controller.select_group("left_arm")
    assert overlay._canvas.image_item.opacity() == pytest.approx(theme.DIMMED_IMAGE_OPACITY)


def test_soloing_a_group_dims_the_photograph(rig) -> None:  # type: ignore[no-untyped-def]
    from alchemypose.gui import theme

    controller, overlay, _layers = rig
    controller.solo_group("right_leg")
    assert overlay._canvas.image_item.opacity() == pytest.approx(theme.DIMMED_IMAGE_OPACITY)


def test_clearing_the_selection_restores_the_photograph(rig) -> None:  # type: ignore[no-untyped-def]
    controller, overlay, _layers = rig
    controller.solo_group("right_leg")
    controller.clear()
    assert overlay._canvas.image_item.opacity() == pytest.approx(1.0)


# --------------------------------------------------------------- staggered draw-in


def test_a_new_result_animates_the_skeleton_in(window) -> None:  # type: ignore[no-untyped-def]
    _show(window, _grid_result())
    animation = window.canvas_overlay._animation
    assert animation is not None
    assert animation.state() == animation.State.Running


def test_a_result_with_nobody_in_it_starts_no_animation(window) -> None:  # type: ignore[no-untyped-def]
    empty = PoseResult(
        topology=COCO_WHOLEBODY_133,
        frames=(PoseFrame(frame_index=0, timestamp_s=None, image_size=(500, 500), people=()),),
        source="x",
        model="m",
    )
    _show(window, empty)
    assert window.canvas_overlay._animation is None


def test_a_press_during_the_draw_in_finishes_it(window) -> None:  # type: ignore[no-untyped-def]
    """Fully transparent items are invisible to Qt's hit test, so a click in the
    first moments after a result lands must end the animation before resolving."""
    _show(window, _grid_result())
    assert window.canvas_overlay._animation is not None
    window.canvas.pressed.emit()
    assert window.canvas_overlay._animation is None
    assert all(item.opacity() > 0.0 for item in window.canvas_overlay.keypoint_items)
