import numpy as np
import pytest

from alchemypose.core.presets import selection_for
from alchemypose.core.schema import PersonPose, PoseFrame, PoseResult
from alchemypose.core.topology import COCO_WHOLEBODY_133, RENDER_GROUPS
from alchemypose.gui.canvas.scene import DIM_FACTOR

pytestmark = pytest.mark.gui

rng = np.random.default_rng(2)


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
    xs = 40.0 * (np.arange(133) % 12) + 30.0
    ys = 40.0 * (np.arange(133) // 12) + 30.0
    keypoints = np.stack([xs, ys], axis=1).astype(np.float32)
    person = PersonPose(
        keypoints=keypoints,
        scores=np.full((133,), 0.9, dtype=np.float32),
        bbox=(0.0, 0.0, 500.0, 500.0),
    )
    frame = PoseFrame(frame_index=0, timestamp_s=None, image_size=(500, 500), people=(person,))
    return PoseResult(topology=COCO_WHOLEBODY_133, frames=(frame,), source="x", model="m")


@pytest.fixture
def rig(qtbot):  # type: ignore[no-untyped-def]
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
    overlay.set_result(_result(), selection_for("all"), confidence_threshold=0.0)
    return SelectionController(overlay, layers), overlay, layers


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


def test_solo_hides_every_other_group(rig) -> None:  # type: ignore[no-untyped-def]
    controller, overlay, _ = rig
    controller.solo_group("left_hand")
    assert controller.soloed == "left_hand"
    assert overlay.keypoint_item_count == len(RENDER_GROUPS["left_hand"])


def test_solo_writes_through_to_the_layer_tree(rig) -> None:  # type: ignore[no-untyped-def]
    controller, _, layers = rig
    controller.solo_group("right_leg")
    assert layers.current_selection().visible == set(RENDER_GROUPS["right_leg"])


def test_clear_restores_what_was_visible_before_the_solo(rig) -> None:  # type: ignore[no-untyped-def]
    controller, overlay, layers = rig
    layers.set_selection(selection_for("upper"))
    before = layers.current_selection().visible
    controller.solo_group("left_hand")
    controller.clear()
    assert layers.current_selection().visible == before
    assert controller.soloed is None


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


def test_clicking_a_keypoint_selects_its_group(qtbot) -> None:  # type: ignore[no-untyped-def]
    from PySide6.QtCore import Qt

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

    view_point = canvas.mapFromScene(item.scenePos())
    qtbot.mouseClick(canvas.viewport(), Qt.MouseButton.LeftButton, pos=view_point)

    assert received == [(item.person_index, item.keypoint_index, item.group_name)]


def test_double_clicking_a_keypoint_emits_double_click(qtbot) -> None:  # type: ignore[no-untyped-def]
    from PySide6.QtCore import Qt

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

    view_point = canvas.mapFromScene(item.scenePos())
    qtbot.mouseDClick(canvas.viewport(), Qt.MouseButton.LeftButton, pos=view_point)

    assert received == [(item.person_index, item.keypoint_index, item.group_name)]
