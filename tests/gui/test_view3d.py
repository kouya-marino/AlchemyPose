"""The 3D panel: what it draws (headless) and how the window shows it."""

from dataclasses import replace

import numpy as np
import pytest

from alchemypose.core.presets import selection_for
from alchemypose.core.schema import KeypointSelection, PersonPose, PoseFrame, PoseResult, Space
from alchemypose.core.topology import COCO_WHOLEBODY_133

pytestmark = pytest.mark.gui

K = 133


def _person(score: float = 0.9) -> PersonPose:
    kp3 = np.zeros((K, 3), dtype=np.float32)
    kp3[:, 1] = np.linspace(-0.8, 0.9, K)          # head (y up = negative) down to feet
    kp3[15], kp3[16] = (0.1, 0.9, 0.0), (-0.1, 0.9, 0.0)
    return PersonPose(
        keypoints=np.zeros((K, 2), dtype=np.float32),
        scores=np.full(K, score, dtype=np.float32),
        keypoints_3d=kp3,
        space_3d=Space.ROOT_REL,
    )


def _result(*people: PersonPose) -> PoseResult:
    return PoseResult(COCO_WHOLEBODY_133, (PoseFrame(0, None, (640, 480), people),), "x", "rtmw3d_x_384x288")


def test_world_axes_put_up_up_and_the_camera_in_front() -> None:
    from alchemypose.gui.view3d.geometry import to_world

    camera_space = np.array([[1.0, -2.0, 3.0]], dtype=np.float32)   # right, up (y down), away
    assert to_world(camera_space).tolist() == [[1.0, 3.0, 2.0]]


def test_the_lowest_body_point_stands_on_the_grid() -> None:
    from alchemypose.gui.view3d.geometry import build

    geometry = build(_result(_person()), selection_for("full_body"), 0.3)
    assert geometry.points[:, 2].min() == pytest.approx(0.0, abs=1e-6)
    assert geometry.points[:, 2].max() > 1.0


def test_the_3d_view_draws_exactly_what_the_canvas_would() -> None:
    from alchemypose.gui.view3d.geometry import build

    hands = build(_result(_person()), selection_for("hands"), 0.3)
    body = build(_result(_person()), selection_for("full_body"), 0.3)
    assert hands.points.shape[0] == 42
    assert body.points.shape[0] != hands.points.shape[0]


def test_doubtful_joints_are_left_out_like_on_the_canvas() -> None:
    from alchemypose.gui.view3d.geometry import build

    assert build(_result(_person(score=0.2)), selection_for("full_body"), 0.3).empty


def test_the_persons_left_is_green_in_3d_too() -> None:
    from alchemypose.gui import theme
    from alchemypose.gui.view3d.geometry import build

    geometry = build(_result(_person()), KeypointSelection(visible=frozenset({9})), 0.3)   # left wrist
    r, g, b = (int(theme.LEFT[i : i + 2], 16) / 255 for i in (1, 3, 5))
    assert geometry.point_colours[0, :3].tolist() == pytest.approx([r, g, b])


def test_two_people_stand_side_by_side_and_one_chosen_stands_alone_at_the_origin() -> None:
    from alchemypose.gui.view3d.geometry import PERSON_SPACING_M, build

    both = build(_result(_person(), _person()), selection_for("full_body"), 0.3)
    xs = both.points[:, 0]
    assert xs.max() - xs.min() == pytest.approx(PERSON_SPACING_M, abs=0.3)
    one = build(_result(_person(), _person()), replace(selection_for("full_body"), persons=frozenset({1})), 0.3)
    assert abs(float(one.points[:, 0].mean())) < 0.2


def test_no_3d_keypoints_means_nothing_to_draw() -> None:
    from alchemypose.gui.view3d.geometry import build

    flat = replace(_person(), keypoints_3d=None, space_3d=None)
    assert build(_result(flat), selection_for("full_body"), 0.3).empty
    assert build(None, selection_for("full_body"), 0.3).empty


def test_offscreen_the_panel_explains_itself_instead_of_crashing(qtbot) -> None:  # type: ignore[no-untyped-def]
    from alchemypose.gui.view3d.geometry import build
    from alchemypose.gui.view3d.widget import Skeleton3DView

    view = Skeleton3DView()
    qtbot.addWidget(view)
    assert not view.available
    assert "display" in (view.unavailable_reason or "")
    view.set_geometry(build(_result(_person()), selection_for("full_body"), 0.3))   # must not raise
    assert not view.last_geometry.empty


def test_the_window_shows_the_3d_panel_for_a_3d_result_and_hides_it_otherwise(qtbot, reporter) -> None:  # type: ignore[no-untyped-def]
    from alchemypose.gui.main_window import MainWindow

    window = MainWindow(reporter=reporter, engine_factory=lambda: None)  # type: ignore[arg-type,return-value]
    qtbot.addWidget(window)
    window.show()
    window.canvas.set_image(np.zeros((480, 640, 3), dtype=np.uint8))
    window._on_result(_result(_person()))
    assert window.view3d.isVisible()
    assert not window.view3d.last_geometry.empty

    flat = _result(replace(_person(), keypoints_3d=None, space_3d=None))
    window._pending_source = None
    window._on_result(flat)
    assert not window.view3d.isVisible()


def test_glb_export_is_offered_only_with_3d(qtbot, reporter) -> None:  # type: ignore[no-untyped-def]
    from alchemypose.gui.main_window import MainWindow

    window = MainWindow(reporter=reporter, engine_factory=lambda: None)  # type: ignore[arg-type,return-value]
    qtbot.addWidget(window)
    window._on_result(_result(replace(_person(), keypoints_3d=None, space_3d=None)))
    assert window._export_writer("glb") is None
    window._on_result(_result(_person()))
    assert window._export_writer("glb") is not None


def test_switching_preset_rebuilds_the_worker_and_retargets_video(qtbot, reporter) -> None:  # type: ignore[no-untyped-def]
    from alchemypose.gui.main_window import MainWindow

    window = MainWindow(reporter=reporter, engine_factory=lambda: None)  # type: ignore[arg-type,return-value]
    qtbot.addWidget(window)
    old_bridge = window._bridge
    window.set_preset("3d")
    assert window.preset == "3d"
    assert window._bridge is not old_bridge
    assert window.video._preset == "3d"
    assert window.preset_box.currentData() == "3d"
    window.set_preset("no-such-preset")
    assert window.preset == "3d"
