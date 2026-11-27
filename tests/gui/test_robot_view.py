"""The G1 in the window, without OpenGL and without the network."""

from pathlib import Path

import numpy as np
import pytest

from alchemypose.robot.urdf import load_urdf

pytestmark = pytest.mark.gui

G1 = load_urdf(Path(__file__).resolve().parents[1] / "fixtures" / "unitree_g1" / "g1_29dof_rev_1_0.urdf")


def test_offscreen_the_robot_panel_explains_itself_and_still_records_poses(qtbot) -> None:  # type: ignore[no-untyped-def]
    from alchemypose.gui.view3d.robot import RobotView

    view = RobotView()
    qtbot.addWidget(view)
    assert not view.available
    view.load(G1, [])
    assert view.loaded
    view.set_pose(np.zeros(29), np.eye(3), 0.8)
    assert view.last_pose is not None


def test_link_transforms_place_the_pelvis_at_the_root_pose() -> None:
    from alchemypose.gui.view3d.robot import link_transforms

    turned = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    poses = link_transforms(G1, np.zeros(29), turned, 0.8)
    assert poses["pelvis"][:3, 3] == pytest.approx([0, 0, 0.8])
    assert poses["pelvis"][:3, :3] == pytest.approx(turned)


def test_baked_lighting_is_two_sided_and_shares_vertices() -> None:
    from alchemypose.gui.view3d.robot import lit_mesh

    up = [[0, 0, 0], [1, 0, 0], [0, 1, 0]]
    down = [[0, 0, 0], [0, 1, 0], [1, 0, 0]]           # same triangle, wound the other way
    vertices, faces, colours = lit_mesh(np.array([up, down], dtype=np.float32), (1.0, 1.0, 1.0, 1.0))
    assert len(vertices) == 3, "shared corners become one vertex"
    assert faces.shape == (2, 3)
    assert np.allclose(colours[0], colours[1])          # inward-wound faces are lit the same


def test_the_g1_toggle_switches_to_3d_and_loads_the_model(qtbot, reporter, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    from alchemypose.gui.main_window import MainWindow

    window = MainWindow(reporter=reporter, engine_factory=lambda: None)  # type: ignore[arg-type,return-value]
    qtbot.addWidget(window)
    window.show()
    asked: list[bool] = []
    monkeypatch.setattr(window.robot, "load_model", lambda: asked.append(True))
    window.robot_action.setChecked(True)
    assert window.preset == "3d"
    assert window.robot_view.isVisible()
    assert asked == [True]
    window.robot_action.setChecked(False)
    assert not window.robot_view.isVisible()


def test_a_finished_motion_poses_the_robot_and_reports_its_speed(qtbot, reporter) -> None:  # type: ignore[no-untyped-def]
    from alchemypose.gui.main_window import MainWindow
    from alchemypose.robot.retarget import RobotMotion, speed_report

    window = MainWindow(reporter=reporter, engine_factory=lambda: None)  # type: ignore[arg-type,return-value]
    qtbot.addWidget(window)
    window.robot_action.blockSignals(True)
    window.robot_action.setChecked(True)
    window.robot_action.blockSignals(False)
    window.robot.robot = G1
    window.robot_view.load(G1, [])
    count = 31
    q = np.zeros((count, 29))
    q[:, 3] = np.linspace(0, 20.0, count)               # far too fast: 20 rad/s on a knee
    motion = RobotMotion(G1.joint_names, np.arange(count) / 30.0, q, np.tile(np.eye(3), (count, 1, 1)),
                         np.full(count, 0.8), np.ones(count, dtype=bool), np.zeros(count), np.eye(3))
    report = speed_report(motion, np.array([j.velocity for j in G1.actuated]))
    window.robot.motion, window.robot.report = motion, report
    window._on_robot_motion(motion, report)
    assert "too fast" in window.status
    assert window.transport.speed < 1.0, "a slower playback rate is suggested"
    assert window.robot_view.last_pose is not None
