"""Reference-motion export: conventions, trimming, root, speed, and MuJoCo agreement."""

from pathlib import Path

import numpy as np
import pytest

from alchemypose.robot import export
from alchemypose.robot.retarget import Retargeter, RobotMotion
from alchemypose.robot.urdf import load_urdf

G1 = load_urdf(Path(__file__).resolve().parents[1] / "fixtures" / "unitree_g1" / "g1_29dof_rev_1_0.urdf")
RETARGET = Retargeter(G1, trajectory_sigma=0.0)


def _yaw(angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _motion(q: np.ndarray, yaw: float = 0.0, valid: np.ndarray | None = None, fps: float = 30.0) -> RobotMotion:
    count = len(q)
    return RobotMotion(
        joint_names=G1.joint_names, times=np.arange(count) / fps, q=q,
        root_rotation=np.tile(_yaw(yaw), (count, 1, 1)),
        root_height=np.array([RETARGET.sole_height(q_t, _yaw(yaw)) for q_t in q]),
        valid=np.ones(count, dtype=bool) if valid is None else valid,
        error=np.zeros(count), camera_to_world=np.eye(3),
    )


def test_an_upright_root_is_0001_in_unitree_order_and_1000_in_mujoco_order() -> None:
    reference = export.build_reference(_motion(np.zeros((10, 29))), RETARGET, root="fixed")
    assert export.unitree_rows(reference)[0, 3:7] == pytest.approx([0, 0, 0, 1])
    assert export.mujoco_qpos(reference)[0, 3:7] == pytest.approx([1, 0, 0, 0])


def test_a_quarter_turn_tells_the_two_quaternion_orders_apart() -> None:
    reference = export.build_reference(_motion(np.zeros((10, 29)), yaw=np.pi / 2), RETARGET, root="fixed")
    s = np.sqrt(0.5)
    assert export.unitree_rows(reference)[0, 3:7] == pytest.approx([0, 0, s, s])     # x y z w
    assert export.mujoco_qpos(reference)[0, 3:7] == pytest.approx([s, 0, 0, s])      # w x y z


def test_the_unitree_csv_has_36_columns_no_header_and_round_trips() -> None:
    rng = np.random.default_rng(0)
    q = rng.uniform(G1.lower, G1.upper, (12, 29)) * 0.2
    reference = export.build_reference(_motion(q, yaw=0.4), RETARGET, root="fixed")
    text = export.unitree_csv(reference)
    lines = text.strip().splitlines()
    assert len(lines) == reference.frames
    assert all(len(line.split(",")) == 36 for line in lines)
    assert not any(c.isalpha() for c in lines[0]), "no header row"
    back = export.from_unitree_rows(np.loadtxt(lines, delimiter=","), G1.joint_names)
    assert back.root_quat == pytest.approx(reference.root_quat, abs=1e-6)
    assert back.joint_pos == pytest.approx(reference.joint_pos, abs=1e-6)


def test_frames_before_and_after_the_person_are_trimmed_and_gaps_filled_and_flagged() -> None:
    q = np.zeros((20, 29))
    q[:, 3] = np.linspace(0, 1, 20)
    valid = np.ones(20, dtype=bool)
    valid[:3] = valid[-2:] = False                 # not there yet / gone
    valid[10] = False                              # one frame missing in the middle
    q[10, 3] = 9.0                                 # whatever the retarget held there
    reference = export.build_reference(_motion(q, valid=valid), RETARGET, root="fixed")
    assert reference.frames == 15
    assert reference.times[0] == 0.0
    assert reference.interpolated.tolist().count(True) == 1
    gap = int(np.flatnonzero(reference.interpolated)[0])
    neighbours = (reference.joint_pos[gap - 1, 3] + reference.joint_pos[gap + 1, 3]) / 2
    assert reference.joint_pos[gap, 3] == pytest.approx(neighbours)


def test_joint_velocities_integrate_back_to_the_joint_positions() -> None:
    t = np.arange(60) / 30.0
    q = np.zeros((60, 29))
    q[:, 3] = 0.5 + 0.4 * np.sin(2 * np.pi * 0.7 * t)
    reference = export.build_reference(_motion(q), RETARGET, root="fixed")
    integrated = reference.joint_pos[0, 3] + np.concatenate([[0.0], np.cumsum(
        (reference.joint_vel[1:, 3] + reference.joint_vel[:-1, 3]) / 2 / reference.fps)])
    assert integrated == pytest.approx(reference.joint_pos[:, 3], abs=0.02)


def test_a_fixed_root_never_travels() -> None:
    rng = np.random.default_rng(1)
    q = rng.uniform(G1.lower, G1.upper, (30, 29)) * 0.3
    reference = export.build_reference(_motion(q), RETARGET, root="fixed")
    assert np.all(reference.root_pos[:, :2] == 0.0)


def test_a_planted_root_stays_put_when_the_robot_stands_still() -> None:
    reference = export.build_reference(_motion(np.zeros((30, 29))), RETARGET, root="planted")
    assert np.abs(reference.root_pos[:, :2]).max() < 1e-9
    assert reference.contacts.all(), "both feet on the floor"


def test_a_planted_root_moves_opposite_to_a_sliding_stance_foot() -> None:
    """Swing the stance leg back from the hip: a planted foot means the body moves forward."""
    count = 30
    q = np.zeros((count, 29))
    q[:, G1.index("right_knee_joint")] = 1.6                     # lift the right foot well clear: left is stance
    q[:, G1.index("right_hip_pitch_joint")] = -0.9
    q[:, G1.index("left_hip_pitch_joint")] = np.linspace(0.0, 0.4, count)   # left leg swings back
    reference = export.build_reference(_motion(q), RETARGET, root="planted")
    assert reference.root_pos[-1, 0] > 0.05
    assert reference.contacts[:, 0].all() and not reference.contacts[:, 1].any()


def test_auto_slowdown_brings_the_fastest_joint_inside_the_budget() -> None:
    count = 31
    q = np.zeros((count, 29))
    knee = G1.index("left_knee_joint")
    q[:, knee] = np.linspace(0.0, 2.5, count)                     # 2.5 rad in 1 s
    limits = np.array([j.velocity for j in G1.actuated])
    budget = 1.25 / limits[knee]                                  # allow only 1.25 rad/s on that knee
    reference = export.build_reference(_motion(q), RETARGET, root="fixed", slowdown="auto", budget=budget)
    assert reference.metadata["slowdown"] == pytest.approx(2.0, rel=0.05)
    assert np.abs(reference.joint_vel[:, knee]).max() <= 1.25 * 1.05
    assert reference.times[-1] == pytest.approx(2.0, abs=0.05)


def test_the_npz_round_trips_with_names_and_conventions(tmp_path: Path) -> None:
    rng = np.random.default_rng(2)
    reference = export.build_reference(_motion(rng.uniform(G1.lower, G1.upper, (8, 29)) * 0.2), RETARGET)
    path = tmp_path / "m.npz"
    export.save_npz(reference, path)
    back = export.load_npz(path)
    assert back.joint_names == G1.joint_names
    assert back.joint_pos == pytest.approx(reference.joint_pos)
    assert back.metadata["quat_order"] == "wxyz"
    assert "never for direct playback on hardware" in back.metadata["kind"]
    with np.load(path) as raw:
        assert str(raw["quat_order"]) == "wxyz"


def test_the_beyondmimic_hint_carries_the_frame_rate() -> None:
    reference = export.build_reference(_motion(np.zeros((5, 29)), fps=25.0), RETARGET)
    assert "--input_fps 25" in export.beyondmimic_hint(Path("dance.csv"), reference)


def test_unitree_rows_of_the_wrong_width_are_refused() -> None:
    with pytest.raises(ValueError, match="36"):
        export.from_unitree_rows(np.zeros((3, 35)), G1.joint_names)


# ------------------------------------------------------- through MuJoCo itself


def test_mujoco_places_the_robot_exactly_where_this_package_does() -> None:
    """The end-to-end convention check: qpos order, quaternion order, joint order."""
    pytest.importorskip("mujoco")
    from alchemypose.robot import assets, simcheck

    if assets.missing():
        pytest.skip("G1 assets not cached (the MJCF needs its meshes)")
    rng = np.random.default_rng(3)
    # A continuous motion, as a clip is: the export low-passes the root height,
    # which is right for real motion and wrong for unrelated poses frame to frame.
    a, b = rng.uniform(G1.lower, G1.upper, (2, 29)) * 0.3
    q = a + np.linspace(0.0, 1.0, 20)[:, None] * (b - a)
    reference = export.build_reference(_motion(q, yaw=0.7), RETARGET, root="fixed")
    report = simcheck.check(reference, assets.g1_dir() / assets.MJCF, G1)
    assert report.fk_max_error_m < 1e-5
    assert report.range_violations == 0
    assert report.lowest_point_m > -0.03, "the soles are on the floor, not under it"
