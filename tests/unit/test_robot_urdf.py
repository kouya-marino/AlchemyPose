"""The G1's kinematic tree and forward kinematics, against MuJoCo's own answer."""

import json
from pathlib import Path

import numpy as np
import pytest

from alchemypose.robot import assets
from alchemypose.robot.urdf import axis_angle, load_urdf, rpy_matrix

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "unitree_g1"
G1 = load_urdf(FIXTURES / "g1_29dof_rev_1_0.urdf")


def test_the_fixture_is_byte_for_byte_the_pinned_upstream_file() -> None:
    import hashlib

    digest = hashlib.sha256((FIXTURES / "g1_29dof_rev_1_0.urdf").read_bytes()).hexdigest()
    assert digest == assets.FILES[assets.URDF]


def test_the_g1_has_29_actuated_joints_in_unitree_motor_order() -> None:
    names = G1.joint_names
    assert len(names) == 29
    assert names[:6] == [
        "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
        "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    ]
    assert names[12:15] == ["waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint"]
    assert names[-1] == "right_wrist_yaw_joint"
    assert G1.root == "pelvis"


def test_joint_limits_are_read() -> None:
    knee = G1.actuated[G1.index("left_knee_joint")]
    assert (knee.lower, knee.upper) == pytest.approx((-0.087267, 2.8798))
    assert np.all(G1.lower < G1.upper)


def test_every_mesh_the_model_draws_is_one_the_asset_registry_fetches() -> None:
    meshes = {visual.mesh for visuals in G1.visuals.values() for visual in visuals}
    assert meshes == {name for name in assets.FILES if name.startswith("meshes/")}


def test_rpy_is_fixed_axis_roll_then_pitch_then_yaw() -> None:
    # A quarter yaw maps x to y; a quarter roll maps y to z; together, x -> y.
    assert rpy_matrix(0, 0, np.pi / 2) @ np.array([1, 0, 0]) == pytest.approx([0, 1, 0])
    assert rpy_matrix(np.pi / 2, 0, 0) @ np.array([0, 1, 0]) == pytest.approx([0, 0, 1])
    combined = rpy_matrix(0.3, -0.2, 1.1)
    assert combined == pytest.approx(rpy_matrix(0, 0, 1.1) @ rpy_matrix(0, -0.2, 0) @ rpy_matrix(0.3, 0, 0))


def test_axis_angle_is_a_rotation() -> None:
    rotation = axis_angle(np.array([1.0, 2.0, -0.5]), 0.7)
    assert rotation @ rotation.T == pytest.approx(np.eye(3))
    assert np.linalg.det(rotation) == pytest.approx(1.0)


@pytest.mark.parametrize("case", range(3))
def test_forward_kinematics_matches_mujoco_on_every_link(case: int) -> None:
    golden = json.loads((FIXTURES / "fk_golden.json").read_text())["cases"][case]
    q = np.array([golden["q"][name] for name in G1.joint_names])
    poses = G1.forward(q)
    assert set(golden["links"]) <= set(poses)
    for link, expected in golden["links"].items():
        assert poses[link][:3, 3] == pytest.approx(expected["pos"], abs=1e-7), link
        assert poses[link][:3, :3].reshape(-1) == pytest.approx(expected["rot"], abs=1e-7), link
