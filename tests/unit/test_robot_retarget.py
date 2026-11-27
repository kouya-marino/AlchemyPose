"""Retargeting human motion onto the G1: tests that can fail."""

from pathlib import Path

import numpy as np
import pytest

from alchemypose.robot import retarget as rt
from alchemypose.robot.retarget import (
    G1_POINTS,
    Retargeter,
    RobotMotion,
    camera_to_world,
    estimate_up,
    pelvis_frame,
    retime,
    speed_report,
)
from alchemypose.robot.urdf import load_urdf

G1 = load_urdf(Path(__file__).resolve().parents[1] / "fixtures" / "unitree_g1" / "g1_29dof_rev_1_0.urdf")
RETARGET = Retargeter(G1)
K = 133

#: Which human keypoint stands at which robot point, for building a human from the robot.
HUMAN_AT = {
    "l_knee": rt.L_KNEE, "r_knee": rt.R_KNEE, "l_ankle": rt.L_ANKLE, "r_ankle": rt.R_ANKLE,
    "l_shoulder": rt.L_SHOULDER, "r_shoulder": rt.R_SHOULDER, "l_elbow": rt.L_ELBOW, "r_elbow": rt.R_ELBOW,
    "l_wrist": rt.L_WRIST, "r_wrist": rt.R_WRIST, "l_heel": rt.L_HEEL, "r_heel": rt.R_HEEL,
}


def _point(poses: dict, name: str) -> np.ndarray:
    link, offset = G1_POINTS[name]
    return (poses[link] @ np.append(offset, 1.0))[:3]


def _human_from_robot(q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """A human whose every bone points the way the robot's does, in the pelvis frame.

    The exact inverse of how ``Retargeter.targets`` builds its goals: hips and
    shoulders at the robot's joints (so the torso frame they define is the
    robot's), and each limb chained from the robot's *anchor-relative* bone
    vectors — limbs start from anchors above the rest knee and elbow.
    """
    poses = G1.forward(q)
    rest = RETARGET.rest
    torso = poses["torso_link"][:3, :3]
    kp = np.zeros((K, 3))
    for s, hip, knee, ankle, toes, heel in (
        ("l", rt.L_HIP, rt.L_KNEE, rt.L_ANKLE, (rt.L_BIG_TOE, rt.L_SMALL_TOE), rt.L_HEEL),
        ("r", rt.R_HIP, rt.R_KNEE, rt.R_ANKLE, (rt.R_BIG_TOE, rt.R_SMALL_TOE), rt.R_HEEL),
    ):
        kp[hip] = poses[f"{'left' if s == 'l' else 'right'}_hip_pitch_link"][:3, 3]
        kp[knee] = kp[hip] + (_point(poses, f"{s}_knee") - rest.hips[s])
        kp[ankle] = kp[knee] + (_point(poses, f"{s}_ankle") - _point(poses, f"{s}_knee"))
        kp[toes[0]] = kp[toes[1]] = kp[ankle] + (_point(poses, f"{s}_toe") - _point(poses, f"{s}_ankle"))
        kp[heel] = kp[ankle] + (_point(poses, f"{s}_heel") - _point(poses, f"{s}_ankle"))
    for s, shoulder, elbow, wrist, root, middle in (
        ("l", rt.L_SHOULDER, rt.L_ELBOW, rt.L_WRIST, rt.L_HAND_ROOT, rt.L_MIDDLE_MCP),
        ("r", rt.R_SHOULDER, rt.R_ELBOW, rt.R_WRIST, rt.R_HAND_ROOT, rt.R_MIDDLE_MCP),
    ):
        kp[shoulder] = _point(poses, f"{s}_shoulder")
        anchor = kp[shoulder] + torso @ rest.arm_anchor[s]
        kp[elbow] = kp[shoulder] + (_point(poses, f"{s}_elbow") - anchor)
        kp[wrist] = kp[elbow] + (_point(poses, f"{s}_wrist") - _point(poses, f"{s}_elbow"))
        kp[root] = kp[wrist]
        kp[middle] = kp[wrist] + (_point(poses, f"{s}_hand") - _point(poses, f"{s}_wrist"))
    scores = np.zeros(K)
    scores[np.any(kp != 0.0, axis=1)] = 0.9
    return kp, scores


def _direction(poses: dict, a: str, b: str) -> np.ndarray:
    v = _point(poses, b) - _point(poses, a)
    return v / np.linalg.norm(v)


BONES = [("l_knee", "l_ankle"), ("r_knee", "r_ankle"), ("l_elbow", "l_wrist"), ("r_elbow", "r_wrist")]


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_a_pose_the_robot_can_make_is_recovered_from_its_own_keypoints(seed: int) -> None:
    rng = np.random.default_rng(seed)
    # Moderate poses: well inside the limits, waist and wrists left near neutral.
    q_true = rng.uniform(G1.lower, G1.upper) * 0.35
    for name in ("waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint"):
        q_true[G1.index(name)] = 0.0
    kp, scores = _human_from_robot(q_true)
    # Warm-started call after call, as a clip does: a held pose settles. One
    # cold solve from zeros would measure the regularisation, not the retarget.
    q = np.zeros(29)
    for _ in range(10):
        q, error = RETARGET.solve(kp, scores, q)
    # Not 0: the wrists are regularised towards neutral for stability on real
    # clips, and a random pose here has them away from it.
    assert error < 0.03, f"robot points miss their targets by {error * 100:.1f} cm"
    truth, found = G1.forward(q_true), G1.forward(q)
    for a, b in BONES:
        cosine = float(_direction(truth, a, b) @ _direction(found, a, b))
        assert cosine > np.cos(np.radians(10)), f"{a}->{b} off by {np.degrees(np.arccos(cosine)):.1f} degrees"


def test_every_solution_respects_the_joint_limits() -> None:
    rng = np.random.default_rng(9)
    kp = rng.normal(0, 0.5, (K, 3))                     # an impossible "pose"
    q, _ = RETARGET.solve(kp, np.full(K, 0.9), np.zeros(29), iterations=50)
    assert np.all(q >= G1.lower - 1e-9) and np.all(q <= G1.upper + 1e-9)


def _standing() -> tuple[np.ndarray, np.ndarray]:
    return _human_from_robot(np.zeros(29))


def test_a_bent_knee_bends_the_robots_knee_forward() -> None:
    q_bent = np.zeros(29)
    q_bent[G1.index("left_hip_pitch_joint")] = -0.6
    q_bent[G1.index("left_knee_joint")] = 1.2
    kp, scores = _human_from_robot(q_bent)
    q, _ = RETARGET.solve(kp, scores, np.zeros(29), iterations=150)
    assert q[G1.index("left_knee_joint")] > 0.8
    assert abs(q[G1.index("right_knee_joint")]) < 0.3


def test_an_arm_raised_forward_is_negative_shoulder_pitch() -> None:
    """Shoulder pitch turns about +y; swinging a hanging arm forward (+x) is a negative angle."""
    kp, scores = _standing()
    shoulder = kp[rt.L_SHOULDER]
    kp[rt.L_ELBOW] = shoulder + np.array([0.19, 0.0, 0.0])
    kp[rt.L_WRIST] = shoulder + np.array([0.37, 0.0, 0.0])
    scores[[rt.L_HAND_ROOT, rt.L_MIDDLE_MCP]] = 0.0
    q, _ = RETARGET.solve(kp, scores, np.zeros(29), iterations=150)
    assert q[G1.index("left_shoulder_pitch_joint")] < -0.8


def test_a_mirrored_human_gives_mirrored_joints() -> None:
    rng = np.random.default_rng(4)
    q_true = rng.uniform(G1.lower, G1.upper) * 0.3
    for name in ("waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint"):
        q_true[G1.index(name)] = 0.0
    kp, scores = _human_from_robot(q_true)
    mirrored = kp.copy()
    mirrored[:, 1] *= -1.0                               # reflect across the sagittal plane
    swap = {rt.L_SHOULDER: rt.R_SHOULDER, rt.L_ELBOW: rt.R_ELBOW, rt.L_WRIST: rt.R_WRIST, rt.L_HIP: rt.R_HIP,
            rt.L_KNEE: rt.R_KNEE, rt.L_ANKLE: rt.R_ANKLE, rt.L_BIG_TOE: rt.R_BIG_TOE,
            rt.L_SMALL_TOE: rt.R_SMALL_TOE, rt.L_HEEL: rt.R_HEEL, rt.L_HAND_ROOT: rt.R_HAND_ROOT,
            rt.L_MIDDLE_MCP: rt.R_MIDDLE_MCP}
    for a, b in list(swap.items()):
        mirrored[[a, b]] = mirrored[[b, a]]
    q, _ = RETARGET.solve(kp, scores, np.zeros(29), iterations=200)
    q_m, _ = RETARGET.solve(mirrored, scores, np.zeros(29), iterations=200)
    for joint in ("knee", "hip_pitch", "elbow", "shoulder_pitch"):
        left, right = G1.index(f"left_{joint}_joint"), G1.index(f"right_{joint}_joint")
        assert q_m[right] == pytest.approx(q[left], abs=0.15), joint
    for joint in ("hip_roll", "shoulder_roll"):                 # lateral joints flip sign
        left, right = G1.index(f"left_{joint}_joint"), G1.index(f"right_{joint}_joint")
        assert q_m[right] == pytest.approx(-q[left], abs=0.15), joint


# ------------------------------------------------------------------- frames


def test_camera_to_world_is_a_rotation_not_a_reflection() -> None:
    for up in ([0, -1, 0], [0, -0.98, -0.2], [0.1, -0.95, 0.1]):
        rotation = camera_to_world(np.array(up, dtype=float))
        assert rotation @ rotation.T == pytest.approx(np.eye(3), abs=1e-9)
        assert np.linalg.det(rotation) == pytest.approx(1.0)


def test_a_performer_facing_the_camera_has_their_left_on_camera_right() -> None:
    rotation = camera_to_world(np.array([0.0, -1.0, 0.0]))
    assert rotation @ np.array([1.0, 0.0, 0.0]) == pytest.approx([0, 1, 0])      # camera right -> left
    assert rotation @ np.array([0.0, 0.0, -1.0]) == pytest.approx([1, 0, 0])     # towards the camera -> forward
    assert rotation @ np.array([0.0, -1.0, 0.0]) == pytest.approx([0, 0, 1])     # camera up -> up


def test_up_is_estimated_from_the_body_not_the_camera() -> None:
    tilt = np.radians(12)
    up = np.array([0.0, -np.cos(tilt), -np.sin(tilt)])                     # a low camera looking up
    kp = np.zeros((K, 3))
    kp[[rt.L_ANKLE, rt.R_ANKLE]] = (0.1, 0.0, 0.0), (-0.1, 0.0, 0.0)
    kp[[rt.L_SHOULDER, rt.R_SHOULDER]] = kp[[rt.L_ANKLE, rt.R_ANKLE]] + up * 1.3
    estimated = estimate_up([(kp, np.full(K, 0.9))] * 5)
    assert estimated == pytest.approx(up, abs=1e-6)


def test_the_pelvis_frame_follows_the_hips_and_keeps_z_up() -> None:
    world = np.zeros((K, 3))
    world[rt.L_HIP], world[rt.R_HIP] = (0.0, 0.1, 0.9), (0.0, -0.1, 0.9)        # facing +x
    assert pelvis_frame(world) == pytest.approx(np.eye(3))
    world[rt.L_HIP], world[rt.R_HIP] = (-0.1, 0.0, 0.9), (0.1, 0.0, 0.9)       # turned to face +y
    frame = pelvis_frame(world)
    assert frame[:, 0] == pytest.approx([0, 1, 0])
    assert frame[:, 2] == pytest.approx([0, 0, 1])


# ------------------------------------------------------------ speed and time


def _motion(q: np.ndarray, fps: float = 30.0) -> RobotMotion:
    count = len(q)
    return RobotMotion(
        joint_names=G1.joint_names, times=np.arange(count) / fps, q=q,
        root_rotation=np.tile(np.eye(3), (count, 1, 1)), root_height=np.full(count, 0.8),
        valid=np.ones(count, dtype=bool), error=np.zeros(count), camera_to_world=np.eye(3),
    )


def test_the_speed_report_finds_the_fastest_joint_and_the_slowdown_it_needs() -> None:
    limits = np.full(29, 10.0)
    q = np.zeros((31, 29))
    q[:, 3] = np.linspace(0, 8.0, 31)                  # 8 rad over 1 s: 8 rad/s on joint 3
    report = speed_report(_motion(q), limits, budget=0.5)
    assert report.worst_joint == G1.joint_names[3]
    assert report.worst_fraction == pytest.approx(0.8)
    assert report.slowdown == pytest.approx(1.6)       # 8 rad/s must come down to 5


def test_a_gentle_motion_needs_no_slowdown() -> None:
    q = np.zeros((31, 29))
    q[:, 0] = np.linspace(0, 0.5, 31)
    assert speed_report(_motion(q), np.full(29, 10.0)).slowdown == 1.0


def test_retiming_stretches_the_clip_and_keeps_the_frame_rate() -> None:
    q = np.zeros((31, 29))
    q[:, 3] = np.linspace(0, 8.0, 31)
    slowed = retime(_motion(q), 2.0)
    assert slowed.fps == pytest.approx(30.0, rel=1e-3)
    assert slowed.times[-1] == pytest.approx(2.0)
    assert slowed.q[-1, 3] == pytest.approx(8.0)
    report = speed_report(slowed, np.full(29, 10.0), budget=0.5)
    assert report.worst_fraction == pytest.approx(0.4, rel=0.05)
    assert slowed.extras["slowdown"] == 2.0


def test_retargeting_never_hyperextends_the_elbow() -> None:
    """The G1's elbow travels past straight; a human's does not, so the retarget stops at straight.

    Without the cap the solver found a mimic — a backwards elbow plus a
    half-turn of shoulder yaw — got stuck in it, and snapped out at 36 rad/s.
    """
    for side in ("left", "right"):
        joint = G1.index(f"{side}_elbow_joint")
        cap = RETARGET.upper[joint]
        assert cap < G1.upper[joint] - 0.3
        s = side[0]

        def inner(value: float) -> float:
            q = np.zeros(29)
            q[joint] = value
            poses = G1.forward(q)
            upper = _point(poses, f"{s}_shoulder") - _point(poses, f"{s}_elbow")
            fore = _point(poses, f"{s}_wrist") - _point(poses, f"{s}_elbow")
            return float(np.degrees(np.arccos(upper @ fore / np.linalg.norm(upper) / np.linalg.norm(fore))))

        # The G1's offset arm is never perfectly straight; the cap is where it is straightest.
        assert inner(cap) > 160.0
        assert inner(cap) >= inner(cap - 0.05) and inner(cap) >= inner(cap + 0.05)
    straight_arm, scores = _standing()
    shoulder = straight_arm[rt.L_SHOULDER]
    straight_arm[rt.L_ELBOW] = shoulder + np.array([0.0, 0.0, -0.19])
    straight_arm[rt.L_WRIST] = shoulder + np.array([0.0, 0.0, -0.38])
    q, _ = RETARGET.solve(straight_arm, scores, np.zeros(29), iterations=100)
    assert q[G1.index("left_elbow_joint")] <= RETARGET.upper[G1.index("left_elbow_joint")] + 1e-9


def test_trajectory_smoothing_is_zero_phase_and_respects_limits() -> None:
    from alchemypose.robot.retarget import smooth_motion

    q = np.zeros((41, 29))
    q[20:, 3] = 1.0                                   # a step half-way through
    q[:, 0] = 50.0                                    # far outside any limit
    smoothed = smooth_motion(_motion(q), G1.lower, G1.upper, sigma_frames=1.5)
    assert smoothed.q[19, 3] + smoothed.q[20, 3] == pytest.approx(1.0, abs=0.15), "centred on the step: no lag"
    assert smoothed.q[0, 3] == 0.0 and smoothed.q[-1, 3] == pytest.approx(1.0)
    assert smoothed.q[:, 0].max() <= G1.upper[0] + 1e-9


def test_a_three_frame_tracking_glitch_is_flattened() -> None:
    from alchemypose.robot.retarget import smooth_motion

    q = np.zeros((41, 29))
    q[20, 18] = 1.2                                   # one-frame spike on an elbow
    smoothed = smooth_motion(_motion(q), G1.lower, G1.upper)
    assert smoothed.q[:, 18].max() < 0.4


def test_a_neutral_human_gives_the_robots_neutral_pose() -> None:
    """Legs and arms hanging straight down must not come out knock-kneed.

    The G1's rest thighs and upper arms slant outwards, so chains started at the
    hip and shoulder joints pulled a straight human limb inwards: ankles 0.11 m
    apart against 0.24 m at rest, ankle roll pinned at its limit. The round-trip
    test cannot see this — it builds its human from the robot. Seen to fail on
    the joint-anchored chains.
    """
    kp, scores = np.zeros((K, 3)), np.zeros(K)

    def put(index: int, point: tuple[float, float, float]) -> None:
        kp[index], scores[index] = point, 0.9

    sides = (
        (rt.L_HIP, rt.L_KNEE, rt.L_ANKLE, rt.L_HEEL, rt.L_BIG_TOE, rt.L_SMALL_TOE,
         rt.L_SHOULDER, rt.L_ELBOW, rt.L_WRIST, 1),
        (rt.R_HIP, rt.R_KNEE, rt.R_ANKLE, rt.R_HEEL, rt.R_BIG_TOE, rt.R_SMALL_TOE,
         rt.R_SHOULDER, rt.R_ELBOW, rt.R_WRIST, -1),
    )
    for hip, knee, ankle, heel, big, small, shoulder, elbow, wrist, side in sides:
        put(hip, (0, 0.09 * side, 0))
        put(knee, (0, 0.09 * side, -0.42))
        put(ankle, (0, 0.09 * side, -0.84))
        put(heel, (-0.05, 0.09 * side, -0.9))
        put(big, (0.15, 0.07 * side, -0.9))
        put(small, (0.15, 0.11 * side, -0.9))
        put(shoulder, (0, 0.18 * side, 0.5))
        put(elbow, (0, 0.18 * side, 0.22))
        put(wrist, (0, 0.18 * side, -0.02))
    q = np.zeros(29)
    for _ in range(10):
        q, _error = RETARGET.solve(kp, scores, q)
    for joint in ("hip_roll", "ankle_roll", "shoulder_roll"):
        for side in ("left", "right"):
            assert abs(q[G1.index(f"{side}_{joint}_joint")]) < 0.03, f"{side} {joint}"
    poses = G1.forward(q)
    rest = G1.forward(np.zeros(29))
    separation = np.linalg.norm(_point(poses, "l_ankle") - _point(poses, "r_ankle"))
    assert separation == pytest.approx(np.linalg.norm(_point(rest, "l_ankle") - _point(rest, "r_ankle")), abs=0.01)
