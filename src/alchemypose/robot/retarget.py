"""Human 3D keypoints -> the G1's 29 joint angles, frame by frame.

**What is matched.** Not positions — a dancer and a 1.3 m robot have different
limbs — but *directions*. Each human chain is rebuilt on the robot's own link
lengths from the robot's own hips and shoulders: the robot's knee target is its
hip plus the human thigh's direction times the robot's thigh length, its ankle
target that knee target plus the human shin's direction times the robot's shin,
and so on out to the elbows, wrists and hands. Rigid end segments are matched
as frames, not as loose directions: the torso by its up and across axes, each
foot by its heel-to-toe line, so the robot is never asked for two foot points
its rigid foot cannot reach at once.

**Frames.** The human 3D is root-relative OpenCV camera space. It is turned into
a world frame — x forward, y left, z up, the robot's convention — by a proper
rotation (determinant +1; a reflection would match every direction and swap
left for right). "Up" is estimated once per clip as the median direction from
the ankles to the shoulders, because a low phone camera's "down" is not
gravity. The pelvis gets a frame each instant — z up, y along the hips, x
forward — whose heading is 1€-smoothed, because a 20 cm hip baseline is a noisy
compass. Joint angles are solved in that pelvis frame, so they do not depend on
which way the dancer faces; the facing goes into the root pose.

**Solving.** Per frame, a few iterations of damped least squares on the stacked
point errors, each step bounded and clamped to the joint limits, warm-started
from the previous frame. The Jacobian is exact (``axis x (p - o)`` per joint).
A target fades in with its keypoints' confidence rather than switching on at a
threshold, since a target that blinks on and off between frames is a jump. A
few joints are regularised where the targets leave them free — see
``JOINT_REGULARISATION`` and the shoulder-twist hold.

**What it is not.** Kinematics only: nothing here knows mass, balance, contact
forces or motor torque, and nothing here talks to a robot. A matched pose can
still be one the G1 cannot hold; :func:`speed_report` flags motion too fast for
its motors, and training a policy in simulation is what makes a motion
balanceable.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

from alchemypose.core.schema import PoseFrame
from alchemypose.core.smoothing import OneEuroFilter
from alchemypose.robot.urdf import Robot

F64 = NDArray[np.float64]
#: Confidences arrive as float32 from the engine; anything floating will do.
Scores = NDArray[np.floating[Any]]

#: COCO-WholeBody indices used here.
L_SHOULDER, R_SHOULDER, L_ELBOW, R_ELBOW, L_WRIST, R_WRIST = 5, 6, 7, 8, 9, 10
L_HIP, R_HIP, L_KNEE, R_KNEE, L_ANKLE, R_ANKLE = 11, 12, 13, 14, 15, 16
L_BIG_TOE, L_SMALL_TOE, L_HEEL, R_BIG_TOE, R_SMALL_TOE, R_HEEL = 17, 18, 19, 20, 21, 22
L_HAND_ROOT, L_MIDDLE_MCP, R_HAND_ROOT, R_MIDDLE_MCP = 91, 100, 112, 121

#: A target's weight ramps from 0 at ``SCORE_FLOOR`` to full at ``SCORE_FULL``
#: (the smallest of its keypoints' confidences). On the reference clip, a hard
#: 0.3 cut-off let the right arm's targets blink on and off at frames 176-177,
#: and the arm jumped each time.
SCORE_FLOOR = 0.2
SCORE_FULL = 0.4
#: The confidence below which a keypoint is not trusted at all (scale, up).
MIN_SCORE = 0.3

#: Robot points, as (link, offset in that link's frame). The foot and hand
#: offsets were read off the meshes' bounds: the G1's sole is 3.5 cm under the
#: ankle-roll frame, its toe 14.2 cm in front, its heel 6.6 cm behind. The
#: wrist point is the wrist-*pitch* joint centre, which is rigid on the
#: forearm; the wrist-yaw origin moves 2 cm when the wrist pitches.
G1_POINTS: dict[str, tuple[str, tuple[float, float, float]]] = {
    "l_knee": ("left_knee_link", (0.0, 0.0, 0.0)),
    "r_knee": ("right_knee_link", (0.0, 0.0, 0.0)),
    "l_ankle": ("left_ankle_roll_link", (0.0, 0.0, 0.0)),
    "r_ankle": ("right_ankle_roll_link", (0.0, 0.0, 0.0)),
    "l_toe": ("left_ankle_roll_link", (0.142, 0.0, -0.035)),
    "r_toe": ("right_ankle_roll_link", (0.142, 0.0, -0.035)),
    "l_heel": ("left_ankle_roll_link", (-0.066, 0.0, -0.035)),
    "r_heel": ("right_ankle_roll_link", (-0.066, 0.0, -0.035)),
    "l_shoulder": ("left_shoulder_pitch_link", (0.0, 0.0, 0.0)),
    "r_shoulder": ("right_shoulder_pitch_link", (0.0, 0.0, 0.0)),
    "l_elbow": ("left_elbow_link", (0.0, 0.0, 0.0)),
    "r_elbow": ("right_elbow_link", (0.0, 0.0, 0.0)),
    "l_wrist": ("left_wrist_pitch_link", (0.0, 0.0, 0.0)),
    "r_wrist": ("right_wrist_pitch_link", (0.0, 0.0, 0.0)),
    "l_hand": ("left_rubber_hand", (0.10, 0.0, 0.01)),
    "r_hand": ("right_rubber_hand", (0.10, 0.0, 0.01)),
}
#: Hip joint frames: where the robot's legs start.
_HIPS = {"l": "left_hip_pitch_link", "r": "right_hip_pitch_link"}
#: The pivot the torso turns about (the waist joints sit at the pelvis origin).
_WAIST = "waist_yaw_link"

#: Per-joint regularisation, by name fragment: (pull towards the previous
#: frame, pull towards zero), each on the scale of a joint's Jacobian term (a
#: 10 cm lever is about 0.01). The wrists' three joints chase one noisy hand
#: point, so the answer is not unique; left free, they flailed at 24 rad/s on
#: the reference clip. A gentle pull keeps the waist upright when the torso
#: targets are weak.
JOINT_REGULARISATION: dict[str, tuple[float, float]] = {
    "wrist": (0.02, 0.005),
    "waist": (0.0, 0.005),
}
#: Shoulder yaw is a twist about the upper arm, which only the elbow's bend
#: reveals: with the arm straight it moves nothing the targets can see. So it is
#: held to the previous frame in proportion to how straight the *human* arm is —
#: free when bent, firmly held as it straightens — rather than by a constant
#: pull that would freeze it when bent too.
TWIST_HOLD = 0.05
TWIST_FREE_BELOW_DEG = 140.0
TWIST_HELD_ABOVE_DEG = 170.0

#: How much each kind of target counts.
WEIGHTS = {
    "knee": 1.0, "ankle": 1.0, "toe": 0.5, "heel": 0.5, "shoulder": 1.5, "elbow": 1.0, "wrist": 1.0, "hand": 0.15,
}

#: The largest change any joint may make in one solver iteration, radians.
#: Twelve bounded steps a frame still allow 3 rad of motion per frame, and the
#: bound keeps a single iteration from leaping to a different arm configuration.
MAX_STEP = 0.25


#: Standard deviation of the zero-phase Gaussian applied to the joint
#: trajectories, in frames. 1.5 frames at 30 fps is 50 ms: it keeps movement up
#: to about 5 Hz — any dance — and removes what the keypoint smoothing cannot,
#: such as the reference clip's frames 174-177, where the tracked right elbow
#: folded from 138 to 71 degrees and back to 130 in a tenth of a second.
TRAJECTORY_SIGMA_FRAMES = 1.5


def _unit(v: F64) -> F64:
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-9 else np.zeros_like(v)


def _ramp(score: float) -> float:
    return float(np.clip((score - SCORE_FLOOR) / (SCORE_FULL - SCORE_FLOOR), 0.0, 1.0))


def _inner_angle(kp: F64, a: int, b: int, c: int) -> float:
    u, w = _unit(kp[a] - kp[b]), _unit(kp[c] - kp[b])
    return float(np.degrees(np.arccos(np.clip(u @ w, -1.0, 1.0))))


# ------------------------------------------------------------------ frames


def camera_to_world(up_cam: F64) -> F64:
    """The proper rotation taking OpenCV camera axes to x-forward, y-left, z-up.

    ``up_cam`` is the clip's estimated up direction in camera axes. Forward is
    *towards* the camera (a performer faces it), made perpendicular to up; left
    completes a right-handed frame. Rows are the world axes in camera terms, so
    ``world = R @ camera``.
    """
    up = _unit(np.asarray(up_cam, dtype=np.float64))
    towards = np.array([0.0, 0.0, -1.0])
    forward = _unit(towards - up * float(towards @ up))
    left = np.cross(up, forward)
    rotation = np.stack([forward, left, up])
    assert np.linalg.det(rotation) > 0.0
    return rotation


def estimate_up(frames: list[tuple[F64, Scores]]) -> F64:
    """The median ankles-to-shoulders direction over a clip, in camera axes.

    ``frames`` are ``(keypoints_3d, scores)`` pairs. A standing dancer's
    full-body line averages out to vertical even when a single frame leans;
    falls back to camera "up" (``-y``) when nothing qualifies.
    """
    directions = []
    for kp, scores in frames:
        needed = (L_SHOULDER, R_SHOULDER, L_ANKLE, R_ANKLE)
        if all(scores[i] >= MIN_SCORE for i in needed):
            line = (kp[L_SHOULDER] + kp[R_SHOULDER]) / 2 - (kp[L_ANKLE] + kp[R_ANKLE]) / 2
            if np.linalg.norm(line) > 1e-6:
                directions.append(_unit(line))
    if not directions:
        return np.array([0.0, -1.0, 0.0])
    return _unit(np.median(np.asarray(directions), axis=0))


def heading_of(world_kp: F64) -> float:
    """The pelvis heading (yaw, radians) from the hips: 0 when facing +x."""
    across = world_kp[L_HIP] - world_kp[R_HIP]
    return float(np.arctan2(-across[0], across[1]))


def pelvis_frame(world_kp: F64, heading: float | None = None) -> F64:
    """Rotation of the human pelvis in world axes: z up, y along the hips (left), x forward."""
    yaw = heading_of(world_kp) if heading is None else heading
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


# ---------------------------------------------------------------- the solve


@dataclass
class _Chain:
    """Robot rest geometry the targets are rebuilt from, measured once at q = 0."""

    hips: dict[str, F64]
    lengths: dict[str, float]
    torso_offsets: dict[str, F64]
    waist: F64
    foot: dict[str, F64]


def _rest(robot: Robot) -> _Chain:
    poses = robot.forward(np.zeros(len(robot.actuated)))

    def point(name: str) -> F64:
        link, offset = G1_POINTS[name]
        return (poses[link] @ np.append(offset, 1.0))[:3]

    hips = {s: poses[_HIPS[s]][:3, 3] for s in "lr"}
    lengths = {}
    for s in "lr":
        lengths[f"{s}_thigh"] = float(np.linalg.norm(point(f"{s}_knee") - hips[s]))
        lengths[f"{s}_shin"] = float(np.linalg.norm(point(f"{s}_ankle") - point(f"{s}_knee")))
        lengths[f"{s}_upper_arm"] = float(np.linalg.norm(point(f"{s}_elbow") - point(f"{s}_shoulder")))
        lengths[f"{s}_forearm"] = float(np.linalg.norm(point(f"{s}_wrist") - point(f"{s}_elbow")))
        lengths[f"{s}_hand"] = float(np.linalg.norm(point(f"{s}_hand") - point(f"{s}_wrist")))
    waist = poses[_WAIST][:3, 3]
    torso = {s: point(f"{s}_shoulder") - waist for s in "lr"}
    # At q = 0 the foot frame is the pelvis frame (x forward, z up), so these
    # offsets are already in foot-frame terms.
    foot = {f"{s}_{part}": point(f"{s}_{part}") - point(f"{s}_ankle") for s in "lr" for part in ("toe", "heel")}
    return _Chain(hips, lengths, torso, waist, foot)


@dataclass
class RobotMotion:
    """A clip as the robot would perform it."""

    joint_names: list[str]
    times: F64                 # (T,)
    q: F64                     # (T, J) radians, in ``joint_names`` order
    root_rotation: F64         # (T, 3, 3) pelvis in world, x forward / y left / z up
    root_height: F64           # (T,) pelvis height that puts the lowest sole on z = 0
    valid: NDArray[np.bool_]   # (T,) whether the frame had enough of a person to solve
    error: F64                 # (T,) RMS distance of matched points from their targets, metres
    camera_to_world: F64       # (3, 3)
    track_id: int = 0
    extras: dict[str, object] = field(default_factory=dict)

    @property
    def fps(self) -> float:
        if len(self.times) < 2:
            return 30.0
        return float((len(self.times) - 1) / max(self.times[-1] - self.times[0], 1e-9))


Targets = dict[str, tuple[F64, float]]


class Retargeter:
    """Turns a tracked person's 3D keypoints into the robot's joint angles."""

    def __init__(
        self,
        robot: Robot,
        iterations: int = 12,
        damping: float = 0.05,
        smoothness: float = 0.002,
        trajectory_sigma: float = TRAJECTORY_SIGMA_FRAMES,
    ) -> None:
        self.robot = robot
        self.trajectory_sigma = trajectory_sigma
        self.iterations = iterations
        self.damping = damping
        self.smoothness = smoothness
        self.rest = _rest(robot)
        self._points = {name: (link, np.asarray(off)) for name, (link, off) in G1_POINTS.items()}
        self.lower, self.upper = robot.lower.copy(), robot.upper.copy()
        # The G1's elbow travels *past* straight: its arm is straightest near
        # q = 1.43 and at the 2.09 limit the forearm bends backwards. A
        # hyperextended elbow plus a half-turn of shoulder yaw imitates an
        # ordinary bend; the solver found that mimic, stuck in it pinned at both
        # limits, and snapped out at 36 rad/s. Human elbows do not hyperextend,
        # so retargeting stops at straight.
        for side in ("left", "right"):
            joint = robot.index(f"{side}_elbow_joint")
            self.upper[joint] = min(self.upper[joint], self._straight_elbow(side))
        self._to_previous = np.full(len(robot.actuated), smoothness)
        self._to_zero = np.zeros(len(robot.actuated))
        for j, name in enumerate(robot.joint_names):
            for fragment, (previous, zero) in JOINT_REGULARISATION.items():
                if fragment in name:
                    self._to_previous[j] = max(smoothness, previous)
                    self._to_zero[j] = zero
        self._twist = {s: robot.index(f"{side}_shoulder_yaw_joint") for s, side in (("l", "left"), ("r", "right"))}

    def _straight_elbow(self, side: str) -> float:
        """The elbow angle at which this arm is straightest, found on the robot's own geometry."""
        joint = self.robot.index(f"{side}_elbow_joint")
        s = side[0]
        best, best_angle = self.robot.upper[joint], -1.0
        for value in np.linspace(self.robot.lower[joint], self.robot.upper[joint], 400):
            q = np.zeros(len(self.robot.actuated))
            q[joint] = value
            poses = self.robot.forward(q)
            points = {}
            for name in (f"{s}_shoulder", f"{s}_elbow", f"{s}_wrist"):
                link, offset = self._points[name]
                points[name] = (poses[link] @ np.append(offset, 1.0))[:3]
            upper = _unit(points[f"{s}_shoulder"] - points[f"{s}_elbow"])
            fore = _unit(points[f"{s}_wrist"] - points[f"{s}_elbow"])
            angle = float(np.arccos(np.clip(upper @ fore, -1.0, 1.0)))
            if angle > best_angle:
                best, best_angle = float(value), angle
        return best

    # ------------------------------------------------------------- targets

    def targets(self, kp: F64, scores: Scores, poses: dict[str, F64]) -> Targets:
        """Where each robot point should be (pelvis frame), and how much it counts.

        ``kp`` is the human, already in the pelvis frame. Arm targets start at
        the robot's *current* shoulders, so they follow the torso as it turns.
        """
        rest, out = self.rest, {}

        def weight(*indices: int) -> float:
            return min(_ramp(float(scores[i])) for i in indices)

        def direction(a: int, b: int) -> F64:
            return _unit(kp[b] - kp[a])

        for s, hip, knee, ankle, toes, heel in (
            ("l", L_HIP, L_KNEE, L_ANKLE, (L_BIG_TOE, L_SMALL_TOE), L_HEEL),
            ("r", R_HIP, R_KNEE, R_ANKLE, (R_BIG_TOE, R_SMALL_TOE), R_HEEL),
        ):
            w_thigh = weight(hip, knee)
            if w_thigh <= 0.0:
                continue
            knee_t = rest.hips[s] + direction(hip, knee) * rest.lengths[f"{s}_thigh"]
            out[f"{s}_knee"] = (knee_t, w_thigh)
            w_shin = min(w_thigh, weight(knee, ankle))
            if w_shin <= 0.0:
                continue
            ankle_t = knee_t + direction(knee, ankle) * rest.lengths[f"{s}_shin"]
            out[f"{s}_ankle"] = (ankle_t, w_shin)
            w_foot = min(w_shin, weight(heel, *toes))
            if w_foot <= 0.0:
                continue
            # One rigid foot frame from the heel-to-toe line, kept level across
            # (the human toe pair is too close together to set a roll).
            forward = _unit((kp[toes[0]] + kp[toes[1]]) / 2 - kp[heel])
            across = np.cross(np.array([0.0, 0.0, 1.0]), forward)
            if np.linalg.norm(across) < 1e-6:
                continue
            across = _unit(across)
            foot = np.stack([forward, across, np.cross(forward, across)], axis=1)
            for part in ("toe", "heel"):
                out[f"{s}_{part}"] = (ankle_t + foot @ rest.foot[f"{s}_{part}"], w_foot)

        w_torso = weight(L_SHOULDER, R_SHOULDER, L_HIP, R_HIP)
        if w_torso > 0.0:
            up = _unit((kp[L_SHOULDER] + kp[R_SHOULDER]) / 2 - (kp[L_HIP] + kp[R_HIP]) / 2)
            across = kp[L_SHOULDER] - kp[R_SHOULDER]
            left = _unit(across - up * float(across @ up))
            torso = np.stack([np.cross(left, up), left, up], axis=1)
            for s in "lr":
                out[f"{s}_shoulder"] = (rest.waist + torso @ rest.torso_offsets[s], w_torso)

        for s, shoulder, elbow, wrist, hand_root, middle in (
            ("l", L_SHOULDER, L_ELBOW, L_WRIST, L_HAND_ROOT, L_MIDDLE_MCP),
            ("r", R_SHOULDER, R_ELBOW, R_WRIST, R_HAND_ROOT, R_MIDDLE_MCP),
        ):
            w_upper = weight(shoulder, elbow)
            if w_upper <= 0.0:
                continue
            link, offset = self._points[f"{s}_shoulder"]
            current = (poses[link] @ np.append(offset, 1.0))[:3]
            elbow_t = current + direction(shoulder, elbow) * rest.lengths[f"{s}_upper_arm"]
            out[f"{s}_elbow"] = (elbow_t, w_upper)
            w_fore = min(w_upper, weight(elbow, wrist))
            if w_fore <= 0.0:
                continue
            wrist_t = elbow_t + direction(elbow, wrist) * rest.lengths[f"{s}_forearm"]
            out[f"{s}_wrist"] = (wrist_t, w_fore)
            w_hand = min(w_fore, weight(hand_root, middle))
            if w_hand > 0.0:
                out[f"{s}_hand"] = (wrist_t + direction(hand_root, middle) * rest.lengths[f"{s}_hand"], w_hand)
        return out

    def twist_hold(self, kp: F64, scores: Scores) -> F64:
        """Extra hold-previous weight per joint: shoulder yaw, as the human arm straightens."""
        hold = np.zeros(len(self.robot.actuated))
        for s, shoulder, elbow, wrist in (("l", L_SHOULDER, L_ELBOW, L_WRIST), ("r", R_SHOULDER, R_ELBOW, R_WRIST)):
            if min(scores[shoulder], scores[elbow], scores[wrist]) < SCORE_FLOOR:
                hold[self._twist[s]] = TWIST_HOLD                  # arm unseen: keep the twist
                continue
            angle = _inner_angle(kp, shoulder, elbow, wrist)
            straight = (angle - TWIST_FREE_BELOW_DEG) / (TWIST_HELD_ABOVE_DEG - TWIST_FREE_BELOW_DEG)
            hold[self._twist[s]] = TWIST_HOLD * float(np.clip(straight, 0.0, 1.0))
        return hold

    # ---------------------------------------------------------------- solve

    def solve(self, kp: F64, scores: Scores, q0: F64, iterations: int | None = None) -> tuple[F64, float]:
        """Joint angles for one frame (``kp`` in the pelvis frame), from ``q0``. Returns ``(q, rms error)``."""
        robot = self.robot
        q = np.clip(q0.copy(), self.lower, self.upper)
        to_previous = self._to_previous + self.twist_hold(kp, scores)
        error = 0.0
        for _ in range(iterations or self.iterations):
            poses, axes, origins = robot.forward_with_axes(q)
            goals = self.targets(kp, scores, poses)
            if not goals:
                break
            rows, residuals, weights = [], [], []
            for name, (goal, confidence) in goals.items():
                link, offset = self._points[name]
                p = (poses[link] @ np.append(offset, 1.0))[:3]
                jac = np.zeros((3, len(q)))
                for j in robot.ancestors[link]:
                    jac[:, j] = np.cross(axes[j], p - origins[j])
                rows.append(jac)
                residuals.append(goal - p)
                weights.append(np.full(3, WEIGHTS[name.split("_", 1)[1]] * confidence))
            jac_all = np.vstack(rows)
            res = np.concatenate(residuals)
            w = np.concatenate(weights)
            error = float(np.sqrt(np.mean(res**2)) * np.sqrt(3))
            jtw = jac_all.T * w
            lhs = jtw @ jac_all + np.diag(self.damping**2 + to_previous + self._to_zero)
            rhs = jtw @ res + to_previous * (q0 - q) - self._to_zero * q
            step = np.clip(np.linalg.solve(lhs, rhs), -MAX_STEP, MAX_STEP)
            q = np.clip(q + step, self.lower, self.upper)
        return q, error

    def sole_height(self, q: F64, rotation: F64) -> float:
        """Pelvis height that puts the lowest toe or heel on the floor."""
        poses = self.robot.forward(q)
        lowest = np.inf
        for name in ("l_toe", "l_heel", "r_toe", "r_heel"):
            link, offset = self._points[name]
            p = rotation @ (poses[link] @ np.append(offset, 1.0))[:3]
            lowest = min(lowest, float(p[2]))
        return -lowest

    def retarget(
        self,
        frames: tuple[PoseFrame, ...],
        track_id: int = 0,
        on_progress: Callable[[int, int], None] | None = None,
    ) -> RobotMotion:
        """Every frame of ``frames`` for ``track_id``. Frames without the person hold the last pose."""
        people = [next((p for p in f.people if p.track_id == track_id), None) for f in frames]
        samples = [
            (p.keypoints_3d.astype(np.float64), p.scores)
            for p in people
            if p is not None and p.keypoints_3d is not None
        ]
        to_world = camera_to_world(estimate_up(samples))
        times = np.array(
            [f.timestamp_s if f.timestamp_s is not None else float(i) for i, f in enumerate(frames)], dtype=np.float64
        )

        count, joints = len(frames), len(self.robot.actuated)
        q_all = np.zeros((count, joints))
        rotations = np.tile(np.eye(3), (count, 1, 1))
        heights = np.zeros(count)
        valid = np.zeros(count, dtype=bool)
        errors = np.full(count, np.nan)
        q = np.zeros(joints)
        rotation = np.eye(3)
        heading_filter = OneEuroFilter(min_cutoff=0.8, beta=0.3)
        unwrapped: float | None = None
        first = True
        for t, person in enumerate(people):
            if person is not None and person.keypoints_3d is not None:
                world = person.keypoints_3d.astype(np.float64) @ to_world.T
                raw = heading_of(world)
                if unwrapped is not None:                      # keep the heading continuous through ±pi
                    raw = unwrapped + (raw - unwrapped + np.pi) % (2 * np.pi) - np.pi
                unwrapped = float(heading_filter(np.array([raw]), float(times[t]))[0])
                rotation = pelvis_frame(world, unwrapped)
                hip_mid = (world[L_HIP] + world[R_HIP]) / 2
                local = (world - hip_mid) @ rotation          # into the pelvis frame
                q, errors[t] = self.solve(local, person.scores, q, iterations=40 if first else None)
                valid[t] = True
                first = False
            q_all[t] = q
            rotations[t] = rotation
            heights[t] = self.sole_height(q, rotation)
            if on_progress is not None:
                on_progress(t + 1, count)
        motion = RobotMotion(
            joint_names=self.robot.joint_names,
            times=times,
            q=q_all,
            root_rotation=rotations,
            root_height=heights,
            valid=valid,
            error=errors,
            camera_to_world=to_world,
            track_id=track_id,
        )
        if self.trajectory_sigma > 0.0:
            motion = smooth_motion(motion, self.lower, self.upper, self.trajectory_sigma)
            motion.root_height = np.array(
                [self.sole_height(q_t, r_t) for q_t, r_t in zip(motion.q, motion.root_rotation, strict=True)]
            )
        return motion


# ---------------------------------------------------------- trajectory smoothing

def smooth_motion(
    motion: RobotMotion, lower: F64, upper: F64, sigma_frames: float = TRAJECTORY_SIGMA_FRAMES
) -> RobotMotion:
    """``motion`` with each joint's trajectory low-passed forwards and backwards, then re-clamped.

    Zero-phase (a symmetric kernel), so a move is not delayed — the robot does
    not lag the dancer it sits beside. Only valid frames are smoothed; frames
    without the person keep the held pose. Root heights are recomputed by the
    caller if needed; this touches joint angles only.
    """
    if sigma_frames <= 0.0 or len(motion.q) < 3:
        return motion
    radius = max(1, int(round(3 * sigma_frames)))
    offsets = np.arange(-radius, radius + 1)
    kernel = np.exp(-0.5 * (offsets / sigma_frames) ** 2)
    kernel /= kernel.sum()
    padded = np.pad(motion.q, ((radius, radius), (0, 0)), mode="edge")
    smoothed = np.stack([np.convolve(padded[:, j], kernel, mode="valid") for j in range(motion.q.shape[1])], axis=1)
    q = np.where(motion.valid[:, None], np.clip(smoothed, lower, upper), motion.q)
    return RobotMotion(
        joint_names=motion.joint_names, times=motion.times, q=q, root_rotation=motion.root_rotation,
        root_height=motion.root_height, valid=motion.valid, error=motion.error,
        camera_to_world=motion.camera_to_world, track_id=motion.track_id,
        extras={**motion.extras, "trajectory_sigma_frames": sigma_frames},
    )


# ------------------------------------------------------------ speed and time


@dataclass(frozen=True)
class SpeedReport:
    """How hard a motion drives the motors, against the URDF's velocity limits."""

    peak_velocity: F64         # (J,) rad/s
    limit: F64                 # (J,) rad/s
    worst_joint: str
    worst_fraction: float      # peak / limit for the worst joint
    slowdown: float            # play at 1/slowdown to keep every joint within ``budget`` of its limit
    budget: float


def speed_report(motion: RobotMotion, limits: F64, budget: float = 0.5) -> SpeedReport:
    """Peak joint speeds of exactly this trajectory, and how much slower it must run to stay in budget.

    The URDF's ``velocity`` is the motor's ceiling, not a comfortable speed;
    ``budget`` is the fraction of it a motion may use (0.5 by default). The
    peak is the true maximum of the trajectory as it will be played or
    exported — not a percentile — so a fast moment is never averaged away.
    """
    q, t = motion.q, motion.times
    if len(t) < 3:
        zero = np.zeros(q.shape[1])
        return SpeedReport(zero, limits, motion.joint_names[0], 0.0, 1.0, budget)
    velocity = np.gradient(q, t, axis=0)
    velocity[~motion.valid] = 0.0
    peak = np.abs(velocity).max(axis=0)
    fraction = peak / limits
    worst = int(np.argmax(fraction))
    slowdown = max(1.0, float(fraction[worst]) / budget)
    return SpeedReport(peak, limits, motion.joint_names[worst], float(fraction[worst]), slowdown, budget)


def retime(motion: RobotMotion, slowdown: float) -> RobotMotion:
    """The same motion played ``slowdown`` times slower, resampled at the original frame rate.

    Joint angles are interpolated linearly between solved frames; root heights
    likewise; root rotations are taken from the nearest frame (they change
    slowly at a frame's scale and a nearest pick keeps them orthonormal).
    """
    if slowdown <= 1.0 + 1e-9:
        return motion
    t = motion.times - motion.times[0]
    fps = motion.fps
    new_t = np.arange(0.0, t[-1] * slowdown + 1e-9, 1.0 / fps)
    source = new_t / slowdown
    q = np.stack([np.interp(source, t, motion.q[:, j]) for j in range(motion.q.shape[1])], axis=1)
    nearest = np.clip(np.searchsorted(t, source), 0, len(t) - 1)
    return RobotMotion(
        joint_names=motion.joint_names,
        times=new_t + motion.times[0],
        q=q,
        root_rotation=motion.root_rotation[nearest],
        root_height=np.interp(source, t, motion.root_height),
        valid=motion.valid[nearest],
        error=motion.error[nearest],
        camera_to_world=motion.camera_to_world,
        track_id=motion.track_id,
        extras={**motion.extras, "slowdown": slowdown},
    )
