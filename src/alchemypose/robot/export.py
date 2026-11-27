"""A retargeted clip as reference motion a G1 policy can be trained to track.

This is the file a trainer reads, so every convention in it is stated rather
than assumed, and the formats that differ are converted explicitly:

* **Native ``.npz``** (:func:`save_npz`) — keyed by name, never by position:
  ``joint_names`` beside ``joint_pos``, ``quat_order`` beside ``root_quat``.
* **Unitree's CSV** (:func:`unitree_csv`) — the layout of Unitree's own
  retargeted G1 motion set, which BeyondMimic also consumes: one row per frame,
  no header, 36 columns — root position xyz, root quaternion **x y z w** (scalar
  last), then the 29 joints in the URDF's order. Read off Unitree's published
  data, not written from memory: its first column 6 is 0.9998, the w of an
  upright pose.
* **MuJoCo ``qpos``** (:func:`mujoco_qpos`) — the same 36 numbers with the
  quaternion **w x y z** (scalar first), MuJoCo's free-joint convention, which is
  what Unitree's own ``g1_29dof_rev_1_0.xml`` loads (its default ``qpos[:7]`` is
  ``0 0 0.793 1 0 0 0``).

**What is measured and what is estimated.** The joint angles come from the
retargeting. The root is the weak part of any single-camera reference: the 3D
keypoints are hip-relative, so there is no world position to read. The root
*height* puts the lowest sole on the floor; the root *orientation* is the
pelvis heading only (yaw — a monocular pelvis has no trustworthy pitch or
roll); and the root *travel* is either held fixed (``root="fixed"``) or
estimated by keeping the stance foot planted (``root="planted"``), which
cannot represent a jump. Both are labelled as estimates in the file.

This is a kinematic reference, not a controller. It is for training and
simulation; streaming it to a robot's motors would ignore balance entirely.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

from alchemypose import __version__
from alchemypose.robot.retarget import TRAJECTORY_SIGMA_FRAMES, Retargeter, RobotMotion, speed_report

F64 = NDArray[np.float64]
RootMode = Literal["fixed", "planted"]

#: A foot is in contact when its sole is within this of the floor, metres.
CONTACT_HEIGHT = 0.02
#: The stance foot changes only when the other one is lower by more than this.
STANCE_HYSTERESIS = 0.01
#: Low-pass on the planted root's horizontal path, frames. Holding the stance
#: foot still turns the legs' frame-to-frame noise into root travel: on the
#: reference clip, a median of 1 cm per frame — 30 cm/s of motion that is not
#: there. 4 frames (130 ms at 30 fps) removes that and keeps real steps.
ROOT_PATH_SIGMA_FRAMES = 4.0

_FEET = {"left": ("l_toe", "l_heel"), "right": ("r_toe", "r_heel")}


@dataclass
class ReferenceMotion:
    """Everything a tracking trainer needs, per frame, plus how it was made."""

    joint_names: list[str]
    fps: float
    times: F64                 # (T,) seconds from the first frame
    root_pos: F64              # (T, 3) metres, world: x forward, y left, z up
    root_quat: F64             # (T, 4) w x y z
    joint_pos: F64             # (T, 29) radians, ``joint_names`` order
    joint_vel: F64             # (T, 29) rad/s
    root_lin_vel: F64          # (T, 3) m/s, world
    root_ang_vel: F64          # (T, 3) rad/s, world
    contacts: NDArray[np.bool_]  # (T, 2) left foot, right foot
    interpolated: NDArray[np.bool_]  # (T,) frames with no person, filled between neighbours
    error: F64                 # (T,) retarget miss, metres (NaN where interpolated)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def frames(self) -> int:
        return int(self.joint_pos.shape[0])


# --------------------------------------------------------------- building it


def _heading(rotation: F64) -> float:
    return float(np.arctan2(rotation[1, 0], rotation[0, 0]))


def _yaw_quat(yaw: F64) -> F64:
    """w x y z for rotations about z; continuous because ``yaw`` is unwrapped."""
    return np.column_stack([np.cos(yaw / 2), np.zeros_like(yaw), np.zeros_like(yaw), np.sin(yaw / 2)])


def _lowpass(values: F64, sigma: float) -> F64:
    if sigma <= 0.0 or len(values) < 3:
        return values
    radius = max(1, int(round(3 * sigma)))
    offsets = np.arange(-radius, radius + 1)
    kernel = np.exp(-0.5 * (offsets / sigma) ** 2)
    kernel /= kernel.sum()
    flat = values.reshape(len(values), -1)
    padded = np.pad(flat, ((radius, radius), (0, 0)), mode="edge")
    out = np.stack([np.convolve(padded[:, j], kernel, mode="valid") for j in range(flat.shape[1])], axis=1)
    return out.reshape(values.shape)


def _trim_and_fill(motion: RobotMotion) -> tuple[RobotMotion, NDArray[np.bool_]]:
    """Cut to the first and last frame with the person; interpolate frames missing between."""
    valid = np.flatnonzero(motion.valid)
    if valid.size == 0:
        raise ValueError("the person was never seen; there is no motion to export")
    first, last = int(valid[0]), int(valid[-1]) + 1
    keep = slice(first, last)
    times = motion.times[keep]
    ok = motion.valid[keep]
    q = motion.q[keep].copy()
    if not ok.all():
        for j in range(q.shape[1]):
            q[~ok, j] = np.interp(times[~ok], times[ok], q[ok, j])
    trimmed = RobotMotion(
        joint_names=motion.joint_names, times=times, q=q, root_rotation=motion.root_rotation[keep],
        root_height=motion.root_height[keep], valid=ok, error=motion.error[keep],
        camera_to_world=motion.camera_to_world, track_id=motion.track_id, extras=dict(motion.extras),
    )
    return trimmed, ~ok


def _soles(retargeter: Retargeter, q: F64, yaw: float, height: float) -> dict[str, F64]:
    """Each foot's sole midpoint in the world, with the root at the origin's xy."""
    c, s = np.cos(yaw), np.sin(yaw)
    base = np.eye(4)
    base[:3, :3] = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    base[2, 3] = height
    poses = retargeter.robot.forward(q, base)
    out = {}
    for foot, names in _FEET.items():
        points = []
        for name in names:
            link, offset = retargeter._points[name]
            points.append((poses[link] @ np.append(offset, 1.0))[:3])
        out[foot] = np.mean(points, axis=0)
    return out


def build_reference(
    motion: RobotMotion,
    retargeter: Retargeter,
    *,
    root: RootMode = "planted",
    slowdown: float | Literal["auto"] = 1.0,
    budget: float = 0.5,
    source: str = "",
) -> ReferenceMotion:
    """Turn a retargeted clip into a training reference.

    ``slowdown`` plays the motion that many times slower; ``"auto"`` picks the
    smallest one that keeps every joint within ``budget`` of its rated speed.
    """
    trimmed, interpolated = _trim_and_fill(motion)
    yaw = np.unwrap(np.array([_heading(r) for r in trimmed.root_rotation]))
    height = _lowpass(trimmed.root_height.astype(np.float64), TRAJECTORY_SIGMA_FRAMES)

    limits = np.array([j.velocity for j in retargeter.robot.actuated])
    factor = speed_report(trimmed, limits, budget).slowdown if slowdown == "auto" else float(slowdown)
    factor = max(1.0, factor)

    t = trimmed.times - trimmed.times[0]
    fps = trimmed.fps
    if factor > 1.0 + 1e-9:
        new_t = np.arange(0.0, t[-1] * factor + 1e-9, 1.0 / fps)
        src = new_t / factor
        q = np.stack([np.interp(src, t, trimmed.q[:, j]) for j in range(trimmed.q.shape[1])], axis=1)
        yaw = np.interp(src, t, yaw)
        height = np.interp(src, t, height)
        nearest = np.clip(np.round(src * fps).astype(int), 0, len(t) - 1)
        interpolated = interpolated[nearest]
        error = trimmed.error[nearest]
        t = new_t
    else:
        q, error = trimmed.q, trimmed.error

    count = len(t)
    soles = [_soles(retargeter, q[i], float(yaw[i]), float(height[i])) for i in range(count)]
    root_xy = np.zeros((count, 2))
    contacts = np.zeros((count, 2), dtype=bool)
    stance = "left" if soles[0]["left"][2] <= soles[0]["right"][2] else "right"
    for i in range(count):
        other = "right" if stance == "left" else "left"
        if soles[i][other][2] < soles[i][stance][2] - STANCE_HYSTERESIS:
            stance = other
        if root == "planted" and i > 0:
            # Keep the stance foot where it was: whatever it moved relative to
            # the root, the root moves the other way.
            moved = soles[i][stance][:2] - soles[i - 1][stance][:2]
            root_xy[i] = root_xy[i - 1] - moved
        elif i > 0:
            root_xy[i] = root_xy[i - 1]
        for k, foot in enumerate(("left", "right")):
            contacts[i, k] = foot == stance or soles[i][foot][2] <= CONTACT_HEIGHT

    if root == "planted":
        root_xy = _lowpass(root_xy, ROOT_PATH_SIGMA_FRAMES)
    root_pos = np.column_stack([root_xy, height])
    dt = 1.0 / fps
    joint_vel = np.gradient(q, dt, axis=0) if count > 1 else np.zeros_like(q)
    root_lin_vel = np.gradient(root_pos, dt, axis=0) if count > 1 else np.zeros_like(root_pos)
    yaw_rate = np.gradient(yaw, dt) if count > 1 else np.zeros(count)
    root_ang_vel = np.column_stack([np.zeros(count), np.zeros(count), yaw_rate])

    from alchemypose.robot import assets

    exported = RobotMotion(
        joint_names=trimmed.joint_names, times=t, q=q, root_rotation=np.tile(np.eye(3), (count, 1, 1)),
        root_height=height, valid=~interpolated, error=error, camera_to_world=trimmed.camera_to_world,
    )
    report = speed_report(exported, limits, budget)
    metadata = {
        "generator": f"alchemypose {__version__}",
        "kind": "kinematic reference motion — for training and simulation, never for direct playback on hardware",
        "robot": "unitree_g1_29dof_rev_1_0",
        "urdf": f"unitreerobotics/unitree_ros@{assets.COMMIT}/robots/g1_description/{assets.URDF}",
        "urdf_sha256": assets.FILES[assets.URDF],
        "units": {"position": "m", "angle": "rad", "time": "s"},
        "frame": "world, x forward, y left, z up; root is the pelvis link",
        "quat_order": "wxyz",
        "root_orientation": "estimated: pelvis heading (yaw) only; pitch and roll are zero",
        "root_height": "estimated: lowest sole on the floor (z = 0), low-passed",
        "root_travel": (
            "estimated: stance foot held planted, path low-passed; cannot represent a jump"
            if root == "planted" else "fixed: no horizontal travel"
        ),
        "root_mode": root,
        "slowdown": factor,
        "speed_budget": budget,
        "peak_speed_fraction": report.worst_fraction,
        "peak_speed_joint": report.worst_joint,
        "trajectory_sigma_frames": TRAJECTORY_SIGMA_FRAMES,
        "track_id": motion.track_id,
        "source": source,
        "interpolated_frames": int(interpolated.sum()),
        "contact_height_m": CONTACT_HEIGHT,
    }
    return ReferenceMotion(
        joint_names=list(trimmed.joint_names), fps=fps, times=t, root_pos=root_pos, root_quat=_yaw_quat(yaw),
        joint_pos=q, joint_vel=joint_vel, root_lin_vel=root_lin_vel, root_ang_vel=root_ang_vel,
        contacts=contacts, interpolated=interpolated, error=error, metadata=metadata,
    )


# ---------------------------------------------------------------- the formats


def save_npz(reference: ReferenceMotion, path: Path) -> None:
    """The native format: every array named, conventions written beside them."""
    np.savez_compressed(
        path,
        joint_names=np.array(reference.joint_names),
        fps=np.array([reference.fps]),
        times=reference.times,
        root_pos=reference.root_pos,
        root_quat=reference.root_quat,
        quat_order=np.array("wxyz"),
        joint_pos=reference.joint_pos,
        joint_vel=reference.joint_vel,
        root_lin_vel=reference.root_lin_vel,
        root_ang_vel=reference.root_ang_vel,
        contact_names=np.array(["left_foot", "right_foot"]),
        contacts=reference.contacts,
        interpolated=reference.interpolated,
        error=reference.error,
        metadata=np.array(json.dumps(reference.metadata, ensure_ascii=False)),
    )


def load_npz(path: Path) -> ReferenceMotion:
    with np.load(path, allow_pickle=False) as data:
        if str(data["quat_order"]) != "wxyz":
            raise ValueError(f"{path} stores quaternions as {data['quat_order']}, expected wxyz")
        return ReferenceMotion(
            joint_names=[str(n) for n in data["joint_names"]], fps=float(data["fps"][0]), times=data["times"],
            root_pos=data["root_pos"], root_quat=data["root_quat"], joint_pos=data["joint_pos"],
            joint_vel=data["joint_vel"], root_lin_vel=data["root_lin_vel"], root_ang_vel=data["root_ang_vel"],
            contacts=data["contacts"], interpolated=data["interpolated"], error=data["error"],
            metadata=json.loads(str(data["metadata"])),
        )


def mujoco_qpos(reference: ReferenceMotion) -> F64:
    """``(T, 36)``: root xyz, quaternion w x y z, 29 joints — MuJoCo's free-joint layout."""
    return np.column_stack([reference.root_pos, reference.root_quat, reference.joint_pos])


def unitree_rows(reference: ReferenceMotion) -> F64:
    """``(T, 36)``: root xyz, quaternion **x y z w**, 29 joints — Unitree's CSV layout."""
    w, x, y, z = reference.root_quat.T
    return np.column_stack([reference.root_pos, x, y, z, w, reference.joint_pos])


def from_unitree_rows(rows: F64, joint_names: list[str], fps: float = 30.0) -> ReferenceMotion:
    """Read Unitree's 36-column layout back (quaternion x y z w) into a reference — e.g. their own data."""
    rows = np.asarray(rows, dtype=np.float64)
    if rows.ndim != 2 or rows.shape[1] != 7 + len(joint_names):
        raise ValueError(f"expected {7 + len(joint_names)} columns, got {rows.shape}")
    count = len(rows)
    x, y, z, w = rows[:, 3], rows[:, 4], rows[:, 5], rows[:, 6]
    zeros3 = np.zeros((count, 3))
    return ReferenceMotion(
        joint_names=list(joint_names), fps=fps, times=np.arange(count) / fps, root_pos=rows[:, :3],
        root_quat=np.column_stack([w, x, y, z]), joint_pos=rows[:, 7:], joint_vel=np.zeros((count, len(joint_names))),
        root_lin_vel=zeros3, root_ang_vel=zeros3, contacts=np.zeros((count, 2), dtype=bool),
        interpolated=np.zeros(count, dtype=bool), error=np.full(count, np.nan), metadata={"source": "unitree rows"},
    )


def unitree_csv(reference: ReferenceMotion) -> str:
    """Unitree's retargeted-motion CSV: no header, one row per frame, at ``reference.fps``."""
    rows = unitree_rows(reference)
    return "\n".join(",".join(f"{v:.6f}" for v in row) for row in rows) + "\n"


def beyondmimic_hint(csv_path: Path, reference: ReferenceMotion) -> str:
    """The command BeyondMimic's own converter takes for this CSV. The CSV carries no frame rate; this does."""
    return (
        "python scripts/csv_to_npz.py "
        f"--input_file {csv_path} --input_fps {reference.fps:g} --output_name {csv_path.stem} --headless"
    )
