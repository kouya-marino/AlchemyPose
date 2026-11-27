"""Play a reference motion through Unitree's own MuJoCo model, kinematically, and check it.

The ``sim`` extra (``pip install "alchemypose[sim]"``, which is MuJoCo). This is
the check to run before handing a file to a trainer: it loads the very MJCF the
trainers use (``g1_29dof_rev_1_0.xml``, from the same pinned commit), writes
each frame into ``qpos`` exactly as the export lays it out, and reports:

* whether MuJoCo's link positions agree with this package's forward kinematics
  — the test that the quaternion order and the joint order really are what the
  file says;
* how far anything goes below the floor;
* which frames have the robot's own parts touching (self-collision);
* any joint outside the MJCF's own ranges.

Kinematics only — ``mj_kinematics`` and collision detection, never
``mj_step``. Whether the motion can be *balanced* is the trainer's question.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from alchemypose.robot.export import ReferenceMotion, mujoco_qpos
from alchemypose.robot.urdf import Robot

F64 = NDArray[np.float64]


@dataclass(frozen=True)
class SimReport:
    frames: int
    fk_max_error_m: float            # MuJoCo body positions vs this package's FK
    lowest_point_m: float            # lowest geom AABB bottom over the clip; < 0 is under the floor
    frames_below_floor: int          # frames with any geom more than 1 cm under z = 0
    self_contact_frames: int         # frames where two robot geoms touch
    range_violations: int            # joint samples outside the MJCF's ranges
    worst_self_contact: tuple[str, str] | None

    def summary(self) -> str:
        example = ""
        if self.worst_self_contact:
            example = f" (e.g. {self.worst_self_contact[0]} / {self.worst_self_contact[1]})"
        lines = [
            f"{self.frames} frames played through MuJoCo (kinematics only)",
            f"  link positions vs this package's FK: worst {self.fk_max_error_m * 1000:.3f} mm",
            f"  lowest point: {self.lowest_point_m * 100:+.1f} cm; "
            f"frames >1 cm under the floor: {self.frames_below_floor}",
            f"  frames with self-contact: {self.self_contact_frames}{example}",
            f"  joint samples outside the MJCF's ranges: {self.range_violations}",
        ]
        return "\n".join(lines)


def _lowest_vertex(model: Any, data: Any) -> float:
    """The lowest mesh vertex in the world. Bounding spheres would overstate a dip by centimetres."""
    import mujoco

    m, d = model, data
    lowest = np.inf
    near = d.geom_xpos[:, 2] - m.geom_rbound < 0.05          # only geoms that could reach the floor
    for g in np.flatnonzero(near & (m.geom_type == mujoco.mjtGeom.mjGEOM_MESH)):
        mesh = m.geom_dataid[g]
        start, count = m.mesh_vertadr[mesh], m.mesh_vertnum[mesh]
        vertices = m.mesh_vert[start : start + count]
        z = vertices @ d.geom_xmat[g].reshape(3, 3)[2] + d.geom_xpos[g, 2]
        lowest = min(lowest, float(z.min()))
    if not np.isfinite(lowest):
        lowest = float((d.geom_xpos[:, 2] - m.geom_rbound).min())
    return lowest


def check(reference: ReferenceMotion, mjcf: Path, robot: Robot) -> SimReport:
    import mujoco

    model = mujoco.MjModel.from_xml_path(str(mjcf))
    data = mujoco.MjData(model)
    names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(model.njnt)]
    if model.jnt_type[0] != mujoco.mjtJoint.mjJNT_FREE or names[1:] != reference.joint_names:
        raise ValueError("the MJCF's joints are not a free base followed by the reference's joints, in order")
    qpos = mujoco_qpos(reference)
    ranges = model.jnt_range[1:]
    limited = model.jnt_limited[1:].astype(bool)
    bodies = {
        name: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        for name in ("pelvis", "left_ankle_roll_link", "right_ankle_roll_link", "left_wrist_yaw_link",
                     "right_wrist_yaw_link", "torso_link")
    }
    worst_fk = 0.0
    lowest = np.inf
    below = contacts = violations = 0
    worst_pair: tuple[str, str] | None = None
    for t in range(len(qpos)):
        data.qpos[:] = qpos[t]
        mujoco.mj_kinematics(model, data)
        mujoco.mj_collision(model, data)
        # This package's FK, with the same root pose.
        w, x, y, z = reference.root_quat[t]
        base = np.eye(4)
        base[:3, :3] = np.array([
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ])
        base[:3, 3] = reference.root_pos[t]
        ours = robot.forward(reference.joint_pos[t], base)
        for name, body in bodies.items():
            if body >= 0:
                worst_fk = max(worst_fk, float(np.abs(data.xpos[body] - ours[name][:3, 3]).max()))
        frame_low = _lowest_vertex(model, data)
        lowest = min(lowest, frame_low)
        below += int(frame_low < -0.01)
        touching = False
        for c in range(data.ncon):
            contact = data.contact[c]
            b1, b2 = model.geom_bodyid[contact.geom1], model.geom_bodyid[contact.geom2]
            if b1 == 0 or b2 == 0 or contact.dist > 0:
                continue
            touching = True
            worst_pair = (
                str(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b1)),
                str(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b2)),
            )
        contacts += int(touching)
        q = qpos[t, 7:]
        violations += int(np.sum(limited & ((q < ranges[:, 0] - 1e-6) | (q > ranges[:, 1] + 1e-6))))
    return SimReport(
        frames=len(qpos), fk_max_error_m=worst_fk, lowest_point_m=lowest, frames_below_floor=below,
        self_contact_frames=contacts, range_violations=violations, worst_self_contact=worst_pair,
    )
