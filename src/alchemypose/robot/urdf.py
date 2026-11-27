"""A URDF read into plain data, and forward kinematics over it, in numpy.

Only what drawing and retargeting need: the kinematic tree (links, joints,
origins, axes, limits) and each link's visual mesh. No dynamics, no
collision geometry. ``xml.etree`` reads it; nothing executes.

Conventions, as the URDF specification defines them and as MuJoCo's own URDF
import agrees (checked against it on random configurations — see the tests):

* A joint's ``origin`` places its child frame in its parent's: translation
  ``xyz``, rotation ``rpy`` = roll about x, then pitch about y, then yaw about z,
  all about the *fixed* axes, so ``R = Rz(yaw) @ Ry(pitch) @ Rx(roll)``.
* A revolute joint then rotates the child about ``axis`` (in the child's
  frame, after the origin) by ``q``: ``T_child = T_parent @ T_origin @ Rot(axis, q)``.
* Robot axes: x forward, y left, z up, metres.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

F64 = NDArray[np.float64]


def rpy_matrix(roll: float, pitch: float, yaw: float) -> F64:
    cr, sr, cp, sp, cy, sy = np.cos(roll), np.sin(roll), np.cos(pitch), np.sin(pitch), np.cos(yaw), np.sin(yaw)
    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return np.asarray(rz @ ry @ rx, dtype=np.float64)


def axis_angle(axis: F64, angle: float) -> F64:
    """Rodrigues: rotation by ``angle`` about the unit vector ``axis``."""
    x, y, z = axis / np.linalg.norm(axis)
    c, s = np.cos(angle), np.sin(angle)
    k = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
    return np.asarray(np.eye(3) + s * k + (1 - c) * (k @ k), dtype=np.float64)


def transform(rotation: F64, translation: F64) -> F64:
    out = np.eye(4)
    out[:3, :3] = rotation
    out[:3, 3] = translation
    return out


@dataclass(frozen=True)
class Visual:
    mesh: str                 # path relative to the URDF's folder
    origin: F64               # 4x4, in the link's frame
    rgba: tuple[float, float, float, float] = (0.7, 0.7, 0.72, 1.0)


@dataclass(frozen=True)
class Joint:
    name: str
    kind: str                 # revolute, continuous, prismatic, fixed
    parent: str
    child: str
    origin: F64               # 4x4
    axis: F64                 # unit 3-vector
    lower: float = -np.inf
    upper: float = np.inf
    velocity: float = np.inf  # rad/s

    @property
    def actuated(self) -> bool:
        return self.kind in ("revolute", "continuous", "prismatic")


@dataclass
class Robot:
    name: str
    root: str
    joints: list[Joint]
    visuals: dict[str, list[Visual]]
    #: Actuated joints in document order — for Unitree's files, the SDK's motor order.
    actuated: list[Joint] = field(default_factory=list)
    _children: dict[str, list[Joint]] = field(default_factory=dict)
    #: For each link, the actuated-joint indices on its path from the root.
    ancestors: dict[str, tuple[int, ...]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.actuated = [j for j in self.joints if j.actuated]
        self._children = {}
        for joint in self.joints:
            self._children.setdefault(joint.parent, []).append(joint)
        index = {j.name: i for i, j in enumerate(self.actuated)}
        self.ancestors = {self.root: ()}
        stack = [self.root]
        while stack:
            parent = stack.pop()
            for joint in self._children.get(parent, ()):
                own = (index[joint.name],) if joint.name in index else ()
                self.ancestors[joint.child] = self.ancestors[parent] + own
                stack.append(joint.child)

    @property
    def joint_names(self) -> list[str]:
        return [j.name for j in self.actuated]

    @property
    def lower(self) -> F64:
        return np.array([j.lower for j in self.actuated])

    @property
    def upper(self) -> F64:
        return np.array([j.upper for j in self.actuated])

    def index(self, name: str) -> int:
        return self.joint_names.index(name)

    def link_names(self) -> list[str]:
        names = [self.root]
        names.extend(j.child for j in self.joints)
        return names

    def forward(self, q: F64, base: F64 | None = None) -> dict[str, F64]:
        """World transform of every link for joint angles ``q`` (actuated order)."""
        return self.forward_with_axes(q, base)[0]

    def forward_with_axes(self, q: F64, base: F64 | None = None) -> tuple[dict[str, F64], F64, F64]:
        """Link poses, plus each actuated joint's world axis and position, ``(J, 3)`` each.

        The axes and positions are what an exact Jacobian needs: a point ``p``
        moved by revolute joint ``j`` moves at ``axis_j x (p - position_j)``.
        """
        values = {j.name: float(v) for j, v in zip(self.actuated, q, strict=True)}
        index = {j.name: i for i, j in enumerate(self.actuated)}
        axes = np.zeros((len(self.actuated), 3))
        origins = np.zeros((len(self.actuated), 3))
        poses = {self.root: np.eye(4) if base is None else np.asarray(base, dtype=np.float64)}
        stack = [self.root]
        while stack:
            parent = stack.pop()
            for joint in self._children.get(parent, ()):
                frame = poses[parent] @ joint.origin
                if joint.name in index:
                    axes[index[joint.name]] = frame[:3, :3] @ (joint.axis / np.linalg.norm(joint.axis))
                    origins[index[joint.name]] = frame[:3, 3]
                local = joint.origin
                if joint.kind in ("revolute", "continuous"):
                    local = local @ transform(axis_angle(joint.axis, values[joint.name]), np.zeros(3))
                elif joint.kind == "prismatic":
                    local = local @ transform(np.eye(3), joint.axis * values[joint.name])
                poses[joint.child] = poses[parent] @ local
                stack.append(joint.child)
        return poses, axes, origins


def _origin(element: ET.Element | None) -> F64:
    if element is None:
        return np.eye(4)
    xyz = np.array([float(v) for v in element.get("xyz", "0 0 0").split()])
    rpy = [float(v) for v in element.get("rpy", "0 0 0").split()]
    return transform(rpy_matrix(*rpy), xyz)


def load_urdf(path: Path) -> Robot:
    tree = ET.parse(path)
    root_el = tree.getroot()
    materials: dict[str, tuple[float, float, float, float]] = {}
    for material in root_el.findall("material"):
        colour = material.find("color")
        if colour is not None and material.get("name"):
            values = [float(v) for v in colour.get("rgba", "0.7 0.7 0.7 1").split()]
            materials[str(material.get("name"))] = (values[0], values[1], values[2], values[3])

    visuals: dict[str, list[Visual]] = {}
    for link in root_el.findall("link"):
        name = str(link.get("name"))
        for visual in link.findall("visual"):
            mesh = visual.find("geometry/mesh")
            if mesh is None or not mesh.get("filename"):
                continue
            rgba = (0.7, 0.7, 0.72, 1.0)
            own = visual.find("material")
            if own is not None:
                colour = own.find("color")
                if colour is not None:
                    v = [float(x) for x in colour.get("rgba", "0.7 0.7 0.7 1").split()]
                    rgba = (v[0], v[1], v[2], v[3])
                elif own.get("name") in materials:
                    rgba = materials[str(own.get("name"))]
            filename = str(mesh.get("filename")).removeprefix("package://")
            visuals.setdefault(name, []).append(Visual(filename, _origin(visual.find("origin")), rgba))

    joints: list[Joint] = []
    for el in root_el.findall("joint"):
        limit = el.find("limit")
        axis = el.find("axis")
        joints.append(
            Joint(
                name=str(el.get("name")),
                kind=str(el.get("type")),
                parent=str(el.find("parent").get("link")),  # type: ignore[union-attr]
                child=str(el.find("child").get("link")),  # type: ignore[union-attr]
                origin=_origin(el.find("origin")),
                axis=np.array([float(v) for v in (axis.get("xyz", "1 0 0") if axis is not None else "1 0 0").split()]),
                lower=float(limit.get("lower", "-inf")) if limit is not None else -np.inf,
                upper=float(limit.get("upper", "inf")) if limit is not None else np.inf,
                velocity=float(limit.get("velocity", "inf")) if limit is not None else np.inf,
            )
        )
    children = {j.child for j in joints}
    roots = [str(link.get("name")) for link in root_el.findall("link") if link.get("name") not in children]
    if len(roots) != 1:
        raise ValueError(f"{path.name} has {len(roots)} root links ({roots}); expected exactly one")
    return Robot(name=str(root_el.get("name")), root=roots[0], joints=joints, visuals=visuals)
