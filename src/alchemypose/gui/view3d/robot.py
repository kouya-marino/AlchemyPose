"""The robot panel: the G1's meshes in an orbitable view, posed frame by frame.

Each link's mesh is uploaded to the GPU once, as its own item; a new pose only
moves the items (one 4x4 transform each), so playback costs forward kinematics
and 35 matrix uploads, not 393,000 triangles re-sent per frame.

Like the skeleton panel, it shows a sentence instead where there is no OpenGL —
and everything that decides *where* each link goes (``robot.urdf.Robot.forward``)
runs without it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QLabel, QVBoxLayout, QWidget

from alchemypose.gui import theme
from alchemypose.gui.view3d.widget import gl_unavailable_reason
from alchemypose.robot.urdf import Robot

F64 = NDArray[np.float64]

#: The G1's two tones, from the URDF's own two materials.
_BODY = (0.82, 0.84, 0.88, 1.0)
_DARK = (0.42, 0.44, 0.48, 1.0)


#: Light directions, in a link's own frame (links stay near upright, so this
#: reads as front-and-above). pyqtgraph's own "shaded" shader lights from
#: behind the viewer's line and drew the robot almost black.
_KEY_LIGHT = np.array([0.7, -0.35, 0.6]) / np.linalg.norm([0.7, -0.35, 0.6])
_FILL_LIGHT = np.array([-0.5, 0.6, 0.2]) / np.linalg.norm([-0.5, 0.6, 0.2])


def lit_mesh(
    triangles: NDArray[np.float32], rgba: tuple[float, float, float, float]
) -> tuple[NDArray[np.float32], NDArray[np.int32], NDArray[np.float32]]:
    """An indexed mesh with lighting baked into smooth per-vertex colours.

    Each triangle is lit two-sided (Unitree's STLs wind some faces inward, and
    one-sided lighting drew those black), then every vertex takes the mean of
    the triangles sharing it, so the GPU interpolates a smooth gradient instead
    of showing the meshes' irregular triangulation as blotches.
    """
    flat = triangles.reshape(-1, 3)
    vertices, inverse = np.unique(np.round(flat, 6), axis=0, return_inverse=True)
    faces = inverse.reshape(-1, 3).astype(np.int32)
    normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    lengths = np.linalg.norm(normals, axis=1, keepdims=True)
    normals = np.divide(normals, lengths, out=np.zeros_like(normals), where=lengths > 0)
    face_light = 0.50 + 0.45 * np.abs(normals @ _KEY_LIGHT) + 0.12 * np.abs(normals @ _FILL_LIGHT)
    total = np.zeros(len(vertices))
    count = np.zeros(len(vertices))
    np.add.at(total, faces.reshape(-1), np.repeat(face_light, 3))
    np.add.at(count, faces.reshape(-1), 1.0)
    light = total / np.maximum(count, 1.0)
    colours = np.empty((len(vertices), 4), dtype=np.float32)
    colours[:, :3] = np.clip(np.outer(light, rgba[:3]), 0.0, 1.0)
    colours[:, 3] = rgba[3]
    return vertices.astype(np.float32), faces, colours


#: One link's ready-to-upload mesh: (link name, visual origin, vertices, faces, colours).
PreparedMesh = tuple[str, F64, NDArray[np.float32], NDArray[np.int32], NDArray[np.float32]]


def prepare_meshes(robot: Robot, mesh_root: Path) -> list[PreparedMesh]:
    """Read and light every link's mesh. Pure numpy, so it runs off the GUI thread (about 1.5 s)."""
    from alchemypose.robot.stl import load_stl

    prepared: list[PreparedMesh] = []
    for link, visuals in robot.visuals.items():
        for visual in visuals:
            triangles, _normals = load_stl(mesh_root / visual.mesh)
            # The URDF's two materials, "white" (0.7) and "dark" (0.2), kept as
            # two tones but lifted: 0.2 vanishes into the dark canvas.
            colour = _DARK if max(visual.rgba[:3]) < 0.4 else _BODY
            vertices, faces, colours = lit_mesh(triangles, colour)
            prepared.append((link, visual.origin, vertices, faces, colours))
    return prepared


def link_transforms(robot: Robot, q: F64, rotation: F64, height: float) -> dict[str, F64]:
    """World transform of every link with the pelvis rotated by ``rotation`` at ``height``."""
    base = np.eye(4)
    base[:3, :3] = rotation
    base[2, 3] = height
    return robot.forward(q, base)


class RobotView(QWidget):
    """Shows one robot. Orbit with the left button, pan with the middle, zoom with the wheel."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._view: Any = None
        self._items: list[tuple[str, F64, Any]] = []     # (link, visual origin, GL item)
        self.robot: Robot | None = None
        self.last_pose: tuple[F64, F64, float] | None = None
        self.unavailable_reason = gl_unavailable_reason()
        self._label = QLabel(self.unavailable_reason or "Loading the Unitree G1 ...")
        self._label.setWordWrap(True)
        self._label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._label.setStyleSheet(f"color: {theme.TEXT_DIM}; background: {theme.CANVAS_BG}; padding: 16px;")
        self._layout.addWidget(self._label)
        if self.unavailable_reason is None:
            self._build_gl()

    @property
    def available(self) -> bool:
        return self._view is not None

    @property
    def loaded(self) -> bool:
        return self.robot is not None

    def _build_gl(self) -> None:
        import pyqtgraph.opengl as gl
        from PySide6.QtGui import QVector3D

        view = gl.GLViewWidget()
        view.setBackgroundColor(theme.CANVAS_BG)
        # In front of the robot (it faces +x), a little to its right, like the skeleton view.
        view.setCameraPosition(distance=2.3, elevation=10, azimuth=-15)
        view.opts["center"] = QVector3D(0.0, 0.0, 0.7)
        grid = gl.GLGridItem()
        grid.setSize(6, 6)
        grid.setSpacing(0.5, 0.5)
        grid.setColor((80, 86, 98, 160))
        view.addItem(grid)
        view.hide()
        self._view = view
        self._layout.addWidget(view)

    def load(self, robot: Robot, meshes: list[PreparedMesh]) -> None:
        """Upload prepared meshes once. Safe without OpenGL (it then only records the robot)."""
        self.robot = robot
        if self._view is None:
            return
        import pyqtgraph.opengl as gl

        for _, _, item in self._items:
            self._view.removeItem(item)
        self._items = []
        for link, origin, vertices, faces, colours in meshes:
            mesh = gl.MeshData(vertexes=vertices, faces=faces, vertexColors=colours)
            # A few zero-area triangles in Unitree's meshes make pyqtgraph's
            # normal computation divide by zero; harmless, and lighting is baked.
            with np.errstate(divide="ignore", invalid="ignore"):
                item = gl.GLMeshItem(meshdata=mesh, smooth=True, shader=None, glOptions="opaque")
            self._view.addItem(item)
            self._items.append((link, origin, item))
        self._label.hide()
        self._view.show()

    def set_pose(self, q: F64, rotation: F64, height: float) -> None:
        self.last_pose = (q, rotation, height)
        if self._view is None or self.robot is None:
            return
        import pyqtgraph as pg

        poses = link_transforms(self.robot, q, rotation, height)
        for link, origin, item in self._items:
            world = poses[link] @ origin
            item.setTransform(pg.Transform3D(*world.astype(np.float32).reshape(-1)))

    def show_message(self, text: str) -> None:
        self._label.setText(text)
        self._label.show()
        if self._view is not None and self.robot is None:
            self._view.hide()
