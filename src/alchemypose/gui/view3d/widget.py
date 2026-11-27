"""The 3D panel: an orbitable pyqtgraph OpenGL view, or a sentence saying why not.

Two things can make OpenGL unavailable, and neither is an error. The ``threed``
extra (``pyqtgraph`` + ``PyOpenGL``) may not be installed; and Qt's ``offscreen``
platform — the one CI and the test suite run on — cannot host a
``QOpenGLWidget`` at all (it reports "not supported on this platform" and the
process then dies at exit). In both cases the panel shows a label instead, and
everything that decides *what* to draw still runs: it lives in ``geometry`` and
needs no GL.

Licensing: this is pyqtgraph (MIT) over PyOpenGL (BSD). ``PySide6.QtQuick3D``
and ``PySide6.QtGraphs`` are never imported anywhere in this project — both are
commercial-or-GPLv3, and importing either would relicense the application.
"""

from __future__ import annotations

from typing import Any

from PySide6.QtCore import Qt
from PySide6.QtGui import QGuiApplication
from PySide6.QtWidgets import QLabel, QVBoxLayout, QWidget

from alchemypose.gui import theme
from alchemypose.gui.view3d.geometry import Geometry

#: Platforms that cannot create an OpenGL widget.
_NO_GL_PLATFORMS = frozenset({"offscreen", "minimal"})


def gl_unavailable_reason() -> str | None:
    """Why the 3D view cannot be drawn here, or ``None`` if it can."""
    if QGuiApplication.platformName() in _NO_GL_PLATFORMS:
        return f"3D view needs a display (Qt platform '{QGuiApplication.platformName()}' has no OpenGL)."
    try:
        import OpenGL  # noqa: F401
        import pyqtgraph.opengl  # noqa: F401
    except ImportError:
        return 'The 3D view needs the threed extra:  pip install "alchemypose[threed]"'
    return None


class Skeleton3DView(QWidget):
    """Shows a ``Geometry``. Orbit with the left button, pan with the middle, zoom with the wheel."""

    def __init__(self, title: str = "3D", parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._title = title
        self._view: Any = None
        self._lines: Any = None
        self._joints: Any = None
        self.unavailable_reason = gl_unavailable_reason()
        self.last_geometry: Geometry = Geometry.nothing()
        if self.unavailable_reason is None:
            self._build_gl()
        else:
            label = QLabel(self.unavailable_reason)
            label.setWordWrap(True)
            label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            label.setStyleSheet(f"color: {theme.TEXT_DIM}; background: {theme.CANVAS_BG}; padding: 16px;")
            self._layout.addWidget(label)

    @property
    def available(self) -> bool:
        return self._view is not None

    @property
    def gl_view(self) -> Any:
        """The ``GLViewWidget``, for anything that adds items of its own (the robot)."""
        return self._view

    def _build_gl(self) -> None:
        import pyqtgraph.opengl as gl

        view = gl.GLViewWidget()
        view.setBackgroundColor(theme.CANVAS_BG)
        # In front of the person, a little above, looking along +y at the hips.
        view.setCameraPosition(distance=3.6, elevation=12, azimuth=-90)
        view.opts["center"] = _vector(0.0, 0.0, 0.9)

        grid = gl.GLGridItem()
        grid.setSize(6, 6)
        grid.setSpacing(0.5, 0.5)
        grid.setColor((80, 86, 98, 160))
        view.addItem(grid)

        self._lines = gl.GLLinePlotItem(mode="lines", width=3.0, antialias=True)
        self._joints = gl.GLScatterPlotItem(size=7.0, pxMode=True)
        view.addItem(self._lines)
        view.addItem(self._joints)
        self._view = view
        self._layout.addWidget(view)

    def set_geometry(self, geometry: Geometry) -> None:
        self.last_geometry = geometry
        if self._view is None:
            return
        import numpy as np

        if geometry.segments.shape[0]:
            self._lines.setData(pos=geometry.segments, color=geometry.segment_colours, mode="lines")
            self._lines.setVisible(True)
        else:
            self._lines.setVisible(False)
        if geometry.points.shape[0]:
            self._joints.setData(pos=geometry.points, color=geometry.point_colours, size=7.0)
            self._joints.setVisible(True)
        else:
            self._joints.setData(pos=np.zeros((0, 3), dtype=np.float32))
            self._joints.setVisible(False)


def _vector(x: float, y: float, z: float) -> Any:
    from PySide6.QtGui import QVector3D

    return QVector3D(x, y, z)
