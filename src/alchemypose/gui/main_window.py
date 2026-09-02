"""The window: it owns what the panels share and delegates everything else."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QHBoxLayout, QMainWindow, QSplitter, QStatusBar, QVBoxLayout, QWidget

from alchemypose import __version__
from alchemypose.gui import theme
from alchemypose.gui.reporting import DialogReporter, Reporter

WINDOW_TITLE = "AlchemyPose"
WINDOW_GEOMETRY = (1440, 900)


class MainWindow(QMainWindow):
    """Canvas on the left, presets and the layer tree on the right."""

    def __init__(self, reporter: Reporter | None = None) -> None:
        super().__init__()
        self.reporter: Reporter = reporter or DialogReporter()
        self.setWindowTitle(f"{WINDOW_TITLE} ({__version__})")
        self.resize(*WINDOW_GEOMETRY)
        self.setStyleSheet(theme.STYLESHEET)

        # NOTE: PoseCanvas, LayerPanel and PresetBar are placeholders until
        # Tasks 13, 15 and 16 land — see src/alchemypose/gui/canvas/view.py
        # and src/alchemypose/gui/panels/{layers,presets}.py.
        from alchemypose.gui.canvas.view import PoseCanvas
        from alchemypose.gui.panels.layers import LayerPanel
        from alchemypose.gui.panels.presets import PresetBar

        self.canvas = PoseCanvas()
        self.presets = PresetBar()
        self.layers = LayerPanel()

        side = QWidget()
        side_layout = QVBoxLayout(side)
        side_layout.setContentsMargins(8, 8, 8, 8)
        side_layout.addWidget(self.presets)
        side_layout.addWidget(self.layers, stretch=1)
        side.setFixedWidth(280)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(self.canvas)
        splitter.addWidget(side)
        splitter.setStretchFactor(0, 1)

        central = QWidget()
        layout = QHBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(splitter)
        self.setCentralWidget(central)

        self._status = QStatusBar()
        self.setStatusBar(self._status)
        self.set_status("Ready.")

    def set_status(self, message: str) -> None:
        self._status.showMessage(message)

    @property
    def status(self) -> str:
        return self._status.currentMessage()
