"""The window: it owns what the panels share and delegates everything else."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QHBoxLayout, QMainWindow, QSplitter, QStatusBar, QVBoxLayout, QWidget

from alchemypose import __version__
from alchemypose.core.presets import matching_preset, selection_for
from alchemypose.core.schema import KeypointSelection
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

        from alchemypose.gui.canvas.scene import PoseOverlay
        from alchemypose.gui.canvas.view import PoseCanvas
        from alchemypose.gui.panels.layers import LayerPanel
        from alchemypose.gui.panels.presets import PresetBar

        self.canvas = PoseCanvas()
        self.canvas_overlay = PoseOverlay(self.canvas)
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

        self.presets.preset_chosen.connect(self._apply_preset)
        self.layers.selection_changed.connect(self._on_selection_changed)
        self.layers.set_selection(selection_for("full_body"))

    def set_status(self, message: str) -> None:
        self._status.showMessage(message)

    @property
    def status(self) -> str:
        return self._status.currentMessage()

    def _apply_preset(self, name: str) -> None:
        """A chip only ever writes to the tree; the tree tells everyone else."""
        self.layers.set_selection(selection_for(name))

    def _on_selection_changed(self, selection: KeypointSelection) -> None:
        self.presets.highlight(matching_preset(selection))
        self.canvas_overlay.set_selection(selection)
