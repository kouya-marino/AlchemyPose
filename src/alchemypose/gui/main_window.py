"""The window: it owns what the panels share and delegates everything else."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

from PySide6.QtCore import Qt
from PySide6.QtGui import QAction, QCloseEvent, QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QSizePolicy,
    QSlider,
    QSplitter,
    QStatusBar,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

from alchemypose import __version__
from alchemypose.core.presets import matching_preset, selection_for
from alchemypose.core.schema import KeypointSelection, PoseResult
from alchemypose.errors import AlchemyPoseError
from alchemypose.gui import theme
from alchemypose.gui.reporting import DialogReporter, Reporter
from alchemypose.inference.engine import PoseEngine

if TYPE_CHECKING:
    from alchemypose.gui.export import SavePrompt

WINDOW_TITLE = "AlchemyPose"
WINDOW_GEOMETRY = (1440, 900)
DEFAULT_CONFIDENCE_THRESHOLD = 0.3
IMAGE_FILTER = "Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.webp)"

#: The Export menu, in order: label, file suffix, and the save dialog's filter.
EXPORT_KINDS: tuple[tuple[str, str, str], ...] = (
    ("Keypoints as JSON…", "json", "JSON (*.json)"),
    ("Keypoints as CSV…", "csv", "CSV (*.csv)"),
    ("Overlay as PNG…", "png", "PNG image (*.png)"),
)


def default_engine_factory() -> PoseEngine:
    """Build the real engine.

    Called by the worker *on its thread*, the first time a job runs. The import
    lives here too, so that neither onnxruntime's import cost nor the CoreML
    compile ever lands on the GUI thread or in window construction.
    """
    from alchemypose.inference.rtmw import RTMWEngine

    return RTMWEngine(preset="performance")


class MainWindow(QMainWindow):
    """Canvas on the left, presets and the layer tree on the right.

    ``engine_factory`` exists for tests and for anyone embedding the window
    with a different model; the default builds RTMW lazily on the worker.
    ``reporter`` and ``save_prompt`` are injected for the same reason as each
    other: both open a modal, and a test must never construct one.
    """

    def __init__(
        self,
        reporter: Reporter | None = None,
        engine_factory: Callable[[], PoseEngine] | None = None,
        save_prompt: SavePrompt | None = None,
    ) -> None:
        super().__init__()
        self.reporter: Reporter = reporter or DialogReporter()
        self.setWindowTitle(f"{WINDOW_TITLE} ({__version__})")
        self.resize(*WINDOW_GEOMETRY)
        self.setStyleSheet(theme.STYLESHEET)

        from alchemypose.gui.bridge import WorkerBridge
        from alchemypose.gui.canvas.interaction import SelectionController
        from alchemypose.gui.canvas.scene import PoseOverlay
        from alchemypose.gui.canvas.view import PoseCanvas
        from alchemypose.gui.export import DialogSavePrompt
        from alchemypose.gui.panels.inspector import Inspector, LegendWidget
        from alchemypose.gui.panels.layers import LayerPanel
        from alchemypose.gui.panels.presets import PresetBar

        self._result: PoseResult | None = None
        self._pending_source: str | None = None
        self._person: int | None = None
        self._confidence_threshold = DEFAULT_CONFIDENCE_THRESHOLD
        self.save_prompt: SavePrompt = save_prompt or DialogSavePrompt()

        self.canvas = PoseCanvas()
        self.canvas_overlay = PoseOverlay(self.canvas)
        self.presets = PresetBar()
        self.layers = LayerPanel()
        self.inspector = Inspector()
        self.legend = LegendWidget()
        self.selection = SelectionController(self.canvas_overlay, self.layers)

        side = QWidget()
        side_layout = QVBoxLayout(side)
        side_layout.setContentsMargins(8, 8, 8, 8)
        side_layout.addWidget(self.presets)
        side_layout.addWidget(self.layers, stretch=1)
        side_layout.addWidget(self.inspector)
        side_layout.addWidget(self.legend)
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

        self._build_toolbar()
        self._build_export_menu()

        self.presets.preset_chosen.connect(self._apply_preset)
        self.layers.selection_changed.connect(self._on_selection_changed)
        self.layers.set_selection(selection_for("full_body"))

        self.canvas.keypoint_clicked.connect(self._on_keypoint_clicked)
        self.canvas.keypoint_double_clicked.connect(self._on_keypoint_double_clicked)
        # The overlay cannot see the view transform, and how much face it draws
        # depends on it; this is the only thing that tells it.
        self.canvas.zoom_changed.connect(self.canvas_overlay.on_zoom_changed)
        self.canvas_overlay.keypoint_hovered.connect(self._on_keypoint_hovered)
        self.canvas_overlay.keypoint_left.connect(self.inspector.clear)

        QShortcut(QKeySequence("Shift+H"), self, self._solo_current)
        QShortcut(QKeySequence("Alt+H"), self, self.selection.clear)
        QShortcut(QKeySequence("Esc"), self, self.selection.clear)

        # The bridge is the only path from the worker thread to this window:
        # both connections below are queued, so the slots run here, not there.
        self._bridge = WorkerBridge(engine_factory or default_engine_factory, parent=self)
        self._bridge.result_ready.connect(self._on_result)
        self._bridge.failed.connect(self._on_failed)
        self._bridge.start()

    # ---------------------------------------------------------------- toolbar

    def _build_toolbar(self) -> None:
        from alchemypose.gui.panels.people import PersonBar

        toolbar = QToolBar("main", self)
        toolbar.setMovable(False)
        toolbar.setFloatable(False)
        self.addToolBar(toolbar)

        self.open_action = QAction("Open", self)
        self.open_action.setShortcut(QKeySequence.StandardKey.Open)
        self.open_action.setToolTip("Open an image (Ctrl+O)")
        self.open_action.triggered.connect(self._choose_file)
        toolbar.addAction(self.open_action)
        toolbar.addSeparator()

        self.people = PersonBar()
        self.people.person_chosen.connect(self._on_person_chosen)
        self.people.everyone_chosen.connect(self._on_everyone_chosen)
        toolbar.addWidget(self.people)

        spacer = QWidget()
        spacer.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        toolbar.addWidget(spacer)

        self._build_confidence_slider(toolbar)

    def _build_confidence_slider(self, toolbar: QToolBar) -> None:
        caption = QLabel("Confidence ≥ ")
        caption.setStyleSheet(f"color: {theme.TEXT_DIM};")
        toolbar.addWidget(caption)

        self.confidence_slider = QSlider(Qt.Orientation.Horizontal)
        self.confidence_slider.setRange(0, 100)
        self.confidence_slider.setValue(round(self._confidence_threshold * 100))
        self.confidence_slider.setFixedWidth(120)
        self.confidence_slider.setToolTip("Hide keypoints whose confidence is below this")
        self.confidence_slider.valueChanged.connect(self._on_slider_moved)
        toolbar.addWidget(self.confidence_slider)

        self._confidence_label = QLabel(f"{self._confidence_threshold:.2f}")
        self._confidence_label.setFixedWidth(36)
        toolbar.addWidget(self._confidence_label)

    # ------------------------------------------------------------------ export

    def _build_export_menu(self) -> None:
        menu = self.menuBar().addMenu("Export")
        self.export_actions: dict[str, QAction] = {}
        for label, suffix, file_filter in EXPORT_KINDS:
            action = QAction(label, self)
            # Defaults, not closure capture: ``triggered`` also passes a
            # checked flag, and a bare closure would read the loop variable
            # after the loop had finished with it.
            action.triggered.connect(lambda _checked=False, s=suffix, f=file_filter: self._export(s, f))
            menu.addAction(action)
            self.export_actions[suffix] = action

    def _export(self, suffix: str, file_filter: str) -> None:
        """Ask where it should go, then write it.

        Nothing that can go wrong here is a bug: there may be no result yet,
        the user may cancel, and the folder they type may not exist. Each of
        those ends as a sentence, not a traceback.
        """
        write = self._export_writer(suffix)
        if write is None:
            self.set_status("Nothing to export yet — open an image first.")
            return

        path = self.save_prompt.ask(self, "Export", f"{self._export_stem()}.{suffix}", file_filter)
        if path is None:
            return

        try:
            write(path)
        except AlchemyPoseError as exc:
            self.reporter.error("Export failed", str(exc))
            self.set_status("Export failed.")
            return
        self.set_status(f"Wrote {path.name}")

    def _export_writer(self, suffix: str) -> Callable[[Path], None] | None:
        """The function that writes this format, or ``None`` if there is nothing to write.

        Answering both questions in one place is what keeps the caller free of
        'this cannot be None by now' assertions: the writer it returns closes
        over the result it just checked.
        """
        from alchemypose.gui.export import export_csv, export_json, export_png

        if suffix == "png":
            if self.canvas.image_item is None:
                return None
            return lambda path: export_png(self.canvas, path)

        result = self._result
        if result is None:
            return None
        if suffix == "json":
            return lambda path: export_json(result, path)
        return lambda path: export_csv(result, path)

    def _export_stem(self) -> str:
        """Name the file after the photograph it came from, when there is one."""
        source = self._pending_source
        return Path(source).stem if source else "pose"

    # ----------------------------------------------------------------- status

    def set_status(self, message: str) -> None:
        self._status.showMessage(message)

    @property
    def status(self) -> str:
        return self._status.currentMessage()

    # ------------------------------------------------------------ open / infer

    def _choose_file(self) -> None:
        name, _ = QFileDialog.getOpenFileName(self, "Open image", "", IMAGE_FILTER)
        if name:
            self.open_image(Path(name))

    def open_image(self, path: Path) -> None:
        """Show the image immediately, then detect in the background."""
        from alchemypose.media.image import load_image

        try:
            image = load_image(path)
        except AlchemyPoseError as exc:
            self.reporter.error("Could not open image", str(exc))
            self.set_status("Could not open that file.")
            return

        self._result = None
        self._person = None
        self._pending_source = str(path)
        self.people.set_people(0)
        self.inspector.clear()
        self.selection.clear()

        self.canvas.set_image(image)
        self.canvas.fit_to_window()
        self.set_status(f"{path.name} — detecting ...")
        self._bridge.submit(image, str(path))

    def _on_result(self, result: PoseResult) -> None:
        """Runs on the GUI thread, via the bridge's queued connection."""
        if self._pending_source is not None and result.source != self._pending_source:
            return  # a straggler for an image we have since replaced
        if not result.frames:
            self._on_failed("the engine returned no frames")
            return

        count = len(result.frames[0].people)
        self._result = result
        self._person = None
        self.people.set_people(count)
        self.canvas_overlay.set_result(result, self._effective_selection(), self._confidence_threshold)
        if count:
            noun = "person" if count == 1 else "people"
            self.set_status(f"{count} {noun} · {len(result.topology)} keypoints · {result.model}")
        else:
            self.set_status("No people detected.")

    def _on_failed(self, message: str) -> None:
        """Runs on the GUI thread, via the bridge's queued connection."""
        self.reporter.error("Pose estimation failed", message)
        self.set_status("Detection failed.")

    # -------------------------------------------------------------- selection

    def _effective_selection(self) -> KeypointSelection:
        """The tree's selection, narrowed to the chosen person if there is one."""
        tree = self.layers.current_selection()
        persons = None if self._person is None else frozenset({self._person})
        return KeypointSelection(visible=tree.visible, persons=persons)

    def _push_selection(self) -> None:
        """Send what should be drawn to the canvas.

        The only caller of ``canvas_overlay.set_selection`` in the app. The two
        halves of "what is drawn" live in different places — which groups in the
        tree, which person in the chips — and only ``_effective_selection`` puts
        them together, so a change to either half comes back through here rather
        than writing to the overlay itself. A second caller would inevitably
        know about one half and quietly reset the other.

        It is not the only *writer* of the overlay's selection: ``_on_result``
        hands it to ``set_result`` instead, because a new result and the
        selection it should be drawn with have to arrive in one call. That path
        composes the two halves through the same ``_effective_selection``, which
        is the invariant that matters — not that every write funnels through
        this method.
        """
        self.canvas_overlay.set_selection(self._effective_selection())

    def _apply_preset(self, name: str) -> None:
        """A chip only ever writes to the tree; the tree tells everyone else.

        Face is the exception, and only because it also moves the *view*: there
        is no face mode and no second canvas, so what a face tab would have
        given is the magnification, and the chip gives that on the same canvas.
        Clicking it in the seconds between opening a photograph and its
        detection returning is harmless — the bounds are empty then, and
        ``zoom_to_rect`` ignores an empty rectangle.
        """
        self.layers.set_selection(selection_for(name))
        if name == "face":
            person = 0 if self._person is None else self._person
            self.canvas.zoom_to_rect(self.canvas_overlay.group_bounds(person, "face"))

    def _on_selection_changed(self, selection: KeypointSelection) -> None:
        """Every group change lands here, whoever made it.

        Presets, the user's own checkboxes and ``SelectionController``'s solo
        all write to the tree and let it announce the result, so the person
        filter is re-applied on top of every one of them.
        """
        self.presets.highlight(matching_preset(selection))
        self._push_selection()

    def _on_person_chosen(self, index: int) -> None:
        self._person = index
        self._push_selection()
        self.set_status(f"Person {index + 1}")

    def _on_everyone_chosen(self) -> None:
        self._person = None
        self._push_selection()

    def _on_keypoint_clicked(self, person: int, index: int, group: str) -> None:
        """Click selects: emphasise the group, hide nothing."""
        self.selection.select_group(group)
        self.set_status(f"{group.replace('_', ' ')} — person {person + 1}")

    def _on_keypoint_double_clicked(self, person: int, index: int, group: str) -> None:
        """Double-click solos: hide everything else."""
        self.selection.solo_group(group)
        self.set_status(f"Solo: {group.replace('_', ' ')} — person {person + 1}")

    def _solo_current(self) -> None:
        """Shift+H: solo whatever is currently emphasised, if anything."""
        if self.selection.emphasised is not None:
            self.selection.solo_group(self.selection.emphasised)

    # -------------------------------------------------------------- inspector

    def _on_keypoint_hovered(self, person: int, index: int, group: str) -> None:
        if self._result is None:
            return
        frame = self._result.frames[0]
        if not 0 <= person < len(frame.people):
            return
        name = self._result.topology.keypoint_names[index]
        score = float(frame.people[person].scores[index])
        self.inspector.show_keypoint(name, score, group)

    # ------------------------------------------------------------- confidence

    @property
    def confidence_threshold(self) -> float:
        """Keypoints scoring below this are not drawn. The stored result is never touched."""
        return self._confidence_threshold

    def set_confidence_threshold(self, value: float) -> None:
        value = min(1.0, max(0.0, value))
        self._confidence_threshold = value
        self._confidence_label.setText(f"{value:.2f}")
        # Keep the slider in step without bouncing back through _on_slider_moved.
        self.confidence_slider.blockSignals(True)
        try:
            self.confidence_slider.setValue(round(value * 100))
        finally:
            self.confidence_slider.blockSignals(False)
        self.canvas_overlay.set_confidence_threshold(value)

    def _on_slider_moved(self, ticks: int) -> None:
        self.set_confidence_threshold(ticks / 100.0)

    # -------------------------------------------------------------- lifecycle

    def closeEvent(self, event: QCloseEvent) -> None:
        """Stop the worker before the widgets it would report to disappear."""
        self._bridge.stop()
        super().closeEvent(event)
