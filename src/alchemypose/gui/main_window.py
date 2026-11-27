"""The window: it owns what the panels share and delegates everything else."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, cast

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QAction, QCloseEvent, QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QComboBox,
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
from alchemypose.inference import weights
from alchemypose.inference.engine import PoseEngine

if TYPE_CHECKING:
    from alchemypose.gui.bridge import WorkerBridge
    from alchemypose.gui.export import SavePrompt
    from alchemypose.inference.video import FrameEngine

#: The preset the window runs. Named once, because the factory below has to
#: fetch exactly the models the engine is about to ask for.
DEFAULT_PRESET = "performance"

WINDOW_TITLE = "AlchemyPose"
WINDOW_GEOMETRY = (1440, 900)
DEFAULT_CONFIDENCE_THRESHOLD = 0.3

#: How each preset reads in the toolbar.
PRESET_LABELS: dict[str, str] = {"lite": "Lite", "balanced": "Balanced", "performance": "Performance", "3d": "3D"}
IMAGE_FILTER = "Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.webp)"
VIDEO_FILTER = "Videos (*.mp4 *.mov *.m4v *.avi *.mkv *.webm)"
MEDIA_FILTER = (
    "Images and videos (*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.webp *.mp4 *.mov *.m4v *.avi *.mkv *.webm);;"
    f"{IMAGE_FILTER};;{VIDEO_FILTER}"
)

#: The Export menu, in order: label, file suffix, and the save dialog's filter.
EXPORT_KINDS: tuple[tuple[str, str, str], ...] = (
    ("Keypoints as JSON…", "json", "JSON (*.json)"),
    ("Keypoints as CSV…", "csv", "CSV (*.csv)"),
    ("Overlay as PNG…", "png", "PNG image (*.png)"),
    ("3D motion as glTF…", "glb", "glTF binary (*.glb)"),
)


def default_engine_factory(preset: str = DEFAULT_PRESET) -> PoseEngine:
    """Build the real engine.

    Called by the worker *on its thread*, the first time a job runs. The import
    lives here too, so that neither onnxruntime's import cost nor the CoreML
    compile ever lands on the GUI thread or in window construction.
    """
    from alchemypose.inference.rtmw import RTMWEngine

    return RTMWEngine(preset=preset)


def _weights_progress(key: str, announce: Callable[[str], None]) -> Callable[[int, int], None]:
    """Turn ``weights.ensure``'s byte counts into sentences for the status bar.

    Runs on the worker thread, and ``announce`` — a signal's ``emit`` — is the
    only thing it is allowed to touch. The status bar is a widget, widgets
    belong to the GUI thread, and the whole reason this returns a string rather
    than writing one is that it is on the wrong side of that line.

    ``ensure`` calls back once with ``(0, total)`` before any bytes move, and
    that opening call is the one that carries the news worth having: which
    model, and how big. The rest only move a percentage, and are throttled to
    whole-percent changes — 213 MB arrives in some three thousand chunks, and
    posting three thousand queued events to redraw the same sentence is work
    the GUI thread has better uses for.
    """
    size = ""
    last = -1

    def report(done: int, total: int) -> None:
        nonlocal size, last
        if done == 0:
            size, last = weights.human_size(total), -1
            announce(f"Downloading {key} ({size}) ...")
            return
        # No Content-Length means no percentage to show; count megabytes instead.
        if total > 0:
            step = done * 100 // total
            said = f"Downloading {key} ({size}) · {step}%"
        else:
            step = done // 1_000_000
            said = f"Downloading {key} · {weights.human_size(done)}"
        if step == last:
            return
        last = step
        announce(said)

    return report


def _fetching_engine_factory(
    announce: Callable[[str], None], preset: str = DEFAULT_PRESET
) -> Callable[[], PoseEngine]:
    """The factory the worker actually calls: fetch the weights aloud, then build.

    ``RTMWEngine`` fetches its own weights and says nothing at all while it
    does, so on a fresh install the first photograph opened sat behind a
    quarter of a gigabyte with the status bar reading "detecting ...". Doing
    the fetch here first puts that download in front of ``announce``; the
    engine's own ``weights.ensure``, moments later, finds every file present
    and verified and returns at the speed of a checksum.

    Nothing is said when nothing is downloaded: ``ensure`` does not call back
    on a cache hit, so the ordinary warm run reaches the engine in silence and
    the status bar keeps whatever ``open_image`` put there.
    """

    def factory() -> PoseEngine:
        spoke = False

        def speak(message: str) -> None:
            nonlocal spoke
            spoke = True
            announce(message)

        for key in weights.PRESETS[preset]:
            weights.ensure(key, progress=_weights_progress(key, speak))
        if spoke:
            # Only after a download, and only because the alternative is a
            # status bar frozen at "100%" for as long as CoreML takes to
            # compile the graph it has just been handed.
            announce("Model weights ready. Loading ...")
        return default_engine_factory(preset)

    return factory


class MainWindow(QMainWindow):
    """Canvas on the left, presets and the layer tree on the right.

    ``engine_factory`` exists for tests and for anyone embedding the window
    with a different model; the default builds RTMW lazily on the worker.
    ``reporter`` and ``save_prompt`` are injected for the same reason as each
    other: both open a modal, and a test must never construct one.
    """

    #: Download news, emitted from the worker thread on its way to the status
    #: bar. It is a signal for the same reason ``WorkerBridge``'s two are:
    #: emitting is thread-safe from anywhere, and a connection to a slot on
    #: this window is delivered by Qt's queued machinery on the GUI thread, so
    #: nothing on the worker side ever touches a widget. It lives here rather
    #: than on the bridge because it is the window's status bar being written
    #: to, and the bridge exists to carry results.
    weights_progress = Signal(str)

    def __init__(
        self,
        reporter: Reporter | None = None,
        engine_factory: Callable[[], PoseEngine] | None = None,
        save_prompt: SavePrompt | None = None,
        video_engine_factory: Callable[[], FrameEngine] | None = None,
    ) -> None:
        super().__init__()
        self.reporter: Reporter = reporter or DialogReporter()
        self.setWindowTitle(f"{WINDOW_TITLE} ({__version__})")
        self.resize(*WINDOW_GEOMETRY)
        self.setStyleSheet(theme.STYLESHEET)

        from alchemypose.gui.canvas.interaction import SelectionController
        from alchemypose.gui.canvas.scene import PoseOverlay
        from alchemypose.gui.canvas.view import PoseCanvas
        from alchemypose.gui.export import DialogSavePrompt
        from alchemypose.gui.panels.inspector import Inspector, LegendWidget
        from alchemypose.gui.panels.layers import LayerPanel
        from alchemypose.gui.panels.presets import PresetBar
        from alchemypose.gui.panels.transport import TransportBar
        from alchemypose.gui.robot import RobotController
        from alchemypose.gui.video import VideoController
        from alchemypose.gui.view3d.robot import RobotView
        from alchemypose.gui.view3d.widget import Skeleton3DView

        self._result: PoseResult | None = None
        self._preset = DEFAULT_PRESET
        self._injected_engine_factory = engine_factory
        self._injected_video_factory = video_engine_factory
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

        # Video mode adds a transport strip under the canvas; it is hidden for stills.
        self.transport = TransportBar()
        self.transport.hide()
        stage = QWidget()
        stage_layout = QVBoxLayout(stage)
        stage_layout.setContentsMargins(0, 0, 0, 0)
        stage_layout.setSpacing(0)
        stage_layout.addWidget(self.canvas, stretch=1)
        stage_layout.addWidget(self.transport)
        self.stage_layout = stage_layout

        # The 3D panel sits between the picture and the controls. It is shown
        # whenever what is on screen carries 3D keypoints (the 3D preset).
        self.view3d = Skeleton3DView()
        self.view3d.setMinimumWidth(320)
        self.view3d.hide()
        # The Unitree G1, posed from the same motion. Shown by the G1 toolbar toggle.
        self.robot_view = RobotView()
        self.robot_view.setMinimumWidth(300)
        self.robot_view.hide()
        self.robot = RobotController(parent=self)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(stage)
        splitter.addWidget(self.view3d)
        splitter.addWidget(self.robot_view)
        splitter.addWidget(side)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)
        self.splitter = splitter

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

        # Every path from the worker thread to this window is a signal, and all
        # three connections below are queued, so the slots run here, not there.
        self.weights_progress.connect(self.set_status)
        self._bridge = self._make_bridge()

        # Video: its own engine, built lazily on the analysis thread. The cast
        # is honest — the fetching factory builds an RTMWEngine, which has
        # ``infer_frame`` — but the image worker's protocol does not promise it.
        self.video = VideoController(self._video_factory(), self._preset, parent=self)
        self._video_first_frame = False
        self.video.frame_ready.connect(self._on_video_frame)
        self.video.analysis_progress.connect(self._on_video_progress)
        self.video.analysis_finished.connect(self._on_video_analysed)
        self.video.analysis_failed.connect(self._on_video_failed)
        self.video.playing_changed.connect(self.transport.set_playing)
        self.transport.position_changed.connect(self.video.show)
        self.transport.play_toggled.connect(self._on_play_toggled)
        self.transport.smoothing_toggled.connect(self.video.set_smoothing)
        from alchemypose.gui.live import LiveController

        self.live = LiveController(lambda image, source: self._bridge.submit(image, source), parent=self)
        self._live_first_frame = False
        self.live.frame_ready.connect(self._on_live_frame)
        self.live.failed.connect(self._on_live_failed)
        self.robot.model_ready.connect(self._on_robot_model)
        self.robot.progress.connect(self._on_robot_progress)
        self.robot.motion_ready.connect(self._on_robot_motion)
        self.robot.failed.connect(self._on_robot_failed)
        self.transport.speed_changed.connect(self.video.set_speed)
        QShortcut(QKeySequence("Space"), self, self._toggle_play)
        QShortcut(QKeySequence("Right"), self, lambda: self._step(1))
        QShortcut(QKeySequence("Left"), self, lambda: self._step(-1))

    # ---------------------------------------------------------------- toolbar

    def _build_toolbar(self) -> None:
        from alchemypose.gui.panels.people import PersonBar

        toolbar = QToolBar("main", self)
        toolbar.setMovable(False)
        toolbar.setFloatable(False)
        self.addToolBar(toolbar)

        self.open_action = QAction("Open", self)
        self.open_action.setShortcut(QKeySequence.StandardKey.Open)
        self.open_action.setToolTip("Open an image or a video (Ctrl+O)")
        self.open_action.triggered.connect(self._choose_file)
        toolbar.addAction(self.open_action)

        self.webcam_action = QAction("Webcam", self)
        self.webcam_action.setCheckable(True)
        self.webcam_action.setToolTip("Live pose from the default camera")
        self.webcam_action.toggled.connect(self._on_webcam_toggled)
        toolbar.addAction(self.webcam_action)

        self.robot_action = QAction("G1", self)
        self.robot_action.setCheckable(True)
        self.robot_action.setToolTip(
            "Show the motion on a Unitree G1 (29 joints). Needs the 3D preset; kinematic only."
        )
        self.robot_action.toggled.connect(self._on_robot_toggled)
        toolbar.addAction(self.robot_action)
        toolbar.addSeparator()

        self.preset_box = QComboBox()
        for name in weights.PRESETS:
            self.preset_box.addItem(PRESET_LABELS.get(name, name), name)
        self.preset_box.setCurrentIndex(list(weights.PRESETS).index(self._preset))
        self.preset_box.setToolTip("Model preset. 3D adds a depth for every keypoint and the 3D view.")
        self.preset_box.currentIndexChanged.connect(
            lambda _i: self.set_preset(str(self.preset_box.currentData()))
        )
        toolbar.addWidget(self.preset_box)
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

    # ----------------------------------------------------------------- preset

    def _make_bridge(self) -> WorkerBridge:
        from alchemypose.gui.bridge import WorkerBridge

        factory = self._injected_engine_factory or _fetching_engine_factory(self.weights_progress.emit, self._preset)
        bridge = WorkerBridge(factory, parent=self)
        bridge.result_ready.connect(self._on_result)
        bridge.failed.connect(self._on_failed)
        bridge.start()
        return bridge

    def _video_factory(self) -> Callable[[], FrameEngine]:
        if self._injected_video_factory is not None:
            return self._injected_video_factory
        # Honest cast: the fetching factory builds an RTMWEngine, which has
        # ``infer_frame``; the image worker's protocol just does not promise it.
        return cast("Callable[[], FrameEngine]", _fetching_engine_factory(self.weights_progress.emit, self._preset))

    @property
    def preset(self) -> str:
        return self._preset

    def set_preset(self, preset: str) -> None:
        """Switch models, and re-run whatever is open with the new ones.

        The image worker is rebuilt rather than reconfigured: its engine is
        built lazily on its own thread, and a fresh worker is the one way to be
        sure no job for the old preset is still in flight to land afterwards.
        """
        if preset == self._preset or preset not in weights.PRESETS:
            return
        self._preset = preset
        index = list(weights.PRESETS).index(preset)
        if self.preset_box.currentIndex() != index:
            self.preset_box.blockSignals(True)
            self.preset_box.setCurrentIndex(index)
            self.preset_box.blockSignals(False)
        self._bridge.stop()
        self._bridge.deleteLater()
        self._bridge = self._make_bridge()
        self.video.set_engine(self._video_factory(), preset)

        source = self._pending_source
        if self.live.active:
            self.live.stop()
            self._on_webcam_toggled(True)
        elif source:
            self.open_path(Path(source))
        self.set_status(f"Preset: {PRESET_LABELS.get(preset, preset)}")

    # -------------------------------------------------------------------- 3D

    def _refresh_3d(self) -> None:
        """Mirror what the canvas draws into the 3D panel, when there is any 3D."""
        from alchemypose.gui.view3d.geometry import build

        result = self._result
        has_3d = result is not None and any(
            p.keypoints_3d is not None for f in result.frames[:1] for p in f.people
        )
        wanted = has_3d or self._preset == "3d"
        if wanted != self.view3d.isVisible():
            self.view3d.setVisible(wanted)
            self._share_width()
        if wanted:
            self.view3d.set_geometry(build(result, self._effective_selection(), self._confidence_threshold))

    def _share_width(self) -> None:
        """Split the width between picture, 3D and controls whenever a panel comes or goes.

        A ``QSplitter`` hands a newly shown widget whatever it likes, which left
        the picture a sliver beside an enormous 3D view. The picture is re-fitted
        afterwards, on the next turn of the event loop, once the splitter has
        actually resized it.
        """
        visible_3d = [w for w in (self.view3d, self.robot_view) if w.isVisibleTo(self)]
        total = max(self.splitter.width(), 1)
        side = 280
        if visible_3d:
            share = (total - side) // (len(visible_3d) + 1)
            sizes = [share]
            sizes += [share if w in visible_3d else 0 for w in self._splitter_panels()]
            sizes.append(side)
        else:
            sizes = [total - side] + [0 for _ in self._splitter_panels()] + [side]
        self.splitter.setSizes(sizes)
        QTimer.singleShot(0, self.canvas.fit_to_window)

    def _splitter_panels(self) -> list[QWidget]:
        """The optional panels between the picture and the controls, in splitter order."""
        panels = (self.splitter.widget(i) for i in range(1, self.splitter.count() - 1))
        return [panel for panel in panels if panel is not None]

    # ------------------------------------------------------------------- G1

    def _on_robot_toggled(self, on: bool) -> None:
        """Show the G1 and put the current motion on it; it needs 3D keypoints, so switch to 3D."""
        if not on:
            self.robot.cancel()
            self.robot_view.hide()
            self._share_width()
            return
        self.robot_view.show()
        self._share_width()
        if self._preset != "3d":
            self.set_status("The G1 needs 3D keypoints — switching to the 3D preset.")
            self.set_preset("3d")
        self.robot_view.show_message("Loading the Unitree G1 ...")
        self.robot.load_model()

    def _on_robot_model(self, robot: object, meshes: object) -> None:
        if not self.robot_view.loaded:
            self.robot_view.load(robot, meshes)  # type: ignore[arg-type]
        self._retarget_current()

    def _robot_track(self) -> int:
        return 0 if self._person is None else self._person

    def _retarget_current(self) -> None:
        """Retarget whatever is on screen: the whole clip, or the one still."""
        if not self.robot_action.isChecked() or self.robot.robot is None:
            return
        if self.video.is_open:
            result = self.video.result
            if result is None:
                self.set_status("G1: waiting for the clip's analysis ...")
                return
            if not any(p.keypoints_3d is not None for f in result.frames for p in f.people):
                self.robot_view.show_message("This clip was analysed without depth. Re-open it with the 3D preset.")
                return
            frames = self.video.smoothed_frames or result.frames
            self.robot.retarget(frames, self._robot_track())
        elif self._result is not None:
            from dataclasses import replace

            frame = self._result.frames[0]
            tracked = replace(frame, people=tuple(replace(p, track_id=i) for i, p in enumerate(frame.people)))
            if any(p.keypoints_3d is not None for p in tracked.people):
                self.robot.retarget((tracked,), self._robot_track())

    def _on_robot_progress(self, done: int, total: int) -> None:
        self.set_status(f"Retargeting to the G1 · {done}/{total} frames ...")

    def _on_robot_motion(self, motion: object, report: object) -> None:
        from alchemypose.robot.retarget import RobotMotion, SpeedReport

        assert isinstance(motion, RobotMotion) and isinstance(report, SpeedReport)
        import numpy as np

        error_cm = float(np.nanmedian(motion.error)) * 100 if motion.valid.any() else float("nan")
        if len(motion.q) == 1:
            self.set_status(f"G1 · posed from person {motion.track_id + 1} · miss {error_cm:.1f} cm")
            self._pose_robot(0)
            return
        if report.slowdown > 1.0:
            pace = f"too fast for the G1 — play at {1 / report.slowdown:.2f}× to stay within {report.budget:.0%}"
        else:
            pace = "within the G1's speed"
        self.set_status(
            f"G1 · {len(motion.q)} frames · median miss {error_cm:.1f} cm · peak {report.worst_fraction:.0%} of "
            f"{report.worst_joint.replace('_joint', '').replace('_', ' ')} motor speed · {pace}"
        )
        self.transport.suggest_speed(1.0 / report.slowdown if report.slowdown > 1.0 else 1.0)
        self._pose_robot(self.video.index if self.video.is_open else 0)

    def _on_robot_failed(self, message: str) -> None:
        self.robot_view.show_message(message)
        self.set_status("G1 unavailable.")

    def _pose_robot(self, index: int) -> None:
        if not self.robot_action.isChecked():
            return
        pose = self.robot.pose(index)
        if pose is not None:
            self.robot_view.set_pose(*pose)

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
        from alchemypose.gui.export import export_csv, export_glb, export_json, export_png

        if suffix == "png":
            if self.canvas.image_item is None:
                return None
            return lambda path: export_png(self.canvas, path)

        # In video mode the data exports are the whole clip, raw — not the one
        # smoothed frame on screen. Until the analysis finishes there is none.
        result = self.video.result if self.video.is_open else self._result
        if result is None:
            return None
        if suffix == "glb":
            if not any(p.keypoints_3d is not None for f in result.frames for p in f.people):
                return None
            track = 0 if self._person is None else self._person
            frames = self.video.smoothed_frames if self.video.is_open else None
            return lambda path: export_glb(result, path, frames=frames, track_id=track)
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
        name, _ = QFileDialog.getOpenFileName(self, "Open image or video", "", MEDIA_FILTER)
        if name:
            self.open_path(Path(name))

    def open_path(self, path: Path) -> None:
        """Open a photograph or a video, by what the file is called."""
        from alchemypose.media.video import is_video_path

        if is_video_path(path):
            self.open_video(path)
        else:
            self.open_image(path)

    def _leave_video(self) -> None:
        if self.video.is_open:
            self.video.close()
        self.transport.hide()
        self._leave_live()

    def _leave_live(self) -> None:
        if self.live.active:
            self.live.stop()
        if self.webcam_action.isChecked():
            self.webcam_action.blockSignals(True)
            self.webcam_action.setChecked(False)
            self.webcam_action.blockSignals(False)

    # ---------------------------------------------------------------- webcam

    def _on_webcam_toggled(self, on: bool) -> None:
        if not on:
            self.live.stop()
            self.set_status("Webcam off.")
            return
        if self.video.is_open:
            self.video.close()
        self.transport.hide()
        self._result = None
        self._person = None
        self._pending_source = None
        self.people.set_people(0)
        self.inspector.clear()
        self.selection.clear()
        self._live_first_frame = True
        try:
            self.live.start(0)
        except Exception as exc:  # noqa: BLE001 - no camera is a sentence, not a crash
            self._leave_live()
            self.reporter.error("Webcam unavailable", str(exc))
            self.set_status("No webcam.")
            return
        self.set_status("Webcam — waiting for the first pose ...")

    def _on_live_frame(self, bgr: object, view: object) -> None:
        import numpy as np

        if not isinstance(view, PoseResult):
            return
        image = np.asarray(bgr, dtype=np.uint8)
        if self._live_first_frame:
            self._live_first_frame = False
            self.canvas.set_image(image)
            self.canvas.fit_to_window()
        else:
            self.canvas.replace_image(image)
        count = len(view.frames[0].people)
        if count != self.people.count:
            self.people.set_people(count)
            self._person = None
        self._result = view
        self.canvas_overlay.set_result(view, self._effective_selection(), self._confidence_threshold)
        self._refresh_3d()
        noun = "person" if count == 1 else "people"
        self.set_status(f"Webcam · {count} {noun} · {view.model}")

    def _on_live_failed(self, message: str) -> None:
        self._leave_live()
        self.reporter.error("Webcam stopped", message)
        self.set_status("Webcam stopped.")

    def open_image(self, path: Path) -> None:
        """Show the image immediately, then detect in the background."""
        from alchemypose.media.image import load_image

        self._leave_video()
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

    # ------------------------------------------------------------------ video

    def open_video(self, path: Path) -> None:
        """Show frame 0 at once, analyse every frame in the background, play meanwhile."""
        self._leave_live()
        self.robot.cancel()
        self._result = None
        self._person = None
        self._pending_source = str(path)
        self.people.set_people(0)
        self.inspector.clear()
        self.selection.clear()
        self._video_first_frame = True
        # Shown before the first frame arrives, so that frame is fitted to the
        # canvas that is left once the strip has taken its height.
        self.transport.show()
        try:
            info = self.video.open(path)
        except AlchemyPoseError as exc:
            self._video_first_frame = False
            self.transport.hide()
            self.reporter.error("Could not open video", str(exc))
            self.set_status("Could not open that file.")
            return
        self.transport.configure(info.frame_count, info.fps)
        self.transport.set_analysis_progress(0, info.frame_count)
        self.transport.show()
        self.set_status(
            f"{path.name} — {info.width}×{info.height}, {info.fps:g} fps, {info.frame_count} frames · analysing ..."
        )

    def _on_video_frame(self, index: int, bgr: object, view: object) -> None:
        import numpy as np

        image = np.asarray(bgr, dtype=np.uint8)
        if self._video_first_frame:
            self._video_first_frame = False
            self.canvas.set_image(image)
            self.canvas.fit_to_window()
        else:
            self.canvas.replace_image(image)
        self.transport.show_position(index)
        self._pose_robot(index)
        if isinstance(view, PoseResult):
            self._result = view
            self.canvas_overlay.set_result(view, self._effective_selection(), self._confidence_threshold)
        else:
            self._result = None
            self.canvas_overlay.set_result(
                _empty_result(image.shape[1], image.shape[0], self._pending_source or ""),
                self._effective_selection(),
                self._confidence_threshold,
            )
        self._refresh_3d()

    def _on_video_progress(self, done: int, total: int) -> None:
        self.transport.set_analysis_progress(done, total)
        if self.video.analysing:
            self.set_status(f"Analysing {done}/{total} frames ...")

    def _on_video_analysed(self, result: PoseResult, cached: bool) -> None:
        tracks = self.video.track_count
        self.people.set_people(tracks)
        noun = "person" if tracks == 1 else "people"
        how = " (from cache)" if cached else ""
        self.set_status(f"{len(result.frames)} frames · {tracks} {noun} tracked · {result.model}{how}")
        self._retarget_current()

    def _on_video_failed(self, message: str) -> None:
        self.reporter.error("Video analysis failed", message)
        self.set_status("Video analysis failed.")

    def _on_play_toggled(self, playing: bool) -> None:
        if playing:
            self.video.play()
        else:
            self.video.pause()

    def _toggle_play(self) -> None:
        if self.video.is_open:
            self._on_play_toggled(not self.video.playing)

    def _step(self, delta: int) -> None:
        if self.video.is_open:
            self.video.pause()
            self.video.show(self.video.index + delta)

    def _on_result(self, result: PoseResult) -> None:
        """Runs on the GUI thread, via the bridge's queued connection."""
        if self.live.owns(result.source):
            self.live.on_result(result)
            return
        if self.video.is_open or self.live.active:
            return  # a still's straggler arriving after a video or the webcam took over
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
        self._refresh_3d()
        self._retarget_current()
        if count:
            self.canvas_overlay.animate_in()
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
        self._refresh_3d()

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
        self._retarget_current()
        self.set_status(f"Person {index + 1}")

    def _on_everyone_chosen(self) -> None:
        self._person = None
        self._push_selection()
        self._retarget_current()

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
        self._refresh_3d()

    def _on_slider_moved(self, ticks: int) -> None:
        self.set_confidence_threshold(ticks / 100.0)

    # -------------------------------------------------------------- lifecycle

    def closeEvent(self, event: QCloseEvent) -> None:
        """Stop the worker before the widgets it would report to disappear."""
        self.live.stop()
        self._bridge.stop()
        self.video.close()
        super().closeEvent(event)


def _empty_result(width: int, height: int, source: str) -> PoseResult:
    """Nobody, on a frame of this size: what a not-yet-analysed video frame draws."""
    from alchemypose.core.schema import PoseFrame
    from alchemypose.core.topology import COCO_WHOLEBODY_133

    frame = PoseFrame(frame_index=0, timestamp_s=None, image_size=(width, height), people=())
    return PoseResult(topology=COCO_WHOLEBODY_133, frames=(frame,), source=source, model="")
