"""The G1 in the window: fetch its model, retarget a clip to it, pose it per frame.

The retarget runs once per clip and person, in the background — 300 frames
take tens of seconds — and playback then reads joint angles by frame index,
exactly as the skeleton reads keypoints. Threading follows ``gui/bridge.py``:
the worker thread only emits; every slot runs on the GUI thread.
"""

from __future__ import annotations

import threading

import numpy as np
from PySide6.QtCore import QObject, Signal

from alchemypose.core.schema import PoseFrame
from alchemypose.robot.retarget import Retargeter, RobotMotion, SpeedReport, speed_report
from alchemypose.robot.urdf import Robot


class RobotController(QObject):
    """Owns the G1 model and the motion retargeted onto it."""

    model_ready = Signal(object, object)            # Robot, prepared meshes
    progress = Signal(int, int)
    motion_ready = Signal(object, object)           # RobotMotion, SpeedReport
    failed = Signal(str)

    _model_loaded = Signal(object, object, int)
    _retargeted = Signal(object, int)
    _failed = Signal(str, int)

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self.robot: Robot | None = None
        self.meshes: list[object] | None = None
        self.motion: RobotMotion | None = None
        self.report: SpeedReport | None = None
        self._generation = 0
        self._stop = threading.Event()
        self._model_loaded.connect(self._on_model_loaded)
        self._retargeted.connect(self._on_retargeted)
        self._failed.connect(self._on_failed)

    @property
    def busy(self) -> bool:
        return self.motion is None and self._generation > 0 and not self._stop.is_set()

    def load_model(self) -> None:
        """Fetch (first time only, 19 MB) and parse the G1, off the GUI thread."""
        if self.robot is not None:
            self.model_ready.emit(self.robot, self.meshes)
            return
        generation = self._generation

        def run() -> None:
            try:
                from alchemypose.gui.view3d.robot import prepare_meshes
                from alchemypose.robot import assets
                from alchemypose.robot.urdf import load_urdf

                root = assets.ensure_g1()
                robot = load_urdf(root / assets.URDF)
                self._model_loaded.emit(robot, prepare_meshes(robot, root), generation)
            except Exception as exc:  # noqa: BLE001 - reported, never raised into Qt
                self._failed.emit(f"could not load the Unitree G1 model: {exc}", generation)

        threading.Thread(target=run, name="alchemypose-g1-model", daemon=True).start()

    def retarget(self, frames: tuple[PoseFrame, ...], track_id: int) -> None:
        """Retarget ``track_id`` through ``frames`` in the background; supersedes any run in flight."""
        robot = self.robot
        if robot is None:
            return
        self._stop.set()
        self._stop = threading.Event()
        self._generation += 1
        self.motion = None
        self.report = None
        generation, stop = self._generation, self._stop

        def run() -> None:
            try:
                retargeter = Retargeter(robot)

                def tick(done: int, total: int) -> None:
                    if stop.is_set():
                        raise _Superseded()
                    if done % 10 == 0 or done == total:
                        self.progress.emit(done, total)

                motion = retargeter.retarget(frames, track_id=track_id, on_progress=tick)
            except _Superseded:
                return
            except Exception as exc:  # noqa: BLE001
                self._failed.emit(f"retargeting failed: {exc}", generation)
                return
            self._retargeted.emit(motion, generation)

        threading.Thread(target=run, name="alchemypose-retarget", daemon=True).start()

    def cancel(self) -> None:
        self._stop.set()
        self.motion = None
        self.report = None

    def pose(self, index: int) -> tuple[np.ndarray, np.ndarray, float] | None:
        motion = self.motion
        if motion is None or not len(motion.q):
            return None
        i = max(0, min(index, len(motion.q) - 1))
        return motion.q[i], motion.root_rotation[i], float(motion.root_height[i])

    def _on_model_loaded(self, robot: Robot, meshes: list[object], generation: int) -> None:
        self.robot, self.meshes = robot, meshes
        self.model_ready.emit(robot, meshes)

    def _on_retargeted(self, motion: RobotMotion, generation: int) -> None:
        if generation != self._generation or self.robot is None:
            return
        self.motion = motion
        self.report = speed_report(motion, np.array([j.velocity for j in self.robot.actuated]))
        self.motion_ready.emit(motion, self.report)

    def _on_failed(self, message: str, generation: int) -> None:
        if generation == self._generation:
            self.failed.emit(message)


class _Superseded(Exception):
    """A newer retarget replaced this one."""
