"""Webcam mode: live frames through the latest-request-wins worker.

This is the policy the worker was built for. The camera produces frames faster
than a model may consume them, and a frame nobody is looking at any more is
worth dropping, so each new frame supersedes any queued one that has not
started. What reaches the screen is always a frame *together with its own
pose* — the frame is held until its result returns — so the skeleton never
trails behind the picture it is drawn on; when the model is slower than the
camera, the display simply runs at the model's rate.

Capture runs on a plain thread of its own, because ``VideoCapture.read`` blocks
until the next frame and must not do that on the GUI thread. It only reads,
stores and submits; the result comes back through the window's existing
``WorkerBridge`` as an ordinary queued signal.

People are tracked and smoothed as they arrive (``core.tracking.Tracker``,
``core.smoothing.PoseSmoother``). Nothing live is stored.
"""

from __future__ import annotations

import threading
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import replace
from typing import Any, Protocol

import numpy as np
from numpy.typing import NDArray
from PySide6.QtCore import QObject, Signal

from alchemypose.core.schema import PoseResult
from alchemypose.core.smoothing import PoseSmoother
from alchemypose.core.tracking import Tracker

SOURCE_PREFIX = "webcam:"
#: Frames held while their pose is computed. A result for an older frame than
#: this many submissions ago has nothing left to be drawn on, and is dropped.
_HELD_FRAMES = 8


class Capture(Protocol):
    """The slice of ``cv2.VideoCapture`` this needs, so tests can stand one in."""

    def isOpened(self) -> bool: ...  # noqa: N802 - OpenCV's spelling

    def read(self) -> tuple[bool, Any]: ...

    def release(self) -> None: ...


def _open_camera(device: int) -> Capture:
    import cv2

    return cv2.VideoCapture(device)


class LiveController(QObject):
    """Owns the camera while webcam mode is on."""

    frame_ready = Signal(object, object)    # bgr, PoseResult (one frame, tracked, smoothed)
    failed = Signal(str)

    _capture_failed = Signal(str)

    def __init__(
        self,
        submit: Callable[[NDArray[np.uint8], str], None],
        parent: QObject | None = None,
        camera_factory: Callable[[int], Capture] = _open_camera,
    ) -> None:
        super().__init__(parent)
        self._submit = submit
        self._camera_factory = camera_factory
        self._capture: Capture | None = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._held: OrderedDict[int, NDArray[np.uint8]] = OrderedDict()
        self._tracker = Tracker()
        self._smoother = PoseSmoother()
        self._started_at = 0.0
        self._capture_failed.connect(self._on_capture_failed)

    @property
    def active(self) -> bool:
        return self._capture is not None

    def owns(self, source: str) -> bool:
        return source.startswith(SOURCE_PREFIX)

    def start(self, device: int = 0) -> None:
        self.stop()
        capture = self._camera_factory(device)
        if not capture.isOpened():
            capture.release()
            raise RuntimeError(f"camera {device} could not be opened")
        self._capture = capture
        self._stop = threading.Event()
        self._tracker, self._smoother = Tracker(), PoseSmoother()
        with self._lock:
            self._held.clear()
        stop = self._stop

        def run() -> None:
            import time

            started = time.monotonic()
            number = 0
            failures = 0
            while not stop.is_set():
                ok, frame = capture.read()
                if not ok or frame is None:
                    failures += 1
                    if failures > 30:
                        self._capture_failed.emit("the camera stopped delivering frames")
                        return
                    time.sleep(0.01)
                    continue
                failures = 0
                bgr = np.ascontiguousarray(frame, dtype=np.uint8)
                with self._lock:
                    self._held[number] = bgr
                    while len(self._held) > _HELD_FRAMES:
                        self._held.popitem(last=False)
                stamp = time.monotonic() - started
                self._submit(bgr, f"{SOURCE_PREFIX}{number}:{stamp:.4f}")
                number += 1

        self._thread = threading.Thread(target=run, name="alchemypose-webcam", daemon=True)
        self._thread.start()

    def stop(self, timeout: float = 1.0) -> None:
        self._stop.set()
        thread, self._thread = self._thread, None
        if thread is not None:
            thread.join(timeout=timeout)
        if self._capture is not None:
            self._capture.release()
        self._capture = None

    def on_result(self, result: PoseResult) -> None:
        """A pose for one held frame. Runs on the GUI thread."""
        if not self.active or not result.frames:
            return
        try:
            _, number_text, stamp_text = result.source.split(":")
            number, stamp = int(number_text), float(stamp_text)
        except ValueError:
            return
        with self._lock:
            bgr = self._held.pop(number, None)
            for older in [n for n in self._held if n < number]:
                del self._held[older]          # superseded: they will never get a pose
        if bgr is None:
            return
        frame = replace(result.frames[0], frame_index=number, timestamp_s=stamp)
        ids = self._tracker.update(frame.people)
        frame = replace(frame, people=tuple(replace(p, track_id=t) for p, t in zip(frame.people, ids, strict=True)))
        frame = self._smoother.frame(frame, fps=30.0)
        self.frame_ready.emit(bgr, replace(result, frames=(frame,)))

    def _on_capture_failed(self, message: str) -> None:
        self.stop()
        self.failed.emit(message)
