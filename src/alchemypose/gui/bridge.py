"""Thread-safe hand-off from the inference worker to the GUI.

``InferenceWorker`` knows nothing about Qt, which is what makes it testable
headless. This object is the only place the two worlds meet.

The worker invokes its callbacks *on its own thread*. Emitting a Qt signal is
the only thread-safe thing to do there: a signal emitted from a non-GUI thread
towards a receiver that lives on the GUI thread is delivered through Qt's
queued-connection machinery, so the connected slots run on the GUI thread on
the next event-loop pass. Nothing in this file touches a widget, and nothing
connected to these signals ever runs on the worker thread.
"""

from __future__ import annotations

import threading
from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray
from PySide6.QtCore import QObject, Signal

from alchemypose.core.schema import PoseResult
from alchemypose.inference.engine import PoseEngine
from alchemypose.inference.worker import InferenceWorker


class WorkerBridge(QObject):
    """Wraps an ``InferenceWorker``, turning its callbacks into Qt signals.

    ``engine_factory`` is called lazily, on the worker thread, the first time a
    job runs — constructing a real engine costs seconds (CoreML compilation),
    and that must never happen on the GUI thread or at window construction.
    """

    result_ready = Signal(object)     # PoseResult
    failed = Signal(str)

    def __init__(self, engine_factory: Callable[[], PoseEngine], parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._alive = threading.Event()
        self._worker = InferenceWorker(
            engine_factory,
            on_result=self._emit_result,
            on_error=self._emit_error,
        )
        # Backstop for an owner that is destroyed without ever being closed
        # (a window dropped by tests, or deleted by its parent): Qt still emits
        # ``destroyed``, and this closure references only plain Python objects
        # — never the bridge itself — so it neither forms a cycle nor touches a
        # dead C++ object. The ordinary path is the owner calling ``stop()``.
        alive, worker = self._alive, self._worker

        def _on_destroyed() -> None:
            alive.clear()
            worker.stop()

        self.destroyed.connect(_on_destroyed)

    @property
    def running(self) -> bool:
        """``True`` between ``start()`` and ``stop()``."""
        return self._alive.is_set()

    # ------------------------------------------------------- worker callbacks
    # Both run on the worker thread. They may only emit.

    def _emit_result(self, result: PoseResult) -> None:
        if self._alive.is_set():
            self.result_ready.emit(result)

    def _emit_error(self, exc: Exception) -> None:
        if self._alive.is_set():
            self.failed.emit(str(exc))

    # -------------------------------------------------------------- lifecycle

    def start(self) -> None:
        self._alive.set()
        self._worker.start()

    def stop(self) -> None:
        """Stop the worker and refuse to emit anything that finishes late.

        ``InferenceWorker.stop`` joins with a timeout; a job that is mid-way
        through building an engine can outlive it. Clearing ``_alive`` first
        means such a straggler finds nobody to tell, instead of emitting on a
        bridge whose owning window may already be gone.
        """
        self._alive.clear()
        self._worker.stop()

    def submit(self, image: NDArray[np.uint8], source: str) -> None:
        self._worker.submit(image, source)
