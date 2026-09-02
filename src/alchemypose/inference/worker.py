# src/alchemypose/inference/worker.py
"""Inference off the UI thread.

No Qt here on purpose: this is plain threading, so it is testable headless.
``gui/bridge.py`` adapts the callbacks to Qt signals.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from alchemypose.core.schema import PoseResult
from alchemypose.inference.engine import PoseEngine


@dataclass(frozen=True)
class _Job:
    image: NDArray[np.uint8]
    source: str


class InferenceWorker:
    """Runs one job at a time; a new submission replaces any job not yet started.

    The engine is constructed lazily *on the worker thread*, so the first
    (slow, CoreML-compiling) session creation never blocks the UI.
    """

    def __init__(
        self,
        engine_factory: Callable[[], PoseEngine],
        on_result: Callable[[PoseResult], None],
        on_error: Callable[[Exception], None],
    ) -> None:
        self._engine_factory = engine_factory
        self._on_result = on_result
        self._on_error = on_error

        self._engine: PoseEngine | None = None
        self._pending: _Job | None = None
        self._condition = threading.Condition()
        self._stopping = False
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._run, name="alchemypose-inference", daemon=True)
        self._thread.start()

    def submit(self, image: NDArray[np.uint8], source: str) -> None:
        """Queue a job, discarding any earlier one that has not begun."""
        with self._condition:
            self._pending = _Job(image=image, source=source)
            self._condition.notify()

    def stop(self, timeout: float = 2.0) -> None:
        with self._condition:
            self._stopping = True
            self._condition.notify()
        thread, self._thread = self._thread, None
        if thread is not None:
            thread.join(timeout=timeout)

    def _run(self) -> None:
        while True:
            with self._condition:
                while self._pending is None and not self._stopping:
                    self._condition.wait()
                if self._stopping:
                    return
                job, self._pending = self._pending, None
            assert job is not None

            try:
                if self._engine is None:
                    self._engine = self._engine_factory()
                self._on_result(self._engine.infer(job.image, job.source))
            except Exception as exc:  # noqa: BLE001 - the worker must outlive any one job
                self._on_error(exc)
