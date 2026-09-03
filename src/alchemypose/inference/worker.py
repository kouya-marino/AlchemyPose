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

    Only the *queued* job is superseded. A job that has already begun always
    runs to completion and always delivers its result. Scrubbing a video
    depends on that: cancelling in-flight work would mean the frame the user
    released the scrub bar on gets thrown away mid-inference, leaving the
    canvas stuck on the frame before it.

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
        """Spin up the worker thread; a no-op while one is already running.

        Restartable on purpose. ``stop()`` leaves ``_stopping`` set, so clearing
        it here is what stops a start-after-stop from handing back a thread that
        returns on its first loop — a worker that looks alive but silently drops
        every later submit, with neither a result nor an ``on_error``.
        """
        with self._condition:
            if self._thread is not None:
                return
            self._stopping = False
            thread = threading.Thread(target=self._run, name="alchemypose-inference", daemon=True)
            self._thread = thread
        thread.start()

    def submit(self, image: NDArray[np.uint8], source: str) -> None:
        """Queue a job, discarding any earlier one that has not begun.

        A job already in flight is never cancelled; its result still arrives.
        """
        with self._condition:
            self._pending = _Job(image=image, source=source)
            self._condition.notify_all()

    def stop(self, timeout: float = 2.0) -> None:
        with self._condition:
            self._stopping = True
            self._pending = None
            thread, self._thread = self._thread, None
            self._condition.notify_all()
        if thread is not None:
            thread.join(timeout=timeout)

    def _run(self) -> None:
        me = threading.current_thread()
        while True:
            with self._condition:
                # ``self._thread is not me`` retires a straggler: a job that
                # outlived ``stop()``'s join timeout finishes, comes back round,
                # and finds it is no longer the current thread — so it exits
                # rather than compete for jobs with the thread a later
                # ``start()`` has since created.
                #
                # That guard also bounds the waiters to one. Reaching ``wait()``
                # below requires being ``self._thread``, which is a single slot,
                # and ``stop()`` empties it under the lock — so a straggler
                # returns here instead of waiting, and no second thread is ever
                # parked alongside the current one. Both ``notify_all`` calls
                # therefore have at most one waiter to wake and would behave
                # identically as plain ``notify``. They stay broad because a
                # missed wake-up is a silent hang, and deliberately untested:
                # with one waiter the two spellings cannot be told apart, and a
                # test implying otherwise would only claim coverage it has not
                # got.
                while self._pending is None and not self._stopping and self._thread is me:
                    self._condition.wait()
                if self._stopping or self._thread is not me:
                    return
                job, self._pending = self._pending, None
            assert job is not None

            try:
                if self._engine is None:
                    self._engine = self._engine_factory()
                self._on_result(self._engine.infer(job.image, job.source))
            except Exception as exc:  # noqa: BLE001 - the worker must outlive any one job
                self._on_error(exc)
