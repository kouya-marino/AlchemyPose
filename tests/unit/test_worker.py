import threading
import time

import numpy as np

from alchemypose.core.schema import PoseFrame, PoseResult
from alchemypose.core.topology import COCO_WHOLEBODY_133
from alchemypose.inference.worker import InferenceWorker


class FakeEngine:
    model_name = "fake"

    def __init__(self, delay: float = 0.0) -> None:
        self.delay = delay
        self.seen: list[str] = []

    def infer(self, image: np.ndarray, source: str) -> PoseResult:
        time.sleep(self.delay)
        self.seen.append(source)
        frame = PoseFrame(frame_index=0, timestamp_s=None, image_size=(1, 1), people=())
        return PoseResult(topology=COCO_WHOLEBODY_133, frames=(frame,), source=source, model="fake")


class GatedEngine(FakeEngine):
    """A FakeEngine whose *first* job parks inside ``infer`` until released.

    That is what makes the supersession test deterministic rather than a race
    against thread start-up: the test can prove a job is in flight before it
    queues the ones behind it, instead of sleeping and hoping.
    """

    def __init__(self) -> None:
        super().__init__()
        self.started = threading.Event()
        self.release = threading.Event()

    def infer(self, image: np.ndarray, source: str) -> PoseResult:
        if not self.started.is_set():
            self.started.set()
            if not self.release.wait(timeout=5.0):
                raise AssertionError("the test never released the first job")
        return super().infer(image, source)


def _image() -> np.ndarray:
    return np.zeros((4, 4, 3), dtype=np.uint8)


def test_result_is_delivered_to_the_callback() -> None:
    done = threading.Event()
    got: list[PoseResult] = []
    worker = InferenceWorker(FakeEngine, on_result=lambda r: (got.append(r), done.set()), on_error=print)
    worker.start()
    worker.submit(_image(), "a.jpg")
    assert done.wait(timeout=5.0)
    worker.stop()
    assert got[0].source == "a.jpg"


def test_a_later_submit_supersedes_an_unstarted_one() -> None:
    """Latest-wins applies to the *queue* only; work already in flight finishes.

    Both halves matter and only the exact sequence pins them both down. An
    assertion that merely says "second.jpg" is absent also passes for a worker
    that throws away the running job — which is how a scrub-bar release would
    lose the very frame the user asked for and leave the canvas a frame behind.

    The gate removes the timing luck: "first.jpg" is provably inside ``infer``
    before "second.jpg" and "third.jpg" are queued, so the worker has to make a
    real supersession decision rather than collapsing three submits into one
    before its thread has dequeued anything.
    """
    engine = GatedEngine()
    done = threading.Event()
    results: list[str] = []

    def on_result(r: PoseResult) -> None:
        results.append(r.source)
        if r.source == "third.jpg":
            done.set()

    worker = InferenceWorker(lambda: engine, on_result=on_result, on_error=print)
    worker.start()
    worker.submit(_image(), "first.jpg")
    assert engine.started.wait(timeout=5.0), "the first job never reached the engine"

    worker.submit(_image(), "second.jpg")
    worker.submit(_image(), "third.jpg")
    engine.release.set()

    assert done.wait(timeout=5.0)
    worker.stop()
    assert results == ["first.jpg", "third.jpg"]
    assert engine.seen == ["first.jpg", "third.jpg"]


def test_an_engine_error_reaches_on_error_and_the_worker_survives() -> None:
    class Boom:
        model_name = "boom"

        def infer(self, image: np.ndarray, source: str) -> PoseResult:
            if source == "bad.jpg":
                raise RuntimeError("kaboom")
            frame = PoseFrame(frame_index=0, timestamp_s=None, image_size=(1, 1), people=())
            return PoseResult(topology=COCO_WHOLEBODY_133, frames=(frame,), source=source, model="boom")

    errors: list[Exception] = []
    recovered = threading.Event()
    worker = InferenceWorker(
        Boom, on_result=lambda r: recovered.set(), on_error=lambda e: errors.append(e)
    )
    worker.start()
    worker.submit(_image(), "bad.jpg")
    # Give the worker thread real wall-clock time to spin up and dequeue "bad.jpg"
    # before the next submit arrives — thread creation is far slower than two
    # back-to-back Python calls on the main thread, so without this pause
    # "good.jpg" reliably (not just occasionally) supersedes "bad.jpg" before it
    # ever starts, and the error path this test targets never runs.
    time.sleep(0.05)
    worker.submit(_image(), "good.jpg")
    assert recovered.wait(timeout=5.0)
    worker.stop()
    assert isinstance(errors[0], RuntimeError)


def test_stop_is_idempotent() -> None:
    worker = InferenceWorker(FakeEngine, on_result=print, on_error=print)
    worker.start()
    worker.stop()
    worker.stop()


def test_a_worker_started_again_after_stop_still_runs_jobs() -> None:
    """A restarted worker must work, not wedge.

    A stop flag left set outranks the new thread's first loop, so ``start()``
    would hand back a thread that returns immediately: the worker then looks
    alive and swallows every submit in silence — no result, no ``on_error``,
    nothing to debug from.
    """
    engine = FakeEngine()
    done = threading.Event()
    got: list[str] = []
    worker = InferenceWorker(
        lambda: engine, on_result=lambda r: (got.append(r.source), done.set()), on_error=print
    )
    worker.start()
    worker.stop()

    worker.start()
    try:
        worker.submit(_image(), "after-restart.jpg")
        assert done.wait(timeout=5.0), "a worker restarted after stop() never ran the job"
    finally:
        worker.stop()
    assert got == ["after-restart.jpg"]
