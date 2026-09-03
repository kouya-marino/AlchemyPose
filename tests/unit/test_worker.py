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


#: The name ``InferenceWorker`` gives its thread, which is how a leaked one can
#: be spotted from the outside without reaching into the worker's private state.
_THREAD_NAME = "alchemypose-inference"


def _worker_threads() -> set[threading.Thread]:
    return {t for t in threading.enumerate() if t.name == _THREAD_NAME and t.is_alive()}


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
    """A failing job is reported, and the next one still runs.

    The gate is what makes that a real test. "bad.jpg" has to be *inside* the
    engine before "good.jpg" is queued: submitted back to back, the second
    supersedes the first while it is still in the queue — thread start-up is far
    slower than two calls on the main thread — and the error path never runs at
    all, leaving a test that passes because nothing was tried.
    """
    entered_bad = threading.Event()

    class Boom:
        model_name = "boom"

        def infer(self, image: np.ndarray, source: str) -> PoseResult:
            if source == "bad.jpg":
                entered_bad.set()
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
    assert entered_bad.wait(timeout=5.0), "the failing job never reached the engine"

    worker.submit(_image(), "good.jpg")
    assert recovered.wait(timeout=5.0), "the worker did not survive the failing job"
    worker.stop()
    assert len(errors) == 1
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


def test_a_job_outliving_stop_does_not_leave_a_second_worker_thread() -> None:
    """``stop()`` only asks; a job inside the engine can outlive the join.

    Closing a file while a frame is being inferred, then opening another, is the
    ordinary way to reach this: ``stop()`` gives up waiting, ``start()`` builds a
    fresh thread, and the old one is still busy. If it went back to the queue
    when it finished, two threads would pull from the same single-slot queue —
    two frames inferred at once, and whichever finished second would paint the
    canvas, so the user could be left looking at the frame they scrubbed away
    from. Retiring it is what keeps "one job at a time" true across a restart.

    ``join`` on the straggler is the assertion, not a wait for one: it returns
    only once that thread has actually exited, and if the guard is gone it never
    will, because the thread is parked in ``wait()`` for the next job.
    """
    engine = GatedEngine()
    worker = InferenceWorker(lambda: engine, on_result=lambda r: None, on_error=print)

    before = _worker_threads()
    worker.start()
    worker.submit(_image(), "outlives-stop.jpg")
    assert engine.started.wait(timeout=5.0), "the first job never reached the engine"
    new_threads = _worker_threads() - before
    assert len(new_threads) == 1, f"expected exactly one worker thread, got {new_threads}"
    straggler = new_threads.pop()

    # A zero timeout *is* the straggler case, with none of the wall-clock
    # guessing: the job is provably still inside infer(), so the join is
    # guaranteed to give up on it, which is what a slow frame does to the
    # default 2s join in the field.
    worker.stop(timeout=0.0)
    worker.start()
    engine.release.set()

    try:
        straggler.join(timeout=5.0)
        assert not straggler.is_alive(), "the thread that outlived stop() went back to the queue"
        live = _worker_threads() - before
        assert straggler not in live
        assert len(live) == 1, f"a restart left {len(live)} live worker threads"
    finally:
        worker.stop()


def test_a_job_queued_before_stop_does_not_run_after_a_restart() -> None:
    """``stop()`` empties the queue, and a restart must not replay it.

    A job still queued when inference is stopped has been abandoned — the user
    closed the file, or scrubbed past that frame and then shut the panel. Left in
    place, it is the first thing the next ``start()`` picks up, so a stale frame
    is drawn over whatever image is on the canvas by then, before anything the
    user has actually asked for.

    Queueing before the thread exists is the shortest route to a job that is
    pending but unstarted; behind a job in flight is the same state by a longer
    path. The wait below cannot fail a correct worker: with the queue cleared
    there is no job to run, so nothing can set ``ran_something`` however long it
    waits. It is there to catch a worker that kept the job, which dequeues it on
    its very first loop.
    """
    engine = FakeEngine()
    ran: list[str] = []
    ran_something = threading.Event()
    delivered = threading.Event()

    def on_result(r: PoseResult) -> None:
        ran.append(r.source)
        ran_something.set()
        if r.source == "fresh.jpg":
            delivered.set()

    worker = InferenceWorker(lambda: engine, on_result=on_result, on_error=print)
    worker.submit(_image(), "stale.jpg")
    worker.stop()

    worker.start()
    try:
        assert not ran_something.wait(timeout=0.5), f"a job queued before stop() ran anyway: {ran}"

        worker.submit(_image(), "fresh.jpg")
        assert delivered.wait(timeout=5.0), "the restarted worker never ran the new job"
    finally:
        worker.stop()
    assert ran == ["fresh.jpg"]
    assert engine.seen == ["fresh.jpg"]
