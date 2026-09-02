import threading

import numpy as np
import pytest

from alchemypose.core.schema import PoseFrame, PoseResult
from alchemypose.core.topology import COCO_WHOLEBODY_133

pytestmark = pytest.mark.gui


class FakeEngine:
    model_name = "fake"

    def __init__(self) -> None:
        self.built_on = threading.current_thread().name

    def infer(self, image: np.ndarray, source: str) -> PoseResult:  # type: ignore[type-arg]
        frame = PoseFrame(frame_index=0, timestamp_s=None, image_size=(4, 4), people=())
        return PoseResult(topology=COCO_WHOLEBODY_133, frames=(frame,), source=source, model=self.built_on)


class BoomEngine:
    model_name = "boom"

    def infer(self, image: np.ndarray, source: str) -> PoseResult:  # type: ignore[type-arg]
        raise RuntimeError("no weights")


def _image() -> np.ndarray:  # type: ignore[type-arg]
    return np.zeros((4, 4, 3), dtype=np.uint8)


def test_a_result_arrives_on_the_gui_thread_as_a_signal(qtbot) -> None:  # type: ignore[no-untyped-def]
    from alchemypose.gui.bridge import WorkerBridge

    bridge = WorkerBridge(FakeEngine)
    bridge.start()
    delivered_on: list[str] = []
    bridge.result_ready.connect(lambda _r: delivered_on.append(threading.current_thread().name))
    with qtbot.waitSignal(bridge.result_ready, timeout=5000) as blocker:
        bridge.submit(_image(), "a.jpg")
    bridge.stop()

    result = blocker.args[0]
    assert result.source == "a.jpg"
    # The engine was built on the worker thread ...
    assert result.model == "alchemypose-inference"
    # ... and the slot ran on this (the GUI) thread, courtesy of the queued connection.
    assert delivered_on == [threading.current_thread().name]


def test_an_engine_failure_arrives_as_a_message_not_an_exception(qtbot) -> None:  # type: ignore[no-untyped-def]
    from alchemypose.gui.bridge import WorkerBridge

    bridge = WorkerBridge(BoomEngine)
    bridge.start()
    with qtbot.waitSignal(bridge.failed, timeout=5000) as blocker:
        bridge.submit(_image(), "a.jpg")
    bridge.stop()
    assert "no weights" in blocker.args[0]


def test_the_engine_is_not_built_until_the_first_job(qtbot) -> None:  # type: ignore[no-untyped-def]
    from alchemypose.gui.bridge import WorkerBridge

    built = 0

    def factory() -> FakeEngine:
        nonlocal built
        built += 1
        return FakeEngine()

    bridge = WorkerBridge(factory)
    bridge.start()
    assert built == 0
    with qtbot.waitSignal(bridge.result_ready, timeout=5000):
        bridge.submit(_image(), "a.jpg")
    bridge.stop()
    assert built == 1


def test_stop_ends_the_worker_thread_and_reports_not_running(qtbot) -> None:  # type: ignore[no-untyped-def]
    from alchemypose.gui.bridge import WorkerBridge

    bridge = WorkerBridge(FakeEngine)
    assert bridge.running is False
    bridge.start()
    assert bridge.running is True
    assert any(t.name == "alchemypose-inference" for t in threading.enumerate())
    bridge.stop()
    assert bridge.running is False
    assert not any(t.name == "alchemypose-inference" for t in threading.enumerate())


def test_a_bridge_whose_owner_is_destroyed_without_stop_still_ends_its_thread(qtbot) -> None:  # type: ignore[no-untyped-def]
    """Owners are not always closed politely; Qt's ``destroyed`` must stop the worker anyway.

    The worker thread keeps the bridge's callbacks (and so the bridge) alive, so
    a *parentless* bridge can only ever be stopped explicitly. The case that
    matters is the real one: a bridge parented to a window that gets destroyed.
    """
    import gc

    from PySide6.QtCore import QObject

    from alchemypose.gui.bridge import WorkerBridge

    owner = QObject()
    bridge = WorkerBridge(FakeEngine, parent=owner)
    bridge.start()
    try:
        assert any(t.name == "alchemypose-inference" for t in threading.enumerate())
        del owner
        gc.collect()
        assert bridge.running is False
        assert not any(t.name == "alchemypose-inference" for t in threading.enumerate())
    finally:
        bridge.stop()   # only matters if the assertion above failed; never leak a thread into the next test
