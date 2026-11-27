"""Video mode: the controller that owns a clip, and the window that shows it."""

from pathlib import Path

import numpy as np
import pytest

from alchemypose.core.schema import PoseResult
from tests.support.fakes import FakeFrameEngine, fake_reader_factory

pytestmark = pytest.mark.gui


@pytest.fixture(autouse=True)
def _private_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ALCHEMYPOSE_MODEL_DIR", str(tmp_path / "cache"))


def _clip(tmp_path: Path) -> Path:
    path = tmp_path / "clip.mp4"
    path.write_bytes(b"stand-in; the fake reader never opens it")
    return path


@pytest.fixture
def controller(qtbot):  # type: ignore[no-untyped-def]
    from alchemypose.gui.video import VideoController

    made = VideoController(lambda: FakeFrameEngine(second_from=6), "lite", reader_factory=fake_reader_factory(12, 10.0))
    yield made
    made.close()


def _finish(qtbot, controller) -> None:  # type: ignore[no-untyped-def]
    qtbot.waitUntil(lambda: controller.result is not None, timeout=5000)


def test_opening_shows_frame_zero_before_any_analysis(qtbot, controller, tmp_path) -> None:  # type: ignore[no-untyped-def]
    with qtbot.waitSignal(controller.frame_ready, timeout=2000) as blocker:
        controller.open(_clip(tmp_path))
    index, bgr, _view = blocker.args
    assert index == 0
    assert np.asarray(bgr).shape == (240, 320, 3)


def test_the_analysis_finishes_with_every_frame_and_tracks_the_walker(qtbot, controller, tmp_path) -> None:  # type: ignore[no-untyped-def]
    controller.open(_clip(tmp_path))
    _finish(qtbot, controller)
    assert len(controller.result.frames) == 12
    assert controller.track_count == 2         # the second person is in 6 of 12 frames: enough for a chip


def test_after_analysis_index_i_is_track_i_on_every_frame(qtbot, controller, tmp_path) -> None:  # type: ignore[no-untyped-def]
    controller.open(_clip(tmp_path))
    _finish(qtbot, controller)
    early, late = controller.view(2), controller.view(9)
    assert isinstance(early, PoseResult) and isinstance(late, PoseResult)
    assert len(early.frames[0].people) == len(late.frames[0].people) == 2
    assert not early.frames[0].people[1].scores.any(), "track 1 has not arrived yet at frame 2"
    assert late.frames[0].people[1].scores.any()


def test_smoothing_changes_the_view_but_never_the_result(qtbot, controller, tmp_path) -> None:  # type: ignore[no-untyped-def]
    controller.open(_clip(tmp_path))
    _finish(qtbot, controller)
    raw = controller.result.frames[5].people[0].keypoints.copy()
    controller.set_smoothing(True)
    smoothed = controller.view(5).frames[0].people[0].keypoints
    controller.set_smoothing(False)
    unsmoothed = controller.view(5).frames[0].people[0].keypoints
    assert np.array_equal(unsmoothed, raw)
    assert not np.array_equal(smoothed, raw)
    assert np.array_equal(controller.result.frames[5].people[0].keypoints, raw)


def test_show_clamps_to_the_clip(qtbot, controller, tmp_path) -> None:  # type: ignore[no-untyped-def]
    controller.open(_clip(tmp_path))
    controller.show(999)
    assert controller.index == 11
    controller.show(-5)
    assert controller.index == 0


def test_playback_advances_and_stops_at_the_end(qtbot, controller, tmp_path) -> None:  # type: ignore[no-untyped-def]
    controller.open(_clip(tmp_path))
    _finish(qtbot, controller)
    controller.play()
    assert controller.playing
    qtbot.waitUntil(lambda: not controller.playing, timeout=5000)     # 12 frames at 10 fps
    assert controller.index == 11


def test_a_straggling_analysis_from_a_replaced_clip_is_ignored(qtbot, controller, tmp_path) -> None:  # type: ignore[no-untyped-def]
    controller.open(_clip(tmp_path))
    old_generation = controller._generation
    controller.open(_clip(tmp_path))
    _finish(qtbot, controller)
    finished: list[object] = []
    controller.analysis_finished.connect(lambda r, c: finished.append(r))
    controller._on_finished(controller.result, False, old_generation)
    assert finished == []


def test_significant_tracks_drops_flicker() -> None:
    from dataclasses import replace

    from alchemypose.core.schema import PoseFrame
    from alchemypose.gui.video import significant_tracks
    from tests.support.fakes import walker

    frames = []
    for i in range(100):
        people = [replace(walker(i), track_id=0)]
        if i < 3:
            people.append(replace(walker(i, x0=500), track_id=1))     # three frames of a passer-by
        frames.append(PoseFrame(i, i / 30, (640, 480), tuple(people)))
    assert significant_tracks(tuple(frames)) == 1


# ---------------------------------------------------------------- the window


@pytest.fixture
def clip_file(tmp_path: Path) -> Path:
    pytest.importorskip("av")
    from tests.support.videos import write_video

    return write_video(tmp_path / "walk.mp4", frames=12, fps=12)


class _NoStills:
    """The still-image engine; these tests never need a real one built."""

    model_name = "none"

    def infer(self, image: np.ndarray, source: str) -> PoseResult:
        from alchemypose.core.schema import PoseFrame
        from alchemypose.core.topology import COCO_WHOLEBODY_133

        frame = PoseFrame(0, None, (image.shape[1], image.shape[0]), ())
        return PoseResult(topology=COCO_WHOLEBODY_133, frames=(frame,), source=source, model="none")


@pytest.fixture
def window(qtbot, reporter):  # type: ignore[no-untyped-def]
    from alchemypose.gui.main_window import MainWindow

    made = MainWindow(reporter=reporter, engine_factory=_NoStills, video_engine_factory=lambda: FakeFrameEngine())
    qtbot.addWidget(made)
    return made


def test_opening_a_video_shows_the_transport_and_analyses_it(qtbot, window, clip_file) -> None:  # type: ignore[no-untyped-def]
    window.open_path(clip_file)
    assert window.transport.isVisibleTo(window)
    assert window.canvas.image_item is not None
    qtbot.waitUntil(lambda: window.video.result is not None, timeout=5000)
    assert window.people.count == 1
    assert "12 frames" in window.status


def test_scrubbing_draws_that_frames_skeleton(qtbot, window, clip_file) -> None:  # type: ignore[no-untyped-def]
    window.open_path(clip_file)
    qtbot.waitUntil(lambda: window.video.result is not None, timeout=5000)
    window.transport.set_position(7)
    assert window.video.index == 7
    assert window.canvas_overlay.keypoint_item_count > 0


def test_json_export_in_video_mode_is_the_whole_clip_raw(qtbot, window, clip_file, tmp_path) -> None:  # type: ignore[no-untyped-def]
    from alchemypose.core.serialize import from_json

    window.open_path(clip_file)
    qtbot.waitUntil(lambda: window.video.result is not None, timeout=5000)
    write = window._export_writer("json")
    assert write is not None
    out = tmp_path / "clip.json"
    write(out)
    exported = from_json(out.read_text())
    assert len(exported.frames) == 12
    assert exported.frames[3].people[0].track_id == 0
    raw = window.video.result.frames[3].people[0].keypoints
    assert np.allclose(exported.frames[3].people[0].keypoints, raw)


def test_opening_a_photograph_leaves_video_mode(qtbot, window, clip_file, tmp_path) -> None:  # type: ignore[no-untyped-def]
    import cv2

    window.open_path(clip_file)
    photo = tmp_path / "still.png"
    cv2.imwrite(str(photo), np.zeros((50, 60, 3), dtype=np.uint8))
    window.open_path(photo)
    assert not window.video.is_open
    assert not window.transport.isVisibleTo(window)
