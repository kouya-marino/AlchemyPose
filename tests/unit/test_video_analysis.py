"""The offline pass: every frame, in order, tracked, and cached."""

from pathlib import Path

import numpy as np
import pytest

from alchemypose.core.schema import PoseFrame
from alchemypose.inference import video as video_module
from alchemypose.inference.video import (
    AnalysisCancelled,
    analyse_video,
    boxes_from_frame,
    cached_analysis,
    load_cached,
)
from tests.support.fakes import FakeFrameEngine, FakeReader, walker


@pytest.fixture(autouse=True)
def _private_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Never write analyses into the real user cache."""
    monkeypatch.setenv("ALCHEMYPOSE_MODEL_DIR", str(tmp_path / "cache"))


def test_every_frame_is_analysed_in_order_with_its_own_index_and_time() -> None:
    result = analyse_video(FakeFrameEngine(), FakeReader(frames=12, fps=10.0))
    assert [f.frame_index for f in result.frames] == list(range(12))
    assert [f.timestamp_s for f in result.frames] == pytest.approx([i / 10.0 for i in range(12)])
    assert result.model == "fake-model"
    assert result.source == "fake.mp4"


def test_people_are_tracked_and_the_longest_present_is_track_zero() -> None:
    """The second person appears at frame 8 — first detected later, present less, so track 1."""
    result = analyse_video(FakeFrameEngine(second_from=8), FakeReader(frames=12))
    assert {p.track_id for p in result.frames[0].people} == {0}
    assert sorted(p.track_id for p in result.frames[10].people) == [0, 1]
    walker_ids = {f.people[0].track_id for f in result.frames}
    assert walker_ids == {0}


def test_progress_is_reported_after_every_frame() -> None:
    seen: list[tuple[int, int, int]] = []
    analyse_video(FakeFrameEngine(), FakeReader(frames=5), on_frame=lambda f, d, t: seen.append((f.frame_index, d, t)))
    assert seen == [(0, 1, 5), (1, 2, 5), (2, 3, 5), (3, 4, 5), (4, 5, 5)]


def test_det_interval_detects_every_nth_frame_and_crops_the_rest_from_the_last_pose() -> None:
    engine = FakeFrameEngine()
    analyse_video(engine, FakeReader(frames=7), det_interval=3)
    detected = [index for index, boxes in engine.calls if boxes is None]
    derived = [index for index, boxes in engine.calls if boxes is not None]
    assert detected == [0, 3, 6]
    assert derived == [1, 2, 4, 5]


def test_det_interval_one_runs_the_detector_on_every_frame() -> None:
    engine = FakeFrameEngine()
    analyse_video(engine, FakeReader(frames=4), det_interval=1)
    assert all(boxes is None for _index, boxes in engine.calls)


def test_a_derived_crop_contains_the_keypoints_with_margin() -> None:
    person = walker(0)
    frame = PoseFrame(0, 0.0, (320, 240), (person,))
    ((box, _score),) = boxes_from_frame(frame)
    xs, ys = person.keypoints[:, 0], person.keypoints[:, 1]
    assert box[0] < xs.min() and box[2] > xs.max()
    assert box[1] < ys.min() and box[3] <= 240.0                 # clipped to the frame


def test_should_stop_cancels_between_frames() -> None:
    engine = FakeFrameEngine()
    calls = iter([False, False, True])
    with pytest.raises(AnalysisCancelled):
        analyse_video(engine, FakeReader(frames=10), should_stop=lambda: next(calls))
    assert len(engine.calls) == 2


def test_a_bad_det_interval_is_refused() -> None:
    with pytest.raises(ValueError):
        analyse_video(FakeFrameEngine(), FakeReader(), det_interval=0)


def test_coordinates_are_stored_exactly_as_the_engine_returned_them() -> None:
    result = analyse_video(FakeFrameEngine(), FakeReader(frames=3))
    assert np.array_equal(result.frames[2].people[0].keypoints, walker(2).keypoints)


# ----------------------------------------------------------------------- cache


def _clip(tmp_path: Path) -> Path:
    path = tmp_path / "clip.mp4"
    path.write_bytes(b"stand-in: the key only reads the path, size and mtime")
    return path


def test_a_second_open_comes_from_the_cache_without_building_an_engine(tmp_path: Path) -> None:
    clip = _clip(tmp_path)
    built: list[FakeFrameEngine] = []

    def factory() -> FakeFrameEngine:
        built.append(FakeFrameEngine())
        return built[-1]

    first, cached_first = cached_analysis(clip, factory, FakeReader(frames=6), preset="lite")
    second, cached_second = cached_analysis(clip, factory, FakeReader(frames=6), preset="lite")
    assert (cached_first, cached_second) == (False, True)
    assert len(built) == 1
    assert len(second.frames) == 6
    assert np.allclose(second.frames[4].people[0].keypoints, first.frames[4].people[0].keypoints)
    assert second.frames[4].people[0].track_id == 0


def test_a_different_preset_or_interval_is_a_different_analysis(tmp_path: Path) -> None:
    clip = _clip(tmp_path)
    cached_analysis(clip, FakeFrameEngine, FakeReader(frames=3), preset="lite")
    assert load_cached(clip, "lite", 1) is not None
    assert load_cached(clip, "performance", 1) is None
    assert load_cached(clip, "lite", 2) is None


def test_editing_the_file_invalidates_its_cached_analysis(tmp_path: Path) -> None:
    clip = _clip(tmp_path)
    cached_analysis(clip, FakeFrameEngine, FakeReader(frames=3), preset="lite")
    clip.write_bytes(b"a different video now, of a different length")
    assert load_cached(clip, "lite", 1) is None


def test_a_corrupt_cache_entry_is_a_miss_not_a_crash(tmp_path: Path) -> None:
    clip = _clip(tmp_path)
    target = video_module.analysis_cache_dir() / f"{video_module.analysis_key(clip, 'lite', 1)}.json"
    target.write_text("{ not json")
    assert load_cached(clip, "lite", 1) is None


def test_no_cache_means_no_file_written(tmp_path: Path) -> None:
    clip = _clip(tmp_path)
    cached_analysis(clip, FakeFrameEngine, FakeReader(frames=3), preset="lite", use_cache=False)
    assert not list(video_module.analysis_cache_dir().glob("*.json"))
