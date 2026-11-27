"""Both video backends, held to the same contract over a generated clip."""

from pathlib import Path

import numpy as np
import pytest

from alchemypose.errors import AlchemyPoseError
from alchemypose.media.video import is_video_path, open_video, upright
from tests.support.videos import HEIGHT, WIDTH, index_of, write_video

pytest.importorskip("av")

FRAMES, FPS = 24, 12


@pytest.fixture(scope="module")
def clip(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return write_video(tmp_path_factory.mktemp("clip") / "clip.mp4", frames=FRAMES, fps=FPS, gop=6)


@pytest.mark.parametrize("backend", ["pyav", "opencv"])
def test_info_describes_the_clip(clip: Path, backend: str) -> None:
    reader = open_video(clip, backend=backend)
    try:
        info = reader.info
        assert (info.width, info.height) == (WIDTH, HEIGHT)
        assert info.fps == pytest.approx(FPS)
        assert info.frame_count == FRAMES
        assert info.backend == backend
        assert info.duration_s == pytest.approx(FRAMES / FPS)
        assert info.timestamp_of(6) == pytest.approx(0.5)
    finally:
        reader.close()


@pytest.mark.parametrize("backend", ["pyav", "opencv"])
def test_sequential_frames_are_every_frame_in_order(clip: Path, backend: str) -> None:
    reader = open_video(clip, backend=backend)
    try:
        seen = [(index, index_of(frame)) for index, frame in reader.frames()]
    finally:
        reader.close()
    assert [i for i, _ in seen] == list(range(FRAMES))
    assert [found for _, found in seen] == list(range(FRAMES))


@pytest.mark.parametrize("backend", ["pyav", "opencv"])
def test_frames_are_bgr_uint8_at_the_upright_size(clip: Path, backend: str) -> None:
    reader = open_video(clip, backend=backend)
    try:
        frame = reader.read(0)
    finally:
        reader.close()
    assert frame.dtype == np.uint8
    assert frame.shape == (HEIGHT, WIDTH, 3)


def test_pyav_random_access_is_frame_accurate_in_any_order(clip: Path) -> None:
    """Backwards, across keyframes, mid-GOP, and repeated — each read is the frame asked for."""
    reader = open_video(clip, backend="pyav")
    try:
        for index in (17, 3, 4, 5, 23, 0, 11, 11, 12, 6, 22, 1):
            assert index_of(reader.read(index)) == index, f"asked for {index}"
    finally:
        reader.close()


def test_pyav_reading_past_the_end_returns_the_last_frame(clip: Path) -> None:
    reader = open_video(clip, backend="pyav")
    try:
        assert index_of(reader.read(FRAMES + 50)) == FRAMES - 1
    finally:
        reader.close()


def test_a_missing_file_is_a_sentence(tmp_path: Path) -> None:
    with pytest.raises(AlchemyPoseError, match="not found"):
        open_video(tmp_path / "nope.mp4")


@pytest.mark.parametrize("backend", ["pyav", "opencv"])
def test_a_file_that_is_not_a_video_is_a_sentence(tmp_path: Path, backend: str) -> None:
    bogus = tmp_path / "bogus.mp4"
    bogus.write_bytes(b"this is not a video")
    with pytest.raises(AlchemyPoseError):
        open_video(bogus, backend=backend)


def test_video_paths_are_recognised_by_suffix() -> None:
    assert is_video_path(Path("a/b/Dance.MP4"))
    assert is_video_path(Path("clip.webm"))
    assert not is_video_path(Path("photo.jpg"))


# --------------------------------------------------------------------- rotation


def _marked() -> np.ndarray:
    """A 2x3 frame whose top-left pixel is the only white one."""
    frame = np.zeros((2, 3, 3), dtype=np.uint8)
    frame[0, 0] = 255
    return frame


def test_no_rotation_is_untouched() -> None:
    frame = _marked()
    assert upright(frame, 0.0) is frame


def test_a_minus_90_display_matrix_turns_the_frame_clockwise() -> None:
    """FFmpeg reports a phone's portrait clip as -90 (counter-clockwise degrees);
    ffplay shows it turned clockwise, so the top-left corner moves to top-right."""
    turned = upright(_marked(), -90.0)
    assert turned.shape[:2] == (3, 2)
    assert turned[0, 1].tolist() == [255, 255, 255]


def test_a_plus_90_display_matrix_turns_the_frame_counter_clockwise() -> None:
    turned = upright(_marked(), 90.0)
    assert turned.shape[:2] == (3, 2)
    assert turned[2, 0].tolist() == [255, 255, 255]


def test_a_half_turn_puts_the_corner_bottom_right() -> None:
    turned = upright(_marked(), 180.0)
    assert turned[1, 2].tolist() == [255, 255, 255]
