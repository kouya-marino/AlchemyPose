# src/alchemypose/media/video.py
"""Reading a video the way the rest of the app expects a frame: BGR, upright, by index.

Two backends behind one small interface. **PyAV** is preferred — it is the
``video`` extra — because it seeks frame-accurately: it seeks to the keyframe
at or before the target and decodes forward to the exact presentation
timestamp, so ``read(i)`` is frame ``i`` whatever the GOP structure. **OpenCV**
is the fallback, because ``opencv-python-headless`` is already a base
dependency and ships FFmpeg on every platform this project supports; its
``CAP_PROP_POS_FRAMES`` seek is only as accurate as the container's index, which
is fine for sequential reading and approximate for random access.

A frame index is the only identity a frame has here. Timestamps are derived
from it (``index / fps``) rather than read off each packet, so a result's
``frame_index`` and ``timestamp_s`` always agree with each other and with the
scrubber, even on a variable-frame-rate phone clip where packet times wobble.

Rotation: phone footage is often stored landscape with a display matrix
saying "turn me 90°". PyAV reports that angle per frame and :func:`upright`
applies it; OpenCV applies it itself (``CAP_PROP_ORIENTATION_AUTO``, on by
default since 4.5). Either way a frame arrives the way a player would show it.
Checked on a real file rather than reasoned out: a clip tagged with FFmpeg's
``rotate`` metadata, read through both backends, came back identical to the
frame FFmpeg's own autorotate produced — same shape, marked corner in the same
place — with PyAV reporting ``frame.rotation == 90`` for it.
"""

from __future__ import annotations

import math
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import cv2
import numpy as np
from numpy.typing import NDArray

from alchemypose.errors import AlchemyPoseError

#: File suffixes the window offers as video. Anything FFmpeg can open would
#: work; this is only what the file dialog filters on.
VIDEO_SUFFIXES: tuple[str, ...] = (".mp4", ".mov", ".m4v", ".avi", ".mkv", ".webm")


def is_video_path(path: Path) -> bool:
    return path.suffix.lower() in VIDEO_SUFFIXES


@dataclass(frozen=True)
class VideoInfo:
    """What a video is, measured once when it is opened."""

    path: str
    width: int                 # upright, after rotation
    height: int
    fps: float
    frame_count: int
    backend: str

    @property
    def duration_s(self) -> float:
        return self.frame_count / self.fps if self.fps > 0 else 0.0

    def timestamp_of(self, index: int) -> float:
        return index / self.fps if self.fps > 0 else 0.0


class VideoReader(Protocol):
    """Frames by index. Sequential reads are cheap; random reads seek."""

    @property
    def info(self) -> VideoInfo: ...

    def read(self, index: int) -> NDArray[np.uint8]: ...

    def frames(self) -> Iterator[tuple[int, NDArray[np.uint8]]]: ...

    def close(self) -> None: ...


def upright(bgr: NDArray[np.uint8], rotation_ccw: float) -> NDArray[np.uint8]:
    """Turn a decoded frame the way its display matrix says it should be shown.

    ``rotation_ccw`` is the display matrix's angle as FFmpeg's
    ``av_display_rotation_get`` reports it — counter-clockwise degrees — and
    showing the frame upright means rotating it back by that much, which is
    what ``ffplay``'s autorotate does (``-90`` → turn clockwise). Only quarter
    turns occur in practice; anything else is rounded to the nearest one.
    """
    turn = int(round(-rotation_ccw / 90.0)) % 4       # clockwise quarter turns
    if turn == 0:
        return bgr
    code = {1: cv2.ROTATE_90_CLOCKWISE, 2: cv2.ROTATE_180, 3: cv2.ROTATE_90_COUNTERCLOCKWISE}[turn]
    return np.ascontiguousarray(cv2.rotate(bgr, code))


class _PyAVReader:
    def __init__(self, path: Path) -> None:
        import av

        try:
            self._container: Any = av.open(str(path))
        except Exception as exc:  # noqa: BLE001 - PyAV raises a family of its own errors
            raise AlchemyPoseError(f"could not open a video from {path}: {exc}") from exc
        if not self._container.streams.video:
            self._container.close()
            raise AlchemyPoseError(f"{path} has no video stream")
        self._stream = self._container.streams.video[0]
        self._stream.thread_type = "AUTO"
        rate = self._stream.average_rate or self._stream.guessed_rate
        fps = float(rate) if rate else 0.0
        if fps <= 0:
            self._container.close()
            raise AlchemyPoseError(f"{path} does not declare a frame rate")
        self._fps = fps
        self._time_base = float(self._stream.time_base)
        self._start_pts = int(self._stream.start_time or 0)

        count = int(self._stream.frames or 0)
        if count <= 0 and self._stream.duration:
            count = int(round(float(self._stream.duration * self._stream.time_base) * fps))
        if count <= 0 and self._container.duration:
            count = int(round(self._container.duration / 1_000_000 * fps))

        # Decode the first frame now: it fixes the upright size (rotation can
        # swap width and height) and proves the stream actually decodes.
        self._decoder: Iterator[Any] | None = None
        self._next_index = 0
        first = self._decode_at(0)
        height, width = first.shape[:2]
        self._info = VideoInfo(
            path=str(path), width=width, height=height, fps=fps, frame_count=count, backend="pyav"
        )

    @property
    def info(self) -> VideoInfo:
        return self._info

    def _index_of(self, frame: Any) -> int:
        if frame.pts is None:
            return self._next_index
        seconds = (frame.pts - self._start_pts) * self._time_base
        return int(math.floor(seconds * self._fps + 0.5))

    def _to_bgr(self, frame: Any) -> NDArray[np.uint8]:
        bgr = frame.to_ndarray(format="bgr24")
        rotation = float(getattr(frame, "rotation", 0) or 0)
        return upright(np.asarray(bgr, dtype=np.uint8), rotation)

    def _decode_at(self, index: int) -> NDArray[np.uint8]:
        if self._decoder is None or index < self._next_index or index > self._next_index + 30:
            target_pts = self._start_pts + int(index / self._fps / self._time_base)
            self._container.seek(target_pts, stream=self._stream, backward=True, any_frame=False)
            self._decoder = self._container.decode(self._stream)
            self._next_index = -1
        last: Any = None
        for frame in self._decoder:
            found = self._index_of(frame)
            self._next_index = found + 1
            last = frame
            if found >= index:
                return self._to_bgr(frame)
        self._decoder = None
        if last is not None:            # the request ran past the last frame
            return self._to_bgr(last)
        raise AlchemyPoseError(f"frame {index} is past the end of {self._container.name}")

    def read(self, index: int) -> NDArray[np.uint8]:
        if index < 0:
            raise IndexError(index)
        return self._decode_at(index)

    def frames(self) -> Iterator[tuple[int, NDArray[np.uint8]]]:
        """Every frame from the start, in order — one decode per frame, no seeking."""
        self._container.seek(self._start_pts, stream=self._stream, backward=True, any_frame=False)
        self._decoder = None
        index = 0
        for frame in self._container.decode(self._stream):
            yield index, self._to_bgr(frame)
            index += 1
        self._next_index = index

    def close(self) -> None:
        self._container.close()


class _OpenCVReader:
    def __init__(self, path: Path) -> None:
        self._capture = cv2.VideoCapture(str(path))
        if not self._capture.isOpened():
            raise AlchemyPoseError(f"could not open a video from {path}")
        fps = float(self._capture.get(cv2.CAP_PROP_FPS) or 0.0)
        if fps <= 0:
            self._capture.release()
            raise AlchemyPoseError(f"{path} does not declare a frame rate")
        count = int(self._capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        self._next_index = 0
        first = self.read(0)
        height, width = first.shape[:2]
        self._info = VideoInfo(
            path=str(path), width=width, height=height, fps=fps, frame_count=count, backend="opencv"
        )

    @property
    def info(self) -> VideoInfo:
        return self._info

    def read(self, index: int) -> NDArray[np.uint8]:
        if index < 0:
            raise IndexError(index)
        if index != self._next_index:
            self._capture.set(cv2.CAP_PROP_POS_FRAMES, float(index))
        ok, frame = self._capture.read()
        if not ok or frame is None:
            raise AlchemyPoseError(f"frame {index} could not be read")
        self._next_index = index + 1
        return np.asarray(frame, dtype=np.uint8)

    def frames(self) -> Iterator[tuple[int, NDArray[np.uint8]]]:
        self._capture.set(cv2.CAP_PROP_POS_FRAMES, 0.0)
        index = 0
        while True:
            ok, frame = self._capture.read()
            if not ok or frame is None:
                break
            yield index, np.asarray(frame, dtype=np.uint8)
            index += 1
        self._next_index = index

    def close(self) -> None:
        self._capture.release()


def pyav_available() -> bool:
    try:
        import av  # noqa: F401
    except ImportError:
        return False
    return True


def open_video(path: Path, backend: str | None = None) -> VideoReader:
    """Open ``path`` with PyAV when it is installed, else OpenCV.

    ``backend`` forces one (``"pyav"`` or ``"opencv"``); tests use it to hold
    both to the same contract.
    """
    if not Path(path).exists():
        raise AlchemyPoseError(f"{path} not found")
    choice = backend or ("pyav" if pyav_available() else "opencv")
    if choice == "pyav":
        return _PyAVReader(Path(path))
    if choice == "opencv":
        return _OpenCVReader(Path(path))
    raise ValueError(f"unknown video backend {backend!r}")
