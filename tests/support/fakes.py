"""Stand-ins for an engine and a video reader, for tests that must not load a model."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import numpy as np

from alchemypose.core.schema import PersonPose, PoseFrame
from alchemypose.media.video import VideoInfo

K = 133
BBox = tuple[float, float, float, float]


def walker(index: int, x0: float = 100.0, step: float = 4.0) -> PersonPose:
    """A person whose keypoints span a box that moves ``step`` px right per frame."""
    rng = np.random.default_rng(index)
    x = x0 + step * index
    points = np.column_stack([x + rng.uniform(0, 80, K), 50 + rng.uniform(0, 160, K)]).astype(np.float32)
    points[0], points[1] = (x, 50.0), (x + 80.0, 210.0)
    return PersonPose(
        keypoints=points, scores=np.full(K, 0.9, dtype=np.float32), bbox=(x, 50.0, x + 80.0, 210.0), bbox_score=0.9
    )


class FakeFrameEngine:
    """Reports one walker per frame (two from ``second_from``), and records how it was called."""

    model_name = "fake-model"

    def __init__(self, second_from: int | None = None) -> None:
        self.calls: list[tuple[int, list[tuple[BBox, float]] | None]] = []
        self.second_from = second_from

    def infer_frame(
        self,
        image: np.ndarray,
        frame_index: int = 0,
        timestamp_s: float | None = None,
        boxes: list[tuple[BBox, float]] | None = None,
    ) -> PoseFrame:
        self.calls.append((frame_index, boxes))
        people = [walker(frame_index)]
        if self.second_from is not None and frame_index >= self.second_from:
            people.append(walker(frame_index, x0=600.0, step=-2.0))
        height, width = image.shape[:2]
        return PoseFrame(frame_index, timestamp_s, (width, height), tuple(people))


class FakeReader:
    """``frames`` synthetic 320x240 frames at ``fps``."""

    def __init__(self, frames: int = 12, fps: float = 10.0, path: str = "fake.mp4") -> None:
        self._info = VideoInfo(path=path, width=320, height=240, fps=fps, frame_count=frames, backend="fake")
        self.closed = False
        self.reads: list[int] = []

    @property
    def info(self) -> VideoInfo:
        return self._info

    def read(self, index: int) -> np.ndarray:
        self.reads.append(index)
        return np.full((240, 320, 3), index % 256, dtype=np.uint8)

    def frames(self) -> Iterator[tuple[int, np.ndarray]]:
        for index in range(self._info.frame_count):
            yield index, self.read(index)

    def close(self) -> None:
        self.closed = True


def fake_reader_factory(frames: int = 12, fps: float = 10.0):  # type: ignore[no-untyped-def]
    def factory(path: Path) -> FakeReader:
        return FakeReader(frames=frames, fps=fps, path=str(path))

    return factory
