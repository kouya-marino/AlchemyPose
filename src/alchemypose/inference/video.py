# src/alchemypose/inference/video.py
"""A whole video in, one multi-frame ``PoseResult`` out.

This is the *offline* pass, and it is a different policy from the worker's on
purpose. ``InferenceWorker`` is latest-request-wins — right for scrubbing a
still or following a webcam, where a frame nobody is looking at any more is
worth dropping. Playing a clip back beside its 3D skeleton and a robot needs
the opposite: every frame, in order, analysed once and kept. So this walks the
video sequentially, runs the engine on each frame, tracks people across them,
and returns the lot; playback then reads from the result by ``frame_index``
and never waits on a model.

Analyses are cached on disk (:func:`cached_analysis`), keyed on the file's
path, size and modification time and on how it was analysed, so reopening the
same clip — the common case while working on one dance — costs a JSON read
rather than another pass through the models.

Coordinates stay raw here, as everywhere they are stored. Smoothing happens
when drawing (``core.smoothing``).
"""

from __future__ import annotations

import hashlib
import logging
import os
import tempfile
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from alchemypose import __version__
from alchemypose.core.schema import Camera, PoseFrame, PoseResult
from alchemypose.core.serialize import from_json, to_json
from alchemypose.core.topology import COCO_WHOLEBODY_133
from alchemypose.core.tracking import Tracker, keypoint_box, renumber_by_presence
from alchemypose.inference import weights
from alchemypose.media.video import VideoReader

_log = logging.getLogger(__name__)

BBox = tuple[float, float, float, float]

#: How much bigger than the keypoints' own extent a derived crop is. The body
#: keypoints stop at the joints, so the crop needs margin for hands, feet and
#: the top of the head — 1.25 is rtmlib's figure for the same job.
DERIVED_BOX_SCALE = 1.25


class FrameEngine(Protocol):
    """What the offline pass needs from an engine: per-frame inference, optional boxes."""

    @property
    def model_name(self) -> str: ...

    def infer_frame(
        self,
        image: NDArray[np.uint8],
        frame_index: int = 0,
        timestamp_s: float | None = None,
        boxes: list[tuple[BBox, float]] | None = None,
    ) -> PoseFrame: ...


def boxes_from_frame(frame: PoseFrame, scale: float = DERIVED_BOX_SCALE) -> list[tuple[BBox, float]]:
    """Crops for the next frame, from this frame's keypoints, scaled about their centre."""
    width, height = frame.image_size
    out: list[tuple[BBox, float]] = []
    for person in frame.people:
        box = keypoint_box(person)
        if box is None:
            continue
        cx, cy = (box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0
        half_w, half_h = (box[2] - box[0]) * scale / 2.0, (box[3] - box[1]) * scale / 2.0
        derived = (
            max(0.0, cx - half_w), max(0.0, cy - half_h),
            min(float(width), cx + half_w), min(float(height), cy + half_h),
        )
        if derived[2] - derived[0] >= 8 and derived[3] - derived[1] >= 8:
            out.append((derived, person.bbox_score if person.bbox_score is not None else 1.0))
    return out


class AnalysisCancelled(Exception):
    """Raised out of :func:`analyse_video` when ``should_stop`` says so."""


def analyse_video(
    engine: FrameEngine,
    reader: VideoReader,
    *,
    det_interval: int = 1,
    on_frame: Callable[[PoseFrame, int, int], None] | None = None,
    should_stop: Callable[[], bool] | None = None,
) -> PoseResult:
    """Run ``engine`` over every frame of ``reader`` and track people through them.

    ``det_interval`` runs the detector on every n-th frame and crops the frames
    in between from the previous frame's keypoints; 1 detects every frame. A
    frame whose derived crops come back empty falls back to the detector.

    ``on_frame(frame, done, total)`` is called after each frame with that frame
    (track ids provisional — they are renumbered by presence at the end). It
    runs on the caller's thread. ``should_stop`` is polled before each frame.
    """
    if det_interval < 1:
        raise ValueError(f"det_interval must be at least 1, got {det_interval}")
    info = reader.info
    tracker = Tracker()
    frames: list[PoseFrame] = []
    previous: PoseFrame | None = None

    for index, image in reader.frames():
        if should_stop is not None and should_stop():
            raise AnalysisCancelled()
        timestamp = info.timestamp_of(index)
        boxes = None
        if previous is not None and index % det_interval != 0:
            boxes = boxes_from_frame(previous) or None
        frame = engine.infer_frame(image, frame_index=index, timestamp_s=timestamp, boxes=boxes)
        ids = tracker.update(frame.people)
        frame = replace(
            frame, people=tuple(replace(p, track_id=t) for p, t in zip(frame.people, ids, strict=True))
        )
        frames.append(frame)
        previous = frame
        if on_frame is not None:
            on_frame(frame, index + 1, max(info.frame_count, index + 1))

    return PoseResult(
        topology=COCO_WHOLEBODY_133,
        frames=renumber_by_presence(tuple(frames)),
        source=info.path,
        model=engine.model_name,
        camera=Camera.default_for(info.width, info.height),
    )


# ------------------------------------------------------------------- caching


def analysis_key(path: Path, preset: str, det_interval: int) -> str:
    """Identifies one analysis of one version of one file."""
    stat = Path(path).stat()
    material = f"{Path(path).resolve()}|{stat.st_size}|{stat.st_mtime_ns}|{preset}|{det_interval}|{__version__}"
    return hashlib.sha256(material.encode("utf-8")).hexdigest()[:32]


def analysis_cache_dir() -> Path:
    path = weights.cache_dir() / "analyses"
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_cached(path: Path, preset: str, det_interval: int) -> PoseResult | None:
    cached = analysis_cache_dir() / f"{analysis_key(path, preset, det_interval)}.json"
    if not cached.exists():
        return None
    try:
        return from_json(cached.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001 - a corrupt cache entry is a miss, not a crash
        _log.warning("ignoring unreadable cached analysis %s (%s)", cached.name, exc)
        return None


def store_cached(path: Path, preset: str, det_interval: int, result: PoseResult) -> Path:
    """Write atomically, so an interrupted write never leaves a half file to be read back."""
    target = analysis_cache_dir() / f"{analysis_key(path, preset, det_interval)}.json"
    fd, tmp = tempfile.mkstemp(dir=target.parent, suffix=".part")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(to_json(result))
        os.replace(tmp, target)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise
    return target


def cached_analysis(
    path: Path,
    engine_factory: Callable[[], FrameEngine],
    reader: VideoReader,
    *,
    preset: str,
    det_interval: int = 1,
    use_cache: bool = True,
    on_frame: Callable[[PoseFrame, int, int], None] | None = None,
    should_stop: Callable[[], bool] | None = None,
) -> tuple[PoseResult, bool]:
    """The analysis of ``path``, from the cache if present. Returns ``(result, was_cached)``.

    The engine is built only on a miss, so a cached clip never loads a model.
    """
    if use_cache:
        hit = load_cached(path, preset, det_interval)
        if hit is not None:
            return hit, True
    result = analyse_video(
        engine_factory(), reader, det_interval=det_interval, on_frame=on_frame, should_stop=should_stop
    )
    if use_cache:
        store_cached(path, preset, det_interval, result)
    return result, False
