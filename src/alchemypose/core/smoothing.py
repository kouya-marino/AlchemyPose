"""Temporal smoothing, applied when drawing — never to what is stored.

v1's rule is that coordinates are stored raw, and smoothing does not get to be
the thing that breaks it: a ``PoseResult`` from the engine, its JSON and its CSV
all carry exactly what the model said, frame by frame. The window and the 3D
and robot views ask for a smoothed *copy* (:func:`smooth_frames`) and draw
that. Turning smoothing off is therefore free and lossless, and an export never
silently contains a filter's opinion.

The filter is the 1€ filter (Casiez, Roussel & Vogel, CHI 2012): a low-pass
whose cutoff rises with speed, so a still joint stops shivering while a fast
one is not dragged behind. It is the standard choice for exactly this — rtmlib
and MediaPipe both use it for keypoints — and it has two knobs a person can
reason about: ``min_cutoff`` (lower = smoother at rest) and ``beta`` (higher =
less lag when moving).

Smoothing is per track, per keypoint. A joint that drops below the confidence
floor is passed through untouched and not fed to the filter, so one bad frame
does not drag a good trajectory towards the crop corner; a track that vanishes
for longer than ``reset_after_s`` starts afresh when it returns.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace

import numpy as np
from numpy.typing import NDArray

from alchemypose.core.schema import PersonPose, PoseFrame

F32 = NDArray[np.float32]


def _alpha(cutoff: F32 | float, dt: float) -> F32:
    tau = 1.0 / (2.0 * math.pi * np.asarray(cutoff, dtype=np.float32))
    return np.asarray(1.0 / (1.0 + tau / dt), dtype=np.float32)


@dataclass
class OneEuroFilter:
    """A 1€ filter over a fixed-shape array of values (e.g. ``(K, 2)`` or ``(K, 3)``)."""

    min_cutoff: float = 1.0
    beta: float = 0.05
    d_cutoff: float = 1.0
    _x: F32 | None = None
    _dx: F32 | None = None
    _t: float | None = None

    def reset(self) -> None:
        self._x = self._dx = None
        self._t = None

    def __call__(self, x: F32, t: float, valid: NDArray[np.bool_] | None = None) -> F32:
        """Filter ``x`` observed at time ``t``. Rows where ``valid`` is False pass through."""
        x = np.asarray(x, dtype=np.float32)
        if self._x is None or self._t is None or self._dx is None or t <= self._t:
            self._x = x.copy()
            self._dx = np.zeros_like(x)
            self._t = t
            return x.copy()
        dt = t - self._t
        dx = (x - self._x) / dt
        a_d = _alpha(self.d_cutoff, dt)
        dx_hat = a_d * dx + (1.0 - a_d) * self._dx
        speed = np.linalg.norm(dx_hat, axis=-1, keepdims=True) if x.ndim > 1 else np.abs(dx_hat)
        cutoff = self.min_cutoff + self.beta * speed
        a = _alpha(cutoff, dt)
        x_hat = a * x + (1.0 - a) * self._x
        if valid is not None:
            keep = np.asarray(valid, dtype=bool)
            if x.ndim > 1:
                keep = keep[:, None]
            x_hat = np.where(keep, x_hat, self._x)          # hold the filter's state
            dx_hat = np.where(keep, dx_hat, self._dx)
            out = np.where(keep, x_hat, x)                  # but show the raw value
        else:
            out = x_hat
        self._x, self._dx, self._t = x_hat.astype(np.float32), dx_hat.astype(np.float32), t
        return np.asarray(out, dtype=np.float32)


@dataclass
class SmoothingSettings:
    min_cutoff: float = 1.0
    beta: float = 0.05
    #: Keypoints below this confidence are shown raw and not fed to the filter.
    min_score: float = 0.3
    #: A track unseen for longer than this restarts its filter on return.
    reset_after_s: float = 0.5


@dataclass
class _TrackState:
    filter_2d: OneEuroFilter
    filter_3d: OneEuroFilter
    last_t: float


@dataclass
class PoseSmoother:
    """Stateful, frame-at-a-time smoothing keyed on ``track_id``."""

    settings: SmoothingSettings = field(default_factory=SmoothingSettings)
    _tracks: dict[int, _TrackState] = field(default_factory=dict)

    def _state(self, track_id: int, t: float) -> _TrackState:
        s = self.settings
        state = self._tracks.get(track_id)
        if state is None or t - state.last_t > s.reset_after_s or t < state.last_t:
            state = _TrackState(
                OneEuroFilter(s.min_cutoff, s.beta), OneEuroFilter(s.min_cutoff, s.beta), t
            )
            self._tracks[track_id] = state
        return state

    def person(self, person: PersonPose, t: float) -> PersonPose:
        if person.track_id is None:
            return person
        state = self._state(person.track_id, t)
        state.last_t = t
        valid = person.scores >= self.settings.min_score
        keypoints = state.filter_2d(person.keypoints, t, valid)
        keypoints_3d = person.keypoints_3d
        if keypoints_3d is not None:
            keypoints_3d = state.filter_3d(keypoints_3d, t, valid)
        return replace(person, keypoints=keypoints, keypoints_3d=keypoints_3d)

    def frame(self, frame: PoseFrame, fps: float) -> PoseFrame:
        t = frame.timestamp_s if frame.timestamp_s is not None else frame.frame_index / fps
        return replace(frame, people=tuple(self.person(p, t) for p in frame.people))


def smooth_frames(
    frames: tuple[PoseFrame, ...], fps: float, settings: SmoothingSettings | None = None
) -> tuple[PoseFrame, ...]:
    """A smoothed copy of ``frames``, in order. The input is not modified."""
    smoother = PoseSmoother(settings or SmoothingSettings())
    return tuple(smoother.frame(frame, fps) for frame in frames)
