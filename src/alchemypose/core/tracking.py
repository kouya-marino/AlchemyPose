"""Giving a person the same ``track_id`` from one frame to the next.

``PersonPose.track_id`` has been in the schema since its first commit and
written as ``null`` in every export so far; this fills it.

The tracker is deliberately small: greedy matching on box overlap, frame to
frame, the way rtmlib's ``PoseTracker`` does it. A pose model already hands
over a box per person, and at video frame rates the same person's box in two
consecutive frames overlaps far more than two different people's do, so IoU is
enough of a signal without appearance features or a motion model. What it does
not survive is two people crossing completely in front of each other — the
identities can swap there — which is the cost of having no appearance model.

Matching uses the box the *keypoints* span (confident ones only), not the
detector's box. In video mode a person's crop for frame ``t`` is often derived
from frame ``t-1``'s keypoints rather than from the detector, and comparing
like with like keeps the overlap honest either way.

Identity is renumbered at the end (:func:`assign_tracks`): the track present in
the most frames becomes 0, the next 1, and so on, so "person 1" in a dance
video is the dancer rather than whoever happened to be detected first.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field, replace

import numpy as np
from numpy.typing import NDArray

from alchemypose.core.schema import PersonPose, PoseFrame

F32 = NDArray[np.float32]
BBox = tuple[float, float, float, float]

#: Keypoints scoring below this do not count towards a person's extent.
EXTENT_MIN_SCORE = 0.3


def keypoint_box(person: PersonPose, min_score: float = EXTENT_MIN_SCORE) -> BBox | None:
    """The box spanned by a person's confident keypoints, or their detector box."""
    mask = person.scores >= min_score
    if int(mask.sum()) >= 2:
        points = person.keypoints[mask]
        x0, y0 = points.min(axis=0)
        x1, y1 = points.max(axis=0)
        if x1 > x0 and y1 > y0:
            return float(x0), float(y0), float(x1), float(y1)
    return person.bbox


def iou(a: BBox, b: BBox) -> float:
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
    ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


@dataclass
class _Track:
    track_id: int
    box: BBox
    last_seen: int


@dataclass
class Tracker:
    """Frame-by-frame identity. Feed it each frame's people in order."""

    iou_threshold: float = 0.3
    #: Frames a track may go unseen before a returning person gets a new id.
    max_age: int = 15
    _tracks: list[_Track] = field(default_factory=list)
    _next_id: int = 0
    _frame: int = -1

    def update(self, people: tuple[PersonPose, ...]) -> list[int]:
        """Track ids for ``people``, in the same order. Advances one frame."""
        self._frame += 1
        self._tracks = [t for t in self._tracks if self._frame - t.last_seen <= self.max_age]
        boxes = [keypoint_box(p) for p in people]

        pairs: list[tuple[float, int, int]] = []
        for pi, box in enumerate(boxes):
            if box is None:
                continue
            for ti, track in enumerate(self._tracks):
                overlap = iou(box, track.box)
                if overlap >= self.iou_threshold:
                    pairs.append((overlap, pi, ti))
        pairs.sort(reverse=True)

        assigned: dict[int, int] = {}
        used_tracks: set[int] = set()
        for _overlap, pi, ti in pairs:
            if pi in assigned or ti in used_tracks:
                continue
            assigned[pi] = ti
            used_tracks.add(ti)

        ids: list[int] = []
        for pi, box in enumerate(boxes):
            if pi in assigned:
                track = self._tracks[assigned[pi]]
                if box is not None:
                    track.box = box
                track.last_seen = self._frame
                ids.append(track.track_id)
            else:
                track = _Track(self._next_id, box or (0.0, 0.0, 0.0, 0.0), self._frame)
                self._next_id += 1
                self._tracks.append(track)
                ids.append(track.track_id)
        return ids


def renumber_by_presence(frames: tuple[PoseFrame, ...]) -> tuple[PoseFrame, ...]:
    """Dense track ids, ``0`` for the person in the most frames.

    Ties go to whoever appeared first. People without a track id are left alone.
    """
    counts: Counter[int] = Counter()
    first_seen: dict[int, int] = {}
    for position, frame in enumerate(frames):
        for person in frame.people:
            if person.track_id is not None:
                counts[person.track_id] += 1
                first_seen.setdefault(person.track_id, position)
    order = sorted(counts, key=lambda tid: (-counts[tid], first_seen[tid]))
    mapping = {old: new for new, old in enumerate(order)}
    return tuple(
        replace(
            frame,
            people=tuple(
                replace(p, track_id=mapping[p.track_id]) if p.track_id is not None else p
                for p in frame.people
            ),
        )
        for frame in frames
    )


def assign_tracks(frames: tuple[PoseFrame, ...], tracker: Tracker | None = None) -> tuple[PoseFrame, ...]:
    """Track every person through ``frames`` and renumber by presence."""
    tracker = tracker or Tracker()
    tracked: list[PoseFrame] = []
    for frame in frames:
        ids = tracker.update(frame.people)
        tracked.append(
            replace(frame, people=tuple(replace(p, track_id=t) for p, t in zip(frame.people, ids, strict=True)))
        )
    return renumber_by_presence(tuple(tracked))


def track_count(frames: tuple[PoseFrame, ...]) -> int:
    """How many distinct track ids appear (assumes dense ids from ``renumber_by_presence``)."""
    ids = {p.track_id for f in frames for p in f.people if p.track_id is not None}
    return max(ids) + 1 if ids else 0


def absent_person(keypoint_count: int) -> PersonPose:
    """A placeholder for a track not seen in this frame: every score 0, so never drawn."""
    return PersonPose(
        keypoints=np.zeros((keypoint_count, 2), dtype=np.float32),
        scores=np.zeros((keypoint_count,), dtype=np.float32),
    )


def by_track(frame: PoseFrame, tracks: int, keypoint_count: int) -> PoseFrame:
    """``frame`` with its people re-ordered so index ``i`` is track ``i``.

    This is the view the window draws in video mode. Person chips, the
    inspector and selection all address people by index, and in a video the
    index has to mean the same person on every frame — so a track that is not
    in this frame is filled with :func:`absent_person`, which scores 0 and is
    therefore never drawn. Untracked people (no ``track_id``) are dropped from
    the view; the stored frame keeps them.
    """
    slots: list[PersonPose] = [absent_person(keypoint_count) for _ in range(tracks)]
    for person in frame.people:
        if person.track_id is not None and 0 <= person.track_id < tracks:
            slots[person.track_id] = person
    return replace(frame, people=tuple(slots))
