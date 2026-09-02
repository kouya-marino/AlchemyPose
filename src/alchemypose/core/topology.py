"""The keypoint schema AlchemyPose renders.

COCO-WholeBody 133 is used throughout because it is what RTMW emits, it maps
cleanly onto H36M-17 for later 3D lifting, and RTMW3D produces the identical
layout with a third coordinate — so v3 changes no index anywhere.
"""

from __future__ import annotations

from dataclasses import dataclass

_BODY = (
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
)
_FEET = (
    "left_big_toe", "left_small_toe", "left_heel",
    "right_big_toe", "right_small_toe", "right_heel",
)
_FINGERS = ("thumb", "index", "middle", "ring", "pinky")


def _hand_names(side: str) -> tuple[str, ...]:
    names = [f"{side}_hand_root"]
    for finger in _FINGERS:
        names += [f"{side}_{finger}_{j}" for j in range(1, 5)]
    return tuple(names)


_FACE = tuple(f"face_{i}" for i in range(68))

KEYPOINT_NAMES: tuple[str, ...] = (
    _BODY + _FEET + _FACE + _hand_names("left") + _hand_names("right")
)


def _hand_edges(base: int) -> tuple[tuple[int, int], ...]:
    """Root to each finger's first joint, then along each finger."""
    edges: list[tuple[int, int]] = []
    for f in range(5):
        first = base + 1 + f * 4
        edges.append((base, first))
        edges += [(first + j, first + j + 1) for j in range(3)]
    return tuple(edges)


_BODY_EDGES = (
    (0, 1), (0, 2), (1, 3), (2, 4),
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16),
)
_FEET_EDGES = ((15, 17), (15, 18), (15, 19), (16, 20), (16, 21), (16, 22))


def _contour(start: int, count: int, closed: bool) -> tuple[tuple[int, int], ...]:
    edges = [(start + i, start + i + 1) for i in range(count - 1)]
    if closed:
        edges.append((start + count - 1, start))
    return tuple(edges)


_FACE_EDGES = (
    _contour(23, 17, closed=False)      # jaw
    + _contour(40, 5, closed=False)     # right brow
    + _contour(45, 5, closed=False)     # left brow
    + _contour(50, 4, closed=False)     # nose bridge
    + _contour(54, 5, closed=False)     # nostrils
    + _contour(59, 6, closed=True)      # right eye
    + _contour(65, 6, closed=True)      # left eye
    + _contour(71, 12, closed=True)     # outer lips
    + _contour(83, 8, closed=True)      # inner lips
)

EDGES: tuple[tuple[int, int], ...] = (
    _BODY_EDGES + _FEET_EDGES + _FACE_EDGES + _hand_edges(91) + _hand_edges(112)
)


def _r(start: int, stop: int) -> tuple[int, ...]:
    return tuple(range(start, stop))


#: Narrow render groups, in the order the layer panel shows them. Every keypoint
#: belongs to exactly one. The broad groups below are unions of these.
RENDER_GROUPS: dict[str, tuple[int, ...]] = {
    "torso_head": (0, 1, 2, 3, 4, 5, 6, 11, 12),
    "left_arm": (7, 9),
    "right_arm": (8, 10),
    "left_leg": (13, 15),
    "right_leg": (14, 16),
    "left_foot": (17, 18, 19),
    "right_foot": (20, 21, 22),
    "face_jaw": _r(23, 40),
    "face_brows": _r(40, 50),
    "face_nose": _r(50, 59),
    "face_eyes": _r(59, 71),
    "face_lips": _r(71, 91),
    "left_hand": _r(91, 112),
    "right_hand": _r(112, 133),
}


def _union(*names: str) -> tuple[int, ...]:
    return tuple(sorted(i for n in names for i in RENDER_GROUPS[n]))


BROAD_GROUPS: dict[str, tuple[int, ...]] = {
    "body": _r(0, 17),
    "feet": _r(17, 23),
    "face": _r(23, 91),
    "left_hand": _r(91, 112),
    "right_hand": _r(112, 133),
    "hands": _r(91, 133),
    "left_side": _union("left_arm", "left_leg", "left_foot", "left_hand"),
    "right_side": _union("right_arm", "right_leg", "right_foot", "right_hand"),
    "upper": _union("torso_head", "left_arm", "right_arm", "left_hand", "right_hand"),
    "lower": _union("left_leg", "right_leg", "left_foot", "right_foot"),
}


@dataclass(frozen=True)
class Topology:
    """A keypoint schema: what the points are called and how they connect."""

    name: str
    keypoint_names: tuple[str, ...]
    edges: tuple[tuple[int, int], ...]
    groups: dict[str, tuple[int, ...]]

    def __len__(self) -> int:
        return len(self.keypoint_names)

    def group_of(self, index: int) -> str | None:
        """The narrowest render group containing ``index``, or None if out of range."""
        for name, indices in RENDER_GROUPS.items():
            if index in indices:
                return name
        return None


COCO_WHOLEBODY_133 = Topology(
    name="coco_wholebody_133",
    keypoint_names=KEYPOINT_NAMES,
    edges=EDGES,
    groups={**RENDER_GROUPS, **BROAD_GROUPS},
)
