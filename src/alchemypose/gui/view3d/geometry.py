"""What the 3D view draws, as plain arrays — built without OpenGL so it tests headless.

The stored 3D is OpenCV camera space (x right, y down, z away from the camera,
metres, origin at the hips). A viewer wants the floor underneath and up being
up, so this module converts to the view's world axes and nowhere else does:

    world = (x, z, -y)        # x right, y away from the camera, z up

and then lifts each person so their lowest confident keypoint stands on the
grid at ``z = 0``. Seen from the default camera — in front of the person,
looking along +y — the skeleton faces the viewer the way it faced the lens.

Selection, the confidence threshold and colours are the 2D canvas's own rules,
so the 3D view never shows a joint the canvas hides, and the person's left is
the same green in both.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from alchemypose.core.schema import KeypointSelection, PersonPose, PoseResult
from alchemypose.gui import theme

F32 = NDArray[np.float32]

#: Spacing between people along x when several share the view, metres.
PERSON_SPACING_M = 1.2
#: Body and feet only decide where the floor is; a hand below the feet is a pose.
_FLOOR_KEYPOINTS = tuple(range(0, 23))


def to_world(points: F32) -> F32:
    """OpenCV camera axes -> the view's z-up world axes."""
    return np.column_stack([points[:, 0], points[:, 2], -points[:, 1]]).astype(np.float32)


def _rgba(hex_colour: str, alpha: float) -> tuple[float, float, float, float]:
    colour = hex_colour.lstrip("#")
    r, g, b = (int(colour[i : i + 2], 16) / 255.0 for i in (0, 2, 4))
    return r, g, b, alpha


@dataclass(frozen=True)
class Geometry:
    """Line segments as consecutive point pairs, and joints, each with an RGBA colour."""

    segments: F32            # (2N, 3)
    segment_colours: F32     # (2N, 4)
    points: F32              # (M, 3)
    point_colours: F32       # (M, 4)

    @property
    def empty(self) -> bool:
        return self.points.shape[0] == 0 and self.segments.shape[0] == 0

    @classmethod
    def nothing(cls) -> Geometry:
        z3, z4 = np.zeros((0, 3), dtype=np.float32), np.zeros((0, 4), dtype=np.float32)
        return cls(z3, z4, z3, z4)


def standing(person: PersonPose, threshold: float) -> F32 | None:
    """``person``'s 3D in world axes, lifted so their lowest confident body point is at z=0."""
    if person.keypoints_3d is None:
        return None
    world = to_world(person.keypoints_3d)
    confident = [i for i in _FLOOR_KEYPOINTS if person.scores[i] >= max(threshold, 1e-6)]
    if not confident:
        return None
    world[:, 2] -= float(world[confident, 2].min())
    return world


def build(result: PoseResult | None, selection: KeypointSelection, threshold: float) -> Geometry:
    """Geometry for the first frame of ``result``, under the canvas's own visibility rules."""
    if result is None or not result.frames:
        return Geometry.nothing()
    topology = result.topology
    people = result.frames[0].people
    chosen = range(len(people)) if selection.persons is None else sorted(selection.persons)
    present = [i for i in chosen if 0 <= i < len(people) and people[i].keypoints_3d is not None]
    # Spread only the people actually drawn, centred, so one person sits at the origin.
    centre = (len(present) - 1) / 2.0
    offsets = {index: (position - centre) * PERSON_SPACING_M for position, index in enumerate(present)}

    seg: list[F32] = []
    seg_c: list[tuple[float, float, float, float]] = []
    pts: list[F32] = []
    pts_c: list[tuple[float, float, float, float]] = []
    for index in present:
        person = people[index]
        world = standing(person, threshold)
        if world is None:
            continue
        world[:, 0] += offsets[index]
        drawn = {
            i for i in selection.visible
            if 0 <= i < len(person) and person.scores[i] > 0.0 and person.scores[i] >= threshold
        }
        for a, b in topology.edges:
            if a in drawn and b in drawn:
                colour = theme.GROUP_COLORS.get(topology.group_of(b) or topology.group_of(a) or "", theme.TEXT_DIM)
                rgba = _rgba(colour, 1.0)
                seg.extend((world[a], world[b]))
                seg_c.extend((rgba, rgba))
        for i in sorted(drawn):
            pts.append(world[i])
            pts_c.append(_rgba(theme.GROUP_COLORS.get(topology.group_of(i) or "", theme.TEXT_DIM), 1.0))

    if not pts and not seg:
        return Geometry.nothing()
    return Geometry(
        segments=np.asarray(seg, dtype=np.float32).reshape(-1, 3),
        segment_colours=np.asarray(seg_c, dtype=np.float32).reshape(-1, 4),
        points=np.asarray(pts, dtype=np.float32).reshape(-1, 3),
        point_colours=np.asarray(pts_c, dtype=np.float32).reshape(-1, 4),
    )
