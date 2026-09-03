"""How much face to draw, given how big the face is on screen.

Sixty-eight dots at fit-to-window is not detail, it is a grey smudge with a jaw
around it: the points land within a pixel or two of each other, so every one of
them costs an item and none of them says anything. The contours through those
same points read as a face at any size, which is why MediaPipe's own demos draw
the mesh and not the vertices.

So the render follows the face's *screen* size rather than its size in the
photograph — a 4000 px portrait at fit-to-window is small on screen, and a 90 px
face at 8x is not. Three levels:

``CONTOURS``
    Outlines only, in one neutral colour. Below roughly a thumbnail there is no
    room for anything else.
``REGIONS``
    The same outlines, each region in its own colour, so brows, eyes, nose and
    lips separate at a glance.
``POINTS``
    The individual keypoints as well, which is also the only level where they
    are worth making hoverable — a point you cannot aim at is not a target.

Nothing here imports Qt: the thresholds are the part worth testing, and they are
testable without a widget.
"""

from __future__ import annotations

from enum import Enum

REGIONS_ABOVE = 80.0
"""Screen pixels at which per-region colour starts to be legible."""

POINTS_ABOVE = 200.0
"""Screen pixels at which the 68 points are far enough apart to aim at.

At 200 px across, adjacent dlib-68 points sit roughly 8-12 px apart — about the
11 px hit radius ``KeypointItem`` claims. Below that the hit shapes overlap and
hovering one means hovering whichever happened to be added last.
"""

FACE_GROUP_PREFIX = "face_"
"""Render groups whose name starts with this are part of the face."""


class FaceDetail(Enum):
    """One of three renderings of the same 68 points."""

    CONTOURS = "contours"
    REGIONS = "regions"
    POINTS = "points"

    @property
    def draws_points(self) -> bool:
        """Whether individual keypoint items are worth creating at this level."""
        return self is FaceDetail.POINTS

    @property
    def colours_by_region(self) -> bool:
        """Whether jaw, brows, nose, eyes and lips get their own colours."""
        return self is not FaceDetail.CONTOURS


def face_detail(face_box_screen_px: float) -> FaceDetail:
    """Pick a detail level from the face's longest on-screen edge, in pixels."""
    if face_box_screen_px >= POINTS_ABOVE:
        return FaceDetail.POINTS
    if face_box_screen_px >= REGIONS_ABOVE:
        return FaceDetail.REGIONS
    return FaceDetail.CONTOURS


def is_face_group(group: str) -> bool:
    """Whether a render group name belongs to the face.

    By prefix rather than by index range: the overlay already knows each item's
    group, and a hard-coded ``23 <= i < 91`` would be a second, silently
    divergent copy of the topology's own layout.
    """
    return group.startswith(FACE_GROUP_PREFIX)


REGION_COLORS: dict[str, str] = {
    "face_jaw": "#c9cdd6",      # the silhouette stays neutral; it is the frame, not a feature
    "face_brows": "#9b8cff",
    "face_nose": "#7fe0d4",
    "face_eyes": "#5cc8f5",
    "face_lips": "#ff7fa8",
}
"""Per-region colours spent at ``REGIONS`` and above, as hex strings.

Hex rather than ``QColor`` so this module stays importable without Qt. The hues
deliberately avoid the green and orange the theme reserves for the person's left
and right: a face is a tight cluster, and a pink lip line that reads as "right
side" would undo the one colour convention the legend has to teach.
"""
