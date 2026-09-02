"""The result types every layer speaks.

One structure serves a still image now, video frames in v2 and 3D in v3: the
slots for ``track_id``, ``keypoints_3d`` and ``camera`` exist from the start so
that adding those features is additive rather than a schema migration.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from alchemypose.core.topology import Topology

F32 = NDArray[np.float32]


class Space(str, Enum):
    """Which coordinate system a set of numbers lives in."""

    IMAGE_PX = "image_px"        # 2D, source-image pixels, origin top-left, y down
    NORMALIZED = "normalized"    # 2D, 0..1 of image size
    CAMERA_M = "camera_m"        # 3D, metres, camera-centred
    ROOT_REL = "root_rel"        # 3D, metres, relative to the pelvis


@dataclass(frozen=True)
class Camera:
    """Pinhole intrinsics. v1 always fills a default; v3 needs the real thing."""

    fx: float
    fy: float
    cx: float
    cy: float
    source: Literal["exif", "estimated", "default"] = "default"

    @classmethod
    def default_for(cls, width: int, height: int) -> Camera:
        """The standard guess when nothing better is known: f = max(w, h), centred."""
        f = float(max(width, height))
        return cls(fx=f, fy=f, cx=width / 2.0, cy=height / 2.0, source="default")


@dataclass(frozen=True)
class PersonPose:
    """One detected person. ``keypoints`` is always the full topology, never a subset."""

    keypoints: F32                     # (K, 2)
    scores: F32                        # (K,)
    bbox: tuple[float, float, float, float] | None = None
    bbox_score: float | None = None
    track_id: int | None = None                 # v2
    keypoints_3d: F32 | None = None             # (K, 3) — v3
    space_2d: Space = Space.IMAGE_PX
    space_3d: Space | None = None

    def __post_init__(self) -> None:
        if self.keypoints.ndim != 2 or self.keypoints.shape[1] != 2:
            raise ValueError(f"keypoints must have shape (K, 2), got {self.keypoints.shape}")
        if self.scores.shape != (self.keypoints.shape[0],):
            raise ValueError(
                f"scores must have shape ({self.keypoints.shape[0]},), got {self.scores.shape}"
            )
        if self.keypoints_3d is not None:
            expected = (self.keypoints.shape[0], 3)
            if self.keypoints_3d.shape != expected:
                raise ValueError(f"keypoints_3d must have shape {expected}, got {self.keypoints_3d.shape}")

    def __len__(self) -> int:
        return int(self.keypoints.shape[0])


@dataclass(frozen=True)
class PoseFrame:
    """One moment. A still image is frame 0 with no timestamp."""

    frame_index: int
    timestamp_s: float | None
    image_size: tuple[int, int]        # (width, height)
    people: tuple[PersonPose, ...]


@dataclass(frozen=True)
class PoseResult:
    """Everything one run produced, plus what produced it."""

    topology: Topology
    frames: tuple[PoseFrame, ...]
    source: str
    model: str
    camera: Camera | None = None
    schema_version: Literal[1] = 1


@dataclass(frozen=True)
class KeypointSelection:
    """What the canvas should draw. Never used to filter stored data."""

    visible: frozenset[int]
    persons: frozenset[int] | None = None      # None means every person

    def is_visible(self, index: int) -> bool:
        return index in self.visible

    def is_edge_visible(self, edge: tuple[int, int]) -> bool:
        """A bone is drawn only when both of its endpoints are."""
        return edge[0] in self.visible and edge[1] in self.visible
