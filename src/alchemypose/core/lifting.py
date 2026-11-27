"""From image pixels plus root-relative depth to a metric 3D skeleton.

RTMW3D gives each keypoint an image position in pixels and a depth in metres
relative to the hips. A 3D skeleton needs all three axes in the same unit, so
the pixel offsets have to become metres — and the scale that does that depends
on how far away the person is, which a single camera cannot know.

The scale is recovered from the body itself. The two shoulder-to-hip segments
are close to one length on adults (``TORSO_M``), and unlike a limb they are
rarely bent or hidden. For each one, its true length ``L`` relates to what is
measured by ``L² = (s · d_px)² + dz²``: ``d_px`` is the segment's length in the
image, ``dz`` its depth extent from the model, ``s`` the unknown metres per pixel.
So ``s = sqrt(L² - dz²) / d_px``, which accounts for a torso leaning towards the
camera instead of mistaking the foreshortening for distance. The two estimates
are averaged; a frame with neither falls back to the shoulder width.

That is weak perspective — one scale for the whole person — which is accurate to
a few percent for a body a few metres from a phone camera, and it is what makes
the result usable for the one thing the 3D side needs from it: the *direction* of
every bone. The absolute size is only as good as ``TORSO_M`` is for this person,
and nothing downstream depends on it.

Axes are OpenCV's camera axes, right-handed: x right, y down, z away from the
camera, metres, origin at the midpoint of the hips (``Space.ROOT_REL``). The
y-up conversion for glTF happens at export and nowhere else.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

F32 = NDArray[np.float32]

#: Shoulder-to-hip length, metres, for COCO's shoulder and hip keypoints.
TORSO_M = 0.50
#: Shoulder-to-shoulder, metres, for the fallback.
SHOULDERS_M = 0.36

L_SHOULDER, R_SHOULDER, L_HIP, R_HIP = 5, 6, 11, 12
#: A keypoint below this confidence is not trusted to set the scale.
MIN_SCORE = 0.3


def root_of(points: F32) -> F32:
    """The midpoint of the hips — MMPose's root for RTMW3D."""
    return np.asarray((points[L_HIP] + points[R_HIP]) / 2.0, dtype=np.float32)


def _segment_scale(kp2d: F32, z: F32, a: int, b: int, length: float) -> float | None:
    d_px = float(np.linalg.norm(kp2d[a] - kp2d[b]))
    if d_px < 2.0:
        return None
    dz = float(z[a] - z[b])
    # A depth extent at or beyond the segment length would mean an impossible
    # pose; keep at least 30% of the length in the image plane.
    in_plane = np.sqrt(max(length**2 - dz**2, (0.3 * length) ** 2))
    return float(in_plane / d_px)


def metres_per_pixel(kp2d: F32, z: F32, scores: F32) -> float | None:
    """The weak-perspective scale for one person, or ``None`` if nothing sets it."""
    estimates = [
        scale
        for a, b in ((L_SHOULDER, L_HIP), (R_SHOULDER, R_HIP))
        if scores[a] >= MIN_SCORE and scores[b] >= MIN_SCORE
        and (scale := _segment_scale(kp2d, z, a, b, TORSO_M)) is not None
    ]
    if estimates:
        return float(np.mean(estimates))
    if scores[L_SHOULDER] >= MIN_SCORE and scores[R_SHOULDER] >= MIN_SCORE:
        return _segment_scale(kp2d, z, L_SHOULDER, R_SHOULDER, SHOULDERS_M)
    return None


def lift(kp2d: F32, z: F32, scores: F32) -> F32 | None:
    """``(K, 3)`` metres, root-relative, OpenCV axes — or ``None`` if no scale is found."""
    scale = metres_per_pixel(kp2d, z, scores)
    if scale is None:
        return None
    xy = (kp2d - root_of(kp2d)) * scale
    out = np.column_stack([xy, z - (z[L_HIP] + z[R_HIP]) / 2.0]).astype(np.float32)
    out[scores <= 0.0] = 0.0
    return out
