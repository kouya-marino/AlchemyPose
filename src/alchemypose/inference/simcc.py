"""Top-down cropping and SimCC decoding for the RTMPose family.

RTMPose predicts, per keypoint, two 1-D probability vectors — one along x, one
along y — at ``simcc_split_ratio`` times the input resolution. Decoding is
therefore an argmax per axis, divided by that ratio. This is the whole trick;
it is why the models are fast.
"""

from __future__ import annotations

import cv2
import numpy as np
from numpy.typing import NDArray

F32 = NDArray[np.float32]


def _box_to_centre_scale(
    bbox: tuple[float, float, float, float], aspect_ratio: float, padding: float = 1.25
) -> tuple[F32, F32]:
    """Centre and a padded, aspect-corrected box size.

    The box is widened or heightened — never squeezed — so the person is
    letterboxed into the model input rather than distorted.
    """
    x0, y0, x1, y1 = bbox
    centre = np.array([(x0 + x1) / 2.0, (y0 + y1) / 2.0], dtype=np.float32)
    w, h = x1 - x0, y1 - y0
    if w > h * aspect_ratio:
        h = w / aspect_ratio
    else:
        w = h * aspect_ratio
    return centre, np.array([w, h], dtype=np.float32) * padding


def top_down_affine(
    bbox: tuple[float, float, float, float], input_size: tuple[int, int], image: NDArray[np.uint8]
) -> tuple[NDArray[np.uint8], F32]:
    """Crop and resize one person into the model's input, returning the warp used.

    ``input_size`` is (width, height). The returned 2x3 affine maps source-image
    pixels to model-input pixels; ``affine_untransform`` inverts it.
    """
    in_w, in_h = input_size
    centre, scale = _box_to_centre_scale(bbox, aspect_ratio=in_w / in_h)

    src = np.array(
        [centre, centre + [0.0, -scale[1] / 2.0], centre + [-scale[0] / 2.0, 0.0]], dtype=np.float32
    )
    dst = np.array(
        [[in_w / 2.0, in_h / 2.0], [in_w / 2.0, 0.0], [0.0, in_h / 2.0]], dtype=np.float32
    )
    affine = cv2.getAffineTransform(src, dst).astype(np.float32)
    warped = cv2.warpAffine(image, affine, (in_w, in_h), flags=cv2.INTER_LINEAR)
    return warped, affine


def affine_untransform(keypoints: F32, affine: F32) -> F32:
    """Map (..., 2) points in model-input space back to source-image pixels."""
    inverse = cv2.invertAffineTransform(affine).astype(np.float32)
    shape = keypoints.shape
    flat = keypoints.reshape(-1, 2)
    homogeneous = np.concatenate([flat, np.ones((flat.shape[0], 1), dtype=np.float32)], axis=1)
    return (homogeneous @ inverse.T).astype(np.float32).reshape(shape)


def decode_simcc(simcc_x: F32, simcc_y: F32, simcc_split_ratio: float = 2.0) -> tuple[F32, F32]:
    """Argmax the two 1-D distributions into ``(keypoints (N, K, 2), scores (N, K))``.

    Coordinates are in model-input pixels. A keypoint whose peak is non-positive
    is reported with zero confidence rather than a plausible-looking location.
    """
    x_locs = np.argmax(simcc_x, axis=2)
    y_locs = np.argmax(simcc_y, axis=2)
    max_x = np.amax(simcc_x, axis=2)
    max_y = np.amax(simcc_y, axis=2)

    keypoints = np.stack([x_locs, y_locs], axis=-1).astype(np.float32) / simcc_split_ratio
    scores = np.minimum(max_x, max_y).astype(np.float32)

    invalid = scores <= 0.0
    scores[invalid] = 0.0
    keypoints[invalid] = 0.0
    return keypoints, scores
