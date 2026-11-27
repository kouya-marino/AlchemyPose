"""Top-down cropping and SimCC decoding for the RTMPose family.

RTMPose predicts, per keypoint, two 1-D vectors of *logits* — one along x, one
along y — at ``simcc_split_ratio`` times the input resolution. Locating a
keypoint is therefore an argmax per axis, divided by that ratio. This is the
whole trick; it is why the models are fast.

**The peak height is not a probability.** Every model in the family is trained
with MMPose's ``KLDiscretLoss``: ``log_softmax(pred * beta)`` against
``softmax(target * label_beta)``, the target being a unit-height Gaussian.
Softmax is shift-invariant and only each row's *shape* is supervised, so the
raw peak of a confidently placed keypoint converges to roughly
``label_beta / beta`` — about 1.0 for the ``beta=10`` body / DWPose configs
and about 10 for the ``beta=1, label_beta=10`` RTMW configs. Measured on the
same photographs, peaks reach 1.03 (RTMW-m), 1.01 (DWPose-l) and 8.6 (RTMW-l).
Neither MMPose's ``get_simcc_maximum`` nor rtmlib's normalises this value, so
a fixed 0..1 threshold on the raw peak is meaningless across models — on
RTMW-l it hides nothing at all. ``decode_simcc`` therefore divides by a
per-model ``peak_scale`` (see ``rtmw.PEAK_SCALE``) and clips, so the
confidence it returns means the same thing whichever model produced it.
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
    return warped.astype(np.uint8, copy=False), affine


def affine_untransform(keypoints: F32, affine: F32) -> F32:
    """Map (..., 2) points in model-input space back to source-image pixels."""
    inverse = cv2.invertAffineTransform(affine).astype(np.float32)
    shape = keypoints.shape
    flat = keypoints.reshape(-1, 2)
    homogeneous = np.concatenate([flat, np.ones((flat.shape[0], 1), dtype=np.float32)], axis=1)
    return (homogeneous @ inverse.T).astype(np.float32).reshape(shape)


def decode_simcc(
    simcc_x: F32, simcc_y: F32, simcc_split_ratio: float = 2.0, *, peak_scale: float = 1.0
) -> tuple[F32, F32]:
    """Argmax the two 1-D logit vectors into ``(keypoints (N, K, 2), confidence (N, K))``.

    Coordinates are in model-input pixels. The confidence is MMPose's score —
    the smaller of the two axis peaks, exactly ``get_simcc_maximum`` — divided
    by ``peak_scale`` and clipped to ``[0, 1]``:

    * ``1.0`` — a peak at least as tall as the training target for a fully
      visible keypoint (rows can overshoot a little; RTMW-m reaches 1.05 raw).
    * ``0.0`` — no peak at all. A non-positive maximum is reported with zero
      confidence and a zeroed location rather than a plausible-looking one.
    * In between — a monotone, model-independent heuristic. It is *not* a
      calibrated probability; nothing in the training objective makes it one.

    ``peak_scale`` is the raw height a confident peak reaches for *this* model
    (``label_beta / beta`` of its loss, verified against real output — see the
    module docstring and ``rtmw.PEAK_SCALE``). It only rescales confidence; it
    never moves a keypoint. The default of ``1.0`` is right for the ``beta=10``
    majority of the family and leaves a unit peak at full confidence.
    """
    if peak_scale <= 0.0:
        raise ValueError(f"peak_scale must be positive, got {peak_scale}")

    x_locs = np.argmax(simcc_x, axis=2)
    y_locs = np.argmax(simcc_y, axis=2)
    max_x = np.amax(simcc_x, axis=2)
    max_y = np.amax(simcc_y, axis=2)

    keypoints = np.stack([x_locs, y_locs], axis=-1).astype(np.float32) / simcc_split_ratio
    raw_peak = np.minimum(max_x, max_y)
    scores = np.clip(raw_peak / peak_scale, 0.0, 1.0).astype(np.float32)

    keypoints[raw_peak <= 0.0] = 0.0
    return keypoints, scores


def decode_simcc_depth(
    simcc_z: NDArray[np.float32], z_range: float, simcc_split_ratio: float = 2.0
) -> NDArray[np.float32]:
    """RTMW3D's depth head -> root-relative depth in metres, ``(N, K)``.

    Exactly MMPose's ``SimCC3DLabel.decode``: the bin index divided by the split
    ratio is a position along a depth axis as long as the model's input width
    (288 for RTMW3D), and the middle of that axis is the root:
    ``z = (z_bin / ratio / (depth / 2) - 1) * z_range``. Verified on real frames
    rather than taken on trust: decoded this way, the predicted root (the
    midpoint of the hips, MMPose's ``root_index=(11, 12)``) lands at
    ``0.00 ± 0.01`` m, as a root-relative quantity must. rtmlib divides by half
    the input *height* instead (384 / 2), which pushes that root to about
    -0.5 m; it is not copied here.

    Positive z is *away* from the camera — the ears decode behind the nose on a
    person facing it — which is the OpenCV camera convention the rest of the 3D
    side uses.
    """
    depth_bins = simcc_z.shape[-1] / simcc_split_ratio
    z_locs = np.argmax(simcc_z, axis=2).astype(np.float32) / simcc_split_ratio
    return ((z_locs / (depth_bins / 2.0) - 1.0) * z_range).astype(np.float32)
