# tests/unit/test_rtmw_preprocess.py
"""Colour-order guard for RTMW preprocessing.

The only integration fixture (``tests/integration/fixtures/two_people.jpg``) is a
fully monochrome 1935 photograph: B == G == R at every pixel, so a BGR/RGB swap is a
mathematical no-op there and the integration suite cannot detect one. These tests
use synthetic pixels with three distinct channel values and assert the literal
expected numbers, which pins the channel order and the ImageNet constants together.
No model weights are needed; this runs in CI.
"""

import numpy as np
import pytest

from alchemypose.inference.rtmw import normalise_crop

# The expected values below are written out as literals on purpose: deriving them
# from the same constants the implementation uses would let a wrong constant or a
# swapped channel cancel out. Each is (value - ImageNet mean) / ImageNet std for the
# *RGB* channel that should land in that output slot.


def test_preprocess_converts_bgr_to_rgb_before_normalising() -> None:
    """A BGR/RGB swap is invisible on the monochrome archival fixture, so the channel
    order is pinned here with synthetic pixels whose channels are all different.
    Getting this backwards yields confident, plausible, anatomically wrong keypoints.
    """
    # OpenCV/numpy convention: array channel order is B, G, R.
    bgr = np.empty((4, 6, 3), dtype=np.uint8)
    bgr[..., 0] = 200  # blue
    bgr[..., 1] = 100  # green
    bgr[..., 2] = 50  # red

    blob = normalise_crop(bgr)

    # Output channel 0 must be RED: (50 - 123.675) / 58.395
    assert blob[0, 0] == pytest.approx(-1.26166624, rel=1e-6)
    # Output channel 1 must be GREEN: (100 - 116.28) / 57.12
    assert blob[0, 1] == pytest.approx(-0.28501401, rel=1e-6)
    # Output channel 2 must be BLUE: (200 - 103.53) / 57.375
    assert blob[0, 2] == pytest.approx(1.68139434, rel=1e-6)


def test_preprocess_channel_order_holds_for_a_second_distinct_triple() -> None:
    """A second, unrelated (B, G, R) triple so a coincidental match on the first cannot pass."""
    bgr = np.empty((3, 5, 3), dtype=np.uint8)
    bgr[..., 0] = 10  # blue
    bgr[..., 1] = 250  # green
    bgr[..., 2] = 170  # red

    blob = normalise_crop(bgr)

    # channel 0 <- red:   (170 - 123.675) / 58.395
    assert blob[0, 0] == pytest.approx(0.79330422, rel=1e-6)
    # channel 1 <- green: (250 - 116.28) / 57.12
    assert blob[0, 1] == pytest.approx(2.34103641, rel=1e-6)
    # channel 2 <- blue:  (10 - 103.53) / 57.375
    assert blob[0, 2] == pytest.approx(-1.63015251, rel=1e-6)


def test_preprocess_produces_contiguous_nchw_float32_for_onnxruntime() -> None:
    """The ONNX session is fed this blob directly: batch-of-one, channels-first, float32, C-contiguous."""
    height, width = 7, 11
    bgr = np.zeros((height, width, 3), dtype=np.uint8)

    blob = normalise_crop(bgr)

    assert blob.shape == (1, 3, height, width)
    assert blob.dtype == np.float32
    assert blob.flags["C_CONTIGUOUS"]


def test_preprocess_maps_spatial_positions_without_transposing_them() -> None:
    """Channel-first transpose must not swap H and W or flip the image: a single lit
    pixel at (row 2, col 5) in the input lands at (row 2, col 5) in every output plane."""
    bgr = np.zeros((4, 8, 3), dtype=np.uint8)
    bgr[2, 5] = (255, 255, 255)

    blob = normalise_crop(bgr)

    for channel in range(3):
        plane = blob[0, channel]
        lit = np.argwhere(plane == plane.max())
        assert lit.tolist() == [[2, 5]]
