"""SimCC decoding: locations from the argmax, confidence from the (rescaled) peak.

The confidence contract pinned here is the one ``PersonPose.scores`` documents:
MMPose's ``get_simcc_maximum`` score — the smaller of the two axis peaks —
divided by the model's ``peak_scale`` and clipped to ``[0, 1]``. Every number
below is a literal on purpose, so a wrong divisor or a swapped reduction cannot
cancel out against the implementation's own constants.
"""

import numpy as np
import pytest

from alchemypose.inference.simcc import affine_untransform, decode_simcc, top_down_affine


def _one_hot_simcc(k: int, w_bins: int, h_bins: int, peaks: list[tuple[int, int]]) -> tuple[np.ndarray, np.ndarray]:
    sx = np.zeros((1, k, w_bins), dtype=np.float32)
    sy = np.zeros((1, k, h_bins), dtype=np.float32)
    for j, (px, py) in enumerate(peaks):
        sx[0, j, px] = 1.0
        sy[0, j, py] = 1.0
    return sx, sy


def test_decode_finds_the_argmax_bin_scaled_by_the_split_ratio() -> None:
    """A unit peak on a unit-scale model (the RTMPose-body / DWPose case) is full confidence."""
    sx, sy = _one_hot_simcc(k=2, w_bins=384, h_bins=512, peaks=[(100, 200), (300, 40)])
    kp, scores = decode_simcc(sx, sy, simcc_split_ratio=2.0)
    assert kp.shape == (1, 2, 2)
    np.testing.assert_allclose(kp[0, 0], [50.0, 100.0])
    np.testing.assert_allclose(kp[0, 1], [150.0, 20.0])
    np.testing.assert_allclose(scores[0], [1.0, 1.0])


def test_decode_confidence_is_the_smaller_of_the_two_axis_peaks() -> None:
    """MMPose reduces with ``min``; rtmlib's 2-D path averages (which would give 0.65 here). Pin min."""
    sx, sy = _one_hot_simcc(k=1, w_bins=10, h_bins=10, peaks=[(3, 4)])
    sx[0, 0, 3] = 0.9
    sy[0, 0, 4] = 0.4
    _, scores = decode_simcc(sx, sy)
    assert scores[0, 0] == np.float32(0.4)


def test_decode_marks_a_negative_peak_as_zero_confidence_and_zeroes_its_location() -> None:
    sx = np.full((1, 1, 8), -0.5, dtype=np.float32)
    sy = np.full((1, 1, 8), -0.5, dtype=np.float32)
    kp, scores = decode_simcc(sx, sy)
    assert scores[0, 0] == np.float32(0.0)
    np.testing.assert_array_equal(kp[0, 0], [0.0, 0.0])


def test_decode_divides_the_peak_by_the_model_peak_scale() -> None:
    """RTMW-l emits ~10x taller logits: an 8.5 peak is 0.85 confidence on a peak_scale of 10, not 8.5."""
    sx, sy = _one_hot_simcc(k=2, w_bins=576, h_bins=768, peaks=[(10, 20), (30, 40)])
    sx[0, 0, 10], sy[0, 0, 20] = 8.5, 9.0
    sx[0, 1, 30], sy[0, 1, 40] = 4.0, 3.5
    _, scores = decode_simcc(sx, sy, peak_scale=10.0)
    np.testing.assert_allclose(scores[0], [0.85, 0.35], rtol=1e-6)


def test_decode_clips_confidence_at_one() -> None:
    """Rows overshoot the training target a little (RTMW-m reaches 1.055 raw); the contract is [0, 1]."""
    sx, sy = _one_hot_simcc(k=1, w_bins=384, h_bins=512, peaks=[(7, 9)])
    sx[0, 0, 7], sy[0, 0, 9] = 1.055, 1.2
    _, unit_scale = decode_simcc(sx, sy)
    assert unit_scale[0, 0] == np.float32(1.0)

    sx[0, 0, 7], sy[0, 0, 9] = 12.0, 11.0
    _, tall_scale = decode_simcc(sx, sy, peak_scale=10.0)
    assert tall_scale[0, 0] == np.float32(1.0)


def test_peak_scale_only_rescales_confidence_and_never_moves_a_keypoint() -> None:
    rng = np.random.default_rng(7)
    sx = (rng.standard_normal((1, 5, 576)) * 3.0).astype(np.float32)
    sy = (rng.standard_normal((1, 5, 768)) * 3.0).astype(np.float32)
    # Random logits of this size peak around 8-12, so scales of 20 and 40 never clip and
    # the two results must differ by exactly the ratio of the scales, nothing else.
    kp_20, sc_20 = decode_simcc(sx, sy, peak_scale=20.0)
    kp_40, sc_40 = decode_simcc(sx, sy, peak_scale=40.0)

    np.testing.assert_array_equal(kp_20, kp_40)
    assert np.all(sc_20 > 0.0) and np.all(sc_20 < 1.0), "test needs unclipped, positive rows"
    np.testing.assert_allclose(sc_40, sc_20 / 2.0, rtol=1e-6)


def test_decode_confidence_is_float32_inside_the_unit_interval_for_arbitrary_logits() -> None:
    rng = np.random.default_rng(11)
    sx = (rng.standard_normal((2, 133, 576)) * 8.0).astype(np.float32)
    sy = (rng.standard_normal((2, 133, 768)) * 8.0).astype(np.float32)
    kp, scores = decode_simcc(sx, sy, peak_scale=10.0)

    assert scores.dtype == np.float32
    assert scores.shape == (2, 133)
    assert np.all(scores >= 0.0) and np.all(scores <= 1.0)
    # A row without a positive peak on both axes carries no location either.
    no_peak = np.minimum(sx.max(axis=2), sy.max(axis=2)) <= 0.0
    assert np.all(kp[no_peak] == 0.0)


def test_decode_rejects_a_non_positive_peak_scale() -> None:
    sx, sy = _one_hot_simcc(k=1, w_bins=8, h_bins=8, peaks=[(1, 1)])
    for bad in (0.0, -1.0):
        with pytest.raises(ValueError, match="peak_scale"):
            decode_simcc(sx, sy, peak_scale=bad)


def test_affine_produces_the_requested_input_size() -> None:
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    warped, _ = top_down_affine((100.0, 50.0, 300.0, 450.0), (192, 256), image)
    assert warped.shape == (256, 192, 3)


def test_affine_round_trip_returns_a_point_to_source_pixels() -> None:
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    bbox = (100.0, 50.0, 300.0, 450.0)
    input_size = (192, 256)
    _, affine = top_down_affine(bbox, input_size, image)

    centre_in_model = np.array([[[input_size[0] / 2, input_size[1] / 2]]], dtype=np.float32)
    back = affine_untransform(centre_in_model, affine)

    expected = np.array([(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2], dtype=np.float32)
    np.testing.assert_allclose(back[0, 0], expected, atol=1.0)


def test_affine_preserves_aspect_ratio_by_padding_not_stretching() -> None:
    """A wide box into a tall input must letterbox, so the mapped width is bounded."""
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    _, affine = top_down_affine((0.0, 200.0, 400.0, 260.0), (192, 256), image)
    corners = np.array([[[0.0, 0.0], [192.0, 0.0]]], dtype=np.float32)
    mapped = affine_untransform(corners, affine)
    width = float(np.linalg.norm(mapped[0, 1] - mapped[0, 0]))
    assert width > 400.0, "a wide box should be padded vertically, not squeezed horizontally"
