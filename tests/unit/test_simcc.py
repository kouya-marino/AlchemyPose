import numpy as np

from alchemypose.inference.simcc import affine_untransform, decode_simcc, top_down_affine


def _one_hot_simcc(k: int, w_bins: int, h_bins: int, peaks: list[tuple[int, int]]) -> tuple[np.ndarray, np.ndarray]:
    sx = np.zeros((1, k, w_bins), dtype=np.float32)
    sy = np.zeros((1, k, h_bins), dtype=np.float32)
    for j, (px, py) in enumerate(peaks):
        sx[0, j, px] = 1.0
        sy[0, j, py] = 1.0
    return sx, sy


def test_decode_finds_the_argmax_bin_scaled_by_the_split_ratio() -> None:
    sx, sy = _one_hot_simcc(k=2, w_bins=384, h_bins=512, peaks=[(100, 200), (300, 40)])
    kp, scores = decode_simcc(sx, sy, simcc_split_ratio=2.0)
    assert kp.shape == (1, 2, 2)
    np.testing.assert_allclose(kp[0, 0], [50.0, 100.0])
    np.testing.assert_allclose(kp[0, 1], [150.0, 20.0])
    np.testing.assert_allclose(scores[0], [1.0, 1.0])


def test_decode_score_is_the_smaller_of_the_two_axis_peaks() -> None:
    sx, sy = _one_hot_simcc(k=1, w_bins=10, h_bins=10, peaks=[(3, 4)])
    sx[0, 0, 3] = 0.9
    sy[0, 0, 4] = 0.4
    _, scores = decode_simcc(sx, sy)
    assert scores[0, 0] == np.float32(0.4)


def test_decode_marks_a_negative_peak_as_zero_confidence() -> None:
    sx = np.full((1, 1, 8), -0.5, dtype=np.float32)
    sy = np.full((1, 1, 8), -0.5, dtype=np.float32)
    _, scores = decode_simcc(sx, sy)
    assert scores[0, 0] == np.float32(0.0)


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
