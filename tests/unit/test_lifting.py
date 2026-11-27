"""Depth decoding and the lift from pixels + depth to a metric skeleton."""

import numpy as np
import pytest

from alchemypose.core.lifting import (
    L_HIP,
    L_SHOULDER,
    R_HIP,
    R_SHOULDER,
    TORSO_M,
    lift,
    metres_per_pixel,
)
from alchemypose.inference.simcc import decode_simcc_depth

K = 133


def _depth_head(bins: list[int], width: int = 576) -> np.ndarray:
    head = np.zeros((1, len(bins), width), dtype=np.float32)
    for k, b in enumerate(bins):
        head[0, k, b] = 1.0
    return head


def test_the_middle_bin_is_the_root() -> None:
    """576 bins at split 2 span a 288-long depth axis; its middle (bin 288) is z = 0."""
    z = decode_simcc_depth(_depth_head([288]), z_range=2.1744869)
    assert z[0, 0] == pytest.approx(0.0, abs=1e-6)


def test_the_ends_of_the_depth_axis_are_plus_and_minus_z_range() -> None:
    z = decode_simcc_depth(_depth_head([0, 575]), z_range=2.0)
    assert z[0, 0] == pytest.approx(-2.0)
    assert z[0, 1] == pytest.approx(2.0 * (575 / 2 / 144 - 1))


def test_the_divisor_is_half_the_depth_axis_not_half_the_input_height() -> None:
    """rtmlib divides by 384 / 2; MMPose by 288 / 2. A quarter of the way along must be -z_range/2."""
    z = decode_simcc_depth(_depth_head([144]), z_range=2.0)
    assert z[0, 0] == pytest.approx(-1.0)


# --------------------------------------------------------------------- lifting


def _body(scale_px_per_m: float = 400.0, lean_dz: float = 0.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """A front-facing torso: shoulders 0.36 m apart, hips 0.24 m.

    ``lean_dz`` tips the shoulders towards the camera by that much depth, keeping
    the shoulder-to-hip length at ``TORSO_M``.
    """
    truth = np.zeros((K, 3), dtype=np.float32)
    # Shoulders sit 0.06 m wider than hips each side, so the segment's vertical
    # extent is what is left of TORSO_M after that offset and the lean.
    in_plane = np.sqrt(TORSO_M**2 - 0.06**2 - lean_dz**2)
    truth[L_SHOULDER] = (0.18, -in_plane, -lean_dz)
    truth[R_SHOULDER] = (-0.18, -in_plane, -lean_dz)
    truth[L_HIP] = (0.12, 0.0, 0.0)
    truth[R_HIP] = (-0.12, 0.0, 0.0)
    pixels = truth[:, :2] * scale_px_per_m + np.array([500.0, 600.0], dtype=np.float32)
    scores = np.zeros(K, dtype=np.float32)
    scores[[L_SHOULDER, R_SHOULDER, L_HIP, R_HIP]] = 0.9
    return pixels.astype(np.float32), truth[:, 2].copy(), scores


def test_an_upright_torso_recovers_its_scale() -> None:
    pixels, z, scores = _body(scale_px_per_m=400.0)
    assert metres_per_pixel(pixels, z, scores) == pytest.approx(1 / 400.0, rel=1e-4)


def test_a_torso_leaning_at_the_camera_is_not_mistaken_for_a_distant_one() -> None:
    """Foreshortening shrinks the torso in the image; the depth extent accounts for it."""
    pixels, z, scores = _body(scale_px_per_m=400.0, lean_dz=0.3)
    assert metres_per_pixel(pixels, z, scores) == pytest.approx(1 / 400.0, rel=1e-4)


def test_lift_puts_the_hips_at_the_origin_in_metres() -> None:
    pixels, z, scores = _body(scale_px_per_m=250.0, lean_dz=0.1)
    lifted = lift(pixels, z, scores)
    assert lifted is not None
    assert np.allclose((lifted[L_HIP] + lifted[R_HIP]) / 2, 0.0, atol=1e-5)
    length = np.linalg.norm(lifted[L_SHOULDER] - lifted[L_HIP])
    assert length == pytest.approx(TORSO_M, rel=1e-3)


def test_without_hips_the_shoulders_set_the_scale() -> None:
    pixels, z, scores = _body(scale_px_per_m=300.0)
    scores[[L_HIP, R_HIP]] = 0.0
    assert metres_per_pixel(pixels, z, scores) == pytest.approx(1 / 300.0, rel=1e-3)


def test_nothing_to_measure_means_no_3d() -> None:
    pixels, z, scores = _body()
    scores[:] = 0.0
    assert lift(pixels, z, scores) is None


def test_a_peakless_keypoint_is_zeroed_in_3d_too() -> None:
    pixels, z, scores = _body()
    lifted = lift(pixels, z, scores)
    assert lifted is not None
    assert np.all(lifted[0] == 0.0)          # the nose scored 0
