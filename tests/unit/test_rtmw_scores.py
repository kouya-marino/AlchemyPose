"""The per-model SimCC peak scale that makes ``PersonPose.scores`` comparable across presets.

No weights are loaded here; the table itself is the contract. Its values were
measured on real model output: RTMW-l's raw peaks reach 8.6 where RTMW-m and
DWPose-l reach about 1.0, a 10x difference that matches the ``label_beta /
beta`` ratio of their training losses.

Checking each entry against its own literal would be circular — it would pass
whatever number the table happened to hold. So the peaks measured on real
photographs are recorded here as ``MEASURED_RAW_PEAKS`` and pushed back
through the real decoder, and the assertions are about the confidences that
come out: a recorded scale is right when it turns that model's own range of
peaks into a usable range of confidence, and wrong when it flattens the range
against either end. That catches a scale that is too small (peaks clip to 1.0
and a threshold filters nothing) and one that is too large (everything falls
to near zero and a threshold filters everything), which the literals cannot.

``tests/integration/test_score_scale.py`` checks the same property end to end
against cached weights.
"""

import numpy as np
import pytest

from alchemypose.errors import WeightsError
from alchemypose.inference import weights
from alchemypose.inference.rtmw import PEAK_SCALE, peak_scale_for
from alchemypose.inference.simcc import decode_simcc

#: Raw (undivided) SimCC peak heights each pose model actually emits, as
#: ``(minimum, median, maximum)`` over every keypoint of every person found in
#: ``tests/integration/fixtures/two_people.jpg``. Measured by reading the peak
#: straight off the ONNX output, before any rescaling or clipping.
#:
#: RTMW-l's spread — 0.94 to 8.58 — is the whole reason this table exists. The
#: two unit-scale models top out just above 1.0, so dividing everything by 1.0
#: looks harmless until RTMW-l ships.
MEASURED_RAW_PEAKS: dict[str, tuple[float, float, float]] = {
    "rtmw_m_256x192": (0.561, 0.920, 1.055),
    "dwpose_l_384x288": (0.062, 0.789, 1.016),
    "rtmw_l_384x288": (0.938, 4.898, 8.576),
    # Measured on Linux, CUDA provider, over the three people YOLOX-s finds.
    "rtmw3d_x_384x288": (0.306, 0.811, 1.002),
}

#: The most confident joint in a photograph must read as confident.
CONFIDENT_JOINT_AT_LEAST = 0.5
#: The least confident joint must read as doubtful, so a threshold can drop it.
DOUBTFUL_JOINT_AT_MOST = 0.7
#: And the two must stay far enough apart that the range in between is usable.
MIN_CONFIDENCE_SPREAD = 0.35


def _confidence(raw_peak: float, peak_scale: float) -> float:
    """Run one raw peak through the real decoder and return the confidence it yields."""
    simcc_x = np.zeros((1, 1, 16), dtype=np.float32)
    simcc_y = np.zeros((1, 1, 16), dtype=np.float32)
    simcc_x[0, 0, 4] = np.float32(raw_peak)
    simcc_y[0, 0, 6] = np.float32(raw_peak)
    _, scores = decode_simcc(simcc_x, simcc_y, peak_scale=peak_scale)
    return float(scores[0, 0])


def test_every_preset_pose_model_has_a_recorded_peak_scale() -> None:
    for _detector, pose_key in weights.PRESETS.values():
        assert peak_scale_for(pose_key) > 0.0
        assert pose_key in MEASURED_RAW_PEAKS, (
            f"{pose_key} has a peak scale but no measured peaks to justify it; measure its raw "
            f"peak range on real photographs and record it in MEASURED_RAW_PEAKS"
        )


def test_measured_peaks_cover_the_whole_scale_table() -> None:
    assert set(MEASURED_RAW_PEAKS) == set(PEAK_SCALE)


def test_rtmw_l_logits_are_ten_times_taller_than_the_unit_scale_models() -> None:
    assert peak_scale_for("rtmw_m_256x192") == 1.0
    assert peak_scale_for("dwpose_l_384x288") == 1.0
    assert peak_scale_for("rtmw_l_384x288") == 10.0 * peak_scale_for("rtmw_m_256x192")


@pytest.mark.parametrize("pose_key", sorted(MEASURED_RAW_PEAKS))
def test_the_recorded_scale_turns_measured_peaks_into_a_usable_confidence_range(pose_key: str) -> None:
    """The scale must map this model's own peak range onto a spread-out confidence range.

    Both ways of getting it wrong are caught here. Too small and the tall peaks
    clip: RTMW-l divided by 1.0 puts its quietest joint at 0.94 and everything
    above it at 1.0, a spread of 0.06 that no threshold can cut through. Too
    large and everything sinks: divided by 100.0 its most confident joint reads
    0.09 and every threshold hides the whole skeleton.
    """
    lowest_raw, median_raw, highest_raw = MEASURED_RAW_PEAKS[pose_key]
    scale = peak_scale_for(pose_key)
    lowest = _confidence(lowest_raw, scale)
    median = _confidence(median_raw, scale)
    highest = _confidence(highest_raw, scale)
    seen = (
        f"{pose_key}: peaks {lowest_raw}/{median_raw}/{highest_raw} over scale {scale}"
        f" -> {lowest:.3f}/{median:.3f}/{highest:.3f}"
    )

    assert highest >= CONFIDENT_JOINT_AT_LEAST, f"the clearest joint reads as doubtful — {seen}"
    assert lowest <= DOUBTFUL_JOINT_AT_MOST, f"the weakest joint reads as confident — {seen}"
    assert highest - lowest >= MIN_CONFIDENCE_SPREAD, (
        f"the model's whole range of certainty is squeezed into {highest - lowest:.3f} of "
        f"confidence, so a threshold has almost nothing to cut through — {seen}"
    )
    assert 0.1 < median < 1.0, (
        f"a typical joint lands at {median:.3f}, against the top or the bottom of the range "
        f"rather than inside it — {seen}"
    )


def test_an_unlisted_model_fails_loudly_instead_of_assuming_unit_scale() -> None:
    """A silent default of 1.0 is exactly how a 10x-scaled score went unnoticed before."""
    with pytest.raises(WeightsError, match="peak scale"):
        peak_scale_for("rtmw_x_384x288")


def test_peak_scale_entries_are_registry_pose_models_not_detectors() -> None:
    detectors = {detector for detector, _pose in weights.PRESETS.values()}
    for key in PEAK_SCALE:
        assert key in weights.REGISTRY
        assert key not in detectors
