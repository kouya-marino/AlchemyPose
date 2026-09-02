"""The per-model SimCC peak scale that makes ``PersonPose.scores`` comparable across presets.

No weights are loaded here; the table itself is the contract. Its values were
measured on real output (score-scale report): RTMW-l's raw peaks reach 8.6 where
RTMW-m and DWPose-l reach about 1.0, a 10x difference that matches the
``label_beta / beta`` ratio of their training losses.
"""

import pytest

from alchemypose.errors import WeightsError
from alchemypose.inference import weights
from alchemypose.inference.rtmw import PEAK_SCALE, peak_scale_for


def test_every_preset_pose_model_has_a_recorded_peak_scale() -> None:
    for _detector, pose_key in weights.PRESETS.values():
        assert peak_scale_for(pose_key) > 0.0


def test_rtmw_l_logits_are_ten_times_taller_than_the_unit_scale_models() -> None:
    assert peak_scale_for("rtmw_l_384x288") == 10.0
    assert peak_scale_for("rtmw_m_256x192") == 1.0
    assert peak_scale_for("dwpose_l_384x288") == 1.0


def test_an_unlisted_model_fails_loudly_instead_of_assuming_unit_scale() -> None:
    """A silent default of 1.0 is exactly how a 10x-scaled score went unnoticed before."""
    with pytest.raises(WeightsError, match="peak scale"):
        peak_scale_for("rtmw_x_384x288")


def test_peak_scale_entries_are_registry_pose_models_not_detectors() -> None:
    detectors = {detector for detector, _pose in weights.PRESETS.values()}
    for key in PEAK_SCALE:
        assert key in weights.REGISTRY
        assert key not in detectors
