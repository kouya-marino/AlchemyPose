"""Runs only when weights are already cached. Marked, so CI skips it.

Proves the ``PersonPose.scores`` contract on real model output for both a
unit-scale model (``lite``: RTMW-m) and the 10x model (``performance``:
RTMW-l): confidences land in ``[0, 1]`` and a 0..1 threshold actually
filters somewhere inside its range, which it did not before RTMW-l's peaks
were rescaled (397 of 399 keypoints scored above 1.0).
"""

from pathlib import Path
from typing import Any

import numpy as np
import pytest

pytestmark = pytest.mark.models

FIXTURE = Path(__file__).parent / "fixtures" / "two_people.jpg"


@pytest.fixture(scope="module", params=["lite", "performance"])
def engine(request: Any) -> Any:
    from alchemypose.inference import weights
    from alchemypose.inference.rtmw import RTMWEngine

    det_key, pose_key = weights.PRESETS[request.param]
    for key in (det_key, pose_key):
        if not (weights.cache_dir() / weights.REGISTRY[key].filename).exists():
            pytest.skip(f"{key} not cached; run `alchemypose models fetch --preset {request.param}`")
    if not FIXTURE.exists():
        pytest.skip("fixture image missing")
    return RTMWEngine(preset=request.param)


@pytest.fixture(scope="module")
def all_scores(engine: Any) -> np.ndarray:
    from alchemypose.media.image import load_image

    result = engine.infer(load_image(FIXTURE), source=str(FIXTURE))
    people = result.frames[0].people
    assert people, "the fixture has people in it"
    return np.concatenate([person.scores for person in people])


def test_scores_are_float32_confidences_in_the_unit_interval(all_scores: np.ndarray) -> None:
    assert all_scores.dtype == np.float32
    assert np.all(all_scores >= 0.0) and np.all(all_scores <= 1.0)


def test_a_zero_to_one_threshold_is_not_inert(all_scores: np.ndarray) -> None:
    """Drawn-count must fall monotonically across the slider and not stay at 'everything'."""
    thresholds = np.linspace(0.0, 1.0, 11)
    drawn = [int((all_scores >= t).sum()) for t in thresholds]
    assert drawn[0] == all_scores.size
    assert drawn == sorted(drawn, reverse=True)
    assert drawn[-1] < all_scores.size, "the top of the slider must hide something"
    assert any(0 < d < all_scores.size for d in drawn), "some threshold must keep a strict subset"
