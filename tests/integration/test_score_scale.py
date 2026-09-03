"""Runs only when weights are already cached. Marked, so CI skips it.

Pins the ``PersonPose.scores`` contract on real model output for a unit-scale
model (``lite``: RTMW-m) and the 10x model (``performance``: RTMW-l).

The property under test is the *shape* of the confidence distribution, not
merely its range, because clipping hides the damage that matters. Divide
RTMW-l's raw SimCC peaks by 1.0 instead of by its measured 10.0 and every
score still lands inside ``[0, 1]`` — ``decode_simcc`` clips — but the peaks
run 0.94 to 8.58, so 397 of this fixture's 399 keypoints pile up at exactly
1.0 and the confidence threshold filters nothing anywhere below its last
notch. A test that only asks "does the drawn count change somewhere" is
satisfied by those two keypoints of movement and reports a dead control as a
working one.

So the assertions below ask instead that the count falls *in proportion*
across the useful part of the range. ``PROFILES`` pins that per preset with
numbers measured on this fixture with the shipped weights, and
``test_the_threshold_has_a_usable_middle`` pins the same idea without any
per-model constants at all.

Every band is two-sided on purpose. A peak scale that is too small leaves the
distribution jammed against 1.0; one that is too large collapses it towards 0;
both make the threshold useless, and a guard that only catches one direction
would let the other ship.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from alchemypose.core.topology import COCO_WHOLEBODY_133

pytestmark = pytest.mark.models

FIXTURE = Path(__file__).parent / "fixtures" / "two_people.jpg"

#: Thresholds a user can actually select, sampled the way the GUI slider steps.
THRESHOLD_GRID = np.round(np.arange(0.0, 1.0001, 0.05), 2)

#: A threshold is doing useful work when it keeps a real subset — neither
#: nearly everything nor nearly nothing.
MIDDLE_BAND = (0.20, 0.80)

#: How much of the range must do that useful work. Measured on this fixture:
#: ``performance`` keeps a middle subset at 11 of the 21 grid thresholds
#: (0.25 through 0.75), ``lite`` at 5 of them (0.75 through 0.95). Both
#: mis-scalings collapse this to one grid point or none.
MIN_USEFUL_THRESHOLDS = 3
MIN_USEFUL_SPAN = 0.10


@dataclass(frozen=True)
class ScoreProfile:
    """What a correctly rescaled run of one preset looks like on this fixture.

    ``at_most`` and ``at_least`` are ``(threshold, fraction)`` pairs bounding
    the share of keypoints that survive that threshold. The ceilings say the
    distribution is not jammed against the top of the range; the floors say it
    has not been crushed towards the bottom.
    """

    median_between: tuple[float, float]
    max_at_least: float
    at_most: tuple[tuple[float, float], ...]
    at_least: tuple[tuple[float, float], ...]


#: Measured on ``two_people.jpg`` with the shipped weights, then given margin.
#:
#: ``lite`` (RTMW-m, peak scale 1.0), 266 keypoints over 2 people: raw peaks
#: 0.56 to 1.05, confidences min 0.56 / median 0.92 / max 1.00, survival
#: 100 % at 0.1, 99 % at 0.6, 64 % at 0.8, 35 % at 0.95.
#:
#: ``performance`` (RTMW-l, peak scale 10.0), 399 keypoints over 3 people: raw
#: peaks 0.94 to 8.58, confidences min 0.09 / median 0.49 / max 0.86, survival
#: 100 % at 0.1, 71 % at 0.3, 62 % at 0.4, 7 % at 0.8, 0 % at 0.95.
#:
#: The two profiles differ because the models genuinely differ — RTMW-m is
#: confident about almost every joint in this photograph and RTMW-l spreads
#: itself across the range — which is exactly why one shared band would either
#: be too loose to catch anything or fail on a healthy model.
PROFILES: dict[str, ScoreProfile] = {
    "lite": ScoreProfile(
        median_between=(0.55, 0.99),
        max_at_least=0.80,
        at_most=((0.80, 0.85), (0.95, 0.60)),
        at_least=((0.10, 0.90), (0.60, 0.80)),
    ),
    "performance": ScoreProfile(
        median_between=(0.30, 0.70),
        max_at_least=0.55,
        at_most=((0.40, 0.75), (0.80, 0.25), (0.95, 0.10)),
        at_least=((0.10, 0.75), (0.30, 0.40)),
    ),
}


def _survival(scores: np.ndarray, threshold: float) -> float:
    """Fraction of keypoints the confidence threshold would keep on screen."""
    return float((scores >= threshold).mean())


@pytest.fixture(scope="module", params=sorted(PROFILES))
def preset(request: Any) -> str:
    return str(request.param)


@pytest.fixture(scope="module")
def engine(preset: str) -> Any:
    from alchemypose.inference import weights
    from alchemypose.inference.rtmw import RTMWEngine

    det_key, pose_key = weights.PRESETS[preset]
    for key in (det_key, pose_key):
        if not (weights.cache_dir() / weights.REGISTRY[key].filename).exists():
            pytest.skip(f"{key} not cached; run `alchemypose models fetch --preset {preset}`")
    if not FIXTURE.exists():
        pytest.skip("fixture image missing")
    return RTMWEngine(preset=preset)


@pytest.fixture(scope="module")
def all_scores(engine: Any) -> np.ndarray:
    from alchemypose.media.image import load_image

    result = engine.infer(load_image(FIXTURE), source=str(FIXTURE))
    people = result.frames[0].people
    assert people, "the fixture has people in it"
    scores = np.concatenate([person.scores for person in people])
    # The bands below are fractions of a real crowd; two people is the minimum
    # that makes them mean anything.
    assert scores.size >= 2 * len(COCO_WHOLEBODY_133)
    assert scores.size % len(COCO_WHOLEBODY_133) == 0
    return scores


def test_scores_are_float32_confidences_in_the_unit_interval(all_scores: np.ndarray) -> None:
    """The declared range of ``PersonPose.scores``.

    This guards the decoder's clip, not the peak-scale table: because
    ``decode_simcc`` clips, an unrescaled RTMW-l run whose raw peaks reach 8.58
    still satisfies every assertion here. The tests below are what catch that.
    """
    assert all_scores.dtype == np.float32
    assert np.all(all_scores >= 0.0) and np.all(all_scores <= 1.0)


def test_the_threshold_has_a_usable_middle(all_scores: np.ndarray, preset: str) -> None:
    """A wide, contiguous stretch of the range must keep a real subset of keypoints.

    No per-model constants: whatever the model's own spread of confidence, a
    correctly scaled one has to pass gradually from "draws everything" to
    "draws almost nothing", so several selectable thresholds must land in
    between. Both ways of getting the scale wrong break this — too small and
    the distribution never leaves the top of the range, too large and it drops
    off the bottom between one notch and the next.
    """
    low, high = MIDDLE_BAND
    useful = [t for t in THRESHOLD_GRID if low <= _survival(all_scores, float(t)) <= high]
    survival = {float(t): round(_survival(all_scores, float(t)), 3) for t in THRESHOLD_GRID}

    assert len(useful) >= MIN_USEFUL_THRESHOLDS, (
        f"{preset}: only {len(useful)} of {THRESHOLD_GRID.size} thresholds keep between "
        f"{low:.0%} and {high:.0%} of keypoints, so the control is effectively a switch; "
        f"survival by threshold: {survival}"
    )
    span = float(max(useful) - min(useful))
    assert span >= MIN_USEFUL_SPAN, (
        f"{preset}: the useful part of the range spans only {span:.2f} "
        f"({min(useful):.2f}..{max(useful):.2f}); survival by threshold: {survival}"
    )


def test_confidence_is_spread_across_the_range_not_pinned_to_one_end(
    all_scores: np.ndarray, preset: str
) -> None:
    """The measured shape of a correctly rescaled run, preset by preset."""
    profile = PROFILES[preset]
    summary = (
        f"{preset}: min {all_scores.min():.3f} median {float(np.median(all_scores)):.3f} "
        f"max {all_scores.max():.3f} over {all_scores.size} keypoints"
    )

    low, high = profile.median_between
    median = float(np.median(all_scores))
    assert low <= median <= high, f"median confidence {median:.3f} outside [{low}, {high}] — {summary}"

    assert all_scores.max() >= profile.max_at_least, (
        f"the most confident keypoint only reaches {all_scores.max():.3f}, below "
        f"{profile.max_at_least} — the peak scale is dividing too hard; {summary}"
    )

    for threshold, ceiling in profile.at_most:
        kept = _survival(all_scores, threshold)
        assert kept <= ceiling, (
            f"at threshold {threshold} the overlay still draws {kept:.1%} of keypoints, above the "
            f"{ceiling:.0%} a rescaled run leaves — the peaks are not being divided enough; {summary}"
        )

    for threshold, floor in profile.at_least:
        kept = _survival(all_scores, threshold)
        assert kept >= floor, (
            f"at threshold {threshold} the overlay draws only {kept:.1%} of keypoints, below the "
            f"{floor:.0%} a rescaled run leaves — the peaks are being divided too hard; {summary}"
        )
