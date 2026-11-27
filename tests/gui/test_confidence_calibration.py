"""The confidence cues against the score distribution they were calibrated on.

``test_polish.py`` checks the *shape* of ``confidence_opacity`` and of the dash
rule, but derives its inputs from the constants themselves, so it slides along
with whatever they happen to be: restoring the values that shipped wrong
(``DASH_BELOW = 0.5``, ``FULL_OPACITY_ABOVE = 0.8``) leaves it green. This file
pins the *outcome* instead. ``fixtures/two_people_scores.json`` holds real
``PersonPose.scores`` for the fixture photograph on both presets, so no weights
are needed, and the assertions are about how much of the default skeleton ends
up dashed and how much reaches full opacity at the default threshold.

Seen to fail: with the old constants, 52 of the 61 body bones drawn on
``performance`` are dashed and not one keypoint is fully opaque; both
performance tests below go red.
"""

import json
from pathlib import Path

import numpy as np
import pytest

from alchemypose.core.presets import selection_for
from alchemypose.core.topology import COCO_WHOLEBODY_133
from alchemypose.gui.canvas.scene import DASH_BELOW, confidence_opacity

pytestmark = pytest.mark.gui

DEFAULT_THRESHOLD = 0.30
FIXTURE = Path(__file__).parent / "fixtures" / "two_people_scores.json"


def _people(preset: str) -> list[np.ndarray]:
    document = json.loads(FIXTURE.read_text())
    return [np.asarray(scores, dtype=np.float32) for scores in document[preset]]


def _drawn_bones(preset: str) -> list[float]:
    """The weaker-endpoint score of every full-body bone drawn at the default threshold."""
    selection = selection_for("full_body")
    weakest: list[float] = []
    for scores in _people(preset):
        for a, b in COCO_WHOLEBODY_133.edges:
            if not selection.is_edge_visible((a, b)):
                continue
            if min(scores[a], scores[b]) >= DEFAULT_THRESHOLD:
                weakest.append(float(min(scores[a], scores[b])))
    return weakest


def _drawn_keypoints(preset: str) -> list[float]:
    selection = selection_for("full_body")
    return [
        float(scores[i])
        for scores in _people(preset)
        for i in sorted(selection.visible)
        if scores[i] >= DEFAULT_THRESHOLD
    ]


def _dashed_fraction(preset: str) -> float:
    bones = _drawn_bones(preset)
    assert len(bones) >= 40, "the fixture should draw a whole skeleton or two"
    return sum(score < DASH_BELOW for score in bones) / len(bones)


def _full_opacity_fraction(preset: str) -> float:
    keypoints = _drawn_keypoints(preset)
    assert len(keypoints) >= 40
    return sum(confidence_opacity(s, DEFAULT_THRESHOLD) == 1.0 for s in keypoints) / len(keypoints)


def test_dashing_is_reserved_for_doubtful_limbs_on_the_default_preset() -> None:
    """At the default threshold almost nothing should be dashed: a cue on everything is no cue."""
    assert _dashed_fraction("performance") <= 0.20


def test_the_default_preset_keeps_a_visible_spread_of_opacity() -> None:
    """Some of the skeleton fully opaque, but not all of it — otherwise the fade says nothing."""
    assert 0.05 <= _full_opacity_fraction("performance") <= 0.60


def test_the_lite_preset_reads_as_confident_because_it_is() -> None:
    """RTMW-m's scores are uniformly high; drawing them faded would misreport them."""
    assert _dashed_fraction("lite") <= 0.20
    assert _full_opacity_fraction("lite") >= 0.80
