"""Runs only when weights are already cached. Marked, so CI skips it.

What only a real run can add is the affine itself: a box a real detector drew
around a real person, a crop taken from a real photograph, and coordinates a
real pose model answered with. ``tests/unit/test_rtmw_wiring.py`` pins the same
contract against stubbed sessions and is the version CI executes; it can only
pin the arithmetic, because it chooses the SimCC peaks itself. The question
here is the one a user would ask of the picture: did the skeleton land on the
person?

That question used to be asked as "are the keypoints inside the image", which
cannot fail. ``decode_simcc`` answers in crop pixels — 0..192 by 0..256 for the
lite preset — and any photograph worth cropping is bigger than its own crop, so
deleting the untransform from ``RTMWEngine.infer`` left every skeleton piled
into the top-left corner of the frame and every assertion in this file green.
A keypoint is a claim about one person, and that person's box is known, so the
bound with teeth is the person rather than the picture.
"""

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from alchemypose.core.topology import COCO_WHOLEBODY_133

pytestmark = pytest.mark.models

FIXTURE = Path(__file__).parent / "fixtures" / "two_people.jpg"

#: Confidence above which a keypoint is treated as placed rather than guessed.
#: The same 0.3-0.5 band the GUI's threshold slider and rtmlib's ``kpt_thr``
#: work in; on this fixture the lite preset is over 0.5 on all 133 of both
#: people, so nothing here rests on where in that band the line is drawn.
CONFIDENCE = 0.5

#: How far outside its own detection box a keypoint may sit, as a fraction of
#: that box's own width and height.
#:
#: Some overshoot is real, not a bug. YOLOX draws its box around what it is sure
#: of, and a few keypoints legitimately fall outside that: the crown of the head
#: above a box that stops at the hairline, an ear, a heel behind the foot. On
#: this fixture the worst of them is the top of person 0's head, 18 px above a
#: box 413 px tall — 4.4% of the box's height — so 15% is about three times the
#: slack anyone needs, and still an order of magnitude tighter than the failures
#: it has to catch. Measured here: delete the untransform and person 0's
#: keypoints sit 292 px outside their box, 191% of its width; untransform
#: everyone with the first person's affine and person 1's sit 183 px outside
#: theirs, 132% of its width.
#:
#: A fraction of the box rather than a count of pixels, because the two people
#: in this fixture differ in size by a tenth and a photograph of a crowd would
#: differ by much more: 20 px is nothing on the person in the foreground and the
#: whole head of someone at the back. The margin has to scale with the person
#: it is describing, which is what makes the same number right for both.
BBOX_MARGIN_FRACTION = 0.15


@pytest.fixture(scope="module")
def engine() -> Any:
    from alchemypose.inference import weights
    from alchemypose.inference.rtmw import RTMWEngine

    det_key, pose_key = weights.PRESETS["lite"]
    for key in (det_key, pose_key):
        if not (weights.cache_dir() / weights.REGISTRY[key].filename).exists():
            pytest.skip(f"{key} not cached; run `alchemypose models fetch --preset lite`")
    return RTMWEngine(preset="lite")


@pytest.fixture(scope="module")
def people(engine: Any) -> Any:
    """Everyone the engine finds in the fixture, inferred once for the coordinate cases."""
    if not FIXTURE.exists():
        pytest.skip("fixture image missing")
    from alchemypose.media.image import load_image

    return engine.infer(load_image(FIXTURE), source=str(FIXTURE)).frames[0].people


def test_detects_the_people_in_the_fixture(engine: Any) -> None:
    if not FIXTURE.exists():
        pytest.skip("fixture image missing")
    from alchemypose.media.image import load_image

    result = engine.infer(load_image(FIXTURE), source=str(FIXTURE))
    people = result.frames[0].people
    assert len(people) == 2


def test_every_person_carries_the_full_topology(engine: Any) -> None:
    if not FIXTURE.exists():
        pytest.skip("fixture image missing")
    from alchemypose.media.image import load_image

    result = engine.infer(load_image(FIXTURE), source=str(FIXTURE))
    for person in result.frames[0].people:
        assert len(person) == len(COCO_WHOLEBODY_133)
        assert person.keypoints.shape == (133, 2)


def test_keypoints_land_inside_the_person_they_describe(people: Any) -> None:
    """Every confident keypoint must sit in its own person's box, give or take the margin.

    This is the invariant "inside the image" was standing in for, and the
    difference is the whole point: crop coordinates are inside the image, and
    they are nowhere near the person. Each keypoint here came out of one
    detection's crop, so the box that produced it is the thing it has to agree
    with — and the margin is a fraction of that box, so the check means the same
    thing for the person at the front of the photograph as for the one behind.

    It fails on any of the ways the mapping can go wrong, not just the one that
    prompted it: the untransform dropped, the affine applied inverted, or one
    person's affine used for another.
    """
    assert people, "no one was detected in the fixture, so this case would check nothing at all"

    for index, person in enumerate(people):
        assert person.bbox is not None, f"person {index} came back without a box to be checked against"
        x0, y0, x1, y1 = person.bbox
        box_w, box_h = x1 - x0, y1 - y0
        confident = person.scores > CONFIDENCE
        assert int(confident.sum()) >= len(COCO_WHOLEBODY_133) // 2, (
            f"only {int(confident.sum())} of person {index}'s {len(COCO_WHOLEBODY_133)} keypoints "
            f"score above {CONFIDENCE}. This case only looks at confident ones, so on this fixture "
            f"it is meant to be looking at nearly all of them; a handful means it has quietly "
            f"stopped measuring the pose it claims to measure"
        )

        keypoints = person.keypoints[confident]
        outside_x = np.maximum(np.maximum(x0 - keypoints[:, 0], keypoints[:, 0] - x1), 0.0) / box_w
        outside_y = np.maximum(np.maximum(y0 - keypoints[:, 1], keypoints[:, 1] - y1), 0.0) / box_h
        outside = np.maximum(outside_x, outside_y)
        worst = int(np.argmax(outside))

        assert float(outside.max()) <= BBOX_MARGIN_FRACTION, (
            f"person {index} was detected at ({x0:.1f}, {y0:.1f}, {x1:.1f}, {y1:.1f}), a box "
            f"{box_w:.1f} x {box_h:.1f}, but {int((outside > BBOX_MARGIN_FRACTION).sum())} of their "
            f"{keypoints.shape[0]} confident keypoints are further outside it than "
            f"{BBOX_MARGIN_FRACTION:.0%} of its own size. The worst sits at "
            f"({keypoints[worst][0]:.1f}, {keypoints[worst][1]:.1f}), out by "
            f"{float(outside_x[worst]) * box_w:.1f} px in x and {float(outside_y[worst]) * box_h:.1f} "
            f"px in y — {float(outside.max()):.1%} of the box. A keypoint describes the person it "
            f"was decoded from, so it cannot be that far from them: either RTMWEngine.infer never "
            f"mapped it out of crop space, or it used the wrong affine to do it"
        )


def test_no_ones_keypoints_land_on_somebody_else(people: Any) -> None:
    """With two people in the frame, neither one's skeleton may be drawn on the other.

    The multi-person form of the same defect, and the one a per-person bound is
    needed to see: reuse one detection's affine for everybody — hoist it out of
    the loop, or index it wrongly — and two people swap skeletons while every
    keypoint stays inside the image, inside a real crop, and inside a real
    person. The boxes in this fixture do not overlap, which is what lets this be
    stated so plainly; the case says so first, rather than assuming it.
    """
    boxes = [person.bbox for person in people]
    assert all(box is not None for box in boxes) and len(boxes) > 1, (
        f"this case needs at least two people with boxes and got {boxes}; with one person there is "
        f"no other skeleton to be confused with and nothing here can fail"
    )

    for index, box in enumerate(boxes):
        for other, their_box in enumerate(boxes):
            if other == index:
                continue
            ax0, ay0, ax1, ay1 = box
            bx0, by0, bx1, by1 = their_box
            assert ax1 < bx0 or bx1 < ax0 or ay1 < by0 or by1 < ay0, (
                f"the boxes of person {index} and person {other} overlap ({box} and {their_box}), "
                f"so a keypoint could legitimately be inside both and this case cannot fail. The "
                f"fixture has changed; state the invariant another way rather than deleting it"
            )

    for index, person in enumerate(people):
        confident = person.scores > CONFIDENCE
        keypoints = person.keypoints[confident]
        for other, (bx0, by0, bx1, by1) in enumerate(boxes):
            if other == index:
                continue
            trespassing = np.flatnonzero(
                (keypoints[:, 0] >= bx0)
                & (keypoints[:, 0] <= bx1)
                & (keypoints[:, 1] >= by0)
                & (keypoints[:, 1] <= by1)
            )
            assert trespassing.size == 0, (
                f"{trespassing.size} of person {index}'s confident keypoints are inside person "
                f"{other}'s box ({bx0:.1f}, {by0:.1f}, {bx1:.1f}, {by1:.1f}) — the first at "
                f"({keypoints[trespassing[0]][0]:.1f}, {keypoints[trespassing[0]][1]:.1f}), while "
                f"person {index} stands at {person.bbox}. RTMWEngine.infer is untransforming this "
                f"person with another person's affine, so the two skeletons have been drawn on the "
                f"wrong bodies"
            )
