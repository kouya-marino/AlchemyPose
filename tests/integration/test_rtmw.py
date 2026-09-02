"""Runs only when weights are already cached. Marked, so CI skips it."""

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from alchemypose.core.topology import COCO_WHOLEBODY_133

pytestmark = pytest.mark.models

FIXTURE = Path(__file__).parent / "fixtures" / "two_people.jpg"


@pytest.fixture(scope="module")
def engine() -> Any:
    from alchemypose.inference import weights
    from alchemypose.inference.rtmw import RTMWEngine

    det_key, pose_key = weights.PRESETS["lite"]
    for key in (det_key, pose_key):
        if not (weights.cache_dir() / weights.REGISTRY[key].filename).exists():
            pytest.skip(f"{key} not cached; run `alchemypose models fetch --preset lite`")
    return RTMWEngine(preset="lite")


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


def test_keypoints_land_inside_the_image(engine: Any) -> None:
    if not FIXTURE.exists():
        pytest.skip("fixture image missing")
    from alchemypose.media.image import load_image

    image = load_image(FIXTURE)
    result = engine.infer(image, source="x")
    w, h = result.frames[0].image_size
    for person in result.frames[0].people:
        confident = person.keypoints[person.scores > 0.5]
        assert np.all(confident[:, 0] >= -10) and np.all(confident[:, 0] <= w + 10)
        assert np.all(confident[:, 1] >= -10) and np.all(confident[:, 1] <= h + 10)
