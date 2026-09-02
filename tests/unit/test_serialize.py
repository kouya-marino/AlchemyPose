import json

import numpy as np

from alchemypose.core.schema import Camera, PersonPose, PoseFrame, PoseResult, Space
from alchemypose.core.serialize import from_json, to_csv, to_json
from alchemypose.core.topology import COCO_WHOLEBODY_133

rng = np.random.default_rng(0)


def _result(n_people: int = 2) -> PoseResult:
    people = tuple(
        PersonPose(
            keypoints=rng.random((133, 2), dtype=np.float32) * 640,
            scores=rng.random((133,), dtype=np.float32),
            bbox=(10.0, 20.0, 300.0, 400.0),
            bbox_score=0.9,
        )
        for _ in range(n_people)
    )
    frame = PoseFrame(frame_index=0, timestamp_s=None, image_size=(640, 480), people=people)
    return PoseResult(
        topology=COCO_WHOLEBODY_133,
        frames=(frame,),
        source="photo.jpg",
        model="rtmw-l_384x288",
        camera=Camera.default_for(640, 480),
    )


def test_json_round_trip_preserves_keypoints_exactly() -> None:
    original = _result()
    restored = from_json(to_json(original))
    a = original.frames[0].people[0]
    b = restored.frames[0].people[0]
    np.testing.assert_allclose(a.keypoints, b.keypoints, rtol=0, atol=1e-6)
    np.testing.assert_allclose(a.scores, b.scores, rtol=0, atol=1e-6)


def test_json_round_trip_preserves_metadata() -> None:
    original = _result()
    restored = from_json(to_json(original))
    assert restored.source == original.source
    assert restored.model == original.model
    assert restored.schema_version == 1
    assert restored.topology.name == "coco_wholebody_133"
    assert restored.camera is not None and restored.camera.fx == 640.0


def test_json_is_self_describing() -> None:
    payload = json.loads(to_json(_result()))
    assert payload["schema_version"] == 1
    assert payload["topology"]["name"] == "coco_wholebody_133"
    assert len(payload["topology"]["keypoint_names"]) == 133
    assert payload["frames"][0]["image_size"] == [640, 480]


def test_json_round_trip_preserves_empty_3d_slots() -> None:
    restored = from_json(to_json(_result()))
    person = restored.frames[0].people[0]
    assert person.keypoints_3d is None
    assert person.space_3d is None
    assert person.space_2d is Space.IMAGE_PX


def test_csv_has_one_row_per_keypoint_per_person_plus_a_header() -> None:
    text = to_csv(_result(n_people=2))
    lines = text.strip().splitlines()
    assert lines[0] == "frame,person,keypoint_index,keypoint_name,x,y,score"
    assert len(lines) == 1 + 2 * 133


def test_csv_names_come_from_the_topology() -> None:
    lines = to_csv(_result(n_people=1)).strip().splitlines()
    assert lines[1].split(",")[3] == "nose"
