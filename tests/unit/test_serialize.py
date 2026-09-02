import json
import logging

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
    np.testing.assert_array_equal(a.keypoints, b.keypoints)
    np.testing.assert_array_equal(a.scores, b.scores)


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


def test_non_finite_values_survive_as_null_and_come_back_as_nan() -> None:
    result = _result()
    kp = result.frames[0].people[0].keypoints.copy()
    kp[5] = [np.nan, np.nan]
    people = (PersonPose(keypoints=kp, scores=result.frames[0].people[0].scores),)
    frame = PoseFrame(frame_index=0, timestamp_s=None, image_size=(640, 480), people=people)
    r = PoseResult(topology=COCO_WHOLEBODY_133, frames=(frame,), source="x", model="m")
    text = to_json(r)
    assert "NaN" not in text, "bare NaN is not valid JSON"
    assert json.loads(text) is not None
    restored = from_json(text)
    assert np.isnan(restored.frames[0].people[0].keypoints[5]).all()


def test_an_unknown_topology_is_rebuilt_from_the_file() -> None:
    payload = json.loads(to_json(_result()))
    payload["topology"]["name"] = "some_future_schema"
    restored = from_json(json.dumps(payload))
    assert restored.topology.name == "some_future_schema"
    assert len(restored.topology.keypoint_names) == 133


def test_a_known_name_with_a_different_layout_warns(caplog) -> None:  # type: ignore[no-untyped-def]
    payload = json.loads(to_json(_result()))
    payload["topology"]["edges"] = [[0, 1]]
    with caplog.at_level(logging.WARNING):
        restored = from_json(json.dumps(payload))
    assert len(restored.topology.edges) == 127, "installed definition wins"
    assert "different keypoint layout" in caplog.text
