"""The animated .glb: structure, axes, and that the skinned bones follow the joints."""

import json
import struct
from dataclasses import replace

import numpy as np
import pytest

from alchemypose.core.gltf import NoMotion, motion_of, skeleton_glb, to_gltf_axes
from alchemypose.core.schema import PersonPose, PoseFrame, PoseResult, Space
from alchemypose.core.topology import COCO_WHOLEBODY_133

K = 133


def _person(t: float, track: int | None = 0, score: float = 0.9) -> PersonPose:
    rng = np.random.default_rng(int(t * 100))
    kp3 = rng.uniform(-0.5, 0.5, (K, 3)).astype(np.float32)
    kp3[:, 0] += t                          # everyone drifts right over time
    return PersonPose(
        keypoints=np.zeros((K, 2), dtype=np.float32),
        scores=np.full(K, score, dtype=np.float32),
        track_id=track,
        keypoints_3d=kp3,
        space_3d=Space.ROOT_REL,
    )


def _result(frames: int = 5, fps: float = 10.0) -> PoseResult:
    return PoseResult(
        topology=COCO_WHOLEBODY_133,
        frames=tuple(PoseFrame(i, i / fps, (640, 480), (_person(i / fps),)) for i in range(frames)),
        source="clip.mp4",
        model="rtmw3d_x_384x288",
    )


def _parse(glb: bytes) -> tuple[dict, bytes]:
    magic, version, total = struct.unpack_from("<III", glb, 0)
    assert (magic, version, total) == (0x46546C67, 2, len(glb))
    json_len, json_type = struct.unpack_from("<II", glb, 12)
    assert json_type == 0x4E4F534A
    document = json.loads(glb[20 : 20 + json_len])
    bin_len, bin_type = struct.unpack_from("<II", glb, 20 + json_len)
    assert bin_type == 0x004E4942
    start = 28 + json_len
    return document, glb[start : start + bin_len]


def _read(document: dict, blob: bytes, accessor: int) -> np.ndarray:
    acc = document["accessors"][accessor]
    view = document["bufferViews"][acc["bufferView"]]
    dtype = {5126: np.float32, 5121: np.uint8, 5123: np.uint16}[acc["componentType"]]
    width = {"SCALAR": 1, "VEC3": 3, "VEC4": 4, "MAT4": 16}[acc["type"]]
    raw = np.frombuffer(blob, dtype=dtype, count=acc["count"] * width, offset=view["byteOffset"])
    return raw.reshape(acc["count"], width) if width > 1 else raw


def test_gltf_axes_are_a_half_turn_about_x() -> None:
    point = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)       # right, down, away
    assert to_gltf_axes(point).tolist() == [[1.0, -2.0, -3.0]]  # right, up, towards the viewer


def test_the_file_has_a_node_per_keypoint_a_skinned_bone_mesh_and_one_animation() -> None:
    document, _ = _parse(skeleton_glb(_result()))
    names = [n.get("name") for n in document["nodes"]]
    assert names[1 : K + 1] == list(COCO_WHOLEBODY_133.keypoint_names)
    assert document["skins"][0]["joints"] == list(range(1, K + 1))
    assert document["meshes"][0]["primitives"][0]["mode"] == 1        # LINES
    assert len(document["animations"][0]["channels"]) == K


def test_animation_times_start_at_zero_and_follow_the_timestamps() -> None:
    document, blob = _parse(skeleton_glb(_result(frames=5, fps=10.0)))
    times = _read(document, blob, document["animations"][0]["samplers"][0]["input"])
    assert times.tolist() == pytest.approx([0.0, 0.1, 0.2, 0.3, 0.4])


def test_the_skinned_bones_land_exactly_on_the_animated_joints() -> None:
    """vertex' = translation(joint at t) @ inverseBind @ bind vertex = joint position at t."""
    result = _result()
    document, blob = _parse(skeleton_glb(result))
    prim = document["meshes"][0]["primitives"][0]
    bind = _read(document, blob, prim["attributes"]["POSITION"])
    ibm = _read(document, blob, document["skins"][0]["inverseBindMatrices"]).reshape(K, 4, 4).transpose(0, 2, 1)
    frame = 3
    for joint in (0, 9, 100):
        sampler = document["animations"][0]["samplers"][joint]
        at_t = _read(document, blob, sampler["output"])[frame]
        world = np.eye(4)
        world[:3, 3] = at_t
        moved = world @ ibm[joint] @ np.append(bind[joint], 1.0)
        expected = to_gltf_axes(result.frames[frame].people[0].keypoints_3d)[joint]
        assert np.allclose(moved[:3], expected, atol=1e-5)


def test_a_doubtful_joint_holds_its_last_confident_position() -> None:
    result = _result(frames=3)
    shaky = result.frames[2].people[0]
    scores = shaky.scores.copy()
    scores[9] = 0.1
    frames = (*result.frames[:2], replace(result.frames[2], people=(replace(shaky, scores=scores),)))
    _, motion = motion_of(frames, 0, K)
    assert np.allclose(motion[2, 9], motion[1, 9])
    assert not np.allclose(motion[2, 8], motion[1, 8])


def test_the_clip_starts_when_the_person_first_appears() -> None:
    result = _result(frames=4)
    frames = (replace(result.frames[0], people=()), *result.frames[1:])
    times, motion = motion_of(frames, 0, K)
    assert motion.shape[0] == 3
    assert times[0] == 0.0


def test_a_track_without_3d_is_refused() -> None:
    result = _result()
    flat = tuple(replace(f, people=(replace(f.people[0], keypoints_3d=None),)) for f in result.frames)
    with pytest.raises(NoMotion):
        skeleton_glb(replace(result, frames=flat))


def test_a_still_exports_its_first_person_without_a_track_id() -> None:
    still = replace(_result(frames=1), frames=(PoseFrame(0, None, (640, 480), (_person(0.0, track=None),)),))
    document, _ = _parse(skeleton_glb(still, track_id=0))
    assert len(document["nodes"]) == K + 2
