"""One tracked person's 3D motion as a binary glTF 2.0 file (``.glb``).

What is in the file, and why it is shaped this way:

* **One node per keypoint**, named after it, each with a ``translation``
  animation sampled at the clip's own timestamps. Any glTF importer (Blender
  imports them as empties) can read the motion without knowing this package.
* **A line mesh of the bones, skinned to those nodes.** Every vertex is one
  keypoint, weighted 1.0 to that keypoint's node, and each node's inverse bind
  matrix undoes its bind position — so as the nodes move, the bones move with
  them and the file plays back as a skeleton, not as a cloud of dots.

Axes are converted here and nowhere else, as the 3D side has always intended:
stored 3D is OpenCV camera space (x right, y down, z away from the camera); glTF
is y up with +z towards the viewer, so ``gltf = (x, -y, -z)`` — a half turn about
x, which keeps the frame right-handed. Units stay metres, which is glTF's unit.

A keypoint the model was not confident about, or a frame the person is absent
from, holds its last confident position rather than snapping to the origin; one
that is never confident sits at the root. The file carries motion, not scores.
"""

from __future__ import annotations

import json
import struct

import numpy as np
from numpy.typing import NDArray

from alchemypose.core.schema import PoseFrame, PoseResult

F32 = NDArray[np.float32]

#: Below this confidence a keypoint's 3D position is not trusted, and held instead.
MIN_SCORE = 0.3

_GLB_MAGIC = 0x46546C67
_CHUNK_JSON = 0x4E4F534A
_CHUNK_BIN = 0x004E4942
_FLOAT, _UNSIGNED_BYTE, _UNSIGNED_SHORT = 5126, 5121, 5123
_LINES = 1
_ARRAY_BUFFER, _ELEMENT_ARRAY_BUFFER = 34962, 34963


class NoMotion(ValueError):
    """Raised when the chosen person has no 3D keypoints to export."""


def to_gltf_axes(points: F32) -> F32:
    """OpenCV camera axes -> glTF's y-up axes."""
    return np.column_stack([points[:, 0], -points[:, 1], -points[:, 2]]).astype(np.float32)


def motion_of(frames: tuple[PoseFrame, ...], track_id: int, keypoints: int) -> tuple[F32, F32]:
    """``(times (T,), positions (T, K, 3))`` in glTF axes for one track, gaps held."""
    times: list[float] = []
    poses: list[F32] = []
    held = np.full((keypoints, 3), np.nan, dtype=np.float32)
    for position, frame in enumerate(frames):
        person = next((p for p in frame.people if p.track_id == track_id), None)
        if person is None and all(p.track_id is None for p in frame.people) and track_id < len(frame.people):
            person = frame.people[track_id]          # a still: untracked, so "track" is the index
        if person is not None and person.keypoints_3d is not None:
            fresh = to_gltf_axes(person.keypoints_3d)
            ok = person.scores >= MIN_SCORE
            held[ok] = fresh[ok]
        elif not poses:
            continue                       # do not start the clip before the person appears
        times.append(frame.timestamp_s if frame.timestamp_s is not None else float(position))
        poses.append(held.copy())
    if not poses:
        raise NoMotion(f"track {track_id} has no 3D keypoints; analyse with the 3d preset")
    stacked = np.stack(poses)
    # Joints never confident anywhere sit at the root; ones confident only later
    # take their first known position for the frames before it.
    for k in range(keypoints):
        column = stacked[:, k]
        known = np.where(~np.isnan(column[:, 0]))[0]
        if known.size == 0:
            column[:] = 0.0
        else:
            column[: known[0]] = column[known[0]]
    t = np.asarray(times, dtype=np.float32)
    return t - t[0], stacked.astype(np.float32)


class _Buffer:
    """Accumulates binary views, 4-byte aligned, and the accessors over them."""

    def __init__(self) -> None:
        self.data = bytearray()
        self.views: list[dict[str, object]] = []
        self.accessors: list[dict[str, object]] = []

    def add(
        self,
        array: np.ndarray,
        component: int,
        kind: str,
        target: int | None = None,
        with_bounds: bool = False,
    ) -> int:
        while len(self.data) % 4:
            self.data.append(0)
        raw = np.ascontiguousarray(array).tobytes()
        view: dict[str, object] = {"buffer": 0, "byteOffset": len(self.data), "byteLength": len(raw)}
        if target is not None:
            view["target"] = target
        self.data.extend(raw)
        self.views.append(view)
        count = int(array.shape[0])
        accessor: dict[str, object] = {
            "bufferView": len(self.views) - 1,
            "componentType": component,
            "count": count,
            "type": kind,
        }
        if with_bounds:
            flat = array.reshape(count, -1).astype(np.float64)
            accessor["min"] = flat.min(axis=0).tolist()
            accessor["max"] = flat.max(axis=0).tolist()
        self.accessors.append(accessor)
        return len(self.accessors) - 1


def skeleton_glb(result: PoseResult, frames: tuple[PoseFrame, ...] | None = None, track_id: int = 0) -> bytes:
    """The ``.glb`` bytes for ``track_id``'s motion through ``frames`` (default: the result's own)."""
    topology = result.topology
    names = topology.keypoint_names
    k = len(names)
    times, motion = motion_of(frames if frames is not None else result.frames, track_id, k)
    bind = motion[0]

    buf = _Buffer()
    positions = buf.add(bind, _FLOAT, "VEC3", _ARRAY_BUFFER, with_bounds=True)
    joints = np.zeros((k, 4), dtype=np.uint8)
    joints[:, 0] = np.arange(k, dtype=np.uint8)
    joint_acc = buf.add(joints, _UNSIGNED_BYTE, "VEC4", _ARRAY_BUFFER)
    weights = np.zeros((k, 4), dtype=np.float32)
    weights[:, 0] = 1.0
    weight_acc = buf.add(weights, _FLOAT, "VEC4", _ARRAY_BUFFER)
    edges = np.asarray(topology.edges, dtype=np.uint16).reshape(-1)
    index_acc = buf.add(edges, _UNSIGNED_SHORT, "SCALAR", _ELEMENT_ARRAY_BUFFER)

    # Column-major 4x4: identity with the translation -bind in the last column.
    inverse_bind = np.tile(np.eye(4, dtype=np.float32), (k, 1, 1))
    inverse_bind[:, :3, 3] = -bind
    ibm_acc = buf.add(inverse_bind.transpose(0, 2, 1).reshape(k, 16), _FLOAT, "MAT4")

    time_acc = buf.add(times, _FLOAT, "SCALAR", with_bounds=True)
    samplers, channels = [], []
    for joint in range(k):
        out_acc = buf.add(motion[:, joint, :], _FLOAT, "VEC3")
        samplers.append({"input": time_acc, "output": out_acc, "interpolation": "LINEAR"})
        channels.append({"sampler": joint, "target": {"node": joint + 1, "path": "translation"}})

    root = {"name": "skeleton", "children": list(range(1, k + 1))}
    joint_nodes = [{"name": names[i], "translation": [float(v) for v in bind[i]]} for i in range(k)]
    mesh_node = {"name": "bones", "mesh": 0, "skin": 0}
    document = {
        "asset": {"version": "2.0", "generator": "AlchemyPose"},
        "scene": 0,
        "scenes": [{"nodes": [0, k + 1]}],
        "nodes": [root, *joint_nodes, mesh_node],
        "meshes": [{
            "name": "bones",
            "primitives": [{
                "attributes": {"POSITION": positions, "JOINTS_0": joint_acc, "WEIGHTS_0": weight_acc},
                "indices": index_acc,
                "mode": _LINES,
            }],
        }],
        "skins": [{"joints": list(range(1, k + 1)), "inverseBindMatrices": ibm_acc, "skeleton": 0}],
        "animations": [{"name": f"track {track_id}", "samplers": samplers, "channels": channels}],
        "accessors": buf.accessors,
        "bufferViews": buf.views,
        "buffers": [{"byteLength": len(buf.data)}],
        "extras": {"source": result.source, "model": result.model, "track_id": track_id},
    }

    json_bytes = json.dumps(document, separators=(",", ":")).encode("utf-8")
    json_bytes += b" " * (-len(json_bytes) % 4)
    bin_bytes = bytes(buf.data) + b"\0" * (-len(buf.data) % 4)
    total = 12 + 8 + len(json_bytes) + 8 + len(bin_bytes)
    return b"".join([
        struct.pack("<III", _GLB_MAGIC, 2, total),
        struct.pack("<II", len(json_bytes), _CHUNK_JSON), json_bytes,
        struct.pack("<II", len(bin_bytes), _CHUNK_BIN), bin_bytes,
    ])
