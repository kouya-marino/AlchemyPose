"""JSON and CSV forms of a result.

JSON is self-describing: it carries the topology inline so a file can be read
years later without this package. Arrays become nested lists; a missing point is
``null``. CSV is the flat form analysts ask for.
"""

from __future__ import annotations

import io
import json
import logging
from typing import Any

import numpy as np
from numpy.typing import NDArray

from alchemypose.core.schema import Camera, PersonPose, PoseFrame, PoseResult, Space
from alchemypose.core.topology import COCO_WHOLEBODY_133, Topology

_log = logging.getLogger(__name__)

_KNOWN_TOPOLOGIES: dict[str, Topology] = {COCO_WHOLEBODY_133.name: COCO_WHOLEBODY_133}


def _encode_array(values: NDArray[np.float32]) -> list[Any]:
    """Nested lists, with non-finite values as JSON null.

    json.dumps would otherwise emit bare NaN/Infinity tokens, which are not valid
    JSON and are rejected by strict parsers — the exact portability this format
    exists to provide.
    """
    if values.ndim == 1:
        return [None if not np.isfinite(v) else float(v) for v in values]
    elif values.ndim == 2:
        return [[None if not np.isfinite(v) else float(v) for v in row] for row in values]
    else:
        raise ValueError(f"Expected 1-D or 2-D array, got {values.ndim}D")


def _decode_array(payload: list[Any], shape_hint: tuple[int, ...]) -> NDArray[np.float32]:
    """Inverse: JSON null becomes NaN."""
    if len(shape_hint) == 1:
        return np.asarray([np.nan if v is None else v for v in payload], dtype=np.float32)
    elif len(shape_hint) == 2:
        return np.asarray(
            [[np.nan if v is None else v for v in row] for row in payload], dtype=np.float32
        )
    else:
        raise ValueError(f"Expected 1-D or 2-D shape, got {shape_hint}")


def _person_to_dict(person: PersonPose) -> dict[str, Any]:
    out: dict[str, Any] = {
        "keypoints": _encode_array(person.keypoints),
        "scores": _encode_array(person.scores),
        "bbox": list(person.bbox) if person.bbox is not None else None,
        "bbox_score": person.bbox_score,
        "track_id": person.track_id,
        "space_2d": person.space_2d.value,
        "keypoints_3d": _encode_array(person.keypoints_3d) if person.keypoints_3d is not None else None,
        "space_3d": person.space_3d.value if person.space_3d is not None else None,
    }
    return out


def _person_from_dict(payload: dict[str, Any]) -> PersonPose:
    kp3d = payload.get("keypoints_3d")
    space3d = payload.get("space_3d")
    return PersonPose(
        keypoints=_decode_array(payload["keypoints"], (len(payload["keypoints"]), 2)),
        scores=_decode_array(payload["scores"], (len(payload["scores"]),)),
        bbox=tuple(payload["bbox"]) if payload.get("bbox") else None,
        bbox_score=payload.get("bbox_score"),
        track_id=payload.get("track_id"),
        keypoints_3d=_decode_array(kp3d, (len(kp3d), 3)) if kp3d is not None else None,
        space_2d=Space(payload.get("space_2d", Space.IMAGE_PX.value)),
        space_3d=Space(space3d) if space3d is not None else None,
    )


def to_json(result: PoseResult) -> str:
    payload = {
        "schema_version": result.schema_version,
        "source": result.source,
        "model": result.model,
        "topology": {
            "name": result.topology.name,
            "keypoint_names": list(result.topology.keypoint_names),
            "edges": [list(e) for e in result.topology.edges],
        },
        "camera": (
            {
                "fx": result.camera.fx, "fy": result.camera.fy,
                "cx": result.camera.cx, "cy": result.camera.cy,
                "source": result.camera.source,
            }
            if result.camera is not None
            else None
        ),
        "frames": [
            {
                "frame_index": f.frame_index,
                "timestamp_s": f.timestamp_s,
                "image_size": list(f.image_size),
                "people": [_person_to_dict(p) for p in f.people],
            }
            for f in result.frames
        ],
    }
    return json.dumps(payload, indent=2, allow_nan=False)


def from_json(text: str) -> PoseResult:
    """Parse a PoseResult from JSON, using the installed topology definition if known.

    If the topology name is known in this environment, the installed definition wins
    (because every index in the rest of the application is interpreted against it).
    The file's own keypoint names and edges are then silently reinterpreted; if they
    differ substantially, a warning is logged.
    """
    payload = json.loads(text)
    name = payload["topology"]["name"]
    topology = _KNOWN_TOPOLOGIES.get(name)
    if topology is not None:
        file_edges = tuple(tuple(e) for e in payload["topology"]["edges"])
        if (tuple(payload["topology"]["keypoint_names"]) != topology.keypoint_names
                or file_edges != topology.edges):
            _log.warning(
                "%s in this file declares a different keypoint layout than the installed "
                "%s (%d vs %d edges). Using the installed definition; indices may not mean "
                "what the file intended.",
                name, name, len(file_edges), len(topology.edges),
            )
    else:
        topology = Topology(
            name=name,
            keypoint_names=tuple(payload["topology"]["keypoint_names"]),
            edges=tuple(tuple(e) for e in payload["topology"]["edges"]),
            groups={},
        )
    cam = payload.get("camera")
    return PoseResult(
        topology=topology,
        frames=tuple(
            PoseFrame(
                frame_index=f["frame_index"],
                timestamp_s=f["timestamp_s"],
                image_size=(f["image_size"][0], f["image_size"][1]),
                people=tuple(_person_from_dict(p) for p in f["people"]),
            )
            for f in payload["frames"]
        ),
        source=payload["source"],
        model=payload["model"],
        camera=Camera(**cam) if cam is not None else None,
        schema_version=payload["schema_version"],
    )


def to_csv(result: PoseResult) -> str:
    buf = io.StringIO()
    buf.write("frame,person,keypoint_index,keypoint_name,x,y,score\n")
    names = result.topology.keypoint_names
    for frame in result.frames:
        for person_index, person in enumerate(frame.people):
            for i in range(len(person)):
                x, y = person.keypoints[i]
                buf.write(
                    f"{frame.frame_index},{person_index},{i},{names[i]},"
                    f"{x:.3f},{y:.3f},{person.scores[i]:.4f}\n"
                )
    return buf.getvalue()
