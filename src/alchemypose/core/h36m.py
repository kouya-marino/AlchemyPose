"""COCO-WholeBody -> Human3.6M 17-joint conversion.

Nothing in v1 calls this. It exists because every monocular 2D->3D lifting model
(MotionBERT, MotionAGFormer, VideoPose3D) consumes H36M-17, and COCO has no
pelvis, thorax, spine or head-top joint — they must be synthesised. Writing the
mapping now pins the conventions while they are still cheap to change.

Convention: input is source-image pixels, origin top-left, y down. Output is the
same. Normalisation into a lifter's expected range is that lifter's business.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from alchemypose.core.schema import PersonPose
from alchemypose.core.topology import KEYPOINT_NAMES

F32 = NDArray[np.float32]

H36M_NAMES: tuple[str, ...] = (
    "pelvis", "right_hip", "right_knee", "right_ankle",
    "left_hip", "left_knee", "left_ankle",
    "spine", "thorax", "neck", "head",
    "left_shoulder", "left_elbow", "left_wrist",
    "right_shoulder", "right_elbow", "right_wrist",
)

#: H36M joints that are a straight copy of a COCO joint.
_DIRECT: dict[str, str] = {
    "right_hip": "right_hip", "right_knee": "right_knee", "right_ankle": "right_ankle",
    "left_hip": "left_hip", "left_knee": "left_knee", "left_ankle": "left_ankle",
    "left_shoulder": "left_shoulder", "left_elbow": "left_elbow", "left_wrist": "left_wrist",
    "right_shoulder": "right_shoulder", "right_elbow": "right_elbow", "right_wrist": "right_wrist",
    "head": "nose",
}

_C: dict[str, int] = {
    name: KEYPOINT_NAMES.index(name)
    for name in set(_DIRECT.values()) | {"left_hip", "right_hip", "left_shoulder", "right_shoulder", "nose"}
}


def to_h36m17(person: PersonPose) -> tuple[F32, F32]:
    """Return ``(keypoints (17, 2), scores (17,))`` in H36M joint order.

    Synthesised joints take the *minimum* confidence of their parents, so a joint
    derived from one unreliable parent is itself marked unreliable.
    """
    kp = person.keypoints
    sc = person.scores

    out_kp = np.zeros((17, 2), dtype=np.float32)
    out_sc = np.zeros((17,), dtype=np.float32)

    for h_name, c_name in _DIRECT.items():
        i = H36M_NAMES.index(h_name)
        j = _C[c_name]
        out_kp[i] = kp[j]
        out_sc[i] = sc[j]

    def mid(a: str, b: str) -> tuple[F32, np.float32]:
        ia, ib = _C[a], _C[b]
        return ((kp[ia] + kp[ib]) / 2.0).astype(np.float32), np.minimum(sc[ia], sc[ib])

    pelvis_xy, pelvis_s = mid("left_hip", "right_hip")
    thorax_xy, thorax_s = mid("left_shoulder", "right_shoulder")

    out_kp[H36M_NAMES.index("pelvis")], out_sc[H36M_NAMES.index("pelvis")] = pelvis_xy, pelvis_s
    out_kp[H36M_NAMES.index("thorax")], out_sc[H36M_NAMES.index("thorax")] = thorax_xy, thorax_s

    spine_xy = ((pelvis_xy + thorax_xy) / 2.0).astype(np.float32)
    out_kp[H36M_NAMES.index("spine")] = spine_xy
    out_sc[H36M_NAMES.index("spine")] = min(pelvis_s, thorax_s)

    # H36M's "neck" sits between thorax and head; COCO offers nothing closer.
    nose_i = _C["nose"]
    neck_xy = ((thorax_xy + kp[nose_i]) / 2.0).astype(np.float32)
    out_kp[H36M_NAMES.index("neck")] = neck_xy
    out_sc[H36M_NAMES.index("neck")] = min(thorax_s, sc[nose_i])

    return out_kp, out_sc
