"""Binary STL, read with numpy: what the robot's meshes are made of.

Every mesh in the G1 description is binary STL — an 80-byte header, a
triangle count, then 50 bytes per triangle (normal, three vertices, a spare
attribute word). That fixed layout reads as one structured ``numpy`` array,
with no parsing loop. An ASCII STL is refused rather than half-read.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from numpy.typing import NDArray

F32 = NDArray[np.float32]

_TRIANGLE = np.dtype([("normal", "<f4", 3), ("vertices", "<f4", (3, 3)), ("attribute", "<u2")])


def load_stl(path: Path) -> tuple[F32, F32]:
    """``(triangles (N, 3, 3), normals (N, 3))`` in the file's units (metres for the G1)."""
    raw = Path(path).read_bytes()
    if len(raw) < 84:
        raise ValueError(f"{path} is too short to be a binary STL")
    count = int(np.frombuffer(raw, dtype="<u4", count=1, offset=80)[0])
    if 84 + 50 * count != len(raw):
        raise ValueError(f"{path} is not a binary STL of {count} triangles (ASCII STL is not supported)")
    triangles = np.frombuffer(raw, dtype=_TRIANGLE, count=count, offset=84)
    return triangles["vertices"].astype(np.float32), triangles["normal"].astype(np.float32)
