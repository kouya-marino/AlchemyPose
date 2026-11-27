"""The Unitree G1 model files, fetched once at runtime and verified like the weights.

Source: ``unitree_ros`` (BSD-3-Clause, © Unitree Robotics), ``robots/g1_description``,
pinned to one commit so the geometry can never change underneath a result. Only
the 29-DoF "rev 1.0" model and the 35 meshes it references are fetched — 19 MB,
not the 115 MB of every variant in that folder. Nothing is vendored in the
repository or the wheel, and nothing is taken from any local copy of the
description: what is drawn is exactly what the pinned upstream commit says.

Why ``g1_29dof_rev_1_0``: the description's README marks the original
``g1_29dof.urdf`` deprecated in favour of it. Robots with a newer
``mode_machine`` (the ``mode_11``–``mode_16`` files) differ in wrist offsets and
motors but share joint names, order and limits, so motion retargeted here
carries over; the geometry is what changes.

The MJCF twin (``g1_29dof_rev_1_0.xml``) from the same commit is fetched too: it
is what MuJoCo loads, and what the robot-motion export is checked against.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import tempfile
from collections.abc import Callable
from pathlib import Path

from alchemypose.errors import WeightsError
from alchemypose.inference import weights

COMMIT = "24d6307f85986fd489807fb54e0995a798c36b37"
BASE_URL = f"https://raw.githubusercontent.com/unitreerobotics/unitree_ros/{COMMIT}/robots/g1_description/"
URDF = "g1_29dof_rev_1_0.urdf"
MJCF = "g1_29dof_rev_1_0.xml"
LICENSE = "BSD-3-Clause, Copyright Unitree Robotics (unitreerobotics/unitree_ros)"

#: Every file, relative to the description folder, and the SHA-256 measured on it.
FILES: dict[str, str] = {
    "g1_29dof_rev_1_0.urdf": "c0ae739c640c3e2c00d1bdd8810b5d6e59601487bd1a3995859f9543269ee5c8",
    "g1_29dof_rev_1_0.xml": "165fa7a5275745e45fa25e1bc78010c1055d81bb81c013b3c7bc8b9347a3b9c3",
    "meshes/head_link.STL": "005fb67fbd3eff94aa8bf4a6e83238174e9f91b6721f7111594322f223724411",
    "meshes/left_ankle_pitch_link.STL": "d49e3abc6f5b12e532062cd575b87b5ef40cd2a3fc18f54a1ca5bba4f773d51d",
    "meshes/left_ankle_roll_link.STL": "c4092af943141d4d9f74232f3cfa345afc6565f46a077793b8ae0e68b39dc33f",
    "meshes/left_elbow_link.STL": "fa752198accd104d5c4c3a01382e45165b944fbbc5acce085059223324e5bed3",
    "meshes/left_hip_pitch_link.STL": "4725168105ee768ee31638ef22b53f6be2d7641bfd7cfefe803488d884776fa4",
    "meshes/left_hip_roll_link.STL": "91f25922ee8a7c3152790051bebad17b4d9cd243569c38fe340285ff93a97acf",
    "meshes/left_hip_yaw_link.STL": "a16d88aa6ddac8083aa7ad55ed317bea44b1fa003d314fba88965b7ed0f3b55b",
    "meshes/left_knee_link.STL": "8d92b9e3d3a636761150bb8025e32514c4602b91c7028d523ee42b3e632de477",
    "meshes/left_rubber_hand.STL": "cff2221a690fa69303f61fce68f2d155c1517b52efb6ca9262dd56e0bc6e70fe",
    "meshes/left_shoulder_pitch_link.STL": "f0d1cfd02fcf0d42f95e678eeca33da3afbcc366ffba5c052847773ec4f31d52",
    "meshes/left_shoulder_roll_link.STL": "fb9df21687773522598dc384f1a2945c7519f11cbc8bd372a49170316d6eee88",
    "meshes/left_shoulder_yaw_link.STL": "1aa97e9748e924336567992181f78c7cd0652fd52a4afcca3df6b2ef6f9e712e",
    "meshes/left_wrist_pitch_link.STL": "b251d8e05047f695d0f536cd78c11973cfa4e78d08cfe82759336cc3471de3a9",
    "meshes/left_wrist_roll_link.STL": "edc387c9a0ba8c2237e9b296d32531426fabeb6f53e58df45c76106bca74148c",
    "meshes/left_wrist_yaw_link.STL": "83f8fb3a726bf9613d65dd14f0f447cb918c3c95b3938042a0c9c09749267d3b",
    "meshes/logo_link.STL": "8571a0a19bc4916fa55f91449f51d5fdefd751000054865a842449429d5f155b",
    "meshes/pelvis.STL": "5ba6bbc888e630550140d3c26763f10206da8c8bd30ed886b8ede41c61f57a31",
    "meshes/pelvis_contour_link.STL": "5cc5c2c7a312329e3feeb2b03d3fc09fc29705bd01864f6767e51be959662420",
    "meshes/right_ankle_pitch_link.STL": "15be426539ec1be70246d4d82a168806db64a41301af8b35c197a33348c787a9",
    "meshes/right_ankle_roll_link.STL": "4b66222ea56653e627711b56d0a8949b4920da5df091da0ceb343f54e884e3a5",
    "meshes/right_elbow_link.STL": "1be925d7aa268bb8fddf5362b9173066890c7d32092c05638608126e59d1e2ab",
    "meshes/right_hip_pitch_link.STL": "e4f3c99d7f4a7d34eadbef9461fc66e3486cb5442db1ec50c86317d459f1a9c6",
    "meshes/right_hip_roll_link.STL": "4c254ef66a356f492947f360dd931965477b631e3fcc841f91ccc46d945d54f6",
    "meshes/right_hip_yaw_link.STL": "e479c2936ca2057e9eb2f7dff6c189b7419d7b8484dea0b298cbb36a2a6aa668",
    "meshes/right_knee_link.STL": "63c4008449c9bbe701a6e2b557b7a252e90cf3a5abcf54cee46862b9a69f8ec8",
    "meshes/right_rubber_hand.STL": "99533b778bca6246144fa511bb9d4e555e075c641f2a0251e04372869cd99d67",
    "meshes/right_shoulder_pitch_link.STL": "24cdb387e0128dfe602770a81c56cdce3a0181d34d039a11d1aaf8819b7b8c02",
    "meshes/right_shoulder_roll_link.STL": "962b97c48f9ce9e8399f45dd9522e0865d19aa9fd299406b2d475a8fc4a53e81",
    "meshes/right_shoulder_yaw_link.STL": "a0b76489271da0c72461a344c9ffb0f0c6e64f019ea5014c1624886c442a2fe5",
    "meshes/right_wrist_pitch_link.STL": "d22f8f3b3127f15a63e5be1ee273cd5075786c3142f1c3d9f76cbf43d2a26477",
    "meshes/right_wrist_roll_link.STL": "a7ee9212ff5b94d6cb7f52bb1bbf3f352194d5b598acff74f4c77d340c5b344f",
    "meshes/right_wrist_yaw_link.STL": "bc9dece2d12509707e0057ba2e48df8f3d56db0c79410212963a25e8a50f61a6",
    "meshes/torso_link_rev_1_0.STL": "11ddb46f2098efbbd8816b1d65893632d8e78be936376c7cdcd6771899ccc723",
    "meshes/waist_roll_link_rev_1_0.STL": "b67c347a05abc3e8ddae600d98a082bee273bb39f8e651647708b0a7140a8a97",
    "meshes/waist_yaw_link_rev_1_0.STL": "ec6db442b11f25eed898b5add07940c85d804f300de24dcbd264ccd8be7d554c",
}


def g1_dir() -> Path:
    path = weights.cache_dir() / "unitree_g1" / COMMIT[:12]
    (path / "meshes").mkdir(parents=True, exist_ok=True)
    return path


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def missing() -> list[str]:
    """Files not yet on disk and verified."""
    root = g1_dir()
    return [name for name, digest in FILES.items() if not (root / name).exists() or _sha256(root / name) != digest]


def ensure_g1(progress: Callable[[int, int], None] | None = None) -> Path:
    """The folder holding the verified URDF, MJCF and meshes, fetching what is missing.

    ``progress(done, total)`` counts files. Each file goes to a temp path and is
    moved into place only once its digest matches, so a failed fetch never
    leaves a corrupt file to be trusted later.
    """
    root = g1_dir()
    todo = missing()
    for done, name in enumerate(todo):
        if progress is not None:
            progress(done, len(todo))
        target = root / name
        fd, temp_name = tempfile.mkstemp(dir=target.parent, suffix=".part")
        os.close(fd)
        temp = Path(temp_name)
        try:
            weights._download(BASE_URL + name, temp)
            actual = _sha256(temp)
            if actual != FILES[name]:
                raise WeightsError(f"checksum mismatch for G1 file {name}: expected {FILES[name]}, got {actual}")
            shutil.move(str(temp), target)
        finally:
            temp.unlink(missing_ok=True)
    if progress is not None and todo:
        progress(len(todo), len(todo))
    return root
