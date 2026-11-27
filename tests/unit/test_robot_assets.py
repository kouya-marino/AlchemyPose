"""The G1's files: what is fetched, how it is verified, and how its meshes are read."""

import hashlib
import struct
from pathlib import Path

import numpy as np
import pytest

from alchemypose.errors import WeightsError
from alchemypose.robot import assets
from alchemypose.robot.stl import load_stl
from alchemypose.robot.urdf import load_urdf

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "unitree_g1"


def test_the_registry_is_pinned_and_complete() -> None:
    assert len(assets.COMMIT) == 40
    assert assets.COMMIT in assets.BASE_URL
    assert {assets.URDF, assets.MJCF} <= set(assets.FILES)
    assert sum(name.startswith("meshes/") for name in assets.FILES) == 35
    assert all(len(digest) == 64 for digest in assets.FILES.values())


def test_the_model_matches_every_mesh_it_needs() -> None:
    robot = load_urdf(FIXTURES / "g1_29dof_rev_1_0.urdf")
    needed = {v.mesh for visuals in robot.visuals.values() for v in visuals}
    assert needed == {name for name in assets.FILES if name.startswith("meshes/")}


def test_fetching_verifies_every_file_and_skips_what_is_already_there(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ALCHEMYPOSE_MODEL_DIR", str(tmp_path))
    contents = {name: f"contents of {name}".encode() for name in ("a.urdf", "meshes/b.STL")}
    monkeypatch.setattr(assets, "FILES", {n: hashlib.sha256(c).hexdigest() for n, c in contents.items()})
    fetched: list[str] = []

    def fake_download(url: str, dest: Path, progress: object = None) -> None:
        name = url.removeprefix(assets.BASE_URL)
        fetched.append(name)
        dest.write_bytes(contents[name])

    monkeypatch.setattr(assets.weights, "_download", fake_download)
    root = assets.ensure_g1()
    assert sorted(fetched) == sorted(contents)
    assert (root / "meshes/b.STL").read_bytes() == contents["meshes/b.STL"]
    assets.ensure_g1()
    assert len(fetched) == 2, "a verified file is not fetched again"


def test_a_file_with_the_wrong_digest_is_refused_and_not_kept(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ALCHEMYPOSE_MODEL_DIR", str(tmp_path))
    monkeypatch.setattr(assets, "FILES", {"a.urdf": "0" * 64})
    monkeypatch.setattr(assets.weights, "_download", lambda url, dest, progress=None: dest.write_bytes(b"tampered"))
    with pytest.raises(WeightsError, match="checksum"):
        assets.ensure_g1()
    assert not (assets.g1_dir() / "a.urdf").exists()


def _binary_stl(triangles: list[list[tuple[float, float, float]]]) -> bytes:
    body = b"".join(struct.pack("<12fH", 0, 0, 1, *[c for v in tri for c in v], 0) for tri in triangles)
    return b"\0" * 80 + struct.pack("<I", len(triangles)) + body


def test_a_binary_stl_is_read_exactly(tmp_path: Path) -> None:
    path = tmp_path / "t.STL"
    path.write_bytes(_binary_stl([[(0, 0, 0), (1, 0, 0), (0, 1, 0)], [(0, 0, 1), (1, 0, 1), (0, 1, 1)]]))
    triangles, normals = load_stl(path)
    assert triangles.shape == (2, 3, 3)
    assert triangles[1, 2].tolist() == [0, 1, 1]
    assert normals[0].tolist() == [0, 0, 1]


def test_an_ascii_stl_is_refused_not_half_read(tmp_path: Path) -> None:
    path = tmp_path / "a.stl"
    path.write_text("solid x\n" + "facet normal 0 0 1\n" * 40 + "endsolid x\n")
    with pytest.raises(ValueError, match="binary STL"):
        load_stl(path)


def test_the_real_meshes_parse_when_they_are_cached() -> None:
    root = assets.g1_dir()
    if assets.missing():
        pytest.skip("G1 meshes not cached")
    triangles, _ = load_stl(root / "meshes/pelvis.STL")
    assert triangles.shape[1:] == (3, 3)
    assert np.abs(triangles).max() < 0.5, "metres, and the pelvis is smaller than a metre"
