"""The README is the PyPI project page, so the facts a newcomer acts on are tested.

Prose saying "keep this current" is not a mechanism. These check the handful of
things someone reads before installing: how to install, how to launch, how much
it downloads, which model it runs and under what licence. Anything derivable
from the code is derived from the code, so a change there that the README does
not follow turns this red.
"""

import inspect
import re
from pathlib import Path

import pytest

from alchemypose.inference import weights
from alchemypose.inference.rtmw import RTMWEngine

README = (Path(__file__).resolve().parents[2] / "README.md").read_text(encoding="utf-8")


def _default_preset() -> str:
    default = inspect.signature(RTMWEngine.__init__).parameters["preset"].default
    assert isinstance(default, str)
    return default


def test_the_install_command_is_named() -> None:
    assert "pip install alchemypose" in README


def test_the_launch_command_is_the_bare_entry_point() -> None:
    """``alchemypose`` on a line of its own, inside a shell block, is what opens the window."""
    blocks = re.findall(r"```(?:bash|console)\n(.*?)```", README, flags=re.DOTALL)
    assert any(line.strip() == "alchemypose" for block in blocks for line in block.splitlines())


def test_the_models_table_marks_the_engines_actual_default() -> None:
    default = _default_preset()
    rows = [line for line in README.splitlines() if line.startswith(f"| `{default}`")]
    assert rows, f"no models-table row for the default preset {default!r}"
    assert "(default)" in rows[0]


def test_every_preset_has_a_row_naming_its_pose_model() -> None:
    labels = {
        "rtmw_m_256x192": "RTMW-m",
        "dwpose_l_384x288": "DWPose-l",
        "rtmw_l_384x288": "RTMW-l",
        "rtmw3d_x_384x288": "RTMW3D-x",
    }
    for preset, (_det, pose) in weights.PRESETS.items():
        rows = [line for line in README.splitlines() if line.startswith(f"| `{preset}`")]
        assert rows, f"no models-table row for preset {preset!r}"
        assert labels[pose] in rows[0], f"{preset!r} row does not name {labels[pose]}"


def test_the_first_run_download_size_is_stated() -> None:
    assert re.search(r"\b\d+ MB of ONNX weights\b", README)


def test_the_stated_download_size_matches_the_files_when_they_are_cached() -> None:
    """Read in binary megabytes, the way the README says its sizes are."""
    _det, _pose = weights.PRESETS[_default_preset()]
    paths = [weights.cache_dir() / weights.REGISTRY[key].filename for key in (_det, _pose)]
    if not all(path.exists() for path in paths):
        pytest.skip("weights not cached; the stated size is only checkable against real files")
    measured = round(sum(path.stat().st_size for path in paths) / 2**20)
    stated = int(re.search(r"\b(\d+) MB of ONNX weights\b", README).group(1))  # type: ignore[union-attr]
    assert stated == measured


def test_images_use_absolute_urls_so_they_render_on_pypi() -> None:
    """The README is the PyPI page, and PyPI cannot resolve a path relative to the repository."""
    images = [
        target for target in re.findall(r"!\[[^\]]*\]\(([^)]+)\)", README)
        if not target.startswith("https://img.shields.io/")           # the badges are not ours to host
    ]
    assert images, "the README should show the window"
    for target in images:
        assert target.startswith("https://raw.githubusercontent.com/kouya-marino/AlchemyPose/main/"), target
        assert (Path(__file__).resolve().parents[2] / target.split("/main/", 1)[1]).exists(), target


def test_the_model_family_and_its_licence_are_named() -> None:
    assert "RTMW" in README
    assert "Apache-2.0" in README
    assert "MMPose" in README
