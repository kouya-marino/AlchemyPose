# src/alchemypose/inference/weights.py
"""Model files: where they come from, where they live, and proving they arrived intact.

Weights are never vendored in the wheel. The package stays tiny and the user
pays the download once, which is how AlchemyFace handles YuNet and SFace.

Every upstream RTMPose/RTMW/YOLOX release ships as a zip bundling the ``.onnx``
file alongside a deploy config, a pipeline description and preview JPEGs
(there is no direct ``.onnx`` URL for any of these on either the OpenMMLab
host or its Hugging Face mirror). ``_download`` therefore extracts the single
``.onnx`` member from the archive it fetches, so ``ensure()`` always hands
back a bare ``.onnx`` file and ``ModelSpec.sha256`` is the digest of that
extracted file, not of the zip.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import tempfile
import urllib.error
import urllib.request
import zipfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from platformdirs import user_cache_dir

from alchemypose.errors import WeightsError

_CHUNK = 1 << 16


@dataclass(frozen=True)
class ModelSpec:
    """One downloadable ONNX file."""

    key: str
    url: str
    sha256: str
    filename: str
    input_size: tuple[int, int]  # (width, height) the model expects


#: Every URL below was fetched and every digest measured by hand (see
#: ``tests/unit/test_weights.py`` and the task report) — none of this is
#: copied from a third party's manifest.
REGISTRY: dict[str, ModelSpec] = {
    "yolox_nano": ModelSpec(
        key="yolox_nano",
        url=(
            "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/"
            "yolox_nano_8xb8-300e_humanart-40f6f0d0.zip"
        ),
        sha256="1450966de24902b18aada1a78913d7efd8fc8dcd51bd4d0d5591476bd4a38821",
        filename="yolox_nano_8xb8-300e_humanart-40f6f0d0.onnx",
        input_size=(416, 416),
    ),
    "yolox_tiny": ModelSpec(
        key="yolox_tiny",
        url=(
            "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/"
            "yolox_tiny_8xb8-300e_humanart-6f3252f9.zip"
        ),
        sha256="ceb11c07298f95c50d7c5abeb906d03340c85f23aa79e3e66966e7fb6c307250",
        filename="yolox_tiny_8xb8-300e_humanart-6f3252f9.onnx",
        input_size=(416, 416),
    ),
    "yolox_s": ModelSpec(
        key="yolox_s",
        url=(
            "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/"
            "yolox_s_8xb8-300e_humanart-3ef259a7.zip"
        ),
        sha256="332e09ea9696e3401049b6c5314851db020b6430f85da159e65f37034ab3aee8",
        filename="yolox_s_8xb8-300e_humanart-3ef259a7.onnx",
        input_size=(640, 640),
    ),
    "rtmw_m_256x192": ModelSpec(
        key="rtmw_m_256x192",
        url=(
            "https://download.openmmlab.com/mmpose/v1/projects/rtmw/onnx_sdk/"
            "rtmw-dw-m-s_simcc-cocktail14_270e-256x192_20231122.zip"
        ),
        sha256="936fca9632e6f9fa295bdc9090b4b936a1454470f0b09ef802d4790b9bbb563c",
        filename="rtmw-dw-m-s_simcc-cocktail14_270e-256x192_20231122.onnx",
        input_size=(192, 256),
    ),
    "dwpose_l_384x288": ModelSpec(
        key="dwpose_l_384x288",
        url=(
            "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/"
            "rtmpose-l_simcc-ucoco_dw-ucoco_270e-384x288-2438fd99_20230728.zip"
        ),
        sha256="8cfecfc2226d8e14b510c2fd28442226c518b4d60690535753187877411d4005",
        filename="rtmpose-l_simcc-ucoco_dw-ucoco_270e-384x288-2438fd99_20230728.onnx",
        input_size=(288, 384),
    ),
    "rtmw_l_384x288": ModelSpec(
        key="rtmw_l_384x288",
        url=(
            "https://download.openmmlab.com/mmpose/v1/projects/rtmw/onnx_sdk/"
            "rtmw-dw-x-l_simcc-cocktail14_270e-384x288_20231122.zip"
        ),
        sha256="bd033156e5104c4f5d2edfe0453e02661e30a2f3da453ec93c8764d561b83054",
        filename="rtmw-dw-x-l_simcc-cocktail14_270e-384x288_20231122.onnx",
        input_size=(288, 384),
    ),
}

#: preset -> (detector key, pose key)
PRESETS: dict[str, tuple[str, str]] = {
    "lite": ("yolox_nano", "rtmw_m_256x192"),
    "balanced": ("yolox_tiny", "dwpose_l_384x288"),
    "performance": ("yolox_s", "rtmw_l_384x288"),
}


def cache_dir() -> Path:
    """Where model files live. ``ALCHEMYPOSE_MODEL_DIR`` overrides the default."""
    override = os.environ.get("ALCHEMYPOSE_MODEL_DIR")
    path = Path(override) if override else Path(user_cache_dir("alchemypose"))
    path.mkdir(parents=True, exist_ok=True)
    return path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(_CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


def _extract_onnx(archive_path: Path, target: Path) -> None:
    """Pull the single ``.onnx`` member out of a downloaded zip and move it into place."""
    with zipfile.ZipFile(archive_path) as archive:
        onnx_members = [name for name in archive.namelist() if name.endswith(".onnx")]
        if len(onnx_members) != 1:
            raise WeightsError(
                f"expected exactly one .onnx file in {archive_path.name}, found {len(onnx_members)}: {onnx_members}"
            )
        with (
            archive.open(onnx_members[0]) as member,
            tempfile.NamedTemporaryFile(delete=False, dir=target.parent) as tmp,
        ):
            shutil.copyfileobj(member, tmp)
            temp_path = Path(tmp.name)
    shutil.move(str(temp_path), target)


def _download(url: str, target: Path, progress: Callable[[int, int], None] | None = None) -> None:
    """Fetch to a temporary file first, so an interrupted download never looks cached.

    Upstream URLs serve a zip bundling the ``.onnx`` file with other artefacts;
    when that happens the archive is downloaded to a temporary file and the
    single ``.onnx`` member is extracted to ``target``, so ``target`` is
    always a bare ONNX file regardless of how the URL served it.
    """
    try:
        with urllib.request.urlopen(url) as response:  # noqa: S310 - the URLs are our own constants
            total = int(response.headers.get("Content-Length", 0))
            done = 0
            with tempfile.NamedTemporaryFile(delete=False, dir=target.parent) as tmp:
                while chunk := response.read(_CHUNK):
                    tmp.write(chunk)
                    done += len(chunk)
                    if progress is not None:
                        progress(done, total)
                temp_path = Path(tmp.name)
        if zipfile.is_zipfile(temp_path):
            try:
                _extract_onnx(temp_path, target)
            finally:
                temp_path.unlink(missing_ok=True)
        else:
            shutil.move(str(temp_path), target)
    except (urllib.error.URLError, OSError, zipfile.BadZipFile) as exc:
        raise WeightsError(f"could not download {url}: {exc}") from exc


def ensure(key: str, *, progress: Callable[[int, int], None] | None = None) -> Path:
    """Return a verified local path to the model, downloading it at most once."""
    spec = REGISTRY.get(key)
    if spec is None:
        raise WeightsError(f"unknown model {key!r}; known: {sorted(REGISTRY)}")

    target = cache_dir() / spec.filename
    if target.exists() and _sha256(target) == spec.sha256:
        return target

    if target.exists():
        target.unlink()
    _download(spec.url, target, progress)

    actual = _sha256(target)
    if actual != spec.sha256:
        target.unlink(missing_ok=True)
        raise WeightsError(
            f"checksum mismatch for {key}: expected {spec.sha256}, got {actual}. "
            f"The download may be corrupt or the upstream file changed."
        )
    return target
