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

Nothing here ever writes unverified bytes to the canonical cache path: every
write — the raw download, the zip extraction, and the final publish in
``ensure`` — goes to a temp file created up front (so its path is always known
for cleanup) and is only ever moved into its destination once it is known to
be complete and, for the canonical path, checksum-verified.
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
    #: Raw SimCC peak height that means "fully confident" — pose models only,
    #: ``None`` for detectors. Measured, not derived from configs; see
    #: ``inference.rtmw`` for why it differs per model and how it was found.
    peak_scale: float | None = None


#: Every URL below was fetched and every digest measured by hand — none of it
#: copied from a third party's manifest. Each ``sha256`` is the hash of the
#: ``.onnx`` member extracted from the zip at ``url``, not of the zip itself, so
#: re-deriving one means downloading the archive and hashing exactly what
#: ``_download`` pulls out of it. ``tests/unit/test_weights.py`` guards the
#: shape of every entry below.
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
        peak_scale=1.0,
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
        peak_scale=1.0,
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
        peak_scale=10.0,
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


def human_size(num_bytes: int) -> str:
    """A download size a person can read: ``58.1 MB``, or ``unknown size`` for 0.

    Decimal units, because that is what the server counts in, what the browser
    would have shown and what the file listing on this platform says. Quoting a
    "203 MiB" for a file every other tool calls 213 MB buys precision nobody
    asked for at the price of a number that appears to disagree with itself.

    ``0`` means the server sent no ``Content-Length``: unknown, not empty.
    """
    if num_bytes <= 0:
        return "unknown size"
    for unit, scale in (("GB", 1_000_000_000), ("MB", 1_000_000), ("kB", 1_000)):
        if num_bytes >= scale:
            return f"{num_bytes / scale:.1f} {unit}"
    return f"{num_bytes} bytes"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(_CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


def _extract_onnx(archive_path: Path, dest: Path) -> None:
    """Stream the single ``.onnx`` member of ``archive_path`` into ``dest``.

    The member is copied by *content* through ``ZipFile.open`` into a freshly
    created temp file, never through ``ZipFile.extract``/``extractall`` — so an
    attacker-controlled entry name (e.g. ``"../../evil.onnx"``, a "zip slip"
    payload) is never interpreted as a filesystem path and can never write
    outside ``dest``'s directory. The temp file's path is captured immediately
    on creation and removed on any failure, so a bad CRC or a disk-full error
    partway through the copy cannot leave a stray partial file behind.
    """
    with zipfile.ZipFile(archive_path) as archive:
        onnx_members = [name for name in archive.namelist() if name.endswith(".onnx")]
        if len(onnx_members) != 1:
            raise WeightsError(
                f"expected exactly one .onnx file in {archive_path.name}, found {len(onnx_members)}: {onnx_members}"
            )
        fd, tmp_name = tempfile.mkstemp(dir=dest.parent)
        temp_path = Path(tmp_name)
        try:
            # os.fdopen(fd, ...) must come first in this "with", not archive.open().
            # A compound "with a, b" evaluates left to right, so if archive.open()
            # raised first, the raw fd from mkstemp would never be wrapped or
            # closed and would leak for the life of the process (the temp file
            # itself is still unlinked below, but that doesn't close the fd).
            # Wrapping the fd first guarantees it is always closed on the way out.
            with os.fdopen(fd, "wb") as out, archive.open(onnx_members[0]) as member:
                shutil.copyfileobj(member, out)
        except BaseException:
            temp_path.unlink(missing_ok=True)
            raise
    shutil.move(str(temp_path), dest)


def _download(url: str, dest: Path, progress: Callable[[int, int], None] | None = None) -> None:
    """Fetch (and, for a zip, extract) ``url`` into ``dest``.

    ``dest`` is a fresh temp path chosen by the caller (``ensure``) — this
    function never writes to the canonical cache filename itself, and it never
    leaves a partial file behind: the raw download goes to its own temp file,
    whose path is captured the instant it is created (before any bytes are
    written) and removed on any failure — a dropped connection mid-download,
    or a zip that turns out to be malformed — so an interrupted transfer can
    never orphan a partially written multi-hundred-megabyte file in the cache
    directory.

    ``progress`` is called once with ``(0, total)`` the moment the response
    headers give a size — before a single byte is written — and then after
    every chunk with the running byte count. That opening call is the whole
    point of the callback for anything but a bar: it is the only moment at
    which a caller can say *which* model is being fetched and *how big* it is
    while the answer is still worth having. Announcing after the first chunk
    would be nearly as early, but "nearly" is not a contract, and a caller
    should not have to know the chunk size to work out whether it has been told
    yet. ``total`` is ``0`` when the server sends no ``Content-Length``, so a
    caller drawing a bar has to read that as "length unknown" rather than
    dividing by it.
    """
    try:
        with urllib.request.urlopen(url) as response:  # noqa: S310 - the URLs are our own constants
            total = int(response.headers.get("Content-Length", 0))
            if progress is not None:
                progress(0, total)
            fd, raw_name = tempfile.mkstemp(dir=dest.parent)
            raw_path = Path(raw_name)
            try:
                done = 0
                with os.fdopen(fd, "wb") as tmp:
                    while chunk := response.read(_CHUNK):
                        tmp.write(chunk)
                        done += len(chunk)
                        if progress is not None:
                            progress(done, total)
            except BaseException:
                raw_path.unlink(missing_ok=True)
                raise
        if zipfile.is_zipfile(raw_path):
            try:
                _extract_onnx(raw_path, dest)
            finally:
                raw_path.unlink(missing_ok=True)
        else:
            shutil.move(str(raw_path), dest)
    except (urllib.error.URLError, OSError, zipfile.BadZipFile) as exc:
        raise WeightsError(f"could not download {url}: {exc}") from exc


def ensure(key: str, *, progress: Callable[[int, int], None] | None = None) -> Path:
    """Return a verified local path to the model, downloading it at most once.

    The canonical cache path only ever receives verified bytes: the download
    (and any zip extraction) lands in a temp file first, that temp file is
    hashed, and only a matching digest gets ``shutil.move``d onto ``target`` —
    so a concurrent reader of ``target`` can never observe a corrupt or
    partially written file, and a failed verification leaves ``target``
    untouched (and cleans up its own temp file) rather than looping.

    ``progress`` is forwarded to ``_download``, and so is called only once this
    function has decided to fetch something: the verified-cache-hit return
    above happens without a single callback. That is what lets a caller be
    completely silent on a warm cache without first having to ask whether the
    file was there — it says nothing because it was told nothing.
    """
    spec = REGISTRY.get(key)
    if spec is None:
        raise WeightsError(f"unknown model {key!r}; known: {sorted(REGISTRY)}")

    cache = cache_dir()
    target = cache / spec.filename
    if target.exists() and _sha256(target) == spec.sha256:
        return target

    if target.exists():
        target.unlink()

    fd, temp_name = tempfile.mkstemp(dir=cache, suffix=".part")
    os.close(fd)
    temp_path = Path(temp_name)
    try:
        _download(spec.url, temp_path, progress)
        actual = _sha256(temp_path)
        if actual != spec.sha256:
            raise WeightsError(
                f"checksum mismatch for {key}: expected {spec.sha256}, got {actual}. "
                f"The download may be corrupt or the upstream file changed."
            )
        shutil.move(str(temp_path), target)
    finally:
        temp_path.unlink(missing_ok=True)
    return target
