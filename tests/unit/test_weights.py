from __future__ import annotations

import hashlib
import io
import os
import tempfile
import zipfile
from pathlib import Path

import pytest

from alchemypose.errors import WeightsError
from alchemypose.inference import weights


def test_every_preset_names_registered_models() -> None:
    for preset, (det_key, pose_key) in weights.PRESETS.items():
        assert det_key in weights.REGISTRY, f"{preset}: unknown detector {det_key}"
        assert pose_key in weights.REGISTRY, f"{preset}: unknown pose model {pose_key}"


def test_the_three_expected_presets_exist() -> None:
    assert set(weights.PRESETS) == {"lite", "balanced", "performance"}


def test_every_registry_entry_is_fully_specified() -> None:
    for key, spec in weights.REGISTRY.items():
        assert spec.url.startswith("https://"), key
        assert len(spec.sha256) == 64, f"{key}: sha256 must be recorded"
        assert spec.filename.endswith(".onnx"), key
        assert len(spec.input_size) == 2, key


def test_cache_dir_honours_the_environment_override(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ALCHEMYPOSE_MODEL_DIR", str(tmp_path))
    assert weights.cache_dir() == tmp_path


def test_ensure_returns_a_cached_file_without_downloading(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ALCHEMYPOSE_MODEL_DIR", str(tmp_path))
    spec = next(iter(weights.REGISTRY.values()))
    payload = b"pretend onnx"
    target = tmp_path / spec.filename
    target.write_bytes(payload)
    monkeypatch.setattr(
        weights, "REGISTRY", {**weights.REGISTRY, spec.key: _with_sha(spec, hashlib.sha256(payload).hexdigest())}
    )

    def explode(*_: object, **__: object) -> None:
        raise AssertionError("must not download when a valid file is cached")

    monkeypatch.setattr(weights, "_download", explode)
    assert weights.ensure(spec.key) == target


def test_ensure_redownloads_once_when_the_cached_file_is_corrupt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ALCHEMYPOSE_MODEL_DIR", str(tmp_path))
    spec = next(iter(weights.REGISTRY.values()))
    good = b"good bytes"
    digest = hashlib.sha256(good).hexdigest()
    monkeypatch.setattr(weights, "REGISTRY", {**weights.REGISTRY, spec.key: _with_sha(spec, digest)})
    (tmp_path / spec.filename).write_bytes(b"corrupt")

    calls: list[int] = []

    def fake_download(url: str, target: Path, progress: object = None) -> None:
        calls.append(1)
        target.write_bytes(good)

    monkeypatch.setattr(weights, "_download", fake_download)
    assert weights.ensure(spec.key).read_bytes() == good
    assert len(calls) == 1


def test_ensure_raises_when_the_download_still_fails_verification(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ALCHEMYPOSE_MODEL_DIR", str(tmp_path))
    spec = next(iter(weights.REGISTRY.values()))
    monkeypatch.setattr(weights, "REGISTRY", {**weights.REGISTRY, spec.key: _with_sha(spec, "0" * 64)})
    monkeypatch.setattr(weights, "_download", lambda url, target, progress=None: target.write_bytes(b"nope"))
    with pytest.raises(WeightsError, match="checksum"):
        weights.ensure(spec.key)


def test_ensure_rejects_an_unknown_key() -> None:
    with pytest.raises(WeightsError, match="unknown model"):
        weights.ensure("no-such-model")


def _make_zip(tmp_path: Path, members: dict[str, bytes], name: str = "archive.zip") -> Path:
    """Build a zip in memory, then persist it so it can be passed as a Path."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for member_name, data in members.items():
            archive.writestr(member_name, data)
    archive_path = tmp_path / name
    archive_path.write_bytes(buffer.getvalue())
    return archive_path


def test_extract_onnx_regression_zip_slip_member_name_cannot_escape_destination(tmp_path: Path) -> None:
    """A malicious ../../evil.onnx entry must never be written outside dest's directory.

    This is the regression guard for the zip-slip class of bug: if
    ``_extract_onnx`` were ever changed to use ``ZipFile.extract``/``extractall``
    (which interprets a member's name as a filesystem path) instead of
    streaming the member's *content* through ``ZipFile.open``, this test would
    fail because the traversal target below would exist.
    """
    # dest lives two directories below tmp_path, so "../../evil.onnx" traverses
    # exactly back up to tmp_path if the member name is ever used as a path.
    dest_dir = tmp_path / "a" / "b"
    dest_dir.mkdir(parents=True)
    dest = dest_dir / "model.onnx"
    archive = _make_zip(tmp_path, {"../../evil.onnx": b"malicious payload"})

    weights._extract_onnx(archive, dest)

    assert dest.read_bytes() == b"malicious payload"
    traversal_target = tmp_path / "evil.onnx"
    assert not traversal_target.exists(), "member name must never be interpreted as a filesystem path"


def test_extract_onnx_raises_when_the_archive_has_no_onnx_member(tmp_path: Path) -> None:
    archive = _make_zip(tmp_path, {"readme.txt": b"not a model"})
    dest = tmp_path / "out.onnx"

    with pytest.raises(WeightsError):
        weights._extract_onnx(archive, dest)

    assert not dest.exists()


def test_extract_onnx_raises_when_the_archive_has_two_onnx_members(tmp_path: Path) -> None:
    archive = _make_zip(tmp_path, {"a.onnx": b"one", "b.onnx": b"two"})
    dest = tmp_path / "out.onnx"

    with pytest.raises(WeightsError):
        weights._extract_onnx(archive, dest)

    assert not dest.exists()


def test_extract_onnx_closes_the_temp_fd_when_archive_open_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Regression for the fd leak: if ``ZipFile.open`` raises (an unsupported
    compression method, a structurally faulty member) before the raw fd from
    ``mkstemp`` is wrapped by ``os.fdopen``, that fd must still end up closed
    rather than leaking for the life of the process.
    """
    archive = _make_zip(tmp_path, {"model.onnx": b"payload"})
    dest = tmp_path / "out.onnx"

    captured_fds: list[int] = []
    real_mkstemp = tempfile.mkstemp

    def spy_mkstemp(*args: object, **kwargs: object) -> tuple[int, str]:
        fd, name = real_mkstemp(*args, **kwargs)  # type: ignore[arg-type]
        captured_fds.append(fd)
        return fd, name

    monkeypatch.setattr(tempfile, "mkstemp", spy_mkstemp)

    def exploding_open(self: zipfile.ZipFile, *args: object, **kwargs: object) -> object:
        raise zipfile.BadZipFile("simulated unsupported compression method")

    monkeypatch.setattr(zipfile.ZipFile, "open", exploding_open)

    with pytest.raises(zipfile.BadZipFile):
        weights._extract_onnx(archive, dest)

    assert captured_fds, "mkstemp must have been called"
    with pytest.raises(OSError):
        os.fstat(captured_fds[0])  # a closed fd raises EBADF; a leaked one would not


def test_extract_onnx_writes_the_exact_bytes_of_a_well_formed_single_member_zip(tmp_path: Path) -> None:
    payload = b"totally a real onnx model, honest"
    archive = _make_zip(tmp_path, {"20230928/yolox_onnx/some_run/end2end.onnx": payload})
    dest = tmp_path / "out.onnx"

    weights._extract_onnx(archive, dest)

    assert dest.read_bytes() == payload


@pytest.mark.parametrize(
    ("num_bytes", "expected"),
    [
        (0, "unknown size"),
        (-1, "unknown size"),
        (512, "512 bytes"),
        (1_000, "1.0 kB"),
        (3_481_706, "3.5 MB"),
        (213_433_855, "213.4 MB"),
        (2_500_000_000, "2.5 GB"),
    ],
)
def test_human_size_reads_as_a_download_size(num_bytes: int, expected: str) -> None:
    """The sizes a person is shown, including the one that means "no idea".

    ``0`` is the ``Content-Length`` a server that sent no ``Content-Length``
    leaves behind, and it has to read as unknown rather than as an empty file —
    "Downloading rtmw_l_384x288 (0 bytes)" would be a confident lie about the
    exact thing the message exists to answer.
    """
    assert weights.human_size(num_bytes) == expected


class _FakeResponse:
    """Just enough of ``urlopen``'s return value to drive ``_download``."""

    def __init__(self, payload: bytes, content_length: str | None) -> None:
        self._body = io.BytesIO(payload)
        self.headers: dict[str, str] = {} if content_length is None else {"Content-Length": content_length}

    def read(self, size: int) -> bytes:
        return self._body.read(size)

    def __enter__(self) -> _FakeResponse:
        return self

    def __exit__(self, *exc: object) -> bool:
        return False


def _serve(monkeypatch: pytest.MonkeyPatch, payload: bytes, content_length: str | None) -> None:
    monkeypatch.setattr(
        weights.urllib.request,
        "urlopen",
        lambda url, *args, **kwargs: _FakeResponse(payload, content_length),
    )


def test_download_reports_the_size_before_it_transfers_a_single_byte(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The opening ``(0, total)`` call is the contract the CLI and the GUI rest on.

    Both of them announce *which* model is being fetched and *how big* it is
    off that one call, and the value of doing so is entirely in its timing: a
    caller that first heard about a 213 MB transfer after the transfer had
    finished would have been told nothing it could use. So this asserts the
    order — a zero-byte call carrying the total, ahead of every call that
    carries bytes — and not merely that the callback happened.
    """
    payload = b"o" * (weights._CHUNK * 2 + 7)
    _serve(monkeypatch, payload, str(len(payload)))
    seen: list[tuple[int, int]] = []

    weights._download(
        "https://example.invalid/model.zip", tmp_path / "model.onnx", lambda done, total: seen.append((done, total))
    )

    assert seen[0] == (0, len(payload)), "the first word must be the size, before any bytes"
    assert [done for done, _ in seen[1:]] == [weights._CHUNK, weights._CHUNK * 2, len(payload)]
    assert {total for _, total in seen} == {len(payload)}, "every call carries the same total"
    assert (tmp_path / "model.onnx").read_bytes() == payload


def test_download_reports_a_zero_total_when_the_server_omits_the_length(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No ``Content-Length`` means no percentage, and callers are told so in the total.

    They are told it *up front*, in the same opening call, rather than
    discovering it by dividing by zero somewhere in the middle of a bar.
    """
    payload = b"z" * 100
    _serve(monkeypatch, payload, None)
    seen: list[tuple[int, int]] = []

    weights._download(
        "https://example.invalid/model.zip", tmp_path / "model.onnx", lambda done, total: seen.append((done, total))
    )

    assert seen == [(0, 0), (100, 0)]


def test_download_survives_being_given_no_progress_callback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``progress`` stays optional: ``RTMWEngine`` still calls ``ensure`` without one."""
    payload = b"q" * 40
    _serve(monkeypatch, payload, str(len(payload)))

    weights._download("https://example.invalid/model.zip", tmp_path / "model.onnx")

    assert (tmp_path / "model.onnx").read_bytes() == payload


def test_ensure_never_calls_progress_for_a_file_it_did_not_download(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Silence on a warm cache is a property of ``ensure``, not of its callers.

    Every bit of the new download messaging is keyed off this: the CLI prints
    nothing and the window says nothing because neither is told anything, and
    neither has to check the cache first to find out whether to speak. If
    ``ensure`` ever started reporting a cache hit as a completed transfer, both
    would start narrating the fast, quiet, common run.
    """
    monkeypatch.setenv("ALCHEMYPOSE_MODEL_DIR", str(tmp_path))
    spec = next(iter(weights.REGISTRY.values()))
    payload = b"already here"
    (tmp_path / spec.filename).write_bytes(payload)
    monkeypatch.setattr(
        weights, "REGISTRY", {**weights.REGISTRY, spec.key: _with_sha(spec, hashlib.sha256(payload).hexdigest())}
    )
    seen: list[tuple[int, int]] = []

    weights.ensure(spec.key, progress=lambda done, total: seen.append((done, total)))

    assert seen == []


def _with_sha(spec: weights.ModelSpec, sha: str) -> weights.ModelSpec:
    from dataclasses import replace

    return replace(spec, sha256=sha)
