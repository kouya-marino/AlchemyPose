import hashlib
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


def _with_sha(spec: weights.ModelSpec, sha: str) -> weights.ModelSpec:
    from dataclasses import replace

    return replace(spec, sha256=sha)
