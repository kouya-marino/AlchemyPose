from pathlib import Path
from typing import Any

import numpy as np
import pytest

from alchemypose.inference import session as session_module
from alchemypose.inference.session import COREML, CPU, CUDA, chosen_providers, make_session, probe_inputs


class _FakeInput:
    """One entry of ``InferenceSession.get_inputs()``: a name, a shape, an ONNX type name."""

    def __init__(self, name: str, shape: list[Any], type_: str = "tensor(float)") -> None:
        self.name = name
        self.shape = shape
        self.type = type_


class _FakeSession:
    """Stands in for a real session so the provider logic can be tested without ONNX files."""

    def __init__(
        self,
        providers: list[str],
        inputs: list[_FakeInput] | None = None,
        run_error: Exception | None = None,
    ) -> None:
        self.providers = providers
        self.inputs = inputs if inputs is not None else [_FakeInput("input", [1, 3, 8, 8])]
        self.run_error = run_error
        self.runs: list[dict[str, Any]] = []

    def get_inputs(self) -> list[_FakeInput]:
        return self.inputs

    def get_providers(self) -> list[str]:
        return self.providers

    def run(self, _outputs: Any, feeds: dict[str, Any]) -> list[Any]:
        self.runs.append(feeds)
        if self.run_error is not None:
            raise self.run_error
        return [np.zeros((1, 1), dtype=np.float32)]


def _install(
    monkeypatch: pytest.MonkeyPatch,
    sessions: dict[str, _FakeSession],
    available: list[str] | None = None,
) -> list[list[str]]:
    """Patch onnxruntime so ``make_session`` builds ``sessions[provider0]``. Returns the calls made."""
    calls: list[list[str]] = []

    def fake_ctor(_path: str, providers: list[str]) -> _FakeSession:
        calls.append(providers)
        return sessions[providers[0]]

    monkeypatch.setattr(session_module.ort, "InferenceSession", fake_ctor)
    monkeypatch.setattr(
        session_module.ort, "get_available_providers", lambda: available or [COREML, CPU]
    )
    return calls


def test_coreml_is_requested_first_when_preferred_and_available() -> None:
    assert chosen_providers(True, [COREML, CPU]) == [COREML, CPU]


def test_cpu_only_when_coreml_is_not_wanted() -> None:
    assert chosen_providers(False, [COREML, CPU]) == [CPU]


def test_cpu_only_when_coreml_is_unavailable() -> None:
    assert chosen_providers(True, [CPU]) == [CPU]


def test_cpu_is_always_last_so_there_is_always_a_fallback() -> None:
    assert chosen_providers(True, [COREML, CPU])[-1] == CPU
    assert chosen_providers(False, [COREML, CPU])[-1] == CPU


def test_probe_inputs_allocates_zeros_at_the_declared_shape() -> None:
    feeds = probe_inputs(_FakeSession([CPU], [_FakeInput("input", [1, 3, 256, 192])]))
    assert feeds is not None
    assert feeds["input"].shape == (1, 3, 256, 192)
    assert feeds["input"].dtype == np.float32
    assert not feeds["input"].any()


def test_probe_inputs_substitutes_one_for_a_symbolic_dimension() -> None:
    feeds = probe_inputs(_FakeSession([CPU], [_FakeInput("input", ["batch", 3, 384, 288])]))
    assert feeds is not None
    assert feeds["input"].shape == (1, 3, 384, 288)


def test_probe_inputs_covers_every_input_and_honours_each_dtype() -> None:
    graph = _FakeSession(
        [CPU],
        [_FakeInput("pixels", [1, 3, 8, 8]), _FakeInput("ids", [1, 4], "tensor(int64)")],
    )
    feeds = probe_inputs(graph)
    assert feeds is not None
    assert set(feeds) == {"pixels", "ids"}
    assert feeds["ids"].dtype == np.int64


def test_probe_inputs_declines_a_dtype_it_cannot_allocate() -> None:
    graph = _FakeSession([CPU], [_FakeInput("text", [1], "tensor(string)")])
    assert probe_inputs(graph) is None


def test_probe_inputs_declines_a_dimension_that_is_not_positive() -> None:
    assert probe_inputs(_FakeSession([CPU], [_FakeInput("input", [1, 3, -1, 8])])) is None


def test_a_working_coreml_session_is_returned_and_proved_by_a_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    coreml = _FakeSession([COREML, CPU])
    calls = _install(monkeypatch, {COREML: coreml, CPU: _FakeSession([CPU])})

    assert make_session(Path("pose.onnx"), prefer_coreml=True) is coreml
    assert calls == [[COREML, CPU]]
    assert len(coreml.runs) == 1, "the session must be proved by an actual run before it is handed back"


def test_a_provider_that_loads_the_graph_but_cannot_run_it_is_demoted_to_cpu(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """YOLOX's failure mode: the session is created happily, then the first run() fails."""
    coreml = _FakeSession([COREML, CPU], run_error=RuntimeError("different ranks"))
    cpu = _FakeSession([CPU])
    calls = _install(monkeypatch, {COREML: coreml, CPU: cpu})

    assert make_session(Path("yolox.onnx"), prefer_coreml=True) is cpu
    assert calls == [[COREML, CPU], [CPU]]


def test_a_coreml_session_that_cannot_even_be_created_is_demoted_to_cpu(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cpu = _FakeSession([CPU])

    def fake_ctor(_path: str, providers: list[str]) -> _FakeSession:
        if providers[0] == COREML:
            raise RuntimeError("provider unavailable")
        return cpu

    monkeypatch.setattr(session_module.ort, "InferenceSession", fake_ctor)
    monkeypatch.setattr(session_module.ort, "get_available_providers", lambda: [COREML, CPU])

    assert make_session(Path("pose.onnx"), prefer_coreml=True) is cpu


def test_a_graph_the_probe_cannot_feed_keeps_its_coreml_session(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An unfeedable input says nothing about the provider, so it must not cost the speed-up."""
    coreml = _FakeSession([COREML, CPU], [_FakeInput("text", [1], "tensor(string)")])
    _install(monkeypatch, {COREML: coreml, CPU: _FakeSession([CPU])})

    assert make_session(Path("pose.onnx"), prefer_coreml=True) is coreml
    assert coreml.runs == []


def test_cpu_pinning_skips_the_probe_entirely(monkeypatch: pytest.MonkeyPatch) -> None:
    cpu = _FakeSession([CPU])
    calls = _install(monkeypatch, {CPU: cpu})

    assert make_session(Path("yolox.onnx"), prefer_coreml=False) is cpu
    assert calls == [[CPU]]
    assert cpu.runs == [], "there is nothing to prove about the fallback provider"


def test_a_cpu_session_that_cannot_be_created_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_ctor(_path: str, providers: list[str]) -> _FakeSession:
        raise RuntimeError("corrupt model file")

    monkeypatch.setattr(session_module.ort, "InferenceSession", fake_ctor)
    monkeypatch.setattr(session_module.ort, "get_available_providers", lambda: [CPU])

    with pytest.raises(RuntimeError, match="corrupt model file"):
        make_session(Path("broken.onnx"), prefer_coreml=False)


@pytest.mark.models
def test_coreml_loads_the_real_yolox_graph_and_then_fails_to_run_it() -> None:
    """The claim the detector's CPU pinning rests on: creation works, execution does not."""
    import onnxruntime as ort

    from alchemypose.inference import weights

    if COREML not in ort.get_available_providers():
        pytest.skip("this onnxruntime build has no CoreML provider")
    path = weights.cache_dir() / weights.REGISTRY["yolox_nano"].filename
    if not path.exists():
        pytest.skip("yolox_nano not cached; run `alchemypose models fetch --preset lite`")

    coreml = ort.InferenceSession(str(path), providers=[COREML, CPU])
    assert coreml.get_providers() == [COREML, CPU], "creation succeeds — the failure is not a load failure"

    feeds = probe_inputs(coreml)
    assert feeds is not None
    with pytest.raises(Exception, match="different ranks"):
        coreml.run(None, feeds)


@pytest.mark.models
def test_make_session_demotes_the_real_yolox_graph_to_cpu() -> None:
    import onnxruntime as ort

    from alchemypose.inference import weights

    if COREML not in ort.get_available_providers():
        pytest.skip("this onnxruntime build has no CoreML provider")
    path = weights.cache_dir() / weights.REGISTRY["yolox_nano"].filename
    if not path.exists():
        pytest.skip("yolox_nano not cached; run `alchemypose models fetch --preset lite`")

    assert make_session(path, prefer_coreml=True).get_providers() == [CPU]


# ------------------------------------------------------------------------ CUDA


def test_cuda_is_requested_first_when_the_gpu_build_offers_it() -> None:
    assert chosen_providers(False, [CUDA, CPU], prefer_cuda=True) == [CUDA, CPU]
    assert chosen_providers(True, [CUDA, CPU], prefer_cuda=True) == [CUDA, CPU]


def test_cuda_is_not_used_unless_asked_for() -> None:
    assert chosen_providers(False, [CUDA, CPU]) == [CPU]


def test_the_detector_path_takes_cuda_though_it_refuses_coreml(monkeypatch: pytest.MonkeyPatch) -> None:
    """``prefer_coreml=False`` is about CoreML's YOLOX failure; CUDA runs that graph fine."""
    cuda = _FakeSession([CUDA, CPU])
    calls = _install(monkeypatch, {CUDA: cuda, CPU: _FakeSession([CPU])}, available=[CUDA, CPU])
    monkeypatch.setattr(session_module.ort, "preload_dlls", lambda: None, raising=False)

    assert make_session(Path("yolox.onnx"), prefer_coreml=False) is cuda
    assert calls == [[CUDA, CPU]]
    assert len(cuda.runs) == 1, "a CUDA session is proved by a run, like a CoreML one"


def test_a_cuda_session_that_cannot_run_is_demoted_to_cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    cuda = _FakeSession([CUDA, CPU], run_error=RuntimeError("cudnn not found"))
    cpu = _FakeSession([CPU])
    calls = _install(monkeypatch, {CUDA: cuda, CPU: cpu}, available=[CUDA, CPU])
    monkeypatch.setattr(session_module.ort, "preload_dlls", lambda: None, raising=False)

    assert make_session(Path("pose.onnx"), prefer_coreml=True) is cpu
    assert calls == [[CUDA, CPU], [CPU]]


def test_prefer_cuda_false_pins_cpu_even_on_the_gpu_build(monkeypatch: pytest.MonkeyPatch) -> None:
    cpu = _FakeSession([CPU])
    calls = _install(monkeypatch, {CPU: cpu}, available=[CUDA, CPU])
    assert make_session(Path("x.onnx"), prefer_coreml=False, prefer_cuda=False) is cpu
    assert calls == [[CPU]]
