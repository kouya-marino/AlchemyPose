"""ONNX session construction, and which execution provider each model gets.

The pose model runs roughly 2.4x faster on CoreML (RTMW-l: 60 ms -> 25 ms per
person on an M5). The YOLOX detector runs on the CPU provider instead, and the
reason is narrower than "CoreML cannot load that graph" — it loads it fine.
``InferenceSession`` returns normally for ``yolox_nano``, ``yolox_tiny`` and
``yolox_s`` alike, and ``get_providers()`` on the result reports
``['CoreMLExecutionProvider', 'CPUExecutionProvider']``. The failure comes on
the first ``run()``::

    FAIL : Non-zero status code returned while running ..._CoreML_..._2 node.
    Status Message: coreml/model/model.mm:67 GetStaticOutputShape
    CoreML static output shape ({1,1,1,3549,3549}) and inferred shape
    ({1,3549}) have different ranks.

Identical on all three detectors, bar the row count: 3549 for the 416x416
``nano``/``tiny`` exports, 8400 for the 640x640 ``yolox_s``.

The NMS region is the part CoreML will not take, but it is not where the run
dies. For ``yolox_nano`` CoreML claims 394 of the graph's 426 nodes and splits
them over 7 partitions; ``NonMaxSuppression`` is one of the 32 it leaves on
CPU. Profiling the failing run gives the order ``_0`` (CoreML) -> three CPU
``Concat``s -> ``_1`` (CoreML) -> ``Unsqueeze``/``Exp`` (CPU) -> ``_2``
(CoreML), and ``_2`` is where it aborts, so the NMS node never executes at
all — it is downstream of the abort. Partition ``_2`` sits *upstream* of NMS
and feeds it: the ``{1,3549}`` inferred shape in the error is one value per
anchor from the box-decode region, which a CPU ``Concat`` then assembles into
the ``(1, 3549, 4)`` box tensor NMS consumes. It is not ``dets`` or
``labels``, which are shaped ``(1, N, 5)`` and ``(1, N)``.

So the thing to guard against is an *execution* failure, not a load failure,
and a guard wrapped around ``InferenceSession(...)`` alone would never see it.
``make_session`` therefore runs the graph once on zeros before handing a
CoreML session back, and demotes to CPU if that run raises.

``YoloxDetector`` still asks for CPU directly rather than letting the probe
discover the breakage for it, because the probe is not free on a graph that
fails: it costs a CoreML session that then gets thrown away, 1.8 s for
``yolox_nano`` and 6.5 s for ``yolox_s``, against 10-12 ms to open the same
file on CPU. Paying seconds at every launch to re-learn a known answer is not
worth it. The probe earns its keep on the pose models, where CoreML does work
today and a future export or onnxruntime release could quietly stop working.

**CUDA**, when the ``onnxruntime-gpu`` build is installed in place of
``onnxruntime``, is preferred over everything else — for the detector too,
which CUDA runs without complaint. Measured on an RTX 4070 laptop GPU: YOLOX-s
12 ms and RTMW-l 14 ms per run, against roughly 850 ms for the pair on that
machine's CPU provider, which is the difference between analysing a 10-second
clip in seconds and in minutes. A CUDA session goes through the same
run-on-zeros probe as CoreML: a missing cuDNN or a driver mismatch is exactly
the kind of failure that only shows when the graph runs.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort
from numpy.typing import NDArray

CPU = "CPUExecutionProvider"
COREML = "CoreMLExecutionProvider"
CUDA = "CUDAExecutionProvider"

_log = logging.getLogger(__name__)

#: onnxruntime input type names -> the numpy dtype the probe allocates for them.
#: Anything absent here is a graph the probe declines to guess at.
_PROBE_DTYPES: dict[str, type[np.generic]] = {
    "tensor(float)": np.float32,
    "tensor(float16)": np.float16,
    "tensor(double)": np.float64,
    "tensor(int64)": np.int64,
    "tensor(int32)": np.int32,
    "tensor(uint8)": np.uint8,
    "tensor(int8)": np.int8,
    "tensor(bool)": np.bool_,
}


def chosen_providers(prefer_coreml: bool, available: Sequence[str], prefer_cuda: bool = False) -> list[str]:
    """The provider list to hand onnxruntime. CPU is always last, as the fallback.

    CUDA wins over CoreML when both are asked for; no platform has both.
    """
    if prefer_cuda and CUDA in available:
        return [CUDA, CPU]
    if prefer_coreml and COREML in available:
        return [COREML, CPU]
    return [CPU]


_dlls_preloaded = False
_shadow_checked = False


def gpu_build_shadowed(installed: Sequence[str], available: Sequence[str]) -> bool:
    """Whether the GPU build is installed but the CPU build's files are the ones loaded.

    ``onnxruntime`` and ``onnxruntime-gpu`` install the same ``onnxruntime``
    package directory. This project depends on the former by name, so any
    ``pip install -U alchemypose`` after a switch to the GPU build reinstalls
    the CPU one over it — both then show in ``pip list``, the CPU files win, and
    CUDA silently disappears. Found exactly that way, on the machine the CUDA
    numbers come from: a 300-frame clip that takes 23 s on the GPU spent
    minutes on the CPU without a word.
    """
    return "onnxruntime-gpu" in installed and "onnxruntime" in installed and CUDA not in available


def _warn_if_gpu_build_shadowed() -> None:
    global _shadow_checked
    if _shadow_checked:
        return
    _shadow_checked = True
    from importlib import metadata

    installed = []
    for name in ("onnxruntime", "onnxruntime-gpu"):
        try:
            metadata.version(name)
        except metadata.PackageNotFoundError:
            continue
        installed.append(name)
    if gpu_build_shadowed(installed, ort.get_available_providers()):
        _log.warning(
            "onnxruntime-gpu is installed but the CPU onnxruntime build is shadowing it, so CUDA is "
            "unavailable and everything runs on the CPU. Fix: pip uninstall -y onnxruntime onnxruntime-gpu "
            '&& pip install "onnxruntime-gpu[cuda,cudnn]"'
        )


def _preload_cuda_libraries() -> None:
    """Let onnxruntime find CUDA and cuDNN from the ``nvidia-*`` pip wheels.

    ``onnxruntime-gpu[cuda,cudnn]`` installs those libraries as wheels rather
    than system packages, and ``preload_dlls`` is how onnxruntime is told to
    look there. Absent on older releases, and harmless when the libraries are
    system-wide anyway.
    """
    global _dlls_preloaded
    if _dlls_preloaded:
        return
    _dlls_preloaded = True
    preload = getattr(ort, "preload_dlls", None)
    if preload is None:
        return
    try:
        preload()
    except Exception as exc:  # noqa: BLE001 - the probe will demote the session if CUDA truly fails
        _log.debug("onnxruntime.preload_dlls failed: %s", exc)


def probe_inputs(session: ort.InferenceSession) -> dict[str, NDArray[Any]] | None:
    """Zero-filled feeds matching every declared input, or ``None`` if they cannot be built.

    Symbolic dimensions — ``'batch'`` on the RTMW graphs, for instance — become
    1. ``None`` comes back when a dimension is still not a positive integer
    afterwards, or when the input dtype is not one this module knows how to
    allocate; the caller then skips the probe rather than inventing a shape and
    blaming the provider for rejecting it.
    """
    feeds: dict[str, NDArray[Any]] = {}
    for spec in session.get_inputs():
        dtype = _PROBE_DTYPES.get(spec.type)
        if dtype is None:
            return None
        shape = [dim if isinstance(dim, int) else 1 for dim in spec.shape]
        if any(dim < 1 for dim in shape):
            return None
        feeds[spec.name] = np.zeros(shape, dtype=dtype)
    return feeds


def _executes(session: ort.InferenceSession, name: str) -> bool:
    """Whether this session's providers can actually run its graph.

    A provider that compiles a subgraph it cannot then execute — which is
    exactly what CoreML does to the YOLOX partition feeding NMS — only reveals
    itself
    when something runs the graph, so this runs it: one inference on zeros.
    Cheap where it succeeds, because the model is already compiled and loaded
    by the time we get here: 4.3 ms for RTMW-m, 15.4 ms for DWPose-l, 38.1 ms
    for RTMW-l, against the 1.0-2.6 s those same sessions spend being created.
    Under 1.5% either way, and it is not even wasted work — it absorbs the
    first-call warm-up the model would otherwise charge to the first real
    frame (RTMW-l: 38 ms cold, 18-22 ms warm).

    A graph the probe cannot build inputs for is given the benefit of the
    doubt, since a wrongly-shaped feed would fail on any provider.
    """
    feeds = probe_inputs(session)
    if feeds is None:
        _log.debug("no probe input could be built for %s; leaving its providers unverified", name)
        return True
    try:
        session.run(None, feeds)
    except Exception as exc:  # noqa: BLE001 - any provider failure must degrade, not crash
        _log.warning("%s cannot execute %s (%s); falling back to CPU", session.get_providers()[0], name, exc)
        return False
    return True


def make_session(path: Path, *, prefer_coreml: bool, prefer_cuda: bool = True) -> ort.InferenceSession:
    """Open a session, degrading to CPU if CoreML cannot create *or run* the graph.

    The run half matters more than the create half: the one provider failure
    this project has actually hit lets the session be created and fails later
    (see the module docstring), so the session is proved with a throwaway
    inference before it is returned. ``prefer_coreml=False`` skips all of that
    and opens the file on CPU directly.

    The first CoreML session pays a compile cost of a few seconds — and so
    does every session after it. Nothing caches the compiled model: on
    ``yolox_nano`` a second session in the same process costs what the first
    did, and a session in a fresh process costs the same again, the spread
    between them being run-to-run noise rather than a cache. That is what
    makes ``YoloxDetector`` ask for CPU outright instead of re-learning a
    known answer at every launch. CoreML does expose a ``ModelCacheDirectory``
    provider option that persists compiled models between sessions; nothing
    here passes it, and what it would buy has not been measured.
    """
    _warn_if_gpu_build_shadowed()
    providers = chosen_providers(prefer_coreml, ort.get_available_providers(), prefer_cuda=prefer_cuda)
    if providers == [CPU]:
        return ort.InferenceSession(str(path), providers=[CPU])
    if providers[0] == CUDA:
        _preload_cuda_libraries()

    try:
        session = ort.InferenceSession(str(path), providers=providers)
    except Exception as exc:  # noqa: BLE001 - any provider failure must degrade, not crash
        _log.warning("%s session for %s could not be created (%s); falling back to CPU", providers[0], path.name, exc)
    else:
        if _executes(session, path.name):
            return session
    return ort.InferenceSession(str(path), providers=[CPU])
