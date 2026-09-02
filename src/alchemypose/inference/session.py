"""ONNX session construction, and which execution provider each model gets.

The pose model runs roughly 2.4x faster on CoreML (RTMW-l: 60 ms -> 25 ms per
person on an M5). The detector must NOT: onnxruntime's CoreML provider raises a
rank mismatch in ``GetStaticOutputShape`` on the YOLOX graph. Hence the split.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path

import onnxruntime as ort

CPU = "CPUExecutionProvider"
COREML = "CoreMLExecutionProvider"

_log = logging.getLogger(__name__)


def chosen_providers(prefer_coreml: bool, available: Sequence[str]) -> list[str]:
    """The provider list to hand onnxruntime. CPU is always last, as the fallback."""
    if prefer_coreml and COREML in available:
        return [COREML, CPU]
    return [CPU]


def make_session(path: Path, *, prefer_coreml: bool) -> ort.InferenceSession:
    """Open a session, degrading to CPU if a CoreML session cannot be created.

    The first CoreML session pays a compile cost of a few seconds; that is a
    one-off per model file, cached by the OS.
    """
    providers = chosen_providers(prefer_coreml, ort.get_available_providers())
    try:
        return ort.InferenceSession(str(path), providers=providers)
    except Exception as exc:  # noqa: BLE001 - any provider failure must degrade, not crash
        if providers == [CPU]:
            raise
        _log.warning("CoreML session failed for %s (%s); falling back to CPU", path.name, exc)
        return ort.InferenceSession(str(path), providers=[CPU])
