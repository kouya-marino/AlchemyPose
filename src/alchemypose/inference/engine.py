# src/alchemypose/inference/engine.py
"""The contract the GUI depends on — so the GUI never imports onnxruntime."""

from __future__ import annotations

from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from alchemypose.core.schema import PoseResult


class PoseEngine(Protocol):
    """Anything that can turn an image into a PoseResult."""

    @property
    def model_name(self) -> str: ...

    def infer(self, image: NDArray[np.uint8], source: str) -> PoseResult: ...
