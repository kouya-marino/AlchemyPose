# src/alchemypose/media/image.py
"""Reading an image the way the rest of the app expects it: BGR, upright."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray

from alchemypose.errors import AlchemyPoseError


def load_image(path: Path) -> NDArray[np.uint8]:
    """Load an image as BGR uint8, applying any EXIF orientation.

    ``cv2.IMREAD_COLOR`` alone ignores EXIF on some builds; ``IMREAD_IGNORE_ORIENTATION``
    is deliberately *not* passed so that phone photos come back upright.
    """
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise AlchemyPoseError(f"could not read an image from {path}")
    return image.astype(np.uint8)
