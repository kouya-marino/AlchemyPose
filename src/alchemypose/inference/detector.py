# src/alchemypose/inference/detector.py
"""YOLOX person detection — the top-down pipeline's first stage.

Runs on the CPU execution provider, always. CoreML is not blocked from
loading this graph — it loads it, then fails the first ``run()`` in a CoreML
partition *upstream* of the baked-in NMS described below. The NMS itself is
left on the CPU provider and never runs at all, the abort landing before it;
``session.py`` carries the error text, the partition breakdown and the
measurements. ``make_session`` would catch that and fall back on its own, but
only after spending 1.8 s (``yolox_nano``) to 6.5 s (``yolox_s``) building a
CoreML session it then discards, so this stage asks for CPU up front and opens
the same file in 8-12 ms.

**The exported graph is not the raw regression head a first guess would
assume.** Loading ``yolox_nano``/``yolox_tiny``/``yolox_s`` with ``onnx.load``
and walking ``graph.node`` shows a full mmdeploy "YOLOX" export: a focus-stem
reshape/transpose straight on ``input`` (no ``Sub``/``Div`` ahead of it, so no
baked-in normalisation — the model wants raw pixels), a CSP-Darknet backbone,
a head, and then a *baked-in* ``NonMaxSuppression`` followed by a ``TopK``.
The NMS is handed scores shaped ``(1, 80, anchors)``, so it suppresses per
class itself, the way the ONNX operator is defined; the boxes it receives are
the plain decoded ``(1, anchors, 4)``, not offset by class id. So the graph
has exactly two outputs:

- ``dets``: ``(1, N, 5)`` float32 — ``x0, y0, x1, y1, score``, xyxy, already
  in the *letterboxed input's* pixel space. Not cx/cy/w/h, not normalised to
  0..1, and NMS has already run — there is nothing left to decode from
  anchors or grids.
- ``labels``: ``(1, N)`` int64 — the class id per row. The head still
  carries all 80 COCO class slots, but only class 0 ("person") holds any
  signal after the Human-Art fine-tune: on the two-person fixture the best
  score anywhere in classes 1-79 is 3e-06, far below the graph's own 0.01 NMS
  threshold, so every row that survives is class 0.

``N`` is not "the number of real objects", and it is not a fixed budget
either: it is ``min(100, rows surviving NMS)``, so it moves with the image —
``yolox_nano`` returns 39 rows on the two-person fixture, 9 on
``multipeople.png`` and 1 on zeros. Most of those rows are near-zero-score
padding, dropped by the score threshold rather than treated as detections.

Preprocessing was verified against the graph and against real images: no
``Sub``/``Div`` node precedes the first convolution, and feeding raw BGR
uint8 pixels (cv2's native order, no RGB conversion, no mean/std scaling)
found both people in the fixture at sensible locations; swapping to RGB
changed detection confidence only in the third decimal place. Raw BGR,
0..255, matches the standard YOLOX preprocessing convention (unlike most
other detectors, YOLOX skips ImageNet normalisation).
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray

from alchemypose.inference.session import make_session

F32 = NDArray[np.float32]
BBox = tuple[float, float, float, float]

_PERSON_CLASS = 0


def letterbox(image: NDArray[np.uint8], size: int) -> tuple[NDArray[np.uint8], float]:
    """Resize preserving aspect ratio into a ``size x size`` canvas, padding with 114."""
    h, w = image.shape[:2]
    ratio = size / max(h, w)
    resized = cv2.resize(image, (int(w * ratio), int(h * ratio)), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((size, size, 3), 114, dtype=np.uint8)
    canvas[: resized.shape[0], : resized.shape[1]] = resized
    return canvas, ratio


def nms(boxes: F32, scores: F32, iou_threshold: float) -> list[int]:
    """Greedy non-maximum suppression. Returns kept indices, best score first.

    Not on ``YoloxDetector``'s hot path — the exported graph already bakes in
    a class-aware NMS (see the module docstring) — but kept as a public, pure,
    independently-tested helper, and used defensively in ``YoloxDetector.detect``
    in case a future model export does not suppress duplicates itself.
    """
    if boxes.shape[0] == 0:
        return []
    x0, y0, x1, y1 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x1 - x0) * (y1 - y0)
    order = scores.argsort()[::-1]

    kept: list[int] = []
    while order.size > 0:
        best = int(order[0])
        kept.append(best)
        if order.size == 1:
            break
        rest = order[1:]
        ix0 = np.maximum(x0[best], x0[rest])
        iy0 = np.maximum(y0[best], y0[rest])
        ix1 = np.minimum(x1[best], x1[rest])
        iy1 = np.minimum(y1[best], y1[rest])
        inter = np.clip(ix1 - ix0, 0, None) * np.clip(iy1 - iy0, 0, None)
        iou = inter / (areas[best] + areas[rest] - inter)
        order = rest[iou <= iou_threshold]
    return kept


class YoloxDetector:
    """Detects people. Always CPU — CoreML compiles this graph but cannot execute it."""

    def __init__(self, model_path: Path, input_size: int = 640, score_threshold: float = 0.3) -> None:
        self._session = make_session(model_path, prefer_coreml=False)
        self._input_name = self._session.get_inputs()[0].name
        self._size = input_size
        self._score_threshold = score_threshold

    def detect(self, image: NDArray[np.uint8]) -> list[tuple[BBox, float]]:
        padded, ratio = letterbox(image, self._size)
        blob = np.ascontiguousarray(padded.transpose(2, 0, 1)[None].astype(np.float32))
        dets, labels = self._session.run(None, {self._input_name: blob})

        boxes, scores = self._parse(dets[0], labels[0])
        if boxes.shape[0] == 0:
            return []
        keep = nms(boxes, scores, iou_threshold=0.45)
        h, w = image.shape[:2]
        out: list[tuple[BBox, float]] = []
        for i in keep:
            x0, y0, x1, y1 = (boxes[i] / ratio).tolist()
            out.append((
                (max(0.0, x0), max(0.0, y0), min(float(w), x1), min(float(h), y1)),
                float(scores[i]),
            ))
        return out

    def _parse(self, dets: F32, labels: NDArray[np.int64]) -> tuple[F32, F32]:
        """Filter the graph's already-NMS'd, already-TopK'd rows down to confident people.

        ``dets`` is ``(N, 5)`` xyxy+score in letterboxed-canvas pixel space;
        ``labels`` is ``(N,)`` class ids. Most rows are near-zero-score
        padding, so the score threshold does the real filtering here — there
        is no objectness/class-score product to compute, and no grid decoding
        left to do (see module docstring).
        """
        is_person = labels == _PERSON_CLASS
        scores = dets[:, 4]
        mask = is_person & (scores >= self._score_threshold)
        boxes = dets[mask, :4].astype(np.float32)
        return boxes, scores[mask].astype(np.float32)
