from pathlib import Path

import numpy as np

from alchemypose.inference import detector as detector_module
from alchemypose.inference.detector import YoloxDetector, letterbox, nms


def test_nms_keeps_the_highest_scoring_of_two_overlapping_boxes() -> None:
    boxes = np.array([[0, 0, 100, 100], [10, 10, 105, 105]], dtype=np.float32)
    scores = np.array([0.6, 0.9], dtype=np.float32)
    assert nms(boxes, scores, iou_threshold=0.5) == [1]


def test_nms_keeps_both_when_they_barely_overlap() -> None:
    boxes = np.array([[0, 0, 50, 50], [200, 200, 260, 260]], dtype=np.float32)
    scores = np.array([0.8, 0.7], dtype=np.float32)
    assert sorted(nms(boxes, scores, iou_threshold=0.5)) == [0, 1]


def test_nms_returns_indices_in_descending_score_order() -> None:
    boxes = np.array([[0, 0, 10, 10], [500, 500, 510, 510], [900, 900, 910, 910]], dtype=np.float32)
    scores = np.array([0.2, 0.9, 0.5], dtype=np.float32)
    assert nms(boxes, scores, iou_threshold=0.5) == [1, 2, 0]


def test_nms_on_empty_input_returns_empty() -> None:
    empty = np.zeros((0, 4), dtype=np.float32)
    assert nms(empty, np.zeros((0,), dtype=np.float32), iou_threshold=0.5) == []


def test_letterbox_pads_to_a_square_without_distorting() -> None:
    image = np.zeros((100, 200, 3), dtype=np.uint8)
    padded, ratio = letterbox(image, 640)
    assert padded.shape == (640, 640, 3)
    assert ratio == 640 / 200          # limited by the longer side


# ------------------------------------------------- what the detector adds on top
#
# The exported graph already filters by class and suppresses duplicates, so a
# stub that only ever emits non-overlapping class-0 rows exercises neither
# behaviour the detector adds. These hand it exactly the rows each one exists
# for. Both were seen to fail: dropping ``is_person`` from ``_parse``'s mask
# turns the first red, keeping every row instead of ``nms(...)`` in ``detect``
# turns the second red.


class _StubSession:
    """Stands in for ``onnxruntime.InferenceSession``: fixed outputs, any input."""

    def __init__(self, dets: np.ndarray, labels: np.ndarray) -> None:
        self._outputs = [dets[None].astype(np.float32), labels[None].astype(np.int64)]

    def get_inputs(self) -> list[object]:
        class _Input:
            name = "input"

        return [_Input()]

    def run(self, _names: object, _feeds: object) -> list[np.ndarray]:
        return self._outputs


def _detector(monkeypatch, dets: np.ndarray, labels: np.ndarray) -> YoloxDetector:  # type: ignore[no-untyped-def]
    session = _StubSession(dets, labels)
    monkeypatch.setattr(detector_module, "make_session", lambda _path, **_kwargs: session)
    return YoloxDetector(Path("/nonexistent/yolox.onnx"), input_size=640, score_threshold=0.3)


def test_a_confident_row_of_another_class_is_not_a_person(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    dets = np.array(
        [
            [10, 10, 110, 310, 0.80],      # a person
            [300, 50, 500, 250, 0.95],     # something else, and more confident
        ]
    )
    labels = np.array([0, 1])
    image = np.zeros((640, 640, 3), dtype=np.uint8)

    found = _detector(monkeypatch, dets, labels).detect(image)

    assert len(found) == 1
    (x0, y0, x1, y1), score = found[0]
    assert (x0, y0, x1, y1) == (10.0, 10.0, 110.0, 310.0)
    assert score == np.float32(0.80)


def test_two_boxes_over_the_same_person_come_back_as_one(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    dets = np.array(
        [
            [100, 100, 300, 600, 0.70],
            [105, 98, 305, 610, 0.90],     # the same person, a few pixels over: IoU ~0.93
            [400, 100, 560, 600, 0.60],    # someone else entirely
        ]
    )
    labels = np.array([0, 0, 0])
    image = np.zeros((640, 640, 3), dtype=np.uint8)

    found = _detector(monkeypatch, dets, labels).detect(image)

    assert [round(score, 2) for _box, score in found] == [0.90, 0.60]
