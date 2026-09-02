import numpy as np

from alchemypose.inference.detector import letterbox, nms


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
