"""``RTMWEngine`` must decode with *its own* model's SimCC peak scale — proved without weights.

``PEAK_SCALE`` records that a confident RTMW-l peak stands about ten times
taller than an RTMW-m or DWPose-l one, and ``tests/unit/test_rtmw_scores.py``
guards those numbers. Neither of them says the engine ever *reaches* for its
entry. Hand ``decode_simcc`` a literal ``1.0`` inside ``RTMWEngine.infer`` and
the table is still correct, every table test is still green, and RTMW-l's
confidences come out ten times too high: on a real photograph 397 of 399
keypoints pile up at exactly 1.0 and the confidence threshold filters nothing.

``tests/integration/test_score_scale.py`` catches that on real output, but only
where the ~486 MB of ONNX files are already cached — so it skips everywhere
else, CI included, and a skip guards nothing. This file closes that gap by
putting fake ONNX sessions behind the real engine: ``weights.ensure`` never
downloads, ``make_session`` hands back stubs, and the pose stub returns SimCC
rows whose peak height *this file* chooses. Everything between ``infer`` and
the confidence on ``PersonPose`` is the shipped code — the detector loop, the
affine crop, the normalisation, the decode, the plumbing into the result.

``EXPECTED`` is written as literals rather than derived from ``PEAK_SCALE`` or
from ``engine.peak_scale``, so a run cannot re-derive whatever the source
happens to be doing and then agree with itself.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray

from alchemypose.core.topology import COCO_WHOLEBODY_133
from alchemypose.inference import detector as detector_module
from alchemypose.inference import rtmw as rtmw_module
from alchemypose.inference import weights
from alchemypose.inference.rtmw import PEAK_SCALE, RTMWEngine

#: Raw SimCC peaks handed to the engine. The first is what RTMW-l emits for a
#: clearly visible joint, the second is a doubtful joint on any of the three
#: models, and the third stands taller than any model's full-confidence height
#: so that dropping the clip to ``[0, 1]`` has somewhere to show itself.
CONFIDENT_PEAK = 8.5
FAINT_PEAK = 0.4
OVERSHOOT_PEAK = 12.0

#: Which keypoint carries which raw peak. The rest of the 133 get
#: ``CONFIDENT_PEAK``, so the whole array stays meaningful.
PROBES: tuple[tuple[int, float], ...] = (
    (0, CONFIDENT_PEAK),
    (1, FAINT_PEAK),
    (2, OVERSHOOT_PEAK),
)

#: The confidence each pose model must turn each of those raw peaks into.
#: Deliberately literal: 8.5 over RTMW-l's peak scale of 10 is 0.85, and the
#: same 8.5 from a unit-scale model is a fully confident 1.0 after the clip.
EXPECTED: dict[str, dict[float, float]] = {
    "rtmw_m_256x192": {CONFIDENT_PEAK: 1.0, FAINT_PEAK: 0.4, OVERSHOOT_PEAK: 1.0},
    "dwpose_l_384x288": {CONFIDENT_PEAK: 1.0, FAINT_PEAK: 0.4, OVERSHOOT_PEAK: 1.0},
    "rtmw_l_384x288": {CONFIDENT_PEAK: 0.85, FAINT_PEAK: 0.04, OVERSHOOT_PEAK: 1.0},
}


class _FakeInput:
    """One entry of ``InferenceSession.get_inputs()``. Only the name is read."""

    def __init__(self, name: str) -> None:
        self.name = name


class _FakeSession:
    """The two things the engine asks of an ONNX session: an input name and ``run``."""

    def __init__(self, name: str, outputs: list[NDArray[Any]]) -> None:
        self._name = name
        self._outputs = outputs
        self.feeds: list[dict[str, NDArray[Any]]] = []

    def get_inputs(self) -> list[_FakeInput]:
        return [_FakeInput(self._name)]

    def run(self, _outputs: Any, feeds: dict[str, NDArray[Any]]) -> list[NDArray[Any]]:
        self.feeds.append(feeds)
        return self._outputs


def _image() -> NDArray[np.uint8]:
    """A small photograph-shaped image. Its content only has to be non-uniform."""
    rows = np.arange(240, dtype=np.uint8)[:, None, None]
    cols = np.arange(320, dtype=np.uint8)[None, :, None]
    return np.broadcast_to(rows + cols, (240, 320, 3)).astype(np.uint8)


def _one_person_detection() -> list[NDArray[Any]]:
    """The YOLOX graph's own outputs for one confident person: ``dets`` and ``labels``.

    ``dets`` is xyxy + score in letterboxed-canvas pixels, class 0 is person —
    the real ``YoloxDetector`` then thresholds, suppresses and unletterboxes it,
    so the crop ``infer`` builds comes out of the shipped detector code.
    """
    dets = np.array([[[40.0, 40.0, 200.0, 400.0, 0.93]]], dtype=np.float32)
    labels = np.zeros((1, 1), dtype=np.int64)
    return [dets, labels]


def _simcc_pair(peaks: NDArray[np.float32], input_size: tuple[int, int]) -> list[NDArray[Any]]:
    """SimCC logits whose per-keypoint peak height is exactly ``peaks``.

    Both axes are zero everywhere but one bin, and that bin is the same for
    every keypoint, so the decoded location is the middle of the crop and the
    only thing that varies between keypoints is the height of the peak — which
    is the number under test.
    """
    width, height = input_size
    count = int(peaks.size)
    simcc_x = np.zeros((1, count, width * 2), dtype=np.float32)
    simcc_y = np.zeros((1, count, height * 2), dtype=np.float32)
    simcc_x[0, :, width] = peaks
    simcc_y[0, :, height] = peaks
    return [simcc_x, simcc_y]


def _peaks() -> NDArray[np.float32]:
    peaks = np.full(len(COCO_WHOLEBODY_133), CONFIDENT_PEAK, dtype=np.float32)
    for index, raw_peak in PROBES:
        peaks[index] = raw_peak
    return peaks


def _engine(monkeypatch: pytest.MonkeyPatch, preset: str) -> tuple[RTMWEngine, _FakeSession]:
    """The real ``RTMWEngine`` for ``preset``, with both ONNX sessions stubbed out.

    Nothing is downloaded and nothing is read from the model cache: ``ensure``
    returns a path neither stub ever opens.
    """
    det_key, pose_key = weights.PRESETS[preset]
    det_session = _FakeSession("images", _one_person_detection())
    pose_session = _FakeSession("input", _simcc_pair(_peaks(), weights.REGISTRY[pose_key].input_size))

    monkeypatch.setattr(weights, "ensure", lambda key, **_kwargs: Path(f"/nonexistent/{key}.onnx"))
    monkeypatch.setattr(detector_module, "make_session", lambda _path, **_kwargs: det_session)
    monkeypatch.setattr(rtmw_module, "make_session", lambda _path, **_kwargs: pose_session)

    engine = RTMWEngine(preset=preset)
    assert engine.model_name == pose_key
    return engine, pose_session


def _scores(monkeypatch: pytest.MonkeyPatch, preset: str) -> NDArray[np.float32]:
    """Run one synthetic frame through the real ``infer`` and return the confidences."""
    engine, pose_session = _engine(monkeypatch, preset)
    result = engine.infer(_image(), source="synthetic")

    assert pose_session.feeds, "infer never ran the pose model, so it decoded nothing"
    people = result.frames[0].people
    assert len(people) == 1, f"expected the one stubbed detection, got {len(people)} people"
    scores = people[0].scores
    assert scores.shape == (len(COCO_WHOLEBODY_133),)
    assert scores.dtype == np.float32
    return scores


@pytest.mark.parametrize("preset", sorted(weights.PRESETS))
def test_infer_decodes_with_the_peak_scale_of_the_model_it_loaded(
    monkeypatch: pytest.MonkeyPatch, preset: str
) -> None:
    """A raw peak of 8.5 is 0.85 of RTMW-l's confidence and all of RTMW-m's.

    The same synthetic SimCC output goes through every preset, so an ``infer``
    that divides by a constant — 1.0, or another model's scale — must disagree
    with at least one of these presets whichever constant it picked.
    """
    pose_key = weights.PRESETS[preset][1]
    scores = _scores(monkeypatch, preset)
    seen = {raw_peak: round(float(scores[index]), 4) for index, raw_peak in PROBES}

    for index, raw_peak in PROBES:
        want = EXPECTED[pose_key][raw_peak]
        got = float(scores[index])
        assert got == pytest.approx(want, abs=1e-4), (
            f"{preset} loads {pose_key}, whose SimCC peak scale is {PEAK_SCALE[pose_key]}, so a raw "
            f"peak of {raw_peak} must decode to a confidence of {want} — infer produced {got:.4f}. "
            f"Either infer is not dividing by this model's own peak scale, or the clip to [0, 1] is "
            f"gone. Raw peak -> confidence for every probe: {seen}"
        )


@pytest.mark.parametrize("preset", sorted(weights.PRESETS))
def test_no_confidence_escapes_the_unit_interval(monkeypatch: pytest.MonkeyPatch, preset: str) -> None:
    """``PersonPose.scores`` promises ``[0, 1]``; a raw peak above the model's scale must clip."""
    scores = _scores(monkeypatch, preset)
    assert float(scores.min()) >= 0.0 and float(scores.max()) <= 1.0, (
        f"{preset}: confidences run {float(scores.min()):.4f}..{float(scores.max()):.4f}, outside the "
        f"[0, 1] range PersonPose.scores declares — the decoder's clip is not being applied"
    )


def test_the_taller_model_does_not_report_taller_confidence(monkeypatch: pytest.MonkeyPatch) -> None:
    """The same faint peak must read ten times lower on RTMW-l than on RTMW-m.

    No literal confidences here, and no reference to ``PEAK_SCALE``: the point
    is only that two models with different peak scales cannot both turn one raw
    peak into the same confidence. Hard-code any single scale in ``infer`` and
    the ratio collapses to 1.
    """
    unit_scale = float(_scores(monkeypatch, "lite")[1])
    ten_times = float(_scores(monkeypatch, "performance")[1])

    assert unit_scale == pytest.approx(FAINT_PEAK, abs=1e-4)
    assert unit_scale / ten_times == pytest.approx(10.0, rel=1e-3), (
        f"a raw peak of {FAINT_PEAK} decoded to {unit_scale:.4f} on lite (RTMW-m) and "
        f"{ten_times:.4f} on performance (RTMW-l). RTMW-l's logits are ten times taller for the "
        f"same certainty, so its confidence for the same peak must be ten times lower; getting the "
        f"same number twice means infer is using one model's scale for both"
    )


def test_every_pose_model_has_an_expected_confidence() -> None:
    """A new entry in the table must arrive with expectations, or the cases above go blind."""
    assert set(EXPECTED) == set(PEAK_SCALE)
    for _det_key, pose_key in weights.PRESETS.values():
        assert pose_key in EXPECTED


def test_the_presets_exercised_here_span_more_than_one_peak_scale() -> None:
    """Otherwise every case above would still pass with the scale hard-coded."""
    scales = {PEAK_SCALE[pose_key] for _det_key, pose_key in weights.PRESETS.values()}
    assert len(scales) > 1, (
        f"every preset now decodes at the same peak scale {scales}, so nothing above can tell a "
        f"per-model scale apart from a hard-coded constant"
    )
