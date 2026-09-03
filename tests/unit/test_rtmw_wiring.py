"""``RTMWEngine.infer`` must build the input the pose model wants and decode with
its peak scale — both proved without weights.

Two properties of ``infer`` are only visible on real output, and real output
needs ~486 MB of cached ONNX files that CI does not have. Both are guarded here
instead by putting fake ONNX sessions behind the *real* engine: ``weights.ensure``
never downloads, ``make_session`` hands back stubs, and everything between
``infer`` and the ``PersonPose`` it appends is the shipped code — the detector
loop, the affine crop, the normalisation, the decode, the plumbing.

**The score scale.** ``PEAK_SCALE`` records that a confident RTMW-l peak stands
about ten times taller than an RTMW-m or DWPose-l one, and
``tests/unit/test_rtmw_scores.py`` guards those numbers. Neither of them says the
engine ever *reaches* for its entry. Hand ``decode_simcc`` a literal ``1.0``
inside ``infer`` and the table is still correct, every table test is still green,
and RTMW-l's confidences come out ten times too high: on a real photograph 397 of
399 keypoints pile up at exactly 1.0 and the confidence threshold filters
nothing. The pose stub returns SimCC rows whose peak height *this file* chooses,
so the confidence that comes out the far end can be checked against a literal.

**The blob.** A stub that returns fixed arrays whatever it is handed witnesses
nothing about the input, and everything ``infer`` does before the session call is
exactly the part no other weights-free test can see. Two edits that leave the
scale cases above green:

* ``blob = normalise_crop(warped)`` reverted to a raw transpose. onnxruntime
  accepts raw 0..255 pixels without complaint and the SimCC output still looks
  healthy; the only symptom is anatomically incoherent keypoints, which is how
  this defect survived the first time (see the ``rtmw`` module docstring).
* ``top_down_affine(bbox, self._pose_spec.input_size, image)`` given the
  *detector's* ``(416, 416)`` instead. Real onnxruntime rejects that outright
  with ``Got: 416 Expected: 256``, so it can never reach a user — but a stub
  swallows it, and a test that claims to drive the real engine would pass.

So ``_FakeSession`` records every feed and the cases below assert on it: the
blob's shape against the size the loaded model declares, its values against the
ImageNet arithmetic written out as literals, and its dtype and layout against
what the session contract requires. ``tests/unit/test_rtmw_preprocess.py`` pins
the same numbers on ``normalise_crop`` directly; the point here is that ``infer``
actually calls it.

``EXPECTED`` and ``EXPECTED_BLOB_SHAPE`` are written as literals rather than
derived from ``PEAK_SCALE``, ``weights.REGISTRY`` or ``engine.peak_scale``, so a
run cannot re-derive whatever the source happens to be doing and then agree with
itself.
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
from alchemypose.inference.detector import letterbox
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

#: The exact array each pose model's session must be handed: NCHW, batch of one,
#: three channels, then the model's own (height, width). Literal for the same
#: reason ``EXPECTED`` is — reading the size out of ``weights.REGISTRY`` here
#: would only re-run the lookup ``infer`` is being tested for making.
EXPECTED_BLOB_SHAPE: dict[str, tuple[int, int, int, int]] = {
    "rtmw_m_256x192": (1, 3, 256, 192),
    "dwpose_l_384x288": (1, 3, 384, 288),
    "rtmw_l_384x288": (1, 3, 384, 288),
}

#: A single colour, in OpenCV's (B, G, R) channel order, painted over the whole
#: synthetic photograph. ``FLAT_PERSON_BOX`` then keeps the padded crop well
#: inside the frame, so every pixel the pose model is handed came from this one
#: triple and the value it must arrive as can be written out as three numbers.
#: Three distinct channels are what makes a missing BGR -> RGB swap visible.
FLAT_BGR: tuple[int, int, int] = (200, 100, 50)

#: What ``FLAT_BGR`` must become in the blob, per output plane: ``(v - mean) / std``
#: for the *RGB* channel that belongs in that plane, using mmpretrain's ImageNet
#: constants. Literals, not a re-run of the implementation's arithmetic — these
#: are the same three numbers ``tests/unit/test_rtmw_preprocess.py`` pins:
#:
#:     plane 0 <- red:   (50 - 123.675) / 58.395
#:     plane 1 <- green: (100 - 116.28) / 57.12
#:     plane 2 <- blue:  (200 - 103.53) / 57.375
NORMALISED_PLANES: tuple[float, float, float] = (-1.26166624, -0.28501401, 1.68139434)

#: Image dimensions and the person's box within them, in source-image pixels.
#: ``top_down_affine`` widens the box to the model's aspect ratio and pads it by
#: 1.25, and every pose model in the registry has the same 3:4 aspect, so this
#: one box lands the same crop for all of them: x 220..420 of 640, y 106..374 of
#: 480. Entirely inside the frame, so ``warpAffine`` never fills a border with
#: black and the crop stays one colour. Move the box outward and the value case
#: below fails loudly rather than quietly checking a weaker thing.
FLAT_IMAGE_SIZE: tuple[int, int] = (480, 640)  # (height, width)
FLAT_PERSON_BOX: tuple[float, float, float, float] = (240.0, 160.0, 400.0, 320.0)

#: The constant ``letterbox`` pads with, and therefore the only value besides
#: ``FLAT_BGR`` that may reach the detector's session.
LETTERBOX_PAD = 114.0


class _FakeInput:
    """One entry of ``InferenceSession.get_inputs()``. Only the name is read."""

    def __init__(self, name: str) -> None:
        self.name = name


class _FakeSession:
    """The two things the engine asks of an ONNX session: an input name and ``run``.

    Returns fixed arrays, so it can prove nothing about its input by failing —
    real onnxruntime rejects a wrongly shaped or wrongly typed blob, this cannot.
    ``feeds`` is the replacement for that: every call is kept so the cases below
    can assert on what ``infer`` actually built.
    """

    def __init__(self, name: str, outputs: list[NDArray[Any]]) -> None:
        self._name = name
        self._outputs = outputs
        self.feeds: list[dict[str, NDArray[Any]]] = []

    def get_inputs(self) -> list[_FakeInput]:
        return [_FakeInput(self._name)]

    def run(self, _outputs: Any, feeds: dict[str, NDArray[Any]]) -> list[NDArray[Any]]:
        self.feeds.append(feeds)
        return self._outputs

    def sole_blob(self) -> NDArray[Any]:
        """The one array this session was handed — the input every case below reads.

        Both assertions guard the same hole: with no call recorded, or the array
        filed under a name the engine invented, there is nothing to check and a
        case that quietly skipped its subject would report success.
        """
        assert len(self.feeds) == 1, (
            f"expected infer to run the {self._name!r} session exactly once for this frame, it ran "
            f"{len(self.feeds)} times. With no recorded call there is no input to check, so every "
            f"assertion about the blob would pass on an engine that never built one"
        )
        feed = self.feeds[0]
        assert sorted(feed) == [self._name], (
            f"the session declares its input as {self._name!r}, but infer fed it {sorted(feed)}. "
            f"onnxruntime raises InvalidArgument on an unknown input name; nothing here would"
        )
        return feed[self._name]


def _image() -> NDArray[np.uint8]:
    """A small photograph-shaped image. Its content only has to be non-uniform."""
    rows = np.arange(240, dtype=np.uint8)[:, None, None]
    cols = np.arange(320, dtype=np.uint8)[None, :, None]
    return np.broadcast_to(rows + cols, (240, 320, 3)).astype(np.uint8)


def _flat_image() -> NDArray[np.uint8]:
    """The same shape of photograph, painted entirely in ``FLAT_BGR``.

    Uniform on purpose: it is what lets the value of the blob be stated as three
    literals instead of recomputed, and it is why the crop's position matters
    (see ``FLAT_PERSON_BOX``).
    """
    return np.full((*FLAT_IMAGE_SIZE, 3), FLAT_BGR, dtype=np.uint8)


def _one_person_detection() -> list[NDArray[Any]]:
    """The YOLOX graph's own outputs for one confident person: ``dets`` and ``labels``.

    ``dets`` is xyxy + score in letterboxed-canvas pixels, class 0 is person —
    the real ``YoloxDetector`` then thresholds, suppresses and unletterboxes it,
    so the crop ``infer`` builds comes out of the shipped detector code.
    """
    dets = np.array([[[40.0, 40.0, 200.0, 400.0, 0.93]]], dtype=np.float32)
    labels = np.zeros((1, 1), dtype=np.int64)
    return [dets, labels]


def _detection_of(box: tuple[float, float, float, float], detector_size: int) -> list[NDArray[Any]]:
    """The same two outputs, for a person standing at ``box`` in *source-image* pixels.

    The graph reports in letterboxed-canvas pixels and ``YoloxDetector`` divides
    the ratio back out, so the box is scaled by the shipped ``letterbox``'s own
    ratio here rather than by arithmetic copied out of it: put image pixels in,
    and the detector hands ``infer`` back the box that went in.
    """
    _canvas, ratio = letterbox(_flat_image(), detector_size)
    x0, y0, x1, y1 = (value * ratio for value in box)
    dets = np.array([[[x0, y0, x1, y1, 0.93]]], dtype=np.float32)
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


def _engine(
    monkeypatch: pytest.MonkeyPatch, preset: str, detection: list[NDArray[Any]] | None = None
) -> tuple[RTMWEngine, _FakeSession, _FakeSession]:
    """The real ``RTMWEngine`` for ``preset``, with both ONNX sessions stubbed out.

    Nothing is downloaded and nothing is read from the model cache: ``ensure``
    returns a path neither stub ever opens.
    """
    det_key, pose_key = weights.PRESETS[preset]
    det_session = _FakeSession("images", detection if detection is not None else _one_person_detection())
    pose_session = _FakeSession("input", _simcc_pair(_peaks(), weights.REGISTRY[pose_key].input_size))

    monkeypatch.setattr(weights, "ensure", lambda key, **_kwargs: Path(f"/nonexistent/{key}.onnx"))
    monkeypatch.setattr(detector_module, "make_session", lambda _path, **_kwargs: det_session)
    monkeypatch.setattr(rtmw_module, "make_session", lambda _path, **_kwargs: pose_session)

    engine = RTMWEngine(preset=preset)
    assert engine.model_name == pose_key
    return engine, det_session, pose_session


def _scores(monkeypatch: pytest.MonkeyPatch, preset: str) -> NDArray[np.float32]:
    """Run one synthetic frame through the real ``infer`` and return the confidences."""
    engine, _det_session, pose_session = _engine(monkeypatch, preset)
    result = engine.infer(_image(), source="synthetic")

    assert pose_session.feeds, "infer never ran the pose model, so it decoded nothing"
    people = result.frames[0].people
    assert len(people) == 1, f"expected the one stubbed detection, got {len(people)} people"
    scores = people[0].scores
    assert scores.shape == (len(COCO_WHOLEBODY_133),)
    assert scores.dtype == np.float32
    return scores


def _flat_frame(monkeypatch: pytest.MonkeyPatch, preset: str) -> tuple[_FakeSession, _FakeSession]:
    """Run one uniform-colour frame through the real ``infer``; return both stubs.

    The detection is placed with the preset's own detector input size, so the
    box the detector reconstructs is ``FLAT_PERSON_BOX`` whichever YOLOX the
    preset carries and the crop lands in the same place every time.
    """
    det_key = weights.PRESETS[preset][0]
    detection = _detection_of(FLAT_PERSON_BOX, weights.REGISTRY[det_key].input_size[0])
    engine, det_session, pose_session = _engine(monkeypatch, preset, detection)

    result = engine.infer(_flat_image(), source="flat")

    people = result.frames[0].people
    assert len(people) == 1, (
        f"expected the one stubbed detection, got {len(people)} people — the frame below never "
        f"reached the pose model as this file intends, so nothing it asserts would mean anything"
    )
    return det_session, pose_session


# --------------------------------------------------------------------------
# What infer hands the session. Nothing downstream of the session call can see
# any of this, because the stub accepts whatever it is given.
# --------------------------------------------------------------------------


@pytest.mark.parametrize("preset", sorted(weights.PRESETS))
def test_infer_crops_to_the_input_size_of_the_pose_model_it_loaded(
    monkeypatch: pytest.MonkeyPatch, preset: str
) -> None:
    """The blob's spatial size must be the loaded pose model's, not another model's.

    RTMW-m takes 256x192 and the two large models take 384x288, so a size read
    from anywhere but this model's own registry entry — the detector's 416x416,
    or another preset's — disagrees with at least one of these three. Real
    onnxruntime refuses the mismatch outright (``Got: 416 Expected: 256``); the
    stub takes it, which is why the size is checked here rather than assumed.
    """
    pose_key = weights.PRESETS[preset][1]
    _det_session, pose_session = _flat_frame(monkeypatch, preset)
    blob = pose_session.sole_blob()
    want = EXPECTED_BLOB_SHAPE[pose_key]

    assert blob.shape == want, (
        f"{preset} loads {pose_key}, which declares an input of {want} (NCHW), but infer fed its "
        f"session {blob.shape}. onnxruntime would reject that with an INVALID_ARGUMENT on the very "
        f"first frame, so the crop is not being taken at the size this model asked for"
    )


@pytest.mark.parametrize("preset", sorted(weights.PRESETS))
def test_infer_normalises_the_crop_before_the_pose_model_sees_it(
    monkeypatch: pytest.MonkeyPatch, preset: str
) -> None:
    """Every pixel of a one-colour crop must arrive as ``(v - ImageNet mean) / std``, in RGB order.

    ``normalise_crop`` is pinned by ``tests/unit/test_rtmw_preprocess.py``; what
    is pinned here is that ``infer`` calls it. Skip it and the blob arrives as
    raw 0..255 BGR — which onnxruntime accepts, SimCC answers confidently, and
    only the anatomy of the decoded pose gives away. Keep the arithmetic but
    drop the channel reversal and the planes come out in the wrong order, which
    the three distinct values catch.
    """
    _det_session, pose_session = _flat_frame(monkeypatch, preset)
    blob = pose_session.sole_blob()
    seen = [(float(blob[0, plane].min()), float(blob[0, plane].max())) for plane in range(3)]

    for plane, want in enumerate(NORMALISED_PLANES):
        low, high = seen[plane]
        assert low == pytest.approx(want, abs=1e-6) and high == pytest.approx(want, abs=1e-6), (
            f"{preset}: the crop is {FLAT_BGR} (B, G, R) at every pixel, so plane {plane} of the "
            f"blob must be {want} everywhere — ImageNet mean/std applied to the RGB channel that "
            f"belongs there. It runs {low:.6f}..{high:.6f}. Values in the hundreds mean infer "
            f"never normalised at all; the right numbers in the wrong planes mean the BGR -> RGB "
            f"reversal is gone; a spread means the crop is no longer one colour and this case is "
            f"measuring the wrong thing. Plane min/max as fed: {seen}"
        )


@pytest.mark.parametrize("preset", sorted(weights.PRESETS))
def test_infer_hands_the_pose_model_a_contiguous_nchw_float32_blob(
    monkeypatch: pytest.MonkeyPatch, preset: str
) -> None:
    """The session contract: batch of one, channels first, float32, C-contiguous.

    onnxruntime enforces the dtype (``Unexpected input data type``) and copies a
    non-contiguous array silently on every frame; the stub does neither.
    """
    _det_session, pose_session = _flat_frame(monkeypatch, preset)
    blob = pose_session.sole_blob()

    assert blob.dtype == np.float32, (
        f"{preset}: infer fed the pose session {blob.dtype}, but the graph's input is float32 — "
        f"onnxruntime raises 'Unexpected input data type' rather than converting"
    )
    assert blob.ndim == 4 and blob.shape[0] == 1 and blob.shape[1] == 3, (
        f"{preset}: infer fed the pose session {blob.shape}; the graph takes NCHW with a batch of "
        f"one and three colour channels, so anything else is not a single RGB crop"
    )
    assert blob.flags["C_CONTIGUOUS"], (
        f"{preset}: the blob fed to the pose session is not C-contiguous, so onnxruntime copies "
        f"the whole {blob.shape} array on every frame before it can run"
    )


@pytest.mark.parametrize("preset", sorted(weights.PRESETS))
def test_infer_hands_the_detector_raw_letterboxed_pixels(
    monkeypatch: pytest.MonkeyPatch, preset: str
) -> None:
    """YOLOX is the exception: raw 0..255 BGR, padded with 114, never normalised.

    The two stages disagree on purpose — the pose graph wants ImageNet-normalised
    RGB and this one wants the pixels as OpenCV read them — so normalising here
    "for consistency" is a plausible edit that would cost detections. On a frame
    painted one colour the whole blob is two numbers: the colour, and the pad.
    """
    det_session, _pose_session = _flat_frame(monkeypatch, preset)
    blob = det_session.sole_blob()

    assert blob.dtype == np.float32 and blob.ndim == 4 and blob.shape[1] == 3, (
        f"{preset}: infer fed the detector session {blob.shape} of {blob.dtype}, not a float32 "
        f"NCHW blob"
    )
    for plane, value in enumerate(FLAT_BGR):
        seen = sorted(float(v) for v in np.unique(blob[0, plane]))
        assert seen == [min(value, LETTERBOX_PAD), max(value, LETTERBOX_PAD)], (
            f"{preset}: the frame is {FLAT_BGR} (B, G, R) at every pixel and letterbox pads with "
            f"{LETTERBOX_PAD}, so plane {plane} of the detector's blob may hold only {value} and "
            f"{LETTERBOX_PAD}. It holds {seen}. YOLOX takes raw BGR 0..255 — normalising it, or "
            f"reversing the channels into RGB, costs detections and nothing downstream says so"
        )


def test_every_pose_model_has_an_expected_blob_shape() -> None:
    """The literal shapes above must still be the sizes the registry declares.

    They are written out so a wrong lookup inside ``infer`` cannot be confirmed
    by the same lookup in the test. That only holds while the literals are
    current, so the two are compared here, once, in the one place a registry
    change should have to be acknowledged.
    """
    assert set(EXPECTED_BLOB_SHAPE) == set(PEAK_SCALE)
    for _det_key, pose_key in weights.PRESETS.values():
        width, height = weights.REGISTRY[pose_key].input_size
        assert EXPECTED_BLOB_SHAPE[pose_key] == (1, 3, height, width), (
            f"{pose_key} now declares an input of {(width, height)} (width, height), so the blob "
            f"shape expected of it is {(1, 3, height, width)}, not {EXPECTED_BLOB_SHAPE[pose_key]}. "
            f"Update the literal — the cases above are checking a size this model no longer takes"
        )


def test_the_presets_exercised_here_span_more_than_one_input_size() -> None:
    """Otherwise the size case above would still pass with the size hard-coded."""
    sizes = {weights.REGISTRY[pose_key].input_size for _det_key, pose_key in weights.PRESETS.values()}
    assert len(sizes) > 1, (
        f"every preset now crops to the same input size {sizes}, so nothing above can tell a "
        f"per-model lookup apart from a hard-coded constant"
    )


# --------------------------------------------------------------------------
# What infer makes of what the session returns.
# --------------------------------------------------------------------------


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
