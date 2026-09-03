"""``RTMWEngine.infer`` must build the input the pose model wants, decode with its
peak scale, and answer in image pixels — all three proved without weights.

Those properties of ``infer`` are only visible on real output, and real output
needs ~486 MB of cached ONNX files that CI does not have. All three are guarded
here instead by putting fake ONNX sessions behind the *real* engine: ``weights.ensure``
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

**Where the keypoints land.** ``decode_simcc`` answers in *crop* pixels — 0..192
by 0..256 for RTMW-m — and ``infer`` is the only thing that maps them back onto
the photograph, through the affine ``top_down_affine`` built for that person.
Reduce ``kp_image = affine_untransform(kp_model, affine)[0]`` to
``kp_image = kp_model[0]`` and every skeleton collapses into a crop-sized corner
at the top left while the boxes stay where the detector drew them; on the
project's own two-person fixture person 0's keypoints come out at x 63..132 for
a box at x 356..509. Nothing downstream objects — the canvas, the exporter and
the serialiser draw whatever they are handed — and a bound like "inside the
image" cannot object either, because crop coordinates are inside any photograph
at least crop-sized. That is not a hypothetical: it is what the integration
test asserted, and it stayed green through the deletion.

The stub chooses which SimCC bin peaks, so the crop coordinate the decoder will
answer is known exactly; the box handed to the detector is known too; so the
image pixel each keypoint must arrive at is known, and the cases below assert
that pixel rather than a range. Two people stand in the frame because the
multi-person form of the bug — one person's affine used for everybody — leaves
every keypoint inside the image, inside a crop, and inside *a* person.

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
from typing import Any, NamedTuple

import numpy as np
import pytest
from numpy.typing import NDArray

from alchemypose.core.schema import PersonPose
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

#: Where in the crop each keypoint's SimCC peak is planted, as a fraction of the
#: pose model's input size along each axis; keypoint ``k`` is given entry
#: ``k % 4``, so all 133 are checked and none of them is checked by accident.
#:
#: Fractions rather than pixels because the presets crop at two different sizes
#: and the image pixel a fraction maps to does not depend on the size: a point
#: ``f`` of the way across the padded crop is the point ``f`` of the way across
#: the padded box, whichever model did the cropping.
#:
#: The four are spread over the crop so a wrong *scale* shows up and not only a
#: wrong offset, and every one of them sits between 0.125 and 0.875 for two
#: reasons: inside that band a keypoint is inside the person's own bounding box,
#: so the containment case below is not just measuring the 1.25 crop padding;
#: and none of them lands on SimCC bin 0, which is where ``decode_simcc`` puts a
#: keypoint it found no peak for.
CROP_FRACTIONS: tuple[tuple[float, float], ...] = (
    (0.5, 0.5),
    (0.125, 0.125),
    (0.875, 0.25),
    (0.25, 0.875),
)


class _Person(NamedTuple):
    """One person standing in the geometry frame, and the crop that follows from them.

    ``centre`` and ``padded`` describe the crop ``top_down_affine`` builds for
    ``bbox``: the centre of the box, and the box's own size times the 1.25
    padding. They are written out rather than computed, because computing them
    would re-run the arithmetic the cases below exist to check.
    ``test_the_person_geometry_here_is_stated_consistently`` holds the literals
    to the box, so a typo in one cannot quietly move the target.
    """

    bbox: tuple[float, float, float, float]
    score: float
    centre: tuple[float, float]
    padded: tuple[float, float]


#: The frame the geometry cases run on, (height, width). Its only job is to be
#: big enough to stand both people below clear of the crop-space corner — the
#: rectangle 0..288 by 0..384 that a keypoint which never left the crop would
#: land in (see ``test_the_people_here_stand_clear_of_the_crop_space_corner``).
GEOMETRY_IMAGE_SIZE: tuple[int, int] = (540, 960)

#: Two people, far enough apart in x that neither one's keypoints could pass for
#: the other's. Both boxes are already 3:4 — the aspect every pose model in the
#: registry takes — so ``top_down_affine`` only pads them, and both sit well
#: inside the frame, so the detector's clamp to the image never moves them. The
#: scores descend because ``YoloxDetector.detect`` returns people in that order,
#: which is what makes "person 0" here the same person as "person 0" there.
GEOMETRY_PEOPLE: tuple[_Person, ...] = (
    _Person((400.0, 120.0, 520.0, 280.0), 0.93, centre=(460.0, 200.0), padded=(150.0, 200.0)),
    _Person((640.0, 200.0, 820.0, 440.0), 0.87, centre=(730.0, 320.0), padded=(225.0, 300.0)),
)

#: How far outside their own box a keypoint may sit, as a fraction of that box's
#: width and height. The number is ``tests/integration/test_rtmw.py``'s, which
#: explains what the slack is for on a real photograph; here every expected
#: keypoint is inside its box already, so the margin is pure headroom and the
#: containment case is sharper than the fraction suggests.
BBOX_MARGIN_FRACTION = 0.15

#: How far a keypoint may miss the image pixel it is owed. The mapping is exact
#: arithmetic on exact inputs; all that is left is the float32 the affine and its
#: inverse are stored in, measured at 6e-5 px over the presets and people here.
#: Every coordinate bug this file guards against is out by hundreds of pixels, so
#: there is no reason to loosen it.
UNTRANSFORM_TOLERANCE_PX = 1e-3


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


def _detections_of(
    boxes: tuple[tuple[float, float, float, float], ...],
    scores: tuple[float, ...],
    detector_size: int,
    image: NDArray[np.uint8],
) -> list[NDArray[Any]]:
    """The same two outputs, for people standing at ``boxes`` in *source-image* pixels.

    The graph reports in letterboxed-canvas pixels and ``YoloxDetector`` divides
    the ratio back out, so the boxes are scaled by the shipped ``letterbox``'s own
    ratio here rather than by arithmetic copied out of it: put image pixels in,
    and the detector hands ``infer`` back the boxes that went in.
    """
    rows: list[list[float]] = []
    _canvas, ratio = letterbox(image, detector_size)
    for box, score in zip(boxes, scores, strict=True):
        rows.append([*(value * ratio for value in box), score])
    dets = np.array([rows], dtype=np.float32)
    labels = np.zeros((1, len(rows)), dtype=np.int64)
    return [dets, labels]


def _detection_of(box: tuple[float, float, float, float], detector_size: int) -> list[NDArray[Any]]:
    """One confident person at ``box``, in the flat frame."""
    return _detections_of((box,), (0.93,), detector_size, _flat_image())


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
    monkeypatch: pytest.MonkeyPatch,
    preset: str,
    detection: list[NDArray[Any]] | None = None,
    pose_outputs: list[NDArray[Any]] | None = None,
) -> tuple[RTMWEngine, _FakeSession, _FakeSession]:
    """The real ``RTMWEngine`` for ``preset``, with both ONNX sessions stubbed out.

    Nothing is downloaded and nothing is read from the model cache: ``ensure``
    returns a path neither stub ever opens.
    """
    det_key, pose_key = weights.PRESETS[preset]
    det_session = _FakeSession("images", detection if detection is not None else _one_person_detection())
    if pose_outputs is None:
        pose_outputs = _simcc_pair(_peaks(), weights.REGISTRY[pose_key].input_size)
    pose_session = _FakeSession("input", pose_outputs)

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


def _geometry_image() -> NDArray[np.uint8]:
    """The frame the two ``GEOMETRY_PEOPLE`` stand in.

    Its pixels never reach an assertion: the detector stub reports the boxes it
    was built with and the pose stub answers the same SimCC whatever it is
    handed, so only the size of this array matters. The fill value carries no
    meaning at all — what does the work here is the geometry, not the picture.
    """
    return np.full((*GEOMETRY_IMAGE_SIZE, 3), 128, dtype=np.uint8)


def _crop_fraction_per_keypoint() -> tuple[tuple[float, float], ...]:
    """``CROP_FRACTIONS`` dealt out over the whole topology, one pair per keypoint."""
    count = len(COCO_WHOLEBODY_133)
    return tuple(CROP_FRACTIONS[index % len(CROP_FRACTIONS)] for index in range(count))


def _simcc_across_the_crop(input_size: tuple[int, int]) -> list[NDArray[Any]]:
    """SimCC logits that put keypoint ``k`` at ``CROP_FRACTIONS[k % 4]`` of the crop.

    ``_simcc_pair`` varies the height of the peak and holds its position still;
    this varies the position and holds the height. ``decode_simcc`` takes the
    argmax bin and divides by the 2.0 split ratio, so a peak in bin
    ``2 * f * size`` decodes to exactly ``f * size`` crop pixels — exactly,
    because ``test_the_crop_fractions_here_land_on_whole_simcc_bins`` keeps every
    one of those bins a whole number.
    """
    width, height = input_size
    count = len(COCO_WHOLEBODY_133)
    simcc_x = np.zeros((1, count, width * 2), dtype=np.float32)
    simcc_y = np.zeros((1, count, height * 2), dtype=np.float32)
    for index, (fraction_x, fraction_y) in enumerate(_crop_fraction_per_keypoint()):
        simcc_x[0, index, round(fraction_x * width * 2)] = CONFIDENT_PEAK
        simcc_y[0, index, round(fraction_y * height * 2)] = CONFIDENT_PEAK
    return [simcc_x, simcc_y]


def _expected_image_keypoints(person: _Person) -> NDArray[np.float32]:
    """Where every keypoint of ``person`` must land, in source-image pixels.

    The whole invariant in one line of arithmetic, and no call into the code
    being tested: the crop is ``person.padded`` pixels of photograph centred on
    ``person.centre``, so whatever the pose model placed ``f`` of the way across
    the crop belongs at ``centre + (f - 0.5) * padded`` in the photograph. The
    model's input size does not appear — a fraction of a 192-wide crop and the
    same fraction of a 288-wide one are the same place on the person.
    """
    centre_x, centre_y = person.centre
    padded_w, padded_h = person.padded
    return np.array(
        [
            (centre_x + (fraction_x - 0.5) * padded_w, centre_y + (fraction_y - 0.5) * padded_h)
            for fraction_x, fraction_y in _crop_fraction_per_keypoint()
        ],
        dtype=np.float32,
    )


def _allowed_box(person: _Person) -> tuple[float, float, float, float]:
    """``person``'s box, grown by ``BBOX_MARGIN_FRACTION`` of its own width and height."""
    x0, y0, x1, y1 = person.bbox
    margin_x = BBOX_MARGIN_FRACTION * (x1 - x0)
    margin_y = BBOX_MARGIN_FRACTION * (y1 - y0)
    return x0 - margin_x, y0 - margin_y, x1 + margin_x, y1 + margin_y


def _inside(keypoints: NDArray[np.float32], box: tuple[float, float, float, float]) -> NDArray[np.bool_]:
    """Which of ``keypoints`` fall within ``box``, as a mask."""
    x0, y0, x1, y1 = box
    within_x = (keypoints[:, 0] >= x0) & (keypoints[:, 0] <= x1)
    within_y = (keypoints[:, 1] >= y0) & (keypoints[:, 1] <= y1)
    return np.asarray(within_x & within_y)


def _geometry_frame(monkeypatch: pytest.MonkeyPatch, preset: str) -> tuple[PersonPose, ...]:
    """Run the two ``GEOMETRY_PEOPLE`` through the real ``infer`` and hand back the poses.

    The detections are placed with the preset's own detector input size and the
    SimCC peaks with its own pose input size, so both people come back at the
    boxes written above whichever pair of models the preset names — and the
    keypoints they carry are the preset's own answer to the same question.
    """
    det_key, pose_key = weights.PRESETS[preset]
    image = _geometry_image()
    detection = _detections_of(
        tuple(person.bbox for person in GEOMETRY_PEOPLE),
        tuple(person.score for person in GEOMETRY_PEOPLE),
        weights.REGISTRY[det_key].input_size[0],
        image,
    )
    pose_outputs = _simcc_across_the_crop(weights.REGISTRY[pose_key].input_size)
    engine, _det_session, _pose_session = _engine(monkeypatch, preset, detection, pose_outputs)

    people = engine.infer(image, source="geometry").frames[0].people
    assert len(people) == len(GEOMETRY_PEOPLE), (
        f"{preset}: the detector stub offered {len(GEOMETRY_PEOPLE)} people and infer came back "
        f"with {len(people)}. Every case below reads person by person, so a frame with the wrong "
        f"people in it would be checking someone who is not there"
    )
    for index, (person, want) in enumerate(zip(people, GEOMETRY_PEOPLE, strict=True)):
        assert person.bbox is not None and person.bbox == pytest.approx(want.bbox, abs=1e-3), (
            f"{preset}: person {index} came back at {person.bbox}, not the {want.bbox} this file "
            f"fed the detector — and the keypoints expected of them are worked out from that box, "
            f"so they would be measured against the wrong crop"
        )
    return people


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


# --------------------------------------------------------------------------
# Where the keypoints end up. The decoder answers in crop pixels; infer is the
# only thing that turns them into the image pixels PersonPose.space_2d
# promises, and nothing downstream can tell the two apart.
# --------------------------------------------------------------------------


@pytest.mark.parametrize("preset", sorted(weights.PRESETS))
def test_infer_returns_keypoints_in_image_pixels_not_crop_pixels(
    monkeypatch: pytest.MonkeyPatch, preset: str
) -> None:
    """Every keypoint must arrive at the one image pixel its crop position maps to.

    There is a single right answer per keypoint here, so this asserts the pixel
    rather than a range. The pose stub decides which SimCC bin peaks, which
    fixes where in the crop ``decode_simcc`` will place each keypoint; the
    detector stub decides where the person stands, which fixes what that crop
    is; between them they fix the image pixel the keypoint is owed, and
    ``_expected_image_keypoints`` states it in one line that never calls the
    transform being checked.

    A bound would not do. "Inside the image" is what the integration test
    asserted, and crop coordinates satisfy it on any photograph at least
    crop-sized — which is why deleting the untransform moved every skeleton into
    a corner of the frame without turning anything red. Both people here stand
    clear of that corner, so crop coordinates cannot satisfy this case for
    either of them, and an affine applied inverted throws the points out the
    other side.
    """
    in_w, in_h = weights.REGISTRY[weights.PRESETS[preset][1]].input_size
    people = _geometry_frame(monkeypatch, preset)

    for index, (person, want_person) in enumerate(zip(people, GEOMETRY_PEOPLE, strict=True)):
        want = _expected_image_keypoints(want_person)
        missed = np.abs(person.keypoints - want).max(axis=1)
        worst = int(np.argmax(missed))
        fraction_x, fraction_y = _crop_fraction_per_keypoint()[worst]
        assert float(missed.max()) <= UNTRANSFORM_TOLERANCE_PX, (
            f"{preset}: person {index} was detected at {want_person.bbox}, so their crop is the "
            f"{want_person.padded} pixels of photograph centred on {want_person.centre}. The pose "
            f"stub put keypoint {worst}'s peak at ({fraction_x}, {fraction_y}) of that crop — "
            f"({fraction_x * in_w:g}, {fraction_y * in_h:g}) in crop pixels — which is "
            f"({want[worst][0]:g}, {want[worst][1]:g}) in the image. infer returned "
            f"({person.keypoints[worst][0]:g}, {person.keypoints[worst][1]:g}), out by "
            f"{float(missed.max()):.4f} px, and {int((missed > UNTRANSFORM_TOLERANCE_PX).sum())} of "
            f"{missed.size} keypoints missed. PersonPose.space_2d promises IMAGE_PX: a keypoint "
            f"still in crop coordinates means the untransform is gone, one thrown clear of the "
            f"frame means the affine was applied the wrong way round"
        )


@pytest.mark.parametrize("preset", sorted(weights.PRESETS))
def test_infer_untransforms_each_person_with_their_own_crop(
    monkeypatch: pytest.MonkeyPatch, preset: str
) -> None:
    """A keypoint must land inside the person it describes, and outside everybody else.

    The portable half of the case above, and the one
    ``tests/integration/test_rtmw.py`` makes against real weights on a real
    photograph: a keypoint is a claim about the person whose crop produced it,
    so it belongs inside that person's box — give or take
    ``BBOX_MARGIN_FRACTION``, which is pure headroom here because every expected
    keypoint is inside the box already — and nowhere near anyone else's.

    This is the multi-person face of the same defect. Hoist ``affine`` out of
    ``infer``'s loop, or index it by anything other than the person being
    decoded, and two people trade skeletons while every keypoint stays inside
    the image, inside a real crop, and inside a real person. The case above
    states more, but only about the frame this file builds; this one is the
    invariant that keeps its meaning on any photograph.
    """
    people = _geometry_frame(monkeypatch, preset)

    for index, person in enumerate(people):
        for other, want_person in enumerate(GEOMETRY_PEOPLE):
            box = _allowed_box(want_person)
            inside = _inside(person.keypoints, box)
            if other == index:
                escaped = np.flatnonzero(~inside)
                first = person.keypoints[escaped[0]] if escaped.size else person.keypoints[0]
                assert escaped.size == 0, (
                    f"{preset}: {escaped.size} of person {index}'s {inside.size} keypoints are "
                    f"outside their own box {want_person.bbox}, widened by "
                    f"{BBOX_MARGIN_FRACTION:.0%} of its own size to "
                    f"{tuple(round(value, 2) for value in box)}. The first is keypoint "
                    f"{int(escaped[0])} at ({first[0]:g}, {first[1]:g}). Every one of them was "
                    f"decoded from inside this person's crop, so it can only leave this person's "
                    f"box if infer mapped it back with the wrong transform"
                )
            else:
                trespassing = np.flatnonzero(inside)
                first = person.keypoints[trespassing[0]] if trespassing.size else person.keypoints[0]
                assert trespassing.size == 0, (
                    f"{preset}: {trespassing.size} of person {index}'s keypoints landed inside "
                    f"person {other}'s box {want_person.bbox}. The first is keypoint "
                    f"{int(trespassing[0])} at ({first[0]:g}, {first[1]:g}), and person {index} "
                    f"stands at {GEOMETRY_PEOPLE[index].bbox}. infer is untransforming this person "
                    f"with somebody else's affine: the skeletons have been swapped, and neither "
                    f"the image bounds nor the confidences would show it"
                )


def test_the_crop_fractions_here_land_on_whole_simcc_bins() -> None:
    """Off a whole bin, the crop position the decoder answers is not the one expected.

    ``_simcc_across_the_crop`` plants keypoint ``k``'s peak in bin
    ``2 * f * size`` and ``decode_simcc`` divides the argmax by the 2.0 split
    ratio, so the decoded position is ``f * size`` crop pixels only while that
    bin is a whole number. Every pose model's input size is checked here,
    because a model added at a size these fractions do not divide would leave
    every expectation in this section quietly rounded.
    """
    for _det_key, pose_key in weights.PRESETS.values():
        input_size = weights.REGISTRY[pose_key].input_size
        for fractions in CROP_FRACTIONS:
            for axis, size in enumerate(input_size):
                bin_index = fractions[axis] * size * 2
                assert bin_index == int(bin_index) and 0 < bin_index < size * 2, (
                    f"{pose_key} takes {input_size} (width, height), so the fraction "
                    f"{fractions[axis]} on axis {axis} asks for SimCC bin {bin_index} of "
                    f"{size * 2}. It has to be a whole bin, and strictly inside the row: bin 0 is "
                    f"where decode_simcc parks a keypoint it found no peak for, so a probe placed "
                    f"there would make a missing peak look like a correct answer"
                )


def test_the_person_geometry_here_is_stated_consistently() -> None:
    """The literals above must still describe the boxes, and the boxes the models.

    Everything the two cases above lean on and neither of them says: every pose
    model crops at 3:4, so a box already at 3:4 is padded and never widened;
    ``centre`` and ``padded`` follow from the box and that 1.25 padding; both
    boxes sit inside the frame, so the detector's clamp to the image never moves
    them; the scores descend, so the order people come back in is the order they
    are written in; every expected keypoint is inside its own person's box,
    without which the containment case would be measuring the crop padding
    rather than the transform; and no two people's allowed boxes touch, without
    which "inside mine, outside theirs" could not fail.
    """
    height, width = GEOMETRY_IMAGE_SIZE
    for _det_key, pose_key in weights.PRESETS.values():
        in_w, in_h = weights.REGISTRY[pose_key].input_size
        assert in_w / in_h == 0.75, (
            f"{pose_key} now crops at {(in_w, in_h)} — an aspect of {in_w / in_h:.4f}, not the 3:4 "
            f"every box here is drawn at. top_down_affine would widen these boxes to suit it, the "
            f"padded sizes written above would stop being the crop, and every expected keypoint "
            f"would be wrong"
        )

    scores = [person.score for person in GEOMETRY_PEOPLE]
    assert scores == sorted(scores, reverse=True) and len(set(scores)) == len(scores), (
        f"the detection scores {scores} are not strictly descending. YoloxDetector.detect returns "
        f"people best score first, which is what makes person N in the result the same person as "
        f"person N in GEOMETRY_PEOPLE; equal scores would leave that pairing to a sort"
    )

    assert len(GEOMETRY_PEOPLE) > 1, (
        "one person in the frame cannot show a crossed affine — infer would untransform them with "
        "their own crop whichever person's affine it reached for"
    )

    for index, person in enumerate(GEOMETRY_PEOPLE):
        x0, y0, x1, y1 = person.bbox
        box_w, box_h = x1 - x0, y1 - y0
        assert 0.0 < x0 < x1 < float(width) and 0.0 < y0 < y1 < float(height), (
            f"person {index}'s box {person.bbox} is not inside the {(width, height)} frame. "
            f"YoloxDetector clamps a box to the image, so a box hanging over the edge would come "
            f"back a different shape and the crop expected of it would be the wrong crop"
        )
        assert box_w / box_h == 0.75, (
            f"person {index}'s box {person.bbox} is {box_w} x {box_h}, an aspect of "
            f"{box_w / box_h:.4f} rather than 3:4. top_down_affine would widen it to the model's "
            f"aspect before padding, so padded={person.padded} would no longer be the crop size"
        )
        assert person.centre == ((x0 + x1) / 2.0, (y0 + y1) / 2.0), (
            f"person {index} is written with centre={person.centre}, but the centre of "
            f"{person.bbox} is {((x0 + x1) / 2.0, (y0 + y1) / 2.0)}"
        )
        assert person.padded == (box_w * 1.25, box_h * 1.25), (
            f"person {index} is written with padded={person.padded}, but top_down_affine pads by "
            f"1.25, which makes the crop of {person.bbox} {(box_w * 1.25, box_h * 1.25)} pixels"
        )
        outside = np.flatnonzero(~_inside(_expected_image_keypoints(person), person.bbox))
        assert outside.size == 0, (
            f"{outside.size} of the keypoints expected of person {index} fall outside their own "
            f"box {person.bbox} — keypoint {int(outside[0])} first. CROP_FRACTIONS has strayed "
            f"outside 0.125..0.875, and the containment case is now measuring how far the 1.25 "
            f"crop padding reaches instead of where the transform put anything"
        )
        for other in GEOMETRY_PEOPLE[index + 1 :]:
            mine, theirs = _allowed_box(person), _allowed_box(other)
            assert (
                mine[2] < theirs[0] or theirs[2] < mine[0] or mine[3] < theirs[1] or theirs[3] < mine[1]
            ), (
                f"the boxes {person.bbox} and {other.bbox}, each widened by "
                f"{BBOX_MARGIN_FRACTION:.0%}, overlap. A keypoint could then be inside both people "
                f"at once, and the half of the containment case that says a keypoint is outside "
                f"everybody else could not fail"
            )


def test_the_people_here_stand_clear_of_the_crop_space_corner() -> None:
    """A keypoint that never left the crop must not be able to pass for anyone's.

    This is what makes the untransform checkable at all, and it is exactly what
    the integration test's old bound was missing. ``decode_simcc`` answers
    somewhere in 0..input width by 0..input height — the top-left corner of any
    photograph big enough to crop from — so if that rectangle reached a person's
    box, keypoints that were never mapped out of crop space would sit inside
    that person, the containment case would pass on them, and the defect would
    be back behind a test that cannot see it.
    """
    for _det_key, pose_key in weights.PRESETS.values():
        in_w, in_h = weights.REGISTRY[pose_key].input_size
        for index, person in enumerate(GEOMETRY_PEOPLE):
            allowed = _allowed_box(person)
            assert allowed[0] > in_w or allowed[1] > in_h, (
                f"{pose_key} decodes into 0..{in_w} by 0..{in_h}, which reaches into the box "
                f"person {index} is allowed, {tuple(round(value, 2) for value in allowed)}. Stand "
                f"them further from the top-left corner of the frame: as written, a keypoint left "
                f"in crop coordinates could land inside this person and the containment case would "
                f"pass on a missing untransform"
            )
