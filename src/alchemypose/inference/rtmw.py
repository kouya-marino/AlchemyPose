# src/alchemypose/inference/rtmw.py
"""Detector plus pose model, assembled into a PoseEngine.

**Preprocessing correction, found empirically:** the RTMW/RTMPose ONNX graphs
have no ``Sub``/``Div`` node ahead of their first convolution, which could be
misread as "feed raw 0..255 pixels". Doing that produces plausible-*looking*
output — SimCC still returns confident-seeming scores and in-bounds
coordinates — but the keypoints are anatomically incoherent (hips decoded above
the shoulders, ankles above the hips, left/right keypoints scattered across the
whole crop). Normalising with mmpretrain's standard ImageNet mean/std before
feeding the network produces coherent, correctly-ordered keypoints on the same
crop. The mean/std subtraction is evidently something the exported SDK pipeline
expects the *caller* to do (the zip's discarded ``pipeline.json``, not the
``.onnx`` graph itself). Neither the graph nor onnxruntime complains when it is
skipped, so the only evidence that the input is wrong is the anatomy of the
output: when changing anything upstream of ``normalise_crop``, check joint
ordering on a real photograph rather than trusting in-range coordinates and
healthy-looking scores.

**Score-scale correction, also found empirically:** the graphs end in a bare
``MatMul`` — no softmax, no sigmoid — so the SimCC peak they emit is a logit
whose height depends on the model's training temperature, not a confidence.
``PEAK_SCALE`` below records that height per model so ``decode_simcc`` can
return a 0..1 confidence that means the same thing for every preset.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from alchemypose.core.lifting import lift
from alchemypose.core.schema import Camera, PersonPose, PoseFrame, PoseResult, Space
from alchemypose.core.topology import COCO_WHOLEBODY_133
from alchemypose.errors import WeightsError
from alchemypose.inference import weights
from alchemypose.inference.detector import YoloxDetector
from alchemypose.inference.session import make_session
from alchemypose.inference.simcc import affine_untransform, decode_simcc, decode_simcc_depth, top_down_affine

F32 = NDArray[np.float32]

#: mmpretrain/mmpose's standard ImageNet normalisation constants, in RGB order.
_IMAGENET_MEAN_RGB: F32 = np.array([123.675, 116.28, 103.53], dtype=np.float32)
_IMAGENET_STD_RGB: F32 = np.array([58.395, 57.12, 57.375], dtype=np.float32)

#: Raw SimCC peak height that means "fully confident" for each pose model — the
#: divisor ``decode_simcc`` uses to turn a raw peak into a 0..1 confidence. The
#: numbers live on ``weights.ModelSpec.peak_scale``, beside everything else that
#: describes a model; this is the per-key view the decoder and tests read.
#:
#: Why it differs per model: ``KLDiscretLoss`` trains ``softmax(pred * beta)``
#: towards ``softmax(target * label_beta)`` with a unit-height Gaussian target, so
#: a confident peak converges to about ``label_beta / beta`` — 1.0 for the RTMPose
#: body / DWPose configs (``beta=10``) and 10.0 for the RTMW configs (``beta=1,
#: label_beta=10``). Every value was checked against real output on four
#: photographs: RTMW-m peaks at 1.03, DWPose-l at 1.01, RTMW-l at 8.6, all with
#: the same ranking of confident versus doubtful joints.
#: RTMW-m's *published* config says ``beta=1`` yet its shipped ONNX (a distilled
#: checkpoint) behaves like a ``beta=10`` model, which is why the table is
#: measured, not derived from configs. Every pose model in ``weights.PRESETS``
#: needs one; a missing one raises rather than silently assuming 1.0 —
#: assuming is exactly how a 10x-scaled score went unnoticed the first time.
PEAK_SCALE: dict[str, float] = {
    key: spec.peak_scale for key, spec in weights.REGISTRY.items() if spec.peak_scale is not None
}


def peak_scale_for(pose_key: str) -> float:
    """The ``peak_scale`` to hand ``decode_simcc`` for the pose model ``pose_key``."""
    try:
        return PEAK_SCALE[pose_key]
    except KeyError:
        raise WeightsError(
            f"no SimCC peak scale recorded for pose model {pose_key!r}; measure its raw peak "
            f"height on real images and set ModelSpec.peak_scale in weights.py (known: {sorted(PEAK_SCALE)})"
        ) from None


def normalise_crop(bgr: NDArray[np.uint8]) -> F32:
    """BGR uint8 crop (HWC, as OpenCV hands it over) -> normalised RGB NCHW float32 blob.

    Pure pixel arithmetic, deliberately kept out of :class:`RTMWEngine` so it can be
    unit-tested without loading any ONNX weights. Output is ``(1, 3, H, W)``,
    C-contiguous, with channel 0 = red. The BGR -> RGB reversal happens *before*
    the mean/std subtraction, so the RGB-ordered constants land on the right channels.
    """
    rgb = bgr[:, :, ::-1].astype(np.float32)
    normalised = (rgb - _IMAGENET_MEAN_RGB) / _IMAGENET_STD_RGB
    return np.ascontiguousarray(normalised.transpose(2, 0, 1)[None].astype(np.float32))


class RTMWEngine:
    """Top-down whole-body pose: YOLOX finds people, RTMW places 133 keypoints on each."""

    def __init__(self, preset: str = "performance") -> None:
        det_key, pose_key = weights.PRESETS[preset]
        det_spec = weights.REGISTRY[det_key]
        self._pose_spec = weights.REGISTRY[pose_key]
        # Resolved before any weights load, so an unlisted model fails fast and cheaply.
        self._peak_scale = peak_scale_for(pose_key)

        self._detector = YoloxDetector(weights.ensure(det_key), input_size=det_spec.input_size[0])
        # CoreML here is the whole speed story: 60 ms -> 25 ms per person for RTMW-l.
        # make_session proves the provider can run the graph before returning the session.
        self._pose = make_session(weights.ensure(pose_key), prefer_coreml=True)
        self._pose_input = self._pose.get_inputs()[0].name
        self._preset = preset

    @property
    def model_name(self) -> str:
        return self._pose_spec.key

    @property
    def peak_scale(self) -> float:
        """Raw SimCC peak height this engine treats as confidence 1.0 (see ``PEAK_SCALE``)."""
        return self._peak_scale

    @property
    def preset(self) -> str:
        return self._preset

    def detect(self, image: NDArray[np.uint8]) -> list[tuple[tuple[float, float, float, float], float]]:
        """The detector's people for ``image``: ``(xyxy box, score)`` pairs."""
        return self._detector.detect(image)

    def infer_frame(
        self,
        image: NDArray[np.uint8],
        frame_index: int = 0,
        timestamp_s: float | None = None,
        boxes: list[tuple[tuple[float, float, float, float], float]] | None = None,
    ) -> PoseFrame:
        """One frame's people. ``boxes`` skips the detector and crops these instead.

        Video uses ``boxes`` to run the detector only every few frames and crop
        the frames between from the previous frame's keypoints, which is how
        rtmlib's ``PoseTracker`` halves the cost of a clip.
        """
        height, width = image.shape[:2]
        people: list[PersonPose] = []
        detections = self._detector.detect(image) if boxes is None else boxes

        for bbox, bbox_score in detections:
            warped, affine = top_down_affine(bbox, self._pose_spec.input_size, image)
            blob = normalise_crop(warped)
            outputs = self._pose.run(None, {self._pose_input: blob})
            simcc_x, simcc_y = outputs[0], outputs[1]

            kp_model, scores = decode_simcc(simcc_x, simcc_y, peak_scale=self._peak_scale)
            kp_image = affine_untransform(kp_model, affine)[0]
            # ``decode_simcc`` zeroes a peakless keypoint in *model-input* space,
            # which the affine then carries to the crop's top-left corner — a
            # plausible-looking spot on the person's box. Zero it again here,
            # in image space, so "no peak" means the image origin it claims to.
            kp_image[scores[0] <= 0.0] = 0.0

            # RTMW3D: a third head, depth. Same keypoints, same order, plus z —
            # lifted to a metric, root-relative skeleton (``core.lifting``).
            keypoints_3d = None
            if len(outputs) >= 3 and self._pose_spec.z_range is not None:
                depth = decode_simcc_depth(outputs[2], self._pose_spec.z_range)[0]
                keypoints_3d = lift(kp_image.astype(np.float32), depth, scores[0].astype(np.float32))

            people.append(
                PersonPose(
                    keypoints=kp_image.astype(np.float32),
                    scores=scores[0].astype(np.float32),
                    bbox=bbox,
                    bbox_score=bbox_score,
                    keypoints_3d=keypoints_3d,
                    space_3d=Space.ROOT_REL if keypoints_3d is not None else None,
                )
            )

        return PoseFrame(
            frame_index=frame_index, timestamp_s=timestamp_s, image_size=(width, height), people=tuple(people)
        )

    def infer(self, image: NDArray[np.uint8], source: str) -> PoseResult:
        height, width = image.shape[:2]
        frame = self.infer_frame(image)
        return PoseResult(
            topology=COCO_WHOLEBODY_133,
            frames=(frame,),
            source=source,
            model=self.model_name,
            camera=Camera.default_for(width, height),
        )
