# src/alchemypose/inference/rtmw.py
"""Detector plus pose model, assembled into a PoseEngine.

**Preprocessing correction, found empirically (see the task report):** the
RTMW/RTMPose ONNX graphs have no ``Sub``/``Div`` node ahead of their first
convolution, which could be misread as "feed raw 0..255 pixels". Doing that
produces plausible-*looking* output — SimCC still returns confident-seeming
scores and in-bounds coordinates — but the keypoints are anatomically
incoherent (hips decoded above the shoulders, ankles above the hips, left/right
keypoints scattered across the whole crop). Normalising with mmpretrain's
standard ImageNet mean/std before feeding the network produces coherent,
correctly-ordered keypoints on the same crop. The mean/std subtraction is
evidently something the exported SDK pipeline expects the *caller* to do
(the zip's discarded ``pipeline.json``, not the ``.onnx`` graph itself) —
exactly the kind of silent, plausible-looking wrongness this task warned
about.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from alchemypose.core.schema import Camera, PersonPose, PoseFrame, PoseResult
from alchemypose.core.topology import COCO_WHOLEBODY_133
from alchemypose.inference import weights
from alchemypose.inference.detector import YoloxDetector
from alchemypose.inference.session import make_session
from alchemypose.inference.simcc import affine_untransform, decode_simcc, top_down_affine

F32 = NDArray[np.float32]

#: mmpretrain/mmpose's standard ImageNet normalisation constants, in RGB order.
_IMAGENET_MEAN_RGB: F32 = np.array([123.675, 116.28, 103.53], dtype=np.float32)
_IMAGENET_STD_RGB: F32 = np.array([58.395, 57.12, 57.375], dtype=np.float32)


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

        self._detector = YoloxDetector(weights.ensure(det_key), input_size=det_spec.input_size[0])
        # CoreML here is the whole speed story: 60 ms -> 25 ms per person for RTMW-l.
        self._pose = make_session(weights.ensure(pose_key), prefer_coreml=True)
        self._pose_input = self._pose.get_inputs()[0].name
        self._preset = preset

    @property
    def model_name(self) -> str:
        return self._pose_spec.key

    def infer(self, image: NDArray[np.uint8], source: str) -> PoseResult:
        height, width = image.shape[:2]
        people: list[PersonPose] = []

        for bbox, bbox_score in self._detector.detect(image):
            warped, affine = top_down_affine(bbox, self._pose_spec.input_size, image)
            blob = normalise_crop(warped)
            simcc_x, simcc_y = self._pose.run(None, {self._pose_input: blob})

            kp_model, scores = decode_simcc(simcc_x, simcc_y)
            kp_image = affine_untransform(kp_model, affine)[0]

            people.append(
                PersonPose(
                    keypoints=kp_image.astype(np.float32),
                    scores=scores[0].astype(np.float32),
                    bbox=bbox,
                    bbox_score=bbox_score,
                )
            )

        frame = PoseFrame(
            frame_index=0, timestamp_s=None, image_size=(width, height), people=tuple(people)
        )
        return PoseResult(
            topology=COCO_WHOLEBODY_133,
            frames=(frame,),
            source=source,
            model=self.model_name,
            camera=Camera.default_for(width, height),
        )
