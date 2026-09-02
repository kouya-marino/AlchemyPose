# AlchemyPose

[![Python 3.12+](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

Whole-body human pose over an image — **a typed Python library and a PySide6 desktop
application**, built on [RTMW](https://github.com/open-mmlab/mmpose/tree/main/projects/rtmpose)
and [YOLOX](https://github.com/open-mmlab/mmpose/tree/main/projects/rtmpose) through
ONNX Runtime.

Drop in a photograph. Every person in it gets 133 keypoints — body, feet, face and
both hands — and you choose what to look at: the whole skeleton, one hand, the left
leg, the face on its own.

> **Status: in development.** The core data model and the inference stack are built
> and tested; the desktop GUI is being assembled. See [Roadmap](#roadmap) for what
> works today.

---

## Why

Most pose demos give you 17 body keypoints and a stick figure. AlchemyPose is built
around the thing that is actually hard once you have 133 of them: **deciding what to
show.** Face landmarks alone outnumber every other part combined, so the interface
is as much of the problem as the model.

It also stays light. No PyTorch, no MediaPipe, no `ultralytics` — the whole inference
path is ONNX Runtime, OpenCV and numpy, and the model files are fetched once at
runtime rather than shipped in the wheel.

---

## The keypoint schema

COCO-WholeBody 133, in contiguous index ranges that map directly onto the interface:

| Part | Indices | Count |
|---|---|---|
| Body | 0–16 | 17 |
| Feet | 17–22 | 6 |
| Face | 23–90 | 68 (dlib-68 order) |
| Left hand | 91–111 | 21 |
| Right hand | 112–132 | 21 |

Those ranges are grouped again into the parts the UI exposes — `torso_head`,
`left_arm`, `right_leg`, `face_eyes`, `left_hand` and so on.

The groups deliberately **overlap on anchor joints**: `left_arm` owns the shoulder
even though `torso_head` does too. Without that, asking for "just the left leg" draws
a shin — the thigh needs the hip, and a bone is only drawn when both of its endpoints
are visible.

```python
from alchemypose.core.presets import selection_for
from alchemypose.core.topology import COCO_WHOLEBODY_133 as topology

selection = selection_for("hands")          # or full_body, upper, lower, left_side, face …
bones = [e for e in topology.edges if selection.is_edge_visible(e)]
```

---

## Install

Not yet on PyPI. From source:

```bash
git clone git@github.com:kouya-marino/AlchemyPose.git
cd AlchemyPose
pyenv virtualenv 3.12.7 alchemypose && pyenv local alchemypose
pip install -e ".[dev]"
```

**Install weight, stated honestly.** The package itself is small, but
`PySide6_Essentials` is about 110 MB and the default model preset downloads roughly
250 MB on first use. Both are one-time. The `lite` preset is about 63 MB if you would
rather trade accuracy for footprint.

---

## Models

Weights are downloaded on first use into `~/Library/Caches/alchemypose`
(override with `ALCHEMYPOSE_MODEL_DIR`) and verified against a recorded SHA-256.
Nothing is vendored in the repository or the wheel.

| Preset | Detector + pose model | Download | Per person on an M5 |
|---|---|---|---|
| `lite` | YOLOX-nano + RTMW-m 256×192 | ~63 MB | ~4 ms (CoreML) |
| `balanced` | YOLOX-tiny + DWPose-l 384×288 | ~148 MB | ~14 ms (CoreML) |
| `performance` *(default)* | YOLOX-s + RTMW-l 384×288 | ~253 MB | ~25 ms (CoreML) |

The person detector runs on the CPU execution provider and the pose model on CoreML.
That split is not a tuning choice — ONNX Runtime's CoreML provider raises a rank
mismatch on the YOLOX graph and cannot create a session for it.

All model weights are Apache-2.0, from the MMPose project.

---

## Architecture

Three layers, with hard import boundaries enforced by a test:

```
core/        topology · schema · presets · serialization · H36M mapping
             no Qt, no onnxruntime — pure data, testable anywhere

inference/   ONNX sessions · YOLOX detection · SimCC decoding · weight cache
             no Qt

gui/         PySide6: canvas, layer tree, presets, export
```

`import alchemypose` pulls in no Qt at all — the application class resolves lazily —
so the library half runs headless, in CI, and on a machine with no display.

One result structure serves every phase. `PersonPose` carries `track_id` and
`keypoints_3d` fields that stay empty in v1, so video and 3D are additive rather than
a schema migration.

---

## Roadmap

**v1 — single image** *(in progress)*

- [x] COCO-WholeBody 133 topology, render groups, presets
- [x] Result schema with reserved video/3D slots
- [x] JSON and CSV serialization
- [x] COCO → H36M-17 mapping (the seam 3D lifting plugs into)
- [x] Model registry with cached, checksum-verified downloads
- [x] ONNX sessions with the CPU/CoreML provider split
- [x] SimCC decoding and top-down affine cropping
- [ ] YOLOX detector and the assembled pose engine
- [ ] Inference worker and CLI
- [ ] Canvas, layer tree, preset chips, selection
- [ ] Face level-of-detail, confidence rendering, export

**v2 — video.** Upload and scrub, per-frame overlay, tracking, temporal smoothing.

**v3 — 3D.** RTMW3D emits the same 133-keypoint topology with a third coordinate, so
the layer panel and presets carry over unchanged; the new work is the viewport.

---

## Development

```bash
pytest tests/unit -q          # core and inference; no weights, no display needed
ruff check src tests
mypy
QT_QPA_PLATFORM=offscreen pytest tests/gui -m gui
pytest tests/integration -m models    # skipped unless weights are cached
```

---

## Licence

MIT — see [LICENSE](LICENSE).

Model weights are Apache-2.0 from [MMPose](https://github.com/open-mmlab/mmpose)
and are downloaded at runtime, not redistributed here.
