# AlchemyPose

[![Python 3.12](https://img.shields.io/badge/python-3.12%20only-blue.svg)](https://www.python.org/)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

Whole-body human pose over an image — **a typed Python library and a PySide6 desktop
application**, built on [RTMW](https://github.com/open-mmlab/mmpose/tree/main/projects/rtmpose)
and [YOLOX](https://github.com/open-mmlab/mmpose/tree/main/projects/rtmpose) through
ONNX Runtime.

Drop in a photograph. Every person in it gets 133 keypoints — body, feet, face and
both hands — and you choose what to look at: the whole skeleton, one hand, the left
leg, the face on its own.

> **Status: v1 works, on single images.** Open a photograph and every person in it
> comes back with 133 keypoints on a zoomable canvas, driven by a layer tree, preset
> chips and a confidence slider. The face resolves from contours to individual points
> as it grows on screen, and the Export menu writes the keypoints as JSON or CSV and
> the overlay as a PNG. The CLI produces the same result with no window at all. Still
> missing: video, and 3D. See [Roadmap](#roadmap).

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
`PySide6_Essentials` is about 110 MB to download and about 335 MB once installed, and
the default model preset adds 253 MB of weights on first use (235 MB over the wire).
Both are one-time. The `lite` preset is 63 MB if you would rather trade accuracy for
footprint.

On a bare Linux box the PySide6 wheel also needs a handful of system libraries it
links against but does not carry (`libegl1`, `libgl1`, `libx11-6`, `libxkbcommon0`,
`libfontconfig1`, `libdbus-1-3`, glib) — see `.github/workflows/ci.yml`, which
installs exactly that set and nothing more.

---

## Using it

### The window

```bash
alchemypose gui
```

`Cmd+O` opens an image — `Ctrl+O` on Windows and Linux, since the action takes Qt's
standard Open sequence. The photograph appears immediately and inference runs on a
worker thread, so the window never freezes; the skeleton lands when the model is done.
`Ctrl`+wheel zooms under the cursor (a bare wheel scrolls the view), and the middle
button drags to pan.

Everything on the right decides what is drawn. The layer tree is the single source
of truth — a two-level, tri-state list of render groups — and the preset chips above
it (Full, Upper, Lower, Left, Right, Hands, Face, All, None) simply set its check
states, so a chip lights up whenever the tree happens to equal it, including when you
got there by hand. With more than one person, the numbered chips in the toolbar
narrow everything to one subject.

On the canvas, clicking a joint emphasises its group and fades the rest; double-click
(or `Shift+H`) solos it and hides everything else; `Esc` or `Alt+H` puts it back.
Hovering reads the keypoint's name, confidence and group into the inspector.

Zooming into a face changes what is drawn on it and not merely how large it is.
Below about 80 pixels of face on screen it is drawn as contours in a single grey;
above that the brows, eyes, nose and lips take their own colours; and only above
about 200 pixels — where adjacent points are finally far enough apart to aim at —
are the 68 individual keypoints drawn at all. Sixty-eight dots on a thumbnail are a
smudge, whereas the contours through them read as a face at any size. The **Face**
chip zooms to the face as well as selecting it, which is what spares the window a
separate face mode.

### Getting it back out

The **Export** menu writes the keypoints as JSON or CSV, and the overlay as a PNG.
The PNG renders the scene rather than the widget, so it arrives at the photograph's
own resolution and is the same file whatever the zoom, the scroll position or the
shape of the window. A PNG needs only an open photograph; the two data formats need
a result. When there is nothing to write yet it says so in the status bar rather
than opening a dialog on an empty canvas.

### One image, from the shell

```console
$ alchemypose image tests/integration/fixtures/two_people.jpg --preset lite --json /tmp/out.json
Wrote /tmp/out.json
```

Without `--json` the result goes to stdout; `--csv` also writes the flat
`frame,person,keypoint_index,keypoint_name,x,y,score` form. The JSON carries the
topology inline, so a result stays readable years later without this package:

```console
$ python - <<'PY'
import json
import numpy as np
r = json.load(open("/tmp/out.json"))
frame = r["frames"][0]
print(r["model"], frame["image_size"], len(frame["people"]), "people")
for i, p in enumerate(frame["people"]):
    s = np.array(p["scores"])
    print(f"person {i}: {len(p['keypoints'])} keypoints, confidence {s.min():.2f}-{s.max():.2f} (median {np.median(s):.2f})")
PY
rtmw_m_256x192 [982, 670] 2 people
person 0: 133 keypoints, confidence 0.56-1.00 (median 0.94)
person 1: 133 keypoints, confidence 0.61-1.00 (median 0.90)
```

Weights are fetched on first use. To pay for that up front:

```console
$ alchemypose models fetch --preset lite
Fetching yolox_nano ...
Fetching rtmw_m_256x192 ...
Done.

$ alchemypose models list
Cache: /Users/you/Library/Caches/alchemypose

lite:
  yolox_nano             cached
  rtmw_m_256x192         cached

balanced:
  yolox_tiny             cached
  dwpose_l_384x288       cached

performance:
  yolox_s                cached
  rtmw_l_384x288         cached
```

### What a score means

Every keypoint carries a confidence in `[0, 1]`, and it means the same thing whichever
preset produced it. That took work: the RTMPose-family graphs end in a bare `MatMul`
— no softmax, no sigmoid — so the raw SimCC peak is a logit whose height depends on
the model's training temperature. Measured on real output, a confident peak reaches
about 1.0 on RTMW-m and DWPose-l but an order of magnitude more on RTMW-l — 8.576 was
the highest of the fixture's 399 — which made a fixed 0..1 threshold meaningless
across presets and inert on the default one. `rtmw.PEAK_SCALE` records the divisor per
model, 1.0 for the first two and 10.0 for RTMW-l, and decoding divides it out and
clips to [0, 1]. It is a monotone confidence, not a calibrated probability — nothing
in the training objective makes it one.

The window's slider filters on exactly that number and starts at **0.30**. Below it a
keypoint is not drawn at all; from there the drawing fades in, reaching full opacity
at 0.55, and a bone whose weaker end is under 0.25 is drawn dashed. Those three
numbers were calibrated against measured score distributions, not chosen by eye.

---

## Models

Weights are downloaded on first use into the platform cache directory
(`~/Library/Caches/alchemypose` on macOS, `~/.cache/alchemypose` on Linux; override
with `ALCHEMYPOSE_MODEL_DIR`) and verified against a recorded SHA-256. Nothing is
vendored in the repository or the wheel.

| Preset | Detector + pose model | On disk | Pose model, per person |
|---|---|---|---|
| `lite` | YOLOX-nano + RTMW-m 256×192 | 63 MB | ~4 ms |
| `balanced` | YOLOX-tiny + DWPose-l 384×288 | 148 MB | ~13 ms |
| `performance` *(default)* | YOLOX-s + RTMW-l 384×288 | 253 MB | ~25 ms |

Sizes are the extracted `.onnx` files as they sit in the cache, in binary megabytes —
the figures `ls -lh` and `du -h` print — so the three rows add up to the 464 MB all six
files occupy together. Upstream ships them inside zips, so the transfer itself is
slightly smaller (59 / 138 / 235 MB). Timings are the pose model alone, per detected
person, on CoreML on an M5 — the detector adds 6 ms (`lite`) to 39 ms (`performance`)
per frame, whatever the head count.

The person detector runs on the CPU execution provider and the pose model on CoreML.
That split is not a tuning choice, though the reason is narrower than "CoreML cannot
load that graph". It loads it — the session is created and reports both providers —
and then raises a rank mismatch on the first `run()`, inside the partition holding
YOLOX's baked-in NMS. Sessions are therefore proved with one throwaway inference on
zeros and demoted to CPU if it fails.

All model weights are Apache-2.0, from the MMPose project.

---

## Architecture

Layers, with the Qt boundary enforced by a test:

```
core/        topology · schema · presets · serialization · H36M mapping
             no Qt, no onnxruntime — pure data, testable anywhere

media/       reading an image as upright BGR, EXIF applied

inference/   ONNX sessions · YOLOX detection · SimCC decoding · weight cache ·
             the worker thread — no Qt, so it is testable headless

gui/         PySide6: canvas · layer tree · preset and person chips ·
             confidence slider · inspector
```

`import alchemypose` pulls in no Qt at all — the application class resolves lazily —
so the library half runs headless, in CI, and on a machine with no display. The
inference worker is plain `threading`; `gui/bridge.py` is the only place it meets Qt,
and it does so by emitting signals, so nothing on a worker thread ever touches a
widget.

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
- [x] YOLOX detector and the assembled pose engine
- [x] Inference worker and CLI
- [x] Canvas, layer tree, preset chips, selection
- [x] Confidence: honest 0..1 scores, threshold slider, fade and dashed bones
- [x] Face level-of-detail
- [x] Export from the window

**v2 — video.** Upload and scrub, per-frame overlay, tracking, temporal smoothing.

**v3 — 3D.** RTMW3D emits the same 133-keypoint topology with a third coordinate, so
the layer panel and presets carry over unchanged; the new work is the viewport.

---

## Development

```bash
pytest tests/unit -q          # core and inference; no display needed
ruff check src tests main.py
mypy
mypy main.py
QT_QPA_PLATFORM=offscreen pytest tests/gui -m gui
pytest tests/integration -m models    # skipped unless weights are cached
```

The GUI tests need a Qt platform plugin, not a display: `offscreen` renders into a
buffer, so there is no xvfb anywhere in this project. They do not need that variable
on the command line, though — `tests/gui/conftest.py` sets it itself, at import,
before any widget can be constructed, so a bare `pytest` runs them headless too.
Setting it explicitly is belt and braces, and it is what the CI step does.

Two of the unit tests are marked `models` and skip without cached weights, so the
first line above reports a couple of skips rather than a clean sweep. `main.py` sits
outside `src/`, so neither `ruff check src tests` nor a bare `mypy` reaches it; it
needs naming, which is why it appears twice above.

CI (`.github/workflows/ci.yml`) runs the lint, type and test gates on Python 3.12 —
the only version, because `requires-python` is pinned to `>=3.12,<3.13`. Nothing on
a runner ever downloads weights, so the `models`-marked tests never actually execute
there. They skip, which is a green tick meaning *not exercised* rather than
*passed*.

**It has run, and it passed.** Three runs so far, every one a push to `main` and
every one successful: `e9aebc2` and `88a659e` on 2 September, `51aba8a` on
3 September (UTC), each finishing in under seventy seconds on `ubuntu-latest`. The
most recent reported `ruff` clean, `mypy` with no issues across 38 source files,
256 passed and 2 skipped in `tests/unit`, and 162 passed in `tests/gui`. It also
printed `Qt 6.11.2 on platform offscreen`, which is the line worth having: the list
of Linux system libraries further up was derived by reading the PySide6 wheel's ELF
headers rather than by watching a runner install them, and a runner has now loaded
the offscreen plugin with exactly that list present and nothing more.

Two things that tick does not cover. It says nothing about the half of the project
that needs weights — CoreML, the timings and the score distributions on this page
have only ever been measured on one Apple-silicon machine. And a workflow is proven
only for the revision that ran it: edits to `ci.yml` after `51aba8a` have not yet
met a runner.

**The suite used to pass and then abort. It no longer does.** It would die during
interpreter shutdown — `libc++abi: terminating due to uncaught exception of type
std::__1::system_error: recursive_mutex lock failed: Invalid argument`, exit status
134, printed *after* the summary line had gone by. The cause was onnxruntime's
telemetry client, which uploads on a thread of its own; at exit the teardown of its
log manager frees a mutex an in-flight upload then tries to lock. `tests/conftest.py`
now sets `ORT_DISABLE_TELEMETRY=1` before anything imports onnxruntime, which is the
only moment the setting is read. Measured here over 40 runs per arm, interleaved:
0 aborts with it, 4 without.

Run the suite as `pytest -q > out 2>&1; rc=$?` and read `rc` anyway. A pipeline such
as `pytest -q | tail -1` reports the exit status of `tail`, so it would have hidden
that abort entirely for as long as it lasted, and it will hide the next one.

---

## Licence

MIT — see [LICENSE](LICENSE).

Model weights are Apache-2.0 from [MMPose](https://github.com/open-mmlab/mmpose)
and are downloaded at runtime, not redistributed here.
