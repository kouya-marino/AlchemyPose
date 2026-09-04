# AlchemyPose

[![PyPI](https://img.shields.io/badge/PyPI-v1.0.0-blue.svg)](https://pypi.org/project/alchemypose/)
[![Python 3.12 | 3.13 | 3.14](https://img.shields.io/badge/python-3.12%20%7C%203.13%20%7C%203.14-blue.svg)](https://www.python.org/)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](https://opensource.org/licenses/MIT)

Whole-body human pose over an image — **a typed Python library and a PySide6 desktop
application**, built on [RTMW](https://github.com/open-mmlab/mmpose/tree/main/projects/rtmpose)
and [YOLOX](https://github.com/open-mmlab/mmpose/tree/main/projects/rtmpose) through
ONNX Runtime.

Drop in a photograph. Every person in it gets 133 keypoints — body, feet, face and
both hands — and you choose what to look at: the whole skeleton, one hand, the left
leg, the face on its own.

> **1.0.0 — single images.** `pip install alchemypose`, type `alchemypose`, and the
> window opens. Every person in the photograph you drop in comes back with 133
> keypoints on a zoomable canvas, driven by a layer tree, preset chips and a
> confidence slider. The face resolves from contours to individual points as it grows
> on screen, and the Export menu writes the keypoints as JSON or CSV and the overlay
> as a PNG. The CLI produces the same result with no window at all. Video and 3D are
> the next two versions — see [Roadmap](#roadmap).

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

## Install

```bash
pip install alchemypose
alchemypose
```

The second line opens the window: `alchemypose` on its own is the application.
`alchemypose --version` prints `1.0.0`, and the subcommands further down do the same
work with no window at all.

From source instead, if you have access to the repository:

```bash
git clone https://github.com/kouya-marino/AlchemyPose.git
cd AlchemyPose
python3 -m venv .venv && source .venv/bin/activate   # 3.12, 3.13 or 3.14
pip install -e ".[dev]"
```

**Python 3.12 or newer.** `requires-python` is `>=3.12` — a floor with no ceiling.
The floor is there because the *development* toolchain needs it: numpy ships type
stubs written in 3.12 syntax, which mypy cannot parse under 3.11. That is a reason
for a floor and not for a ceiling, so there is no upper bound; every dependency here
already supports 3.13 and 3.14. The suite has been run in full on 3.12 and on 3.14,
and CI runs it on 3.12, 3.13 and 3.14 on every push.

**Install weight, stated honestly.** The package itself is tiny — a 79 KB wheel of
pure Python — but two things behind it are not. `PySide6_Essentials` is a 106 MB
download on macOS, whose wheel is a universal binary, and 55 to 76 MB on Linux and
Windows; installed, it takes 328 MB on disk here. And the default model preset
fetches 253 MB of ONNX weights the first time you run inference — 235 MB over the
wire, since upstream ships them zipped. Both are one-time. The `lite` preset is
63 MB if you would rather trade accuracy for footprint.

On a bare Linux box the PySide6 wheel also needs a handful of system libraries it
links against but does not carry: `libegl1`, `libgl1`, `libx11-6`, `libxkbcommon0`,
`libfontconfig1`, `libdbus-1-3` and glib. The CI workflow installs exactly that set
and nothing more, and a runner has since loaded Qt's offscreen plugin with it.

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

Those ranges are grouped again into the 14 parts the UI exposes — `torso_head`,
`left_arm`, `right_leg`, `face_eyes`, `left_hand` and so on — joined by 127 edges.

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

## Using it

### The window

```bash
alchemypose
```

That is the whole of it: no subcommand, no arguments. (`alchemypose gui` does the
same thing explicitly, and `python -m alchemypose gui` works too.)

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

That path is the test fixture in the source tree — a public-domain 1935 photograph of
two people — so substitute any photograph of your own and the numbers below become
yours. ONNX Runtime also writes a few `[W:onnxruntime:…]` lines to stderr while it
builds a session, about unused initializers and how much of the graph CoreML took;
they are notices, not errors, and stdout carries only the line above.

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

| Preset | Detector + pose model | On disk | Detection | Pose model, per person |
|---|---|---|---|---|
| `lite` | YOLOX-nano + RTMW-m 256×192 | 63 MB | 8 ms | 4 ms |
| `balanced` | YOLOX-tiny + DWPose-l 384×288 | 148 MB | 15 ms | 14 ms |
| `performance` *(default)* | YOLOX-s + RTMW-l 384×288 | 253 MB | 42 ms | 25 ms |

Sizes are the extracted `.onnx` files as they sit in the cache, in binary megabytes —
the figures `ls -lh` and `du -h` print — so the three rows add up to the 464 MB all six
files occupy together. Upstream ships them inside zips, so the transfer itself is
slightly smaller (59 / 138 / 235 MB).

The timings are medians of fifteen warm runs on one 982×670 photograph on an M5, with
the split described below — the detector on the CPU provider, the pose model on
CoreML. Detection is per frame whatever the head count; the pose column is per
detected person. End to end that photograph takes 17 ms on `lite` (2 people found),
58 ms on `balanced` and 114 ms on `performance` (3 people found). Opening the models
costs another 1–3 seconds once per process, before the first inference.

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
`keypoints_3d` fields that stay empty in 1.0.0, so video and 3D are additive rather
than a schema migration.

---

## Roadmap

**v1 — single image.** Shipped in 1.0.0.

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
- [x] Face level of detail
- [x] Export from the window

**v2 — video.** Upload and scrub, per-frame overlay, tracking, temporal smoothing.
`PersonPose.track_id` and `PoseFrame.frame_index` have been in the schema since its
first commit, and the worker is already latest-request-wins, which is what scrubbing
needs.

**v3 — 3D.** RTMW3D emits the same 133-keypoint topology with a third coordinate, so
the layer panel and presets carry over unchanged; the new work is the viewport.

---

## Development

```bash
pytest                        # the whole suite; the GUI half runs headless by itself
pytest tests/unit -q          # core and inference; no display needed
ruff check src tests main.py
mypy
mypy main.py
QT_QPA_PLATFORM=offscreen pytest tests/gui -m gui
pytest tests/integration -m models    # skipped unless weights are cached
```

At 1.0.0 that is 511 tests, all passing, on a machine with the weights cached. Twelve
of them are marked `models` — two in `tests/unit`, ten in `tests/integration` — and
check the cache rather than downloading anything, so without weights the same run
reports 499 passed and 12 skipped. `main.py` sits outside `src/`, so neither
`ruff check src tests` nor a bare `mypy` reaches it; it needs naming, which is why it
appears twice above.

The GUI tests need a Qt platform plugin, not a display: `offscreen` renders into a
buffer, so there is no xvfb anywhere in this project. They do not need that variable
on the command line, though — `tests/gui/conftest.py` sets it itself, at import,
before any widget can be constructed, so a bare `pytest` runs them headless too.
Setting it explicitly is belt and braces, and it is what the CI step does.

CI (`.github/workflows/ci.yml`) runs the test suite on **Python 3.12, 3.13 and 3.14**
— the whole of what `requires-python = ">=3.12"` now claims — and runs `ruff` and
`mypy` exactly once, in a job of their own. Both are static analysers configured
against fixed targets (`target-version = "py312"`, `python_version = "3.12"`), so
they return the same answer whichever interpreter invokes them; three copies of one
answer is not a broader gate, only a longer log. The tests are the part that actually
varies by interpreter, so the tests are the part that gets the matrix.

Nothing on a runner ever downloads weights, so the `models`-marked tests never
actually execute there. They skip, which is a green tick meaning *not exercised*
rather than *passed*.

**It has run, and it passed** — though on the single-interpreter shape this workflow
had until 1.0.0, so read the numbers below as 3.12's. Six runs so far, every one a
push to `main` and every one successful, each finishing in about a minute on
`ubuntu-latest`. The most recent reported `ruff` clean, `mypy` with no issues across
38 source files and then across `main.py`, 289 passed and 2 skipped in `tests/unit`,
179 passed in `tests/gui`, and 10 skipped in the weights-only step. It also printed
`Qt 6.11.2 on platform offscreen`, which is the line worth having: the list of Linux
system libraries further up was derived by reading the PySide6 wheel's ELF headers
rather than by watching a runner install them, and a runner has now loaded the
offscreen plugin with exactly that list present and nothing more.

**The 3.13 and 3.14 legs are new, and have not had a push to run on yet.** What has
been measured for the widened range is local: the full suite under Python 3.14.7, in
a clean virtualenv built from the released dependency set, with the package installed
from a source tree of its own so that nothing resolved back to the development
checkout — 511 passed, 0 skipped, `pytest` exit status 0. Weights were present, so
the twelve `models` tests ran against the real ONNX graphs rather than skipping.
`ruff` and `mypy` were run under 3.14 as well and returned exactly what they return
under 3.12, which is the measurement behind giving them one job instead of three. The
same tree's 1.0.0 wheel installs and runs on 3.14; built with the old `<3.13` ceiling
it does not, and pip says so — `requires a different Python: 3.14.7 not in
'<3.13,>=3.12'` — which is what the ceiling was doing to everyone on a current
interpreter. 3.13 is the gap: no 3.13 interpreter exists on the machine these numbers
were taken on, so that leg is claimed by the matrix and not yet by a run.

What that tick does not cover is the half of the project that needs weights. CoreML,
the timings and the score distributions on this page have only ever been measured on
one Apple-silicon machine.

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

### Releasing

A release is a tag. Pushing `v1.0.0` runs `.github/workflows/publish.yml`, which
re-runs every gate above on the tagged commit and then uploads to PyPI through
trusted publishing, so no API token exists anywhere in this repository. It is the one
irreversible thing here: PyPI never allows re-uploading a version, and yanking a
release does not give its number back. So the tag is spent only after the artefacts
themselves have been read — the tag is matched against the version in
`src/alchemypose/__init__.py`, then against the metadata inside the built wheel and
sdist; the wheel's `Requires-Python` is compared against the one in `pyproject.toml`,
because the string pip enforces is the one baked into the artefact and a wrong range
published there cannot be edited afterwards, only yanked; both archives are searched
for weights, `test-data/`, `docs/` and the test suite; the wheel is checked for a
top-level `main`, because `main.py` is a checkout convenience and not ours to install
into anyone's `site-packages`; and the wheel is installed into a throwaway virtualenv
where each entry point is run and asked its version. That first check reads the attribute with `ast`, the way setuptools'
`attr:` does, rather than reading `project.version` — with the version declared once
and exposed through `dynamic = ["version"]`, that key does not exist here.

The PyPI and Python badges at the top of this file are **static and version-pinned**,
not the dynamic `shields.io/pypi/…` kind. Those proved unreliable: PyPI's JSON API
serves a stale `info.version` for some time after an upload, shields.io caches
whatever it read for three hours, and GitHub's image proxy then caches shields. The
three compound, and the badge showed a version four releases behind. A pinned badge
cannot be stale — and the release fails if either badge stops agreeing with what the
package declares, so it cannot be forgotten either. The Python badge is held to the
whole `Programming Language :: Python ::` classifier list rather than just to the
`requires-python` floor: now that it names a range, checking only its first number
would wave the rest of it through.

---

## Licence

MIT — see the `LICENSE` file in the source distribution.

Model weights are Apache-2.0 from [MMPose](https://github.com/open-mmlab/mmpose)
and are downloaded at runtime, not redistributed here.
