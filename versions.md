# Versions

A curated overview of where AlchemyPose has been and where it is going.
`CHANGELOG.md` is the detailed log; this file is the map.

From v2 onward, every version is a branch → pull request → review → merge → tag,
and each one leaves the project working and independently publishable. v1 is the
exception: it was built directly on `main`, by agreement, because the repository was
one commit old and had no collaborators, so a feature branch would have merged into
an empty `main`.

**Every version that changes what a user can do updates `README.md` in the same
pull request.** The README is the PyPI project page, so deferring it means
publishing a description that is wrong. AlchemyFace shipped three releases with a
README describing a package it had stopped being, because the update was
scheduled for 1.0.0. That mistake is not worth repeating.

---

## In progress

### v1 — single image · unreleased

A photograph goes in; every person in it comes back with 133 keypoints, drawn as
an interactive skeleton whose parts can be shown, hidden, isolated and inspected.

33 commits on `main`, all 2026-09-02. No tag, nothing on PyPI.

**What it contains**

- **`core/`** — `PersonPose` / `PoseFrame` / `PoseResult`, the COCO-WholeBody 133
  topology (133 named keypoints, 127 edges, 14 render groups), nine presets, JSON
  and CSV serialisation, and the COCO→H36M-17 mapping. No Qt, no onnxruntime.
- **`inference/`** — YOLOX detection on the CPU execution provider, RTMW pose on
  CoreML, vendored SimCC decoding, and a weight cache with SHA-256 verification.
  No PyTorch anywhere in the stack. The provider split is forced, not chosen —
  though not for the obvious reason: ONNX Runtime's CoreML provider *loads* the
  YOLOX graph quite happily and only fails on the first `run()`, with a rank
  mismatch inside the partition holding the graph's baked-in NMS. Sessions are
  therefore proved with a throwaway run on zeros, not merely constructed.
- **A worker thread with no Qt in it**, adapted to signals by a single bridge, on
  latest-request-wins — the policy that matters for v2 scrubbing.
- **The window** — dark canvas with zoom and pan, gradient bones under
  constant-size joints, a tri-state layer tree that is the single source of truth
  for visibility, preset chips, person chips, a confidence slider, a hover
  inspector and a person's-left/right legend. Click to emphasise, double-click or
  `Shift+H` to solo, `Esc` / `Alt+H` to restore.
- **A CLI that never touches Qt** — `alchemypose image`, `models list|fetch|clear`.

**Verified**, on this machine (Apple M5, macOS 26.6.2, Python 3.12.7, onnxruntime
1.29.0, PySide6 6.11.2):

- **343 tests pass.** 240 run with no display; 103 need
  `QT_QPA_PLATFORM=offscreen`; 7 of the 240 also need cached weights and skip
  without them. `ruff check src tests` clean; `mypy` reports zero errors across
  36 source files.
- **The CLI produces real output.**
  `alchemypose image tests/integration/fixtures/two_people.jpg --preset performance`
  finds 3 people in a 982×670 photograph and writes a 400-line CSV — one header
  plus 399 rows, 3 × 133 — and a JSON document carrying `track_id: null` and
  `keypoints_3d: null` on every person, which is the v2/v3 seam sitting there
  doing nothing on purpose.
- **The library half is genuinely headless.** `import alchemypose` loads zero
  `PySide6` modules, asserted by a subprocess test; running the full engine in a
  subprocess also loads zero.
- **Timings, measured warm on that fixture, end to end** — detection plus pose for
  every person, not the pose model alone. `lite` (YOLOX-nano + RTMW-m 256×192):
  **17 ms** for 2 people. `performance` (YOLOX-s + RTMW-l 384×288): **112 ms** for
  3 people, about 37 ms per person. The first CoreML session pays a compile delay
  of a few seconds, which is why these are warm numbers.
- **Confidence is a real 0–1 quantity.** On `performance` the fixture's 399 scores
  run min 0.094, median 0.490, max 0.858, and the drawn keypoint count falls
  399 · 335 · 247 · 153 · 28 · 0 as the threshold moves 0.0 · 0.2 · 0.4 · 0.6 ·
  0.8 · 0.9. Before the peak-scale fix the same scores ran up to 8.576 and the
  slider hid two of them across its whole range.
- **The founding use case draws correctly.** "Right hand and left leg" renders 22
  bones with the thigh attached; every single render group renders connected, with
  zero isolated dots.
- Six model files, 464 MB, downloaded and hash-verified for real.

**What v1 does not have yet.** Face level-of-detail — the 68 face points are drawn
at every zoom rather than resolving from contours as the face grows on screen — and
export from the window. CI is written but has never run: the workflow file is not
committed, so GitHub has never seen it, and its hand-derived list of the system
libraries PySide6 links against is unproven. The README has no screenshot, which is
a strange omission on a page about a visual tool. The checklist is in `todo.md`.

---

## Planned

### v2 — video

Upload and scrub, per-frame overlay, tracking, temporal smoothing.

**Why it exists.** A still image proves the model; a video is what makes a pose
tool useful and what makes it look like anything in a screen recording. It is also
where the interface question v1 is built around — *what do you show, out of 133
points per person* — stops being a nicety, because a face-full of dots flickering
frame to frame is unreadable in a way a still never is.

**What makes it cheap.** `PersonPose.track_id` has existed since the first commit
of the schema, unpopulated. Without it, video would force a schema change, and the
lifting models — which need contiguous per-person sequences — would have nothing to
consume. `PoseFrame` already carries `frame_index` and `timestamp_s`, which are 0
and `None` for a still, and `PoseResult.frames` is already a tuple of length one
rather than a single frame. The worker is already latest-request-wins, which is
exactly scrub semantics: a new frame supersedes any queued frame that has not
started. Reading frames is the new dependency (PyAV, for frame-accurate seeking,
which is part of why the project targets Python 3.12 rather than 3.10).

The per-model peak scale matters more here than it did in v1. Per-image score
normalisation was rejected partly because it would make a keypoint's confidence
depend on who else is in the frame, and flicker frame to frame in exactly this
version.

### v3 — 3D

RTMW3D, and an orbitable skeleton beside the image.

**Why it exists.** It is the point of choosing this model family. RTMW3D emits the
**same 133-keypoint topology** with a third coordinate, so it is not a different
skeleton with a different layer panel — it is the same skeleton, seen from
somewhere else.

**What makes it cheap.** `keypoints_3d` and `space_3d` exist on `PersonPose`
already, and `Space` already names `CAMERA_M` and `ROOT_REL`. Because the topology
is unchanged, the layer tree, the render groups, the presets, `group_of`, the
side-colour convention and the selection logic all carry over untouched — the new
work is the viewport, not the model of what is being drawn. `core/h36m.py` ships in
v1 with nothing calling it: it maps COCO-WholeBody to H36M-17, synthesising pelvis,
thorax, spine and head, and every 2D→3D lifting model plugs in at exactly that
seam. It is 93 lines — the design guessed twenty, and the difference is mostly the
reasoning written down beside the constants — and putting it in v1 forced the
coordinate conventions to be settled while they were free to change — 2D is source-image
pixels with the origin top-left and y down; 3D will be camera-space, right-handed,
metres, OpenCV convention, converted to y-up only at glTF export.

`pyqtgraph` and `PyOpenGL` are already declared as the optional `threed` extra.
Qt3D is deprecated and QtQuick3D and QtGraphs are commercial-or-GPLv3, so neither
can ever be imported here without relicensing the distributed application.

---

## Decisions that shaped v1

| | Alternative | Chosen, and why |
|---|---|---|
| Pose model | Ultralytics YOLO-pose | **RTMW**, Apache-2.0. Ultralytics is AGPL-3.0, which extends to weights, and offers 17 body points — no face, no hands |
| Bindings | PyQt6 | **PySide6_Essentials**, LGPL and dynamically loaded. ~111 MB against 443 MB for full PySide6 |
| SimCC decoding | depend on `rtmlib` | **vendored**, 116 lines. `rtmlib` pulls in both `opencv-python` and `opencv-contrib-python`, whose bundled Qt plugins conflict with PySide6 |
| Render groups | a partition of the 133 indices | **a cover** — limbs own their anchor joint, or a selected leg draws as a shin |
| Keypoint scores | the raw SimCC peak, as the references ship it | **divided by a measured per-model peak scale**, so a 0–1 threshold means something |
| Execution providers | CoreML for everything | **CPU for the detector, CoreML for the pose model** — forced; CoreML loads the YOLOX graph and then fails the first `run()` on it |
| Weights | vendored in the wheel | **fetched at runtime, SHA-256 verified**, following AlchemyFace |
| Editing keypoints | in scope | **out** — the moment a joint can be dragged this becomes a second AlchemyAnnotate |
