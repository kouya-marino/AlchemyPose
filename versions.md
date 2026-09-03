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

Every commit so far is on `main`. No tag, nothing on PyPI.

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
- **Face level of detail** — the 68 face points resolve from contours, to
  contours coloured per region, to the individual points, on thresholds of 80 px
  and 200 px of face measured *on screen* rather than in the photograph. The
  canvas announces every change of scale and the overlay rebuilds only when one
  of the two thresholds is crossed. The **Face** chip magnifies the face on the
  same canvas, which is what a face mode would have been for.
- **Export** — keypoints as JSON or CSV, the overlay as a PNG rendered from the
  scene at the source photograph's own resolution, so it is the same file
  whatever the zoom and the window size happen to be.

**Verified in two places, which cover different ground.** On this machine — Apple
M5, macOS 26.6.2, Python 3.12.7, onnxruntime 1.29.0, PySide6 6.11.2 — the whole
suite runs, cached weights and CoreML included. On a clean `ubuntu-latest` runner,
CI runs everything that needs neither. Both are green.

- **442 tests pass**, with `QT_QPA_PLATFORM=offscreen` set. 277 of them run
  with no display at all; 165 are the GUI suite; 11 additionally need cached
  weights and skip without them. `ruff check src tests` is clean and `mypy` reports
  zero errors across 38 source files. Treat the total as a reading rather than a
  constant — it moves whenever a test is added, and `pytest -q` re-reads it.

  The GUI tests do not need that variable on the command line, which the older
  wording here implied. `tests/gui/conftest.py` sets it itself, at import, before a
  widget can exist, so a bare `pytest` runs them headless too; passing it explicitly
  is belt and braces and is what the CI step does.

- **CI has run three times and passed three times.** The workflow went in with
  commit `e9aebc2`, and GitHub registered and ran it on that same push.

  | Run | Commit | Started (UTC) | Result | Duration |
  |---|---|---|---|---|
  | 33684801971 | `e9aebc2` | 2026-09-02 21:22 | success | 55s |
  | 33685597889 | `88a659e` | 2026-09-02 21:31 | success | 1m 02s |
  | 33700786465 | `51aba8a` | 2026-09-03 00:44 | success | 58s |

  Read out of the most recent log rather than off the tick: `ruff check` clean;
  `mypy` "Success: no issues found in 38 source files"; `pytest tests/unit` 256
  passed and 2 skipped; `pytest tests/gui` 162 passed — lower than the local counts
  above because tests have been added since `51aba8a`, and the 2 skips are the two
  `models`-marked unit tests finding no weights. The probe step ahead of the
  GUI tests printed `Qt 6.11.2 on platform offscreen`. That is the line that
  settles the one part of the workflow which was reasoning rather than observation:
  the list of Linux system libraries the PySide6 wheel links against was read out
  of the wheel's ELF headers, and a runner has now loaded the offscreen plugin with
  exactly that list installed and nothing more.

- **The shutdown abort is fixed, and the cause was telemetry.** For most of v1 the
  suite would report green, print its pass line, and then die during interpreter
  shutdown — `recursive_mutex lock failed` out of `libc++abi`, exit status 134.
  onnxruntime's POSIX build carries a telemetry client that batches events and
  uploads them on a thread of its own; at exit the static teardown of the log
  manager frees a mutex that an upload still in flight then tries to lock, which
  fails with `EINVAL` and becomes an uncaught `std::system_error`. It is a race
  against a network round-trip, which is why it was intermittent and why no single
  test directory ever showed it alone: the upload does not leave until about nine
  and a half seconds after the first `import onnxruntime`, and a short run has
  already exited. `tests/conftest.py` sets `ORT_DISABLE_TELEMETRY=1` before anything
  imports onnxruntime (commit `f1dd076`), which is the only point at which the
  setting is read.

  Re-measured here, interleaved, 40 runs per arm against trees identical but for
  that one line: **0 aborts with the variable set, 4 without.** Pooling every sample
  taken in this project puts the untreated rate near one run in ten and the treated
  rate at zero in eighty; the earlier "three in ten" was a nineteen-run sample. The
  rate moves with machine load and network latency, so the durable statement is that
  it was frequent enough to matter and is now absent. Read the status separately from
  the summary line regardless — `pytest -q > out 2>&1; rc=$?`, never
  `pytest -q | tail -1`, which reports `tail`'s status and hides an abort
  completely.
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

**What v1 does not have yet.** The half of the project that needs real weights has
still only ever been exercised here: CI downloads nothing, so every `models`-marked
test skips on a runner, and CoreML, the timings and the score distributions above
are single-machine facts. A green tick also belongs only to the revision that earned
it, and `ci.yml` has been edited since `51aba8a` without going through a runner
again. The opacity recalibration has no test that can detect it being undone —
restoring the old constants leaves the suite green, which is recorded in `todo.md`.
The README has no screenshot, which is a strange omission on a page about a visual
tool. Nothing is tagged, nothing is on PyPI, and the repository is still private. A
handful of smaller loose ends are carried in `todo.md`, which is the checklist.

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
