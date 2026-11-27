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

## Shipped

### v4 — the Unitree G1 · 4.0.0 · 2026-11-27

**Status: shipped in 4.0.0.** The same motion on Unitree's 29-joint G1, beside the
clip and its 3D skeleton. The model is fetched at runtime from a pinned commit
and verified per file; forward kinematics agree with MuJoCo. Retargeting matches
bone directions on the robot's own proportions, inside its limits, and checks
the result against its motors' speed.

What getting it right took, recorded because each was found the hard way on the
reference clip: a regularisation weight that was a bias rather than a smoother
(15.6 cm off a pose the robot can make); an elbow that travels past straight,
where the solver found a hyperextended mimic and snapped out of it at 36 rad/s;
a wrist point that was not rigid on the forearm; foot points from two
independent directions that a rigid foot cannot meet at once; targets blinking
at a hard confidence cut-off; and a three-frame tracking glitch only a trajectory
filter removes. Each spike frame was attributed before anything was changed.

`core/h36m.py` still has no caller: the retargeting reads COCO-WholeBody directly,
which carries the toes, heels and hand landmarks H36M-17 lacks.

### v3 — 3D · 3.0.0 · 2026-11-27

**Status: shipped in 3.0.0.** RTMW3D runs as the `3d` preset, `keypoints_3d` is
filled with a metric, root-relative skeleton, an orbitable panel shows it beside
the picture, and glTF exports it. What the plan below got right: the topology
really is identical, and the layer tree, presets and selection carried over
untouched. What it did not say: MMPose publishes RTMW3D only as a PyTorch
checkpoint, so the weights are a pinned third-party ONNX export of it; the depth
decoding had to be checked against real frames, because rtmlib's differs; and
the depth alone is not a 3D skeleton — the pixel offsets need a metric scale,
which `core.lifting` recovers from the torso. `core/h36m.py` still has no caller;
the G1 retargeting in v4 is where it gets one.


**What it is.** RTMW3D in place of RTMW, `keypoints_3d` and `space_3d` populated,
and an orbitable skeleton beside the image — the same pose, seen from somewhere
the camera was not.

**Why it exists.** It is the point of choosing this model family.

**Why it is cheap.** RTMW3D emits the **same 133-keypoint topology** with a third
coordinate. Not a similar skeleton, not a remapping: the same indices, in the same
order, with a z. So the layer panel, the render groups, the presets, `group_of`,
the side-colour convention, the selection and solo logic and the confidence
threshold all carry over unchanged, and the new work is the viewport rather than
the model of what is being drawn. `keypoints_3d` and `space_3d` are already on
`PersonPose`, and `Space` already names `CAMERA_M` and `ROOT_REL`. `core/h36m.py`
ships in v1 with nothing calling it: it maps COCO-WholeBody to H36M-17, synthesising
pelvis, thorax, spine and head, and every 2D→3D lifting model plugs in at exactly
that seam. It is 93 lines, most of them the reasoning written down beside
the constants, and putting it in v1 forced the coordinate conventions to be settled
while they were free to change: 2D is source-image pixels with the origin top-left
and y down; 3D will be camera-space, right-handed, metres, OpenCV convention,
converted to y-up only at glTF export.

`pyqtgraph` and `PyOpenGL` are already declared as the optional `threed` extra.
Qt3D is deprecated and QtQuick3D and QtGraphs are commercial-or-GPLv3, so neither
can ever be imported here without relicensing the distributed application.

---

### v2 — video · 2.0.0 · 2026-11-27

**Status: shipped in 2.0.0.** Video opens, analyses in
the background with a disk cache, plays in real time and scrubs by frame index;
people are tracked and smoothed at draw time; the webcam runs live; and
`alchemypose video` does the same headless. One design decision went the other way
from the plan below: scrubbing does *not* run on the latest-request-wins worker.
The worker drops frames by design, and playback beside a 3D skeleton and a robot
needs every one, so a clip gets an offline pass that analyses each frame once and
playback reads from that. The worker keeps its policy for stills and the webcam,
where dropping a stale frame is the right call. Measured on Linux with an RTX 4070
laptop GPU: the 300-frame reference clip analyses in 22 s on `performance`.

**What it is.** Open a video instead of a photograph: a scrubber under the canvas,
the overlay redrawn per frame, `track_id` populated so a person keeps their
identity across frames, and temporal smoothing so the skeleton stops shivering.
Webcam arrives with the same work.

**Why it exists.** A still image proves the model; a video is what makes a pose
tool useful and what makes it look like anything in a screen recording. It is also
where the interface question v1 is built around — *what do you show, out of 133
points per person* — stops being a nicety, because a face-full of dots flickering
frame to frame is unreadable in a way a still never is.

**Why it is cheap.** The result schema was written for it and has been carrying the
empty seats since its first commit. `PersonPose.track_id` is already there,
serialised as `null` on every person in every export v1 has ever written — so
filling it in changes no schema, no reader and no file format. `PoseFrame` already
carries `frame_index` and `timestamp_s`, which are 0 and `None` for a still, and
`PoseResult.frames` is already a tuple that happens to have one element rather than
a single frame. The worker is already latest-request-wins, which *is* scrub
semantics: a new frame supersedes any queued frame that has not started. The layer
tree, the presets, the confidence slider and the export functions all take a
`PoseResult` and neither know nor care how many frames are in it. Reading frames is
the one genuinely new dependency — PyAV, for frame-accurate seeking, which is part
of why the project targets Python 3.12 rather than 3.10.

The per-model peak scale matters more here than it did in v1. Per-image score
normalisation was rejected partly because it would make a keypoint's confidence
depend on who else is in the frame, and flicker frame to frame in exactly this
version.

### v1.1 — the v1 leftovers · 1.1.0 · 2026-11-27

No new capability: the open items `todo.md` carried out of 1.0.0, closed. Tests
that can fail now guard the detector's person filter, its defensive NMS and the
confidence-cue calibration (each seen red against its own revert); the README's
facts are tested; the photograph dims on selection and new results draw in; a
peakless keypoint lands on the image origin; the peak scale moved onto
`ModelSpec`. The window was opened on Linux for the first time, and the README
gained its screenshot.

### v1 — single image · 1.0.0 · 2026-09-03

A photograph goes in; every person in it comes back with 133 keypoints, drawn as
an interactive skeleton whose parts can be shown, hidden, isolated and inspected.

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
- **A CLI that never touches Qt** — `alchemypose image`, `models list|fetch|clear`
  — under a bare `alchemypose` that opens the window when it is given nothing to do.
- **Face level of detail** — the 68 face points resolve from contours, to
  contours coloured per region, to the individual points, on thresholds of 80 px
  and 200 px of face measured *on screen* rather than in the photograph. The
  canvas announces every change of scale and the overlay rebuilds only when one
  of the two thresholds is crossed. The **Face** chip magnifies the face on the
  same canvas, which is what a face mode would have been for.
- **Export** — keypoints as JSON or CSV, the overlay as a PNG rendered from the
  scene at the source photograph's own resolution, so it is the same file
  whatever the zoom and the window size happen to be.
- **A wheel** — the package and `py.typed`, no weights and no fixtures.

**Verified: what was actually run, and what came back.** Everything in this list
was measured on the release tree rather than carried over from a draft. The
machine is one Apple M5 laptop — macOS 26.6.2, Python 3.12.7, onnxruntime 1.29.0,
PySide6 6.11.2, numpy 2.5.2 — and the runner is `ubuntu-latest`, which has no
weights and no display.

- **492 tests pass here**, with cached weights and CoreML in play: 303 unit, 179
  GUI, 10 integration. On a machine with no weights the same command reports
  **480 passed, 12 skipped** — those 12 are the `models`-marked tests, 2 in
  `tests/unit` and 10 in `tests/integration`, which check for the files and skip
  rather than downloading them. `ruff check src tests main.py` passes; `mypy`
  reports no issues across 38 source files, and `mypy main.py` no issues in 1.
  Treat the total as a reading rather than a constant — it moves whenever a test
  is added.

  The GUI tests do not need `QT_QPA_PLATFORM` on the command line, which older
  wording here implied. `tests/gui/conftest.py` sets it itself, at import, before a
  widget can exist, so a bare `pytest` runs them headless too; passing it explicitly
  is belt and braces and is what the CI step does.

- **CI has run six times and passed six times**, every one a push to `main`. The
  workflow went in with `e9aebc2` and GitHub registered and ran it on that push.

  | Run | Commit | Started (UTC) | Result | Duration |
  |---|---|---|---|---|
  | 33684801971 | `e9aebc2` | 2026-09-02 21:22 | success | 55s |
  | 33685597889 | `88a659e` | 2026-09-02 21:31 | success | 1m 02s |
  | 33700786465 | `51aba8a` | 2026-09-03 00:44 | success | 58s |
  | 33735124492 | `03a701f` | 2026-09-03 08:45 | success | 1m 01s |
  | 33738463693 | `4ba3cdb` | 2026-09-03 09:22 | success | 59s |
  | 33751784998 | `f3d42ab` | 2026-09-03 11:49 | success | 1m 03s |

  Read out of the most recent log rather than off the tick: `ruff check` clean;
  `mypy` "Success: no issues found in 38 source files" and then the same for
  `main.py`; `pytest tests/unit` 289 passed and 2 skipped; `pytest tests/gui` 179
  passed; the weights-only step 10 skipped. The probe step ahead of the GUI tests
  printed `Qt 6.11.2 on platform offscreen`. That is the line that settles the one
  part of the workflow which was reasoning rather than observation: the list of
  Linux system libraries the PySide6 wheel links against was read out of the
  wheel's ELF headers, and a runner has now loaded the offscreen plugin with
  exactly that list installed and nothing more. `ci.yml` was last edited in
  `4ba3cdb`, and the last two runs are on that revision or later, so the workflow
  as it stands is the workflow that has been through a runner.

- **The CLI produces real output.** `alchemypose image
  tests/integration/fixtures/two_people.jpg --preset performance` finds 3 people in
  a 982×670 photograph and writes a 400-line CSV — one header plus 399 rows,
  3 × 133 — and a JSON document carrying `track_id: null` and `keypoints_3d: null`
  on every person, which is the v2/v3 seam sitting there doing nothing on purpose.
  The same image on `--preset lite` finds 2.

- **Confidence is a real 0–1 quantity.** On `performance` the fixture's 399 scores
  run min 0.094, median 0.490, max 0.858, and the drawn keypoint count falls
  399 · 335 · 247 · 153 · 28 · 0 as the threshold moves 0.0 · 0.2 · 0.4 · 0.6 ·
  0.8 · 0.9. Before the peak-scale fix the same scores ran up to 8.576 and the
  slider hid two of them across its whole range.

- **Timings, measured warm on that fixture** — median of 15 runs each, after three
  warm-up passes, in one process.

  | Preset | People found | Detection | Pose, per person | End to end |
  |---|---|---|---|---|
  | `lite` | 2 | 7.6 ms | 4.5 ms | 16.9 ms |
  | `balanced` | 3 | 15.3 ms | 14.2 ms | 57.9 ms |
  | `performance` | 3 | 42.1 ms | 24.5 ms | 113.9 ms |

  Those are warm numbers on purpose. In a fresh process, opening the models costs
  1.1 s on `lite` and 2.9 s on `performance` before the first inference — session
  creation, the CoreML compile, and the throwaway run that proves the provider can
  execute the graph — and the first inference itself is then about 25% slower than
  the ones after it (20.8 ms against 16.3 ms on `lite`, 132.2 ms against 107.9 ms
  on `performance`).

- **The library half is genuinely headless.** `import alchemypose` loads zero
  `PySide6` modules, asserted by a subprocess test; running the full engine in a
  subprocess also loads zero, checked again for this release.

- **The founding use case draws correctly.** "Right hand and left leg" renders 22
  bones with the thigh attached; every render group renders connected, with zero
  isolated dots.

- **Six model files, 464 MB, downloaded and hash-verified for real** — 63 MB for
  `lite`, 148 MB for `balanced`, 253 MB for `performance`, which is what the three
  presets cost on disk and what they add up to.

- **The artefact is what it claims to be.** Built from the release tree, the wheel
  is 79 KB over 45 entries — the 38 source files, `py.typed` and the dist-info —
  with no `.onnx`, `.jpg` or `.png` anywhere in it, and the sdist is 74 KB with no
  tests and no fixtures. `twine check` passes on both.

**What v1 does not have.** The half of the project that needs real weights has
only ever been exercised on the machine above: CI downloads nothing, so every
`models`-marked test skips on a runner, and CoreML, the timings and the score
distributions are single-machine facts. The window has never been opened on
Windows or on Linux — the GUI suite runs headless on a runner, which proves the
widgets construct and behave, not that the thing looks right on a desktop. Three
fixes have no test that can detect them being undone: the opacity recalibration,
the detector's person-class filter and its defensive `nms()` call, all recorded in
`todo.md`. The README has no screenshot, which is a strange omission on a page
about a visual tool. Nothing is video and nothing is 3D.

---

## Planned

### v4.1 — a robot-motion export

Root pose, the 29 joint angles, velocities and foot contacts, keyed by joint name,
in a layout MuJoCo and Isaac Lab trainers load, with a MuJoCo kinematic playback
check — the file a policy is trained from. Root travel and ground contact are
estimates from one camera, and labelled as such.

## Decisions that shaped v1

| | Alternative | Chosen, and why |
|---|---|---|
| Pose model | Ultralytics YOLO-pose | **RTMW**, Apache-2.0. Ultralytics is AGPL-3.0, which extends to weights, and offers 17 body points — no face, no hands |
| Bindings | PyQt6 | **PySide6_Essentials**, LGPL and dynamically loaded. Its macOS wheel is 106 MB against 423 MB for Essentials plus Addons |
| SimCC decoding | depend on `rtmlib` | **vendored**, 116 lines. `rtmlib` pulls in both `opencv-python` and `opencv-contrib-python`, whose bundled Qt plugins conflict with PySide6 |
| Render groups | a partition of the 133 indices | **a cover** — limbs own their anchor joint, or a selected leg draws as a shin |
| Keypoint scores | the raw SimCC peak, as the references ship it | **divided by a measured per-model peak scale**, so a 0–1 threshold means something |
| Execution providers | CoreML for everything | **CPU for the detector, CoreML for the pose model** — forced; CoreML loads the YOLOX graph and then fails the first `run()` on it |
| Weights | vendored in the wheel | **fetched at runtime, SHA-256 verified**, following AlchemyFace |
| Bare `alchemypose` | print the help | **open the window** — it is what someone types first, and every subcommand still reaches its own command |
| Editing keypoints | in scope | **out** — the moment a joint can be dragged this becomes a second AlchemyAnnotate |
