# Changelog

All notable changes to this project are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project follows [Semantic Versioning](https://semver.org/).

## [Unreleased]

Nothing yet. v2 is video — upload and scrub, per-frame overlay, tracking,
temporal smoothing. `versions.md` says why it is cheap.

## [1.0.0] — 2026-09-03

The first release, and the whole of v1: a photograph goes in, and every person
in it comes back with 133 COCO-WholeBody keypoints on a canvas whose parts can
be shown, hidden, isolated and inspected. Everything below shipped in it, so
this entry is also the entire history of the project — every commit to date sits
on `main`, from the scaffold to here.

The layering was deliberate and is enforced rather than intended. `core/` —
topology, schema, presets, serialisation, the H36M mapping — imports neither Qt
nor onnxruntime. `inference/` imports no Qt, and is driven end to end by a CLI
that runs in a process with zero Qt modules loaded. Qt appears only under `gui/`.
The point was that the window would have nothing left to prove but itself.

It earned its keep in an unexpected direction. The two worst defects below are
*inference* defects, and neither produced an error, an exception or a failing
test. Both were found by rendering the output and looking at it — which is the
defect class the whole test strategy cannot catch, and worth saying plainly. One
of the two never reached a commit, which is why it is recorded under Notes rather
than Fixed: nothing in this repository was ever broken by it.

### Added

- **`alchemypose.core` — the data model, importing neither Qt nor onnxruntime.**
  `PersonPose` / `PoseFrame` / `PoseResult`, the COCO-WholeBody 133 topology
  (133 named keypoints, 127 edges, 14 render groups), nine named presets, JSON
  and CSV serialisation, and a COCO→H36M-17 mapping. All 133 keypoints are
  always stored; visibility is a `KeypointSelection` applied at render time and
  never filters the data. Coordinates are kept as raw source-image pixels beside
  `image_size`, never pre-normalised.

  `core/h36m.py` ships with nothing calling it. It exists so that the coordinate
  conventions — 2D in source pixels, origin top-left, y down; 3D to be
  camera-space, right-handed, metres, OpenCV convention — are settled now, while
  changing them is free. Its neck joint interpolates a quarter of the way from
  the thorax toward the nose rather than half, because the midpoint lands at chin
  level; the reasoning is recorded in the source so the constant is not later
  mistaken for a measurement.

- **`alchemypose.inference` — ONNX Runtime only, no PyTorch.** YOLOX person
  detection, RTMW pose estimation, vendored SimCC decoding and top-down affine
  cropping, and a weight registry that downloads on first use into
  `~/Library/Caches/alchemypose` and verifies a recorded SHA-256. Nothing is
  vendored in the repository or the wheel. Every openmmlab artefact turns out to
  be a zip, so the recorded hash is of the *extracted* `.onnx`, and a download is
  verified before it is published into the cache rather than after.

  The detector runs on the CPU execution provider and the pose model on CoreML.
  That is not a tuning preference, and the reason is narrower than "CoreML cannot
  load that graph". It loads it: `InferenceSession` returns normally for all three
  YOLOX exports and `get_providers()` reports CoreML ahead of CPU. The failure
  comes on the first `run()`, inside the partition holding the graph's baked-in
  NMS — a `GetStaticOutputShape` rank mismatch between a CoreML static shape of
  `{1,1,1,3549,3549}` and an inferred `{1,3549}`. So the split is genuinely forced,
  and what has to be guarded against is an *execution* failure rather than a load
  failure: a CoreML session is proved with one throwaway inference on zeros before
  it is handed back, and demoted to CPU if that raises.

- **`inference/worker.py`** — a plain `threading.Thread` over a `queue.Queue`
  with latest-request-wins semantics, and no Qt import anywhere in it, so it is
  tested headless. `gui/bridge.py` is the only adapter between it and Qt signals.

- **`alchemypose` CLI** — the bare command, plus `gui`, `image`,
  `models list|fetch|clear`, `version` and `--version`. Typing `alchemypose` with
  no subcommand opens the window, because that is what someone types straight
  after installing it and a page of help is not what they came for; a subcommand,
  and a *mistyped* subcommand, both still reach the command they named.
  `alchemypose image photo.jpg --preset performance --json out.json --csv out.csv`
  runs the whole pipeline with no display and no Qt in the process.

- **The desktop window.** A dark `QGraphicsView` canvas with zoom and pan, the
  skeleton drawn as gradient bones under constant-size joints, a tri-state layer
  tree, preset chips, person chips, a confidence slider, a hover inspector and a
  left/right legend.

  **The layer tree is the single source of truth for what is visible.** Preset
  chips *write* to the tree; canvas gestures *write* to the tree; the overlay only
  ever mirrors what the tree reports back. Nothing changes what is drawn without
  the tree reflecting it. The legend says "person's left", not "left", because
  image-left is the opposite.

- **Selection and solo are separate gestures**, following Blender rather than the
  annotation tools: a click emphasises a group and dims the rest, a double-click
  or `Shift+H` hides everything else, and `Esc` / `Alt+H` restores. Solo records
  the tree state it replaced so there is something to return to.

- **Face level of detail, driven by the face's size on screen rather than in the
  photograph.** Sixty-eight dots at fit-to-window are not detail, they are a grey
  smudge with a jaw around it, so `gui/canvas/lod.py` picks one of three
  renderings from the longest on-screen edge of the largest drawn face:
  `CONTOURS` below 80 px, `REGIONS` from 80 px, `POINTS` from 200 px. The
  contours are drawn at every level — it is the individual keypoint items that
  are skipped below `POINTS`, which is also the only level where they are worth
  making hoverable, since at 200 px across adjacent dlib-68 points sit about
  8–12 px apart against the 11 px hit radius a joint claims. From `REGIONS` up,
  jaw, brows, nose, eyes and lips take their own colours in place of the theme's
  single face grey, in hues chosen to avoid the green and orange the legend
  reserves for the person's left and right.

  `PoseCanvas` gained `zoom_changed`, emitted from `zoom_by`, `fit_to_window` and
  `zoom_to_rect`, and the overlay rebuilds only when a zoom actually crosses one
  of the two thresholds — a rebuild is every item in the scene, and a wheel event
  arrives dozens of times a second. A face with no extent at all is treated as
  unmeasured rather than small and keeps its points: contours through coincident
  points are zero-length strokes, so thinning there would leave nothing at all.
  `lod.py` imports no Qt, so the thresholds are tested without a widget.

  The **Face** chip also calls `zoom_to_rect` on the face bounds, so it delivers
  the magnification a separate face mode would have given, on the same canvas and
  with no second viewport. Clicking it in the seconds before a detection returns
  is harmless: the bounds are empty then, and `zoom_to_rect` ignores an empty
  rectangle.

- **Export, as three plain functions under an Export menu.** `gui/export.py`
  offers `export_json`, `export_csv` and `export_png`, each taking what it needs
  and a path — no dialog, no window, no status bar — which is what lets the whole
  of export be tested without a modal. The PNG renders the **scene**, not the
  widget: rendering the widget would bake in the zoom, the scroll position, the
  window's aspect ratio and the empty canvas either side of a portrait, whereas
  the scene is the photograph plus the overlay in image-pixel coordinates, so
  1:1 gives back the source image's own resolution and the same file whatever
  the view is doing.

  Nothing that can go wrong in an export is a bug — there may be no result yet,
  the user may cancel, the folder they typed may not exist, the image may be too
  large to allocate — so each of those ends as a sentence in the status bar or a
  reported error rather than a traceback. A PNG needs only a photograph; JSON and
  CSV need a result. The suggested filename follows the photograph's own stem.
  The save dialog sits behind a `SavePrompt` protocol and is injected, for the
  same reason `Reporter` is: no test opens one.

- **`track_id`, `keypoints_3d` and `space_3d` exist on `PersonPose` from the first
  commit of the schema (`3c6d490`), unpopulated**, as do `PoseFrame.frame_index`
  and `timestamp_s`. They cost nothing now and they are what makes v2 and v3
  additive instead of a schema migration. See `versions.md`.

- **Packaging, so the thing can be installed rather than cloned.** A wheel and an
  sdist that carry the package and `py.typed` and nothing else: measured on the
  1.0.0 artefacts, the wheel is 79 KB across 45 entries with no `.onnx`, `.jpg`
  or `.png` in it, and the sdist is 74 KB with no tests and no fixtures. The
  version has one home — `alchemypose.__version__`, read out of the source by
  setuptools' `attr:` without importing anything — so `pip show`, `alchemypose
  --version` and the package cannot disagree. `twine check` passes on both files.

- **A README that describes what exists**, and a CI workflow. The README covers the
  window and its gestures, the CLI with transcripts taken from real runs, and what a
  confidence score means; its timing table says what it measures — the pose model
  alone, per detected person, with the detector's per-frame cost stated separately —
  rather than leaving the reader to guess. CI runs ruff, mypy, the unit tests, the
  GUI suite under `QT_QPA_PLATFORM=offscreen`, and last the weights-only tests that
  have nothing to run on a runner and say so. Python 3.12 only. No step on a runner
  downloads weights, so a push can never pull the 464 MB of ONNX files; the cache
  check inside the `models`-marked tests is the backstop, not the guarantee.

  **That workflow has run, and it passed.** It went in with `e9aebc2` and GitHub
  registered it on that same push. Six runs have completed, all pushes to `main`,
  all successful — `e9aebc2` (55s), `88a659e` (62s), `51aba8a` (58s), `03a701f`
  (61s), `4ba3cdb` (59s) and `f3d42ab` (63s), on `ubuntu-latest`. `ci.yml` was last
  edited in `4ba3cdb`, and two of those runs are on it or later, so the workflow as
  it stands is the workflow that has been exercised. The last run logged `ruff`
  clean, `mypy` reporting no issues across 38 source files and then across `main.py`,
  289 passed and 2 skipped in `tests/unit`, 179 passed in `tests/gui`, and 10 skipped
  in the weights-only step — 480 tests as of that commit, fewer than the local totals
  below, because tests have been added since. The probe step ahead of the GUI tests
  printed `Qt 6.11.2 on platform offscreen`. That last line is the one that mattered:
  the list of system libraries the PySide6 wheel links against was derived by reading
  the wheel's ELF headers, and it was reasoning rather than observation until a runner
  loaded the offscreen plugin with exactly that list installed and nothing more.

  One thing it still cannot speak to. CI never touches the weights, so CoreML, the
  timings and every score distribution recorded here remain single-machine facts,
  measured on one Apple-silicon laptop. Separately, the README still has no
  screenshot, on a project whose stated goal is that it look striking in one.

### Fixed

- **Render groups were a strict partition, so selecting one limb drew only its
  distal segment.** A bone is drawn only when both of its endpoints are visible.
  With `left_leg` owning exactly the knee and the ankle, "show me the left leg"
  drew the shin and nothing else — the thigh needs the hip, and the hip lived in
  `torso_head`. `left_foot` was worse: all three of its bones originate at the
  ankle, so it rendered as three isolated dots and zero bones. And no edge
  connected either wrist to its hand root, so the hands were detached from the
  arms even in the show-everything view.

  Groups are now a **cover, not a partition**: each limb and extremity owns its
  proximal anchor joint (`left_arm` 5, 7, 9; `left_leg` 11, 13, 15; `left_foot`
  15, 17, 18, 19, and the mirrors), and two connector edges join wrist 9 → 91 and
  wrist 10 → 112. Edge count 125 → 127. The use case the whole interface is built
  around — show only the right hand and the left leg — went from 21 bones with a
  floating shin to 22 with the thigh attached. `group_of` gained an explicit
  priority order so a shared joint resolves to the limb, which is why a shoulder
  reads as person's-left green rather than midline cyan.

  The fix broke something on the way in, which is worth recording: the broad
  groups (`upper`, `lower`, `left_side`, `right_side`) are built by concatenating
  render groups, and that union was sorted but not deduplicated. Harmless while
  the groups were disjoint; the moment they overlapped it emitted the shoulders
  and ankles twice. Caught before the fix was accepted, not after.

- **The exported YOLOX graph is not the head that was assumed.** It bakes in a
  class-aware `NonMaxSuppression` followed by a `TopK`, and emits `dets (1, N, 5)`
  as xyxy + score in letterboxed pixel space plus `labels (1, N)` — not the raw
  cx/cy/w/h/objectness output a first guess reaches for, and nothing to decode
  from anchors or grids. `N` is whatever the baked-in `TopK` leaves behind:
  the graph declares it as a dynamic dimension (`Gatherdets_dim_1`) and it moves
  with the image and with the export — 39 rows for `yolox_nano` on the fixture at
  416×416, 12 for `yolox_tiny`, 10 for `yolox_s` at 640×640, and 1 for an all-zeros
  input. It is not a count of real objects either, so most rows are near-zero-score
  padding dropped by the score threshold. The hand-rolled `nms()` written for it was
  solving an already-solved problem; it stays as a tested, public helper used
  defensively in case a future export does not suppress duplicates itself, but it
  is not on the hot path. Unlike most detectors, YOLOX also genuinely does want
  raw BGR with no normalisation — verified against the graph, not assumed.

- **SimCC peaks are unnormalised, so the confidence slider did nothing.**
  `KLDiscretLoss` trains `log_softmax(pred * beta)` against
  `softmax(target * label_beta)`, and softmax is shift-invariant, so a confident
  peak converges on roughly `label_beta / beta` raw units above background —
  about 1 for the beta=10 models and about 10 for RTMW-l. On the integration
  fixture with the `performance` preset, scores ran min 0.938, median 4.898, max
  8.576, and the 0–1 slider hid **2 keypoints out of 399 across its entire
  range**. The default 0.3 threshold was hiding nothing at all.

  This is not a missing step relative to the references — it is a defect the
  references share. MMPose's `get_simcc_maximum` applies no softmax by default,
  and rtmlib ships `kpt_thr=0.43` on the very same RTMW-l weights, where it is
  just as inert. Softmax was tried and rejected (peak probability ~0.005 at
  temperature 1 for the beta=10 models — flat), as was MMPose's visibility recipe
  (median 0.9975: a presence flag, not a confidence) and per-image normalisation
  (it would inflate a garbage frame to 1.0 and flicker between v2 video frames).

  Scores are now divided by a **measured per-model peak scale** and clipped to
  [0, 1] — 10.0 for `rtmw_l_384x288`, 1.0 for the others. An unregistered pose
  model raises rather than silently assuming 1.0, since a silent default is how
  this went unnoticed. Keypoint positions are bit-identical before and after
  (`np.array_equal` across all five people on both models); only the scores moved.
  Re-measured for this release on the same fixture, the `performance` preset's 399
  scores run min 0.094, median 0.490, max 0.858, and the same sweep draws 399, 335,
  247, 153, 28 and 0 keypoints at thresholds 0.0, 0.2, 0.4, 0.6, 0.8 and 0.9.

  The scale table rests on measurement rather than on published configs, and one
  reason is recorded rather than hidden: RTMW-m's config says `beta=1`, but the
  shipped ONNX — a DWPose-style distillation — behaves as `beta=10`. The config is
  not the recipe that produced those weights.

- **`PoseCanvas.set_image` called `scene.clear()`, destroying items the overlay
  still held.** The next selection change raised
  `libshiboken: Internal C++ object (KeypointItem) already deleted`. It had been
  unreachable only because nothing called `set_image` yet; wiring up the Open
  action is exactly what makes it reachable. The canvas now emits `scene_cleared`
  immediately *before* wiping the scene, and the overlay drops its item lists
  without calling `removeItem` — the scene is about to delete them itself, and
  doing both is a double free.

  Dropping the items was not sufficient. The overlay's cached `PoseResult` still
  belonged to the previous photograph, so a layer toggle in the window between
  opening image B and B's inference returning redrew **image A's skeleton over
  image B**. `_forget_items` now clears the result too, and a regression test
  asserts zero items in that window rather than 42.

  Two competing dimming mechanisms were collapsed in the same change. The
  selection controller had been writing opacity onto items directly from outside
  the overlay, which any rebuild then silently discarded; a single
  `PoseOverlay.set_emphasis` now owns it, and every item's opacity is derived from
  the overlay's own fields, so a rebuild *reproduces* the visual state instead of
  forgetting it.

- **The person chips were being discarded by the layer tree.** `SelectionController`
  wrote the tree's raw selection straight to the overlay, so soloing a group while
  one person was selected quietly put everybody back on screen while the chip still
  claimed one. Visibility now has exactly one writer: the controller writes to the
  tree and stops there, the tree announces, and the window composes both halves —
  which groups, and which person — in `_effective_selection` before anything reaches
  the canvas.

- **mypy had been checking nothing.** `[tool.mypy] python_version` was `"3.11"`
  while the installed numpy 2.5.2 ships stubs written with PEP 695 syntax, so
  mypy aborted on the stubs before it reached a single project file. Every "mypy
  clean" claim before that point was a no-op run. The virtualenv, `requires-python`,
  the ruff target and the mypy target now all say 3.12. Dropping 3.11 costs
  nothing — PyAV, which v2 needs, requires ≥3.11 anyway.

  It then took until the GUI package existed for mypy to actually go green: a
  `TYPE_CHECKING` forward reference to `alchemypose.gui.app` had been erroring
  since the project scaffold, and for most of that time nobody could see it.

- Smaller, but real: `_extract_onnx` leaked a file descriptor on one error path —
  `mkstemp` hands back a raw fd, and it was only adopted by the *second* of two
  context managers, so an `archive.open` failure in the first left it open for the
  life of the process. Introduced by an earlier fix in the same function, which is
  the usual way.

### Notes

- **492 tests, all of them passing**, measured on the release tree with weights
  cached. 313 run with no display at all (303 unit, 10 integration); 179 are the
  GUI suite, which sets `QT_QPA_PLATFORM=offscreen` for itself in
  `tests/gui/conftest.py` rather than requiring it on the command line; 12 of the
  492 additionally need cached weights, and on a machine without them the run is
  480 passed and 12 skipped. `ruff check src tests main.py` is clean, and `mypy`
  reports zero errors across the 38 source files and then across `main.py`, which
  needs naming separately because it sits outside `src/`. The total is a reading
  taken at the release, not a fixed property — it climbs as tests are added.
- **The suite used to pass and then abort, and that is now fixed.** Every test would
  report green, the run would print its pass line, and then the process would die
  during interpreter shutdown with `libc++abi: terminating due to uncaught exception
  of type std::__1::system_error: recursive_mutex lock failed: Invalid argument`,
  leaving pytest with exit status 134 — a teardown fault rather than a failing
  assertion, but a red build to anything that reads an exit code, which is every CI
  system there is.

  The cause is onnxruntime's telemetry client. Its POSIX build batches events and
  uploads them over HTTP on a thread of its own, and the static destructor that tears
  the log manager down at exit frees a `recursive_mutex` that an upload still in
  flight then locks; the lock fails with `EINVAL`, libc++ turns that into an uncaught
  `std::system_error`, and the process takes SIGABRT. That makes it a race against a
  network round-trip, which explains both the intermittency and why no single test
  directory ever reproduced it: the upload does not leave until about nine and a half
  seconds after the first `import onnxruntime`, and a short run has exited before then.
  A whole-suite run lands squarely in that window.

  The delay is observed rather than inferred. Watching a bare
  `import onnxruntime; sleep` process with `lsof -a -p <pid> -i` shows it open exactly
  one outbound socket — a single TCP connection to port 443 on an address registered
  to Microsoft Corporation — first visible 9.5 to 9.6 seconds after the import, on
  every one of four runs. With `ORT_DISABLE_TELEMETRY=1` set, the same process opens
  no socket at all in twenty-five seconds, on every one of three runs. So the
  variable does not merely quieten the logging; it stops the connection being made.

  `tests/conftest.py` sets `ORT_DISABLE_TELEMETRY=1` at import, ahead of every test
  module and therefore ahead of the first `import onnxruntime`, which is the only
  point at which onnxruntime reads it (commit `f1dd076`). It uses `setdefault`, so a
  run that wants telemetry back can ask for it in the environment.
  `onnxruntime.disable_telemetry_events()` is not a substitute: it stops later events
  being logged but leaves the manager and its uploader running.

  Measured interleaved, 40 runs per arm against trees identical but for that one
  line: **0 aborts with the variable set, 4 without.** Pooling that with every
  earlier sample taken in this project — 1 in 12, 1 in 20, 2 in 28, 3 in 19, and the
  5 in 40 recorded on `f1dd076` — gives 16 aborts in 159 untreated runs, about one
  in ten, against 0 in 80 treated ones. The headline figure this entry used to carry,
  roughly two runs in ten, came from the nineteen-run sample alone and was high. The
  rate moves with machine load and network latency, so the durable claim is the
  contrast, not the percentage: frequent enough to matter, and now absent. Measure
  any run as `pytest -q > out 2>&1; rc=$?` in any case — `pytest -q | tail -1`
  reports `tail`'s status and hides an abort completely.
- **The import boundary is enforced, not just intended.** A subprocess test
  asserts that `import alchemypose` loads zero `PySide6` modules, and running the
  full engine headless was measured at 0 Qt modules loaded. `core/` imports
  neither Qt nor onnxruntime.
- **RTMW nearly shipped fed with raw 0–255 BGR, and nothing would have said so.**
  Not a fix — there is no bad revision to bisect to, because the first commit of
  `rtmw.py` already normalises correctly — but the trap is real enough to record.
  The exported graph has no `Sub`/`Div` ahead of its first convolution, which
  reads like "wants raw pixels". It does not: it wants ImageNet mean/std
  normalisation in **RGB** order. The first draft handed it the raw BGR array
  straight from OpenCV, and got no error, no warning, and 133 confident, in-range
  keypoints in anatomically impossible positions — hips above shoulders on a
  photograph of two people standing up. Nothing in the pipeline could tell. It
  was caught by drawing the overlay and looking at it, before the file was
  committed, and the next graph exported this way will look exactly as innocent.

  The regression test came later and could not use the integration fixture: the
  only one is a monochrome 1935 archival photograph, on which a BGR/RGB swap is
  arithmetically a no-op — the three channels are identical to the last unit.
  It uses synthetic pixels with three distinct channel values, and its expected
  numbers are independent literals rather than the normalisation formula
  re-implemented from the test's own constants — an earlier draft did exactly
  that, and a wrong constant would have cancelled out on both sides and passed.

- The opacity constants shipped wrong for a while and are worth flagging.
  `DASH_BELOW = 0.5` and `FULL_OPACITY_ABOVE = 0.8` were chosen while scores were
  unbounded logits, so against honest confidences they dashed 84% of the default
  preset's bones — 51 of the 61 it draws on the fixture — and let not one of them
  reach full opacity. A cue that applies to everything is not a cue. They are now
  0.25 and 0.55, calibrated against measured score distributions on both models,
  and the source says to re-measure rather than re-tune by eye if the score
  pipeline changes again.
- **Fixes are checked by reverting them and confirming the new test goes red, and
  the record of which ones is worth keeping honest.** Two tests were found unable
  to fail: the channel-order guard described above, which was rewritten, and one
  that asserted a preset chip toggles itself — disproved with a standalone script
  (a checkable `QToolButton` does toggle itself on click) and deleted rather than
  kept as a false assertion.

  The peak-scale fix was the one the claim did not cover. Dropping
  `PEAK_SCALE["rtmw_l_384x288"]` back to 1.0 left the whole integration suite
  green while the confidence slider went inert again, because the only assertions
  within reach were that scores land in `[0, 1]` — which the clip satisfies — and
  that *some* threshold keeps a strict subset, which 397 scores pinned at exactly
  1.0 also satisfies. Both guards were rebuilt around the *shape* of the
  distribution rather than its bounds: the unit tests push each model's recorded
  raw peak range back through the real decoder and require the resulting
  confidences to stay spread out at both ends, and the integration tests pin,
  per preset, where the median sits, how much survives each of several
  thresholds, and how much of the 0–1 range does useful filtering.

  Every row below was re-measured against the release tree rather than quoted from
  an earlier draft: one edit to the shipped source, the whole suite run after it,
  the source put back:

  | Undone in the shipped source | Tests that go red |
  |---|---|
  | `PEAK_SCALE["rtmw_l_384x288"]` → 1.0 | 7 (5 of them need no weights) |
  | `normalise_crop` fed BGR instead of RGB | 5 |
  | `_effective_selection` stops narrowing to the chosen person | 3 |
  | `letterbox` pads with 0 instead of 114 | 3 |
  | `DASH_BELOW` → 0.5 and `FULL_OPACITY_ABOVE` → 0.8 | **0** |
  | `YoloxDetector._parse` stops filtering on the person class | **0** |
  | `YoloxDetector.detect` skips its defensive `nms()` call | **0** |

  The three zeros are the honest part of the table. The opacity constants have no
  guard because `tests/gui/test_polish.py` builds its expectations *out of* the
  constants under test, so it pins the way `confidence_opacity` interpolates
  between them and slides along with whatever the endpoints happen to be. The two
  detector rows are a gap of a different kind: `_parse` and `detect` are exercised
  — `tests/unit/test_rtmw_wiring.py` drives the real detector over stubbed
  sessions — but every stub emits class-0 rows that do not overlap, so neither the
  class filter nor the suppression has anything to do. All three are carried as
  open items in `todo.md`. The accurate form of the claim, then: reverting is the
  habit here, seven of this release's behaviours have been through it rather than
  all of them, and it has found three that no test can detect.
- Two assumptions written down before the code was run turned out to be wrong, and
  both were caught by running it: an error-recovery test that failed 100 times out of
  100 because two back-to-back submissions meant the second always superseded the
  first (latest-wins working exactly as designed, the test wrong), and an assertion
  about typer's stdout and stderr that does not hold under typer 0.27.2, which has no
  click dependency and keeps the streams separate.
- Six model files, 464 MB, are cached locally and SHA-256 verified. The
  integration fixture is a 1935 Farm Security Administration photograph by Ben
  Shahn, a US government work in the public domain, with its provenance recorded
  in `tests/integration/fixtures/NOTICE.md`.
- v1 was built directly on `main` rather than on a branch, by agreement: the
  repository was one commit old and had no collaborators, so a feature branch would
  have merged into an empty `main`. From v2 the branch-and-pull-request convention in
  `todo.md` applies.

[Unreleased]: https://github.com/kouya-marino/AlchemyPose/compare/v1.0.0...HEAD
[1.0.0]: https://github.com/kouya-marino/AlchemyPose/releases/tag/v1.0.0
