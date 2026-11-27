# Todo

Actionable work, grouped by the version it belongs to. `versions.md` explains
*why* each version exists; this is the checklist.

Convention: one branch and one pull request per version. Tick items as they
merge, not as they are written. *(v1 is the exception — it was built directly on
`main` by agreement, because the repository was one commit old and had no
collaborators. The convention starts at v2.)*

**Every version updates `README.md` in its own pull request.** It is the PyPI
project page, so a deferred README means publishing a description that is wrong.
AlchemyFace shipped three releases that way because the update was scheduled for
1.0.0. It nearly happened here too: the README was written a third of the way
through v1 and described the GUI as "being assembled" until the very end.

**A test only counts once it has been seen to fail.** A fix is checked by
reverting it and confirming the new test goes red. Two tests in this build could
not fail and were rewritten or deleted; a test that cannot fail is worse than no
test, because it reports safety.

The peak scale is why that rule is written down rather than assumed. Its guards
asserted only that confidences land in `[0, 1]` and that *some* threshold keeps a
subset — both of which survive the fix being reverted, because clipping produces
them — so reverting it left the suite green while the slider went inert again.
They were rebuilt around the shape of the score distribution instead.

Eight of v1's behaviours have now actually been through a revert — one edit to
the shipped source, the whole suite run after it, the source put back. The first
four rows were measured on the 1.0.0 release tree on the M5; the last four after
the post-release tests went in, on Linux with weights cached:

| Undone in the shipped source | Tests that go red |
|---|---|
| `peak_scale` of `rtmw_l_384x288` → 1.0 | 7 (5 of them need no weights) |
| `normalise_crop` fed BGR instead of RGB | 5 |
| `_effective_selection` stops narrowing to the chosen person | 3 |
| `letterbox` pads with 0 instead of 114 | 3 |
| `DASH_BELOW` → 0.5 and `FULL_OPACITY_ABOVE` → 0.8 | 3 *(was 0)* |
| `YoloxDetector._parse` stops filtering on the person class | 1 *(was 0)* |
| `YoloxDetector.detect` skips its defensive `nms()` call | 1 *(was 0)* |
| `infer` stops re-zeroing a peakless keypoint in image space | 1 |

The three zeros were the open items below, and are closed. The second-order lesson holds: do not write
"every fix was verified this way" unless every fix has been. An overclaim about
test rigour is worse than no claim, because it stops anyone checking.

---

## Done — v1, shipped in 1.0.0

### Scaffold
- [x] `pyproject.toml`, `src/` layout, `py.typed`, MIT `LICENSE`, ruff + mypy
      configuration, pyenv virtualenv `alchemypose` on Python 3.12.7.
- [x] Lazy `App` resolution in `alchemypose/__init__.py`, with a subprocess test
      asserting `import alchemypose` loads zero `PySide6` modules.
- [x] Target Python 3.12 throughout. *(Started at 3.11, which silently disabled
      mypy entirely: numpy 2.5.2's stubs use PEP 695 syntax and mypy aborted on
      them before reaching a project file.)*

### `core/` — no Qt, no onnxruntime
- [x] COCO-WholeBody 133 topology: names, 127 edges, index ranges, render groups.
- [x] Render groups as a **cover, not a partition** — each limb owns its proximal
      anchor joint, plus wrist→hand connector edges. *(As a partition, "show the
      left leg" drew a shin and "show the left foot" drew three isolated dots.)*
- [x] Deduplicate the broad-group unions, which the overlap change broke.
- [x] `PersonPose` / `PoseFrame` / `PoseResult` / `Camera` / `Space` /
      `KeypointSelection`, with `track_id`, `keypoints_3d` and `space_3d` present
      and unpopulated from day one.
- [x] Nine named presets; `matching_preset` for chip highlighting.
- [x] JSON and CSV serialisation, round-tripping. Non-finite values encode as
      `null` rather than invalid JSON.
- [x] COCO-WholeBody → H36M-17 mapping, with pelvis, thorax, spine and head
      synthesised and the joint order pinned by a test.

### `inference/` — no Qt
- [x] Weight registry, cache directory, `ALCHEMYPOSE_MODEL_DIR` override, SHA-256
      verification of the **extracted** `.onnx` (every openmmlab artefact is a
      zip), and verify-before-publish so a failed download leaves no files behind.
- [x] Vendored SimCC decoding and top-down affine cropping.
- [x] ONNX session construction and the CPU/CoreML provider split, with a CPU
      fallback driven by a throwaway run on zeros rather than by a guard around
      session creation. *(CoreML creates the YOLOX session without complaint and
      reports both providers; it fails on the first `run()`, so a guard that only
      wraps the constructor would never fire.)*
- [x] YOLOX detector. *(The exported graph bakes in NMS and emits
      `dets(1,N,5)` + `labels(1,N)`, not a raw regression head.)*
- [x] `RTMWEngine` implementing `PoseEngine`. *(ImageNet normalisation in RGB
      order — raw 0–255 BGR produced confident, anatomically impossible keypoints
      with no error at all.)*
- [x] Per-model peak scale, so `PersonPose.scores` is a real 0–1 quantity and the
      confidence threshold is not inert.
- [x] Worker thread over a queue, latest-request-wins, no Qt import.
- [x] `alchemypose image | models list | fetch | clear | version | gui`.
- [x] Integration fixture with recorded provenance and a public-domain licence.

### `gui/`
- [x] Theme, palette, person's-left/right/midline colour convention, `Reporter`
      protocol injected so no test opens a modal.
- [x] Application shell and main window.
- [x] `PoseCanvas` — image, zoom anchored under the cursor, middle-drag pan.
- [x] Skeleton overlay: gradient bones, constant-size joints, per-group colour.
- [x] Tri-state layer tree as the single source of truth for visibility.
- [x] Preset chips that write to the tree and highlight on exact match.
- [x] Click to select, double-click or `Shift+H` to solo, `Esc` / `Alt+H` to
      restore, hover to brighten.
- [x] Open an image, run inference off the UI thread, show the result. Stale
      results from a superseded image are discarded rather than drawn.
- [x] Drop overlay items **and** the cached result when the image is replaced.
      *(`scene.clear()` was destroying items the overlay still referenced; and the
      stale result redrew photo A's skeleton over photo B.)*
- [x] Confidence slider, hover inspector, legend. The stored result is never
      mutated by the threshold.
- [x] One emphasis mechanism, on the overlay, so a rebuild reproduces the dimming
      instead of discarding it.
- [x] One writer for visibility. `SelectionController` writes to the layer tree
      and stops there; the window composes the tree's groups with the person
      chips in `_effective_selection` before anything reaches the canvas, so a
      solo or a clear can no longer put everybody back on screen while a chip
      still claims one person. *(Dropping that narrowing now turns three tests
      red.)*
- [x] Confidence-driven opacity and dashed low-confidence bones, recalibrated
      against real score distributions after the peak-scale fix.
- [x] Staggered load animation, torso → limbs → feet → hands → face.

### `gui/` — face level of detail
- [x] `gui/canvas/lod.py`: `FaceDetail` (`CONTOURS` / `REGIONS` / `POINTS`) and
      `face_detail(face_box_screen_px)`, thresholds at 80 px and 200 px. Nothing
      in it imports Qt, so the thresholds are tested without a widget.
- [x] Skip the individual face points below `POINTS`; the contours are drawn at
      every level. At fit-to-window 68 dots are a grey smudge, and 200 px is
      where adjacent dlib-68 points clear the 11 px hit radius a joint claims.
- [x] From `REGIONS` up, jaw, brows, nose, eyes and lips take their own colours
      in place of the theme's single face grey — in hues that avoid the green and
      orange the legend reserves for the person's left and right.
- [x] `PoseCanvas.zoom_changed = Signal(float)`, emitted from `zoom_by`,
      `fit_to_window` and `zoom_to_rect`; `PoseOverlay.on_zoom_changed` rebuilds
      only when the zoom crosses one of the two thresholds.
- [x] A face with no extent at all counts as unmeasured rather than tiny and keeps
      its points. *(Contours through coincident points are zero-length strokes, so
      thinning there would leave nothing drawn at all.)*
- [x] The **Face** preset chip calls `zoom_to_rect` on the face bounds, so it
      delivers the magnification a face tab would have given without the mode.
      Clicking it before a detection returns is harmless — the bounds are empty
      then and an empty rectangle is ignored.

### `gui/` — export
- [x] `gui/export.py`: `export_json`, `export_csv`, `export_png`. The PNG renders
      the **scene**, not the widget, so it comes out at the source image's own
      resolution regardless of zoom or window size.
- [x] An Export menu wired to file dialogs, refusing politely when there is
      nothing to export. The dialog sits behind an injected `SavePrompt` protocol,
      for the same reason `Reporter` does, so no test opens one; a cancelled
      dialog, a folder that does not exist and an image too large to allocate all
      end as a status line or a reported error rather than a traceback.
- [x] Settled what a PNG export of a *soloed* selection contains: the visible
      overlay, because the PNG renders the scene as it stands. Narrowing `all` to
      `hands` takes one person's overlay from 260 items to 82 — 133 keypoints plus
      127 bones, against 42 plus 40. A PNG needs only an open photograph; JSON and
      CSV need a result.

### README, CI and the 1.0.0 release
- [x] Rewrite the README for what v1 actually does: the window and its gestures,
      the CLI with real transcripts, what a confidence score means and why it
      needed work, and a roadmap with the right boxes ticked. *(It had gone stale
      almost immediately — it still described the GUI as "being assembled" with
      the detector, engine, worker, CLI, canvas and layer tree all listed as
      outstanding.)*
- [x] Say plainly what the README's timing table measures — the pose model alone,
      per detected person, with the detector's per-frame cost stated separately —
      and re-measure both for the release rather than carrying the numbers over.
- [x] `.github/workflows/ci.yml` — ruff, mypy, `pytest tests/unit`, the GUI suite
      under `QT_QPA_PLATFORM=offscreen`, and the weights-only tests last. Python
      3.12 only, not a matrix: a 3.11 entry would be a guaranteed red X, since
      numpy's stubs will not parse under it. No weights are ever fetched on a
      runner, so the `models`-marked tests skip there; the in-test cache check is
      the backstop, not the guarantee.
- [x] **CI has run, and it is green.** Six runs to date, all pushes to `main`, all
      successful: `e9aebc2` (run 33684801971, 55s), `88a659e` (33685597889, 1m 02s),
      `51aba8a` (33700786465, 58s), `03a701f` (33735124492, 1m 01s), `4ba3cdb`
      (33738463693, 59s) and `f3d42ab` (33751784998, 1m 03s). The most recent log
      shows `ruff` clean, `mypy` with no issues across 38 source files and then
      across `main.py`, `tests/unit` at 289 passed and 2 skipped, `tests/gui` at
      179 passed, and the weights-only step at 10 skipped. The step ahead of the
      GUI tests printed `Qt 6.11.2 on platform offscreen`, so the hand-derived
      list of system libraries the PySide6 wheel links against is now observed
      rather than reasoned. *(The AlchemyFace precedent that prompted the
      suspicion — a publish workflow advertising "the same checks as CI" that
      silently skipped every GUI test, and a wheel check reporting a leak that did
      not exist because it ran from the wrong directory — is why the log was read
      line by line instead of trusting the tick.)*
- [x] Re-run CI after the round of workflow edits that followed `51aba8a`.
      `ci.yml` was last touched in `4ba3cdb`, and the runs on `4ba3cdb` and
      `f3d42ab` were both green, so the workflow as it stands is the workflow that
      has met a runner.
- [x] **The shutdown abort is fixed, and it was onnxruntime telemetry.** The suite
      used to report green, print its pass line and then die in interpreter shutdown
      with `libc++abi: terminating due to uncaught exception of type
      std::__1::system_error: recursive_mutex lock failed: Invalid argument`, exit
      status 134. `lsof -a -p <pid> -i` on a bare `import onnxruntime` process shows
      it open exactly one outbound socket — TCP 443 to an address registered to
      Microsoft — first visible 9.5 s after the import; the exit-time teardown of the
      telemetry log manager frees a `recursive_mutex` that an upload still in flight
      then tries to lock. `tests/conftest.py` now sets `ORT_DISABLE_TELEMETRY=1`
      ahead of the first onnxruntime import, which is the only moment it is read
      (commit `f1dd076`); with it set, the process opens no socket at all. Measured
      interleaved, 40 runs per arm on trees identical but for that one line:
      0 aborts with it, 4 without. Keep measuring runs as
      `pytest -q > out 2>&1; rc=$?` regardless — `pytest -q | tail -1` reports
      `tail`'s status and would hide the next one exactly as well as it hid this one.
- [x] Release metadata: version read from `alchemypose.__version__` and nowhere
      else, SPDX licence expression with `license-files`, the full classifier and
      keyword set, and `project.urls`. Verified on the built artefacts — the wheel
      is 79 KB over 45 entries with no weights and no fixtures, the sdist 74 KB
      with no tests, and `twine check` passes on both.
- [x] Make the bare `alchemypose` open the window rather than print a page of
      help, with `--version` answered during parsing and every subcommand — and
      every *mistyped* subcommand — still reaching what it named.

---

## Closed after 1.0.0 — the v1 leftovers

### Missing tests, in the order they matter
- [x] *Closed: two unit tests in `tests/unit/test_detector.py` over hand-built
      rows; each goes red against its own revert.* **`YoloxDetector._parse` and `detect` have no coverage that can fail.**
      `tests/unit/test_detector.py` tests `nms` and `letterbox` and nothing else,
      and while `tests/unit/test_rtmw_wiring.py` does drive the real detector over
      stubbed sessions, every stub it feeds emits class-0 rows that do not overlap
      — so nothing in the suite exercises either behaviour the two methods add.
      Measured on the release tree: dropping `is_person` from the mask in `_parse`
      leaves the whole suite green, and replacing `keep = nms(...)` in `detect`
      with every row kept leaves it green too. What is missing is a pair of unit
      tests over a hand-built `(dets, labels)` pair — one carrying a high-scoring
      class-1 row that must not come back as a person, one carrying two boxes over
      the same person that must come back as one. Both are cheap: neither needs a
      session, a model file or a display. *(The third candidate is now covered:
      `letterbox` padding with 0 instead of 114 turns three tests red, since
      `test_infer_hands_the_detector_raw_letterboxed_pixels` went in with
      `4ba3cdb`.)*
- [x] *Closed: `tests/gui/test_confidence_calibration.py`, over real scores
      recorded in `tests/gui/fixtures/two_people_scores.json`, so it needs no
      weights. The old constants turn all three of its tests red: 52 of 61 bones
      dashed on `performance`, none fully opaque.*
      **Nothing in the suite can detect a miscalibrated opacity constant.**
      `tests/gui/test_polish.py` derives its expectations *from* the constants it
      is checking — `CONFIDENT = (1.0 + FULL_OPACITY_ABOVE) / 2.0`, and the dash
      cases are written at `DASH_BELOW` and `DASH_BELOW - 0.01` — so it pins the
      shape of `confidence_opacity` and slides along with whatever the constants
      happen to be. Restoring the values that shipped wrong, `DASH_BELOW = 0.5`
      and `FULL_OPACITY_ABOVE = 0.8`, leaves the whole suite green. What is
      missing is a test against a realistic score distribution — the fixture
      percentiles the constants were calibrated on — asserting that the dashed
      fraction and the full-opacity fraction stay inside the intended band. With
      the old constants that fraction was 51 dashed out of 61 bones drawn and not
      one at full opacity, so the band is easy to state.
- [x] *Closed: seven tests, the download size checked against the cached files.*
      `tests/unit/test_readme.py` — assert the README names the install command,
      the launch command, the download size, the model and its licence. Prose
      about being kept current is not a mechanism; a test is.

### Deferred minors, to batch into one commit
- [x] *Closed — and it exposed a hit-testing gap: Qt skips fully transparent
      items, so a click mid-animation missed. A canvas press now finishes the
      draw-in first.* Nothing calls `PoseOverlay.animate_in()`. It exists, it is tested, and
      `MainWindow._on_result` is where it belongs — one line.
- [x] *Closed: select and solo dim, clear restores; tested at the call sites.*
      Nothing calls `PoseOverlay.set_image_dimmed()` either. The photograph is
      meant to fall back to ~60% opacity once a selection exists, so the skeleton
      reads as the subject — `theme.DIMMED_IMAGE_OPACITY` is already 0.6 and the
      method already applies it. `SelectionController`'s `select_group` /
      `solo_group` / `clear` are the call sites.
- [x] *Closed: `ModelSpec.peak_scale`; `rtmw.PEAK_SCALE` is derived from it.*
      Move `PEAK_SCALE` onto `ModelSpec` in `weights.py`. It is a per-model
      constant like `input_size`, and it currently lives in `rtmw.py` away from
      everything else that describes a model. Mechanical.
- [x] *Closed as deliberate: still no panel reads emphasis from both.*
      `SelectionController._emphasised` and `PoseOverlay._emphasis` track
      overlapping state and can legitimately disagree while soloed. Deliberate for
      now; revisit if any panel reads emphasis from both.
- [x] *Closed: re-zeroed in image space in `infer`, with a test.*
      A keypoint with no peak is zeroed in *model-input* space and then affine
      untransformed, so it lands at the crop's top-left corner rather than at the
      image origin — confirmed by pushing an all-zero SimCC pair through the real
      decoder and transform. Harmless while it is hidden, but at threshold 0.0 it
      is drawn there.
- [x] *Closed: not adopted — it would collapse the column-aligned comments the
      code reads by. The README says so.* `ruff format` reports "34 files would be reformatted, 43 files already
      formatted" over `src tests main.py`. The project gates on `ruff check`,
      which passes; decide whether to adopt the formatter and do it in one commit,
      or say in the README that it is deliberately not used.

### The release itself
- [x] *Done: the `v1.0.0` tag exists and PyPI serves 1.0.0 as the only release.*
      Tag `v1.0.0` and upload the built wheel and sdist to PyPI. This is the last
      step of the 1.0.0 release and the only one still outstanding: the artefacts
      build clean and pass `twine check`, and `CHANGELOG.md`'s link references
      point at that exact tag name, so the tag has to be `v1.0.0` and not `1.0.0`.
      PyPI does not allow a version to be re-uploaded, so anything wrong in the
      artefact costs a version number.
- [x] *Done: `assets/screenshot.png`, taken from the window running on Linux,
      referenced from the README. The repository went public with 3.0.0, and
      the README now points at absolute raw URLs (tested), so PyPI renders it too;
      the 1.1.0 and 2.0.0 pages show broken images.* A screenshot of the window with a multi-person image and the skeleton
      visible. This is a portfolio piece and the README is the first thing seen —
      and it is currently a page about a visual tool with no picture on it. A
      screenshot on the PyPI page needs an absolute URL to a publicly reachable
      image, which the repository cannot serve while it is private.
- [ ] *Half done: opened on Linux (Ubuntu 22.04, X11, PySide6 6.12.0) with the
      fixture — three people found on `performance`, drawn correctly. Windows has
      still never been seen.* Open the window on Windows and on Linux at least once. The GUI suite proves
      the widgets construct and behave headless on `ubuntu-latest`; nobody has yet
      seen the application drawn on either platform, and the `gui-scripts` entry
      point deliberately left out of `pyproject.toml` is waiting on the same.

---

## Later

### v2 · video — built on branch `v2-video`
- [x] `PyAV` for frame-accurate seeking, as an extra rather than a base dependency.
      *OpenCV is the fallback; both are held to one contract over generated clips.*
- [x] Populate `track_id`. The field has been there since the schema's first
      commit and every export written so far carries it as `null`; this is the
      version that fills it.
- [x] *Done differently, on purpose:* scrubbing reads from an offline pass that
      analyses every frame once, because playback beside 3D and a robot needs every
      frame and latest-request-wins drops them. The worker keeps its policy for
      stills and the webcam. ~~Scrubbing on the existing latest-request-wins worker.~~
- [x] *Decided: applied at render time; stored coordinates stay raw.*
      Temporal smoothing, and a decision about whether smoothed coordinates are
      stored or applied at render time. *(v1's rule is that coordinates are stored
      raw; smoothing should not be the thing that breaks it.)*
- [x] *Its two ideas are in: IoU matching, and `--det-interval` cropping from the
      last pose between detections.* Revisit `rtmlib`'s `PoseTracker` as a
      reference for the tracking half.
- [x] Webcam, which arrives with the video work. *Not yet run against a real camera:
      the tests drive it with a fake capture.*
- [ ] Export the overlay as a video (MP4), not just the frame on screen as PNG.
- [ ] Identity can swap when two people cross completely in front of each other;
      IoU has no appearance model. Revisit if multi-person clips matter.

### v3 · 3D — shipped in 3.0.0
- [x] *As the `3d` preset rather than in place of RTMW, so the 2D presets keep their
      speed.* RTMW3D in place of RTMW. Same 133-keypoint topology plus a third
      coordinate, so the layer tree, render groups, presets and selection logic
      carry over unchanged.
- [x] Populate `keypoints_3d` and set `space_3d`.
- [x] The orbitable viewport, on the declared `threed` extra (`pyqtgraph` +
      `PyOpenGL`). **Never** `PySide6.QtQuick3D` or `PySide6.QtGraphs` — both are
      commercial-or-GPLv3 with no LGPL option, and importing either relicenses the
      distributed application.
- [x] glTF export, converting to y-up at the boundary and nowhere else.
- [ ] `core/h36m.py` finally gets a caller. *Moved to v4: the G1 retargeting.*

### v4 · the Unitree G1
- [ ] Load `g1_29dof_rev_1_0.urdf` and its meshes at runtime, pinned and verified.
- [ ] Retarget the 3D skeleton to the 29 joints, within the joint limits.
- [ ] The robot beside the clip and its 3D skeleton, in step.
- [ ] A robot-motion export for training: root pose, joint angles, velocities,
      foot contacts, in a MuJoCo-loadable layout, with a playback check.

---

## Open questions

- [ ] Should the peak scale be exposed at all — a `PersonPose.score_scale`, or a
      note in the exported JSON? Exports written before the fix carry raw 10×
      scores under the same `schema_version: 1`, so nothing flags them on reload.
      Bumping the schema version for a field that did not change is heavy-handed;
      doing nothing means two files can disagree silently.
- [ ] `Tab` to cycle the selected person was part of the intended keyboard map
      and is not implemented — nothing in `src/` binds it. The numbered person
      chips in the toolbar cover the same ground. Add the shortcut, or accept the
      chips as the answer and stop carrying it as an omission?
- [ ] rtmlib ships `kpt_thr=0.43` against unbounded SimCC peaks on the same RTMW-l
      weights used here, where it filters almost nothing. Worth reporting upstream.
- [ ] The `balanced` preset is DWPose-l rather than an RTMW variant, so it is the
      one model in the table from a different lineage. Keep it for the size/quality
      point it makes, or drop it for a consistent family?
