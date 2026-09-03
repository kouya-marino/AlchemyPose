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
They were rebuilt around the shape of the score distribution instead, and putting
`PEAK_SCALE["rtmw_l_384x288"]` back to `1.0` now turns seven tests red, only two
of which need cached weights.

The rule is applied, not universal, and the gap between those two words is the
whole point. Four fixes have actually been through a revert. Restoring the
BGR→RGB normalisation defect turns two tests red; undoing the person-filter fix
turns twelve red; the peak scale turns seven. The fourth found a second hole:
putting the old opacity constants back — `DASH_BELOW = 0.5`,
`FULL_OPACITY_ABOVE = 0.8` — leaves the entire suite green, so that fix is
currently unguarded. It is carried as an open item below. The second-order lesson
holds: do not write "every fix was verified this way" unless every fix has been.
An overclaim about test rigour is worse than no claim, because it stops anyone
checking.

---

## Done — v1 · the library and the engine

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
      `hands` takes the same export from 260 items to 82. A PNG needs only an open
      photograph; JSON and CSV need a result.

---

## Open — v1

### Deferred minors, to batch into one commit
- [ ] `SelectionController._sync_overlay_from_layers` pushes the bare tree
      selection, bypassing the window's person filter — so a solo or a clear drops
      the chosen person until the next tree or chip change. Either the controller
      asks the window for its effective selection, or the overlay holds the person
      filter separately from `visible`.
- [ ] Nothing calls `PoseOverlay.animate_in()`. It exists, it is tested, and
      `MainWindow._on_result` is where it belongs — one line.
- [ ] Nothing calls `PoseOverlay.set_image_dimmed()` either. The photograph is
      meant to fall back to ~60% opacity once a selection exists, so the skeleton
      reads as the subject — `theme.DIMMED_IMAGE_OPACITY` is already 0.6 and the
      method already applies it. `SelectionController`'s `select_group` /
      `solo_group` / `clear` are the call sites.
- [ ] Move `PEAK_SCALE` onto `ModelSpec` in `weights.py`. It is a per-model
      constant like `input_size`, and it currently lives in `rtmw.py` away from
      everything else that describes a model. Mechanical.
- [ ] `SelectionController._emphasised` and `PoseOverlay._emphasis` track
      overlapping state and can legitimately disagree while soloed. Deliberate for
      now; revisit if any panel reads emphasis from both.
- [ ] A keypoint with no peak is zeroed in *model-input* space and then affine
      untransformed, so it lands at the crop's top-left corner rather than at the
      image origin. Harmless while it is hidden, but at threshold 0.0 it is drawn
      there.
- [ ] `ruff format` would reformat 34 of the 76 files under `src` and `tests`. The
      project gates on `ruff check`, which passes; decide whether to adopt the
      formatter and do it in one commit, or say in the README that it is
      deliberately not used.
- [ ] **Nothing in the suite can detect a miscalibrated opacity constant.**
      `tests/gui/test_polish.py` derives its expectations *from* the constants it
      is checking — `CONFIDENT = (1.0 + FULL_OPACITY_ABOVE) / 2.0`, and the dash
      cases are written at `DASH_BELOW` and `DASH_BELOW - 0.01` — so it pins the
      shape of `confidence_opacity` and slides along with whatever the constants
      happen to be. Restoring the values that shipped wrong, `DASH_BELOW = 0.5`
      and `FULL_OPACITY_ABOVE = 0.8`, leaves the whole suite green. What is
      missing is a test against a realistic score distribution — the fixture
      percentiles the constants were calibrated on — asserting that the dashed
      fraction and the full-opacity fraction stay inside the intended band. Until
      that exists, the recalibration is a fix with no guard on it.

### README, CI and release
- [x] Rewrite the README for what v1 actually does: the window and its gestures,
      the CLI with real transcripts, what a confidence score means and why it
      needed work, and a roadmap with the right boxes ticked. *(It had gone stale
      almost immediately — it still described the GUI as "being assembled" with
      the detector, engine, worker, CLI, canvas and layer tree all listed as
      outstanding.)*
- [x] Say plainly what the README's timing table measures. It is now the pose
      model alone, per detected person, with the detector's per-frame cost stated
      separately. End to end on the fixture, warm, on this machine: `lite` 17 ms
      for 2 people, `performance` 112 ms for 3 — consistent with the parts.
- [x] `.github/workflows/ci.yml` — ruff, mypy, `pytest tests/unit`, then the GUI
      suite under `QT_QPA_PLATFORM=offscreen`. Python 3.12 only, not a matrix: a
      3.11 entry would be a guaranteed red X, since numpy's stubs will not parse
      under it. No weights are ever fetched on a runner, so the `models`-marked
      tests skip there; the in-test cache check is the backstop, not the guarantee.
- [x] **CI has run, and it is green.** The workflow went in with commit `e9aebc2`
      and GitHub registered it on that same push. Three runs to date, all pushes
      to `main`, all successful: `e9aebc2` (run 33684801971, 55s), `88a659e`
      (33685597889, 1m 02s) and `51aba8a` (33700786465, 58s), the first two on
      2 September and the last on 3 September, UTC. The most recent log shows
      `ruff` clean, `mypy` with no issues across 38 source files, `tests/unit` at
      256 passed and 2 skipped, and `tests/gui` at 162 passed. The step ahead of
      the GUI tests printed `Qt 6.11.2 on platform offscreen`, so the hand-derived
      list of system libraries the PySide6 wheel links against is now observed
      rather than reasoned — a runner loaded the offscreen plugin with exactly
      that list installed and nothing more. *(The AlchemyFace precedent that
      prompted the suspicion — a publish workflow advertising "the same checks as
      CI" that silently skipped every GUI test, and a wheel check reporting a leak
      that did not exist because it ran from the wrong directory — is why the log
      was read line by line instead of trusting the tick.)*
- [ ] Re-run CI after the current round of workflow edits. A green tick belongs to
      the revision that earned it: `ci.yml` has been changed since `51aba8a`, and
      those changes have not been through a runner. Nothing is wrong with them —
      they simply have no evidence behind them yet.
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
- [ ] `tests/unit/test_readme.py` — assert the README names the install command,
      the launch command, the download size, the model and its licence. Prose
      about being kept current is not a mechanism; a test is.
- [ ] A screenshot of the window with a multi-person image and the skeleton
      visible. This is a portfolio piece and the README is the first thing seen —
      and it is currently a page about a visual tool with no picture on it.
- [ ] Tag and publish. The wheel must stay small — no weights, no fixtures.

---

## Later

### v2 · video
- [ ] `PyAV` for frame-accurate seeking, as an extra rather than a base dependency.
- [ ] Populate `track_id`. The field has been there since the schema's first
      commit; this is the version that fills it.
- [ ] Scrubbing on the existing latest-request-wins worker — a new frame
      supersedes any queued frame that has not started, which is already how it
      behaves.
- [ ] Temporal smoothing, and a decision about whether smoothed coordinates are
      stored or applied at render time. *(v1's rule is that coordinates are stored
      raw; smoothing should not be the thing that breaks it.)*
- [ ] Revisit `rtmlib`'s `PoseTracker` as a reference for the tracking half.
- [ ] Webcam, which arrives with the video work.

### v3 · 3D
- [ ] RTMW3D in place of RTMW. Same 133-keypoint topology plus a third
      coordinate, so the layer tree, render groups, presets and selection logic
      carry over unchanged.
- [ ] Populate `keypoints_3d` and set `space_3d`.
- [ ] The orbitable viewport, on the declared `threed` extra (`pyqtgraph` +
      `PyOpenGL`). **Never** `PySide6.QtQuick3D` or `PySide6.QtGraphs` — both are
      commercial-or-GPLv3 with no LGPL option, and importing either relicenses the
      distributed application.
- [ ] glTF export, converting to y-up at the boundary and nowhere else.
- [ ] `core/h36m.py` finally gets a caller.

---

## Open questions

- [ ] Should the peak scale be exposed at all — a `PersonPose.score_scale`, or a
      note in the exported JSON? Exports written before the fix carry raw 10×
      scores under the same `schema_version: 1`, so nothing flags them on reload.
      Bumping the schema version for a field that did not change is heavy-handed;
      doing nothing means two files can disagree silently.
- [ ] `Tab` to cycle the selected person was part of the intended keyboard map
      and is not implemented. The numbered person chips in the toolbar cover the
      same ground. Add the shortcut, or accept the chips as the answer and stop
      carrying it as an omission?
- [ ] rtmlib ships `kpt_thr=0.43` against unbounded SimCC peaks on the same RTMW-l
      weights used here, where it filters almost nothing. Worth reporting upstream.
- [ ] The `balanced` preset is DWPose-l rather than an RTMW variant, so it is the
      one model in the table from a different lineage. Keep it for the size/quality
      point it makes, or drop it for a consistent family?
