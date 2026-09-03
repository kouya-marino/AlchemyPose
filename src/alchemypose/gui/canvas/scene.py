"""Turns a PoseResult plus a KeypointSelection into graphics items.

Rebuilding from scratch on every change is deliberate: a few thousand items is
well inside Graphics View's range, and SLEAP — a PySide pose GUI — clears and
rebuilds its scene every video frame. Simplicity beats a diffing algorithm here.

Because everything is rebuilt, every *visual* state an item can be in must be
reproducible from the overlay's own fields: confidence fades from the result and
threshold, emphasis dimming from ``_emphasis``, and how much of the face is
drawn at all from ``_zoom``. An item's opacity is always
``_target_opacity(item)`` — nothing sets it to an ad-hoc value that a rebuild
would then forget.
"""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import QEasingCurve, QObject, QPointF, QRectF, QVariantAnimation, Signal
from PySide6.QtGui import QColor

from alchemypose.core.schema import KeypointSelection, PersonPose, PoseResult
from alchemypose.gui import theme
from alchemypose.gui.canvas import lod
from alchemypose.gui.canvas.items import BoneItem, KeypointItem
from alchemypose.gui.canvas.view import PoseCanvas

# --- Confidence cue calibration -------------------------------------------
# These three numbers are NOT self-evident and were NOT guessed. They were
# calibrated against measured ``PersonPose.scores`` distributions from real
# RTMW output (RTMW-m = preset ``lite``, RTMW-l = preset ``performance``) on
# ``tests/integration/fixtures/two_people.jpg`` at the default 0.3 threshold,
# *after* scores were normalised to honest [0, 1] confidences at the engine
# boundary (``simcc.decode_simcc`` / ``rtmw.PEAK_SCALE``). Their first values
# (dash 0.5, full 0.8) were set while scores were unnormalised logits and,
# against honest confidences, dashed ~72% of the default preset's skeleton and
# let none of it reach full opacity — a cue that applies to everything is no
# cue. If the score pipeline changes again, RE-MEASURE rather than re-tune by
# eye: run both presets over that fixture, collect the ``PersonPose.scores`` of
# the keypoints that survive the default threshold, and read the three numbers
# back off that distribution — the dash cut low in its tail, the full-opacity
# cut around its upper middle, the floor just under its lowest survivors.

DASH_BELOW = 0.25
"""A bone whose weaker endpoint confidence falls under this is drawn dashed.

Calibrated (see above): on honest confidences this dashes 0% of surviving
body bones for both models at the 0.3 threshold — dashing is reserved for
genuinely doubtful limbs, which appear once the user lowers the threshold.
"""

FULL_OPACITY_ABOVE = 0.55
"""Scores at or above this draw fully opaque; below it the item fades towards the threshold.

Calibrated (see above): ``lite`` scores are uniformly high (median ~0.92) and
render fully opaque, which is the honest picture; ``performance`` keeps a
visible spread (median opacity ~0.66, 17% at full) without reading as broken.
"""

MIN_OPACITY = 0.35
"""The faintest a keypoint that survived the threshold is allowed to get.

Re-checked (not assumed) in the same calibration: on ``performance`` at the
0.3 threshold, 15% of surviving body keypoints sit within 0.10 of this floor
(their scores are 0.30-0.36), so the floor is what keeps a quarter of the
default preset's skeleton legible at all. Raising it would compress the
0.35-1.0 spread the knee retune just restored; lowering it would make those
joints vanish into the dark canvas. Unchanged on evidence, not by default.
"""

DIM_FACTOR = 0.25
"""Multiplier applied to every item outside the emphasised group."""

#: Draw-in stage per render group: torso first, then limbs, feet, hands; face last.
ANIMATION_STAGE: dict[str, int] = {
    "torso_head": 0,
    "left_arm": 1,
    "right_arm": 1,
    "left_leg": 1,
    "right_leg": 1,
    "left_foot": 2,
    "right_foot": 2,
    "left_hand": 3,
    "right_hand": 3,
}
_LAST_STAGE = 4
_STAGE_COUNT = _LAST_STAGE + 1


def confidence_opacity(score: float, threshold: float) -> float:
    """Fade a keypoint towards the threshold, so uncertainty is *shown*, not hidden.

    Pure and Qt-free on purpose. ``1.0`` at or above ``FULL_OPACITY_ABOVE``;
    linear down to ``MIN_OPACITY`` at the threshold; never below ``MIN_OPACITY``
    (a score under the threshold is not drawn at all, so it never gets here in
    practice, but the function still returns something sane if it does).
    """
    if score >= FULL_OPACITY_ABOVE:
        return 1.0
    span = max(FULL_OPACITY_ABOVE - threshold, 1e-6)
    fraction = min(1.0, max(0.0, (score - threshold) / span))
    return MIN_OPACITY + fraction * (1.0 - MIN_OPACITY)


class PoseOverlay(QObject):
    """Owns every skeleton item in the canvas's scene.

    Inherits ``QObject`` (rather than holding a separate signals object) purely
    so hover can be surfaced as ordinary Qt signals — ``KeypointItem`` itself is
    not a ``QObject`` (``QGraphicsItem`` does not support multiple inheritance
    with it cleanly), so its hover events are routed here through a plain
    callback and re-emitted as ``keypoint_hovered`` / ``keypoint_left``, which
    ``MainWindow`` connects to the inspector panel.

    The canvas owns the scene and wipes it whenever a new image is shown; the
    overlay caches references to items *in* that scene. Neither is wrong, but
    ``QGraphicsScene.clear()`` destroys items at the C++ level immediately, so
    the overlay listens to ``PoseCanvas.scene_cleared`` and forgets its
    references before they dangle — see ``_forget_items``.
    """

    keypoint_hovered = Signal(int, int, str)   # person_index, keypoint_index, group
    keypoint_left = Signal()

    def __init__(self, canvas: PoseCanvas) -> None:
        super().__init__(canvas)
        self._canvas = canvas
        self._result: PoseResult | None = None
        self._selection = KeypointSelection(visible=frozenset())
        self._threshold = 0.3
        self._emphasis: str | None = None
        #: The view's current scale, mirrored here from ``PoseCanvas.zoom_changed``.
        #: Keypoint items ignore the view transform, so how big the face is *on
        #: screen* — the only thing the detail level may depend on — is its scene
        #: extent times this, and the overlay cannot read it off an item.
        self._zoom = 1.0
        self.keypoint_items: list[KeypointItem] = []
        self.bone_items: list[BoneItem] = []
        #: Confidence-derived opacity per live item, keyed by ``id(item)``.
        #: Bones carry no score of their own, and ``KeypointItem`` only bakes
        #: its score into a tooltip, so the overlay keeps it here. Ids are only
        #: stable while an item is alive, which is exactly as long as it is in
        #: one of the two lists above — the three are always cleared together.
        self._confidence: dict[int, float] = {}
        self._animation: QVariantAnimation | None = None
        canvas.scene_cleared.connect(self._forget_items)

    # ----------------------------------------------------------------- counts

    @property
    def keypoint_item_count(self) -> int:
        return len(self.keypoint_items)

    @property
    def bone_item_count(self) -> int:
        return len(self.bone_items)

    @property
    def visible_item_count(self) -> int:
        return self.keypoint_item_count + self.bone_item_count

    @property
    def emphasis(self) -> str | None:
        """The render group currently emphasised, or ``None`` for no dimming."""
        return self._emphasis

    # ------------------------------------------------------- face detail level

    @property
    def face_detail(self) -> lod.FaceDetail:
        """How much face the current zoom earns — see ``canvas.lod``.

        A face with no *drawn* extent is not a small face but an unmeasured
        one, and the trade the lower levels make buys nothing there. Either
        way round: with every point on one coordinate (a stub result, or a
        detection that collapsed) the contours are zero-length strokes, so
        dropping the points would leave nothing at all; with no point clearing
        the confidence threshold there is nothing to drop in the first place.
        Unmeasured therefore keeps the points; only a face with real drawn
        extent is thresholded.
        """
        size = self._face_screen_size()
        if size <= 0.0:
            return lod.FaceDetail.POINTS
        return lod.face_detail(size)

    def _is_confident(self, person: PersonPose, index: int) -> bool:
        """Whether one keypoint is trustworthy enough to put on screen at all.

        A score of exactly 0.0 is not "zero confidence" but a sentinel:
        ``decode_simcc`` clips to [0, 1], so 0.0 arises only from a
        non-positive SimCC peak, whose *position* it also zeroes.
        Un-transformed, that (0, 0) lands on the crop's top-left corner. The
        slider can reach 0.0, and ``>=`` alone would then pile such joints on
        each person's box corner; skip them regardless of threshold.
        """
        score = float(person.scores[index])
        return score > 0.0 and score >= self._threshold

    def _is_drawable(self, person: PersonPose, index: int) -> bool:
        """Whether one keypoint of one person is actually drawn: visible *and* confident.

        The two halves are deliberately separable. Confidence decides whether
        the detector's answer belongs on screen at all; the layer selection is
        the user choosing which parts of what is on screen to look at. Only the
        first is a fact about the photograph, which is why ``_face_screen_size``
        consults ``_is_confident`` alone.
        """
        return self._selection.is_visible(index) and self._is_confident(person, index)

    def _face_screen_size(self) -> float:
        """The longest on-screen edge of the largest *drawn* face, in pixels.

        Drawn, not merely detected. A face whose points all fall under the
        confidence threshold renders nothing, and measuring it anyway can only
        ever push the level *up* — never adding a face the user can see, only
        spending 68 items on some other, smaller face they cannot read. That is
        precisely the cost this whole trade exists to avoid, so the box covers
        the points that clear ``_is_confident``.

        Confidence only, *not* ``_is_drawable``: hiding all but the lips does
        not make the head on screen any smaller, and measuring only the
        surviving layer would strip the region colours from a region at the
        moment the user isolated it — backwards.

        ``group_bounds`` is deliberately not reused either. It answers "where
        is this group", which is what the Face preset needs to frame the head,
        and framing a head whose confidence the user dropped below the slider
        is still the right thing for that button to do.
        """
        if self._result is None:
            return 0.0
        people = self._result.frames[0].people
        chosen = self._selection.persons
        indices = range(len(people)) if chosen is None else sorted(chosen)
        face = self._result.topology.groups.get("face", ())
        # The largest, not the first: with the view framed on one big face,
        # a smaller one beside it is still drawn, and hiding *its* points on
        # the strength of a neighbour's size would be arbitrary.
        longest = 0.0
        for index in indices:
            person = people[index] if 0 <= index < len(people) else None
            if person is None:
                continue
            drawn = [i for i in face if self._is_confident(person, i)]
            if not drawn:
                continue
            xs = [float(person.keypoints[i][0]) for i in drawn]
            ys = [float(person.keypoints[i][1]) for i in drawn]
            longest = max(longest, max(xs) - min(xs), max(ys) - min(ys))
        return longest * self._zoom

    def on_zoom_changed(self, zoom: float) -> None:
        """Rebuild only when the zoom has crossed a face detail threshold.

        Connected to ``PoseCanvas.zoom_changed``, which fires on every wheel
        notch. A rebuild is hundreds of items, and three detail levels are
        separated by only two thresholds, so the level changes just twice
        across the whole range: sweeping ``MIN_ZOOM`` to ``MAX_ZOOM`` a notch
        at a time is 48 calls and 2 rebuilds. Those two notches are the only
        ones allowed to pay for one.
        """
        before = self.face_detail
        self._zoom = zoom
        if self.face_detail is not before:
            self._rebuild()

    # ------------------------------------------------------------------ state

    def set_result(
        self, result: PoseResult, selection: KeypointSelection, confidence_threshold: float
    ) -> None:
        self._result = result
        self._selection = selection
        self._threshold = confidence_threshold
        self._rebuild()

    def set_selection(self, selection: KeypointSelection) -> None:
        self._selection = selection
        self._rebuild()

    def set_confidence_threshold(self, value: float) -> None:
        self._threshold = value
        self._rebuild()

    def set_emphasis(self, group: str | None) -> None:
        """Dim every group except ``group``; ``None`` restores everything.

        Recorded, not just applied: a later rebuild reproduces it, so toggling a
        layer no longer silently undoes the user's selection. Existing items are
        re-shaded in place — no rebuild, item identities are preserved.
        """
        self._emphasis = group
        self._apply_opacities()

    def set_image_dimmed(self, dimmed: bool) -> None:
        """Push the photograph back so the skeleton reads as the subject.

        Touches only the pixmap. Skeleton opacity is owned by
        ``_target_opacity`` and is not consulted here.
        """
        item = self._canvas.image_item
        if item is not None:
            item.setOpacity(theme.DIMMED_IMAGE_OPACITY if dimmed else 1.0)

    # ------------------------------------------------------------------ build

    def _forget_items(self) -> None:
        """Drop every cached item reference *without* touching the scene.

        Connected to ``PoseCanvas.scene_cleared``, which fires just before
        ``QGraphicsScene.clear()``. The scene is about to delete the items
        itself; calling ``removeItem`` on them here as well would be a
        double-free, and calling it a moment later would hit already-deleted
        C++ objects. Harmless when there is nothing to forget.

        The result goes too: it belongs to the photograph that was just
        replaced. Keeping it would let a layer toggle, arriving before the new
        image's inference does, redraw the *old* skeleton over the *new* photo —
        plausible-looking and completely wrong. Drawing nothing is the honest
        answer until ``set_result`` is called again.
        """
        self._stop_animation()
        self._result = None
        self.keypoint_items.clear()
        self.bone_items.clear()
        self._confidence.clear()

    def _clear(self) -> None:
        """Remove our items from a scene that is *not* being wiped."""
        self._stop_animation()
        scene = self._canvas.scene()
        for item in (*self.keypoint_items, *self.bone_items):
            scene.removeItem(item)
        self.keypoint_items.clear()
        self.bone_items.clear()
        self._confidence.clear()

    def _rebuild(self) -> None:
        self._clear()
        if self._result is None:
            return

        scene = self._canvas.scene()
        topology = self._result.topology
        frame = self._result.frames[0]
        # Read once: it walks every person's face box, and the answer cannot
        # change while one rebuild is in flight.
        detail = self.face_detail

        for person_index, person in enumerate(frame.people):
            if self._selection.persons is not None and person_index not in self._selection.persons:
                continue

            def drawable(i: int, person: PersonPose = person) -> bool:
                return self._is_drawable(person, i)

            for a, b in topology.edges:
                if not (drawable(a) and drawable(b)):
                    continue
                pa = QPointF(float(person.keypoints[a][0]), float(person.keypoints[a][1]))
                pb = QPointF(float(person.keypoints[b][0]), float(person.keypoints[b][1]))
                weakest = min(float(person.scores[a]), float(person.scores[b]))
                bone = BoneItem(
                    pa,
                    pb,
                    self._color_for(topology.group_of(a) or "", detail),
                    self._color_for(topology.group_of(b) or "", detail),
                    confident=weakest >= DASH_BELOW,
                )
                scene.addItem(bone)
                self.bone_items.append(bone)
                self._confidence[id(bone)] = confidence_opacity(weakest, self._threshold)
                bone.setOpacity(self._target_opacity(bone))

            for i in range(len(person)):
                if not drawable(i):
                    continue
                group_name = topology.group_of(i) or ""
                if lod.is_face_group(group_name) and not detail.draws_points:
                    # The contours through these points are drawn either way;
                    # the dots are what would smear them.
                    continue
                score = float(person.scores[i])
                item = KeypointItem(
                    person_index=person_index,
                    keypoint_index=i,
                    group_name=group_name,
                    name=topology.keypoint_names[i],
                    color=self._color_for(group_name, detail),
                    score=score,
                    on_hover_enter=self._on_keypoint_hover_enter,
                    on_hover_leave=self._on_keypoint_hover_leave,
                )
                item.setPos(float(person.keypoints[i][0]), float(person.keypoints[i][1]))
                scene.addItem(item)
                self.keypoint_items.append(item)
                self._confidence[id(item)] = confidence_opacity(score, self._threshold)
                item.setOpacity(self._target_opacity(item))

    def _color_for(self, group: str, detail: lod.FaceDetail) -> QColor:
        """The colour a group draws in, which for the face depends on the detail level.

        The theme paints every face group one grey, because at a normal zoom
        that is all the face is: an outline. Once it is large enough for the
        regions to be told apart, they are given their own colours — the same
        strokes, saying more.
        """
        if detail.colours_by_region and group in lod.REGION_COLORS:
            return QColor(lod.REGION_COLORS[group])
        return theme.color_for(group)

    # --------------------------------------------------------------- opacity

    def _emphasis_factor(self, item: KeypointItem | BoneItem) -> float:
        """``DIM_FACTOR`` for anything outside the emphasised group.

        A bone spans two joints and carries no group of its own, so when a
        group is emphasised every bone dims — matching what
        ``SelectionController`` has always done.
        """
        if self._emphasis is None:
            return 1.0
        if isinstance(item, KeypointItem) and item.group_name == self._emphasis:
            return 1.0
        return DIM_FACTOR

    def _target_opacity(self, item: KeypointItem | BoneItem) -> float:
        """The opacity ``item`` *should* have right now: confidence × emphasis."""
        return self._confidence.get(id(item), 1.0) * self._emphasis_factor(item)

    def _apply_opacities(self) -> None:
        for item in self.keypoint_items:
            item.setOpacity(self._target_opacity(item))
        for bone in self.bone_items:
            bone.setOpacity(self._target_opacity(bone))

    # ------------------------------------------------------------- animation

    def animate_in(self, duration_ms: int = 400) -> None:
        """Stagger the skeleton into view: torso, then limbs, feet, hands, face.

        Costs nothing at runtime and is most of why a screenshot looks designed.
        Every item lands on its *own* target opacity — a marginal keypoint fades
        in to its faded value, not to 1.0 — and targets are read live, so an
        emphasis change mid-flight is honoured rather than overwritten.
        """
        self._stop_animation()
        for item in (*self.keypoint_items, *self.bone_items):
            item.setOpacity(0.0)

        animation = QVariantAnimation(self)
        animation.setStartValue(0.0)
        animation.setEndValue(1.0)
        animation.setDuration(duration_ms)
        animation.setEasingCurve(QEasingCurve.Type.OutCubic)
        animation.valueChanged.connect(self._on_animation_tick)
        animation.finished.connect(self._on_animation_finished)
        # Keep a reference: with only Qt's parent link, Python may collect the
        # wrapper (and its connected slots) mid-flight and the animation
        # silently never visibly runs.
        self._animation = animation
        animation.start()

    def _on_animation_tick(self, progress: float) -> None:
        # Start/end values are Python floats, so the QVariant unwraps to one.
        for item in self.keypoint_items:
            stage = ANIMATION_STAGE.get(item.group_name, _LAST_STAGE)
            local = min(1.0, max(0.0, progress * _STAGE_COUNT - stage))
            item.setOpacity(self._target_opacity(item) * local)
        for bone in self.bone_items:
            bone.setOpacity(self._target_opacity(bone) * progress)

    def _on_animation_finished(self) -> None:
        self._animation = None
        self._apply_opacities()

    def _stop_animation(self) -> None:
        """Abandon a running draw-in; the caller will set final opacities itself."""
        animation, self._animation = self._animation, None
        if animation is not None:
            animation.valueChanged.disconnect(self._on_animation_tick)
            animation.finished.disconnect(self._on_animation_finished)
            animation.stop()
            animation.deleteLater()

    # ----------------------------------------------------------------- hover

    def _on_keypoint_hover_enter(self, item: KeypointItem) -> None:
        self.keypoint_hovered.emit(item.person_index, item.keypoint_index, item.group_name)

    def _on_keypoint_hover_leave(self) -> None:
        self.keypoint_left.emit()

    # ----------------------------------------------------------------- bounds

    def group_bounds(self, person_index: int, group: str) -> QRectF:
        """The scene rectangle covering one group of one person — used to zoom to it.

        An empty rectangle for anything unanswerable — no result yet, no such
        person, no such group — because the callers are a zoom and a
        measurement, and both already do nothing with an empty one. The Face
        chip is clickable in the seconds between opening a photograph and its
        detection returning, and must be harmless there.
        """
        result = self._result
        person = self._person_at(person_index)
        if result is None or person is None:
            return QRectF()
        indices = result.topology.groups.get(group, ())
        if not indices:
            return QRectF()
        xs = [float(person.keypoints[i][0]) for i in indices]
        ys = [float(person.keypoints[i][1]) for i in indices]
        return QRectF(min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys))

    def _person_at(self, person_index: int) -> PersonPose | None:
        """One person of the current result, or ``None`` if there is no such person."""
        if self._result is None:
            return None
        people = self._result.frames[0].people
        if not 0 <= person_index < len(people):
            return None
        return people[person_index]

    def person_bounds(self, person_index: int) -> QRectF:
        person = self._person_at(person_index)
        if person is None:
            return QRectF()
        if person.bbox is not None:
            x0, y0, x1, y1 = person.bbox
            return QRectF(x0, y0, x1 - x0, y1 - y0)
        xs, ys = person.keypoints[:, 0], person.keypoints[:, 1]
        return QRectF(float(xs.min()), float(ys.min()), float(np.ptp(xs)), float(np.ptp(ys)))
