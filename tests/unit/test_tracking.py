"""Frame-to-frame identity: who is who across a clip."""

from dataclasses import replace

import numpy as np

from alchemypose.core.schema import PersonPose, PoseFrame
from alchemypose.core.tracking import (
    Tracker,
    absent_person,
    assign_tracks,
    by_track,
    iou,
    keypoint_box,
    renumber_by_presence,
    track_count,
)

K = 133


def _person(x0: float, y0: float, w: float = 100.0, h: float = 200.0, score: float = 0.9) -> PersonPose:
    """133 keypoints spread over a box, so the keypoint extent is that box."""
    rng = np.random.default_rng(int(x0 * 7 + y0))
    points = np.column_stack([x0 + rng.uniform(0, w, K), y0 + rng.uniform(0, h, K)]).astype(np.float32)
    points[0], points[1] = (x0, y0), (x0 + w, y0 + h)          # pin the extent exactly
    return PersonPose(keypoints=points, scores=np.full(K, score, dtype=np.float32), bbox=(x0, y0, x0 + w, y0 + h))


def _frame(index: int, *people: PersonPose) -> PoseFrame:
    return PoseFrame(frame_index=index, timestamp_s=index / 30, image_size=(1000, 800), people=people)


def test_iou_of_identical_boxes_is_one_and_of_disjoint_boxes_zero() -> None:
    assert iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0
    assert iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0


def test_keypoint_box_ignores_doubtful_joints() -> None:
    person = _person(100, 100)
    scores = person.scores.copy()
    keypoints = person.keypoints.copy()
    keypoints[5] = (900, 700)          # a wild outlier ...
    scores[5] = 0.05                   # ... that the model was not sure of
    box = keypoint_box(replace(person, keypoints=keypoints, scores=scores))
    assert box == (100.0, 100.0, 200.0, 300.0)


def test_two_people_keep_their_ids_as_they_move_and_the_list_order_flips() -> None:
    tracker = Tracker()
    first = tracker.update((_person(100, 100), _person(600, 100)))
    # Next frame: both drift right a little, and the detector lists them the other way round.
    second = tracker.update((_person(610, 105), _person(110, 105)))
    assert second == [first[1], first[0]]


def test_a_newcomer_gets_a_fresh_id() -> None:
    tracker = Tracker()
    (a,) = tracker.update((_person(100, 100),))
    ids = tracker.update((_person(105, 100), _person(700, 300)))
    assert ids[0] == a
    assert ids[1] not in (a,)


def test_someone_gone_longer_than_max_age_returns_as_someone_new() -> None:
    tracker = Tracker(max_age=3)
    (a,) = tracker.update((_person(100, 100),))
    for _ in range(4):
        tracker.update(())
    (b,) = tracker.update((_person(100, 100),))
    assert b != a


def test_a_brief_occlusion_within_max_age_keeps_the_id() -> None:
    tracker = Tracker(max_age=5)
    (a,) = tracker.update((_person(100, 100),))
    tracker.update(())
    tracker.update(())
    (b,) = tracker.update((_person(104, 100),))
    assert b == a


def test_renumbering_makes_the_longest_present_track_zero() -> None:
    frames = (
        _frame(0, replace(_person(0, 0), track_id=7)),
        _frame(1, replace(_person(0, 0), track_id=7), replace(_person(500, 0), track_id=3)),
        _frame(2, replace(_person(500, 0), track_id=3)),
        _frame(3, replace(_person(500, 0), track_id=3)),
    )
    renumbered = renumber_by_presence(frames)
    assert renumbered[3].people[0].track_id == 0          # track 3 was in three frames
    assert renumbered[0].people[0].track_id == 1          # track 7 in two
    assert track_count(renumbered) == 2


def test_assign_tracks_follows_one_walker_across_a_clip() -> None:
    frames = tuple(_frame(i, _person(100 + 5 * i, 100)) for i in range(20))
    tracked = assign_tracks(frames)
    assert {p.track_id for f in tracked for p in f.people} == {0}


def test_by_track_puts_each_track_at_its_own_index_and_fills_gaps_with_nobody() -> None:
    frame = _frame(0, replace(_person(500, 0), track_id=2), replace(_person(0, 0), track_id=0))
    view = by_track(frame, tracks=3, keypoint_count=K)
    assert len(view.people) == 3
    assert view.people[0].track_id == 0
    assert view.people[2].track_id == 2
    assert np.all(view.people[1].scores == 0.0), "track 1 is absent here: score 0 is never drawn"


def test_by_track_drops_untracked_people_from_the_view_only() -> None:
    frame = _frame(0, _person(0, 0), replace(_person(500, 0), track_id=0))
    view = by_track(frame, tracks=1, keypoint_count=K)
    assert len(view.people) == 1 and view.people[0].track_id == 0
    assert len(frame.people) == 2


def test_an_absent_person_is_all_zero() -> None:
    person = absent_person(K)
    assert person.keypoints.shape == (K, 2)
    assert not person.scores.any()
