"""The 1€ filter and the per-track smoother built on it."""

from dataclasses import replace

import numpy as np

from alchemypose.core.schema import PersonPose, PoseFrame
from alchemypose.core.smoothing import OneEuroFilter, PoseSmoother, SmoothingSettings, smooth_frames

K = 133
FPS = 30.0


def _person(points: np.ndarray, score: float = 0.9, track: int | None = 0) -> PersonPose:
    return PersonPose(
        keypoints=points.astype(np.float32), scores=np.full(K, score, dtype=np.float32), track_id=track
    )


def test_a_still_signal_comes_out_unchanged() -> None:
    f = OneEuroFilter()
    x = np.full((K, 2), 123.0, dtype=np.float32)
    for i in range(30):
        out = f(x, i / FPS)
    assert np.allclose(out, 123.0)


def test_jitter_on_a_still_joint_is_reduced() -> None:
    rng = np.random.default_rng(0)
    f = OneEuroFilter(min_cutoff=1.0, beta=0.05)
    raw, filtered = [], []
    for i in range(120):
        x = (100.0 + rng.normal(0, 3.0, (K, 2))).astype(np.float32)
        raw.append(x)
        filtered.append(f(x, i / FPS))
    assert np.std(np.array(filtered[30:])) < 0.5 * np.std(np.array(raw[30:]))


def test_a_fast_move_is_followed_rather_than_dragged() -> None:
    """beta raises the cutoff with speed: after a sustained move the lag stays small."""
    f = OneEuroFilter(min_cutoff=1.0, beta=0.05)
    for i in range(60):
        out = f(np.full((K, 2), 10.0 * i, dtype=np.float32), i / FPS)    # 300 px/s
    assert abs(float(out[0, 0]) - 590.0) < 30.0


def test_invalid_rows_are_shown_raw_and_do_not_disturb_the_state() -> None:
    f = OneEuroFilter()
    still = np.full((K, 2), 50.0, dtype=np.float32)
    for i in range(10):
        f(still, i / FPS)
    wild = still.copy()
    wild[3] = (900.0, 900.0)
    valid = np.ones(K, dtype=bool)
    valid[3] = False
    out = f(wild, 10 / FPS, valid)
    assert out[3].tolist() == [900.0, 900.0]                 # shown as the model said
    after = f(still, 11 / FPS)
    assert np.allclose(after[3], 50.0, atol=1e-3)            # but never fed to the filter


def test_smoothing_never_modifies_the_frames_it_is_given() -> None:
    rng = np.random.default_rng(1)
    frames = tuple(
        PoseFrame(i, i / FPS, (640, 480), (_person(rng.uniform(0, 640, (K, 2))),)) for i in range(10)
    )
    before = [f.people[0].keypoints.copy() for f in frames]
    smoothed = smooth_frames(frames, FPS)
    assert all(np.array_equal(f.people[0].keypoints, b) for f, b in zip(frames, before, strict=True))
    assert not np.array_equal(smoothed[5].people[0].keypoints, frames[5].people[0].keypoints)


def test_tracks_are_smoothed_independently() -> None:
    smoother = PoseSmoother()
    a = _person(np.full((K, 2), 0.0), track=0)
    b = _person(np.full((K, 2), 500.0), track=1)
    for i in range(5):
        out = smoother.frame(PoseFrame(i, i / FPS, (640, 480), (a, b)), FPS)
    assert np.allclose(out.people[0].keypoints, 0.0)
    assert np.allclose(out.people[1].keypoints, 500.0)


def test_a_track_that_returns_after_a_long_gap_starts_afresh() -> None:
    smoother = PoseSmoother(SmoothingSettings(reset_after_s=0.5))
    smoother.person(_person(np.full((K, 2), 0.0)), 0.0)
    smoother.person(_person(np.full((K, 2), 0.0)), 0.1)
    back = smoother.person(_person(np.full((K, 2), 400.0)), 2.0)    # 1.9 s later, elsewhere
    assert np.allclose(back.keypoints, 400.0), "no smear from where it was two seconds ago"


def test_untracked_people_pass_through() -> None:
    person = _person(np.full((K, 2), 7.0), track=None)
    assert PoseSmoother().person(person, 0.0) is person


def test_3d_keypoints_are_smoothed_too() -> None:
    smoother = PoseSmoother()
    rng = np.random.default_rng(2)
    raw, out = [], []
    for i in range(90):
        kp3 = (1.0 + rng.normal(0, 0.02, (K, 3))).astype(np.float32)
        raw.append(kp3)
        person = replace(_person(np.zeros((K, 2))), keypoints_3d=kp3)
        out.append(smoother.person(person, i / FPS).keypoints_3d)
    assert np.std(np.array(out[30:])) < 0.5 * np.std(np.array(raw[30:]))
