"""Feature extraction: the same hand pose must give the same 80 numbers wherever
the hand is, however big it looks, however it is rotated and whichever hand it is.

Each invariance has its own switch for the ablation study, so these tests also
check that switching one off really loses that invariance (and only that one).
"""

from __future__ import annotations

import numpy as np
import pytest

from airtouch.features import (
    NUM_FEATURES,
    extra_features,
    feature_matrix,
    feature_vector,
    normalize_landmarks,
)
from airtouch.landmarks import (
    INDEX_MCP,
    MIDDLE_MCP,
    NUM_LANDMARKS,
    PINKY_MCP,
    WRIST,
    anchor_point,
    hand_size,
    is_valid,
    palm_center,
    to_metric,
)
from ml.synthetic_data import GESTURES, HandPose, make_hand

ASPECT = 4 / 3
TOL = 1e-9
ALL_FLAGS = [
    dict(translate=t, rotate=r, scale=s) for t in (True, False) for r in (True, False) for s in (True, False)
]


def random_hand(gesture: str = "open_palm", seed: int = 0) -> np.ndarray:
    """A realistic hand: random position, size, tilt, side and jitter."""
    return make_hand(gesture, np.random.default_rng(seed))


def translated(lm: np.ndarray, dx: float = 0.13, dy: float = -0.07, dz: float = 0.02) -> np.ndarray:
    return lm + np.array([dx, dy, dz])


def scaled(lm: np.ndarray, k: float) -> np.ndarray:
    """Uniform scale in square units; x, y and z all shrink/grow together."""
    return lm * k


def rotated(lm: np.ndarray, degrees: float, aspect: float = ASPECT) -> np.ndarray:
    """In-plane rotation about the image centre, in square units (x is per-width)."""
    a = np.radians(degrees)
    m = to_metric(lm, aspect)
    x, y = m[:, 0] - aspect / 2, m[:, 1] - 0.5
    out = lm.copy()
    out[:, 0] = (np.cos(a) * x - np.sin(a) * y + aspect / 2) / aspect
    out[:, 1] = np.sin(a) * x + np.cos(a) * y + 0.5
    return out


def mirrored(lm: np.ndarray) -> np.ndarray:
    """The same hand seen as the other hand (flip left-right in the image)."""
    out = lm.copy()
    out[:, 0] = 1.0 - out[:, 0]
    return out


def difference(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.abs(a - b).max())


# ------------------------------------------------------------------ invariances
@pytest.mark.parametrize("gesture", GESTURES)
def test_translation_does_not_change_the_features(gesture):
    h = random_hand(gesture)
    assert difference(feature_vector(translated(h)), feature_vector(h)) < TOL


@pytest.mark.parametrize("gesture", GESTURES)
@pytest.mark.parametrize("k", [0.4, 2.5])
def test_uniform_scale_does_not_change_the_features(gesture, k):
    h = random_hand(gesture)
    assert difference(feature_vector(scaled(h, k)), feature_vector(h)) < TOL


@pytest.mark.parametrize("gesture", GESTURES)
@pytest.mark.parametrize("degrees", [17, 90, -120, 179])
def test_in_plane_rotation_does_not_change_the_features(gesture, degrees):
    h = random_hand(gesture)
    assert difference(feature_vector(rotated(h, degrees)), feature_vector(h)) < TOL


@pytest.mark.parametrize("gesture", GESTURES)
def test_left_and_right_hand_give_the_same_features(gesture):
    h = random_hand(gesture)
    assert difference(feature_vector(mirrored(h)), feature_vector(h)) < TOL


@pytest.mark.parametrize("gesture", GESTURES)
@pytest.mark.parametrize("seed", [1, 2, 3])
def test_all_transforms_at_once(gesture, seed):
    h = random_hand(gesture, seed)
    moved = translated(scaled(rotated(mirrored(h), 63), 0.6), 0.2, 0.1, -0.03)
    assert difference(feature_vector(moved), feature_vector(h)) < TOL


def test_generator_left_and_right_hands_match_with_any_roll():
    # Same seed = same finger shape; only the side and the in-plane angle differ.
    right = make_hand("point", np.random.default_rng(5), pose=HandPose(roll=0.0), noise=0.0)
    left = make_hand("point", np.random.default_rng(5), pose=HandPose(roll=30.0, mirror=True), noise=0.0)
    assert difference(feature_vector(left), feature_vector(right)) < TOL


def test_same_physical_hand_matches_across_camera_shapes():
    pose = HandPose(cx=0.5, cy=0.5, size=0.2, roll=20.0, pitch=10.0, yaw=-10.0)
    wide = make_hand("scroll", np.random.default_rng(3), pose=pose, aspect=16 / 9, noise=0.0)
    std = make_hand("scroll", np.random.default_rng(3), pose=pose, aspect=4 / 3, noise=0.0)
    assert difference(feature_vector(wide, 16 / 9), feature_vector(std, 4 / 3)) < TOL


def test_the_aspect_ratio_matters():
    h = random_hand("point", 1)
    assert difference(feature_vector(h, 16 / 9), feature_vector(h, 4 / 3)) > 0.05


# ------------------------------------------------------------------ ablation flags
@pytest.mark.parametrize("gesture", GESTURES)
def test_rotate_off_loses_rotation_and_side_invariance(gesture):
    h = random_hand(gesture)
    base = feature_vector(h, rotate=False)
    assert difference(feature_vector(rotated(h, 40), rotate=False), base) > 0.1
    assert difference(feature_vector(mirrored(h), rotate=False), base) > 0.1


@pytest.mark.parametrize("gesture", GESTURES)
def test_rotate_off_keeps_translation_and_scale_invariance(gesture):
    h = random_hand(gesture)
    base = feature_vector(h, rotate=False)
    assert difference(feature_vector(translated(h), rotate=False), base) < TOL
    assert difference(feature_vector(scaled(h, 2.0), rotate=False), base) < TOL


@pytest.mark.parametrize("gesture", GESTURES)
def test_translate_off_loses_translation_invariance(gesture):
    h = random_hand(gesture)
    base = feature_vector(h, translate=False)
    assert difference(feature_vector(translated(h), translate=False), base) > 0.1


@pytest.mark.parametrize("gesture", GESTURES)
def test_scale_off_loses_scale_invariance_only(gesture):
    h = random_hand(gesture)
    base = feature_vector(h, scale=False)
    assert difference(feature_vector(scaled(h, 2.0), scale=False), base) > 0.05
    assert difference(feature_vector(translated(h), scale=False), base) < TOL
    assert difference(feature_vector(rotated(h, 40), scale=False), base) < TOL
    assert difference(feature_vector(mirrored(h), scale=False), base) < TOL


def test_all_flags_off_is_just_the_metric_hand():
    h = random_hand("point", 2)
    out = normalize_landmarks(h, ASPECT, translate=False, rotate=False, scale=False)
    assert np.array_equal(out, to_metric(h, ASPECT))


@pytest.mark.parametrize("flags", ALL_FLAGS, ids=str)
def test_every_flag_combination_gives_a_different_view_of_the_hand(flags):
    h = random_hand("point", 2)
    others = [f for f in ALL_FLAGS if f != flags]
    base = feature_vector(h, **flags)
    assert all(difference(feature_vector(h, **f), base) > 1e-3 for f in others)


# --------------------------------------------------------------- canonical form
@pytest.mark.parametrize("seed", range(8))
def test_normalised_hand_is_in_its_canonical_pose(seed):
    n = normalize_landmarks(random_hand("open_palm", seed))
    assert np.allclose(n[WRIST], 0.0)
    assert np.allclose(n[MIDDLE_MCP, :2], [0.0, -1.0])     # one hand-size straight up
    assert n[INDEX_MCP, 0] < n[PINKY_MCP, 0]               # index knuckle on the left


def test_normalize_does_not_modify_its_input():
    h = random_hand("point", 3)
    before = h.copy()
    normalize_landmarks(h)
    feature_vector(h)
    assert np.array_equal(h, before)


# ---------------------------------------------------------------- shape and size
def test_feature_count_is_80():
    assert NUM_FEATURES == 80


@pytest.mark.parametrize("flags", ALL_FLAGS, ids=str)
def test_feature_vector_length_for_every_flag_combination(flags):
    assert feature_vector(random_hand(), **flags).shape == (NUM_FEATURES,)


def test_extra_features_are_the_last_seventeen_numbers():
    h = random_hand("fist", 4)
    n = normalize_landmarks(h)
    extra = extra_features(n)
    assert extra.shape == (NUM_FEATURES - NUM_LANDMARKS * 3,)
    assert np.array_equal(feature_vector(h), np.concatenate([n.reshape(-1), extra]))


def test_feature_matrix_stacks_the_vectors_and_forwards_the_flags():
    hands = [random_hand(g, i) for i, g in enumerate(GESTURES)]
    matrix = feature_matrix(hands, ASPECT, rotate=False)
    assert matrix.shape == (len(hands), NUM_FEATURES)
    for row, hand in zip(matrix, hands):
        assert np.array_equal(row, feature_vector(hand, ASPECT, rotate=False))


def test_feature_matrix_accepts_an_array_of_hands():
    hands = np.stack([random_hand(g, i) for i, g in enumerate(GESTURES)])
    assert feature_matrix(hands).shape == (len(GESTURES), NUM_FEATURES)


# ------------------------------------------------------------- degenerate hands
def degenerate_hands():
    collapsed_palm = random_hand("point", 5)
    collapsed_palm[MIDDLE_MCP] = collapsed_palm[WRIST]     # zero-length reference vector
    collinear = np.zeros((21, 3))
    collinear[:, 1] = np.linspace(0.0, 1.0, 21)
    return {
        "all-zero": np.zeros((21, 3)),
        "single-point": np.full((21, 3), 0.5),
        "collapsed-palm": collapsed_palm,
        "collinear": collinear,
    }


@pytest.mark.parametrize("name", list(degenerate_hands()))
@pytest.mark.parametrize("flags", ALL_FLAGS, ids=str)
def test_degenerate_hand_never_produces_nan_or_inf(name, flags):
    out = feature_vector(degenerate_hands()[name], **flags)
    assert out.shape == (NUM_FEATURES,)
    assert np.all(np.isfinite(out))


# ----------------------------------------------------------- what the numbers say
def test_a_closed_pinch_shows_up_in_the_extra_features():
    closed = make_hand("pinch_index", np.random.default_rng(0), pose=HandPose(), thumb_gap=0.0, noise=0.0)
    wide = make_hand("open_palm", np.random.default_rng(0), pose=HandPose(), noise=0.0)
    thumb_to_index = NUM_LANDMARKS * 3 + 5            # after the 5 tip-to-wrist distances
    assert feature_vector(closed)[thumb_to_index] == pytest.approx(0.0, abs=1e-9)
    assert feature_vector(wide)[thumb_to_index] > 0.5


def test_different_gestures_are_further_apart_than_tremor_on_one_gesture():
    rng = np.random.default_rng(0)
    pose = HandPose()
    fist = make_hand("fist", rng, pose=pose, noise=0.0)
    palm = make_hand("open_palm", rng, pose=pose, noise=0.0)
    trembling = fist + rng.normal(0, 0.0015, fist.shape)
    tremor = np.linalg.norm(feature_vector(trembling) - feature_vector(fist))
    between = np.linalg.norm(feature_vector(palm) - feature_vector(fist))
    assert between > 5 * tremor


# ------------------------------------------------------------- landmark helpers
def test_is_valid_accepts_only_finite_21_by_3():
    good = random_hand()
    assert is_valid(good)
    assert is_valid(good.tolist())
    assert not is_valid(None)
    assert not is_valid(np.zeros((20, 3)))
    assert not is_valid(np.zeros((21, 2)))
    assert not is_valid(np.zeros(63))
    bad = good.copy()
    bad[7, 1] = np.nan
    assert not is_valid(bad)
    bad[7, 1] = np.inf
    assert not is_valid(bad)


@pytest.mark.parametrize("bad", [
    [["bad", 0, 0]] * 21,
    [[0], [0, 1]],
    [[None] * 3] * 21,
    [[10 ** 400] * 3] * 21,
], ids=["text", "ragged", "missing-values", "overflow"])
def test_malformed_landmarks_are_rejected_without_an_exception(bad):
    assert not is_valid(bad)


@pytest.mark.parametrize("bad", [[["bad", 0, 0]] * 21, [[0], [0, 1]]], ids=["text", "ragged"])
def test_engine_treats_malformed_landmarks_as_a_lost_hand(bad):
    from airtouch.config import Config
    from airtouch.controller import FakeBackend
    from airtouch.engine import GestureEngine

    engine = GestureEngine(Config(), FakeBackend())
    result = engine.process(bad, 0.0)
    assert result.hands == 0 and result.events == []

    result = engine.process_hands([bad], 0.1)
    assert result.hands == 0 and result.events == []


def test_to_metric_scales_only_x_and_copies():
    h = random_hand()
    m = to_metric(h, 2.0)
    assert np.array_equal(m[:, 0], h[:, 0] * 2.0)
    assert np.array_equal(m[:, 1:], h[:, 1:])
    m[0, 0] = 123.0
    assert h[0, 0] != 123.0


def test_hand_size_is_never_zero():
    assert hand_size(np.zeros((21, 3))) > 0


def test_hand_size_and_palm_center_match_the_generators_pose():
    pose = HandPose(cx=0.4, cy=0.6, size=0.2)
    m = to_metric(make_hand("point", np.random.default_rng(0), pose=pose, noise=0.0), ASPECT)
    assert hand_size(m) == pytest.approx(pose.size)
    assert palm_center(m) == pytest.approx([pose.cx * ASPECT, pose.cy])


def test_anchor_point_ignores_the_fingertips():
    # pinching moves the fingertips, never the knuckles or wrist: the cursor must not jump
    h = random_hand("point", 6)
    moved_tips = h.copy()
    moved_tips[[4, 8, 12, 16, 20], :2] += 0.1
    assert np.array_equal(anchor_point(h), anchor_point(moved_tips))
