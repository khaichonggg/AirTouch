"""Rule-based recogniser: the six gestures on synthetic hands, plus the pinch
logic that decides when a button may go down and when it must come up."""

from __future__ import annotations

import numpy as np
import pytest

from airtouch.gestures_rule import (
    EXTENDED_RATIO,
    GESTURES,
    HOLD_REACH,
    NONE,
    PRESS_REACH,
    RuleRecognizer,
    classify,
    finger_extension,
    finger_reach,
    pinch_gap,
    pinch_gaps,
    pinch_held,
    which_pinch,
)
from airtouch.landmarks import INDEX_MCP, INDEX_TIP, MIDDLE_MCP, MIDDLE_TIP, THUMB_TIP, to_metric
from ml.synthetic_data import ALL_GESTURES, DIFFICULTY, HandPose, generate_dataset, make_hand
from ml.synthetic_data import GESTURES as GENERATOR_GESTURES

ASPECT = 4 / 3
PINCH_ON = 0.20
PINCH_OFF = 0.32

# Fingertips of the stick hand sit on two rays, 12 degrees either side of straight up.
_RAY_X, _RAY_Y = float(np.sin(np.radians(12))), float(np.cos(np.radians(12)))


def stick_hand(thumb, index_reach: float = 1.2, middle_reach: float = 1.2) -> np.ndarray:
    """Metric hand holding only the points the pinch tests read.

    Both knuckles sit exactly one hand-size from the wrist (at the origin), so
    reach and gaps can be read straight off the coordinates.
    """
    m = np.zeros((21, 3))
    m[INDEX_MCP] = m[MIDDLE_MCP] = (0.0, 1.0, 0.0)
    m[INDEX_TIP] = (-_RAY_X * index_reach, _RAY_Y * index_reach, 0.0)
    m[MIDDLE_TIP] = (_RAY_X * middle_reach, _RAY_Y * middle_reach, 0.0)
    m[THUMB_TIP] = (thumb[0], thumb[1], 0.0)
    return m


def index_tip(reach: float = 1.2) -> np.ndarray:
    return np.array([-_RAY_X * reach, _RAY_Y * reach])


def middle_tip(reach: float = 1.2) -> np.ndarray:
    return np.array([_RAY_X * reach, _RAY_Y * reach])


def left_pinch(gap: float, reach: float = 1.2) -> np.ndarray:
    """Thumb exactly ``gap`` hand-sizes from the index tip, on the far side from the middle tip."""
    return stick_hand(index_tip(reach) - [gap, 0.0], index_reach=reach)


def right_pinch(gap: float, reach: float = 1.2) -> np.ndarray:
    """Thumb exactly ``gap`` hand-sizes from the middle tip, on the far side from the index tip."""
    return stick_hand(middle_tip(reach) + [gap, 0.0], middle_reach=reach)


def accuracy(gesture: str, seed: int, n: int = 300, **kwargs) -> float:
    rng = np.random.default_rng(seed)
    return float(np.mean([classify(make_hand(gesture, rng, **kwargs)) == gesture for _ in range(n)]))


@pytest.fixture(scope="module")
def standard_set():
    """The project's default data set: 300 normal samples per gesture, seed 0."""
    return generate_dataset(300, GENERATOR_GESTURES, "normal", seed=0)


# ---------------------------------------------------------------- core gestures
def test_rules_and_generator_agree_on_the_gesture_set():
    assert set(GESTURES) == set(GENERATOR_GESTURES)


@pytest.mark.parametrize("gesture", GESTURES)
def test_each_gesture_is_recognised_on_300_normal_samples(standard_set, gesture):
    hands, labels = standard_set
    samples = hands[labels == gesture]
    assert len(samples) == 300
    assert np.mean([classify(h) == gesture for h in samples]) >= 0.98


@pytest.mark.parametrize("gesture", GESTURES)
def test_recognition_does_not_hinge_on_one_lucky_seed(gesture):
    # pinch_middle is the weakest gesture: its true rate is about 98.5%, so a single
    # 300-sample estimate can dip below 98% (97.7% seen on seed 2); more samples cannot.
    assert np.mean([accuracy(gesture, seed=s) for s in (1, 2, 3, 4)]) >= 0.98


@pytest.mark.parametrize("gesture", GESTURES)
def test_each_gesture_is_recognised_on_a_wide_camera(gesture):
    aspect = 16 / 9
    rng = np.random.default_rng(7)
    hits = [classify(make_hand(gesture, rng, aspect=aspect), aspect=aspect) == gesture for _ in range(100)]
    assert np.mean(hits) >= 0.95


@pytest.mark.parametrize("gesture", ALL_GESTURES)
@pytest.mark.parametrize("difficulty", list(DIFFICULTY))
def test_classify_only_ever_returns_known_labels(gesture, difficulty):
    rng = np.random.default_rng(1)
    allowed = set(GESTURES) | {NONE}
    assert {classify(make_hand(gesture, rng, difficulty=difficulty)) for _ in range(20)} <= allowed


def test_degenerate_hand_does_not_crash():
    assert classify(np.zeros((21, 3))) in set(GESTURES) | {NONE}
    assert classify(np.full((21, 3), 0.5)) in set(GESTURES) | {NONE}


# ------------------------------------------------------------------- invariance
def transformed_copies(lm: np.ndarray):
    """The same hand moved, resized, rotated and seen as the other hand."""
    yield lm + [0.2, -0.1, 0.05]
    yield lm + [-0.3, 0.25, 0.0]
    yield lm * 1.7
    yield lm * 0.4
    a = np.radians(35.0)
    m = to_metric(lm, ASPECT)
    out = lm.copy()
    out[:, 0] = (np.cos(a) * m[:, 0] - np.sin(a) * m[:, 1]) / ASPECT
    out[:, 1] = np.sin(a) * m[:, 0] + np.cos(a) * m[:, 1]
    yield out
    flipped = lm.copy()
    flipped[:, 0] = 1.0 - flipped[:, 0]
    yield flipped


@pytest.mark.parametrize("gesture", GESTURES)
def test_classify_does_not_care_where_the_hand_is_or_how_big_it_looks(gesture):
    rng = np.random.default_rng(3)
    for _ in range(40):
        hand = make_hand(gesture, rng)
        label = classify(hand)
        assert all(classify(other) == label for other in transformed_copies(hand))


# -------------------------------------------------------- pinch vs. curled fingers
CURLED_PINCH = {"index": 0.6, "middle": 0.7, "ring": 0.8, "pinky": 0.8}


def test_heavily_curled_pinch_facing_the_camera_is_still_a_pinch():
    rng = np.random.default_rng(0)
    labels = [classify(make_hand("pinch_index", rng, pose=HandPose(), curls=CURLED_PINCH, thumb_gap=0.03))
              for _ in range(300)]
    assert labels.count("pinch_index") / 300 >= 0.98


def test_heavily_curled_pinch_in_random_poses_is_mostly_still_a_pinch():
    # Tilting the hand towards the camera foreshortens the reach, so a share of these
    # really do look tucked; the point is that most are told apart from a fist.
    rng = np.random.default_rng(0)
    labels = [classify(make_hand("pinch_index", rng, curls=CURLED_PINCH, thumb_gap=0.03)) for _ in range(300)]
    assert labels.count("pinch_index") / 300 >= 0.80


def test_fists_are_rejected_by_the_same_thresholds():
    rng = np.random.default_rng(0)
    labels = [classify(make_hand("fist", rng)) for _ in range(300)]
    assert labels.count("fist") == 300


def test_which_pinch_is_none_for_500_random_fists():
    rng = np.random.default_rng(0)
    assert all(which_pinch(to_metric(make_hand("fist", rng), ASPECT), PINCH_ON) is None for _ in range(500))


def test_false_press_rate_on_fists_stays_tiny_across_seeds():
    # A fist's reach (max ~0.99) and thumb gap (min ~0.18) overlap the press test in very
    # rare corner cases; in the app the debounced "fist" label blocks those presses as well.
    presses = 0
    for seed in range(1, 6):
        rng = np.random.default_rng(seed)
        presses += sum(which_pinch(to_metric(make_hand("fist", rng), ASPECT), PINCH_ON) is not None
                       for _ in range(500))
    assert presses / 2500 <= 0.002


def test_which_pinch_is_never_right_for_500_random_point_poses():
    # In the pointing pose the thumb rests on the curled middle finger, close to its tip.
    rng = np.random.default_rng(0)
    assert all(which_pinch(to_metric(make_hand("point", rng), ASPECT), PINCH_ON) != "right" for _ in range(500))


@pytest.mark.parametrize("gesture, button", [("pinch_index", "left"), ("pinch_middle", "right")])
def test_real_pinches_pick_the_right_button(standard_set, gesture, button):
    hands, labels = standard_set
    hits = [which_pinch(to_metric(h, ASPECT), PINCH_ON) == button for h in hands[labels == gesture]]
    assert np.mean(hits) >= 0.98


# ------------------------------------------------------------------- which_pinch
def test_thumb_on_the_index_tip_is_left_and_on_the_middle_tip_is_right():
    assert which_pinch(stick_hand(index_tip()), PINCH_ON) == "left"
    assert which_pinch(stick_hand(middle_tip()), PINCH_ON) == "right"


def test_thumb_far_from_both_is_no_pinch():
    assert which_pinch(stick_hand((5.0, 5.0)), PINCH_ON) is None


def test_press_threshold_is_strict():
    assert which_pinch(left_pinch(PINCH_ON - 0.01), PINCH_ON) == "left"
    assert which_pinch(left_pinch(PINCH_ON + 0.01), PINCH_ON) is None
    assert which_pinch(right_pinch(PINCH_ON - 0.01), PINCH_ON) == "right"
    assert which_pinch(right_pinch(PINCH_ON + 0.01), PINCH_ON) is None


@pytest.mark.parametrize("mix, expected", [(0.4, "left"), (0.5, "left"), (0.6, "right")])
def test_the_nearer_finger_wins_when_both_are_close(mix, expected):
    # Tips ~0.5 apart; the wide threshold puts both within reach of the thumb.
    thumb = index_tip() + mix * (middle_tip() - index_tip())
    assert which_pinch(stick_hand(thumb), threshold=0.45) == expected


def test_a_tucked_index_finger_cannot_press():
    assert which_pinch(left_pinch(0.0, reach=PRESS_REACH["left"] - 0.02), PINCH_ON) is None
    assert which_pinch(left_pinch(0.0, reach=PRESS_REACH["left"] + 0.02), PINCH_ON) == "left"


def test_a_tucked_middle_finger_cannot_press():
    assert which_pinch(right_pinch(0.0, reach=PRESS_REACH["right"] - 0.02), PINCH_ON) is None
    assert which_pinch(right_pinch(0.0, reach=PRESS_REACH["right"] + 0.02), PINCH_ON) == "right"


def test_the_middle_finger_has_the_stricter_press_test():
    # Reach 0.90 is good enough for the index finger but not the middle finger:
    # that is what stops the thumb resting on a curled middle finger right-clicking.
    assert PRESS_REACH["right"] > 0.90 > PRESS_REACH["left"]
    assert which_pinch(left_pinch(0.0, reach=0.90), PINCH_ON) == "left"
    assert which_pinch(right_pinch(0.0, reach=0.90), PINCH_ON) is None


def test_a_tucked_finger_lets_the_other_finger_win_even_if_it_is_nearer():
    # Index tucked and nearer to the thumb -> the middle finger gets the press.
    index, middle = index_tip(0.80), middle_tip(1.0)
    thumb = index + 0.4 * (middle - index)
    assert which_pinch(stick_hand(thumb, index_reach=0.80, middle_reach=1.0), threshold=0.4) == "right"
    # Middle tucked and nearer to the thumb -> the index finger gets the press.
    index, middle = index_tip(1.2), middle_tip(0.90)
    thumb = index + 0.6 * (middle - index)
    assert which_pinch(stick_hand(thumb, index_reach=1.2, middle_reach=0.90), threshold=0.4) == "left"


def test_pinch_is_measured_in_hand_sizes():
    near = left_pinch(0.1)
    far_from_camera = near * 0.25
    assert which_pinch(far_from_camera, PINCH_ON) == "left"
    assert pinch_gaps(far_from_camera) == pytest.approx(pinch_gaps(near))


# -------------------------------------------------------------------- pinch_held
def test_held_button_tolerates_the_zone_between_the_two_thresholds():
    assert pinch_held(left_pinch(0.26), "left", PINCH_OFF)
    assert pinch_held(right_pinch(0.26), "right", PINCH_OFF)


def test_held_button_lets_go_when_the_fingers_clearly_part():
    assert not pinch_held(left_pinch(PINCH_OFF + 0.01), "left", PINCH_OFF)
    assert not pinch_held(right_pinch(PINCH_OFF + 0.01), "right", PINCH_OFF)


@pytest.mark.parametrize("button, make", [("left", left_pinch), ("right", right_pinch)])
def test_hold_test_is_looser_than_the_press_test(button, make):
    assert HOLD_REACH[button] < PRESS_REACH[button]
    between = (HOLD_REACH[button] + PRESS_REACH[button]) / 2
    hand = make(0.05, reach=between)
    assert which_pinch(hand, PINCH_ON) is None      # could not start a press...
    assert pinch_held(hand, button, PINCH_OFF)      # ...but does not drop a drag either


@pytest.mark.parametrize("button, make", [("left", left_pinch), ("right", right_pinch)])
def test_finger_tucked_below_the_hold_limit_lets_go_even_when_touching(button, make):
    assert not pinch_held(make(0.0, reach=HOLD_REACH[button] - 0.02), button, PINCH_OFF)


def test_pinch_gap_picks_the_button_finger():
    m = left_pinch(0.1)
    d_index, d_middle = pinch_gaps(m)
    assert d_index == pytest.approx(0.1)
    assert d_middle > d_index
    assert pinch_gap(m, "left") == d_index
    assert pinch_gap(m, "right") == d_middle


# ------------------------------------------------------------- finger measurements
def test_finger_reach_reads_tip_distance_over_knuckle_distance():
    m = stick_hand((5.0, 5.0), index_reach=0.8, middle_reach=1.3)
    assert finger_reach(m, "index") == pytest.approx(0.8)
    assert finger_reach(m, "middle") == pytest.approx(1.3)


def test_straight_fingers_count_as_extended_and_curled_ones_do_not():
    rng = np.random.default_rng(0)
    straight = make_hand("open_palm", rng, pose=HandPose(), curls=dict.fromkeys(("index", "middle", "ring", "pinky"), 0.0), noise=0.0)
    curled = make_hand("fist", rng, pose=HandPose(), curls=dict.fromkeys(("index", "middle", "ring", "pinky"), 1.0), noise=0.0)
    assert all(v > EXTENDED_RATIO for v in finger_extension(to_metric(straight, ASPECT)).values())
    assert all(v < EXTENDED_RATIO for v in finger_extension(to_metric(curled, ASPECT)).values())


def test_reach_separates_a_real_pinch_from_a_fist():
    rng = np.random.default_rng(0)
    pinch = np.mean([finger_reach(to_metric(make_hand("pinch_index", rng), ASPECT), "index") for _ in range(200)])
    fist = np.mean([finger_reach(to_metric(make_hand("fist", rng), ASPECT), "index") for _ in range(200)])
    assert pinch > PRESS_REACH["left"] > fist


# ---------------------------------------------------------------- RuleRecognizer
@pytest.mark.parametrize("gesture", GESTURES)
def test_recognizer_matches_classify_with_full_confidence(gesture):
    rng = np.random.default_rng(1)
    rec = RuleRecognizer(ASPECT, PINCH_ON)
    for _ in range(10):
        hand = make_hand(gesture, rng)
        assert rec.predict(hand) == (classify(hand, ASPECT, PINCH_ON), 1.0)


def test_recognizer_uses_its_aspect_and_pinch_threshold():
    hand = make_hand("pinch_index", np.random.default_rng(0), pose=HandPose(), curls=CURLED_PINCH, thumb_gap=0.05, noise=0.0)
    assert RuleRecognizer(ASPECT, 0.20).predict(hand)[0] == "pinch_index"
    assert RuleRecognizer(ASPECT, 1e-6).predict(hand)[0] != "pinch_index"
    stretched = make_hand("point", np.random.default_rng(0), pose=HandPose(), aspect=16 / 9, noise=0.0)
    assert RuleRecognizer(16 / 9, 0.20).predict(stretched)[0] == "point"
