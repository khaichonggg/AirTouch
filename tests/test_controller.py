"""MouseController with a FakeBackend: real-time clicking, dragging, scrolling,
dwell and the camera-to-screen mapping. The real mouse is never touched.

Multi-frame tests hold a pose the way a real hand does: one fixed finger shape
per test (``make_hand`` re-rolls joint noise on every call, so a noiseless
template is generated once), moved as a whole, with optional small tremor.
"""

from __future__ import annotations

import re

import numpy as np
import pytest

from airtouch.config import Config
from airtouch.controller import FakeBackend, MouseController
from airtouch.gestures_rule import HOLD_REACH, PRESS_REACH, finger_reach, pinch_gaps
from airtouch.landmarks import INDEX_TIP, MIDDLE_TIP, THUMB_TIP, anchor_point, hand_size, to_metric
from ml.synthetic_data import HandPose, make_hand

ASPECT = 4 / 3
DT = 1 / 30
POSE = HandPose(cx=0.5, cy=0.55, size=0.16)
TREMOR = 0.0015            # per-landmark jitter of a hand held still


def shape(gesture: str, index: float, middle: float, ring: float, pinky: float, pose=POSE, seed=1, **kw) -> np.ndarray:
    """One fixed, noiseless finger shape."""
    curls = {"index": index, "middle": middle, "ring": ring, "pinky": pinky}
    return make_hand(gesture, np.random.default_rng(seed), pose=pose, curls=curls, noise=0.0, **kw)


POINT = shape("point", 0.05, 0.9, 0.9, 0.9)
V_SIGN = shape("scroll", 0.05, 0.05, 0.9, 0.9)
FIST = shape("fist", 0.9, 0.9, 0.9, 0.9)
OPEN = shape("open_palm", 0.05, 0.05, 0.05, 0.05)
PINCH_LEFT = shape("pinch_index", 0.4, 0.7, 0.7, 0.7, thumb_gap=0.0)
PINCH_RIGHT = shape("pinch_middle", 0.05, 0.45, 0.7, 0.7, thumb_gap=0.0)


def moved(hand: np.ndarray, dx: float = 0.0, dy: float = 0.0) -> np.ndarray:
    """The whole hand shifted, in normalised image units."""
    return hand + np.array([dx, dy, 0.0])


def with_gap(hand: np.ndarray, button: str, gap: float) -> np.ndarray:
    """Copy of ``hand`` with the thumb tip exactly ``gap`` hand-sizes from the finger tip.

    The thumb goes on the far side from the other fingertip, so the gap to the
    button's own finger is always the smaller of the two.
    """
    m = to_metric(hand, ASPECT)
    tip, other = (INDEX_TIP, MIDDLE_TIP) if button == "left" else (MIDDLE_TIP, INDEX_TIP)
    away = m[tip, :2] - m[other, :2]
    target = m[tip, :2] + away / np.linalg.norm(away) * gap * hand_size(m)
    out = hand.copy()
    out[THUMB_TIP, :2] = target[0] / ASPECT, target[1]
    return out


TOUCHING = with_gap(PINCH_LEFT, "left", 0.05)
APART = with_gap(PINCH_LEFT, "left", 0.6)
TOUCHING_RIGHT = with_gap(PINCH_RIGHT, "right", 0.05)
APART_RIGHT = with_gap(PINCH_RIGHT, "right", 0.6)


class Rig:
    """A controller on a fake backend, with a 30 fps clock."""

    def __init__(self, screen=(1920, 1080), **settings):
        self.cfg = Config(**settings)
        self.be = FakeBackend(*screen)
        self.c = MouseController(self.cfg, self.be)
        self.t = 0.0

    def step(self, label: str, hand, **settings):
        self.t += DT
        return self.c.update(label, hand, self.t, **settings)

    def hold(self, label: str, hand, frames: int):
        return [self.step(label, hand) for _ in range(frames)]

    def moves(self) -> int:
        return self.be.count("move")

    def scrolls(self) -> list[float]:
        return [e[1] for e in self.be.events if e[0] == "scroll"]


def test_the_templates_have_the_geometry_these_tests_rely_on():
    for hand, button, finger in ((TOUCHING, "left", "index"), (TOUCHING_RIGHT, "right", "middle")):
        m = to_metric(hand, ASPECT)
        assert finger_reach(m, finger) > PRESS_REACH[button]
    assert pinch_gaps(to_metric(TOUCHING, ASPECT))[0] == pytest.approx(0.05)
    assert pinch_gaps(to_metric(TOUCHING_RIGHT, ASPECT))[1] == pytest.approx(0.05)
    assert pinch_gaps(to_metric(APART, ASPECT))[0] == pytest.approx(0.6)


# ------------------------------------------------------------------ left button
def test_button_goes_down_on_the_first_contact_frame_even_if_the_label_is_still_point():
    rig = Rig()
    assert rig.step("point", APART).button is None
    assert rig.be.count("down") == 0
    r = rig.step("point", TOUCHING)       # the debounced label has not caught up yet
    assert "down:left" in r.events
    assert r.button == "left"
    assert rig.be.down == {"left"}
    assert rig.be.count("down", "left") == 1


@pytest.mark.parametrize("before", ["first-frame", "moving", "paused"])
def test_a_new_pinch_moves_to_its_target_before_pressing(before):
    rig = Rig()
    hand = TOUCHING
    if before != "first-frame":
        rig.step("point", APART)
        if before == "paused":
            rig.step("open_palm", OPEN)
            rig.be.move(20, 30)       # user moved the physical mouse while idle
        else:
            hand = moved(TOUCHING, dx=0.1)
    rig.be.events.clear()

    r = rig.step("pinch_index", hand)

    assert rig.be.events == [("move", *r.cursor), ("down", "left")]


def test_resuming_motion_restores_the_target_after_the_physical_mouse_moved():
    rig = Rig()
    target = rig.step("point", APART).cursor
    rig.c.halt()
    rig.be.move(20, 30)
    rig.be.events.clear()

    rig.step("point", APART)

    assert rig.be.pos == target
    assert rig.be.events == [("move", *target)]


def test_button_comes_up_on_the_frame_the_fingers_part():
    rig = Rig()
    rig.step("point", APART)
    rig.step("point", TOUCHING)
    r = rig.step("pinch_index", with_gap(PINCH_LEFT, "left", rig.cfg.pinch_off + 0.01))
    assert r.events.count("up:left") == 1
    assert r.button is None
    assert rig.be.down == set()


def test_holding_a_pinch_sends_no_more_button_events():
    rig = Rig()
    rig.step("point", TOUCHING)
    for r in rig.hold("pinch_index", TOUCHING, 30):
        assert r.button == "left"
        assert r.events == []
    assert rig.be.count("down") == 1
    assert rig.be.count("up") == 0


def test_a_click_is_one_down_and_one_up():
    rig = Rig()
    rig.hold("point", APART, 3)
    rig.step("point", TOUCHING)
    rig.step("pinch_index", TOUCHING)
    rig.hold("point", APART, 3)
    assert rig.be.count("down", "left") == 1
    assert rig.be.count("up", "left") == 1
    assert rig.be.count("down", "right") == 0


@pytest.mark.parametrize("pinch_on", [0.10, 0.20, 0.30])
def test_press_threshold_comes_from_the_config(pinch_on):
    below, above = with_gap(PINCH_LEFT, "left", pinch_on - 0.01), with_gap(PINCH_LEFT, "left", pinch_on + 0.01)
    rig = Rig(pinch_on=pinch_on, pinch_off=pinch_on + 0.15)
    rig.step("point", above)
    assert rig.c.button is None
    rig.step("point", below)
    assert rig.c.button == "left"


def test_the_thumb_moving_does_not_move_the_cursor():
    # The cursor hangs on the knuckles, so a click lands where the user aimed.
    rig = Rig()
    rig.hold("point", APART, 30)
    aimed = rig.c.cursor
    r = rig.step("point", TOUCHING)
    assert r.cursor == aimed


# ------------------------------------------------------------------------- drag
def test_drag_is_down_then_moves_then_up():
    rig = Rig()
    rig.hold("point", APART, 5)
    rig.be.events.clear()
    rig.step("point", TOUCHING)
    start = rig.c.cursor
    for i in range(1, 16):
        r = rig.step("pinch_index", moved(TOUCHING, dx=0.01 * i))
        assert r.button == "left"
    travelled = r.cursor[0] - start[0]
    r = rig.step("point", moved(APART, dx=0.15))
    assert r.button is None

    kinds = [e[0] for e in rig.be.events]
    down, up = kinds.index("down"), kinds.index("up")
    assert down < up
    assert kinds[down:up].count("move") >= 10         # the OS sees movement while the button is down
    assert rig.be.count("down") == 1 and rig.be.count("up") == 1
    assert travelled > 150                            # ~0.15 of the width (~300 px), minus filter lag


def test_drag_survives_a_slightly_loose_pinch():
    # Wobble: the gap keeps drifting into the in-between zone but keeps coming back.
    rig = Rig()
    rig.step("point", TOUCHING)
    loose = with_gap(PINCH_LEFT, "left", (rig.cfg.pinch_on + rig.cfg.pinch_off) / 2)
    for _ in range(10):
        rig.hold("pinch_index", loose, rig.cfg.debounce_frames - 1)
        rig.step("pinch_index", TOUCHING)
    assert rig.c.button == "left"
    assert rig.be.count("up") == 0


@pytest.mark.parametrize("label", ["none", "fist", "scroll", "open_palm", "point"])
def test_a_wobbling_label_never_drops_a_held_button(label):
    # A held button is governed only by the live finger distance.
    rig = Rig()
    rig.step("point", TOUCHING)
    x_before = rig.c.cursor[0]
    for i in range(1, 11):
        r = rig.step(label, moved(TOUCHING, dx=0.02 * i))
        assert r.button == "left"
        assert r.events == []
    assert r.cursor[0] > x_before                    # and the drag keeps following the hand


# ------------------------------------------------------------------ right button
def test_right_click():
    rig = Rig()
    rig.hold("point", APART_RIGHT, 3)
    r = rig.step("point", TOUCHING_RIGHT)
    assert "down:right" in r.events
    assert r.button == "right"
    assert rig.be.down == {"right"}
    r = rig.step("pinch_middle", APART_RIGHT)
    assert "up:right" in r.events
    assert r.button is None
    assert rig.be.down == set()
    assert rig.be.count("down", "left") == 0 and rig.be.count("up", "left") == 0


def test_only_one_button_at_a_time():
    rig = Rig()
    rig.step("point", TOUCHING)
    assert rig.c.button == "left"
    # the same thumb can't be on the other finger without first leaving this one
    rig.step("pinch_middle", TOUCHING_RIGHT)
    assert rig.c.button == "right"
    assert rig.be.down == {"right"}
    assert rig.be.count("up", "left") == 1


def test_right_drag_follows_the_hand_too():
    rig = Rig()
    rig.step("point", TOUCHING_RIGHT)
    first = rig.c.cursor
    for i in range(1, 11):
        r = rig.step("pinch_middle", moved(TOUCHING_RIGHT, dy=0.01 * i))
    assert r.button == "right"
    assert r.cursor[1] > first[1]


# ---------------------------------------------------------- blocked and tucked
@pytest.mark.parametrize("label", ["fist", "scroll", "none"])
def test_these_labels_can_never_start_a_click(label):
    rig = Rig()
    for r in rig.hold(label, TOUCHING, 10):
        assert r.button is None
    assert rig.be.count("down") == 0


def test_open_palm_label_can_still_start_a_click():
    rig = Rig()
    r = rig.step("open_palm", TOUCHING)
    assert "down:left" in r.events


def test_a_finger_tucked_into_the_palm_cannot_press():
    tucked = with_gap(shape("fist", 1.0, 1.0, 1.0, 1.0), "left", 0.05)
    assert finger_reach(to_metric(tucked, ASPECT), "index") < PRESS_REACH["left"]
    rig = Rig()
    for r in rig.hold("point", tucked, 10):
        assert r.button is None
    assert rig.be.count("down") == 0


@pytest.mark.parametrize("mode", ["absolute", "relative"])
def test_a_finger_tucked_below_the_hold_limit_lets_go_at_once(mode):
    tucked = with_gap(shape("fist", 1.0, 1.0, 1.0, 1.0, pose=HandPose(pitch=20.0)), "left", 0.05)
    assert finger_reach(to_metric(tucked, ASPECT), "index") < HOLD_REACH["left"]
    rig = Rig(cursor_mode=mode)
    rig.step("point", TOUCHING)
    if mode == "relative":
        rig.hold("point", TOUCHING, 9)     # first establish a real held drag
    r = rig.step("pinch_index", tucked)
    assert "up:left" in r.events
    assert rig.be.down == set()


def test_fist_releases_a_held_button():
    rig = Rig()
    rig.step("point", TOUCHING)
    rig.hold("fist", FIST, rig.cfg.debounce_frames)
    assert rig.c.button is None
    assert rig.be.count("up", "left") == 1
    rig.hold("fist", FIST, 10)                        # and it stays up: no new clicks
    assert rig.be.count("down") == 1


def test_random_fists_release_a_held_button():
    # A small share (~1.5%) keep the thumb on a not-quite-tucked index tip: to the
    # controller that is still a pinch, and it lets go as soon as the thumb moves away.
    rng = np.random.default_rng(0)
    released = 0
    for _ in range(300):
        rig = Rig()
        rig.step("point", TOUCHING)
        rig.hold("fist", make_hand("fist", rng), rig.cfg.debounce_frames)
        released += rig.c.button is None
    assert released / 300 >= 0.97


def test_a_thumb_still_on_a_curled_finger_keeps_the_drag_even_if_labelled_fist():
    # A deep pinch can look like a fist to the labeller; the drag must not drop.
    curled = with_gap(shape("pinch_index", 0.7, 0.8, 0.8, 0.8, thumb_gap=0.0), "left", 0.05)
    reach = finger_reach(to_metric(curled, ASPECT), "index")
    assert HOLD_REACH["left"] < reach < PRESS_REACH["left"]
    rig = Rig()
    rig.step("point", TOUCHING)
    for r in rig.hold("fist", curled, 10):
        assert r.button == "left"


# -------------------------------------------------------------------- hand lost
def test_losing_the_hand_releases_the_button():
    rig = Rig()
    rig.step("point", TOUCHING)
    r = rig.step("none", None)
    assert "up:left" in r.events
    assert r.button is None
    assert rig.be.down == set()


def test_losing_the_hand_keeps_the_cursor_where_it_is():
    rig = Rig()
    rig.hold("point", POINT, 10)
    before = rig.c.cursor
    moves = rig.moves()
    r = rig.step("none", None)
    assert r.cursor == before
    assert r.events == []
    assert rig.moves() == moves


def test_no_hand_from_the_start_is_harmless():
    rig = Rig()
    r = rig.step("none", None)
    assert r.cursor is None and r.button is None and r.events == []
    assert rig.be.events == []


def test_release_all_and_halt_only_report_a_real_release():
    rig = Rig()
    events: list[str] = []
    rig.c.release_all(events)
    rig.c.halt(events)
    assert events == []
    assert rig.be.count("up") == 0

    rig.step("point", TOUCHING)
    rig.c.halt(events)
    assert events == ["up:left"]
    rig.c.halt(events)                                # idempotent
    assert events == ["up:left"]
    assert rig.be.count("up", "left") == 1


def test_release_all_without_an_event_list_still_releases():
    rig = Rig()
    rig.step("point", TOUCHING)
    rig.c.release_all()
    assert rig.c.button is None
    assert rig.be.down == set()


# ---------------------------------------------------------------- cursor movement
def test_point_moves_the_cursor_to_the_mapped_position():
    rig = Rig()
    r = rig.step("point", POINT)
    assert r.cursor == rig.c.to_screen(*anchor_point(POINT))
    assert rig.be.pos == r.cursor
    assert rig.be.events == [("move", *r.cursor)]


def test_cursor_follows_the_hand_in_both_axes():
    rig = Rig()
    rig.hold("point", POINT, 30)
    x0, y0 = rig.c.cursor
    r = rig.hold("point", moved(POINT, dx=0.1, dy=0.1), 60)[-1]
    assert r.cursor[0] > x0            # hand right in the image -> cursor right
    assert r.cursor[1] > y0            # hand down in the image -> cursor down


def test_no_move_event_when_the_pixel_does_not_change():
    rig = Rig()
    rig.hold("point", POINT, 40)
    moves = rig.moves()
    rig.hold("point", POINT, 40)
    assert rig.moves() == moves


def test_a_still_hand_with_tremor_gives_a_steady_cursor_and_no_clicks():
    rng = np.random.default_rng(0)
    rig = Rig()
    cursors = [rig.step("point", POINT + rng.normal(0, TREMOR, POINT.shape)).cursor for _ in range(300)]
    xs, ys = np.array(cursors[30:]).T
    assert xs.max() - xs.min() <= 12
    assert ys.max() - ys.min() <= 12
    assert rig.be.count("down") == 0


@pytest.mark.parametrize(
    "label, hand",
    [("open_palm", OPEN), ("fist", FIST), ("none", POINT)],
    ids=["open_palm", "fist", "unknown-pose"],
)
def test_these_labels_freeze_the_cursor(label, hand):
    rig = Rig()
    rig.hold("point", POINT, 10)
    before, moves = rig.c.cursor, rig.moves()
    for i in range(1, 21):
        r = rig.step(label, moved(hand, dx=0.01 * i, dy=-0.005 * i))
        assert r.cursor == before
        assert r.events == []
    assert rig.moves() == moves


# ----------------------------------------------------------------------- no fly-in
PAUSES = {
    "fist": ("fist", FIST),
    "open_palm": ("open_palm", OPEN),
    "unknown-pose": ("none", POINT),
    "scroll": ("scroll", V_SIGN),
    "hand-lost": ("none", None),
}


@pytest.mark.parametrize("pause", list(PAUSES))
def test_after_a_pause_the_cursor_jumps_straight_to_the_hand(pause):
    rig = Rig()
    rig.hold("point", POINT, 30)                     # the filter has a history at the old spot
    far = moved(POINT, dx=0.25, dy=-0.15)
    label, hand = PAUSES[pause]
    rig.step(label, None if hand is None else moved(hand, dx=0.25, dy=-0.15))
    r = rig.step("point", far)
    assert r.cursor == rig.c.to_screen(*anchor_point(far))


def test_without_a_pause_the_filter_still_glides():
    # The control for the test above: a jump during movement is smoothed, not teleported.
    rig = Rig()
    rig.hold("point", POINT, 30)
    start = rig.c.cursor
    far = moved(POINT, dx=0.25, dy=-0.15)
    r = rig.step("point", far)
    target = rig.c.to_screen(*anchor_point(far))
    assert r.cursor != target
    assert min(start[0], target[0]) < r.cursor[0] < max(start[0], target[0])


# -------------------------------------------------------------- relative movement
def relative_rig(gain=1.5, **settings) -> Rig:
    settings.setdefault("frame_margin", 0.14)  # active box is 0.96 wide x 0.72 high
    return Rig(cursor_mode="relative", cursor_gain=gain, **settings)


class VirtualDesktopBackend(FakeBackend):
    def screen_bounds(self):
        return -1280, -720, 4480, 2160


def relative_virtual_rig(start) -> Rig:
    rig = relative_rig(gain=1.0)
    rig.be = VirtualDesktopBackend()
    rig.be.pos = start
    rig.c = MouseController(rig.cfg, rig.be)
    return rig


@pytest.mark.parametrize("axis, displacement, expected", [
    (0, 100 * 0.96 / 1919, (1060, 540)),
    (1, 50 * 0.72 / 1079, (960, 590)),
])
def test_relative_point_follows_only_the_index_tip_when_the_palm_is_still(axis, displacement, expected):
    rig = relative_rig(gain=1.0)
    rig.step("point", POINT)
    fingertip = POINT.copy()
    fingertip[INDEX_TIP, axis] += displacement

    rig.hold("point", fingertip, 180)

    assert rig.c.cursor == expected
    assert rig.be.count("down") == 0


def test_relative_point_does_not_follow_palm_motion_when_the_index_tip_stays_fixed():
    rig = relative_rig(gain=1.0)
    rig.step("point", POINT)
    palm = moved(POINT, dx=-0.03)
    palm[INDEX_TIP] = POINT[INDEX_TIP]

    rig.hold("point", palm, 180)

    assert rig.c.cursor == (960, 540)
    assert rig.be.events == []


@pytest.mark.parametrize("label, touching, button", [
    ("point", TOUCHING, "left"),
    ("pinch_index", TOUCHING, "left"),
    ("pinch_middle", TOUCHING_RIGHT, "right"),
])
def test_relative_tip_to_pinch_switch_reanchors_at_the_actual_pointer_without_a_jump(label, touching, button):
    rig = relative_rig(gain=1.0)
    rig.hold("point", POINT, 10)
    rig.be.pos = (430, 440)    # the physical pointer may have changed meanwhile
    rig.be.events.clear()

    result = rig.step(label, moved(touching, dx=0.2, dy=-0.1))

    assert result.cursor == (430, 440)
    assert result.button is None                 # a short pinch is pending, not held
    assert rig.be.events == []
    # A stationary palm cannot produce a delayed jump from the previous tip filter.
    rig.hold(label, moved(touching, dx=0.2, dy=-0.1), 30)
    assert rig.be.events == [("down", button)]


@pytest.mark.parametrize("touching, apart, button", [
    (TOUCHING, APART, "left"),
    (TOUCHING_RIGHT, APART_RIGHT, "right"),
])
def test_relative_pinch_release_switches_back_to_tip_without_a_jump(touching, apart, button):
    rig = relative_rig(gain=1.0)
    rig.step("point", touching)
    rig.hold("point", moved(touching, dx=0.05), 30)
    rig.be.pos = (720, 500)
    rig.be.events.clear()

    result = rig.step("point", moved(apart, dx=-0.2, dy=0.1))

    assert result.cursor == (720, 500)
    assert result.button is None
    assert rig.be.events == [("up", button)]
    rig.hold("point", moved(apart, dx=-0.2, dy=0.1), 30)
    assert rig.be.events == [("up", button)]


def test_relative_held_drag_uses_the_palm_even_when_the_label_is_point():
    rig = relative_rig(gain=1.0)
    rig.step("pinch_index", TOUCHING)
    tip_and_thumb = TOUCHING.copy()
    tip_and_thumb[[INDEX_TIP, THUMB_TIP], 0] += 0.03

    rig.hold("point", tip_and_thumb, 30)

    assert rig.be.pos == (960, 540)
    assert rig.be.events == [("down", "left")]
    rig.hold("point", moved(tip_and_thumb, dx=100 * 0.96 / 1919), 180)
    assert rig.be.pos == (1060, 540)
    assert rig.be.down == {"left"}


@pytest.mark.parametrize("touching, pinch, button", [
    (TOUCHING, PINCH_LEFT, "left"),
    (TOUCHING_RIGHT, PINCH_RIGHT, "right"),
])
def test_relative_held_pinch_keeps_the_hysteresis_gap_for_more_than_thirty_frames(touching, pinch, button):
    rig = relative_rig(gain=1.0)
    rig.step("point", touching)
    rig.hold("point", touching, 9)
    loose = with_gap(pinch, button, 0.26)

    for i in range(1, 46):
        result = rig.step("point", moved(loose, dx=i * 0.001))
        assert result.button == button
        assert result.events == []

    assert rig.be.count("down", button) == 1
    assert rig.be.count("up", button) == 0
    assert rig.be.pos[0] > 960
    released = rig.step("point", with_gap(pinch, button, 0.34))
    assert released.button is None
    assert released.events == [f"up:{button}"]


@pytest.mark.parametrize("mode", ["relative", "absolute"])
@pytest.mark.parametrize("label, touching", [("point", TOUCHING), ("pinch_middle", TOUCHING_RIGHT)])
def test_a_disallowed_new_press_never_clicks_even_when_the_fingers_touch(mode, label, touching):
    rig = Rig(cursor_mode=mode)

    for _ in range(5):
        result = rig.step(label, touching, allow_press=False)
        assert result.button is None
        assert result.events == []

    assert rig.be.down == set()
    assert rig.be.count("down") == 0


@pytest.mark.parametrize("mode", ["relative", "absolute"])
@pytest.mark.parametrize("touching, apart, button", [
    (TOUCHING, APART, "left"),
    (TOUCHING_RIGHT, APART_RIGHT, "right"),
])
def test_disallowing_new_presses_keeps_existing_drag_and_live_release_working(mode, touching, apart, button):
    rig = Rig(cursor_mode=mode)
    rig.step("point", touching)
    if mode == "relative":
        rig.hold("point", touching, 9)
    start = rig.be.pos

    for i in range(1, 11):
        result = rig.step("point", moved(touching, dx=i * 0.01), allow_press=False)
        assert result.button == button
        assert result.events == []

    assert rig.be.pos[0] > start[0]
    released = rig.step("point", moved(apart, dx=0.1), allow_press=False)
    assert released.button is None
    assert released.events == [f"up:{button}"]
    assert rig.be.count("down", button) == 1
    assert rig.be.count("up", button) == 1


@pytest.mark.parametrize("start", [(-120, 300), (2200, 300), (100, -120), (100, 1200)])
def test_relative_stationary_hand_keeps_the_actual_cursor_on_a_secondary_screen(start):
    rig = relative_virtual_rig(start)

    results = rig.hold("point", POINT, 30)

    assert all(result.cursor == start for result in results)
    assert rig.be.pos == start
    assert rig.be.events == []


@pytest.mark.parametrize("start, expected", [((-120, 300), (-20, 350)),
                                            ((2200, 300), (2300, 350))])
def test_relative_secondary_screen_uses_the_existing_primary_screen_speed(start, expected):
    rig = relative_virtual_rig(start)
    rig.step("point", POINT)

    rig.hold("point", moved(POINT, dx=100 * 0.96 / 1919,
                            dy=50 * 0.72 / 1079), 180)

    assert rig.c.cursor == expected
    assert rig.be.pos == expected


@pytest.mark.parametrize("direction, start, edge", [
    (-1, (-1270, -710), (-1280, -720)),
    (1, (3190, 1430), (3199, 1439)),
])
def test_relative_virtual_screen_edges_discard_overshoot_and_allow_reverse_motion(direction, start, edge):
    rig = relative_virtual_rig(start)
    rig.step("point", POINT)
    rig.hold("point", moved(POINT, dx=0.3 * direction, dy=0.3 * direction), 180)
    assert rig.c.cursor == edge
    rig.hold("point", moved(POINT, dx=0.6 * direction, dy=0.6 * direction), 180)
    assert rig.c.cursor == edge

    result = rig.step("point", moved(POINT, dx=0.59 * direction, dy=0.59 * direction))

    assert direction * (result.cursor[0] - edge[0]) < 0
    assert direction * (result.cursor[1] - edge[1]) < 0


def test_relative_negative_screen_coordinates_keep_subpixel_motion():
    rig = relative_virtual_rig((-120, 300))
    rig.step("point", POINT)
    for i in range(1, 31):
        rig.step("point", moved(POINT, dx=-i * 0.1 * 0.96 / 1919))
    rig.hold("point", moved(POINT, dx=-3 * 0.96 / 1919), 180)

    assert rig.c.cursor == (-123, 300)


def test_relative_activation_anchors_at_the_actual_cursor_without_a_move():
    rig = relative_rig()
    rig.be.move(230, 340)
    rig.be.events.clear()

    r = rig.step("point", moved(POINT, dx=0.25, dy=-0.2))

    assert r.cursor == (230, 340)
    assert rig.be.pos == (230, 340)
    assert rig.be.events == []


@pytest.mark.parametrize("gain, expected", [(1.0, (1060, 590)), (2.0, (1160, 640))])
def test_relative_travel_uses_the_active_box_scale_and_gain(gain, expected):
    # A 100x50-pixel displacement at gain 1 in this camera active box.
    rig = relative_rig(gain=gain)
    rig.step("point", POINT)
    rig.hold("point", moved(POINT, dx=100 * 0.96 / 1919,
                            dy=50 * 0.72 / 1079), 180)

    assert rig.c.cursor == expected
    assert rig.be.pos == expected


@pytest.mark.parametrize("pause", list(PAUSES))
def test_relative_a_to_b_reposition_then_b_to_c_continues_without_a_jump(pause):
    rig = relative_rig(gain=1.0)
    rig.step("point", POINT)
    rig.hold("point", moved(POINT, dx=100 * 0.96 / 1919), 180)
    assert rig.c.cursor == (1060, 540)                    # reached B
    label, hand = PAUSES[pause]
    rig.step(label, hand)
    rig.be.events.clear()

    resumed = rig.step("point", moved(POINT, dx=-0.25))  # reposition the hand

    assert resumed.cursor == (1060, 540)
    assert rig.be.events == []
    rig.hold("point", moved(POINT, dx=-0.25 + 100 * 0.96 / 1919), 180)
    assert rig.c.cursor == (1160, 540)                   # reached C


def test_relative_halt_resumes_from_a_physical_mouse_move():
    rig = relative_rig()
    rig.hold("point", POINT, 30)
    rig.c.halt()
    rig.be.move(20, 30)
    rig.be.events.clear()

    r = rig.step("point", moved(POINT, dx=0.2))

    assert r.cursor == (20, 30)
    assert rig.be.events == []
    rig.hold("point", moved(POINT, dx=0.25), 180)
    assert rig.c.cursor[0] > 20


@pytest.mark.parametrize("direction, start, edge", [(1, (1900, 1000), (1919, 1079)),
                                                  (-1, (20, 20), (0, 0))])
def test_relative_screen_edges_discard_overshoot_so_returning_moves_at_once(direction, start, edge):
    rig = relative_rig(gain=1.0)
    rig.be.pos = start
    rig.step("point", POINT)
    rig.hold("point", moved(POINT, dx=0.3 * direction, dy=0.3 * direction), 180)
    assert rig.c.cursor == edge
    rig.hold("point", moved(POINT, dx=0.6 * direction, dy=0.6 * direction), 180)
    assert rig.c.cursor == edge

    r = rig.step("point", moved(POINT, dx=0.59 * direction, dy=0.59 * direction))

    assert direction * (r.cursor[0] - edge[0]) < 0
    assert direction * (r.cursor[1] - edge[1]) < 0


def test_relative_subpixel_motion_accumulates_instead_of_being_rounded_away():
    rig = relative_rig(gain=1.0)
    rig.step("point", POINT)
    # Each frame advances by 0.1 pixel; after settling, total travel is 3 pixels.
    for i in range(1, 31):
        rig.step("point", moved(POINT, dx=i * 0.1 * 0.96 / 1919))
    rig.hold("point", moved(POINT, dx=3 * 0.96 / 1919), 180)

    assert rig.c.cursor == (963, 540)


def test_relative_movement_retains_one_euro_smoothing():
    rig = relative_rig(gain=1.0)
    rig.step("point", POINT)

    r = rig.step("point", moved(POINT, dx=100 * 0.96 / 1919))

    assert 960 < r.cursor[0] < 1060
    rig.hold("point", moved(POINT, dx=100 * 0.96 / 1919), 180)
    assert rig.c.cursor == (1060, 540)


@pytest.mark.parametrize("label, touching, apart, button", [
    ("pinch_index", TOUCHING, APART, "left"),
    ("pinch_middle", TOUCHING_RIGHT, APART_RIGHT, "right"),
])
def test_relative_pinch_starts_at_current_pointer_and_can_drag(label, touching, apart, button):
    rig = relative_rig(gain=1.0)
    rig.be.pos = (230, 340)

    pressed = rig.step(label, touching)

    assert pressed.cursor == (230, 340)
    assert pressed.button is None
    assert rig.be.events == []
    rig.hold(label, touching, 9)
    assert rig.be.events == [("down", button)]
    rig.hold(label, moved(touching, dx=100 * 0.96 / 1919), 180)
    assert rig.c.cursor == (330, 340)
    assert rig.be.down == {button}
    released = rig.step("point", moved(apart, dx=100 * 0.96 / 1919))
    assert released.button is None
    assert released.events == [f"up:{button}"]
    assert rig.be.down == set()


# -------------------------------------------------------------------------- scroll
@pytest.mark.parametrize("direction, sign", [(-1, 1), (1, -1)], ids=["hand-up", "hand-down"])
def test_scroll_direction(direction, sign):
    rig = Rig()
    rig.step("scroll", V_SIGN)
    events = []
    for i in range(1, 21):
        events += rig.step("scroll", moved(V_SIGN, dy=direction * 0.01 * i)).events
    notches = rig.scrolls()
    assert notches and all(np.sign(n) == sign for n in notches)
    assert len(events) == len(notches)
    assert all(re.fullmatch(r"scroll:[+-]\d+\.\d\d", e) for e in events)
    assert all((e[7] == "+") == (sign > 0) for e in events)


def test_scroll_amount_follows_travel_times_gain():
    rig = Rig()                                       # gain 40 notches per frame height
    rig.step("scroll", V_SIGN)
    for i in range(1, 21):
        rig.step("scroll", moved(V_SIGN, dy=-0.01 * i))
    rig.hold("scroll", moved(V_SIGN, dy=-0.2), 30)    # let the smoothing settle
    assert sum(rig.scrolls()) == pytest.approx(0.2 * rig.cfg.scroll_gain, abs=0.45)


def test_scroll_gain_comes_from_the_config():
    slow, fast = Rig(scroll_gain=20.0), Rig(scroll_gain=80.0)
    for rig in (slow, fast):
        rig.step("scroll", V_SIGN)
        for i in range(1, 21):
            rig.step("scroll", moved(V_SIGN, dy=-0.01 * i))
        rig.hold("scroll", moved(V_SIGN, dy=-0.2), 30)
    assert sum(fast.scrolls()) == pytest.approx(4 * sum(slow.scrolls()), rel=0.1)


def test_movement_inside_the_deadzone_does_not_scroll():
    rig = Rig()
    rig.step("scroll", V_SIGN)
    rig.hold("scroll", moved(V_SIGN, dy=-0.5 * rig.cfg.scroll_deadzone), 60)
    assert rig.scrolls() == []


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_a_still_v_hand_with_realistic_tremor_does_not_scroll(seed):
    rng = np.random.default_rng(seed)
    rig = Rig()
    for _ in range(300):
        rig.step("scroll", V_SIGN + rng.normal(0, TREMOR, V_SIGN.shape))
    assert rig.scrolls() == []


def test_scrolling_does_not_move_the_cursor_or_click():
    rig = Rig()
    rig.hold("point", POINT, 10)
    before, moves = rig.c.cursor, rig.moves()
    rig.step("scroll", V_SIGN)
    for i in range(1, 21):
        r = rig.step("scroll", moved(V_SIGN, dy=-0.01 * i))
        assert r.cursor == before
    assert rig.moves() == moves
    assert rig.be.count("down") == 0


def test_re_entering_scroll_at_a_new_height_does_not_jump():
    rig = Rig()
    rig.step("scroll", V_SIGN)
    for i in range(1, 11):
        rig.step("scroll", moved(V_SIGN, dy=-0.01 * i))
    before = len(rig.scrolls())
    rig.hold("point", POINT, 3)                       # leave scroll mode
    rig.hold("scroll", moved(V_SIGN, dy=0.2), 5)      # back, somewhere else entirely
    assert len(rig.scrolls()) == before


def test_a_held_button_takes_priority_over_scroll():
    rig = Rig()
    rig.step("point", TOUCHING)
    rig.step("scroll", moved(TOUCHING, dy=-0.1))
    rig.step("scroll", moved(TOUCHING, dy=-0.2))
    assert rig.scrolls() == []
    assert rig.c.button == "left"


# --------------------------------------------------------------------------- dwell
def dwell_rig(**settings) -> Rig:
    return Rig(dwell_enabled=True, dwell_ms=300, dwell_radius_px=18, **settings)


def px_per_unit(rig: Rig) -> float:
    """Screen pixels per unit of normalised image x."""
    return (rig.c.to_screen(0.6, 0.5)[0] - rig.c.to_screen(0.5, 0.5)[0]) / 0.1


def test_dwell_clicks_after_the_hold_time():
    rig = dwell_rig()
    results = rig.hold("point", POINT, 30)
    frames = [i for i, r in enumerate(results) if "click:dwell" in r.events]
    assert len(frames) == 1
    assert 8 <= frames[0] <= 10        # 300 ms is 9 frames at 30 fps, counted from the first frame
    assert rig.be.count("down", "left") == 1
    assert rig.be.count("up", "left") == 1
    assert rig.c.button is None
    assert rig.be.down == set()


def test_dwell_clicks_once_then_rearms_only_after_leaving_the_radius():
    rig = dwell_rig()
    rig.hold("point", POINT, 60)                      # 2 s: far longer than dwell_ms
    assert rig.be.count("down", "left") == 1

    # wandering around inside the radius, for a long time: still only that one click
    near = moved(POINT, dx=8 / px_per_unit(rig))
    for _ in range(30):
        rig.step("point", near)
        rig.step("point", POINT)
    assert rig.be.count("down", "left") == 1

    # leaving the radius re-arms it, and holding still at the new spot clicks again
    away = moved(POINT, dx=0.1)
    rig.hold("point", away, 60)
    assert rig.be.count("down", "left") == 2
    rig.hold("point", away, 60)
    assert rig.be.count("down", "left") == 2


def test_dwell_timer_restarts_if_the_cursor_leaves_before_it_fires():
    rig = dwell_rig()
    rig.hold("point", POINT, 5)                       # 167 ms: not long enough
    rig.hold("point", moved(POINT, dx=0.1), 5)        # moves away, still gliding
    assert rig.be.count("down") == 0


def test_dwell_off_means_no_clicks():
    rig = Rig(dwell_enabled=False, dwell_ms=300)
    rig.hold("point", POINT, 90)
    assert rig.be.count("down") == 0


def test_dwell_does_not_fire_while_a_button_is_held():
    rig = dwell_rig()
    rig.step("point", TOUCHING)
    results = rig.hold("pinch_index", TOUCHING, 60)
    assert not any("click:dwell" in r.events for r in results)
    assert rig.be.count("down") == 1


@pytest.mark.parametrize("touching, apart", [(TOUCHING, APART), (TOUCHING_RIGHT, APART_RIGHT)])
def test_a_manual_pinch_restarts_the_dwell_timer_on_release(touching, apart):
    rig = dwell_rig()
    rig.c.update("point", apart, 0.0)
    rig.c.update("point", apart, 0.2)
    rig.c.update("point", touching, 0.21)
    rig.c.update("point", touching, 1.0)

    released = rig.c.update("point", apart, 1.01)
    assert "click:dwell" not in released.events
    assert rig.be.count("down") == 1

    after_wait = rig.c.update("point", apart, 1.32)
    assert after_wait.events == ["click:dwell"]
    assert rig.be.count("down") == 2


def test_a_failed_dwell_release_can_be_retried_by_halt():
    class FailsFirstRelease(FakeBackend):
        def __init__(self):
            super().__init__()
            self.failed = False

        def release(self, button):
            if not self.failed:
                self.failed = True
                raise OSError("temporary input failure")
            super().release(button)

    backend = FailsFirstRelease()
    controller = MouseController(Config(dwell_enabled=True, dwell_ms=300), backend)
    controller.update("point", POINT, 0.0)
    with pytest.raises(OSError, match="temporary input failure"):
        controller.update("point", POINT, 0.4)

    controller.halt()

    assert backend.down == set()
    assert backend.count("up", "left") == 1


def test_dwell_does_not_fire_while_paused():
    rig = dwell_rig()
    rig.hold("fist", FIST, 60)
    rig.hold("open_palm", OPEN, 60)
    assert rig.be.count("down") == 0


def test_dwell_rearms_after_a_pause():
    rig = dwell_rig()
    rig.hold("point", POINT, 30)
    assert rig.be.count("down", "left") == 1
    rig.step("fist", FIST)
    rig.hold("point", POINT, 30)
    assert rig.be.count("down", "left") == 2


# ------------------------------------------------------------ equal sensitivity
def test_equal_sensitivity_4_3_camera_on_a_16_9_screen():
    rig = Rig()
    d = 0.1                                           # physical distance, in image heights
    cx, cy = 0.5, 0.5
    x0, y0 = rig.c.to_screen(cx, cy)
    # the same physical distance is d/aspect of the width but d of the height
    x1, _ = rig.c.to_screen(cx + d / ASPECT, cy)
    _, y1 = rig.c.to_screen(cx, cy + d)
    assert abs((x1 - x0) - (y1 - y0)) <= 2
    assert x1 - x0 > 100                              # a real movement, not rounding noise


def test_equal_sensitivity_through_the_full_update_path():
    d = 0.1

    def settled_travel(dx: float, dy: float) -> tuple[int, int]:
        rig = Rig()
        rig.hold("point", POINT, 60)
        start = rig.c.cursor
        rig.hold("point", moved(POINT, dx=dx, dy=dy), 180)
        return rig.c.cursor[0] - start[0], rig.c.cursor[1] - start[1]

    sideways, _ = settled_travel(d / ASPECT, 0.0)
    _, downwards = settled_travel(0.0, d)
    assert sideways > 100
    assert abs(sideways - downwards) <= 2


@pytest.mark.parametrize(
    "camera, screen",
    [
        ((640, 480), (1920, 1080)),
        ((640, 480), (1366, 768)),
        ((640, 480), (2560, 1440)),
        ((640, 480), (3440, 1440)),   # ultrawide: the box hits its width limit
        ((640, 480), (1280, 1024)),
        ((640, 480), (1080, 1920)),   # portrait screen
        ((1280, 720), (1920, 1080)),
        ((1280, 720), (3440, 1440)),
        ((640, 640), (1920, 1080)),
    ],
    ids=str,
)
def test_equal_sensitivity_for_other_cameras_and_screens(camera, screen):
    rig = Rig(screen=screen, frame_width=camera[0], frame_height=camera[1])
    _, _, w, h = rig.c.active_box
    d = 0.2 * min(w * rig.cfg.aspect, h)              # stays inside the box on both axes
    x0, y0 = rig.c.to_screen(0.5, 0.5)
    x1, _ = rig.c.to_screen(0.5 + d / rig.cfg.aspect, 0.5)
    _, y1 = rig.c.to_screen(0.5, 0.5 + d)
    assert abs((x1 - x0) - (y1 - y0)) <= 2


# ------------------------------------------------------------------------ mapping
def test_to_screen_clamps_to_the_screen():
    rig = Rig()
    assert rig.c.to_screen(-1.0, -1.0) == (0, 0)
    assert rig.c.to_screen(2.0, 2.0) == (1919, 1079)
    assert rig.c.to_screen(-1.0, 2.0) == (0, 1079)
    assert rig.c.to_screen(2.0, -1.0) == (1919, 0)
    assert rig.c.to_screen(1e12, -1e12) == (1919, 0)


def test_to_screen_maps_the_active_box_onto_the_whole_screen():
    rig = Rig()
    x0, y0, w, h = rig.c.active_box
    assert rig.c.to_screen(x0, y0) == (0, 0)
    assert rig.c.to_screen(x0 + w, y0 + h) == (1919, 1079)
    assert rig.c.to_screen(x0 + w / 2, y0 + h / 2) == (960, 540)


def test_to_screen_returns_plain_ints():
    x, y = Rig().c.to_screen(0.4, 0.6)
    assert type(x) is int and type(y) is int


def test_a_hand_leaving_the_box_pins_the_cursor_to_the_edge():
    rig = Rig()
    far_left = moved(POINT, dx=-0.9, dy=-0.9)
    r = rig.hold("point", far_left, 30)[-1]
    assert r.cursor == (0, 0)
    assert rig.be.pos == (0, 0)


CAMERAS_AND_SCREENS = [
    ((640, 480), (1920, 1080), 0.15),
    ((640, 480), (1920, 1080), 0.0),
    ((640, 480), (1920, 1080), 0.4),
    ((1280, 720), (1920, 1080), 0.15),
    ((640, 480), (3440, 1440), 0.15),
    ((640, 640), (1920, 1080), 0.0),
    ((480, 640), (1280, 1024), 0.1),
    ((640, 480), (1080, 1920), 0.15),
]


@pytest.mark.parametrize("camera, screen, margin", CAMERAS_AND_SCREENS, ids=str)
def test_active_box_stays_inside_the_image_and_matches_the_screen_shape(camera, screen, margin):
    rig = Rig(screen=screen, frame_width=camera[0], frame_height=camera[1], frame_margin=margin)
    x0, y0, w, h = rig.c.active_box
    assert w > 0 and h > 0
    assert 0.0 <= x0 and x0 + w <= 1.0
    assert 0.0 <= y0 and y0 + h <= 1.0
    assert x0 + w / 2 == pytest.approx(0.5)
    assert y0 + h / 2 == pytest.approx(0.5)
    assert w * rig.cfg.aspect / h == pytest.approx(screen[0] / screen[1])


def test_frame_margin_sets_the_box_height_when_it_fits():
    assert Rig(frame_margin=0.15).c.active_box[3] == pytest.approx(0.70)
    assert Rig(frame_margin=0.25).c.active_box[3] == pytest.approx(0.50)


# ----------------------------------------------------------- in-between zone
def test_a_thumb_lingering_in_the_in_between_zone_releases_after_debounce_frames():
    rig = Rig()
    assert rig.cfg.pinch_on < 0.26 < rig.cfg.pinch_off
    lingering = with_gap(PINCH_LEFT, "left", 0.26)
    rig.step("point", TOUCHING)
    for i in range(rig.cfg.debounce_frames - 1):
        r = rig.step("point", lingering)
        assert r.button == "left", f"released too early, on lingering frame {i + 1}"
    r = rig.step("point", lingering)
    assert "up:left" in r.events
    assert r.button is None
    assert rig.be.down == set()


@pytest.mark.parametrize("frames", [1, 2, 5, 10])
def test_the_in_between_grace_comes_from_the_config(frames):
    rig = Rig(debounce_frames=frames)
    lingering = with_gap(PINCH_LEFT, "left", 0.26)
    rig.step("point", TOUCHING)
    rig.hold("point", lingering, frames - 1)
    assert rig.c.button == "left"
    rig.step("point", lingering)
    assert rig.c.button is None


def test_no_button_stays_stuck_with_a_thumb_resting_nearby():
    rig = Rig()
    lingering = with_gap(PINCH_LEFT, "left", 0.26)
    rig.step("point", TOUCHING)
    rig.hold("point", lingering, 300)
    assert rig.c.button is None
    assert rig.be.count("down") == 1                  # and it does not re-press from there either


def test_the_grace_count_restarts_on_a_new_press():
    rig = Rig()
    lingering = with_gap(PINCH_LEFT, "left", 0.26)
    rig.step("point", TOUCHING)
    rig.hold("point", lingering, rig.cfg.debounce_frames)         # released
    assert rig.c.button is None
    rig.step("point", TOUCHING)                                   # pressed again
    rig.hold("point", lingering, rig.cfg.debounce_frames - 1)
    assert rig.c.button == "left"


# --------------------------------------------------------------------------- zoom
@pytest.mark.parametrize("steps, word", [(1.5, "in"), (-1.5, "out")])
def test_zoom_goes_to_the_backend_and_is_reported(steps, word):
    rig = Rig()
    events: list[str] = []
    rig.c.zoom(steps, events)
    assert events == [f"zoom:{word}"]
    assert rig.be.events == [("zoom", steps)]


# -------------------------------------------------------------------- fake backend
def test_fake_backend_records_everything():
    be = FakeBackend(800, 600)
    assert be.screen_size() == (800, 600)
    assert be.pos == (400, 300)
    be.move(1, 2)
    be.press("left")
    be.release("left")
    be.press("right")
    be.scroll(1.5)
    be.zoom(-2.0)
    assert be.pos == (1, 2)
    assert be.down == {"right"}
    assert be.events == [("move", 1, 2), ("down", "left"), ("up", "left"), ("down", "right"),
                         ("scroll", 1.5), ("zoom", -2.0)]
    assert be.count("down") == 2
    assert be.count("down", "left") == 1
    assert be.count("scroll") == 1
