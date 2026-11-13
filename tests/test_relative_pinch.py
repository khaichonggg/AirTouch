"""Relative tap/hold semantics through the real controller, without OS input."""
import numpy as np
import pytest

from airtouch.config import Config
from airtouch.controller import FakeBackend, MouseController
from airtouch.landmarks import INDEX_TIP, MIDDLE_TIP, THUMB_TIP, hand_size, to_metric
from ml.synthetic_data import HandPose, make_hand


def pinch(button="left", gap=.05, dx=0):
    label = "pinch_index" if button == "left" else "pinch_middle"
    hand = make_hand(label, np.random.default_rng(1),
                     pose=HandPose(cx=.5, cy=.55, size=.16), noise=0,
                     curls={"index": .4 if button == "left" else .05,
                            "middle": .7 if button == "left" else .45,
                            "ring": .7, "pinky": .7}, thumb_gap=0)
    metric = to_metric(hand, 4 / 3)
    tip, other = (INDEX_TIP, MIDDLE_TIP) if button == "left" else (MIDDLE_TIP, INDEX_TIP)
    away = metric[tip, :2] - metric[other, :2]
    target = metric[tip, :2] + away / np.linalg.norm(away) * gap * hand_size(metric)
    hand[THUMB_TIP, :2] = target[0] / (4 / 3), target[1]
    return hand + (dx, 0, 0)


def rig():
    backend = FakeBackend()
    return MouseController(Config(cursor_mode="relative"), backend), backend


@pytest.mark.parametrize("button", ["left", "right"])
def test_short_pinch_never_holds_the_os_button_and_clicks_once_on_release(button):
    controller, backend = rig()
    label = "pinch_index" if button == "left" else "pinch_middle"
    controller.update(label, pinch(button), 0)
    controller.update(label, pinch(button), .1)
    assert backend.down == set()
    assert backend.count("down") == 0

    result = controller.update("point", pinch(button, gap=.28), .15, allow_press=False)

    assert result.events == [f"down:{button}", f"up:{button}"]
    assert backend.events == [("down", button), ("up", button)]
    assert backend.down == set()
    controller.update("point", pinch(button, gap=.6), .2)
    assert backend.count("down") == backend.count("up") == 1


@pytest.mark.parametrize("release_label,preserve_anchor", [("point", False), ("pinch_index", False), ("point", True)])
def test_pending_click_target_does_not_move_while_the_palm_or_fingertip_moves(release_label, preserve_anchor):
    controller, backend = rig()
    controller.update("pinch_index", pinch(), 0)
    controller.update("pinch_index", pinch(dx=.05), .1)
    controller.update(release_label, pinch(gap=.28, dx=.1), .15,
                      allow_press=False, preserve_anchor=preserve_anchor)
    assert backend.pos == (960, 540)
    assert backend.events == [("down", "left"), ("up", "left")]


@pytest.mark.parametrize("invalid", [None, np.zeros((20, 3)), np.full((21, 3), np.nan)])
def test_invalid_landmarks_cancel_a_pending_tap_without_input(invalid):
    controller, backend = rig()
    controller.update("pinch_index", pinch(), 0)
    controller.update("point", invalid, .1)
    controller.update("point", pinch(gap=.6), .2)
    assert backend.count("down") == 0


@pytest.mark.parametrize("button", ["left", "right"])
def test_continuous_pinch_presses_after_hold_delay_then_drags_without_a_click(button):
    controller, backend = rig()
    label = "pinch_index" if button == "left" else "pinch_middle"
    controller.update(label, pinch(button), 0)
    controller.update(label, pinch(button, dx=.03), .24)
    assert backend.events == []
    held = controller.update(label, pinch(button, dx=.04), .26)
    assert held.button == button
    assert backend.events[0] == ("down", button)
    assert backend.count("up") == 0
    assert backend.pos[0] > 960
    for i in range(10):
        controller.update("point", pinch(button, gap=.27, dx=.05 + .003 * i), .3 + i / 30)
    assert backend.count("down", button) == 1
    assert backend.count("up") == 0
    released = controller.update("point", pinch(button, gap=.34, dx=.1), .7)
    assert released.events == [f"up:{button}"]
    assert backend.down == set()


@pytest.mark.parametrize("cancel", ["loss", "fist", "scroll", "unknown", "halt", "suspend", "release_all"])
def test_cancelled_pending_pinch_never_becomes_a_click_later(cancel):
    controller, backend = rig()
    controller.update("pinch_index", pinch(), 0)
    if cancel == "loss":
        controller.update("none", None, .1)
    elif cancel in ("fist", "scroll", "unknown"):
        pose = (make_hand("fist", np.random.default_rng(1), noise=0,
                          curls=dict.fromkeys(("index", "middle", "ring", "pinky"), 1.0))
                if cancel == "fist" else pinch())
        controller.update("none" if cancel == "unknown" else cancel, pose, .1)
    else:
        getattr(controller, "suspend_motion" if cancel == "suspend" else cancel)()
    controller.update("point", pinch(gap=.6), .2)
    assert backend.count("down") == backend.count("up") == 0
    assert backend.down == set()


def test_two_short_pinches_produce_two_complete_clicks():
    controller, backend = rig()
    for t in (0, .3):
        controller.update("pinch_index", pinch(), t)
        controller.update("point", pinch(gap=.28), t + .1, allow_press=False)
    assert backend.events == [("down", "left"), ("up", "left")] * 2


def test_small_gap_wobbles_do_not_split_a_pending_pinch():
    controller, backend = rig()
    controller.update("pinch_index", pinch(), 0)
    for i in range(1, 6):
        controller.update("point", pinch(gap=.24 if i % 2 else .18), i * .03, allow_press=False)
    assert backend.events == []
    controller.update("point", pinch(gap=.28), .2, allow_press=False)
    assert backend.events == [("down", "left"), ("up", "left")]


def test_switching_to_a_different_pinch_cancels_instead_of_completing_the_first_click():
    controller, backend = rig()
    controller.update("pinch_index", pinch("left"), 0)
    controller.update("pinch_middle", pinch("right"), .1)
    assert backend.events == []
    assert backend.down == set()


def test_tap_release_failure_remains_tracked_for_halt_to_retry():
    class ReleaseFailsOnce(FakeBackend):
        failed = False

        def release(self, button):
            if not self.failed:
                self.failed = True
                raise OSError("temporary release failure")
            super().release(button)

    backend = ReleaseFailsOnce()
    controller = MouseController(Config(cursor_mode="relative"), backend)
    controller.update("pinch_index", pinch(), 0)
    with pytest.raises(OSError, match="temporary release failure"):
        controller.update("point", pinch(gap=.28), .1, allow_press=False)
    assert controller.button == "left"
    assert backend.down == {"left"}
    controller.halt()
    assert controller.button is None
    assert backend.events == [("down", "left"), ("up", "left")]


def test_a_physical_mouse_move_cancels_the_pending_click_instead_of_clicking_elsewhere():
    controller, backend = rig()
    controller.update("pinch_index", pinch(), 0)
    backend.pos = (300, 200)
    controller.update("point", pinch(gap=.28), .1, allow_press=False)
    assert backend.count("down") == 0
    assert backend.pos == (300, 200)


@pytest.mark.parametrize("button", ["left", "right"])
def test_real_recognizer_completes_a_short_tap_through_the_slightly_open_pose(button):
    from airtouch.engine import GestureEngine

    backend = FakeBackend()
    engine = GestureEngine(Config(cursor_mode="relative"), backend)
    for t in (0, .04, .08):
        engine.process(pinch(button, gap=.05), t)
    assert backend.events == []

    transition = engine.process(pinch(button, gap=.24), .12)
    assert backend.events == []
    released = engine.process(pinch(button, gap=.28), .16)

    if button == "left":
        # The 2-D classifier calls the bent index finger a fist during opening.
        assert transition.raw_label == released.raw_label == "fist"
    assert released.events == [f"down:{button}", f"up:{button}"]
    assert backend.events == [("down", button), ("up", button)]
    assert backend.down == set()


def test_real_recognizer_does_not_turn_a_prolonged_ambiguous_fist_into_a_drag():
    from airtouch.engine import GestureEngine

    backend = FakeBackend()
    engine = GestureEngine(Config(cursor_mode="relative"), backend)
    for t in (0, .04, .08):
        engine.process(pinch(gap=.05), t)
    for t in (.12, .2, .34):
        result = engine.process(pinch(gap=.24), t)
        assert result.raw_label == "fist"
        assert result.button is None
    engine.process(pinch(gap=.28), .4)
    assert backend.events == []


@pytest.mark.parametrize("fps", [15, 24, 30, 60])
@pytest.mark.parametrize("button", ["left", "right"])
def test_release_on_the_first_frame_past_the_hold_deadline_completes_the_pending_tap(fps, button):
    import math
    from airtouch.engine import GestureEngine

    backend = FakeBackend()
    engine = GestureEngine(Config(cursor_mode="relative"), backend)
    # Confirmation starts on frame 1. The last closed frame is still before
    # its hold deadline; the next camera frame already shows fingers apart.
    release_frame = math.ceil(1 + MouseController.PINCH_DRAG_HOLD_S * fps)
    for frame in range(release_frame):
        engine.process(pinch(button), frame / fps)
    assert backend.events == []

    result = engine.process(pinch(button, gap=.28), release_frame / fps)

    assert result.events == [f"down:{button}", f"up:{button}"]
    assert backend.down == set()


def test_real_recognizer_cancels_a_pending_tap_when_the_index_really_tucks_into_a_fist():
    from airtouch.engine import GestureEngine
    from airtouch.gestures_rule import PRESS_REACH, finger_reach

    backend = FakeBackend()
    engine = GestureEngine(Config(cursor_mode="relative"), backend)
    for t in (0, .04, .08):
        engine.process(pinch(gap=.05), t)
    tucked = make_hand("fist", np.random.default_rng(1), noise=0,
                       curls=dict.fromkeys(("index", "middle", "ring", "pinky"), 1.0))
    assert finger_reach(to_metric(tucked, 4 / 3), "index") <= PRESS_REACH["left"]
    engine.process(tucked, .12)
    engine.process(pinch(gap=.28), .16)
    assert backend.events == []


@pytest.mark.parametrize("seed", [11, 73, 92])
def test_a_natural_fist_with_the_index_near_the_knuckle_cancels_a_pending_tap(seed):
    from airtouch.engine import GestureEngine

    backend = FakeBackend()
    engine = GestureEngine(Config(cursor_mode="relative"), backend)
    engine.process(pinch(), 0)
    engine.process(pinch(), .04)
    result = engine.process(make_hand("fist", np.random.default_rng(seed), noise=0), .12)
    assert result.raw_label == "fist"
    assert backend.events == []


def test_natural_fist_variation_cancels_pending_taps_without_false_clicks():
    from airtouch.engine import GestureEngine

    for seed in range(500):
        backend = FakeBackend()
        engine = GestureEngine(Config(cursor_mode="relative"), backend)
        engine.process(pinch(), 0)
        engine.process(pinch(), .04)
        result = engine.process(make_hand("fist", np.random.default_rng(seed), noise=0), .12)
        assert result.raw_label == "fist", seed
        assert backend.events == [], seed
