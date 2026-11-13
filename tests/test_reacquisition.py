"""Real engine/controller; synthetic hands and FakeBackend only."""
import runpy
from pathlib import Path

import pytest

from airtouch.config import PROJECT_ROOT

helpers = runpy.run_path(str(PROJECT_ROOT / "tests/test_relative_engine.py"))
hand = helpers["hand"]
primed_engine = helpers["primed_engine"]


@pytest.mark.parametrize("pose,button", [("pinch_index", "left"), ("pinch_middle", "right")])
@pytest.mark.parametrize("mode", ["relative", "absolute"])
@pytest.mark.parametrize("returned_at", [.25, .26], ids=["deadline", "after-deadline"])
def test_reacquired_pinch_after_expired_hand_loss_releases_before_new_confirmation(pose, button, mode, returned_at):
    engine, backend, _ = primed_engine(pose, mode=mode)
    assert backend.down == {button}
    engine.process(None, .10)
    backend.pos = (300, 400)
    backend.events.clear()

    # No invalid frames arrive while tracking is delayed. The deadline is
    # nevertheless elapsed when the first valid hand arrives again.
    returned = engine.process(hand(pose, dx=.25), returned_at)

    assert returned.button is None
    assert backend.down == set()
    assert returned.events == [f"up:{button}"]
    assert backend.count("down") == 0
    assert backend.count("move") == 0
    assert backend.pos == (300, 400)
    if mode == "relative":
        assert returned.cursor == (300, 400)
    # Relative mode confirms after two raw frames; absolute mode retains its
    # established three-frame label debounce after a full loss/reset.
    confirmation_frames = 2 if mode == "relative" else 3
    for index in range(1, confirmation_frames):
        confirmed = engine.process(hand(pose, dx=.25), returned_at + index * .03)
    if mode == "relative":
        assert confirmed.button is None
        assert backend.down == set()
        assert backend.count("down") == 0
        confirmed = engine.process(hand(pose, dx=.25), returned_at + .30)
    assert confirmed.button == button
    assert backend.count("down", button) == 1


@pytest.mark.parametrize("pose,button", [("pinch_index", "left"), ("pinch_middle", "right")])
@pytest.mark.parametrize("mode", ["relative", "absolute"])
def test_reacquired_pinch_before_deadline_retains_drag_without_an_extra_press(pose, button, mode):
    engine, backend, _ = primed_engine(pose, mode=mode)
    engine.process(None, .10)
    backend.events.clear()

    returned = engine.process(hand(pose, dx=.25), .249)

    assert returned.button == button
    assert backend.down == {button}
    assert backend.count("down") == 0
    assert backend.count("up") == 0


@pytest.mark.parametrize("pose,button", [("pinch_index", "left"), ("pinch_middle", "right")])
def test_disable_resume_requires_fresh_pinch_confirmation_and_current_cursor(pose, button):
    engine, backend, _ = primed_engine(pose)
    engine.set_enabled(False)
    assert backend.down == set()
    backend.pos = (300, 400)
    backend.events.clear()
    assert engine.process(hand(pose), .12).enabled is False
    engine.set_enabled(True)

    first = engine.process(hand(pose, dx=.25), .15)
    assert first.button is None and first.cursor == (300, 400)
    assert backend.events == []
    second = engine.process(hand(pose, dx=.25), .18)
    assert second.button is None
    assert backend.events == []
    held = engine.process(hand(pose, dx=.25), .45)
    assert held.button == button
    assert backend.pos == (300, 400)
    assert backend.count("down", button) == 1


@pytest.mark.parametrize("single_hand", [True, False], ids=["one-hand", "no-hand"])
def test_direct_single_hand_api_ends_zoom_span_before_two_hands_return(single_hand):
    engine, backend, _ = primed_engine()
    left, right, farther = hand(dx=-.15), hand(dx=.15), hand(dx=.35)
    engine.process_hands([left, right], .10)
    engine.process(left if single_hand else None, .13)
    backend.events.clear()

    restarted = engine.process_hands([left, farther], .16)

    assert restarted.label == "zoom"
    assert restarted.events == []
    assert backend.count("zoom") == 0
    moved = engine.process_hands([left, hand(dx=.45)], .19)
    assert any(event.startswith("zoom:") for event in moved.events)
