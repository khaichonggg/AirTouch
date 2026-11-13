"""D-key dwell toggles start a fresh hold without disturbing manual drags."""
import runpy

import pytest

from airtouch.app import LiveSession
from airtouch.config import Config, PROJECT_ROOT
from airtouch.controller import FakeBackend
from airtouch.engine import GestureEngine

helpers = runpy.run_path(str(PROJECT_ROOT / "tests/test_relative_engine.py"))
hand, Recognizer = helpers["hand"], helpers["Recognizer"]


def rig(mode, label="point"):
    cfg = Config(cursor_mode=mode, debounce_frames=1, dwell_enabled=True, dwell_ms=900)
    backend, recognizer = FakeBackend(), Recognizer()
    recognizer.label = label
    engine = GestureEngine(cfg, backend, recognizer)
    live = LiveSession(cfg, None, None, engine, False, None)
    return engine, backend, live


@pytest.mark.parametrize("mode", ["absolute", "relative"])
@pytest.mark.parametrize("previous_click", [False, True], ids=["pending", "already-clicked"])
@pytest.mark.parametrize("disabled_frame", [False, True], ids=["no-frame-between-toggles", "disabled-frames"])
def test_reenabled_dwell_requires_a_new_full_hold(mode, previous_click, disabled_frame):
    engine, backend, live = rig(mode)
    engine.process(hand(), 0)
    engine.process(hand(), .3)
    if previous_click:
        engine.process(hand(), 1.0)
        assert backend.count("down", "left") == 1
    backend.events.clear()

    live._toggle_dwell()
    if disabled_frame:
        engine.process(hand(), 3.0)
    assert backend.events == []
    live._toggle_dwell()

    engine.process(hand(), 3.01)
    assert backend.events == []
    engine.process(hand(), 3.90)  # 890ms of the new 900ms hold
    assert backend.events == []
    engine.process(hand(), 4.0)
    assert backend.events == [("down", "left"), ("up", "left")]
    assert backend.down == set()


@pytest.mark.parametrize("mode", ["absolute", "relative"])
@pytest.mark.parametrize("pose,button", [("pinch_index", "left"), ("pinch_middle", "right")])
def test_dwell_toggles_do_not_release_or_repress_manual_drag(mode, pose, button):
    engine, backend, live = rig(mode, pose)
    times = (-.30, -.27, -.24, 0, .03, .06) if mode == "relative" else (0, .03, .06)
    for time in times:
        engine.process(hand(pose), time)
    assert backend.down == {button}
    backend.events.clear()
    live._toggle_dwell()
    live._toggle_dwell()
    engine.process(hand(pose, dx=.01), .09)
    assert backend.down == {button}
    assert backend.count("down") == backend.count("up") == 0
    engine.process(hand(pose, touching=False), .12)
    assert backend.down == set()
    assert backend.count("up", button) == 1
