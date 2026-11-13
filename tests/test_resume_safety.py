"""Processing-gap recovery, using fake clocks/devices and the real engine."""
import runpy
import threading
from pathlib import Path

import numpy as np
import pytest
import cv2

from airtouch import app, session
from airtouch.config import Config, PROJECT_ROOT
from airtouch.controller import FakeBackend
from airtouch.engine import GestureEngine

helpers = runpy.run_path(str(PROJECT_ROOT / "tests/test_relative_engine.py"))
hand = helpers["hand"]


@pytest.mark.parametrize("api", ["tray", "cli"])
def test_pause_during_image_preparation_discards_result_before_input(monkeypatch, api):
    cfg, engine, backend = primed("pinch_index")
    clock = Clock()
    camera = Camera(clock, frame=True, gap=.01)
    tracker = Tracker([hand("pinch_index", dx=.25)])
    backend.pos = (300, 400)

    def paused_flip(frame, _code):
        clock.now += 10
        return frame

    monkeypatch.setattr(cv2, "flip", paused_flip)
    if api == "tray":
        monkeypatch.setattr(session, "time", clock)
        with pytest.raises(EndProbe):
            session._capture_loop(threading.Event(), cfg, camera, tracker, engine)
    else:
        monkeypatch.setattr(app, "time", clock)
        assert app.LiveSession(cfg, camera, tracker, engine, window=False).step()
    assert backend.events == [("up", "left")]
    assert backend.down == set()
    assert backend.pos == (300, 400)


@pytest.fixture(autouse=True)
def no_real_devices_or_windows(monkeypatch):
    def forbidden(*_args, **_kwargs):
        raise AssertionError("resume simulation reached a real device, window or input backend")

    for name in ("make_backend", "ThreadedCamera", "HandTracker"):
        monkeypatch.setattr(app, name, forbidden)
    monkeypatch.setattr("airtouch.controller.make_backend", forbidden)
    for name in ("VideoCapture", "namedWindow", "imshow", "waitKey", "getWindowProperty", "destroyAllWindows"):
        monkeypatch.setattr(cv2, name, forbidden)


class Clock:
    now = .06

    def perf_counter(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds


class EndProbe(Exception):
    pass


class Camera:
    def __init__(self, clock, frame=True, gap=10.0, age=None):
        self.clock, self.frame, self.gap = clock, frame, gap
        self.calls = 0
        self.seconds_since_last_frame = age

    def read_if_new(self, last_seq):
        self.calls += 1
        if self.calls > 1:
            raise EndProbe()
        self.clock.now += self.gap
        return 1, np.zeros((4, 6, 3), dtype=np.uint8) if self.frame else None


class Tracker:
    def __init__(self, hands):
        self.hands, self.calls = hands, 0

    def detect(self, frame, timestamp):
        self.calls += 1
        return self.hands


class Recognizer:
    def __init__(self, label):
        self.label = label

    def predict(self, _hand):
        return self.label, 1.0


def primed(label="point", mode="relative", dwell=False, zoom=False):
    cfg = Config(cursor_mode=mode, dwell_enabled=dwell)
    backend = FakeBackend()
    engine = GestureEngine(cfg, backend, Recognizer(label))
    hands = [hand(label)] if not zoom else [hand(dx=-.15), hand(dx=.15)]
    # Establish a real held drag before the recovery probe's clock origin.
    # Relative pinches now wait for a hold rather than pressing immediately.
    timestamps = ((-.54, -.27, .06) if mode == "relative" and label.startswith("pinch_")
                  else (0.0, .03, .06))
    for timestamp in timestamps:
        engine.process_hands(hands, timestamp)
    backend.events.clear()
    return cfg, engine, backend


def resume(monkeypatch, api, cfg, engine, hands, frame=True, gap=10.0, age=None):
    clock = Clock()
    camera, tracker = Camera(clock, frame, gap, age), Tracker(hands)
    if api == "tray":
        monkeypatch.setattr(session, "time", clock)
        with pytest.raises(EndProbe):
            session._capture_loop(threading.Event(), cfg, camera, tracker, engine)
    else:
        monkeypatch.setattr(app, "time", clock)
        live = app.LiveSession(cfg, camera, tracker, engine, window=False)
        assert live.step()
    return tracker


@pytest.mark.parametrize("api", ["tray", "cli"])
@pytest.mark.parametrize("mode", ["relative", "absolute"])
@pytest.mark.parametrize("label,button", [("pinch_index", "left"), ("pinch_middle", "right")])
def test_long_gap_releases_old_drag_before_resumed_frame(monkeypatch, api, mode, label, button):
    cfg, engine, backend = primed(label, mode)
    assert backend.down == {button}
    backend.pos = (300, 400)
    resume(monkeypatch, api, cfg, engine, [hand(label, dx=.25)])
    assert backend.events == [("up", button)]
    assert backend.down == set()
    assert backend.pos == (300, 400)
    for index in range(1, 11 if mode == "relative" else 3):
        fresh = engine.process(hand(label, dx=.25), 10.06 + index * .03)
        if mode == "relative" and index < 10:
            assert fresh.button is None   # new confirmation and hold are both required
    assert backend.count("down", button) == 1
    assert backend.down == {button}


@pytest.mark.parametrize("api", ["tray", "cli"])
def test_long_gap_reanchors_relative_point_at_current_cursor(monkeypatch, api):
    cfg, engine, backend = primed()
    backend.pos = (300, 400)
    resume(monkeypatch, api, cfg, engine, [hand(dx=.25)])
    assert backend.events == []
    assert backend.pos == (300, 400)


@pytest.mark.parametrize("api", ["tray", "cli"])
def test_elapsed_pause_cannot_complete_a_pending_dwell_click(monkeypatch, api):
    cfg, engine, backend = primed(dwell=True)
    resume(monkeypatch, api, cfg, engine, [hand()])
    assert backend.count("down") == 0, backend.events
    engine.process(hand(), 10.5)
    assert backend.count("down") == 0
    engine.process(hand(), 11.0)
    assert backend.count("down", "left") == 1
    assert backend.down == set()


@pytest.mark.parametrize("api", ["tray", "cli"])
def test_resumed_two_hand_span_does_not_reuse_old_zoom_distance(monkeypatch, api):
    cfg, engine, backend = primed(zoom=True)
    resume(monkeypatch, api, cfg, engine, [hand(dx=-.15), hand(dx=.35)])
    assert backend.count("zoom") == 0
    engine.process_hands([hand(dx=-.15), hand(dx=.5)], 11.0)
    assert backend.count("zoom") == 1


@pytest.mark.parametrize("api", ["tray", "cli"])
def test_waking_without_a_frame_releases_drag_without_camera_lost_warning(monkeypatch, api, capsys):
    cfg, engine, backend = primed("pinch_index")
    resume(monkeypatch, api, cfg, engine, [], frame=False, gap=3600.0)
    assert backend.events == [("up", "left")]
    assert backend.down == set()
    assert "摄像头" not in capsys.readouterr().out


@pytest.mark.parametrize("api", ["tray", "cli"])
def test_explicitly_stale_resumed_frame_is_not_recognized(monkeypatch, api):
    cfg, engine, backend = primed("pinch_index")
    tracker = resume(monkeypatch, api, cfg, engine, [hand("pinch_index", dx=.25)], age=10.0)
    assert tracker.calls == 0
    assert backend.events == [("up", "left")]
    assert backend.down == set()


@pytest.mark.parametrize("api", ["tray", "cli"])
def test_fresh_frames_after_stale_resume_start_a_new_relative_stroke(monkeypatch, api):
    cfg, engine, backend = primed()
    backend.pos = (300, 400)
    clock = Clock()

    class BufferedCamera:
        calls = 0
        seconds_since_last_frame = 10.0

        def read_if_new(self, last_seq):
            self.calls += 1
            if self.calls > 3:
                raise EndProbe()
            clock.now += 10.0 if self.calls == 1 else .03
            self.seconds_since_last_frame = 10.0 if self.calls == 1 else 0.0
            return self.calls, np.full((4, 6, 3), self.calls, dtype=np.uint8)

    class FreshTracker:
        seen = []

        def detect(self, frame, timestamp):
            identifier = int(frame[0, 0, 0])
            self.seen.append(identifier)
            return [hand(dx=.25 if identifier == 2 else .30)]

    camera, tracker = BufferedCamera(), FreshTracker()
    if api == "tray":
        monkeypatch.setattr(session, "time", clock)
        with pytest.raises(EndProbe):
            session._capture_loop(threading.Event(), cfg, camera, tracker, engine)
    else:
        monkeypatch.setattr(app, "time", clock)
        live = app.LiveSession(cfg, camera, tracker, engine, window=False)
        for _ in range(3):
            assert live.step()
    assert tracker.seen == [2, 3]
    assert backend.count("move") == 1
    assert backend.pos[0] > 300
    assert backend.count("down") == 0


@pytest.mark.parametrize("api", ["tray", "cli"])
def test_short_processing_gap_keeps_drag_grace(monkeypatch, api):
    cfg, engine, backend = primed("pinch_index")
    resume(monkeypatch, api, cfg, engine, [hand("pinch_index")], gap=.1)
    assert backend.down == {"left"}
    assert backend.count("up") == 0
    assert backend.count("down") == 0


@pytest.mark.parametrize("api", ["tray", "cli"])
def test_resume_does_not_enable_a_paused_engine(monkeypatch, api):
    cfg, engine, backend = primed("pinch_index")
    engine.set_enabled(False)
    backend.events.clear()
    resume(monkeypatch, api, cfg, engine, [hand("pinch_index")])
    assert engine.enabled is False
    assert backend.events == []


def run_in_call_pause(monkeypatch, api, label="point", mode="relative", dwell=False,
                      zoom=False, pause=10.0, no_frame_after=False):
    cfg = Config(cursor_mode=mode, dwell_enabled=dwell, dwell_ms=200)
    backend = FakeBackend()
    engine = GestureEngine(cfg, backend, Recognizer(label))
    clock = Clock()
    clock.now = 0.0
    warm_times = ((0.0, .1, .39, .4) if mode == "relative" and label.startswith("pinch_")
                  else (0.0, .1, .19, .2))

    class TimedCamera:
        calls = 0
        seconds_since_last_frame = 0.0

        def read_if_new(self, last_seq):
            self.calls += 1
            if self.calls <= 4:
                clock.now = warm_times[self.calls - 1]
                if self.calls == 4:
                    backend.events.clear()
                    backend.pos = (300, 400)
                return self.calls, np.zeros((4, 6, 3), dtype=np.uint8)
            if self.calls == 5 and no_frame_after:
                clock.now += .01
                return 4, None
            raise EndProbe()

    class PausingTracker:
        calls = 0

        def detect(self, frame, timestamp):
            self.calls += 1
            if self.calls == 4:
                clock.now += pause
            if zoom:
                return [hand(dx=-.15), hand(dx=.35 if self.calls == 4 else .15)]
            offset = .25 if self.calls == 4 and not dwell else 0.0
            return [hand(label, dx=offset)]

    camera, tracker = TimedCamera(), PausingTracker()
    if api == "tray":
        monkeypatch.setattr(session, "time", clock)
        with pytest.raises(EndProbe):
            session._capture_loop(threading.Event(), cfg, camera, tracker, engine)
    else:
        monkeypatch.setattr(app, "time", clock)
        live = app.LiveSession(cfg, camera, tracker, engine, window=False)
        for _ in range(5 if no_frame_after else 4):
            assert live.step()
    return cfg, engine, backend


@pytest.mark.parametrize("api", ["tray", "cli"])
@pytest.mark.parametrize("mode", ["relative", "absolute"])
@pytest.mark.parametrize("label,button", [("pinch_index", "left"), ("pinch_middle", "right")])
def test_pause_inside_tracking_discards_drag_result_before_input(monkeypatch, api, mode, label, button):
    _cfg, engine, backend = run_in_call_pause(monkeypatch, api, label, mode)
    assert backend.events == [("up", button)]
    assert backend.down == set()
    assert backend.pos == (300, 400)
    resumed_at = 10.4 if mode == "relative" else 10.2
    for index in range(1, 12 if mode == "relative" else 4):
        result = engine.process(hand(label, dx=.25), resumed_at + index * .03)
        if index == 1 or mode == "relative" and index < 11:
            assert result.button is None
    assert result.button == button
    assert backend.count("down", button) == 1


@pytest.mark.parametrize("api", ["tray", "cli"])
def test_pause_inside_tracking_cannot_complete_due_dwell(monkeypatch, api):
    _cfg, engine, backend = run_in_call_pause(monkeypatch, api, dwell=True)
    assert backend.count("down") == 0, backend.events
    engine.process(hand(), 10.23)
    engine.process(hand(), 10.4)
    assert backend.count("down") == 0
    engine.process(hand(), 10.44)
    assert backend.count("down", "left") == 1


@pytest.mark.parametrize("api", ["tray", "cli"])
def test_pause_inside_tracking_cannot_emit_a_stale_zoom(monkeypatch, api):
    _cfg, engine, backend = run_in_call_pause(monkeypatch, api, zoom=True)
    assert backend.count("zoom") == 0
    engine.process_hands([hand(dx=-.15), hand(dx=.35)], 10.23)
    assert backend.count("zoom") == 0
    engine.process_hands([hand(dx=-.15), hand(dx=.5)], 10.26)
    assert backend.count("zoom") == 1


@pytest.mark.parametrize("api", ["tray", "cli"])
def test_pause_inside_tracking_cannot_move_a_pointing_cursor(monkeypatch, api):
    _cfg, _engine, backend = run_in_call_pause(monkeypatch, api)
    assert backend.events == []
    assert backend.pos == (300, 400)


@pytest.mark.parametrize("api", ["tray", "cli"])
def test_in_call_pause_updates_camera_wait_baseline(monkeypatch, api, capsys):
    _cfg, _engine, backend = run_in_call_pause(monkeypatch, api, "pinch_index", no_frame_after=True)
    assert backend.events == [("up", "left")]
    assert backend.down == set()
    assert "摄像头" not in capsys.readouterr().out


@pytest.mark.parametrize("api", ["tray", "cli"])
def test_short_tracking_delay_keeps_existing_drag(monkeypatch, api):
    _cfg, _engine, backend = run_in_call_pause(monkeypatch, api, "pinch_index", pause=.01)
    assert backend.down == {"left"}
    assert backend.count("up") == 0
    assert backend.count("down") == 0
