"""Prediction pauses expire the whole frame before input, then recover fresh."""
import threading

import cv2
import numpy as np
import pytest

from airtouch import engine as engine_module, gesture_classifier, session
from airtouch.config import Config
from airtouch.controller import FakeBackend
from ml.synthetic_data import HandPose, make_hand


@pytest.mark.parametrize("mode,held,stage,before,prediction,expired", [
    ("absolute", False, "predict", 0, 10, True),
    ("absolute", True, "predict", 0, 10, True),
    ("absolute", False, "flip", 1.2, 1.2, True),
    ("absolute", False, "detect", 1.2, 1.2, True),
    ("absolute", False, "prepare", 2.1, 0, True),
    ("absolute", False, "detect", 1.2, .2, False),
    ("absolute", False, "predict", 0, 0, False),
    ("relative", False, "predict", 0, 10, True),
], ids=["predict-pause", "held-drag-pause", "flip-plus-predict", "detect-plus-predict",
        "expired-before-predict", "short-combined-control", "normal-control", "relative-recovery"])
def test_expired_prediction_releases_old_input_and_recovers(monkeypatch, mode, held, stage,
                                                          before, prediction, expired):
    stop = threading.Event()
    backend = FakeBackend()
    hand = make_hand("pinch_index", np.random.default_rng(92), pose=HandPose(), thumb_gap=0, noise=0)
    priming_frames = 1 if held else 0
    target = priming_frames + 1
    fresh_frames = 3 if mode == "relative" else 1
    after_target, after_first_fresh, after_second_fresh = [], [], []
    prediction_calls = []

    class Clock:
        now = 0.0

        def perf_counter(self):
            return self.now

        def sleep(self, seconds):
            self.now += seconds

    clock = Clock()

    class Camera:
        reads = 0
        released = False

        def read_if_new(self, _last_seq):
            self.reads += 1
            if self.reads == target:
                assert backend.down == ({"left"} if held else set())
                backend.events.clear()
            elif self.reads == target + 1:
                after_target.append((list(backend.events), set(backend.down)))
                backend.events.clear()
            elif self.reads == target + 2 and mode == "relative":
                after_first_fresh.append(set(backend.down))
            elif self.reads == target + 3 and mode == "relative":
                after_second_fresh.append(set(backend.down))
            if self.reads > target + fresh_frames:
                stop.set()
                return self.reads, None
            # The first two fresh contacts only confirm a pending tap. A
            # deliberate hold must elapse before the third may start a drag.
            clock.now += .26 if mode == "relative" and self.reads == target + 3 else .01
            return self.reads, np.zeros((8, 8, 3), dtype=np.uint8)

        def release(self):
            self.released = True

    class Tracker:
        closed = False

        def detect(self, _frame, _timestamp):
            if camera.reads == target and stage == "detect":
                clock.now += before
            return [hand]

        def close(self):
            self.closed = True

    class Recognizer:
        name = "rule"

        def predict(self, _hand):
            prediction_calls.append(camera.reads)
            if camera.reads == target:
                clock.now += prediction
            return "pinch_index", 1.0

    camera, tracker = Camera(), Tracker()
    original_valid = engine_module.is_valid
    prepared = False

    def valid(value):
        nonlocal prepared
        if camera.reads == target and stage == "prepare" and not prepared:
            prepared = True
            clock.now += before
        return original_valid(value)

    def flip(frame, _code):
        if camera.reads == target and stage == "flip":
            clock.now += before
        return frame

    monkeypatch.setattr(engine_module, "is_valid", valid)
    monkeypatch.setattr(cv2, "flip", flip)
    monkeypatch.setattr(session, "time", clock)
    monkeypatch.setattr(gesture_classifier, "load_recognizer", lambda cfg: (Recognizer(), "test"))
    session.run_headless(stop, Config(debounce_frames=1, cursor_mode=mode),
                         camera_factory=lambda cfg: camera, tracker_factory=lambda cfg: tracker,
                         backend=backend)

    assert camera.released and tracker.closed and backend.down == set()
    assert camera.reads == target + fresh_frames + 1  # the active run recovered; it did not turn OFF
    if expired:
        assert after_target == [([("up", "left")] if held else [], set())]
        assert backend.count("down", "left") == backend.count("up", "left") == 1
    else:
        assert after_target[0][1] == {"left"}
        assert sum(event == ("down", "left") for event in after_target[0][0]) == 1
        assert backend.count("down", "left") == 0
        assert backend.count("up", "left") == 1
    if stage == "prepare":
        assert target not in prediction_calls  # do not start inference on an already-expired frame
    if mode == "relative":
        assert after_first_fresh == [set()]  # stale contact cannot count toward fresh confirmation
        assert after_second_fresh == [set()]  # confirmation alone cannot start a long press


def test_off_during_long_prediction_takes_priority_over_frame_recovery(monkeypatch):
    stop = threading.Event()
    backend = FakeBackend()
    hand = make_hand("pinch_index", np.random.default_rng(93), pose=HandPose(), thumb_gap=0, noise=0)
    now = [0.0]
    metrics, statuses = [], []

    class Camera:
        released = False

        def read_if_new(self, _seq):
            return 1, np.zeros((8, 8, 3), dtype=np.uint8)

        def release(self):
            self.released = True

    class Tracker:
        closed = False

        def detect(self, _frame, _timestamp):
            return [hand]

        def close(self):
            self.closed = True

    class Recognizer:
        name = "rule"

        def predict(self, _hand):
            now[0] = 10.0
            stop.set()
            return "pinch_index", 1.0

    class Monitor:
        def __init__(self, *_args):
            pass

        def start(self, _now):
            metrics.append("start")

        def idle(self, _now):
            metrics.append("idle")

        def frame(self, *_args, **_kwargs):
            metrics.append("frame")

        def finish(self, _now):
            metrics.append("finish")

    clock = type("Clock", (), {"perf_counter": staticmethod(lambda: now[0])})()
    camera, tracker = Camera(), Tracker()
    monkeypatch.setattr(session, "time", clock)
    monkeypatch.setattr(session, "PerformanceMonitor", Monitor)
    monkeypatch.setattr(gesture_classifier, "load_recognizer", lambda cfg: (Recognizer(), "test"))
    session.run_headless(stop, Config(debounce_frames=1, mirror=False), statuses.append,
                         camera_factory=lambda cfg: camera, tracker_factory=lambda cfg: tracker,
                         backend=backend)
    assert camera.released and tracker.closed and backend.events == []
    assert metrics == ["start", "finish"]  # OFF must not enter the expired-frame recovery path
    assert session.MSG_CLOSED in statuses
    assert not any(message.startswith(session.ERROR_PREFIX) for message in statuses)
