"""An OFF request during a blocked dependency must discard its returned frame."""
import threading

import cv2
import numpy as np
import pytest

from airtouch import gesture_classifier, session
from airtouch.config import Config
from airtouch.controller import FakeBackend
from ml.synthetic_data import HandPose, make_hand


@pytest.mark.parametrize("stage", ["read", "flip", "detect", "predict"])
@pytest.mark.parametrize("cancel", [True, False], ids=["cancelled", "active-control"])
@pytest.mark.parametrize("held", [False, True], ids=["new-press", "held-drag"])
def test_off_during_dependency_call_cannot_click_on_its_return(monkeypatch, stage, cancel, held):
    stop, entered, unblock = threading.Event(), threading.Event(), threading.Event()
    hand = make_hand("pinch_index", np.random.default_rng(1), pose=HandPose(), thumb_gap=0, noise=0)
    backend = FakeBackend()
    failures = []
    phase = 2 if held else 1
    calls = {}
    shifted = hand.copy()
    shifted[:, 0] += .12

    def pause(where):
        calls[where] = calls.get(where, 0) + 1
        if stage == where and calls[where] == phase:
            entered.set()
            assert unblock.wait(2), "test failed to unblock its fake dependency"

    class Camera:
        released = False
        calls = 0

        def read_if_new(self, _last_seq):
            self.calls += 1
            if self.calls > phase:
                stop.set()
                return self.calls, None
            pause("read")
            return self.calls, np.zeros((8, 8, 3), dtype=np.uint8)

        def release(self):
            self.released = True

    class Tracker:
        calls = 0
        closed = False

        def detect(self, _frame, _timestamp):
            self.calls += 1
            pause("detect")
            return [hand if self.calls == 1 else shifted]

        def close(self):
            self.closed = True

    class Recognizer:
        name = "rule"

        def predict(self, _hand):
            pause("predict")
            return "pinch_index", 1.0

    camera, tracker = Camera(), Tracker()
    monkeypatch.setattr(gesture_classifier, "load_recognizer", lambda cfg: (Recognizer(), "test"))

    def flip(frame, _code):
        pause("flip")
        return frame

    monkeypatch.setattr(cv2, "flip", flip)

    def run():
        try:
            session.run_headless(stop, Config(debounce_frames=1), camera_factory=lambda cfg: camera,
                                 tracker_factory=lambda cfg: tracker, backend=backend)
        except BaseException as error:
            failures.append(error)

    thread = threading.Thread(target=run, daemon=True)
    try:
        thread.start()
        assert entered.wait(2)
        if cancel:
            assert backend.down == ({"left"} if held else set())
            backend.events.clear()
            stop.set()  # equivalent to the tray's OFF request; no real input is used
        unblock.set()
        thread.join(2)
        assert not thread.is_alive()
    finally:
        stop.set()
        unblock.set()
        thread.join(2)
    assert not failures
    assert camera.released and tracker.closed
    assert backend.down == set()
    if cancel:
        assert backend.events == ([("up", "left")] if held else [])
        assert tracker.calls == phase - (stage in ("read", "flip"))
    else:
        assert backend.count("down", "left") == 1
        assert backend.count("up", "left") == 1
        assert tracker.calls == phase
