"""Frame age includes classifier time; all time/devices/input are simulated."""
import cv2
import numpy as np
import pytest

from airtouch import app
from airtouch.config import Config
from airtouch.controller import FakeBackend
from airtouch.engine import GestureEngine
from ml.synthetic_data import HandPose, make_hand


@pytest.mark.parametrize("flip_gap,predict_gap", [(0, 10), (1.2, 1.2), (0, 0)],
                         ids=["prediction-pause", "combined-pause", "normal-control"])
def test_cli_discards_expired_prediction_and_recovers_on_fresh_frame(monkeypatch, flip_gap, predict_gap):
    class Clock:
        now = .06

        def perf_counter(self):
            return self.now

    clock = Clock()
    monkeypatch.setattr(app, "time", clock)
    flip_calls = []

    def flip(frame, _code):
        flip_calls.append(1)
        if len(flip_calls) == 1:
            clock.now += flip_gap
        return frame

    monkeypatch.setattr(cv2, "flip", flip)
    hand = make_hand("pinch_index", np.random.default_rng(91), pose=HandPose(), thumb_gap=0, noise=0)

    class Tracker:
        def detect(self, *_args):
            return [hand]

    class Recognizer:
        name = "rule"
        calls = 0

        def predict(self, _hand):
            self.calls += 1
            if self.calls == 1:
                clock.now += predict_gap
            return "pinch_index", 1.0

    cfg = Config(debounce_frames=1)
    backend, recognizer = FakeBackend(), Recognizer()
    engine = GestureEngine(cfg, backend, recognizer)
    live = app.LiveSession(cfg, None, Tracker(), engine, window=False)
    frame = np.zeros((8, 8, 3), dtype=np.uint8)
    expired = flip_gap + predict_gap > app.ASLEEP_GAP_S
    try:
        live._on_frame(frame, clock.now)
        assert engine.recognizer is recognizer
        if expired:
            assert backend.events == []
            assert backend.down == set()
            assert live.frames == 0
            assert live.last_frame_t == live.last_poll_t == clock.now
            clock.now += .03
            live._on_frame(frame, clock.now)
            assert engine.recognizer is recognizer
        assert live.frames == 1
        assert backend.down == {"left"}
        assert backend.count("down", "left") == 1
    finally:
        engine.reset()
    assert backend.down == set()
    assert backend.count("up", "left") == 1
