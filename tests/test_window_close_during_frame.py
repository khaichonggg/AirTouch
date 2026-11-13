"""Closing the fake preview during frame work must not resurrect it or send input."""
from pathlib import Path
import runpy

import cv2
import numpy as np
import pytest

from airtouch import app
from airtouch.config import Config, PROJECT_ROOT
from airtouch.controller import FakeBackend

helpers = runpy.run_path(str(PROJECT_ROOT / "tests/test_app.py"))


@pytest.mark.parametrize("stage", ["flip", "detect"])
@pytest.mark.parametrize("held", [False, True], ids=["new-press", "held-drag"])
def test_close_during_tracking_discards_returned_hand(monkeypatch, stage, held):
    backend = FakeBackend()

    class Window(helpers["FakeGui"]):
        visible = True
        recreated = 0

        def imshow(self, name, img):
            if not self.visible:
                self.recreated += 1
                self.visible = True
            super().imshow(name, img)

        def getWindowProperty(self, name, prop):
            return float(self.visible)

        def user_closes(self):
            assert backend.down == ({"left"} if held else set())
            self.visible = False
            backend.events.clear()

    window = Window(monkeypatch)
    rng = np.random.default_rng(84)
    pinch = helpers["pinch_hand"](rng)

    class Tracker(helpers["FakeTracker"]):
        def detect(self, frame, t_ms=None):
            if stage == "detect" and self.calls == 1:
                window.user_closes()
            return super().detect(frame, t_ms)

    flips = []

    def flip(frame, _code):
        flips.append(1)
        if stage == "flip" and len(flips) == 2:
            window.user_closes()
        return frame

    monkeypatch.setattr(cv2, "flip", flip)
    monkeypatch.setattr(app, "draw_hud", lambda *_args, **_kwargs: None)
    tracker = Tracker([pinch if held else None, pinch, pinch])
    code, camera, _ = helpers["run"](tracker, cfg=Config(debounce_frames=1), backend=backend,
                                     show_window=True)
    assert code == 0
    assert window.recreated == 0
    assert window.shown == 1 and tracker.calls == 2
    assert backend.events == ([("up", "left")] if held else [])
    assert backend.down == set()
    assert camera.released == tracker.closed == window.destroyed == 1


def test_close_during_hud_drawing_does_not_recreate_preview(monkeypatch):
    backend = FakeBackend()

    class Window(helpers["FakeGui"]):
        visible = True
        recreated = 0

        def imshow(self, name, img):
            if not self.visible:
                self.recreated += 1
                self.visible = True
            super().imshow(name, img)

        def getWindowProperty(self, name, prop):
            return float(self.visible)

    window = Window(monkeypatch)
    draws = []

    def draw(*_args, **_kwargs):
        draws.append(1)
        if len(draws) == 2:
            window.visible = False
            backend.events.clear()

    monkeypatch.setattr(app, "draw_hud", draw)
    pinch = helpers["pinch_hand"](np.random.default_rng(85))
    tracker = helpers["FakeTracker"]([None, pinch, pinch])
    code, camera, _ = helpers["run"](tracker, cfg=Config(debounce_frames=1), backend=backend,
                                     show_window=True)
    assert code == 0
    assert window.recreated == 0
    assert window.shown == 1 and tracker.calls == 2
    assert backend.events == [("up", "left")]
    assert backend.down == set()
    assert camera.released == tracker.closed == window.destroyed == 1
