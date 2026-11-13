"""Unavailable configuration/device/model paths with injected failures only."""
from pathlib import Path
import runpy
import threading

import pytest

from airtouch.config import Config, PROJECT_ROOT
from airtouch.controller import FakeBackend
from airtouch import session

helpers = runpy.run_path(str(PROJECT_ROOT / "tests/test_session.py"))
FakeCamera, FakeTracker = helpers["FakeCamera"], helpers["FakeTracker"]


def test_config_stat_and_read_permission_failure_falls_back_safely(tmp_path, monkeypatch):
    path = tmp_path / "config.json"
    original_exists, original_read = Path.exists, Path.read_text

    def denied_exists(self):
        if self == path:
            raise PermissionError("config directory unavailable")
        return original_exists(self)

    def denied_read(self, *a, **kw):
        if self == path:
            raise PermissionError("config unavailable")
        return original_read(self, *a, **kw)

    monkeypatch.setattr(Path, "exists", denied_exists)
    monkeypatch.setattr(Path, "read_text", denied_read)
    assert Config.load(path) == Config()


def test_camera_permission_failure_stays_off_and_retry_can_recover():
    backend = FakeBackend()
    stop = threading.Event()
    statuses = []

    def denied(_cfg):
        raise PermissionError("camera permission denied")

    with pytest.raises(RuntimeError, match=session.MSG_NO_CAMERA):
        session.run_headless(stop, Config(), statuses.append,
                             camera_factory=denied, tracker_factory=lambda _cfg: pytest.fail("tracker must not open"),
                             backend=backend)
    assert backend.events == []
    camera = FakeCamera()

    class Tracker(FakeTracker):
        def detect(self, frame, timestamp):
            stop.set()
            return []

    tracker = Tracker()
    session.run_headless(stop, Config(), statuses.append,
                         camera_factory=lambda _cfg: camera, tracker_factory=lambda _cfg: tracker,
                         backend=backend)
    assert camera.released and tracker.closed
    assert session.MSG_READY in statuses and session.MSG_CLOSED in statuses
    assert backend.events == []


def test_model_permission_failure_closes_camera_and_retry_can_recover():
    backend = FakeBackend()
    stop = threading.Event()
    camera = FakeCamera()

    def denied(_cfg):
        raise PermissionError("model unavailable")

    with pytest.raises(RuntimeError, match=session.MSG_NO_TRACKER):
        session.run_headless(stop, Config(), camera_factory=lambda _cfg: camera,
                             tracker_factory=denied, backend=backend)
    assert camera.released
    assert backend.events == []
    recovered = FakeCamera()

    class Tracker(FakeTracker):
        def detect(self, frame, timestamp):
            stop.set()
            return []

    tracker = Tracker()
    session.run_headless(stop, Config(), camera_factory=lambda _cfg: recovered,
                         tracker_factory=lambda _cfg: tracker, backend=backend)
    assert recovered.released and tracker.closed
    assert backend.events == []
