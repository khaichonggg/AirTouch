"""Repairing config.json takes effect on the next CLI or tray activation."""
import threading

import pytest

from airtouch import app, config, session
from airtouch.config import Config
from airtouch.controller import FakeBackend


class Resource:
    def __init__(self):
        self.closed = False

    def start(self):
        return self

    def close(self):
        self.closed = True

    release = close


class Monitor:
    def __init__(self, *_args):
        pass

    def finish(self, _now):
        pass


@pytest.mark.parametrize("entry", ["tray", "cli"])
@pytest.mark.parametrize("broken", [b'{"dwell_enabled":tru', b'\xff', b'[' * 1500 + b']' * 1500],
                         ids=["truncated", "invalid-encoding", "excessive-nesting"])
def test_repaired_config_is_read_on_next_activation(tmp_path, monkeypatch, entry, broken):
    path = tmp_path / "config.json"
    path.write_bytes(broken)
    monkeypatch.setattr(config, "DEFAULT_CONFIG_PATH", path)
    monkeypatch.setattr(session, "PerformanceMonitor", Monitor)
    seen = []
    monkeypatch.setattr(session, "_capture_loop", lambda stop, cfg, *_args: seen.append(cfg))
    monkeypatch.setattr(app.LiveSession, "run", lambda self: seen.append(self.cfg))

    def activate():
        camera, tracker, backend = Resource(), Resource(), FakeBackend()
        if entry == "tray":
            session.run_headless(threading.Event(), camera_factory=lambda cfg: camera,
                                 tracker_factory=lambda cfg: tracker, backend=backend)
        else:
            assert app.main(show_window=False, camera=camera, tracker=tracker, backend=backend) == 0
        assert camera.closed and tracker.closed
        assert backend.down == set()

    activate()
    assert seen[-1] == Config()
    assert path.read_bytes() == broken

    repaired = Config(frame_width=960, frame_height=720, cursor_mode="relative", dwell_enabled=False)
    repaired.save(path)
    activate()
    assert len(seen) == 2
    assert seen[-1] == repaired
