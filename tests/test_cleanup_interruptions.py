"""Cancellation during teardown must not abandon resources owned by this run."""
import threading
from types import SimpleNamespace

import pytest

from airtouch import app, engine as engine_module, gesture_classifier, session
from airtouch.config import Config
from airtouch.controller import FakeBackend


class Resource:
    def __init__(self, name, events, failures):
        self.name, self.events, self.failures = name, events, failures
        self.closed = False

    def _close(self, *_args):
        self.events.append(self.name)
        failure = self.failures.get(self.name)
        if failure is not None:
            raise failure
        self.closed = True

    reset = close = release = finish = _close


def wire_resources(monkeypatch, failures, *, primary=None):
    events = []
    parts = {name: Resource(name, events, failures)
             for name in ("engine", "monitor", "halo", "tracker", "camera", "window")}
    parts["camera"].start = lambda: parts["camera"]
    recognizer = SimpleNamespace(name="rule")
    monkeypatch.setattr(gesture_classifier, "load_recognizer", lambda _cfg: (recognizer, "rule"))
    monkeypatch.setattr(app, "load_recognizer", lambda _cfg: (recognizer, "rule"))
    monkeypatch.setattr(engine_module, "GestureEngine", lambda *_args: parts["engine"])
    monkeypatch.setattr(app, "GestureEngine", lambda *_args: parts["engine"])
    monkeypatch.setattr(session, "PerformanceMonitor", lambda *_args: parts["monitor"])
    from airtouch import cursor_halo
    monkeypatch.setattr(cursor_halo, "CursorHalo", lambda: parts["halo"])

    def end_loop(*_args):
        if primary is not None:
            raise primary

    monkeypatch.setattr(session, "_capture_loop", end_loop)
    monkeypatch.setattr(app.LiveSession, "__init__", lambda self, *_args: None)
    monkeypatch.setattr(app.LiveSession, "run", end_loop)
    monkeypatch.setattr(app, "_open_window", lambda: True)
    monkeypatch.setattr(app.cv2, "destroyAllWindows", parts["window"].close)
    monkeypatch.setattr(app, "_say_started", lambda *_args: None)
    return parts, events


def run_headless(parts):
    session.run_headless(threading.Event(), Config(cursor_halo=True),
                         camera_factory=lambda _cfg: parts["camera"],
                         tracker_factory=lambda _cfg: parts["tracker"], backend=FakeBackend())


def run_cli(parts):
    try:
        return app.main(Config(), show_window=True, camera=parts["camera"],
                        tracker=parts["tracker"], backend=FakeBackend())
    except KeyboardInterrupt:
        pytest.fail("CLI leaked a cleanup Ctrl+C instead of finishing its normal exit")


@pytest.mark.parametrize("stage", ["engine", "monitor", "halo", "tracker"])
@pytest.mark.parametrize("failure_type", [KeyboardInterrupt, SystemExit])
def test_headless_cleanup_interruption_closes_remaining_resources(monkeypatch, stage, failure_type):
    interruption = failure_type("cancel cleanup")
    parts, events = wire_resources(monkeypatch, {stage: interruption})
    with pytest.raises(failure_type) as caught:
        run_headless(parts)
    assert caught.value is interruption
    assert events == ["engine", "monitor", "halo", "tracker", "camera"]
    assert parts["camera"].closed


@pytest.mark.parametrize("stage", ["engine", "tracker", "camera"])
@pytest.mark.parametrize("failure_type", [KeyboardInterrupt, SystemExit])
def test_cli_cleanup_interruption_still_closes_window_and_later_resources(monkeypatch, stage, failure_type):
    interruption = failure_type("cancel cleanup")
    parts, events = wire_resources(monkeypatch, {stage: interruption})
    if failure_type is KeyboardInterrupt:
        assert run_cli(parts) == 0  # Ctrl+C is the public CLI's normal exit.
    else:
        with pytest.raises(failure_type) as caught:
            run_cli(parts)
        assert caught.value is interruption
    assert events == ["engine", "tracker", "camera", "window"]
    assert parts["window"].closed
    if stage != "camera":
        assert parts["camera"].closed


@pytest.mark.parametrize("primary_type", [ValueError, KeyboardInterrupt, SystemExit])
def test_headless_primary_failure_survives_cleanup_interruption(monkeypatch, primary_type):
    primary = primary_type("first failure")
    parts, events = wire_resources(monkeypatch, {"tracker": SystemExit("cleanup failure")}, primary=primary)
    with pytest.raises(primary_type) as caught:
        run_headless(parts)
    assert caught.value is primary
    assert events == ["engine", "monitor", "halo", "tracker", "camera"]
    assert parts["camera"].closed


@pytest.mark.parametrize("cleanup_type", [KeyboardInterrupt, SystemExit])
def test_cli_primary_system_exit_survives_cleanup_interruption(monkeypatch, cleanup_type):
    primary = SystemExit("first exit")
    parts, events = wire_resources(monkeypatch, {"tracker": cleanup_type("cleanup failure")}, primary=primary)
    with pytest.raises(SystemExit) as caught:
        run_cli(parts)
    assert caught.value is primary
    assert events == ["engine", "tracker", "camera", "window"]
    assert parts["camera"].closed and parts["window"].closed


def test_cli_reported_error_keeps_failure_code_after_cleanup_ctrl_c(monkeypatch):
    parts, events = wire_resources(monkeypatch, {"tracker": KeyboardInterrupt()},
                                   primary=ValueError("loop failure"))
    assert run_cli(parts) == 1
    assert events == ["engine", "tracker", "camera", "window"]
    assert parts["camera"].closed and parts["window"].closed


def test_headless_first_cleanup_interruption_is_preserved_after_later_interruptions(monkeypatch):
    first = KeyboardInterrupt("first cancellation")
    parts, events = wire_resources(monkeypatch, {"engine": first, "halo": SystemExit("second cancellation")})
    with pytest.raises(KeyboardInterrupt) as caught:
        run_headless(parts)
    assert caught.value is first
    assert events == ["engine", "monitor", "halo", "tracker", "camera"]
    assert parts["camera"].closed and parts["tracker"].closed


def test_cli_later_system_exit_is_not_swallowed_after_cleanup_ctrl_c(monkeypatch):
    exit_signal = SystemExit("exit during camera release")
    parts, events = wire_resources(monkeypatch, {"engine": KeyboardInterrupt(), "camera": exit_signal})
    with pytest.raises(SystemExit) as caught:
        run_cli(parts)
    assert caught.value is exit_signal
    assert events == ["engine", "tracker", "camera", "window"]
    assert parts["tracker"].closed and parts["window"].closed


@pytest.mark.parametrize("mode", ["headless", "cli"])
def test_callers_handled_exception_does_not_swallow_this_runs_cleanup_exit(monkeypatch, mode):
    interruption = SystemExit("cleanup exit")
    parts, events = wire_resources(monkeypatch, {"tracker": interruption})
    try:
        raise ValueError("already handled by caller")
    except ValueError:
        with pytest.raises(SystemExit) as caught:
            (run_headless if mode == "headless" else run_cli)(parts)
    assert caught.value is interruption
    assert parts["camera"].closed
    assert events[-1] == ("camera" if mode == "headless" else "window")


@pytest.mark.parametrize("failure_type", [KeyboardInterrupt, SystemExit])
def test_cli_error_reporting_interruption_survives_later_cleanup_exit(monkeypatch, failure_type):
    report_failure = failure_type("interrupted error report")
    parts, events = wire_resources(monkeypatch, {"tracker": SystemExit("later cleanup exit")},
                                   primary=ValueError("loop failure"))
    def interrupted_report(message):
        if message.startswith("程序遇到问题"):
            raise report_failure
    monkeypatch.setattr(app, "say", interrupted_report)
    with pytest.raises(failure_type) as caught:
        app.main(Config(), show_window=True, camera=parts["camera"],
                 tracker=parts["tracker"], backend=FakeBackend())
    assert caught.value is report_failure
    assert events == ["engine", "tracker", "camera", "window"]
    assert parts["camera"].closed and parts["window"].closed
