"""Missed device/restart boundaries; uses no camera, mouse or native model."""
import threading
import time
import logging

import cv2
import numpy as np
import pytest

from airtouch import camera as camera_module, tray as tray_module
from airtouch.camera import ThreadedCamera
from airtouch.session import CaptureWorker, ERROR_PREFIX
from airtouch.tray import TrayApp


@pytest.fixture(autouse=True)
def no_real_camera(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("real camera forbidden")
    monkeypatch.setattr(cv2, "VideoCapture", forbidden)


def wait_for(predicate, timeout=1):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(.001)
    return bool(predicate())


class BlockedCapture:
    def __init__(self):
        self.calls = 0
        self.blocked = threading.Event()
        self.unblock = threading.Event()
        self.closed = threading.Event()

    def isOpened(self):
        return True

    def set(self, *_args):
        return True

    def read(self):
        self.calls += 1
        if self.calls > 1:
            self.blocked.set()
            self.unblock.wait(3)
        return True, np.zeros((4, 6, 3), dtype=np.uint8)

    def release(self):
        self.closed.set()


@pytest.mark.parametrize("fresh_instance", [False, True])
def test_restart_waits_for_old_camera_driver_and_then_recovers(monkeypatch, fresh_instance):
    monkeypatch.setattr(camera_module, "JOIN_TIMEOUT_S", .02)
    old, new = BlockedCapture(), BlockedCapture()
    opens = []

    def open_capture(index):
        cap = old if not opens else new
        opens.append(cap)
        return cap

    camera = ThreadedCamera(capture_factory=open_capture, first_frame_timeout=.1).start()
    old_reader = camera._thread
    restart = ThreadedCamera(capture_factory=open_capture, first_frame_timeout=.1) if fresh_instance else camera
    second_error = None
    try:
        assert old.blocked.wait(1)
        camera.release()
        assert not old.closed.is_set(), "the original driver call is still active"
        try:
            restart.start()
        except RuntimeError as error:
            second_error = error
        assert opens == [old], "restart opened the same camera while its old read is still active"
        assert second_error is not None, "caller needs a clear retry-later error"
        old.unblock.set()
        assert old.closed.wait(1)
        old_reader.join(1)
        assert not old_reader.is_alive()
        restart.start()
        assert opens == [old, new], "device should become reusable after actual driver closure"
    finally:
        old.unblock.set()
        new.unblock.set()
        camera.release()
        restart.release()
        old_reader.join(1)
        assert old.closed.wait(1)


def test_late_error_from_previous_run_cannot_switch_off_a_successful_restart():
    waiting_error = threading.Event()
    second_running = threading.Event()
    attempts = []

    def loop(stop, status):
        attempts.append(1)
        if len(attempts) == 1:
            raise RuntimeError("first camera failed")
        second_running.set()
        stop.wait(3)

    worker = CaptureWorker(loop_fn=loop)
    app = TrayApp(worker=worker)
    old_thread = None

    class PendingError(logging.Handler):
        def emit(self, record):
            if record.getMessage().startswith(ERROR_PREFIX):
                waiting_error.set()  # _on_status logs just before acquiring the tray lock.

    handler = PendingError()
    old_level = tray_module.log.level
    tray_module.log.addHandler(handler)
    tray_module.log.setLevel(logging.INFO)
    try:
        with app._lock:  # the real GUI/menu lock delays the old status delivery.
            app.set_on(True)
            assert waiting_error.wait(1)
            old_thread = next(t for t in threading.enumerate()
                              if t.name == "airtouch-capture")
            app.set_on(False)
            app.set_on(True)
            assert second_running.wait(1)
        old_thread.join(1)
        assert not old_thread.is_alive()
        assert app.on is True, "the old failure switched OFF the new active capture"
        assert worker.is_running
    finally:
        worker.stop(1)
        tray_module.log.removeHandler(handler)
        tray_module.log.setLevel(old_level)


def test_a_stuck_first_frame_cannot_fall_back_into_a_second_active_reader(monkeypatch):
    monkeypatch.setattr(camera_module, "JOIN_TIMEOUT_S", .02)
    old = BlockedCapture()
    old.calls = 1  # block even the first read.
    fallback_calls = []
    monkeypatch.setattr(camera_module, "_default_openers", lambda: [lambda _i: old,
                         lambda _i: fallback_calls.append(1)])
    camera = ThreadedCamera(first_frame_timeout=.01)
    try:
        with pytest.raises(RuntimeError):
            camera.start()
        assert old.blocked.is_set() and not old.closed.is_set()
        assert fallback_calls == [], "fallback must not open over a reader still inside the driver"
    finally:
        old.unblock.set()
        camera.release()
        assert old.closed.wait(1)


def test_a_blocked_device_does_not_prevent_a_different_camera_index(monkeypatch):
    monkeypatch.setattr(camera_module, "JOIN_TIMEOUT_S", .02)
    old, other = BlockedCapture(), BlockedCapture()
    first = ThreadedCamera(index=7, capture_factory=lambda _i: old, first_frame_timeout=.1)
    second = ThreadedCamera(index=8, capture_factory=lambda _i: other, first_frame_timeout=.1)
    old_reader = None
    try:
        first.start()
        old_reader = first._thread
        assert old.blocked.wait(1)
        first.release()
        second.start()
        assert second.is_open
    finally:
        old.unblock.set()
        other.unblock.set()
        first.release()
        second.release()
        if old_reader is not None:
            old_reader.join(1)
            assert not old_reader.is_alive()


def test_camera_thread_start_failure_releases_device_for_a_retry(monkeypatch):
    failed, healthy = BlockedCapture(), BlockedCapture()
    camera = ThreadedCamera(index=6, capture_factory=lambda _i: failed, first_frame_timeout=.1)

    def cannot_start(_thread):
        raise RuntimeError("thread creation failed")

    with monkeypatch.context() as failure:
        failure.setattr(camera_module.threading.Thread, "start", cannot_start)
        with pytest.raises(RuntimeError, match="thread creation failed"):
            camera.start()
    assert failed.closed.is_set()
    replacement = ThreadedCamera(index=6, capture_factory=lambda _i: healthy, first_frame_timeout=.1)
    try:
        replacement.start()
        assert replacement.is_open
    finally:
        healthy.unblock.set()
        replacement.release()


@pytest.mark.parametrize("interruption", [KeyboardInterrupt, SystemExit])
def test_interrupted_camera_open_closes_capture_and_keeps_device_reusable(interruption):
    class InterruptedCapture(BlockedCapture):
        def isOpened(self):
            raise interruption()

    interrupted, healthy = InterruptedCapture(), BlockedCapture()
    index = 9 if interruption is KeyboardInterrupt else 10
    with pytest.raises(interruption):
        ThreadedCamera(index=index, capture_factory=lambda _i: interrupted).start()
    assert interrupted.closed.is_set()
    replacement = ThreadedCamera(index=index, capture_factory=lambda _i: healthy, first_frame_timeout=.1)
    try:
        replacement.start()
        assert replacement.is_open
    finally:
        healthy.unblock.set()
        replacement.release()


def test_quit_invalidates_pending_error_before_waiting_for_worker_stop():
    class Worker:
        on_status = None
        is_running = False

        def start(self):
            self.is_running = True

        def stop(self, timeout=3):
            self.on_status(ERROR_PREFIX + "late shutdown error")
            self.is_running = False
            return True

    class Icon:
        HAS_NOTIFICATION = True
        stopped = False

        def __init__(self):
            self.notifications = []

        def update_menu(self):
            pass

        def notify(self, message, title):
            self.notifications.append(message)

        def stop(self):
            self.stopped = True

    worker = Worker()
    app = TrayApp(worker=worker)
    app.icon = Icon()
    app.set_on(True)
    app.quit()
    assert not app.on and app.icon.stopped
    assert app.icon.notifications == [], "quit should invalidate in-flight status before stopping"


@pytest.mark.parametrize("interruption", [KeyboardInterrupt, SystemExit])
def test_cancellation_while_waiting_first_frame_stops_unowned_reader(monkeypatch, interruption):
    class NoFrames(BlockedCapture):
        def read(self):
            self.calls += 1
            return False, None

    index = 4 if interruption is KeyboardInterrupt else 5
    empty, healthy = NoFrames(), BlockedCapture()
    camera = ThreadedCamera(index=index, capture_factory=lambda _i: empty, first_frame_timeout=.5)
    replacement = ThreadedCamera(index=index, capture_factory=lambda _i: healthy, first_frame_timeout=.1)

    def cancel_wait(_seconds):
        raise interruption()

    try:
        with monkeypatch.context() as cancellation:
            cancellation.setattr(camera_module.time, "sleep", cancel_wait)
            with pytest.raises(interruption):
                camera.start()
        assert not camera.is_open, "start never returned ownership to its caller"
        assert empty.closed.is_set(), "interrupted first-frame wait leaked its reader/capture"
        replacement.start()
        assert replacement.is_open, "a fresh instance must be able to retry this device"
    finally:
        camera.release()  # cleanup also runs against the unfixed RED implementation.
        healthy.unblock.set()
        replacement.release()


@pytest.mark.parametrize("failure", [RuntimeError, KeyboardInterrupt, SystemExit])
def test_worker_thread_start_failure_keeps_stop_and_retry_usable(monkeypatch, failure):
    started = threading.Event()

    def loop(stop, _status):
        started.set()
        stop.wait(2)

    worker = CaptureWorker(loop_fn=loop)

    def cannot_start(_thread):
        raise failure("thread creation failed")

    try:
        with monkeypatch.context() as broken_start:
            broken_start.setattr(threading.Thread, "start", cannot_start)
            with pytest.raises(failure):
                worker.start()
        assert not worker.is_running
        assert worker.stop(0) is True, "stop must not join an unstarted published thread"
        worker.start()
        assert started.wait(1)
        assert worker.stop(1) is True
    finally:
        if worker._thread is not None and worker._thread.ident is not None:
            worker.stop(1)


def test_tray_worker_start_failure_returns_to_off_and_can_retry(monkeypatch):
    class Icon:
        HAS_NOTIFICATION = True

        def __init__(self):
            self.notifications = []

        def update_menu(self):
            pass

        def notify(self, message, title):
            self.notifications.append(message)

        def stop(self):
            pass

    entered = threading.Event()

    def loop(stop, _status):
        entered.set()
        stop.wait(2)

    worker = CaptureWorker(loop_fn=loop)
    app = TrayApp(worker=worker)
    app.icon = Icon()

    def cannot_start(_thread):
        raise RuntimeError("can't start new thread")

    try:
        with monkeypatch.context() as broken_start:
            broken_start.setattr(threading.Thread, "start", cannot_start)
            app.set_on(True)
        assert not app.on and not worker.is_running
        assert worker.stop(0) is True
        assert len(app.icon.notifications) == 1
        assert any("\u4e00" <= char <= "\u9fff" for char in app.icon.notifications[0])
        app.set_on(True)
        assert app.on and entered.wait(1)
    finally:
        if worker._thread is not None and worker._thread.ident is not None:
            app.quit()
