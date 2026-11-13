"""Read-only regression probes against the isolated snapshot; no real devices."""
import threading
import time
import pytest
from airtouch import session, camera as camera_module


def wait_for(predicate, timeout=1):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.001)
    return bool(predicate())


def test_retry_after_restart_timeout_does_not_overlap_old_capture(monkeypatch):
    monkeypatch.setattr(session, "WIND_DOWN_S", .03)
    release = threading.Event()
    entered = threading.Event()
    lock = threading.Lock()
    active = [0]
    peak = [0]
    starts = []
    statuses = []

    def stuck_loop(stop, status):
        with lock:
            active[0] += 1
            peak[0] = max(peak[0], active[0])
            starts.append(len(starts) + 1)
        entered.set()
        release.wait(2)
        with lock:
            active[0] -= 1

    worker = session.CaptureWorker(loop_fn=stuck_loop, on_status=statuses.append)
    try:
        worker.start()
        assert entered.wait(1)
        worker.stop(timeout=0)
        worker.start()  # existing guard must time out against still-active first run
        assert wait_for(lambda: any(session.MSG_STILL_CLOSING in s for s in statuses))
        assert len(starts) == 1
        worker.start()  # a normal retry must still respect that first run
        wait_for(lambda: len(starts) > 1, .15)
        assert peak[0] == 1, f"{peak[0]} loops overlapped after a second ON"
    finally:
        release.set()
        worker.stop(timeout=1)
        assert wait_for(lambda: active[0] == 0)


def test_camera_open_status_error_releases_constructed_capture():
    class BrokenCapture:
        releases = 0
        def isOpened(self):
            raise OSError("driver status query failed")
        def release(self):
            self.releases += 1
    cap = BrokenCapture()
    camera = camera_module.ThreadedCamera(capture_factory=lambda index: cap)
    with pytest.raises(RuntimeError, match="摄像头"):
        camera.start()
    assert cap.releases == 1, "constructed capture leaked when isOpened raises"


def test_stop_reports_unfinished_capture_after_restart_timeout(monkeypatch):
    monkeypatch.setattr(session, "WIND_DOWN_S", .03)
    release = threading.Event()
    entered = threading.Event()
    finished = threading.Event()
    statuses = []

    def stuck_loop(stop, status):
        entered.set()
        release.wait(2)
        finished.set()

    worker = session.CaptureWorker(loop_fn=stuck_loop, on_status=statuses.append)
    try:
        worker.start()
        assert entered.wait(1)
        worker.stop(timeout=0)
        worker.start()
        assert wait_for(lambda: any(session.MSG_STILL_CLOSING in s for s in statuses))
        assert worker.stop(timeout=.01) is False
    finally:
        release.set()
        worker.stop(timeout=1)
        assert finished.wait(1)


def test_camera_failed_close_is_not_retried_during_open_failure():
    class ClosedCapture:
        releases = 0
        def isOpened(self):
            return False
        def release(self):
            self.releases += 1
            raise OSError("driver cleanup failed")
    cap = ClosedCapture()
    camera = camera_module.ThreadedCamera(capture_factory=lambda index: cap)
    with pytest.raises(RuntimeError, match="摄像头"):
        camera.start()
    assert cap.releases == 1
