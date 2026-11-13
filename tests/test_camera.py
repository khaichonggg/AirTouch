"""ThreadedCamera, driven by a fake capture: no webcam is ever opened."""

from __future__ import annotations

import math
import threading
import time

import cv2
import numpy as np
import pytest

from airtouch import camera as camera_module
from airtouch.camera import OPEN_FAILED_MESSAGE, RateMeter, ThreadedCamera


@pytest.fixture(autouse=True)
def no_real_camera(monkeypatch):
    """A slip in a test must fail, not switch on the person's webcam."""

    def forbidden(*args, **kwargs):
        raise AssertionError("a test tried to open the real camera")

    monkeypatch.setattr(cv2, "VideoCapture", forbidden)


class CountingFrame(np.ndarray):
    """A frame that counts how often it gets copied."""

    copies = 0

    def copy(self, *args, **kwargs):
        CountingFrame.copies += 1
        return super().copy(*args, **kwargs)


def make_frame(counter: int, width: int = 64, height: int = 48) -> np.ndarray:
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    frame[0, 0] = (counter & 0xFF, (counter >> 8) & 0xFF, (counter >> 16) & 0xFF)
    return frame


def counter_of(frame: np.ndarray) -> int:
    b, g, r = (int(v) for v in frame[0, 0])
    return b | g << 8 | r << 16


class FakeCapture:
    """Stands in for cv2.VideoCapture. Frames carry the number of the read that made them.

    delay:   seconds a read takes
    fail:    fail(i) -> True makes read number i fail; may also raise
    good:    after this many good frames, every read fails (the camera goes silent)
    """

    def __init__(self, opened=True, delay=0.001, fail=None, good=None, width=64, height=48,
                 frame_type=None):
        self.opened = opened
        self.delay = delay
        self.fail = fail
        self.good = good
        self.size = (width, height)
        self.frame_type = frame_type
        self.reads = 0
        self.good_reads = 0
        self.releases = 0
        self.props = {}

    def isOpened(self):
        return self.opened

    def set(self, prop, value):
        self.props[prop] = value
        return True

    def read(self):
        self.reads += 1
        if self.delay:
            time.sleep(self.delay)
        if self.good is not None and self.good_reads >= self.good:
            return False, None
        if self.fail is not None and self.fail(self.reads):
            return False, None
        self.good_reads += 1
        frame = make_frame(self.reads, *self.size)
        if self.frame_type is not None:
            frame = frame.view(self.frame_type)
        return True, frame

    def release(self):
        self.releases += 1
        self.opened = False


def wait_until(condition, timeout=3.0) -> bool:
    deadline = time.perf_counter() + timeout
    while time.perf_counter() < deadline:
        if condition():
            return True
        time.sleep(0.001)
    return condition()


def camera_with(capture, **kwargs) -> ThreadedCamera:
    return ThreadedCamera(capture_factory=lambda index: capture, **kwargs)


# -- opening -------------------------------------------------------------------
def test_start_raises_plain_message_when_capture_cannot_open():
    capture = FakeCapture(opened=False)
    with pytest.raises(RuntimeError) as err:
        camera_with(capture).start()
    assert str(err.value) == OPEN_FAILED_MESSAGE
    assert "摄像头" in str(err.value) and "Zoom" in str(err.value)
    assert capture.releases == 1


def test_start_raises_when_camera_opens_but_never_delivers():
    # What a camera in use by another program often does on Windows.
    capture = FakeCapture(fail=lambda i: True)
    camera = camera_with(capture, first_frame_timeout=0.05)
    with pytest.raises(RuntimeError):
        camera.start()
    assert capture.releases == 1
    assert not camera.is_open


def test_start_turns_a_factory_exception_into_the_plain_message():
    def broken(index):
        raise cv2.error("driver exploded")

    with pytest.raises(RuntimeError) as err:
        ThreadedCamera(capture_factory=broken).start()
    assert str(err.value) == OPEN_FAILED_MESSAGE


def test_settings_are_applied_and_index_is_passed_on():
    capture = FakeCapture()
    indexes = []

    def factory(index):
        indexes.append(index)
        return capture

    with ThreadedCamera(3, width=320, height=240, fps=25, capture_factory=factory):
        pass
    assert indexes == [3]
    assert capture.props[cv2.CAP_PROP_BUFFERSIZE] == 1
    assert capture.props[cv2.CAP_PROP_FRAME_WIDTH] == 320
    assert capture.props[cv2.CAP_PROP_FRAME_HEIGHT] == 240
    assert capture.props[cv2.CAP_PROP_FPS] == 25


def test_a_capture_without_settings_support_still_works():
    capture = FakeCapture()
    capture.set = None      # calling it raises TypeError
    with camera_with(capture) as camera:
        assert camera.is_open


def test_start_twice_opens_once():
    capture = FakeCapture()
    opened = []

    def factory(index):
        opened.append(index)
        return capture

    camera = ThreadedCamera(capture_factory=factory)
    assert camera.start() is camera
    assert camera.start() is camera
    camera.release()
    assert len(opened) == 1


def test_frame_size_reports_what_the_camera_really_delivers():
    with camera_with(FakeCapture(width=128, height=72), width=640, height=480) as camera:
        assert camera.frame_size == (128, 72)


class FakeVideoCaptureClass:
    """Replaces cv2.VideoCapture itself, to test how real cameras get opened."""

    def __init__(self, dshow, default):
        self.captures = {cv2.CAP_DSHOW: dshow, None: default}
        self.calls = []

    def __call__(self, index, backend=None):
        self.calls.append((index, backend))
        return self.captures[backend]


def test_windows_opens_directshow_first(monkeypatch):
    fake = FakeVideoCaptureClass(FakeCapture(), FakeCapture())
    monkeypatch.setattr(cv2, "VideoCapture", fake)
    monkeypatch.setattr(camera_module.sys, "platform", "win32")
    with ThreadedCamera(1):
        pass
    assert fake.calls == [(1, cv2.CAP_DSHOW)]


def test_windows_falls_back_when_directshow_will_not_open(monkeypatch):
    fake = FakeVideoCaptureClass(FakeCapture(opened=False), FakeCapture())
    monkeypatch.setattr(cv2, "VideoCapture", fake)
    monkeypatch.setattr(camera_module.sys, "platform", "win32")
    with ThreadedCamera(0) as camera:
        assert camera.is_open
    assert fake.calls == [(0, cv2.CAP_DSHOW), (0, None)]


def test_windows_falls_back_when_directshow_opens_but_stays_dark(monkeypatch):
    dark = FakeCapture(fail=lambda i: True)
    fake = FakeVideoCaptureClass(dark, FakeCapture())
    monkeypatch.setattr(cv2, "VideoCapture", fake)
    monkeypatch.setattr(camera_module.sys, "platform", "win32")
    with ThreadedCamera(0, first_frame_timeout=0.05) as camera:
        seq, frame = camera.read_latest()
        assert frame is not None and seq >= 1
    assert dark.releases == 1


def test_no_camera_at_all_gives_the_plain_message(monkeypatch):
    fake = FakeVideoCaptureClass(FakeCapture(opened=False), FakeCapture(opened=False))
    monkeypatch.setattr(cv2, "VideoCapture", fake)
    monkeypatch.setattr(camera_module.sys, "platform", "win32")
    with pytest.raises(RuntimeError) as err:
        ThreadedCamera(0).start()
    assert str(err.value) == OPEN_FAILED_MESSAGE


# -- reading -------------------------------------------------------------------
def test_read_if_new_returns_none_without_copying_when_nothing_is_new():
    capture = FakeCapture(good=1, frame_type=CountingFrame)
    with camera_with(capture) as camera:
        seq, frame = camera.read_if_new(0)
        assert seq == 1 and frame is not None
        CountingFrame.copies = 0
        for _ in range(2000):
            again, nothing = camera.read_if_new(seq)
            assert again == seq and nothing is None
        assert CountingFrame.copies == 0


def test_read_if_new_copies_exactly_once_for_a_new_frame():
    capture = FakeCapture(good=1, frame_type=CountingFrame)
    with camera_with(capture) as camera:
        CountingFrame.copies = 0
        assert camera.read_if_new(0)[1] is not None
        assert CountingFrame.copies == 1


def test_read_if_new_before_any_frame_returns_none():
    camera = ThreadedCamera(capture_factory=lambda index: FakeCapture())
    assert camera.read_if_new(-1) == (0, None)
    assert camera.read_latest() == (0, None)


def test_new_frames_arrive_and_seq_strictly_increases():
    with camera_with(FakeCapture()) as camera:
        seen = []
        last = 0
        deadline = time.perf_counter() + 3.0
        while len(seen) < 20 and time.perf_counter() < deadline:
            seq, frame = camera.read_if_new(last)
            if frame is not None:
                assert seq > last
                assert counter_of(frame) == seq     # the frame and its number belong together
                seen.append(seq)
                last = seq
    assert len(seen) == 20
    assert seen == sorted(set(seen))


def test_only_the_newest_frame_is_kept_for_a_slow_reader():
    with camera_with(FakeCapture(delay=0.001)) as camera:
        first, _ = camera.read_if_new(0)
        assert wait_until(lambda: camera.read_latest()[0] > first + 5)
        seq, frame = camera.read_if_new(first)
        assert seq > first + 5                  # it skipped ahead rather than queueing
        assert counter_of(frame) == seq


def test_read_if_new_returns_a_private_copy():
    with camera_with(FakeCapture(good=1)) as camera:
        seq, frame = camera.read_if_new(0)
        original = counter_of(frame)
        frame[:] = 77                           # the app draws its overlay on this
        _, latest = camera.read_latest()
        assert counter_of(latest) == original
        assert not np.shares_memory(frame, latest)


def test_read_latest_returns_a_copy_even_of_a_seen_frame():
    with camera_with(FakeCapture(good=1)) as camera:
        seq, frame = camera.read_if_new(0)
        seq2, one = camera.read_latest()
        seq3, two = camera.read_latest()
        assert seq == seq2 == seq3
        assert one is not two and not np.shares_memory(one, two)
        np.testing.assert_array_equal(one, frame)


# -- trouble -------------------------------------------------------------------
def test_survives_intermittent_read_failures():
    capture = FakeCapture(delay=0.0005, fail=lambda i: i % 3 != 0)     # two of three reads fail
    with camera_with(capture) as camera:
        assert camera.is_open
        assert wait_until(lambda: camera.read_latest()[0] >= 10)
        assert camera.is_open


def test_a_read_that_raises_does_not_kill_the_thread():
    def fail(i):
        if i % 2:
            raise cv2.error("USB hiccup")
        return False

    with camera_with(FakeCapture(delay=0.0005, fail=fail)) as camera:
        assert wait_until(lambda: camera.read_latest()[0] >= 5)
        assert camera.is_open


def test_failed_reads_do_not_spin():
    capture = FakeCapture(delay=0, good=1)      # after one frame: instant failures forever
    with camera_with(capture):
        assert wait_until(lambda: capture.reads >= 2)
        reads_before = capture.reads
        time.sleep(0.3)
        reads = capture.reads - reads_before
    # A spinning loop would do tens of thousands of reads in 0.3 s; READ_RETRY_S allows ~30.
    assert 0 < reads < 60


def test_release_is_idempotent_and_stops_the_thread():
    capture = FakeCapture()
    camera = camera_with(capture).start()
    thread = camera._thread
    assert thread.is_alive() and camera.is_open
    camera.release()
    assert not thread.is_alive()
    assert not camera.is_open
    assert capture.releases == 1
    reads = capture.reads
    time.sleep(0.02)
    assert capture.reads == reads
    camera.release()
    camera.release()
    assert capture.releases == 1


def test_release_before_start_is_harmless():
    ThreadedCamera(capture_factory=lambda index: FakeCapture()).release()


def test_no_frames_after_release():
    camera = camera_with(FakeCapture()).start()
    seq, _ = camera.read_latest()
    camera.release()
    assert camera.read_if_new(0)[1] is None
    assert camera.read_latest()[1] is None
    assert camera.read_latest()[0] >= seq


def test_seq_keeps_increasing_across_a_restart():
    captures = [FakeCapture(good=3), FakeCapture(good=3)]
    camera = ThreadedCamera(capture_factory=lambda index: captures.pop(0))
    camera.start()
    assert wait_until(lambda: camera.read_latest()[0] == 3)
    camera.release()
    camera.start()
    assert wait_until(lambda: camera.read_latest()[0] == 6)
    camera.release()


def test_context_manager_releases_even_after_an_error():
    capture = FakeCapture()
    with pytest.raises(ValueError):
        with camera_with(capture) as camera:
            assert camera.is_open
            raise ValueError("boom")
    assert capture.releases == 1
    assert not camera.is_open


class StuckCapture(FakeCapture):
    """A driver whose read() hangs until the test lets it go."""

    def __init__(self):
        super().__init__()
        self.unstick = threading.Event()
        self.stuck = threading.Event()

    def read(self):
        if self.reads:      # first read works, so start() succeeds
            self.stuck.set()
            self.unstick.wait(5.0)
        return super().read()


def test_release_does_not_hang_on_a_stuck_driver(monkeypatch):
    monkeypatch.setattr(camera_module, "JOIN_TIMEOUT_S", 0.05)
    capture = StuckCapture()
    camera = camera_with(capture).start()
    try:
        assert capture.stuck.wait(3.0)
        thread = camera._thread
        t0 = time.perf_counter()
        camera.release()
        assert time.perf_counter() - t0 < 1.0
        # Closing a capture underneath a read in progress can crash the process.
        assert capture.releases == 0
        assert not camera.is_open
    finally:
        capture.unstick.set()
    thread.join(3.0)
    assert not thread.is_alive()
    assert camera.read_latest()[1] is None      # the late frame is thrown away
    assert capture.releases == 1              # the reader closes it once read() finally returns


# -- speed figures ---------------------------------------------------------------
def test_status_before_any_frame():
    camera = ThreadedCamera(capture_factory=lambda index: FakeCapture())
    assert camera.seconds_since_last_frame == math.inf
    assert camera.measured_fps == 0.0
    assert camera.frame_size is None
    assert not camera.is_open


def test_measured_fps_reflects_the_capture_rate():
    with camera_with(FakeCapture(delay=0.01)) as camera:
        assert wait_until(lambda: camera.read_latest()[0] >= 30)
        # Nominally 100 per second minus overhead. sleep() may overshoot on a busy
        # machine, so only the order of magnitude is checked.
        assert 10 < camera.measured_fps < 110


def test_a_stalled_camera_reads_as_stalled():
    with camera_with(FakeCapture(delay=0.002, good=20)) as camera:
        assert wait_until(lambda: camera.read_latest()[0] == 20)
        fresh = camera.seconds_since_last_frame
        time.sleep(0.3)
        assert camera.seconds_since_last_frame >= fresh + 0.25
        assert camera.measured_fps < 4      # 1 / 0.3 s, not the 100+ it was running at


def test_rate_meter_averages_gaps():
    meter = RateMeter(alpha=0.5)
    assert meter.tick(0.0) == 0.0           # one event alone says nothing about a rate
    assert meter.tick(0.1) == pytest.approx(10.0)
    assert meter.tick(0.2) == pytest.approx(10.0)
    # gap average 0.5 * 0.3 + 0.5 * 0.1 = 0.2 s, so 5 per second (averaging the rates would give 6.67)
    assert meter.tick(0.5) == pytest.approx(5.0)
