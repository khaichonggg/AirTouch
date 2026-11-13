"""The background worker behind the tray switch, and its headless gesture loop.

Everything here is fake (camera, hand tracker, mouse). An autouse fixture makes
it impossible to reach the real webcam or the real mouse from these tests.
"""

from __future__ import annotations

import subprocess
import sys
import threading
import time
import types

import numpy as np
import pytest

from airtouch import session
from airtouch.config import PROJECT_ROOT, Config
from airtouch.controller import FakeBackend
from airtouch.session import CaptureWorker, run_headless
from ml.synthetic_data import HandPose, make_hand

RNG = np.random.default_rng(0)
PINCH = make_hand("pinch_index", RNG, pose=HandPose(), thumb_gap=0.0, noise=0.0)
POINT = make_hand("point", RNG, pose=HandPose(), noise=0.0)


@pytest.fixture(autouse=True)
def no_real_hardware(monkeypatch):
    def forbidden(*_args, **_kwargs):
        raise AssertionError("a test tried to use the real camera, tracker or mouse")

    monkeypatch.setitem(sys.modules, "airtouch.camera", types.SimpleNamespace(ThreadedCamera=forbidden))
    monkeypatch.setitem(sys.modules, "airtouch.hand_tracker", types.SimpleNamespace(HandTracker=forbidden))
    monkeypatch.setattr("airtouch.controller.make_backend", forbidden)


def wait_until(predicate, timeout: float = 5.0) -> bool:
    deadline = time.perf_counter() + timeout
    while time.perf_counter() < deadline:
        if predicate():
            return True
        time.sleep(0.002)
    return predicate()


# ------------------------------------------------------------------ test doubles
class FakeCamera:
    """Delivers ``frames`` new frames (None = endless), then stalls.

    ``interval`` spaces them out in real time so that a starved test thread can
    never look like a stalled camera."""

    def __init__(self, frames: int | None = None, interval: float = 0.0):
        self.frames = frames
        self.interval = interval
        self.seq = 0
        self.reads = 0
        self.released = False
        self._next_t = 0.0

    def read_if_new(self, last_seq: int):
        self.reads += 1
        now = time.perf_counter()
        if (self.frames is not None and self.seq >= self.frames) or now < self._next_t:
            return self.seq, None
        self._next_t = now + self.interval
        self.seq += 1
        frame = np.zeros((4, 6, 3), dtype=np.uint8)
        frame[:, :3] = 255                      # left half white, to detect mirroring
        return self.seq, frame

    def release(self) -> None:
        self.released = True


class FakeTracker:
    def __init__(self, hands=()):
        self.hands = list(hands)
        self.timestamps: list[int] = []
        self.first_pixels: list[int] = []
        self.closed = False

    def detect(self, frame, t_ms: int):
        self.timestamps.append(t_ms)
        self.first_pixels.append(int(frame[0, 0, 0]))
        return self.hands

    def close(self) -> None:
        self.closed = True


class Rig:
    """One run_headless on its own thread, fully faked."""

    def __init__(self, camera=None, tracker=None, cfg=None):
        self.camera = camera if camera is not None else FakeCamera()
        self.tracker = tracker if tracker is not None else FakeTracker()
        self.backend = FakeBackend()
        self.statuses: list[str] = []
        self.stop_event = threading.Event()
        self.failure: BaseException | None = None
        self.cfg = cfg if cfg is not None else Config()
        self.thread = threading.Thread(target=self._body, daemon=True)

    def _body(self):
        try:
            run_headless(self.stop_event, self.cfg, self.statuses.append,
                         camera_factory=lambda cfg: self.camera,
                         tracker_factory=lambda cfg: self.tracker,
                         backend=self.backend)
        except BaseException as e:
            self.failure = e

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *_exc):
        self.stop()

    def stop(self):
        self.stop_event.set()
        self.thread.join(5)
        assert not self.thread.is_alive()


# ------------------------------------------------------------------ run_headless
def test_frames_flow_through_to_the_mouse_and_stop_cleans_up():
    with Rig(tracker=FakeTracker([POINT])) as rig:
        assert wait_until(lambda: len(rig.tracker.timestamps) >= 10 and rig.backend.count("move") >= 1)
    assert rig.failure is None
    assert rig.camera.released and rig.tracker.closed
    assert session.MSG_READY in rig.statuses and session.MSG_CLOSED in rig.statuses


def test_halo_follows_active_control_and_is_closed_with_the_worker(monkeypatch):
    class Halo:
        def __init__(self):
            self.updates = []
            self.closed = False

        def update(self, position, pressed=False, active=True):
            self.updates.append((position, pressed, active))

        def close(self):
            self.closed = True

    halo = Halo()
    monkeypatch.setitem(sys.modules, "airtouch.cursor_halo", types.SimpleNamespace(CursorHalo=lambda: halo))
    with Rig(tracker=FakeTracker([POINT]), cfg=Config(cursor_halo=True)) as rig:
        assert wait_until(lambda: any(active for _, _, active in halo.updates))
    assert halo.closed and rig.camera.released and rig.tracker.closed
    assert any(pos is not None for pos, _, active in halo.updates if active)


def test_halo_is_hidden_when_no_hand_is_detected(monkeypatch):
    updates = []
    halo = types.SimpleNamespace(update=lambda pos, **kwargs: updates.append((pos, kwargs)), close=lambda: None)
    monkeypatch.setitem(sys.modules, "airtouch.cursor_halo", types.SimpleNamespace(CursorHalo=lambda: halo))
    with Rig(cfg=Config(cursor_halo=True)) as rig:
        assert wait_until(lambda: len(updates) >= 3)
    assert all(not kwargs["active"] for _, kwargs in updates)


def test_halo_is_closed_if_tracking_fails(monkeypatch):
    closed = []
    halo = types.SimpleNamespace(update=lambda *_a, **_k: None, close=lambda: closed.append(True))
    monkeypatch.setitem(sys.modules, "airtouch.cursor_halo", types.SimpleNamespace(CursorHalo=lambda: halo))

    class Broken(FakeTracker):
        def detect(self, *_args):
            raise RuntimeError("tracker failed")

    with Rig(tracker=Broken(), cfg=Config(cursor_halo=True)) as rig:
        assert wait_until(lambda: rig.failure is not None)
    assert closed == [True] and rig.camera.released


def test_final_performance_sample_excludes_camera_teardown(monkeypatch):
    camera = FakeCamera()
    ended = []

    class Monitor:
        def __init__(self, *_args):
            pass

        def start(self, _now):
            pass

        def frame(self, *_args, **_kwargs):
            pass

        def idle(self, _now):
            pass

        def finish(self, _now):
            ended.append(camera.released)

    monkeypatch.setattr(session, "PerformanceMonitor", Monitor)
    with Rig(camera=camera) as rig:
        assert wait_until(lambda: len(rig.tracker.timestamps) >= 3)
    assert ended == [False]
    assert camera.released


@pytest.mark.parametrize("width,height,aspect", [
    (1280, 720, 16 / 9), (1920, 1200, 1.6), (3840, 2160, 16 / 9),
])
def test_actual_camera_shape_reaches_tracker_engine_and_recognizer(monkeypatch, width, height, aspect):
    from airtouch.engine import GestureEngine

    stop = threading.Event()
    camera, tracker = FakeCamera(), FakeTracker()
    camera.frame_size = (width, height)  # the driver ignored the requested 640 x 480
    cfg = Config()
    seen = {}

    def make_tracker(actual_cfg):
        seen["tracker"] = (actual_cfg.frame_width, actual_cfg.frame_height)
        return tracker

    class RecordingEngine(GestureEngine):
        def __init__(self, actual_cfg, backend, recognizer):
            seen["engine"] = actual_cfg.aspect
            seen["recognizer"] = recognizer.aspect
            super().__init__(actual_cfg, backend, recognizer)

        def process_hands(self, hands, t):
            stop.set()
            return super().process_hands(hands, t)

    monkeypatch.setattr("airtouch.engine.GestureEngine", RecordingEngine)
    run_headless(stop, cfg, camera_factory=lambda _cfg: camera,
                 tracker_factory=make_tracker, backend=FakeBackend())
    assert seen["tracker"] == (width, height)
    assert seen["engine"] == pytest.approx(aspect)
    assert seen["recognizer"] == pytest.approx(aspect)
    assert (cfg.frame_width, cfg.frame_height) == (640, 480)
    assert camera.released and tracker.closed


def test_backend_sets_up_dpi_before_opening_camera_and_tracker(monkeypatch):
    order = []
    stop = threading.Event()
    camera, tracker = FakeCamera(), FakeTracker()

    def make_backend():
        order.append("backend")
        return FakeBackend()

    def make_camera(_cfg):
        order.append("camera")
        return camera

    def make_tracker(_cfg):
        order.append("tracker")
        stop.set()
        return tracker

    monkeypatch.setattr("airtouch.controller.make_backend", make_backend)
    run_headless(stop, Config(), camera_factory=make_camera, tracker_factory=make_tracker)
    assert order == ["backend", "camera", "tracker"]
    assert camera.released and tracker.closed


def test_timestamps_are_strictly_increasing():
    with Rig() as rig:      # an endless fake camera delivers many frames per millisecond
        assert wait_until(lambda: len(rig.tracker.timestamps) >= 50)
    stamps = rig.tracker.timestamps
    assert all(isinstance(t, int) for t in stamps)
    assert all(b > a for a, b in zip(stamps, stamps[1:]))


@pytest.mark.parametrize("mirror, expected", [(True, 0), (False, 255)])
def test_frame_is_flipped_only_when_mirror_is_on(mirror, expected):
    with Rig(camera=FakeCamera(frames=3), cfg=Config(mirror=mirror)) as rig:
        assert wait_until(lambda: len(rig.tracker.first_pixels) == 3)
    assert rig.tracker.first_pixels == [expected] * 3


def test_stop_releases_a_held_button():
    with Rig(tracker=FakeTracker([PINCH])) as rig:
        assert wait_until(lambda: "left" in rig.backend.down)
    assert rig.backend.down == set()
    assert rig.backend.count("up", "left") == 1
    assert rig.camera.released and rig.tracker.closed


def test_stalled_camera_releases_a_held_button_while_still_running():
    with Rig(camera=FakeCamera(frames=5), tracker=FakeTracker([PINCH])) as rig:
        assert wait_until(lambda: "left" in rig.backend.down)
        assert wait_until(lambda: rig.camera.seq == 5)
        started = time.perf_counter()
        assert wait_until(lambda: not rig.backend.down)
        assert time.perf_counter() - started >= session.STALL_S * 0.8
        assert rig.thread.is_alive()
        assert not rig.camera.released


def test_slow_camera_does_not_drop_a_held_button():
    with Rig(camera=FakeCamera(interval=0.03), tracker=FakeTracker([PINCH])) as rig:
        assert wait_until(lambda: "left" in rig.backend.down)
        time.sleep(0.6)
        assert "left" in rig.backend.down
        assert rig.backend.count("up", "left") == 0


def test_waiting_for_frames_never_busy_spins():
    with Rig(camera=FakeCamera(frames=0)) as rig:
        time.sleep(0.3)
    assert 10 < rig.camera.reads < 300      # 2 ms sleeps: at most ~150; a spin would be many thousands


class FakeClock:
    """Stands in for the ``time`` module, so seconds of waiting take no time."""

    def __init__(self):
        self.now = 1000.0

    def perf_counter(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.now += seconds


class ScriptedCamera(FakeCamera):
    def __init__(self, on_read):
        super().__init__(frames=0)
        self.on_read = on_read

    def read_if_new(self, last_seq):
        self.on_read()
        return last_seq, None


def run_with_clock(monkeypatch, on_read, tracker=None):
    clock = FakeClock()
    monkeypatch.setattr(session, "time", clock)
    camera = ScriptedCamera(lambda: on_read(clock))
    tracker = tracker if tracker is not None else FakeTracker()
    run_headless(threading.Event(), Config(), None, lambda cfg: camera, lambda cfg: tracker, FakeBackend())
    return camera, tracker


def test_camera_that_never_delivers_a_frame_is_an_error(monkeypatch):
    camera = tracker = None
    clock = FakeClock()
    monkeypatch.setattr(session, "time", clock)
    camera, tracker = FakeCamera(frames=0), FakeTracker()
    with pytest.raises(RuntimeError, match="摄像头"):
        run_headless(threading.Event(), Config(), None, lambda cfg: camera, lambda cfg: tracker, FakeBackend())
    assert clock.now - 1000.0 == pytest.approx(session.CAMERA_LOST_S, abs=0.01)
    assert camera.released and tracker.closed


def test_computer_going_to_sleep_is_not_a_lost_camera(monkeypatch):
    reads = {"n": 0}

    def on_read(clock):
        reads["n"] += 1
        if reads["n"] == 100:
            clock.now += 3600.0         # lid closed for an hour
        if reads["n"] == 200:
            raise KeyboardInterrupt     # got well past the jump without an error

    with pytest.raises(KeyboardInterrupt):
        run_with_clock(monkeypatch, on_read)


def test_button_is_released_at_the_camera_stall_limit(monkeypatch):
    clock = FakeClock()
    monkeypatch.setattr(session, "time", clock)
    backend = FakeBackend()
    stop = threading.Event()
    released_after = []

    class FewFramesThenNothing(FakeCamera):
        def read_if_new(self, last_seq):
            if self.seq < 5:                # all at clock time 0; the gesture takes 3 frames to settle
                self.seq += 1
                return self.seq, np.zeros((4, 6, 3), dtype=np.uint8)
            if "left" not in backend.down:
                released_after.append(clock.now - 1000.0)
                stop.set()
            return self.seq, None

    run_headless(stop, Config(), None, lambda cfg: FewFramesThenNothing(), lambda cfg: FakeTracker([PINCH]), backend)
    assert backend.count("down", "left") == 1
    expected = session.STALL_S             # grace starts at the last actual frame
    assert released_after[0] == pytest.approx(expected, abs=0.02)


def test_camera_failure_gives_a_plain_message_without_building_the_tracker():
    trackers = []

    def broken(cfg):
        raise OSError("MSMF: can't grab frame, error -1072873821")

    with pytest.raises(RuntimeError) as err:
        run_headless(threading.Event(), Config(), None, broken, lambda cfg: trackers.append(cfg), FakeBackend())
    assert str(err.value) == session.MSG_NO_CAMERA
    assert not trackers


def test_camera_error_already_in_chinese_is_kept():
    def broken(cfg):
        raise RuntimeError("摄像头被占用了")

    with pytest.raises(RuntimeError, match="^摄像头被占用了$"):
        run_headless(threading.Event(), Config(), None, broken, lambda cfg: FakeTracker(), FakeBackend())


def test_broken_model_gives_a_plain_message_and_closes_the_camera():
    camera = FakeCamera()

    def broken(cfg):
        raise ValueError("model asset buffer is corrupt")

    with pytest.raises(RuntimeError) as err:
        run_headless(threading.Event(), Config(), None, lambda cfg: camera, broken, FakeBackend())
    assert str(err.value) == session.MSG_NO_TRACKER
    assert camera.released


def test_stop_while_tracker_starts_closes_the_camera_and_tracker():
    stop = threading.Event()
    camera = FakeCamera()
    tracker = FakeTracker()

    def slow_tracker(cfg):
        stop.set()                      # the user switched off while the model was loading
        return tracker

    run_headless(stop, Config(), None, lambda cfg: camera, slow_tracker, FakeBackend())
    assert camera.released and tracker.closed


def test_exception_in_the_loop_still_cleans_up_everything():
    class Exploding(FakeTracker):
        def detect(self, frame, t_ms):
            if len(self.timestamps) >= 3:
                raise ValueError("boom")
            return super().detect(frame, t_ms)

    rig = Rig(tracker=Exploding([PINCH]))
    rig.thread.start()
    rig.thread.join(5)
    assert isinstance(rig.failure, ValueError)
    assert rig.backend.count("down", "left") == 1 and rig.backend.down == set()
    assert rig.camera.released and rig.tracker.closed


def test_a_failing_cleanup_step_does_not_skip_the_others():
    class BadClose(FakeTracker):
        def close(self):
            raise OSError("close failed")

    with Rig(tracker=BadClose([PINCH])) as rig:
        assert wait_until(lambda: "left" in rig.backend.down)
    assert rig.failure is None
    assert rig.backend.down == set()
    assert rig.camera.released


def test_broken_status_callback_does_not_stop_the_loop():
    def bad_status(_message):
        raise ValueError("status display broke")

    stop = threading.Event()
    camera, tracker = FakeCamera(), FakeTracker()
    thread = threading.Thread(target=run_headless, daemon=True, args=(
        stop, Config(), bad_status, lambda cfg: camera, lambda cfg: tracker, FakeBackend()))
    thread.start()
    assert wait_until(lambda: len(tracker.timestamps) >= 3)
    stop.set()
    thread.join(5)
    assert not thread.is_alive() and camera.released


def test_real_camera_and_tracker_are_built_with_the_agreed_arguments(monkeypatch):
    """The default wiring, with stand-ins for airtouch.camera / airtouch.hand_tracker."""
    built = {}
    camera, tracker = FakeCamera(frames=2), FakeTracker()

    class Camera:
        def __init__(self, *args):
            built["camera"] = args

        def start(self):
            return camera

    def make_tracker(cfg):
        built["tracker"] = cfg
        return tracker

    backend = FakeBackend()
    monkeypatch.setitem(sys.modules, "airtouch.camera", types.SimpleNamespace(ThreadedCamera=Camera))
    monkeypatch.setitem(sys.modules, "airtouch.hand_tracker", types.SimpleNamespace(HandTracker=make_tracker))
    monkeypatch.setattr("airtouch.controller.make_backend", lambda: backend)

    cfg = Config(camera_index=2, frame_width=320, frame_height=240, camera_fps=15)
    stop = threading.Event()
    thread = threading.Thread(target=run_headless, args=(stop, cfg), daemon=True)
    thread.start()
    assert wait_until(lambda: len(tracker.timestamps) == 2)
    stop.set()
    thread.join(5)
    assert built == {"camera": (2, 320, 240, 15), "tracker": cfg}
    assert camera.released and tracker.closed


def test_config_is_read_from_file_when_not_given(monkeypatch):
    seen = []
    monkeypatch.setattr(Config, "load", classmethod(lambda cls, path=None: seen.append(1) or cls()))
    stop = threading.Event()
    stop.set()
    run_headless(stop, None, None, lambda cfg: FakeCamera(), lambda cfg: FakeTracker(), FakeBackend())
    assert seen == [1]


# ----------------------------------------------------------------- CaptureWorker
def looping(stop_event, on_status):
    on_status("running")
    stop_event.wait(10)


def test_start_runs_the_loop_and_stop_joins_it():
    worker = CaptureWorker(loop_fn=looping)
    assert not worker.is_running
    worker.start()
    assert worker.is_running
    started = time.perf_counter()
    assert worker.stop() is True
    assert time.perf_counter() - started < 1.0
    assert not worker.is_running


def test_the_loop_sees_the_stop_event():
    seen = []

    def loop(stop_event, _on_status):
        assert not stop_event.is_set()
        stop_event.wait(10)
        seen.append(stop_event.is_set())

    worker = CaptureWorker(loop_fn=loop)
    worker.start()
    worker.stop()
    assert seen == [True]


def test_the_thread_is_a_daemon():
    worker = CaptureWorker(loop_fn=looping)
    worker.start()
    try:
        assert worker._thread.daemon
    finally:
        worker.stop()


def test_restart_after_stop_gets_a_fresh_stop_event():
    events = []

    def loop(stop_event, _on_status):
        events.append(stop_event)
        stop_event.wait(10)

    worker = CaptureWorker(loop_fn=loop)
    for _ in range(3):
        worker.start()
        assert worker.is_running
        assert wait_until(lambda: bool(events))
        worker.stop()
        assert not worker.is_running
    assert len(events) == 3 and len({id(e) for e in events}) == 3


def test_double_start_runs_only_one_loop():
    calls = []

    def loop(stop_event, _on_status):
        calls.append(1)
        stop_event.wait(10)

    worker = CaptureWorker(loop_fn=loop)
    worker.start()
    worker.start()
    worker.start()
    assert wait_until(lambda: calls == [1])
    time.sleep(0.05)
    worker.stop()
    assert calls == [1]


def test_stop_is_safe_when_never_started_or_already_stopped():
    worker = CaptureWorker(loop_fn=looping)
    assert worker.stop() is True
    worker.start()
    assert worker.stop() is True
    assert worker.stop() is True
    assert not worker.is_running


def test_stop_without_waiting_still_ends_the_loop():
    worker = CaptureWorker(loop_fn=looping)
    worker.start()
    worker.stop(timeout=0)
    assert wait_until(lambda: not worker.is_running)


def test_exception_reports_error_and_leaves_nothing_running():
    statuses = []
    running_when_told = []

    def on_status(message):
        statuses.append(message)
        running_when_told.append(worker.is_running)

    def loop(_stop_event, _on_status):
        raise ValueError("internal detail the user can't read")

    worker = CaptureWorker(loop_fn=loop, on_status=on_status)
    worker.start()
    assert wait_until(lambda: bool(statuses))
    assert statuses == [session.ERROR_PREFIX + session.MSG_UNKNOWN]
    assert running_when_told == [False]         # no window where the error is known but it still "runs"
    assert not worker.is_running
    assert wait_until(lambda: not any(t.name == "airtouch-capture" for t in threading.enumerate()))


def test_error_written_in_chinese_reaches_the_user_unchanged():
    statuses = []

    def loop(_stop_event, _on_status):
        raise RuntimeError("打不开摄像头")

    worker = CaptureWorker(loop_fn=loop, on_status=statuses.append)
    worker.start()
    assert wait_until(lambda: bool(statuses))
    assert statuses == ["error: 打不开摄像头"]


def test_worker_can_start_again_after_an_exception():
    runs = []

    def loop(stop_event, _on_status):
        runs.append(1)
        if len(runs) == 1:
            raise ValueError("first run fails")
        stop_event.wait(10)

    worker = CaptureWorker(loop_fn=loop)
    worker.start()
    assert wait_until(lambda: not worker.is_running)
    worker.start()
    assert wait_until(lambda: len(runs) == 2)
    assert worker.is_running
    worker.stop()


def test_loop_ending_by_itself_is_reported_and_not_running():
    statuses = []
    running_when_told = []

    def on_status(message):
        statuses.append(message)
        running_when_told.append(worker.is_running)

    worker = CaptureWorker(loop_fn=lambda _stop, _status: None, on_status=on_status)
    worker.start()
    assert wait_until(lambda: bool(statuses))
    assert statuses == [session.MSG_ENDED] and running_when_told == [False]


def test_failure_after_being_told_to_stop_is_not_reported():
    statuses = []

    def loop(stop_event, _on_status):
        stop_event.wait(10)
        raise ValueError("camera driver grumbled while closing")

    worker = CaptureWorker(loop_fn=loop, on_status=statuses.append)
    worker.start()
    worker.stop()
    assert statuses == []
    assert not worker.is_running


def test_quick_restart_waits_for_the_previous_loop_to_finish():
    active, most_active, starts = [0], [0], []
    lock = threading.Lock()

    def loop(stop_event, _on_status):
        with lock:
            active[0] += 1
            most_active[0] = max(most_active[0], active[0])
            starts.append(1)
        stop_event.wait(10)
        time.sleep(0.15)                # closing the camera takes a moment
        with lock:
            active[0] -= 1

    worker = CaptureWorker(loop_fn=loop)
    worker.start()
    assert wait_until(lambda: len(starts) == 1)
    worker.stop(timeout=0)
    worker.start()                      # ON again before the old loop has finished
    assert wait_until(lambda: len(starts) == 2)
    worker.stop()
    assert most_active[0] == 1


def test_off_on_off_before_the_old_loop_finished_ends_with_nothing_running():
    starts = []

    def loop(stop_event, _on_status):
        starts.append(1)
        stop_event.wait(10)
        time.sleep(0.1)

    worker = CaptureWorker(loop_fn=loop)
    worker.start()
    assert wait_until(lambda: len(starts) == 1)
    worker.stop(timeout=0)
    worker.start()
    worker.stop(timeout=0)
    assert wait_until(lambda: not worker.is_running)
    time.sleep(0.15)
    assert starts == [1]                # the second run never needed to open anything


def test_restart_gives_up_if_the_previous_loop_is_stuck(monkeypatch):
    monkeypatch.setattr(session, "WIND_DOWN_S", 0.02)
    statuses = []
    release = threading.Event()

    def loop(stop_event, _on_status):
        release.wait(10)                # a camera driver that hangs and ignores the stop

    worker = CaptureWorker(loop_fn=loop, on_status=statuses.append)
    worker.start()
    worker.stop(timeout=0)
    worker.start()
    assert wait_until(lambda: bool(statuses))
    assert statuses == [session.ERROR_PREFIX + session.MSG_STILL_CLOSING]
    assert not worker.is_running
    release.set()


def test_a_failing_status_callback_cannot_break_the_worker():
    def bad_status(_message):
        raise ValueError("no")

    worker = CaptureWorker(loop_fn=looping, on_status=bad_status)
    worker.start()
    assert worker.is_running
    assert worker.stop() is True


def test_stop_from_inside_a_status_callback_does_not_deadlock():
    statuses = []

    def on_status(message):
        statuses.append(message)
        if message == "running":
            statuses.append(worker.stop())      # can't wait for its own thread

    worker = CaptureWorker(loop_fn=looping, on_status=on_status)
    worker.start()
    assert wait_until(lambda: not worker.is_running)
    assert statuses == ["running", False]


def test_default_loop_is_run_headless_with_this_workers_settings(monkeypatch):
    calls = []

    def fake_run_headless(stop_event, cfg, on_status):
        calls.append((cfg, on_status))
        on_status("hello")

    monkeypatch.setattr(session, "run_headless", fake_run_headless)
    cfg, statuses = Config(), []
    worker = CaptureWorker(cfg=cfg, on_status=statuses.append)
    worker.start()
    assert wait_until(lambda: not worker.is_running)
    assert calls[0][0] is cfg
    assert statuses[0] == "hello"


def test_default_worker_uses_real_loop_end_to_end_with_fakes(monkeypatch):
    """CaptureWorker + run_headless together, the way the tray uses them."""
    camera, tracker, backend = FakeCamera(), FakeTracker([PINCH]), FakeBackend()
    real = session.run_headless

    def wired(stop_event, cfg, on_status):
        real(stop_event, cfg, on_status, lambda c: camera, lambda c: tracker, backend)

    monkeypatch.setattr(session, "run_headless", wired)
    statuses = []
    worker = CaptureWorker(cfg=Config(), on_status=statuses.append)
    worker.start()
    assert wait_until(lambda: "left" in backend.down)
    assert worker.stop() is True
    assert backend.down == set()
    assert camera.released and tracker.closed
    assert statuses[-1] == session.MSG_CLOSED


@pytest.mark.parametrize("error, expected", [
    (ValueError("bad"), session.MSG_UNKNOWN),
    (RuntimeError(""), session.MSG_UNKNOWN),
    (RuntimeError("摄像头坏了"), "摄像头坏了"),
])
def test_plain_message(error, expected):
    assert session.plain_message(error) == expected


def test_all_error_messages_are_plain_chinese():
    for name in dir(session):
        if name.startswith("MSG_"):
            text = getattr(session, name)
            assert any("一" <= ch <= "鿿" for ch in text), name
            for jargon in ("FPS", "landmark", "debounce", "thread", "exception"):
                assert jargon.lower() not in text.lower(), name


def test_parked_tray_does_not_load_the_heavy_libraries():
    """A parked icon must stay light: OpenCV & co. load only when switched on."""
    code = ("import sys, airtouch.session, airtouch.tray;"
            "heavy = {'cv2', 'mediapipe', 'numpy', 'sklearn', 'scipy'} & set(sys.modules);"
            "sys.exit(repr(sorted(heavy)) if heavy else 0)")
    done = subprocess.run([sys.executable, "-c", code], cwd=PROJECT_ROOT, capture_output=True, text=True, timeout=60)
    assert done.returncode == 0, done.stderr
