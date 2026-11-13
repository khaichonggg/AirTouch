"""Runs the gesture loop on a background thread - the tray switch's ON/OFF.

OFF really means off: the loop has ended and the camera and hand tracker are
closed, so the switch saves power. OpenCV, MediaPipe and the mouse backend are
imported only when the loop starts, so a parked tray icon stays light.

Status messages go to ``on_status`` in plain Chinese. A failure arrives as
``"error: <message>"``; the loop has already ended by then.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import replace
from typing import Callable

from .config import Config
from .performance import PerformanceMonitor

log = logging.getLogger("airtouch.session")

StatusFn = Callable[[str], None]
LoopFn = Callable[[threading.Event, StatusFn], None]

ERROR_PREFIX = "error: "

IDLE_SLEEP_S = 0.002        # waiting for the next frame; never busy-spin
STALL_S = 0.3               # no new frame for this long: tell the engine there is no hand
CAMERA_LOST_S = 10.0        # no new frame for this long: the camera is gone, turn off
ASLEEP_GAP_S = 2.0          # the loop was frozen this long: the computer slept, not the camera
WIND_DOWN_S = 10.0          # how long a restart waits for the previous loop to finish

MSG_PREPARING = "正在准备，第一次开启要等几秒..."
MSG_READY = "已开启，可以用手势控制鼠标了。"
MSG_CLOSED = "已关闭，摄像头也关掉了。"
MSG_ENDED = "手势控制已经结束。"
MSG_NO_CAMERA = "打不开摄像头。请看看摄像头有没有插好，或者是不是正被别的软件（比如视频会议）占用，然后再开启一次。"
MSG_CAMERA_LOST = "摄像头没有画面了。请看看摄像头有没有插好，然后再开启一次。"
MSG_NO_TRACKER = "手部识别模型打不开。请双击 setup.bat 重新安装一次。"
MSG_STILL_CLOSING = "上一次的摄像头还没有关好，请过一会儿再开启。"
MSG_UNKNOWN = "手势控制出了点问题，已经自动关闭。可以再开启一次试试。"


def plain_message(exc: BaseException, fallback: str = MSG_UNKNOWN) -> str:
    """The exception's own text if it is already written for the user, else ``fallback``."""
    text = str(exc).strip()
    return text if any("一" <= ch <= "鿿" for ch in text) else fallback


def _say(on_status: StatusFn | None, message: str) -> None:
    """A broken status callback must never take the camera loop down with it."""
    if on_status is None:
        return
    try:
        on_status(message)
    except Exception:
        log.exception("status callback failed")


class _CaptureCancelled(Exception):
    """A completed prediction belongs to a run that was switched off."""


class _PredictionFrameExpired(Exception):
    """A processing pause made this frame too old to control input."""


class _CancellableRecognizer:
    def __init__(self, recognizer, stop_event=None, *, clock=None):
        self._recognizer = recognizer
        self._stop_event = stop_event
        self._clock = clock
        self._frame_origin = None
        self.name = getattr(recognizer, "name", "rule")

    def __getattr__(self, name):
        return getattr(self._recognizer, name)

    def begin_frame(self, origin):
        self._frame_origin = origin

    def _check_frame(self):
        if self._stop_event is not None and self._stop_event.is_set():
            raise _CaptureCancelled()
        now = self._clock() if self._clock is not None else time.perf_counter()
        if self._frame_origin is not None and now - self._frame_origin > ASLEEP_GAP_S:
            raise _PredictionFrameExpired()

    def predict(self, landmarks):
        self._check_frame()
        result = self._recognizer.predict(landmarks)
        self._check_frame()
        return result


def _open_tracker(cfg: Config, factory):
    try:
        if factory is not None:
            return factory(cfg)
        from .hand_tracker import HandTracker

        return HandTracker(cfg)
    except Exception as e:
        raise RuntimeError(plain_message(e, MSG_NO_TRACKER)) from e


def _open_camera(cfg: Config, factory):
    try:
        if factory is not None:
            return factory(cfg)
        from .camera import ThreadedCamera

        return ThreadedCamera(cfg.camera_index, cfg.frame_width, cfg.frame_height, cfg.camera_fps).start()
    except Exception as e:
        raise RuntimeError(plain_message(e, MSG_NO_CAMERA)) from e


def _capture_loop(stop_event: threading.Event, cfg: Config, camera, tracker, engine,
                  monitor=None, halo=None) -> None:
    import cv2

    t0 = prev = last_frame = time.perf_counter()
    if monitor is not None:
        monitor.start(t0)
    last_seq, last_ms = 0, -1
    while not stop_event.is_set():
        seq, frame = camera.read_if_new(last_seq)
        if stop_event.is_set():
            return  # a pending read may finish after the tray was switched OFF
        now = time.perf_counter()
        resumed = now - prev > ASLEEP_GAP_S
        if resumed:
            # A long processing pause invalidates all input history, even if
            # a new frame is ready immediately. Never resume an old drag/dwell
            # timer or zoom span across sleep (or an equivalent process freeze).
            engine.reset()
            if halo is not None:
                halo.update(None, active=False)
            last_frame = now
        prev = now

        age = getattr(camera, "seconds_since_last_frame", None) if frame is not None else None
        if resumed and frame is not None and age is not None and age > STALL_S:
            # A latest frame can have been captured just before the freeze.
            # Consume its sequence without feeding stale landmarks to input.
            last_seq, frame = seq, None

        if frame is None:
            waited = now - last_frame
            if waited > STALL_S:
                # The hand was last seen at last_frame, not when the stall alarm
                # fired: don't add another lost-hand grace delay after STALL_S.
                engine.process_hands([], last_frame - t0)
                engine.process_hands([], now - t0)
            if halo is not None and waited > STALL_S:
                halo.update(None, active=False)
            if monitor is not None:
                monitor.idle(now)
            if waited > CAMERA_LOST_S:
                raise RuntimeError(MSG_CAMERA_LOST)
            time.sleep(IDLE_SLEEP_S)
            continue

        last_seq, last_frame = seq, now
        if cfg.mirror:
            frame = cv2.flip(frame, 1)
        if stop_event.is_set():
            return
        last_ms = max(int((now - t0) * 1000), last_ms + 1)     # MediaPipe needs strictly increasing times
        tracking_start = time.perf_counter()
        hands = tracker.detect(frame, last_ms)
        if stop_event.is_set():
            return  # discard the result; run_headless's finally releases old input
        tracking_end = time.perf_counter()
        if tracking_end - now > ASLEEP_GAP_S:
            # Suspension can occur during image preparation or the native call,
            # after the loop's earlier check. Release input before this old result
            # can move/click/zoom, and give the camera a fresh wait deadline.
            engine.reset()
            if halo is not None:
                halo.update(None, active=False)
            prev = last_frame = tracking_end
            if monitor is not None:
                monitor.idle(tracking_end)
            continue
        recognizer = getattr(engine, "recognizer", None)
        if isinstance(recognizer, _CancellableRecognizer):
            recognizer.begin_frame(now)
        try:
            result = engine.process_hands(hands, now - t0)
        except _PredictionFrameExpired:
            # Include image preparation and tracking in the age budget. A
            # delayed prediction must not act before the next loop resets it.
            engine.reset()
            if halo is not None:
                halo.update(None, active=False)
            prev = last_frame = time.perf_counter()
            if monitor is not None:
                monitor.idle(last_frame)
            continue
        control_end = time.perf_counter()
        if halo is not None:
            active = bool(result.hands and result.enabled
                          and (result.button or result.label in ("point", "pinch_index", "pinch_middle")))
            halo.update(result.cursor, pressed=bool(result.button), active=active)
        if monitor is not None:
            monitor.frame(control_end, seq, result.hands, result.raw_label, result.label,
                          result.confidence, (tracking_end - tracking_start) * 1000,
                          (control_end - tracking_end) * 1000,
                          None if age is None else age * 1000,
                          cursor=result.cursor, events=result.events,
                          landmarks=hands[0] if hands and result.hands else None,
                          tracker_hands=len(hands), frame=frame)


def run_headless(
    stop_event: threading.Event,
    cfg: Config | None = None,
    on_status: StatusFn | None = None,
    camera_factory: Callable[[Config], object] | None = None,
    tracker_factory: Callable[[Config], object] | None = None,
    backend=None,
) -> None:
    """Gesture control without a preview window, until ``stop_event`` is set.

    ``camera_factory(cfg)`` returns a started camera (``read_if_new``, ``release``)
    and ``tracker_factory(cfg)`` a tracker (``detect``, ``close``); ``backend`` is
    the mouse. They default to the real webcam, MediaPipe and the real mouse.

    Raises RuntimeError with a plain Chinese message if the camera or the model
    can't be used. Held mouse buttons are always released on the way out.
    """
    cfg = cfg if cfg is not None else Config.load()
    _say(on_status, MSG_PREPARING)
    engine = tracker = camera = halo = monitor = None
    unwinding = False
    try:
        from .controller import make_backend
        from .engine import GestureEngine
        from .gesture_classifier import load_recognizer

        # The backend declares DPI awareness before any camera/native UI work.
        backend = backend if backend is not None else make_backend()
        if stop_event.is_set():
            return
        camera = _open_camera(cfg, camera_factory)
        if stop_event.is_set():
            return
        size = getattr(camera, "frame_size", None)
        if size is not None and tuple(size) != (cfg.frame_width, cfg.frame_height):
            cfg = replace(cfg)
            # These are measured dimensions, not user input to clamp.
            cfg.frame_width, cfg.frame_height = size
        tracker = _open_tracker(cfg, tracker_factory)
        if stop_event.is_set():
            return
        recognizer, description = load_recognizer(cfg)
        log.info("gesture recogniser: %s", description)
        engine = GestureEngine(cfg, backend, _CancellableRecognizer(recognizer, stop_event))
        monitor = PerformanceMonitor(cfg, camera, getattr(recognizer, "name", "rule"))
        if cfg.cursor_halo:
            from .cursor_halo import CursorHalo
            halo = CursorHalo()
        _say(on_status, MSG_READY)
        _capture_loop(stop_event, cfg, camera, tracker, engine, monitor, halo)
    except _CaptureCancelled:
        pass  # normal OFF; finally still releases input and closes devices
    except BaseException:
        unwinding = True
        raise
    finally:
        interruption = None
        # One failing step must not skip the others; the mouse button comes first.
        for part, method in ((engine, "reset"), (halo, "close"), (tracker, "close"),
                             (camera, "release")):
            if part is not None:
                try:
                    getattr(part, method)()
                except BaseException as exc:
                    log.exception("cleanup failed: %s", method)
                    if not isinstance(exc, Exception) and interruption is None:
                        interruption = exc
            if part is engine and monitor is not None:
                # Mouse release comes first. Snapshot while capture is still
                # live: slow device teardown must not look like slow tracking.
                try:
                    monitor.finish(time.perf_counter())
                except BaseException as exc:
                    log.exception("could not finish performance metrics")
                    if not isinstance(exc, Exception) and interruption is None:
                        interruption = exc
                monitor = None
        # Cancellation is still cancellation, but it must not abandon the
        # resources after the interrupted step or replace the original failure.
        if interruption is not None and not unwinding:
            raise interruption
    _say(on_status, MSG_CLOSED)


class CaptureWorker:
    """Start and stop ``loop_fn(stop_event, on_status)`` on a daemon thread.

    Every start gets its own stop event, and a restart waits for the previous
    loop to finish, so there is only ever one camera open and ``start`` / ``stop``
    can be called in any order from any thread.
    """

    def __init__(self, loop_fn: LoopFn | None = None, cfg: Config | None = None,
                 on_status: StatusFn | None = None):
        self.cfg = cfg
        self.on_status = on_status
        self._loop_fn = loop_fn if loop_fn is not None else self._default_loop
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._stop_event: threading.Event | None = None
        # Keep retiring loops even if a restart times out waiting for them.
        # Forgetting one would let the next ON open a second camera loop.
        self._threads: dict[threading.Thread, threading.Event] = {}

    def _default_loop(self, stop_event: threading.Event, on_status: StatusFn) -> None:
        run_headless(stop_event, self.cfg, on_status)     # cfg None: re-read config.json each time

    @property
    def is_running(self) -> bool:
        thread = self._thread
        return thread is not None and thread.is_alive()

    def start(self) -> None:
        """Start the loop in the background. Does nothing if it is already running."""
        with self._lock:
            current = self._thread
            previous_stop = self._stop_event
            if current is not None and current.is_alive() and not self._stop_event.is_set():
                return
            previous = tuple(self._threads)
            stop_event = threading.Event()
            thread = threading.Thread(target=self._run, args=(stop_event, previous, self.on_status),
                                      name="airtouch-capture", daemon=True)
            self._thread, self._stop_event = thread, stop_event
            self._threads[thread] = stop_event
            try:
                thread.start()
            except BaseException:
                stop_event.set()
                if thread.ident is None:
                    # A failed spawn must not publish a thread that stop()
                    # cannot join, or discard an older loop still winding down.
                    self._threads.pop(thread, None)
                    self._thread, self._stop_event = current, previous_stop
                raise

    def stop(self, timeout: float | None = 3) -> bool:
        """Ask the loop to end and wait up to ``timeout`` seconds (0 = don't wait).

        Returns True once it has ended. A loop that takes longer still ends on
        its own and still closes the camera.
        """
        with self._lock:
            threads = tuple(self._threads.items())
        if not threads:
            return True
        for _thread, stop_event in threads:
            stop_event.set()
        if any(thread is threading.current_thread() for thread, _event in threads):
            return False                    # called from on_status: can't wait for ourselves
        deadline = None if timeout is None else time.perf_counter() + timeout
        for thread, _event in threads:
            remaining = None if deadline is None else max(0.0, deadline - time.perf_counter())
            thread.join(remaining)
        return not any(thread.is_alive() for thread, _event in threads)

    def _run(self, stop_event: threading.Event, previous: tuple[threading.Thread, ...],
             on_status: StatusFn | None) -> None:
        # Callback ownership belongs to this run, including a terminal message
        # already in flight when a newer run replaces the public callback.
        emit = lambda message: _say(on_status, message)
        message = MSG_ENDED
        try:
            deadline = time.perf_counter() + WIND_DOWN_S
            for thread in previous:
                thread.join(max(0.0, deadline - time.perf_counter()))
                if thread.is_alive():
                    raise RuntimeError(MSG_STILL_CLOSING)
            if not stop_event.is_set():
                self._loop_fn(stop_event, emit)
        except BaseException as e:          # even sys.exit() in a library must not leave the switch ON
            log.exception("capture loop failed")
            message = ERROR_PREFIX + plain_message(e)
        finally:
            # Cleared before the last message, so whoever receives it sees is_running == False.
            with self._lock:
                self._threads.pop(threading.current_thread(), None)
                if self._thread is threading.current_thread():
                    self._thread = None
        if not stop_event.is_set():         # nobody cares why a loop died after being told to stop
            emit(message)
