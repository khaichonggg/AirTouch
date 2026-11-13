"""Threaded webcam reader that only ever keeps the newest frame.

``cv2.VideoCapture.read`` blocks until the camera delivers and drivers queue
old frames, so reading in the main loop would add lag and tie the loop to the
camera's pace. A background thread keeps draining the camera instead, and the
main loop just asks "anything new?".

The main loop asks that hundreds of times a second, so ``read_if_new`` has to
stay cheap: it compares sequence numbers and copies a frame only when there
really is a new one. (Copying a 640x480 frame on every poll was the original
project's biggest CPU hog.)
"""

from __future__ import annotations

import math
import sys
import threading
import time

import cv2
import numpy as np

OPEN_FAILED_MESSAGE = (
    "打不开摄像头。请按下面的顺序检查一下：\n"
    "1. 摄像头有没有插好？（笔记本电脑看看有没有镜头盖或隐私开关）\n"
    "2. 是不是别的程序正在用它？（比如 Zoom、Teams、微信视频，先把它们关掉）\n"
    "3. Windows 有没有允许使用摄像头？（设置 > 隐私和安全性 > 摄像头）\n"
    "检查完再运行一次就好。"
)

READ_RETRY_S = 0.01         # pause after a failed read, so a dead camera can't spin a CPU core
FIRST_FRAME_POLL_S = 0.005
JOIN_TIMEOUT_S = 2.0

DEVICE_CLOSING_MESSAGE = "上一次的摄像头还在关闭，请稍等一下再开启。"
_DEVICE_OWNERS: dict[int, object] = {}
_DEVICE_OWNERS_LOCK = threading.Lock()


def _claim_device(index: int) -> object:
    # A stopped reader may still be inside a native driver call. Ownership
    # outlives its ThreadedCamera instance until that reader actually closes.
    with _DEVICE_OWNERS_LOCK:
        if index in _DEVICE_OWNERS:
            raise RuntimeError(DEVICE_CLOSING_MESSAGE)
        token = object()
        _DEVICE_OWNERS[index] = token
        return token


def _release_device(index: int, token: object) -> None:
    with _DEVICE_OWNERS_LOCK:
        if _DEVICE_OWNERS.get(index) is token:
            del _DEVICE_OWNERS[index]


class RateMeter:
    """Smoothed events per second. Averages the *gaps* between events, because
    averaging the rates themselves over-weights short gaps and reads too high."""

    def __init__(self, alpha: float = 0.1):
        self.alpha = alpha
        self.last: float | None = None
        self.gap: float | None = None

    def tick(self, now: float) -> float:
        if self.last is not None:
            gap = now - self.last
            self.gap = gap if self.gap is None else self.alpha * gap + (1 - self.alpha) * self.gap
        self.last = now
        return self.rate

    @property
    def rate(self) -> float:
        return 1.0 / self.gap if self.gap else 0.0


def _default_openers() -> list:
    """Ways to open a real camera, best first."""
    try:
        # A missing camera makes OpenCV log scary technical lines;
        # OPEN_FAILED_MESSAGE already says everything the person needs.
        cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_SILENT)
    except AttributeError:
        pass
    if sys.platform == "win32":
        # DirectShow opens in about a second; the default (Media Foundation) can take ten.
        return [lambda index: cv2.VideoCapture(index, cv2.CAP_DSHOW), cv2.VideoCapture]
    return [cv2.VideoCapture]


class ThreadedCamera:
    """The newest camera frame, read on a background thread.

    Every frame gets a strictly increasing ``seq`` (1, 2, 3, ...). It keeps
    counting across ``release()`` + ``start()``, so a caller's ``last_seq``
    stays meaningful.

    capture_factory(index) replaces ``cv2.VideoCapture`` (tests use a fake).
    first_frame_timeout: how long ``start()`` waits for a first picture. Waiting
    is what catches a camera in use by another program: on Windows it often
    "opens" without complaint and then never delivers anything.
    """

    def __init__(self, index: int = 0, width: int = 640, height: int = 480, fps: int = 30,
                 capture_factory=None, first_frame_timeout: float = 5.0):
        self.index = index
        self.width = width
        self.height = height
        self.fps = fps
        self.first_frame_timeout = first_frame_timeout
        self._factory = capture_factory
        self._lock = threading.Lock()
        self._frame: np.ndarray | None = None
        self._seq = 0
        self._rate = RateMeter()
        self._cap = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()

    # -- lifecycle -------------------------------------------------------------
    def start(self) -> "ThreadedCamera":
        """Open the camera and wait for its first frame. Safe to call twice.

        Raises RuntimeError with a plain message if no picture can be had."""
        if self.is_open:
            return self
        openers = [self._factory] if self._factory is not None else _default_openers()
        for opener in openers:
            if self._try_open(opener):
                return self
        raise RuntimeError(OPEN_FAILED_MESSAGE)

    def _try_open(self, opener) -> bool:
        token = _claim_device(self.index)
        cap = None
        try:
            cap = opener(self.index)
            opened = cap.isOpened()
        except BaseException as error:
            try:
                if cap is not None:
                    try:
                        cap.release()
                    except Exception:
                        pass
            finally:
                _release_device(self.index, token)
            if not isinstance(error, Exception):
                raise                  # cancellation is not a driver fallback
            return False
        if not opened:
            try:
                cap.release()
            except Exception:
                pass
            finally:
                _release_device(self.index, token)
            return False
        try:
            self._configure(cap)
            self._stop = threading.Event()
            seq_before = self._seq
            self._cap = cap
            self._thread = threading.Thread(target=self._read_loop, args=(cap, self._stop, token),
                                            name="airtouch-camera", daemon=True)
            self._thread.start()
        except BaseException:
            self._stop.set()
            # Once started, only the reader may close its native capture.
            if self._thread is None or self._thread.ident is None:
                try:
                    cap.release()
                finally:
                    self._thread = self._cap = None
                    _release_device(self.index, token)
            raise
        try:
            deadline = time.perf_counter() + self.first_frame_timeout
            while self._seq == seq_before and time.perf_counter() < deadline:
                time.sleep(FIRST_FRAME_POLL_S)
            if self._seq != seq_before:
                return True
        except BaseException:
            # start() did not hand ownership to its caller. Ask the reader to
            # stop and close itself even when first-frame waiting is cancelled.
            self.release()
            raise
        self.release()
        return False

    def _configure(self, cap) -> None:
        settings = (
            (cv2.CAP_PROP_BUFFERSIZE, 1),       # no queue of stale frames in the driver
            (cv2.CAP_PROP_FRAME_WIDTH, self.width),
            (cv2.CAP_PROP_FRAME_HEIGHT, self.height),
            (cv2.CAP_PROP_FPS, self.fps),
        )
        for prop, value in settings:
            try:
                cap.set(prop, value)    # a camera may ignore any of these, which is fine
            except Exception:
                pass

    def _read_loop(self, cap, stop: threading.Event, token: object) -> None:
        try:
            while not stop.is_set():
                try:
                    ok, frame = cap.read()
                except Exception:   # a driver hiccup must not kill the thread
                    ok, frame = False, None
                if not ok or frame is None:
                    stop.wait(READ_RETRY_S)
                    continue
                now = time.perf_counter()
                with self._lock:
                    if stop.is_set():   # a read that was stuck until after release()
                        break
                    self._seq += 1
                    self._frame = frame     # replaced, never written into: see read_if_new
                    self._rate.tick(now)
        finally:
            # Only the reader closes its capture, including after a delayed read.
            # Releasing from another thread while read() is inside a driver can crash.
            try:
                cap.release()
            except Exception:
                pass
            finally:
                _release_device(self.index, token)

    def release(self) -> None:
        """Stop reading and wait briefly for closure. A stuck read closes on return.

        Safe to call any number of times; never close underneath a driver read.
        """
        thread = self._thread
        self._thread = self._cap = None
        self._stop.set()
        if thread is not None:
            thread.join(JOIN_TIMEOUT_S)
        with self._lock:
            self._frame = None

    def __enter__(self) -> "ThreadedCamera":
        return self.start()

    def __exit__(self, *exc) -> None:
        self.release()

    # -- reading ---------------------------------------------------------------
    def read_if_new(self, last_seq: int) -> tuple[int, np.ndarray | None]:
        """(seq, frame) if a frame newer than ``last_seq`` exists, else (seq, None).

        The nothing-new path touches no pixels, so polling is free. The copy
        happens outside the lock: the reader thread replaces ``_frame`` rather
        than writing into it, so it can't change under us, and it never has to
        wait for the copy. The caller owns the copy and may draw on it.
        """
        with self._lock:
            seq, frame = self._seq, self._frame
        if frame is None or seq <= last_seq:
            return seq, None
        return seq, frame.copy()

    def read_latest(self) -> tuple[int, np.ndarray | None]:
        """(seq, copy of the newest frame), even if already seen. (seq, None) before the first."""
        with self._lock:
            seq, frame = self._seq, self._frame
        return seq, None if frame is None else frame.copy()

    # -- status ----------------------------------------------------------------
    @property
    def is_open(self) -> bool:
        thread = self._thread
        return thread is not None and thread.is_alive()

    @property
    def frame_size(self) -> tuple[int, int] | None:
        """(width, height) the camera really delivers, which may differ from what was asked."""
        frame = self._frame
        return None if frame is None else (frame.shape[1], frame.shape[0])

    @property
    def seconds_since_last_frame(self) -> float:
        """Infinity until the first frame has arrived."""
        with self._lock:
            last = self._rate.last
        return math.inf if last is None else time.perf_counter() - last

    @property
    def measured_fps(self) -> float:
        """Smoothed capture rate. Falls towards 0 while the camera is stalled, even
        though no new samples arrive to pull the average down."""
        with self._lock:
            last, gap = self._rate.last, self._rate.gap
        if last is None or not gap:
            return 0.0
        return 1.0 / max(gap, time.perf_counter() - last)
