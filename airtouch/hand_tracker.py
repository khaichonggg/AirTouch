"""MediaPipe hand tracking (Tasks API, VIDEO mode).

Two traps this class exists to hide:

* MediaPipe's file loader fails on non-ASCII paths (the project may live in a
  folder like "... 【 Final Year Project 】"), so the model is loaded from
  bytes with ``model_asset_buffer``.
* ``detect_for_video`` raises ValueError unless timestamps strictly increase,
  so ``detect`` bumps any value that would break that.
"""

from __future__ import annotations

import contextlib
import os
import threading
import time
from pathlib import Path

import cv2
import numpy as np

from .config import HAND_MODEL_PATH, Config

MODEL_MISSING_MESSAGE = (
    "找不到手部识别模型：{path}\n"
    "请先运行 setup.bat，或者在命令行里运行：python -m airtouch.download_model"
)
MODEL_BROKEN_MESSAGE = (
    "手部识别模型打不开，文件可能坏了：{path}\n"
    "请删除这个文件，再运行 setup.bat 或 python -m airtouch.download_model 重新下载。"
)

_NATIVE_STDERR_LOCK = threading.RLock()


@contextlib.contextmanager
def _quiet_native_stderr():
    """Suppress native startup chatter where descriptor replacement is safe.

    Windows native libraries can retain their own CRT's borrowed stderr HANDLE.
    dup2 closes that handle, even when Python's fd2 is subsequently restored;
    repeated MediaPipe creation can then terminate with STATUS_INVALID_HANDLE.
    Keep Windows stderr stable. The tray entry point already sends it to its
    persistent log before importing native libraries.
    """
    if os.name == "nt":
        yield
        return
    # Descriptor 2 belongs to the whole process. Overlapping startup scopes
    # must restore it in order; otherwise the last scope can leave it muted.
    with _NATIVE_STDERR_LOCK:
        try:
            saved = os.dup(2)
        except OSError:     # no real stderr: nothing to hide
            saved = None
        if saved is None:
            yield
            return
        null = os.open(os.devnull, os.O_WRONLY)
        try:
            os.dup2(null, 2)
            yield
        finally:
            os.dup2(saved, 2)
            os.close(saved)
            os.close(null)


class HandTracker:
    """Finds hands in BGR frames. ``detect`` returns a list of (21, 3) float64
    arrays of MediaPipe normalised coordinates (see ``airtouch.landmarks``)."""

    def __init__(self, cfg: Config | None = None, model_path: str | Path = HAND_MODEL_PATH,
                 num_hands: int | None = None):
        cfg = cfg or Config()
        path = Path(model_path)
        if not path.is_file():
            raise FileNotFoundError(MODEL_MISSING_MESSAGE.format(path=path))

        # Imported here because it takes seconds, and `--help`, tests with a fake
        # tracker or the fallback paths should never have to pay for it.
        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions, vision

        self._mp = mp
        self._last_ts = -1
        options = vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_buffer=path.read_bytes()),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=cfg.max_hands if num_hands is None else num_hands,
            min_hand_detection_confidence=cfg.min_detection_confidence,
            # Below this the tracker falls back to a full palm search: same role
            # as Config's tracking threshold, which has no separate presence value.
            min_hand_presence_confidence=cfg.min_tracking_confidence,
            min_tracking_confidence=cfg.min_tracking_confidence,
        )
        with _quiet_native_stderr():
            try:
                self._landmarker = vision.HandLandmarker.create_from_options(options)
            except Exception as e:
                raise RuntimeError(MODEL_BROKEN_MESSAGE.format(path=path)) from e
            # One blank frame now: the native side logs a last warning on its first
            # frame, and the first real frame then isn't slower than the rest.
            try:
                self.detect(np.zeros((cfg.frame_height, cfg.frame_width, 3), dtype=np.uint8), 0)
            except BaseException:
                # __init__ did not return, so its caller cannot close the tracker.
                with contextlib.suppress(Exception):
                    self.close()
                raise

    def detect(self, frame_bgr: np.ndarray, t_ms: int | None = None) -> list[np.ndarray]:
        """Hands in this frame. ``t_ms`` defaults to a monotonic clock."""
        if self._landmarker is None:
            raise RuntimeError("HandTracker is closed")
        # cvtColor always returns a new contiguous array, which mp.Image requires
        # (a [..., ::-1] view would not do).
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
        result = self._landmarker.detect_for_video(image, self._next_timestamp(t_ms))
        return [np.array([(p.x, p.y, p.z) for p in hand], dtype=np.float64)
                for hand in result.hand_landmarks]

    detect_hands = detect

    def _next_timestamp(self, t_ms: int | None) -> int:
        ts = int(time.monotonic() * 1000) if t_ms is None else int(t_ms)
        self._last_ts = max(ts, self._last_ts + 1)
        return self._last_ts

    def close(self) -> None:
        """Free the native tracker. Safe to call any number of times."""
        landmarker, self._landmarker = self._landmarker, None
        if landmarker is not None:
            landmarker.close()

    def __enter__(self) -> "HandTracker":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
