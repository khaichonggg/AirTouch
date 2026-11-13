"""The live webcam app: camera -> hand tracker -> gesture engine -> real mouse.

``main`` owns setup and teardown, ``LiveSession`` is the loop itself and
``process_frame`` the work done on one new frame. The camera, tracker and mouse
backend can all be injected, so the whole loop runs in tests with no webcam and
without touching the real mouse.

Whatever goes wrong, ``main`` lets go of the mouse on the way out.
"""

from __future__ import annotations

import dataclasses
import os
import sys
import time
import traceback

import cv2

from .camera import RateMeter, ThreadedCamera
from .config import Config
from .console import use_utf8_console
from .controller import make_backend
from .download_model import ensure_model
from .engine import GestureEngine
from .gesture_classifier import load_recognizer
from .hand_tracker import HandTracker
from .hud import draw_hud
from .session import ASLEEP_GAP_S, _CancellableRecognizer, _PredictionFrameExpired

WINDOW = "AirTouch"
IDLE_SLEEP_S = 0.002        # pause when there is no new frame, so the loop can't spin a CPU core
STALL_RELEASE_S = 0.3       # no frame for this long: treat as "no hand", which lets go of the mouse
STALL_WARN_S = 3.0          # ...and after this long, tell the person
REPORT_EVERY_S = 1.0        # how often lite mode prints its speed line
ESC = 27

ENGINE_WORDS = {"rule": "规则识别（默认方式）", "ml": "训练好的手势模型"}
NO_MODEL_MESSAGE = "训练好的手势模型现在用不了（可能还没有训练过），先用默认方式识别手势。"


def say(message: str) -> None:
    """Tell the person at the console. A broken console must never stop the mouse loop."""
    try:
        print(message, flush=True)
    except (OSError, ValueError):
        pass


def speed_words(fps: float) -> str:
    if fps >= 20:
        return "很流畅"
    if fps >= 12:
        return "还可以"
    return "有点慢，把房间弄亮一点、关掉其他程序会好很多"


def _track_frame(frame, tracker, t: float, mirror: bool):
    """Keep image/landmark coordinates aligned, without sending any input."""
    if mirror:
        frame = cv2.flip(frame, 1)
    hands = tracker.detect(frame, int(t * 1000))
    return frame, hands


def process_frame(frame, tracker, engine, t: float, mirror: bool = True):
    """One new camera frame -> one engine step. Returns (frame, hands, result).

    The picture is flipped BEFORE tracking, so the landmarks match what is shown
    on screen and moving the hand to the right moves the cursor to the right.
    """
    frame, hands = _track_frame(frame, tracker, t, mirror)
    return frame, hands, engine.process_hands(hands, t)


class LiveSession:
    """One run of the main loop. ``step`` is one pass; False means time to quit."""

    def __init__(self, cfg: Config, camera, tracker, engine, window: bool = False,
                 max_frames: int | None = None):
        self.cfg = cfg      # the engine's own Config object: keys change it live
        self.camera = camera
        self.tracker = tracker
        self.engine = engine
        self.window = window
        self.max_frames = max_frames
        self.engine_name = getattr(engine.recognizer, "name", "rule")
        self.frames = 0
        self.last_seq = -1
        self.speed = RateMeter()
        self.last_frame_t = self.last_report_t = time.perf_counter()
        self.last_poll_t = self.last_frame_t
        self.stall_warned = False

    def run(self) -> None:
        while self.step():
            pass

    def step(self) -> bool:
        seq, frame = self.camera.read_if_new(self.last_seq)
        now = time.perf_counter()
        resumed = now - self.last_poll_t > ASLEEP_GAP_S
        self.last_poll_t = now
        if resumed:
            self.engine.reset()
            self.last_frame_t = self.last_report_t = now
            self.stall_warned = False
            self.speed = RateMeter()
            age = getattr(self.camera, "seconds_since_last_frame", None) if frame is not None else None
            if frame is not None and age is not None and age > STALL_RELEASE_S:
                self.last_seq, frame = seq, None
        # imshow recreates a closed native window. Check before drawing again,
        # after the first frame has made the newly created window visible.
        if self.window and self.frames and self._window_closed():
            return False
        if frame is None:
            self._on_no_frame(now)
        else:
            self.last_seq = seq
            self._on_frame(frame, now)
        if self.window and not self._pump_window():
            return False
        if frame is None:
            time.sleep(IDLE_SLEEP_S)
        return self.max_frames is None or self.frames < self.max_frames

    # -- camera events ---------------------------------------------------------
    def _on_frame(self, frame, t: float) -> None:
        check_window = self.window and self.frames > 0
        tracking_start = time.perf_counter()
        frame, hands = _track_frame(frame, self.tracker, t, self.cfg.mirror)
        tracking_end = time.perf_counter()
        if check_window and self._window_closed():
            return  # the person may close the preview while tracking is busy
        if tracking_end - tracking_start > ASLEEP_GAP_S:
            # Do not act on a result captured before an in-call pause.
            self._reset_after_processing_gap(tracking_end)
            return
        if self.stall_warned:
            self.stall_warned = False
            say("摄像头又有画面了。")
        recognizer = self.engine.recognizer
        guarded = _CancellableRecognizer(recognizer, clock=lambda: time.perf_counter())
        guarded.begin_frame(t)
        self.engine.recognizer = guarded
        try:
            result = self.engine.process_hands(hands, t)
        except _PredictionFrameExpired:
            self._reset_after_processing_gap(time.perf_counter())
            return
        finally:
            self.engine.recognizer = recognizer
        self.frames += 1
        fps = self.speed.tick(t)
        if self.window:
            if self.cfg.show_hud:
                draw_hud(frame, result, hands, fps, self.engine_name,
                         self.engine.controller.active_box, self.engine.enabled,
                         self.cfg.dwell_enabled)
            if check_window and self._window_closed():
                return  # imshow would otherwise resurrect a window closed while drawing
            cv2.imshow(WINDOW, frame)
        elif t - self.last_report_t >= REPORT_EVERY_S:
            self.last_report_t = t
            say(f"速度：每秒处理 {fps:.0f} 张画面（{speed_words(fps)}）")
        # Measured from here, so a slow tracking step never looks like a silent camera.
        self.last_frame_t = time.perf_counter()

    def _reset_after_processing_gap(self, now: float) -> None:
        self.engine.reset()
        self.last_frame_t = self.last_report_t = self.last_poll_t = now
        self.stall_warned = False
        self.speed = RateMeter()

    def _on_no_frame(self, t: float) -> None:
        silent_for = t - self.last_frame_t
        if silent_for <= STALL_RELEASE_S:
            return
        # A stalled camera must NEVER leave a mouse button stuck down. Nothing has been
        # seen since the last frame, so that is when the hand went missing: reporting
        # it from then on means the engine's own (shorter) grace period for a lost hand
        # is already over, and it lets go right now.
        self.engine.process_hands([], self.last_frame_t)
        self.engine.process_hands([], t)
        if silent_for > STALL_WARN_S and not self.stall_warned:
            self.stall_warned = True
            say("摄像头好像没有画面了（可能被拔掉或卡住了），鼠标按键已经松开。"
                "请检查一下摄像头；如果一直没恢复，关掉程序重新打开。")

    # -- window ----------------------------------------------------------------
    def _pump_window(self) -> bool:
        """Keep the window responsive and handle keys. False = the person asked to quit."""
        key = cv2.waitKey(1) & 0xFF
        if key == ESC or key in (ord("q"), ord("Q")):
            return False
        if key in (ord("p"), ord("P")):
            self._toggle_pause()
        elif key in (ord("d"), ord("D")):
            self._toggle_dwell()
        elif key in (ord("m"), ord("M")):
            self._switch_recognizer()
        return not self._window_closed()

    def _window_closed(self) -> bool:
        # Must be checked: imshow silently re-creates a window closed with its X
        # button, so without this the app could never be closed that way.
        try:
            return cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1
        except cv2.error:
            return True

    def _toggle_pause(self) -> None:
        self.engine.set_enabled(not self.engine.enabled)
        say("继续：手势又能控制鼠标了。" if self.engine.enabled
            else "已暂停：手势不再控制鼠标。再按 P 继续。")

    def _toggle_dwell(self) -> None:
        self.cfg.dwell_enabled = not self.cfg.dwell_enabled
        self.engine.controller.reset_dwell()
        say("已打开停留点击：鼠标停住不动一小会儿，就会自动点一下。" if self.cfg.dwell_enabled
            else "已关闭停留点击。")

    def _switch_recognizer(self) -> None:
        self.engine.reset()     # a model can take a second to load: don't hold a button through it
        self.cfg.engine = "rule" if self.cfg.engine == "ml" else "ml"
        wanted = self.cfg.engine
        recognizer, _ = load_recognizer(self.cfg)
        self.engine.set_recognizer(recognizer)
        self.engine_name = self.cfg.engine = getattr(recognizer, "name", "rule")
        if wanted == "ml" and self.engine_name != "ml":
            say(NO_MODEL_MESSAGE)
        else:
            say(f"现在用的是：{ENGINE_WORDS.get(self.engine_name, self.engine_name)}。")


def _match_camera_shape(cfg: Config, camera) -> Config:
    """Use the picture shape the camera really delivers. It may ignore the size we
    asked for, and every hand measurement depends on the width/height ratio."""
    size = getattr(camera, "frame_size", None)
    if size is None or tuple(size) == (cfg.frame_width, cfg.frame_height):
        return cfg
    cfg = dataclasses.replace(cfg)
    # Config bounds apply to requested settings, not dimensions already delivered
    # by a driver (which may exceed those bounds, e.g. 1920 x 1200).
    cfg.frame_width, cfg.frame_height = size
    return cfg


def _open_window() -> bool:
    try:
        cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE)
        return True
    except cv2.error:
        say("这台电脑没法显示画面窗口，改为不显示窗口运行。退出时按 Ctrl+C。")
        return False


def _say_started(engine_name: str, window: bool) -> None:
    say(f"AirTouch 已启动。现在用的是：{ENGINE_WORDS.get(engine_name, engine_name)}。")
    if window:
        say("退出：按 Q 或 Esc，或者点窗口右上角的 ×（输入法是中文状态时按键可能没反应，点 × 就行）。")
        say("其他按键：P 暂停/继续，D 打开/关闭停留点击，M 换一种识别方式。")
    else:
        say("现在不显示画面，最省电脑资源。退出：在这个窗口里按 Ctrl+C。")


def main(cfg: Config | None = None, show_window: bool = True, camera=None, tracker=None,
         backend=None, max_frames: int | None = None) -> int:
    """Run the live app until the person quits. Returns the exit code."""
    use_utf8_console()
    # Private copy: keys like D change it live, the caller's object stays as it was.
    cfg = dataclasses.replace(cfg) if cfg is not None else Config.load()
    engine = None
    window = False
    code = 0
    unwinding = False
    try:
        say("AirTouch 正在启动，请稍等几秒……")
        # Before anything else: the real backend declares the program DPI-aware (so
        # cursor coordinates are real pixels), which Windows only allows up front.
        if backend is None:
            backend = make_backend()
        try:
            if tracker is None:
                ensure_model()
            if camera is None:
                camera = ThreadedCamera(cfg.camera_index, cfg.frame_width, cfg.frame_height,
                                        cfg.camera_fps)
            if hasattr(camera, "start"):
                camera.start()
            cfg = _match_camera_shape(cfg, camera)
            if tracker is None:
                tracker = HandTracker(cfg)
        except (RuntimeError, FileNotFoundError) as e:
            say(str(e))     # these already carry a plain message for the person
            return 1

        wanted_ml = cfg.engine == "ml"
        recognizer, _ = load_recognizer(cfg)
        cfg.engine = getattr(recognizer, "name", "rule")
        if wanted_ml and cfg.engine != "ml":
            say(NO_MODEL_MESSAGE)
        engine = GestureEngine(cfg, backend, recognizer)

        window = show_window and _open_window()
        _say_started(cfg.engine, window)
        LiveSession(cfg, camera, tracker, engine, window, max_frames).run()
    except KeyboardInterrupt:
        pass        # Ctrl+C is the normal way out of lite mode
    except Exception as e:
        # A traceback means nothing to the person using this; `finally` still lets go
        # of the mouse. Set AIRTOUCH_DEBUG=1 to see the traceback when developing.
        unwinding = True  # reporting can itself be interrupted before we handle e
        say(f"程序遇到问题，已经停止。问题信息：{type(e).__name__}: {e}")
        if os.environ.get("AIRTOUCH_DEBUG"):
            traceback.print_exc()
        code = 1
        unwinding = False
    except BaseException:
        unwinding = True
        raise
    finally:
        interruption = None
        # Mouse first: nothing that happens below may leave a button stuck down.
        steps = (getattr(engine, "reset", None), getattr(tracker, "close", None),
                 getattr(camera, "release", None), cv2.destroyAllWindows if window else None)
        for step in steps:
            try:
                if step is not None:
                    step()
            except BaseException as exc:
                # Ctrl+C remains the CLI's normal way out, including during
                # teardown. Defer other cancellation until every resource has
                # been attempted, without replacing an already-unwinding exit.
                if not isinstance(exc, (Exception, KeyboardInterrupt)) and interruption is None:
                    interruption = exc
        if interruption is not None and not unwinding:
            raise interruption
    say("AirTouch 已退出。")
    return code


if __name__ == "__main__":
    sys.exit(main())
