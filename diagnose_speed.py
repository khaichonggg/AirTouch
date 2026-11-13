"""How fast can AirTouch run on THIS computer, and what is holding it back?

Measures the camera (speed and brightness) and the two steps the computer does
for every picture (colour conversion, hand tracking), then explains the result
in plain Chinese with concrete tips. Saves nothing unless --report is given.

Run:  python diagnose_speed.py [--no-camera] [--camera N] [--seconds 5] [--report FILE]
"""

from __future__ import annotations

import argparse
import dataclasses
import statistics
import sys
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from airtouch.camera import ThreadedCamera
from airtouch.config import Config
from airtouch.console import use_utf8_console
from airtouch.hand_tracker import HandTracker

# Average brightness, 0 = black, 255 = white.
BLACK_BRIGHTNESS = 8        # only a covered lens or a privacy switch gets this dark
DIM_BRIGHTNESS = 60         # webcams lengthen their exposure below about this, and so slow down
# Pictures per second from the camera.
CAMERA_GOOD_FPS = 24
CAMERA_SLOW_FPS = 15
# Milliseconds of hand tracking per picture (33 ms is what 30 pictures a second allows).
TRACKER_GOOD_MS = 33
TRACKER_SLOW_MS = 66

KEEP_FRAMES = 30            # recent camera pictures kept for the timing tests
TRACKER_REPEATS = 30
TRACKER_WARMUP = 3          # the first few detections are slower and would skew the result
CONVERT_REPEATS = 50

SLOW_COMPUTER_TIPS = ("可以试试：关掉不用的程序（尤其是浏览器标签、视频和游戏）；"
                      "笔记本接上电源；用 run.bat --lite 运行（不显示画面，最省电脑资源）。")


@dataclass
class Measurements:
    camera_tested: bool = False
    camera_error: str = ""
    camera_fps: float | None = None
    brightness: float | None = None
    convert_ms: float | None = None
    tracker_ms: float | None = None
    tracker_error: str = ""
    real_frames: bool = False


def synthetic_frame(width: int = 640, height: int = 480) -> np.ndarray:
    """A busy picture, so the tracker does a realistic amount of work with no camera."""
    return np.random.default_rng(0).integers(0, 256, (height, width, 3), dtype=np.uint8)


def measure_camera(camera, seconds: float = 5.0):
    """Watch a started camera. Returns (pictures per second, brightness, recent frames)."""
    frames: deque = deque(maxlen=KEEP_FRAMES)
    levels = []
    first_seq = last_seq = -1
    first_t = last_t = 0.0
    deadline = time.perf_counter() + seconds
    while time.perf_counter() < deadline:
        seq, frame = camera.read_if_new(last_seq)
        now = time.perf_counter()
        if frame is None:
            time.sleep(0.002)
            continue
        if not levels:
            first_seq, first_t = seq, now
        last_seq, last_t = seq, now
        frames.append(frame)
        levels.append(float(frame.mean()))
    if not levels:
        return None, None, []
    # Counted by sequence number, so pictures this loop happened to skip still count.
    fps = (last_seq - first_seq) / (last_t - first_t) if last_t > first_t else 0.0
    return fps, statistics.fmean(levels), list(frames)


def measure_convert(frame: np.ndarray, repeats: int = CONVERT_REPEATS) -> float:
    """Median milliseconds for the BGR -> RGB conversion the tracker does on each picture."""
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        times.append(time.perf_counter() - t0)
    return statistics.median(times) * 1000


def measure_tracker(tracker, frames: list, repeats: int = TRACKER_REPEATS) -> float:
    """Median milliseconds per ``tracker.detect``."""
    times = []
    for i in range(repeats + TRACKER_WARMUP):
        frame = frames[i % len(frames)]
        t0 = time.perf_counter()
        tracker.detect(frame, i * 33)
        times.append(time.perf_counter() - t0)
    return statistics.median(times[TRACKER_WARMUP:]) * 1000


def expected_speed(m: Measurements) -> float | None:
    """Pictures per second the whole chain can manage: whichever step is slowest."""
    if m.tracker_ms is None:
        return m.camera_fps
    computer = 1000.0 / max(m.tracker_ms + (m.convert_ms or 0.0), 0.001)
    return computer if m.camera_fps is None else min(computer, m.camera_fps)


def verdict(m: Measurements) -> list[str]:
    """The conclusion, in everyday words."""
    lines: list[str] = []
    if not m.camera_tested:
        lines.append("这次没有打开摄像头，只测了电脑本身的速度。")
    elif m.camera_error:
        lines.append(m.camera_error)
    elif m.camera_fps is None:
        lines.append("摄像头打开了，但一张画面都没有送来。请关掉其他正在用摄像头的程序，再试一次。")
    else:
        too_dark = False
        if m.brightness is not None and m.brightness < BLACK_BRIGHTNESS:
            too_dark = True
            lines.append("画面几乎全黑：看看镜头盖有没有打开，笔记本上的摄像头隐私开关有没有关上。")
        elif m.brightness is not None and m.brightness < DIM_BRIGHTNESS:
            too_dark = True
            lines.append("画面比较暗。光线不够时摄像头会自动放慢，手也更难认出来。"
                         "开灯或者面向窗户坐，这是提速最有效的办法。")
        if m.camera_fps >= CAMERA_GOOD_FPS:
            lines.append("摄像头的速度正常。")
        elif not too_dark:
            lines.append("摄像头偏慢。最常见的原因是光线不够，把房间弄亮一点通常马上就快了；"
                         "也可以关掉其他正在用摄像头的程序，或者换一个 USB 口。")

    if m.tracker_ms is None:
        lines.append(m.tracker_error or "没有测到电脑认手的速度。")
    elif m.tracker_ms <= TRACKER_GOOD_MS:
        lines.append("这台电脑认手的速度完全够用。")
    elif m.tracker_ms <= TRACKER_SLOW_MS:
        lines.append("这台电脑认手的速度勉强够用，稍微有点吃力。" + SLOW_COMPUTER_TIPS)
    else:
        lines.append("这台电脑认手比较吃力，这是现在最慢的一步。" + SLOW_COMPUTER_TIPS)

    speed = expected_speed(m)
    if speed is not None:
        feel = "用起来会很流畅" if speed >= 20 else "能用，但会感觉慢一点" if speed >= 12 else "会感觉明显卡顿"
        lines.append(f"综合下来，每秒大约能处理 {speed:.0f} 张画面，{feel}。")
    return lines


def summary(m: Measurements) -> list[str]:
    """The measured numbers, in everyday words."""
    lines = ["—— 测量结果 ——"]
    if m.camera_fps is not None:
        lines.append(f"摄像头：每秒送来 {m.camera_fps:.1f} 张画面")
    elif m.camera_tested:
        lines.append("摄像头：没有拿到画面")
    else:
        lines.append("摄像头：这次没有测")
    if m.brightness is not None:
        lines.append(f"画面亮度：{m.brightness:.0f}（0 是全黑，255 是全白，60 以上比较好）")
    if m.convert_ms is not None:
        lines.append(f"整理画面颜色：每张 {m.convert_ms:.2f} 毫秒")
    if m.tracker_ms is not None:
        source = "用的是摄像头拍到的画面" if m.real_frames else "用的是电脑生成的测试画面"
        lines.append(f"认出手的位置：每张 {m.tracker_ms:.1f} 毫秒（{source}）")
    return lines


def diagnose(cfg: Config, use_camera: bool = True, seconds: float = 5.0, camera=None,
             tracker=None) -> Measurements:
    """Run every measurement. Nothing here may crash on a missing camera or model."""
    m = Measurements(camera_tested=use_camera)
    frames: list = []
    if use_camera:
        if camera is None:
            camera = ThreadedCamera(cfg.camera_index, cfg.frame_width, cfg.frame_height, cfg.camera_fps)
        print(f"正在测试摄像头，大约 {seconds:.0f} 秒。这段时间可以把手放在镜头前……", flush=True)
        try:
            camera.start()
            m.camera_fps, m.brightness, frames = measure_camera(camera, seconds)
        except RuntimeError as e:
            m.camera_error = str(e)
        finally:
            camera.release()
    m.real_frames = bool(frames)
    if not frames:
        frames = [synthetic_frame(cfg.frame_width, cfg.frame_height)]

    m.convert_ms = measure_convert(frames[0])

    print("正在测试这台电脑认手的速度……", flush=True)
    try:
        if tracker is None:
            tracker = HandTracker(cfg)
        m.tracker_ms = measure_tracker(tracker, frames)
    except FileNotFoundError as e:
        m.tracker_error = str(e)
    except Exception as e:      # e.g. a damaged model file: report it, don't crash
        m.tracker_error = f"没有测到电脑认手的速度：{e}"
    finally:
        if tracker is not None:
            tracker.close()
    return m


def main(argv=None, camera=None, tracker=None) -> int:
    use_utf8_console()
    ap = argparse.ArgumentParser(description="检查 AirTouch 在这台电脑上能跑多快，以及什么地方拖慢了它。", add_help=False)
    ap.add_argument("-h", "--help", action="help", help="显示这份说明")
    ap.add_argument("--no-camera", action="store_true", help="不打开摄像头，只测电脑本身的速度")
    ap.add_argument("--camera", type=int, metavar="N", help="用第几个摄像头（0 是第一个）")
    ap.add_argument("--seconds", type=float, default=5.0, help="摄像头测多少秒（默认 5 秒）")
    ap.add_argument("--report", metavar="FILE", help="把结果另存到这个文件")
    args = ap.parse_args(argv)

    cfg = Config.load()
    if args.camera is not None:
        cfg = dataclasses.replace(cfg, camera_index=args.camera)
    print("AirTouch 速度检查", flush=True)
    m = diagnose(cfg, not args.no_camera, args.seconds, camera, tracker)

    report = ["", *summary(m), "", "—— 结论 ——", *[f"· {line}" for line in verdict(m)]]
    print("\n".join(report), flush=True)
    if args.report:
        try:
            path = Path(args.report)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("\n".join(report).strip() + "\n", encoding="utf-8")
            print(f"\n结果已保存到：{path}", flush=True)
        except OSError as e:
            print(f"\n结果没能保存：{e}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
