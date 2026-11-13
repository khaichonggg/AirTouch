"""Record real hand landmarks from the webcam, to train on real data.

Keys: 1-6 pick the gesture, SPACE starts / pauses recording, S saves, C throws
away what is not saved yet, Q / Esc quits (saving first, so forgetting S loses
nothing).

A frame is kept only while exactly one hand is visible, because with two hands
there is no telling which one the label belongs to. Rows go to
data/recorded.csv in exactly the ``ml.synthetic_data`` format, so
``python -m ml.train_model --data data/recorded.csv`` just works. Only landmark
numbers are stored, never a picture.

The loop and the key handling take a fake camera, tracker and window, so all of
it is tested with no webcam and no real window. ``check_gestures.py`` shares
the window, the camera watchdog and the CSV writer.

CLI:  python -m ml.record_gestures [--out data/recorded.csv]
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from airtouch.config import PROJECT_ROOT, Config
from airtouch.console import use_utf8_console
from airtouch.hud import draw_landmarks
from airtouch.landmarks import is_valid
from ml.synthetic_data import GESTURES, HEADER

RECORDED_CSV = PROJECT_ROOT / "data" / "recorded.csv"
WINDOW_TITLE = "AirTouch recorder"      # English: Windows garbles non-ASCII OpenCV window titles

ESC, SPACE = 27, 32
QUIT_KEYS = (ESC, ord("q"), ord("Q"))
SAVE_KEYS = (ord("s"), ord("S"))
CLEAR_KEYS = (ord("c"), ord("C"))
KEY_LABELS = {ord(str(i + 1)): gesture for i, gesture in enumerate(GESTURES)}   # key "1" -> "point", ...

GESTURE_WORDS = {
    "point": "伸出食指",
    "pinch_index": "拇指碰食指",
    "pinch_middle": "拇指碰中指",
    "scroll": "V 字手势",
    "fist": "握拳",
    "open_palm": "张开手掌",
}

CAMERA_LOST_S = 5.0     # no new frame for this long: the camera is gone
CAMERA_LOST_MESSAGE = (
    "摄像头没有画面了。请看看摄像头有没有插好，或者是不是被别的软件（比如视频会议）占用了，然后再运行一次。"
)


def say(message: str) -> None:
    print(message, flush=True)


# ------------------------------------------------------------------ pure logic
def label_for_key(key: int) -> str | None:
    """The gesture picked by a ``cv2.waitKey`` code, or None for any other key."""
    return KEY_LABELS.get(key)


def append_csv(path: str | Path, hands, labels) -> int:
    """Append rows in exactly the ``save_csv`` format. The header is written only if the file is new."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [[label] + [f"{v:.6f}" for v in np.asarray(hand).reshape(-1)] for hand, label in zip(hands, labels)]
    is_new = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if is_new:
            writer.writerow(HEADER)
        writer.writerows(rows)
    return len(rows)


class SampleBuffer:
    """Samples recorded but not on disk yet."""

    def __init__(self):
        self.hands: list[np.ndarray] = []
        self.labels: list[str] = []

    def __len__(self) -> int:
        return len(self.labels)

    def add(self, hand, label: str) -> None:
        self.hands.append(np.array(hand, dtype=np.float64))
        self.labels.append(label)

    def counts(self) -> Counter:
        return Counter(self.labels)

    def clear(self) -> None:
        self.hands, self.labels = [], []

    def flush(self, path: str | Path) -> int:
        """Append everything to ``path``. The buffer is emptied only if the write worked."""
        n = append_csv(path, self.hands, self.labels)
        self.clear()
        return n


class Recorder:
    """Keyboard state plus the unsaved samples. Knows nothing about cameras or windows."""

    def __init__(self, out: str | Path = RECORDED_CSV, say=say):
        self.out = Path(out)
        self.say = say
        self.label = GESTURES[0]
        self.recording = False
        self.buffer = SampleBuffer()
        self.saved: Counter = Counter()

    def on_key(self, key: int) -> bool:
        """Handle one ``cv2.waitKey`` code. False means time to quit."""
        if key in QUIT_KEYS:
            return False
        label = label_for_key(key)
        if label is not None:
            self.label = label
            self.recording = False      # the hand is about to change pose: those frames would be mislabelled
        elif key == SPACE:
            self.recording = not self.recording
        elif key in SAVE_KEYS:
            self.save()
        elif key in CLEAR_KEYS:
            dropped = len(self.buffer)
            self.buffer.clear()
            self.recording = False
            self.say(f"已丢掉还没保存的 {dropped} 条。")
        return True

    def on_hands(self, hands) -> bool:
        """Offer one frame's detections. True if a sample was kept."""
        if not self.recording or len(hands) != 1 or not is_valid(hands[0]):
            return False
        self.buffer.add(hands[0], self.label)
        return True

    def count(self, label: str) -> int:
        """Samples of ``label`` so far in this session, saved or not."""
        return self.saved[label] + self.buffer.labels.count(label)

    @property
    def saved_total(self) -> int:
        return sum(self.saved.values())

    def save(self) -> int:
        """Write the unsaved samples. A failed write keeps them, so S can be pressed again."""
        if not len(self.buffer):
            return 0
        counts = self.buffer.counts()
        try:
            n = self.buffer.flush(self.out)
        except OSError:
            self.say(f"保存失败，写不进这个文件：{self.out}\n是不是用 Excel 打开着？关掉它再按一次 S。")
            return 0
        self.saved.update(counts)
        self.say(f"已保存 {n} 条到 {self.out}")
        return n


# ---------------------------------------------------------- camera and window
class CameraLost(RuntimeError):
    """The camera stopped delivering pictures."""


class FrameReader:
    """Asks the camera "anything new?" without ever blocking, and notices when it goes silent."""

    def __init__(self, camera, clock=time.monotonic, stall_s: float = CAMERA_LOST_S):
        self.camera = camera
        self.clock = clock
        self.stall_s = stall_s
        self.seq = -1
        self.frames = 0
        self.first_t: float | None = None
        self.last_t = clock()

    def poll(self) -> np.ndarray | None:
        """The next new frame, or None if there is none yet. Raises CameraLost after a long silence."""
        try:
            seq, frame = self.camera.read_if_new(self.seq)
        except Exception as e:
            raise CameraLost(CAMERA_LOST_MESSAGE) from e
        now = self.clock()
        if frame is None:
            if now - self.last_t > self.stall_s:
                raise CameraLost(CAMERA_LOST_MESSAGE)
            return None
        self.seq, self.last_t = seq, now
        self.frames += 1
        if self.first_t is None:
            self.first_t = now
        return frame

    @property
    def fps(self) -> float:
        if self.first_t is None or self.frames < 2 or self.last_t <= self.first_t:
            return 0.0
        return (self.frames - 1) / (self.last_t - self.first_t)


class CvWindow:
    """The preview window. Tests swap in a fake with the same methods, so no real window opens."""

    def __init__(self, title: str = WINDOW_TITLE):
        self.title = title
        cv2.namedWindow(title, cv2.WINDOW_AUTOSIZE)
        try:
            # Opened from a console it can land behind it, where its instructions go unseen.
            cv2.setWindowProperty(title, cv2.WND_PROP_TOPMOST, 1)
        except cv2.error:
            pass

    def show(self, img: np.ndarray) -> None:
        cv2.imshow(self.title, img)

    def poll_key(self, wait_ms: int = 1) -> int:
        """Key code, 255 if none. Also what keeps the window alive: call it on every pass."""
        return cv2.waitKey(wait_ms) & 0xFF

    def closed(self) -> bool:
        # imshow silently re-creates a window closed with its X, so this must be checked.
        try:
            return cv2.getWindowProperty(self.title, cv2.WND_PROP_VISIBLE) < 1
        except cv2.error:
            return True

    def close(self) -> None:
        try:
            cv2.destroyWindow(self.title)
            cv2.waitKey(1)      # lets Windows actually take it off the screen
        except cv2.error:
            pass


# ------------------------------------------------------------------- recording
def _put(img, text, org, scale=0.5, color=(255, 255, 255), thick=1) -> None:
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thick + 2, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


def draw_overlay(img: np.ndarray, recorder: Recorder, n_hands: int) -> np.ndarray:
    """State bar, per-gesture counts and the key map. English, as OpenCV can't draw Chinese."""
    h, w = img.shape[:2]
    cv2.rectangle(img, (0, 0), (w, 34), (25, 25, 25), -1)
    state = "REC" if recorder.recording else "paused"
    color = (60, 60, 255) if recorder.recording else (200, 200, 200)
    _put(img, f"{state}: {recorder.label}", (10, 23), 0.65, color, 2)
    _put(img, f"unsaved {len(recorder.buffer)}  saved {recorder.saved_total}", (max(w - 220, 300), 23))
    if recorder.recording and n_hands != 1:
        _put(img, "Need exactly ONE hand: not recording", (10, 60), 0.6, (0, 200, 255), 2)
    entries = [f"{i + 1} {gesture}: {recorder.count(gesture)}" for i, gesture in enumerate(GESTURES)]
    lines = ("   ".join(entries[:3]), "   ".join(entries[3:]), "SPACE rec/pause   S save   C clear   Q quit")
    for i, line in enumerate(lines):
        _put(img, line, (10, h - 48 + i * 19), 0.45)
    return img


def run_recorder(camera, tracker, cfg: Config, recorder: Recorder, window, clock=time.monotonic) -> None:
    """The camera loop. Polls the keyboard on every pass, frame or no frame, so quitting always works."""
    reader = FrameReader(camera, clock)
    while True:
        frame = reader.poll()
        if frame is not None:
            if cfg.mirror:
                frame = cv2.flip(frame, 1)      # same picture as the live app, so the data matches it
            hands = tracker.detect(frame, int(clock() * 1000))
            kept = recorder.on_hands(hands)
            for hand in hands:
                draw_landmarks(frame, hand, (0, 0, 255) if kept else (0, 220, 0))
            window.show(draw_overlay(frame, recorder, len(hands)))
        key = window.poll_key(1 if frame is not None else 5)
        if not recorder.on_key(key) or window.closed():
            return


def _print_help(out: Path) -> None:
    say("录制手势数据（只保存手指位置的数字，不保存任何照片）")
    say("按数字键选手势：" + "  ".join(f"{i + 1} {GESTURE_WORDS[g]}" for i, g in enumerate(GESTURES)))
    say("空格键：开始 / 暂停录制（画面里只有一只手时才会记录）")
    say("S 保存   C 丢掉还没保存的   Q 或 Esc 退出（退出时会自动保存）")
    say("输入法是中文状态时按键可能没反应，先切换到英文，或者直接点窗口右上角的 ×。")
    say(f"数据会保存到：{out}")


def main(argv=None) -> int:
    use_utf8_console()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(RECORDED_CSV), help="CSV to append to")
    a = ap.parse_args(argv)

    cfg = Config.load()
    recorder = Recorder(a.out)
    camera = tracker = window = None
    code = 0
    try:
        _print_help(recorder.out)
        # Imported here because mediapipe takes seconds to load.
        from airtouch.camera import ThreadedCamera
        from airtouch.download_model import ensure_model
        from airtouch.hand_tracker import HandTracker

        ensure_model()
        # Asked for two hands, or a second hand could never be noticed.
        tracker = HandTracker(cfg, num_hands=2)
        camera = ThreadedCamera(cfg.camera_index, cfg.frame_width, cfg.frame_height, cfg.camera_fps).start()
        size = getattr(camera, "frame_size", None)
        if size is not None and tuple(size) != (cfg.frame_width, cfg.frame_height):
            say(f"注意：摄像头实际画面是 {size[0]}x{size[1]}。训练时请加上 --aspect {size[0] / size[1]:.4f}")
        window = CvWindow()
        run_recorder(camera, tracker, cfg, recorder, window)
    except KeyboardInterrupt:
        pass
    except (RuntimeError, FileNotFoundError) as e:     # these already carry a plain message
        say(str(e))
        code = 1
    except Exception as e:
        say(f"录制时遇到问题，已经停止。问题信息：{type(e).__name__}: {e}")
        code = 1
    finally:
        recorder.save()
        for step in (getattr(camera, "release", None), getattr(tracker, "close", None),
                     getattr(window, "close", None)):
            try:
                if step is not None:
                    step()
            except Exception:
                pass    # keep going: the rest must still be let go of
    say(f"一共保存了 {recorder.saved_total} 条。")
    return code


if __name__ == "__main__":
    sys.exit(main())
