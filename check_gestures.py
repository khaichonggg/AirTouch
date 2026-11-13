"""Guided real-hand self-check: about a minute in front of the webcam.

Everything else in the project is validated on synthetic hands, and in the
original project a real hand was misrecognised in ways that could not be
reproduced headlessly. This tool closes that loop: it walks the person through
the six gestures, measures them with the REAL engine code, prints a plain verdict
and writes the numbers a developer can tune from (logs/hand_check_report.json).

Only landmark numbers are ever kept, never a picture, and the mouse is a fake
one: nothing on the computer gets clicked during the check.

    python check_gestures.py              guided check with the camera
    python check_gestures.py --apply      ...and write suggested settings to config.json
    python check_gestures.py --save       ...and append the landmarks to data/recorded.csv
    python check_gestures.py --no-camera  the whole analysis on synthetic hands (demo / tests)

Pure logic (``analyse``, ``suggest_settings``, ``apply_settings``, ``CheckFlow``)
runs without a camera. The camera path takes injected devices, so tests drive it
with a fake camera, tracker, window and clock.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import sys
import time
import traceback
import unicodedata
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from airtouch import gestures_rule as gr
from airtouch.config import PROJECT_ROOT, Config
from airtouch.console import use_utf8_console
from airtouch.controller import FakeBackend
from airtouch.engine import GestureEngine
from airtouch.hud import blank_canvas, draw_landmarks
from airtouch.landmarks import is_valid, to_metric
from ml.record_gestures import (
    CAMERA_LOST_MESSAGE,
    GESTURE_WORDS,
    QUIT_KEYS,
    RECORDED_CSV,
    SPACE,
    CameraLost,
    CvWindow,
    FrameReader,
    append_csv,
    say,
)
from ml.synthetic_data import make_hand, random_pose

REPORT_PATH = PROJECT_ROOT / "logs" / "hand_check_report.json"
DEMO_REPORT_PATH = PROJECT_ROOT / "logs" / "hand_check_report_demo.json"
ERROR_LOG_PATH = PROJECT_ROOT / "logs" / "hand_check_error.log"
WINDOW_TITLE = "AirTouch hand check"

SEND_ME = "把 logs/hand_check_report.json 发给我就能帮你调准"

# Timing of the guided flow (seconds): 6 x (3 + 3) plus a moment to get the hand up.
COUNTDOWN_S = 3.0
HOLD_S = 3.0
HAND_WAIT_S = 1.5       # a hand steadily in view for this long starts the check by itself...
INTRO_LIMIT_S = 20.0    # ...and it starts anyway after this, so a key lost to the IME can't strand anyone

# Verdicts: "score" is the share of ALL frames where the gesture was in effect correctly.
GOOD, FAIR, BAD = "好", "一般", "不行"
GOOD_SCORE, FAIR_SCORE = 0.90, 0.70
GOOD_FALSE_PRESS, FAIR_FALSE_PRESS = 0.02, 0.10     # a wrong click hurts far more than a missed one
LOW_DETECTION = 0.85

# Settings suggestions, in hand-size units (the same units as pinch_on / pinch_off).
MIN_FRAMES_FOR_ADVICE = 10      # fewer frames than this say nothing about a person's hand
TOUCH_MARGIN = 1.15             # pinch_on should sit this far above the widest normal touch...
REST_MARGIN = 0.85              # ...and this far below the closest a resting pose gets to pressing
MAX_PLAUSIBLE_TOUCH = 0.50      # a wider "touch" means the fingers weren't touching at all
MIN_CHANGE = 0.02               # smaller changes wouldn't be felt
HOLD_RATIO = 1.6                # pinch_off / pinch_on, the proportion of the defaults (0.32 / 0.20)

PINCH_FINGER = {"pinch_index": "index", "pinch_middle": "middle"}
PINCH_BUTTON = {"pinch_index": "left", "pinch_middle": "right"}
FINGER_BUTTON = {"index": "left", "middle": "right"}


@dataclass(frozen=True)
class Step:
    gesture: str
    name: str               # short plain name for the results table
    instruction: str        # big line on the video
    instruction_en: str     # shown only if the computer has no Chinese font


STEPS = (
    Step("point", GESTURE_WORDS["point"], "请伸出食指（其他手指收起）", "Point: index finger up, others curled"),
    Step("pinch_index", GESTURE_WORDS["pinch_index"], "请用拇指碰食指指尖", "Touch thumb to index fingertip"),
    Step("pinch_middle", GESTURE_WORDS["pinch_middle"], "请用拇指碰中指指尖", "Touch thumb to middle fingertip"),
    Step("scroll", GESTURE_WORDS["scroll"], "请伸出食指和中指（V 字）", "V sign: index and middle up"),
    Step("fist", GESTURE_WORDS["fist"], "请握拳", "Make a fist"),
    Step("open_palm", GESTURE_WORDS["open_palm"], "请张开手掌", "Open your hand"),
)


# ---------------------------------------------------------------------- analysis
def _rate(part: int, whole: int) -> float:
    return round(part / whole, 4) if whole else 0.0


def _summary(values: list[float]) -> dict:
    """Median / p10 / p90, rounded: the numbers a threshold gets tuned from."""
    if not values:
        return {"median": None, "p10": None, "p90": None}
    p10, median, p90 = np.percentile(values, [10, 50, 90])
    return {"median": round(float(median), 4), "p10": round(float(p10), 4), "p90": round(float(p90), 4)}


def press_gap(m: np.ndarray) -> float:
    """The pinch threshold at which ``which_pinch`` would fire on this frame: the
    thumb gap of the closest finger allowed to press, or inf if neither may press."""
    gap_index, gap_middle = gr.pinch_gaps(m)
    gaps = []
    if gr.finger_reach(m, "index") > gr.PRESS_REACH["left"]:
        gaps.append(gap_index)
    if gr.finger_reach(m, "middle") > gr.PRESS_REACH["right"]:
        gaps.append(gap_middle)
    return min(gaps, default=math.inf)


def _closest_press_gap(gaps: list[float]) -> float | None:
    """The press gap 95% of the frames stay above. Using a low percentile rather
    than the minimum means one bad frame can't dictate the settings."""
    if not gaps:
        return None
    value = sorted(gaps)[int(0.05 * (len(gaps) - 1))]
    return round(value, 4) if math.isfinite(value) else None


def verdict_for(score: float, false_press: float | None = None) -> str:
    fp = false_press or 0.0
    if score >= GOOD_SCORE and fp <= GOOD_FALSE_PRESS:
        return GOOD
    if score >= FAIR_SCORE and fp <= FAIR_FALSE_PRESS:
        return FAIR
    return BAD


def _analyse_gesture(gesture: str, frames, cfg: Config, fps: float) -> dict:
    """Feed one gesture's frames through the real engine with a fake mouse."""
    engine = GestureEngine(cfg, FakeBackend())
    wanted_button = PINCH_BUTTON.get(gesture)
    is_pinch = gesture in PINCH_FINGER

    raw, debounced, press_gaps = [], [], []
    gaps = {"index": [], "middle": []}
    reach = {"index": [], "middle": []}
    extension = {name: [] for name in ("index", "middle", "ring", "pinky")}
    presses: Counter = Counter()
    fired = held = 0

    for i, hand in enumerate(frames):
        valid = is_valid(hand)
        result = engine.process(hand if valid else None, i / fps)
        presses.update(ev.split(":")[1] for ev in result.events if ev.startswith("down:"))
        if not valid:
            continue
        raw.append(result.raw_label)
        debounced.append(result.label)
        held += wanted_button is not None and result.button == wanted_button
        m = to_metric(np.asarray(hand, dtype=np.float64), cfg.aspect)
        gap_index, gap_middle = gr.pinch_gaps(m)
        gaps["index"].append(gap_index)
        gaps["middle"].append(gap_middle)
        for finger in reach:
            reach[finger].append(gr.finger_reach(m, finger))
        for finger, value in gr.finger_extension(m).items():
            extension[finger].append(value)
        fired += gr.which_pinch(m, cfg.pinch_on) is not None
        press_gaps.append(press_gap(m))

    total, with_hand = len(frames), len(raw)
    # The debouncer needs a few frames to switch on at the start of a hold. That
    # delay is by design, so it must not count against the gesture.
    settled = debounced[cfg.debounce_frames - 1:]
    recognised = _rate(raw.count(gesture), with_hand)
    recognised_debounced = _rate(settled.count(gesture), len(settled))
    detection = _rate(with_hand, total)
    score = round(recognised_debounced * detection, 4)
    false_press = None if is_pinch else _rate(fired, with_hand)
    stats = {f"pinch_gap_{f}": _summary(v) for f, v in gaps.items()}
    stats.update({f"reach_{f}": _summary(v) for f, v in reach.items()})
    stats.update({f"extension_{f}": _summary(v) for f, v in extension.items()})
    return {
        "name": GESTURE_WORDS.get(gesture, gesture),
        "frames": total,
        "frames_with_hand": with_hand,
        "detection_rate": detection,
        "recognised_rate": recognised,
        "recognised_rate_debounced": recognised_debounced,
        "score": score,
        "verdict": verdict_for(score, false_press),
        "seen_as": dict(Counter(raw).most_common()),
        "false_press_rate": false_press,
        "closest_press_gap": None if is_pinch else _closest_press_gap(press_gaps),
        "button": {
            "left_presses": presses["left"],
            "right_presses": presses["right"],
            "held_rate": _rate(held, with_hand) if is_pinch else None,
        },
        "stats": stats,
    }


def suggest_settings(report: dict, cfg: Config) -> dict[str, float]:
    """Suggested ``pinch_on`` / ``pinch_off`` if this hand doesn't fit the current ones, else {}.

    Two measurements bound pinch_on: it must clear the widest normal touch (p90
    of the touching gap in the pinch gestures), and it must stay under the closest
    a non-pinch pose gets to pressing (fingers tucked into the palm can't press,
    so their small gaps don't count). If those two overlap, no single value
    works and a new setting would only trade missed clicks for wrong ones, so
    nothing is suggested and the developer looks at the report instead.
    """
    gestures = report.get("gestures", {})
    touches = [
        gestures[g]["stats"][f"pinch_gap_{finger}"]["p90"]
        for g, finger in PINCH_FINGER.items()
        if g in gestures and gestures[g]["frames_with_hand"] >= MIN_FRAMES_FOR_ADVICE
    ]
    rests = [
        g["closest_press_gap"]
        for name, g in gestures.items()
        if name not in PINCH_FINGER and g["frames_with_hand"] >= MIN_FRAMES_FOR_ADVICE
        and g["closest_press_gap"] is not None
    ]
    touch = max(touches, default=None)
    rest = min(rests, default=None)
    if touch is not None and touch > MAX_PLAUSIBLE_TOUCH:
        touch = None
    if touch is not None and rest is not None and touch >= rest:
        return {}

    wanted = cfg.pinch_on
    floor = touch * TOUCH_MARGIN if touch is not None else None
    ceiling = rest * REST_MARGIN if rest is not None else None
    if floor is not None and ceiling is not None and floor > ceiling:
        wanted = (touch + rest) / 2     # a narrow window: split it evenly
    else:
        if floor is not None:
            wanted = max(wanted, floor)
        if ceiling is not None:
            wanted = min(wanted, ceiling)
    if abs(wanted - cfg.pinch_on) < MIN_CHANGE:
        return {}
    # Config clamps both values into their safe ranges and keeps pinch_off > pinch_on.
    tuned = Config(pinch_on=wanted, pinch_off=max(wanted * HOLD_RATIO, wanted + 0.08))
    return {"pinch_on": round(tuned.pinch_on, 3), "pinch_off": round(tuned.pinch_off, 3)}


def _overall(gestures: dict, complete: bool) -> dict:
    names = {v: [g["name"] for g in gestures.values() if g["verdict"] == v] for v in (GOOD, FAIR, BAD)}
    if names[BAD]:
        verdict, text = BAD, f"「{'」「'.join(names[BAD])}」经常认不准，需要调一调。"
    elif names[FAIR]:
        verdict, text = FAIR, f"基本能用，不过「{'」「'.join(names[FAIR])}」有时会认错。"
    else:
        verdict, text = GOOD, "全部手势都认得很准，可以放心使用。"
    if not complete:
        text = "这次没有做完全部手势，结果只供参考。" + text
    frames = sum(g["frames"] for g in gestures.values())
    with_hand = sum(g["frames_with_hand"] for g in gestures.values())
    resting = [g for g in gestures.values() if g["false_press_rate"] is not None]
    resting_frames = sum(g["frames_with_hand"] for g in resting)
    false_presses = sum(g["false_press_rate"] * g["frames_with_hand"] for g in resting)
    return {
        "verdict": verdict,
        "conclusion": text,
        "detection_rate": _rate(with_hand, frames),
        "false_press_rate": round(false_presses / resting_frames, 4) if resting_frames else None,
        "advice": [],
    }


def _advice(report: dict, cfg: Config) -> list[str]:
    """Plain-language next steps, most useful first."""
    gestures = report["gestures"]
    advice = []
    if report["overall"]["detection_rate"] < LOW_DETECTION:
        advice.append("有不少时候摄像头看不到你的手。请把灯开亮一点，把手放到画面中间，手心对着摄像头。")
    for g, finger in PINCH_FINGER.items():
        info = gestures.get(g)
        median = info["stats"][f"reach_{finger}"]["median"] if info else None
        if median is not None and median < gr.PRESS_REACH[FINGER_BUTTON[finger]]:
            advice.append(f"做「{info['name']}」时那根手指弯得太厉害，程序不敢算作点击。"
                          "捏的时候让手指伸直一点，用指尖去碰。")
    pressed = [g["name"] for g in gestures.values() if (g["false_press_rate"] or 0.0) > GOOD_FALSE_PRESS]
    if pressed:
        advice.append(f"做「{'」「'.join(pressed)}」时，程序有时会以为你在按鼠标。")
    suggested = report.get("suggested_settings") or {}
    if "pinch_on" in suggested:
        before, after = cfg.pinch_on, suggested["pinch_on"]
        why = ("你的手指碰到一起时，程序量到的距离比现在的设置大。" if after > before
               else "你的手没有捏的时候，手指也离得比较近，容易误按。")
        advice.append(f"{why}建议把“算作碰到”的距离从 {before:.2f} 调到 {after:.2f}。")
    return advice


def analyse(samples: dict, cfg: Config | None = None, *, source: str = "unknown",
            fps: float | None = None, frame_size: tuple[int, int] | None = None,
            complete: bool | None = None) -> dict:
    """Turn recorded frames into the report.

    samples: gesture -> list of frames, each a (21, 3) landmark array or None
             when no hand was found (those count in the detection rate only).
    """
    cfg = cfg if cfg is not None else Config()
    fps = fps if fps and fps > 0 else float(cfg.camera_fps)
    ordered = [s.gesture for s in STEPS if s.gesture in samples]
    ordered += [g for g in samples if g not in ordered]
    gestures = {g: _analyse_gesture(g, samples[g], cfg, fps) for g in ordered if len(samples[g])}
    if complete is None:
        complete = all(s.gesture in gestures for s in STEPS)
    report = {
        "version": 1,
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "source": source,
        "complete": complete,
        "contains": "numbers only: no pictures and no raw hand positions",
        "camera": {
            "frame_size": list(frame_size) if frame_size else [cfg.frame_width, cfg.frame_height],
            "frames_per_second": round(float(fps), 1),
        },
        "config": dataclasses.asdict(cfg),
        "thresholds": {
            "pinch_on": cfg.pinch_on,
            "pinch_off": cfg.pinch_off,
            "debounce_frames": cfg.debounce_frames,
            "extended_ratio": gr.EXTENDED_RATIO,
            "press_reach": dict(gr.PRESS_REACH),
            "hold_reach": dict(gr.HOLD_REACH),
            "verdict_good_score": GOOD_SCORE,
            "verdict_fair_score": FAIR_SCORE,
            "verdict_good_false_press": GOOD_FALSE_PRESS,
            "verdict_fair_false_press": FAIR_FALSE_PRESS,
        },
        "gestures": gestures,
    }
    report["overall"] = _overall(gestures, complete)
    report["suggested_settings"] = suggest_settings(report, cfg)
    report["overall"]["advice"] = _advice(report, cfg)
    return report


def write_report(report: dict, path: str | Path = REPORT_PATH) -> Path:
    """Strict JSON, written via a temporary name so a crash can't leave half a report."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".part")
    tmp.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(path)
    return path


def apply_settings(suggested: dict, path: str | Path) -> dict:
    """Merge ``suggested`` into config.json, keeping every other key (even ones this
    version doesn't know). Returns what the file now holds."""
    path = Path(path)
    data: dict = {}
    if path.exists():
        try:
            # utf-8-sig: some Windows editors add a BOM, and that must not look like corruption
            loaded = json.loads(path.read_text(encoding="utf-8-sig"))
        except ValueError:
            loaded = None
        if isinstance(loaded, dict):
            data = loaded
        else:
            path.replace(path.with_name(path.name + ".bak"))    # unusable, but not ours to destroy
    data.update(suggested)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".part")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp.replace(path)
    return data


def save_samples(samples: dict, path: str | Path = RECORDED_CSV) -> int:
    """Append every frame that has a hand to the standard training CSV (numbers only)."""
    hands, labels = [], []
    for gesture, frames in samples.items():
        for hand in frames:
            if is_valid(hand):
                hands.append(hand)
                labels.append(gesture)
    return append_csv(path, hands, labels) if hands else 0


def _display_width(text: str) -> int:
    return sum(2 if unicodedata.east_asian_width(ch) in "WF" else 1 for ch in text)


def _pad(text: str, width: int) -> str:
    return text + " " * max(width - _display_width(text), 0)


def _percent(value: float | None) -> str:
    return "-" if value is None else f"{value * 100:.0f}%"


def format_results(report: dict) -> str:
    """The plain verdict table and conclusion for the console."""
    rows = [("手势", "结果", "认对了", "看到手", "误按")]
    for g in report["gestures"].values():
        rows.append((g["name"], g["verdict"], _percent(g["recognised_rate_debounced"]),
                     _percent(g["detection_rate"]), _percent(g["false_press_rate"])))
    widths = [max(_display_width(row[i]) for row in rows) + 3 for i in range(len(rows[0]))]
    lines = ["".join(_pad(cell, width) for cell, width in zip(row, widths)).rstrip() for row in rows]
    lines.insert(1, "-" * sum(widths))
    lines += ["", f"总结：{report['overall']['conclusion']}"]
    lines += [f"  · {tip}" for tip in report["overall"]["advice"]]
    return "\n".join(lines)


# ---------------------------------------------------------------- synthetic demo
def synthetic_samples(frames: int = 90, seed: int = 0, aspect: float = 4 / 3,
                      dropout: float = 0.02) -> dict:
    """A believable check without a camera: each gesture is one held pose with a
    little tracker jitter, and the tracker misses the odd frame."""
    rng = np.random.default_rng(seed)
    samples = {}
    for step in STEPS:
        held = make_hand(step.gesture, rng, aspect=aspect, pose=random_pose(rng), noise=0.0)
        samples[step.gesture] = [
            None if rng.random() < dropout else held + rng.normal(0.0, 0.0015, held.shape)
            for _ in range(frames)
        ]
    return samples


# ---------------------------------------------------------------- the guided flow
class CheckFlow:
    """What the check is doing right now: intro -> (countdown -> hold) x 6 -> done.
    Driven by the clock and fed one hand per camera frame; knows nothing about
    cameras or drawing."""

    def __init__(self, steps=STEPS, countdown_s: float = COUNTDOWN_S, hold_s: float = HOLD_S,
                 hand_wait_s: float = HAND_WAIT_S, intro_limit_s: float = INTRO_LIMIT_S):
        self.steps = tuple(steps)
        self.countdown_s = countdown_s
        self.hold_s = hold_s
        self.hand_wait_s = hand_wait_s
        self.intro_limit_s = intro_limit_s
        self.samples: dict[str, list] = {s.gesture: [] for s in self.steps}
        self.phase = "intro"
        self.index = 0
        self._phase_t: float | None = None
        self._hand_t: float | None = None

    @property
    def done(self) -> bool:
        return self.phase == "done"

    @property
    def step(self) -> Step:
        return self.steps[min(self.index, len(self.steps) - 1)]

    def _enter(self, phase: str, now: float) -> None:
        self.phase, self._phase_t = phase, now

    def update(self, now: float, hand, start: bool = False) -> None:
        if self._phase_t is None:
            self._phase_t = now
        elapsed = now - self._phase_t
        if self.phase == "intro":
            if hand is None:
                self._hand_t = None
            elif self._hand_t is None:
                self._hand_t = now
            steady = self._hand_t is not None and now - self._hand_t >= self.hand_wait_s
            if start or steady or elapsed >= self.intro_limit_s:
                self._enter("countdown", now)
        elif self.phase == "countdown":
            if elapsed >= self.countdown_s:
                self._enter("hold", now)
        elif self.phase == "hold":
            if elapsed >= self.hold_s:
                self.index += 1
                self._enter("done" if self.index >= len(self.steps) else "countdown", now)
            else:
                self.samples[self.step.gesture].append(hand)

    def seconds_left(self, now: float) -> float:
        limit = self.countdown_s if self.phase == "countdown" else self.hold_s
        return max(0.0, limit - (now - (self._phase_t if self._phase_t is not None else now)))

    def progress(self, now: float) -> float:
        return min(1.0, 1.0 - self.seconds_left(now) / self.hold_s) if self.hold_s > 0 else 1.0


# ------------------------------------------------------------------ drawing
WHITE, GREY, YELLOW, GREEN = (255, 255, 255), (200, 200, 200), (60, 220, 255), (90, 220, 90)  # BGR
HAND_COLOR = (0, 220, 0)
CJK_FONTS = (
    r"C:\Windows\Fonts\msyh.ttc",
    r"C:\Windows\Fonts\msyhbd.ttc",
    r"C:\Windows\Fonts\simhei.ttf",
    r"C:\Windows\Fonts\simsun.ttc",
    "/System/Library/Fonts/PingFang.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
)

# (Chinese, English): the English is drawn only if no Chinese font exists.
TEXT_WAIT = ("正在准备，请稍等几秒...", "Getting ready...")
TEXT_TITLE = ("手势检查，大约一分钟", "Hand check: about one minute")
TEXT_RAISE = ("请把手举到摄像头前，手心对着镜头", "Hold up your hand, palm to the camera")
TEXT_NO_HAND_YET = ("还没看到手，放到画面中间试试", "No hand yet: try the middle of the picture")
TEXT_HAND_SEEN = ("看到手了，马上开始", "Hand found: starting soon")
TEXT_GET_READY = ("摆好手势，倒数结束后保持不动", "Get ready, then hold still")
TEXT_HOLD = ("保持不动...", "Hold still...")
TEXT_LOST = ("没看到手，请把手放回画面中间", "Hand lost: back to the middle please")
TEXT_QUIT = ("按 Q 或 Esc 退出，也可以点窗口的 ×", "Q or Esc to quit")


def find_font(candidates=CJK_FONTS) -> str | None:
    for path in candidates:
        try:
            ImageFont.truetype(path, 16)
            return path
        except OSError:
            continue
    return None


class Painter:
    """Big simple-Chinese text on video frames.

    ``cv2.putText`` can't draw Chinese, so PIL renders each distinct string once;
    after that it is only a numpy blend, which keeps the camera loop fast.
    """

    def __init__(self, fonts=CJK_FONTS):
        self.font_path = find_font(fonts)
        self._cache: dict = {}

    def line(self, img: np.ndarray, text: tuple[str, str], y: float, size: float, color=WHITE) -> None:
        """One centred line. ``y`` (its top) and ``size`` are fractions of the picture height."""
        h, w = img.shape[:2]
        px, top = max(12, int(size * h)), int(y * h)
        if self.font_path is None:
            self._latin(img, text[1], top, px, color)
            return
        bgr, alpha = self._patch(text[0], px, color, int(w * 0.96))
        rows = min(alpha.shape[0], h - top)
        if rows <= 0:
            return
        left = max((w - alpha.shape[1]) // 2, 0)
        roi = img[top:top + rows, left:left + alpha.shape[1]]
        a = alpha[:rows]
        roi[:] = (a * bgr[:rows] + (1.0 - a) * roi).astype(np.uint8)

    def _patch(self, text: str, px: int, color, max_width: int):
        key = (text, px, color, max_width)
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        font = ImageFont.truetype(self.font_path, px)
        width = font.getlength(text)
        if width > max_width:       # too long for this picture: shrink rather than clip
            font = ImageFont.truetype(self.font_path, max(10, int(px * max_width / width)))
        stroke = max(1, font.size // 14)
        left, top, right, bottom = font.getbbox(text, stroke_width=stroke)
        canvas = Image.new("RGBA", (max(right - left, 1), max(bottom - top, 1)), (0, 0, 0, 0))
        ImageDraw.Draw(canvas).text((-left, -top), text, font=font, fill=(*color[::-1], 255),
                                    stroke_width=stroke, stroke_fill=(0, 0, 0, 255))
        rgba = np.asarray(canvas, dtype=np.float32)
        patch = rgba[..., 2::-1], rgba[..., 3:] / 255.0     # BGR, alpha
        self._cache[key] = patch
        return patch

    @staticmethod
    def _latin(img: np.ndarray, text: str, top: int, px: int, color) -> None:
        h, w = img.shape[:2]
        font, scale = cv2.FONT_HERSHEY_SIMPLEX, px / 30.0
        (width, height), _ = cv2.getTextSize(text, font, scale, 2)
        if width > w * 0.96:
            scale *= w * 0.96 / width
            (width, height), _ = cv2.getTextSize(text, font, scale, 2)
        org = ((w - width) // 2, min(top + height, h - 1))
        cv2.putText(img, text, org, font, scale, (0, 0, 0), 5, cv2.LINE_AA)
        cv2.putText(img, text, org, font, scale, color, 2, cv2.LINE_AA)


def _shade(img: np.ndarray, top: float, bottom: float) -> None:
    """Darken a band so the text on it stays readable over any video."""
    h = img.shape[0]
    band = img[int(top * h):int(bottom * h)]
    band[:] = (band * 0.4).astype(np.uint8)


def render(img: np.ndarray, flow: CheckFlow, hand, painter: Painter, now: float) -> np.ndarray:
    """Draw the current instruction on the frame, in place."""
    h, w = img.shape[:2]
    if hand is not None:
        draw_landmarks(img, hand, HAND_COLOR)
    _shade(img, 0.0, 0.30)
    _shade(img, 0.92, 1.0)
    painter.line(img, TEXT_QUIT, 0.935, 0.04, GREY)
    if flow.phase == "intro":
        painter.line(img, TEXT_TITLE, 0.03, 0.085)
        painter.line(img, TEXT_RAISE, 0.135, 0.055)
        painter.line(img, TEXT_NO_HAND_YET if hand is None else TEXT_HAND_SEEN, 0.21, 0.05,
                     YELLOW if hand is None else GREEN)
        return img
    number, total = flow.index + 1, len(flow.steps)
    painter.line(img, (f"第 {number} 步，共 {total} 步", f"Step {number} of {total}"), 0.02, 0.045, GREY)
    painter.line(img, (flow.step.instruction, flow.step.instruction_en), 0.085, 0.085)
    if flow.phase == "countdown":
        painter.line(img, TEXT_GET_READY, 0.205, 0.05, YELLOW)
        digit = str(max(1, math.ceil(flow.seconds_left(now))))
        painter.line(img, (digit, digit), 0.38, 0.32, YELLOW)
    else:
        painter.line(img, TEXT_HOLD if hand is not None else TEXT_LOST, 0.205, 0.05,
                     GREEN if hand is not None else YELLOW)
        cv2.rectangle(img, (0, int(0.90 * h)), (int(w * flow.progress(now)), int(0.92 * h)), GREEN, -1)
    return img


# ---------------------------------------------------------------- the camera path
@dataclass
class SessionResult:
    samples: dict
    cfg: Config
    complete: bool = False
    stop_reason: str = ""       # "" (finished), "quit", "window" or "camera"
    fps: float = 0.0
    frame_size: tuple[int, int] | None = None


def run_session(camera, tracker, cfg: Config, window, painter: Painter | None = None,
                flow: CheckFlow | None = None, clock=time.monotonic) -> SessionResult:
    """The camera loop. Polls the keyboard on every pass, frame or no frame, so
    quitting always works; records only landmark numbers."""
    painter = painter if painter is not None else Painter()
    flow = flow if flow is not None else CheckFlow()
    reader = FrameReader(camera, clock)
    result = SessionResult(samples=flow.samples, cfg=cfg)
    start = False
    while not flow.done:
        try:
            frame = reader.poll()
        except CameraLost:
            result.stop_reason = "camera"
            break
        if frame is not None:
            frame = cv2.flip(frame, 1) if cfg.mirror else frame     # same picture as the live app
            result.frame_size = (frame.shape[1], frame.shape[0])
            now = clock()
            hands = tracker.detect(frame, int(now * 1000))
            hand = hands[0] if len(hands) else None
            flow.update(now, hand, start)
            start = False
            if flow.done:
                break
            window.show(render(frame, flow, hand, painter, now))
        key = window.poll_key(1 if frame is not None else 5)
        if key in QUIT_KEYS:
            result.stop_reason = "quit"
            break
        if key == SPACE:
            start = True
        if window.closed():
            result.stop_reason = "window"
            break
    result.complete = flow.done
    result.fps = reader.fps
    return result


def _open_real_tracker(cfg: Config):
    # Imported here because mediapipe takes seconds to load.
    from airtouch.download_model import ensure_model
    from airtouch.hand_tracker import HandTracker

    ensure_model()
    return HandTracker(cfg, num_hands=1)


def _open_real_camera(cfg: Config):
    from airtouch.camera import ThreadedCamera

    return ThreadedCamera(cfg.camera_index, cfg.frame_width, cfg.frame_height, cfg.camera_fps).start()


def run_camera_check(cfg: Config, open_camera=None, open_tracker=None, open_window=None,
                     clock=time.monotonic, flow: CheckFlow | None = None) -> SessionResult:
    """Open the devices, run the session and ALWAYS let go of everything again."""
    window = tracker = camera = None
    painter = Painter()
    try:
        window = (open_window or CvWindow)(WINDOW_TITLE)
        splash = blank_canvas(cfg.frame_width, cfg.frame_height)
        painter.line(splash, TEXT_WAIT, 0.44, 0.07)
        window.show(splash)
        window.poll_key(1)      # paints it: the steps below can take a few seconds
        # The tracker first: if the model is broken the camera light never comes on.
        tracker = (open_tracker or _open_real_tracker)(cfg)
        camera = (open_camera or _open_real_camera)(cfg)
        # Every hand measurement depends on the picture's real shape, and a camera
        # may ignore the size it was asked for. The live app corrects for this too.
        size = getattr(camera, "frame_size", None)
        if size is not None and tuple(size) != (cfg.frame_width, cfg.frame_height):
            cfg = dataclasses.replace(cfg, frame_width=size[0], frame_height=size[1])
        return run_session(camera, tracker, cfg, window, painter, flow, clock)
    finally:
        for part, method in ((camera, "release"), (tracker, "close"), (window, "close")):
            if part is not None:
                try:
                    getattr(part, method)()
                except Exception:
                    pass    # keep going: the rest must still be let go of


# ---------------------------------------------------------------------- the CLI
def _has_chinese(text: str) -> bool:
    return any("\u4e00" <= ch <= "\u9fff" for ch in text)


def _holds_real_report(path: Path) -> bool:
    try:
        return json.loads(path.read_text(encoding="utf-8")).get("source") == "camera"
    except (OSError, ValueError, AttributeError):
        return False


def _ask_yes(question: str) -> bool:
    if sys.stdin is None or not sys.stdin.isatty():
        return False
    try:
        return input(question).strip().lower() in ("y", "yes", "是", "好")
    except (EOFError, OSError):
        return False


def _offer_settings(report: dict, want_apply: bool, config_path: Path) -> None:
    suggested = report["suggested_settings"]
    if not suggested:
        if want_apply:
            say("现在的设置和你的手很合适，不需要改。")
        return
    if not report["complete"]:
        say("这次没有做完全部手势，先不改设置。做完一整遍再说。")
        return
    if want_apply or _ask_yes("要我现在帮你把设置改好吗？输入 y 再按回车（直接按回车就是不改）："):
        apply_settings(suggested, config_path)
        say(f"已经改好了（{config_path}）。下次启动 AirTouch 就会用新的设置。")
    else:
        say("设置没有改。想让程序自动改好，下次运行时加上 --apply。")


def _run(a, open_camera, open_tracker, open_window, clock, flow) -> int:
    config_path = Path(a.config) if a.config else PROJECT_ROOT / "config.json"
    cfg = Config.load(config_path)
    demo = a.no_camera
    if demo:
        say("这是演示：用的是程序生成的假手数据，没有打开摄像头。")
        session = SessionResult(samples=synthetic_samples(seed=a.seed, aspect=cfg.aspect), cfg=cfg,
                                complete=True, fps=float(cfg.camera_fps),
                                frame_size=(cfg.frame_width, cfg.frame_height))
    else:
        say("手势检查大约需要一分钟。只记录手指位置的数字，不保存任何照片，也不会动你的鼠标。")
        say("退出：按 Q 或 Esc，或者点窗口右上角的 ×（输入法是中文状态时按键可能没反应，点 × 就行）。")
        session = run_camera_check(cfg, open_camera, open_tracker, open_window, clock, flow)

    code = 0
    if session.stop_reason == "camera":
        say(CAMERA_LOST_MESSAGE)
        code = 1
    if not any(len(frames) for frames in session.samples.values()):
        if session.stop_reason in ("quit", "window"):
            say("已退出。这次还没有录到手势，所以没有生成报告。")
        return code

    report = analyse(session.samples, session.cfg, source="synthetic" if demo else "camera",
                     fps=session.fps, frame_size=session.frame_size, complete=session.complete)
    say("")
    say(format_results(report))
    say("")

    if a.report:
        report_path = Path(a.report)
    elif demo and _holds_real_report(REPORT_PATH):     # a demo must never overwrite real-hand numbers
        report_path = DEMO_REPORT_PATH
    else:
        report_path = REPORT_PATH
    write_report(report, report_path)
    say(f"报告已保存：{report_path}（里面只有数字，没有任何照片）")

    if demo:
        # Made-up hands say nothing about this person's hand or real data.
        if a.apply or a.save:
            say("演示模式用的是假手的数据，所以不会改你的设置，也不会保存数据。")
        return code
    if a.save:
        say(f"已把 {save_samples(session.samples)} 条手指位置数字加到 {RECORDED_CSV}")
    _offer_settings(report, a.apply, config_path)
    say(SEND_ME if report_path == REPORT_PATH else f"把 {report_path} 发给我就能帮你调准")
    return code


def _log_crash() -> Path | None:
    try:
        ERROR_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with ERROR_LOG_PATH.open("a", encoding="utf-8") as f:
            f.write(f"--- {datetime.now().isoformat(timespec='seconds')}\n{traceback.format_exc()}\n")
        return ERROR_LOG_PATH
    except OSError:
        return None


def main(argv=None, open_camera=None, open_tracker=None, open_window=None,
         clock=time.monotonic, flow: CheckFlow | None = None) -> int:
    """Returns the exit code. Whatever goes wrong, the person sees a plain message, never a traceback."""
    use_utf8_console()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--apply", action="store_true", help="write the suggested pinch settings into config.json")
    ap.add_argument("--save", action="store_true", help="append the recorded landmarks to data/recorded.csv")
    ap.add_argument("--no-camera", action="store_true", help="run on synthetic hands, without the camera")
    ap.add_argument("--report", help="where to write the report (default: logs/hand_check_report.json)")
    ap.add_argument("--config", help="config file to read and to --apply to (default: config.json)")
    ap.add_argument("--seed", type=int, default=0, help="random seed for --no-camera")
    a = ap.parse_args(argv)
    try:
        return _run(a, open_camera, open_tracker, open_window, clock, flow)
    except KeyboardInterrupt:
        say("已退出。")
        return 0
    except (RuntimeError, FileNotFoundError) as e:
        if _has_chinese(str(e)):    # the camera and tracker explain themselves in plain words
            say(str(e))
            return 1
        logged = _log_crash()
    except Exception:
        logged = _log_crash()
    say("检查时遇到问题，已经停止。" + (f"把 {logged} 发给我就能帮你看。" if logged else ""))
    return 1


if __name__ == "__main__":
    sys.exit(main())
