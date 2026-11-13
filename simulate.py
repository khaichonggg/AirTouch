"""Headless virtual test: scripted synthetic hands drive the REAL GestureEngine.

No camera and no real mouse are involved: every action goes to a FakeBackend,
so this is safe to run while someone is using the computer. Each scenario
scripts a hand frame by frame and checks what the engine did on every frame:

    move, left_click, drag, right_click, scroll_up, scroll_down, pause_fist,
    idle_open_palm, hand_lost, short_dropout, zoom_in, zoom_out, dwell_click

``research/stress_test.py`` builds on the same ``Hand`` and ``Session`` helpers.

Usage:
    python simulate.py                  all scenarios, rule engine then ML engine
    python simulate.py --quiet          only the result table
    python simulate.py --engine rule    skip the ML engine
    python simulate.py --no-render      do not write images/sim_*.png

The exit code is 1 if any check failed.
"""

from __future__ import annotations

import argparse
import math
import sys
import traceback
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from airtouch import gestures_rule as gr
from airtouch.config import GESTURE_MODEL_PATH, PROJECT_ROOT, Config
from airtouch.console import use_utf8_console
from airtouch.engine import FrameResult, GestureEngine
from airtouch.controller import FakeBackend
from airtouch.gesture_classifier import GestureClassifier
from airtouch.hud import blank_canvas, draw_hud, save_png
from airtouch.landmarks import anchor_point, to_metric
from ml.synthetic_data import SPEC, HandPose, make_hand, random_pose

FPS = 30
DT = 1.0 / FPS
ASPECT = Config().aspect
JITTER = 0.0015         # camera noise per landmark, normalised units (about 1 px at 640x480)
IMAGES_DIR = PROJECT_ROOT / "images"
FINGER_NAMES = ("index", "middle", "ring", "pinky")
PINCH_GESTURE = {"left": "pinch_index", "right": "pinch_middle"}

# Used only to look at the thumb's approach direction in the hand's own coordinates.
_HAND_FRAME = HandPose(cx=0.0, cy=0.0, size=1.0)
_NOMINAL_CURLS = dict(index=0.45, middle=0.45, ring=0.7, pinky=0.7)


# ------------------------------------------------------------ scripted hands
def sample_curls(gesture: str, rng: np.random.Generator) -> dict[str, float]:
    """One finger shape for a gesture, picked once and then held for a whole trial."""
    spec = SPEC[gesture]
    return {f: float(rng.uniform(*spec[f][rng.integers(len(spec[f]))])) for f in FINGER_NAMES}


def ease(s: float) -> float:
    """Smoothstep: a hand speeds up and slows down instead of moving at constant speed."""
    return s * s * (3.0 - 2.0 * s)


def _unit(v: np.ndarray) -> np.ndarray:
    return v / max(float(np.linalg.norm(v)), 1e-9)


def _thumb_seed(rng: np.random.Generator) -> int:
    """A ``make_hand`` seed whose thumb opens away from the fingertip like a real one.

    ``make_hand`` offsets the thumb from the fingertip along a random 3-D
    direction. Held for a whole trial, a direction straight at the camera would
    be invisible to any 2-D tracker, and one pointing across the hand would walk
    the thumb through the other fingers. Real thumbs open towards their own base.
    """
    while True:
        seed = int(rng.integers(1 << 31))
        for gesture, tip in (("pinch_index", 8), ("pinch_middle", 12)):
            lm = make_hand(gesture, np.random.default_rng(seed), pose=_HAND_FRAME, curls=_NOMINAL_CURLS,
                           thumb_gap=1.0, noise=0.0, aspect=1.0)
            away = lm[4] - lm[tip]
            if float(_unit(away) @ _unit(lm[2] - lm[tip])) < 0.5 or float(np.linalg.norm(away[:2])) < 0.7:
                break
        else:
            return seed


class Hand:
    """One person's hand during one trial: noise-free shapes + per-frame jitter.

    ``make_hand`` re-draws the bone noise, the curls and the thumb direction on
    every call, which would make a held pose twitch far more than a real one.
    Re-seeding it with the trial's seed and passing fixed curls yields the same
    bones and thumb direction on every call and for every gesture, so shapes can
    be blended into each other.
    """

    def __init__(self, rng: np.random.Generator, pose: HandPose | None = None, jitter: float = JITTER):
        self.rng = rng
        self.pose = pose if pose is not None else random_pose(rng, "normal")
        self.jitter = jitter
        self.seed = _thumb_seed(rng)
        self.curls = {g: sample_curls(g, rng) for g in SPEC}
        self._gap_scale: float | None = None

    # -- noise-free shapes -----------------------------------------------------
    def shape(self, gesture: str, curls: dict | None = None, thumb_gap: float | None = None) -> np.ndarray:
        return make_hand(gesture, np.random.default_rng(self.seed), pose=self.pose, aspect=ASPECT,
                         curls=curls if curls is not None else self.curls[gesture],
                         thumb_gap=thumb_gap, noise=0.0)

    def pinch(self, button: str, gap: float, curls: dict | None = None) -> np.ndarray:
        """Pinch shape whose gap is ``gap`` as the controller measures it: 2-D, in
        hand-size units, so ``gap=0.20`` sits exactly on ``pinch_on``."""
        gesture = PINCH_GESTURE[button]
        if self._gap_scale is None:
            m = to_metric(self.shape(gesture, curls, thumb_gap=1.0), ASPECT)
            self._gap_scale = gr.pinch_gap(m, button)
        return self.shape(gesture, curls, thumb_gap=gap / self._gap_scale)

    # -- frames ----------------------------------------------------------------
    def noisy(self, shape: np.ndarray) -> np.ndarray:
        return shape + self.rng.normal(0.0, self.jitter, shape.shape)

    def hold(self, shape: np.ndarray, n: int) -> list[np.ndarray]:
        return [self.noisy(shape) for _ in range(n)]

    def blend(self, a: np.ndarray, b: np.ndarray, n: int) -> list[np.ndarray]:
        """``n`` in-between frames, excluding both end shapes."""
        return [self.noisy(a + (b - a) * ease((i + 1) / (n + 1))) for i in range(n)]

    def travel(self, shape: np.ndarray, dx: float, dy: float, n: int) -> list[np.ndarray]:
        """The whole hand moves by (dx, dy) over ``n`` frames, ending exactly there."""
        return [self.noisy(moved(shape, dx * ease((i + 1) / n), dy * ease((i + 1) / n))) for i in range(n)]

    def gap_path(self, button: str, start: float, end: float, n: int, curls: dict | None = None) -> list[np.ndarray]:
        """The thumb goes from gap ``start`` to ``end`` over ``n`` frames; fingers stay put."""
        return [self.noisy(self.pinch(button, start + (end - start) * ease((i + 1) / n), curls)) for i in range(n)]


def moved(shape: np.ndarray, dx: float, dy: float) -> np.ndarray:
    return shape + np.array([dx, dy, 0.0])


def shifted(frames: list[np.ndarray], dx: float, dy: float) -> list[np.ndarray]:
    return [moved(f, dx, dy) for f in frames]


def measured_gap(lm: np.ndarray, button: str) -> float:
    """The thumb-finger gap exactly as the controller sees it on this frame."""
    return gr.pinch_gap(to_metric(lm, ASPECT), button)


def touching(lm: np.ndarray, button: str, cfg: Config) -> bool:
    """True if this frame is a touch the controller is allowed to press on."""
    return gr.which_pinch(to_metric(lm, ASPECT), cfg.pinch_on) == button


# ------------------------------------------------------------------ sessions
class Session:
    """One engine with a FakeBackend and a clock, plus a log of every frame."""

    def __init__(self, recognizer=None, cfg: Config | None = None):
        self.cfg = cfg or Config()
        self.backend = FakeBackend()
        self.engine = GestureEngine(self.cfg, self.backend, recognizer)
        self.t = 0.0
        self.frames: list = []
        self.results: list[FrameResult] = []

    def feed(self, frame) -> FrameResult:
        """``frame``: a (21, 3) hand, ``None`` for no hand, or a tuple of two hands."""
        if isinstance(frame, tuple):
            res = self.engine.process_hands(list(frame), self.t)
        else:
            res = self.engine.process(frame, self.t)
        self.frames.append(frame)
        self.results.append(res)
        self.t += DT
        return res

    def play(self, frames) -> list[FrameResult]:
        return [self.feed(f) for f in frames]

    # -- what happened ---------------------------------------------------------
    def event_frames(self, event: str, start: int = 0, stop: int | None = None) -> list[int]:
        """Frame numbers whose events include ``event`` (exact string)."""
        return [i for i in range(start, len(self.results) if stop is None else stop)
                if event in self.results[i].events]

    def events(self, start: int = 0, stop: int | None = None) -> list[str]:
        return [e for r in self.results[start:stop] for e in r.events]

    def notches(self, start: int = 0, stop: int | None = None) -> list[float]:
        """Wheel notches sent in the frame range, read back from the event strings."""
        return [float(e.split(":")[1]) for e in self.events(start, stop) if e.startswith("scroll:")]

    def first_touch(self, button: str, start: int = 0) -> int | None:
        """First frame the live geometry says the thumb touches this button's finger."""
        for i in range(start, len(self.frames)):
            f = self.frames[i]
            if isinstance(f, np.ndarray) and touching(f, button, self.cfg):
                return i
        return None

    def expected_cursor(self, frame: np.ndarray) -> tuple[int, int]:
        """Where the cursor belongs for this hand, with no smoothing at all."""
        ax, ay = anchor_point(frame)
        return self.engine.controller.to_screen(float(ax), float(ay))

    def cursor_path(self, start: int, stop: int | None = None) -> np.ndarray:
        return np.array([r.cursor for r in self.results[start:stop]], dtype=np.float64)

    def log_lines(self) -> list[str]:
        return [f"  t={r.t:6.3f} raw={r.raw_label:<12} label={r.label:<12} cursor={str(r.cursor):<13} "
                f"button={str(r.button):<5} {' '.join(r.events) or '-'}" for r in self.results]


# ------------------------------------------------------------------ scenarios
@dataclass
class Check:
    what: str
    ok: bool
    info: str = ""


@dataclass
class Ctx:
    """What a scenario needs: a seeded hand, sessions, checks and key frames to render."""

    name: str
    recognizer: object = None
    seed: int = 7
    checks: list[Check] = field(default_factory=list)
    sessions: list[Session] = field(default_factory=list)
    snaps: list[tuple] = field(default_factory=list)

    def hand(self, **pose) -> Hand:
        rng = np.random.default_rng(self.seed)
        return Hand(rng, HandPose(**{"cx": 0.5, "cy": 0.55, "size": 0.16, "roll": 8.0, **pose}))

    def session(self, **cfg) -> Session:
        s = Session(self.recognizer, Config(**cfg))
        self.sessions.append(s)
        return s

    def check(self, what: str, ok: bool, info: str = "") -> None:
        self.checks.append(Check(what, bool(ok), info))

    def snap(self, s: Session, tag: str, frame_no: int | None = None) -> None:
        self.snaps.append((tag, s, len(s.results) - 1 if frame_no is None else frame_no))


def _check_one_click(x: Ctx, s: Session, button: str) -> tuple[int | None, int | None]:
    other = "right" if button == "left" else "left"
    downs, ups = s.event_frames(f"down:{button}"), s.event_frames(f"up:{button}")
    x.check(f"恰好按下并松开一次 {button} 键", len(downs) == 1 and len(ups) == 1, f"按下帧 {downs}, 松开帧 {ups}")
    x.check(f"没有碰到 {other} 键", not any(other in e for e in s.events()), " ".join(s.events()))
    x.check("结束时没有按键卡住", not s.backend.down, str(s.backend.down))
    return (downs[0] if downs else None), (ups[0] if ups else None)


def _expected_release(s: Session, button: str, down: int) -> int | None:
    """The frame on which the click model says the button must come up."""
    loose = 0
    for i in range(down + 1, len(s.frames)):
        gap = measured_gap(s.frames[i], button)
        loose = loose + 1 if gap > s.cfg.pinch_on else 0
        if gap >= s.cfg.pinch_off or loose >= s.cfg.debounce_frames:
            return i
    return None


def _check_trip(x: Ctx, s: Session, name: str, start: int, stop: int, want: np.ndarray) -> float:
    """One straight cursor trip: right direction, right distance, no back-and-forth."""
    path = s.cursor_path(start, stop)
    got = path[-1] - path[0]
    dist, want_dist = float(np.linalg.norm(got)), float(np.linalg.norm(want))
    u = want / want_dist
    steps = np.diff(path, axis=0) @ u
    backwards = float(-steps[steps < 0].sum())
    x.check(f"{name}: 光标方向正确", float(got @ u) > 0.98 * dist, f"实际 {got}, 应为 {want}")
    x.check(f"{name}: 光标距离正确", abs(float(got @ u) - want_dist) < 0.05 * want_dist,
            f"实际 {dist:.0f} 像素, 应为 {want_dist:.0f} 像素")
    x.check(f"{name}: 光标基本不往回走", backwards < 0.05 * dist, f"往回 {backwards:.1f} / {dist:.0f} 像素")
    return dist


def scn_move(x: Ctx) -> None:
    hand = x.hand(cx=0.32, cy=0.64)
    shape = hand.shape("point")
    s = x.session()
    s.play(hand.hold(shape, 10))
    x.check("手势在第 3 帧确认为 point", s.results[2].label == "point", s.results[2].label)
    x.check("从第 3 帧起光标跟手", s.results[2].cursor is not None, str(s.results[2].cursor))
    x.snap(s, "start")

    speeds = {}
    for name, tag, dx, dy in (("向右", "right", 0.36, 0.0), ("向上", "up", 0.0, -0.28),
                              ("向左", "left", -0.36, 0.0), ("向下", "down", 0.0, 0.28)):
        start = len(s.results)
        after = moved(shape, dx, dy)
        want = np.subtract(s.expected_cursor(after), s.expected_cursor(shape), dtype=np.float64)
        s.play(hand.travel(shape, dx, dy, 36) + hand.hold(after, 20))
        speeds[name] = _check_trip(x, s, name, start, len(s.results), want) / math.hypot(dx * ASPECT, dy)
        shape = after
        x.snap(s, f"after_{tag}")
    x.check("横向和纵向灵敏度相同", abs(speeds["向右"] / speeds["向上"] - 1) < 0.05,
            f"手每移动一个单位, 光标移动: 横 {speeds['向右']:.0f} 像素, 纵 {speeds['向上']:.0f} 像素")
    x.check("移动时没有任何点击", not any(e.startswith(("down", "up")) for e in s.events()), " ".join(s.events()))


def _light_curls(hand: Hand, button: str) -> dict:
    """The thumb comes to a finger that stays up (the rules still say point); the
    other fingers do not move at all."""
    curls = dict(hand.curls["point"])
    curls["index" if button == "left" else "middle"] = 0.28 if button == "left" else 0.45
    return curls


def _click_scenario(x: Ctx, button: str) -> None:
    hand = x.hand()
    point = hand.shape("point")
    curls = _light_curls(hand, button)
    far = hand.pinch(button, 0.45, curls)
    s = x.session()
    s.play(hand.hold(point, 12))
    x.snap(s, "before")

    # the thumb closes on the finger over 8 frames, holds 3 frames, opens over 5
    s.play(hand.blend(point, far, 2) + hand.gap_path(button, 0.45, 0.03, 8, curls))
    s.play(hand.hold(hand.pinch(button, 0.03, curls), 3))
    held_until = len(s.results)
    s.play(hand.gap_path(button, 0.03, 0.45, 5, curls) + hand.blend(far, point, 2) + hand.hold(point, 8))

    down, up = _check_one_click(x, s, button)
    if down is None or up is None:
        return
    first = s.first_touch(button)
    x.check("手指碰到的第一帧就按下 (最多晚 1 帧)", first is not None and 0 <= down - first <= 1,
            f"第 {first} 帧碰到, 第 {down} 帧按下, 当时间距 {measured_gap(s.frames[down], button):.3f}")
    # Click timing follows live geometry. A recogniser may already have
    # confirmed the pinch, or its debounced label may still be pointing.
    pinch_label = PINCH_GESTURE[button]
    x.check("按下时手势为 point 或对应捏合", s.results[down].label in ("point", pinch_label),
            s.results[down].label)
    x.check("保持捏合后确认对应手势", s.results[held_until - 1].label == pinch_label,
            s.results[held_until - 1].label)
    x.check(f"按下时状态显示 {button} 键", s.results[down].button == button, str(s.results[down].button))
    want_up = _expected_release(s, button, down)
    x.check("手指一分开就松开", up == want_up, f"第 {up} 帧松开, 应为第 {want_up} 帧")
    path = s.cursor_path(down, held_until)
    drift = float(np.linalg.norm(path - path[0], axis=1).max())
    x.check("点击时光标不乱跑", drift <= 4.0, f"最多偏移 {drift:.1f} 像素")
    x.snap(s, "touch", down)
    x.snap(s, "release", up)


def scn_left_click(x: Ctx) -> None:
    _click_scenario(x, "left")


def scn_right_click(x: Ctx) -> None:
    _click_scenario(x, "right")


def _pressed_hand(hand: Hand) -> tuple[list[np.ndarray], np.ndarray, np.ndarray, dict]:
    """Frames that end with the left button held, plus the point and pinched shapes."""
    point = hand.shape("point")
    curls = _light_curls(hand, "left")
    shut = hand.pinch("left", 0.05, curls)
    frames = hand.hold(point, 12) + hand.blend(point, hand.pinch("left", 0.45, curls), 2)
    return frames + hand.gap_path("left", 0.45, 0.05, 6, curls) + hand.hold(shut, 3), point, shut, curls


def scn_drag(x: Ctx) -> None:
    hand = x.hand(cx=0.35, cy=0.62)
    frames, _point, shut, curls = _pressed_hand(hand)
    dx, dy = 0.30, -0.20
    s = x.session()
    s.play(frames)
    x.snap(s, "press")
    start = len(s.results)
    want = np.subtract(s.expected_cursor(moved(shut, dx, dy)), s.expected_cursor(shut), dtype=np.float64)
    s.play(hand.travel(shut, dx, dy, 40))
    x.snap(s, "dragging", start + 20)
    s.play(hand.hold(moved(shut, dx, dy), 15))
    stop = len(s.results)
    s.play(shifted(hand.gap_path("left", 0.05, 0.45, 5, curls), dx, dy))

    down, up = _check_one_click(x, s, "left")
    if down is None or up is None:
        return
    x.snap(s, "release", up)
    x.check("拖动全程左键一直按着", all(r.button == "left" for r in s.results[down:up]), "")
    x.check("松手发生在拖动结束之后", up >= stop, f"第 {up} 帧松开, 拖动到第 {stop} 帧")
    _check_trip(x, s, "拖动", start, stop, want)


def _scroll_scenario(x: Ctx, direction: int) -> None:
    """direction: +1 = the hand moves up (wheel up), -1 = the hand moves down."""
    travel = 0.22
    hand = x.hand(cy=0.66 if direction > 0 else 0.44)
    shape = hand.shape("scroll")
    dy = -direction * travel
    s = x.session()
    s.play(hand.hold(shape, 12))
    x.check("手势确认为 scroll", s.results[-1].label == "scroll", s.results[-1].label)
    x.check("手不动时不滚动", not s.notches(), str(s.notches()))
    x.snap(s, "start")
    s.play(hand.travel(shape, 0.0, dy, 30))
    x.snap(s, "scrolling", len(s.results) - 15)
    s.play(hand.hold(moved(shape, 0.0, dy), 20))
    notches = s.notches()
    want = direction * travel * s.cfg.scroll_gain
    x.check("滚轮方向正确", bool(notches) and all(n * direction > 0 for n in notches), str(notches))
    x.check("滚动量正确", abs(sum(notches) - want) < 0.15 * abs(want), f"实际 {sum(notches):+.2f} 格, 应约为 {want:+.2f} 格")
    tail = s.notches(len(s.results) - 10)
    x.check("手停下后滚动也停下", not tail, str(tail))
    x.check("滚动时光标不动", s.backend.count("move") == 0, f"{s.backend.count('move')} 次移动")
    x.check("滚动时没有点击", s.backend.count("down") == 0, f"{s.backend.count('down')} 次按下")


def scn_scroll_up(x: Ctx) -> None:
    _scroll_scenario(x, +1)


def scn_scroll_down(x: Ctx) -> None:
    _scroll_scenario(x, -1)


def _frozen_scenario(x: Ctx, gesture: str) -> None:
    """The hand wanders around in ``gesture``: the cursor stays put and nothing is
    clicked. Then pointing resumes and the cursor is at the hand at once, no glide.

    The pose changes in one cut: the in-between shapes of a real change of gesture
    are what ``transition_no_false_click`` in the stress test is about."""
    hand = x.hand(cx=0.40, cy=0.58)
    point, pose = hand.shape("point"), hand.shape(gesture)
    s = x.session()
    s.play(hand.hold(point, 12) + hand.hold(pose, 6))
    x.check(f"手势确认为 {gesture}", s.results[-1].label == gesture, s.results[-1].label)
    frozen_at, moves_before = s.results[-1].cursor, s.backend.count("move")
    x.snap(s, "entered")
    dx, dy = 0.20, -0.12
    s.play(hand.travel(pose, dx, dy, 30))
    x.snap(s, "moving")
    x.check("手在动, 光标不动", s.backend.count("move") == moves_before and s.results[-1].cursor == frozen_at,
            f"光标 {frozen_at} -> {s.results[-1].cursor}")
    x.check("没有点击", s.backend.count("down") == 0, f"{s.backend.count('down')} 次按下")
    x.check("没有滚动", s.backend.count("scroll") == 0, f"{s.backend.count('scroll')} 次滚动")

    start = len(s.results)
    s.play(hand.hold(moved(point, dx, dy), 6))
    first = next((i for i in range(start, len(s.results)) if s.results[i].label == "point"), None)
    x.check("恢复 point", first is not None, s.results[-1].label)
    if first is not None:
        x.check("恢复后光标立刻回到手的位置", s.results[first].cursor == s.expected_cursor(s.frames[first]),
                f"光标 {s.results[first].cursor}, 手在 {s.expected_cursor(s.frames[first])}")
    x.snap(s, "resumed")


def scn_pause_fist(x: Ctx) -> None:
    _frozen_scenario(x, "fist")


def scn_idle_open_palm(x: Ctx) -> None:
    _frozen_scenario(x, "open_palm")


def scn_hand_lost(x: Ctx) -> None:
    hand = x.hand(cx=0.40, cy=0.58)
    frames, point, _shut, _curls = _pressed_hand(hand)
    s = x.session()
    s.play(frames)
    x.check("手消失前左键是按下的", s.results[-1].button == "left", str(s.results[-1].button))
    frozen = s.results[-1].cursor
    gone_at = s.t
    x.snap(s, "pressed")

    s.play([None] * 15)                 # half a second without a hand
    ups = s.event_frames("up:left")
    x.check("手消失后松开左键, 只松开一次", len(ups) == 1 and not s.backend.down, f"松开帧 {ups}")
    if ups:
        waited, grace = s.results[ups[0]].t - gone_at, s.engine.LOST_GRACE_S
        x.check("等够容忍时间才松开, 也不多等", grace <= waited < grace + 1.5 * DT,
                f"手消失 {waited:.3f} 秒后松开, 容忍 {grace} 秒")
    x.check("手不在时光标不动", s.results[-1].cursor == frozen, f"{frozen} -> {s.results[-1].cursor}")
    x.snap(s, "gone")

    # back somewhere else: the very first moving frame is already at the hand
    start = len(s.results)
    s.play(hand.hold(moved(point, 0.22, -0.14), 8))
    first = next((i for i in range(start, len(s.results)) if s.results[i].cursor != frozen), None)
    x.check("手回来后恢复控制", first is not None, "光标一直没动")
    if first is not None:
        x.check("没有飞入: 第一帧就在手的新位置", s.results[first].cursor == s.expected_cursor(s.frames[first]),
                f"光标 {s.results[first].cursor}, 手在 {s.expected_cursor(s.frames[first])}")
    x.check("手回来后没有自己按键", s.backend.count("down") == 1, f"{s.backend.count('down')} 次按下")
    x.snap(s, "reacquired")


def scn_short_dropout(x: Ctx) -> None:
    hand = x.hand(cx=0.38, cy=0.60)
    frames, _point, shut, _curls = _pressed_hand(hand)
    path = hand.travel(shut, 0.18, -0.10, 30)
    path[10] = path[11] = None          # tracker flicker: 2 frames lost, 1 more later
    path[20] = None
    s = x.session()
    s.play(frames)
    start = len(s.results)
    s.play(path)
    x.snap(s, "during_dropout", start + 11)
    held = [r.button for r in s.results[start:]]
    x.check("拖动中短暂丢手, 左键一直按着", all(b == "left" for b in held), str(held))
    x.check("没有多余的松开和按下", not s.event_frames("up:left") and s.backend.count("down") == 1,
            " ".join(s.events()))
    x.check("丢手后光标继续跟手", s.results[-1].cursor != s.results[start + 11].cursor,
            f"{s.results[start + 11].cursor} -> {s.results[-1].cursor}")
    x.snap(s, "after_dropout")


def _zoom_scenario(x: Ctx, start: float, end: float, word: str, opposite: str) -> None:
    """Two hands, mirror images of each other, ``spread`` apart (fraction of the image width)."""
    left = x.hand(size=0.13, roll=-8.0)
    right = Hand(np.random.default_rng(x.seed + 1), HandPose(cx=0.5, cy=0.55, size=0.13, roll=8.0, mirror=True))
    shape_l, shape_r = left.shape("open_palm"), right.shape("open_palm")
    n = 30
    spreads = [start] * 5 + [start + (end - start) * ease((i + 1) / n) for i in range(n)] + [end] * 8
    s = x.session()
    s.play([(left.noisy(moved(shape_l, -d / 2, 0.0)), right.noisy(moved(shape_r, d / 2, 0.0))) for d in spreads])

    x.check("两只手时状态是 zoom", all(r.label == "zoom" for r in s.results), "")
    events = s.events()
    x.check(f"产生 zoom:{word}", f"zoom:{word}" in events, " ".join(events))
    x.check(f"没有相反方向的 zoom:{opposite}", f"zoom:{opposite}" not in events, " ".join(events))
    steps = [e[1] for e in s.backend.events if e[0] == "zoom"]
    want = math.log(end / start) * s.cfg.zoom_gain
    x.check("缩放量大致正确", abs(sum(steps) - want) < 0.25 * abs(want), f"实际 {sum(steps):+.2f}, 应约为 {want:+.2f}")
    still = s.events(0, 5) + s.events(len(s.results) - 3)
    x.check("手不动时不缩放", not still, " ".join(still))
    x.check("缩放时不点击也不移动光标", s.backend.count("down") == 0 and s.backend.count("move") == 0, "")
    x.snap(s, "start", 4)
    x.snap(s, "middle", 5 + n // 2)
    x.snap(s, "end")


def scn_zoom_in(x: Ctx) -> None:
    _zoom_scenario(x, 0.30, 0.65, "in", "out")


def scn_zoom_out(x: Ctx) -> None:
    _zoom_scenario(x, 0.65, 0.30, "out", "in")


def scn_dwell_click(x: Ctx) -> None:
    hand = x.hand(cx=0.40, cy=0.58)
    point = hand.shape("point")
    s = x.session(dwell_enabled=True)
    s.play(hand.hold(point, 3))         # the label settles and the cursor starts
    start_t = s.results[-1].t
    s.play(hand.hold(point, 36))        # 1.2 s standing still
    clicks = s.event_frames("click:dwell")
    x.check("停留约 1 秒后恰好点击一次", len(clicks) == 1, f"点击帧 {clicks}")
    if clicks:
        waited_ms = (s.results[clicks[0]].t - start_t) * 1000
        x.check("不提前也不拖太久", s.cfg.dwell_ms - 1000 * DT <= waited_ms <= s.cfg.dwell_ms + 200,
                f"停留 {waited_ms:.0f} 毫秒后点击")
        x.snap(s, "dwell_click", clicks[0])
    s.play(hand.hold(point, 45))        # keeps standing still
    x.check("不移开就不再点击", len(s.event_frames("click:dwell")) == 1, str(s.event_frames("click:dwell")))

    dx, dy = 0.15, 0.05
    s.play(hand.travel(point, dx, dy, 12) + hand.hold(moved(point, dx, dy), 45))
    x.check("移开再停留, 又点击一次", len(s.event_frames("click:dwell")) == 2, str(s.event_frames("click:dwell")))
    n_clicks = len(s.event_frames("click:dwell"))
    x.check("每次点击都有按下和松开", s.backend.count("down", "left") == s.backend.count("up", "left") == n_clicks,
            f"{s.backend.count('down')} 次按下, {s.backend.count('up')} 次松开")
    x.snap(s, "second_click")


# name -> (description, function, also run with the ML engine)
SCENARIOS = {
    "move": ("移动光标: 手往哪走, 光标往哪走", scn_move, True),
    "left_click": ("左键单击: 拇指碰食指", scn_left_click, True),
    "drag": ("拖动: 捏住不放同时移动手", scn_drag, True),
    "right_click": ("右键单击: 拇指碰中指", scn_right_click, True),
    "scroll_up": ("向上滚动: 比 V 字, 手往上", scn_scroll_up, True),
    "scroll_down": ("向下滚动: 比 V 字, 手往下", scn_scroll_down, True),
    "pause_fist": ("暂停: 握拳时光标不动, 也不点击", scn_pause_fist, True),
    "idle_open_palm": ("休息: 张开手时光标不动", scn_idle_open_palm, True),
    "hand_lost": ("手离开画面又回来", scn_hand_lost, False),
    "short_dropout": ("拖动中短暂丢手, 拖动不断", scn_short_dropout, False),
    "zoom_in": ("两只手张开: 放大", scn_zoom_in, False),
    "zoom_out": ("两只手合拢: 缩小", scn_zoom_out, False),
    "dwell_click": ("停留点击: 手不动约 1 秒自动点击", scn_dwell_click, False),
}


# -------------------------------------------------------------------- running
def _render(x: Ctx, prefix: str, engine_name: str) -> None:
    for k, (tag, s, i) in enumerate(x.snaps, start=1):
        frame = s.frames[i]
        hands = list(frame) if isinstance(frame, tuple) else ([] if frame is None else [frame])
        img = draw_hud(blank_canvas(s.cfg.frame_width, s.cfg.frame_height), s.results[i], hands, fps=float(FPS),
                       engine_name=engine_name, active_box=s.engine.controller.active_box,
                       dwell=s.cfg.dwell_enabled)
        save_png(IMAGES_DIR / f"sim_{prefix}{x.name}_{k}_{tag}.png", img)


def run_scenario(name: str, recognizer=None, verbose: bool = False, render: bool = False,
                 engine_name: str = "rule") -> dict:
    title, fn, _ = SCENARIOS[name]
    x = Ctx(name, recognizer)
    prefix = "ml_" if engine_name == "ml" else ""
    try:
        fn(x)
        crash = None
    except Exception:           # a crash is a finding like any other: report it and go on
        crash = traceback.format_exc(limit=4)
    failed = [c for c in x.checks if not c.ok]
    passed = crash is None and not failed and bool(x.checks)
    if verbose:
        print(f"\n--- {prefix}{name}: {title}")
        for s in x.sessions:
            print("\n".join(s.log_lines()))
        for c in x.checks:
            print(f"  [{'通过' if c.ok else '失败'}] {c.what}" + (f"  ({c.info})" if c.info else ""))
    if render:
        _render(x, prefix, engine_name)
    if passed:
        details = f"{len(x.checks)}/{len(x.checks)} 项检查通过"
    else:
        lines = [f"{c.what} ({c.info})" for c in failed]
        details = "; ".join(lines + ([f"程序出错: {crash}"] if crash else []))
    return {"scenario": prefix + name, "passed": passed, "details": details}


def _load_ml(path: Path) -> tuple[object, str]:
    """(classifier, "") or (None, a plain explanation of why not)."""
    if not Path(path).exists():
        return None, f"还没有训练好的手势模型 ({Path(path).name}), 跳过 ML 模式。"
    cfg = Config()
    try:
        return GestureClassifier(path, cfg.ml_confidence, cfg.aspect), ""
    except Exception as e:      # half-written or incompatible file
        return None, f"手势模型读取失败: {e}"


def run_all(verbose: bool = False, render: bool = True, engine: str = "rule",
            model_path: str | Path = GESTURE_MODEL_PATH) -> list[dict]:
    """Run every scenario and return a list of {scenario, passed, details}.

    ``engine="ml"`` reruns the core scenarios with the trained GestureClassifier;
    without a model file the result is one entry marked ``skipped``.
    """
    if engine == "ml":
        recognizer, why = _load_ml(Path(model_path))
        if recognizer is None:
            return [{"scenario": "ml", "passed": Path(model_path).exists() is False, "skipped": True, "details": why}]
        names = [n for n, (_, _, with_ml) in SCENARIOS.items() if with_ml]
    else:
        recognizer, names = None, list(SCENARIOS)
    return [run_scenario(n, recognizer, verbose, render, engine) for n in names]


def print_table(results: list[dict]) -> None:
    print(f"\n{'场景':<20}结果")
    for r in results:
        word = "跳过" if r.get("skipped") else ("通过" if r["passed"] else "失败")
        print(f"{r['scenario']:<22}{word}  {r['details']}")


def main(argv=None) -> int:
    use_utf8_console()
    ap = argparse.ArgumentParser(description="AirTouch 虚拟测试: 不开摄像头, 不动真鼠标, 检查每个动作。")
    ap.add_argument("--quiet", action="store_true", help="只显示结果表, 不显示每一帧")
    ap.add_argument("--no-render", action="store_true", help="不生成 images/sim_*.png")
    ap.add_argument("--engine", choices=("rule", "ml", "both"), default="both", help="用哪种手势识别 (默认两种都测)")
    a = ap.parse_args(argv)

    results: list[dict] = []
    for engine in (("rule", "ml") if a.engine == "both" else (a.engine,)):
        print(f"\n===== {'规则' if engine == 'rule' else '机器学习 (ML)'} 手势识别 =====")
        part = run_all(verbose=not a.quiet, render=not a.no_render, engine=engine)
        print_table(part)
        results += part
    bad = [r for r in results if not r["passed"]]
    done = [r for r in results if not r.get("skipped")]
    print(f"\n一共 {len(done)} 个场景, {len(done) - len(bad)} 个通过, {len(bad)} 个失败。")
    if not a.no_render:
        print(f"关键画面已保存到 {IMAGES_DIR}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
