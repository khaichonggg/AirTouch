"""Fitts'-law target-acquisition task for the FYP2 user study (ISO 9241-9 style).

A ring of circular targets is shown; the participant selects them one after
another, jumping across the ring each time. Every selection - hit or miss - is
logged, so ``evaluation.analyze_results`` can combine speed and accuracy into
one throughput score.

Both input conditions use the same window and the same targets:

    gesture  camera + hand tracker + GestureEngine, but the cursor is drawn inside
             the window (``WindowBackend``): the real mouse is never moved
    mouse    the real mouse inside the window, read through cv2.setMouseCallback

The camera and tracker are imported only inside ``run_gesture``, and the window
is reached only through ``CvDisplay``, so everything else - layout, hit testing,
sequencing, logging, even the two main loops - runs headlessly.

Run:  python -m evaluation.task_runner --participant P01 --input gesture --blocks 2
"""

from __future__ import annotations

import argparse
import csv
import math
import random
import re
import sys
import time
import zlib
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from airtouch.config import Config
from airtouch.console import use_utf8_console
from airtouch.engine import GestureEngine
from airtouch.gesture_classifier import load_recognizer
from airtouch.hud import blank_canvas, draw_landmarks

Point = tuple[float, float]

RESULTS_DIR = Path(__file__).resolve().parent / "results"
INPUTS = ("gesture", "mouse")
WINDOW_TITLE = "AirTouch target task"
WINDOW_SIZE = (1280, 720)

# Ring diameters and target diameters, in window pixels. Together: 9 conditions
# spanning about 2.2 to 4.3 bits, with neighbouring targets never overlapping.
AMPLITUDES = (250, 400, 550)
WIDTHS = (30, 50, 70)
NUM_TARGETS = 11

TOP_BAR = 40            # status line above the ring
MARGIN = 10
QUIT_KEYS = (27, ord("q"), ord("Q"))

CSV_FIELDS = (
    "participant", "input", "block", "trial", "amplitude", "width",
    "target_x", "target_y", "start_x", "start_y", "click_x", "click_y",
    "hit", "movement_time_ms", "timestamp",
)

# BGR
RING_COLOR = (110, 110, 110)
TARGET_COLOR = (90, 200, 60)
CURSOR_COLOR = (255, 255, 255)
PRESSED_COLOR = (0, 165, 255)
MISS_COLOR = (70, 70, 255)
BAR_COLOR = (25, 25, 25)
TEXT_COLOR = (235, 235, 235)


# ------------------------------------------------------------ layout and logic
def ring_positions(center: Point, amplitude: float, n: int) -> list[Point]:
    """``n`` target centres evenly spaced on a circle of diameter ``amplitude``:
    the first at the top, going clockwise on screen."""
    cx, cy = center
    r = amplitude / 2
    return [(cx + r * math.sin(2 * math.pi * i / n), cy - r * math.cos(2 * math.pi * i / n))
            for i in range(n)]


def alternating_order(n: int) -> list[int]:
    """ISO 9241-9 visiting order: across the ring and back, moving on by one each
    time, so every target is visited once. For n=5: 0 3 1 4 2."""
    half = (n + 1) // 2
    return [i // 2 + (half if i % 2 else 0) for i in range(n)]


def hit_test(x: float, y: float, target: Point, width: float) -> bool:
    """True if (x, y) is inside the circular target of diameter ``width``."""
    return math.hypot(x - target[0], y - target[1]) <= width / 2


def ring_center(size: tuple[int, int]) -> Point:
    width, height = size
    return width / 2, (height + TOP_BAR) / 2


def check_fits(size: tuple[int, int], amplitudes, widths) -> None:
    """Conditions must stay identical for every participant, so a window that is
    too small is an error rather than a reason to shrink the targets."""
    need = max(amplitudes) + max(widths)
    room = min(size[0], size[1] - TOP_BAR) - 2 * MARGIN
    if need > room:
        raise ValueError(f"窗口太小了：最大的一圈目标需要 {need} 像素，现在只有 {room} 像素。"
                         f"请把窗口设大一点（--size）。")


@dataclass(frozen=True)
class Condition:
    block: int
    amplitude: int
    width: int


def plan_sequences(blocks: int, amplitudes, widths, seed) -> list[Condition]:
    """Every A x W once per block, in a new random order each block."""
    rng = random.Random(seed)
    plan: list[Condition] = []
    for block in range(1, blocks + 1):
        cells = [Condition(block, a, w) for a in amplitudes for w in widths]
        rng.shuffle(cells)
        plan += cells
    return plan


@dataclass(frozen=True)
class TrialRecord:
    participant: str
    input: str
    block: int
    trial: int
    amplitude: int
    width: int
    target: Point
    start: Point
    click: Point
    hit: bool
    movement_time_ms: float
    timestamp: str

    def as_row(self) -> dict:
        return {
            "participant": self.participant, "input": self.input,
            "block": self.block, "trial": self.trial,
            "amplitude": self.amplitude, "width": self.width,
            "target_x": round(self.target[0], 2), "target_y": round(self.target[1], 2),
            "start_x": round(self.start[0], 2), "start_y": round(self.start[1], 2),
            "click_x": round(self.click[0], 2), "click_y": round(self.click[1], 2),
            "hit": int(self.hit),
            "movement_time_ms": round(self.movement_time_ms, 1),
            "timestamp": self.timestamp,
        }


def clean_participant(name: str) -> str:
    """Participant code, safe inside a file name (nothing like ``../x``)."""
    code = str(name).strip()
    if not re.fullmatch(r"[\w-]{1,40}", code):
        raise ValueError(f"参与者编号“{name}”不能用：只能用字母、数字、下划线和减号，比如 P01。")
    return code


class FittsSession:
    """The whole task as a state machine: call ``select`` for each button press.

    Each sequence starts on a highlighted target. Selecting it starts the clock
    and logs nothing, since no timed movement has happened yet; until then the
    participant can rest for as long as they like (mid-air pointing is tiring).
    After that every selection, hit or miss, logs one trial and moves on: repeating
    a missed target would hide the very errors the analysis needs to see.
    The ring is closed, so a sequence of n targets has n timed movements.
    """

    def __init__(
        self,
        participant: str,
        input_name: str,
        blocks: int = 2,
        size: tuple[int, int] = WINDOW_SIZE,
        amplitudes=AMPLITUDES,
        widths=WIDTHS,
        n_targets: int = NUM_TARGETS,
        seed=None,
        on_trial=None,
        now=datetime.now,
    ):
        self.participant = clean_participant(participant)
        if input_name not in INPUTS:
            raise ValueError(f"操作方式只能是 {' 或 '.join(INPUTS)}，现在是“{input_name}”。")
        if blocks < 1:
            raise ValueError("轮数（--blocks）至少要 1。")
        # Only with an odd count do all the jumps have the same length.
        if n_targets < 3 or n_targets % 2 == 0:
            raise ValueError("目标个数（--targets）必须是 3 或以上的单数，比如 11。")
        check_fits(size, amplitudes, widths)
        self.input = input_name
        self.size = tuple(size)
        self.n_targets = n_targets
        self.on_trial = on_trial
        self.now = now
        self.center = ring_center(size)
        self.widest_ring = max(amplitudes) + max(widths)
        if seed is None:
            seed = zlib.crc32(f"{self.participant}/{input_name}".encode())
        self.plan = plan_sequences(blocks, amplitudes, widths, seed)
        self.blocks = blocks
        self.index = 0
        self.last: TrialRecord | None = None
        self.trials_done = 0
        self.hits = 0
        order = alternating_order(n_targets)
        self._visits = order + order[:1]    # start target, then n movements back to it
        self._begin_sequence()

    def _begin_sequence(self) -> None:
        self._step = 0
        self._from: Point | None = None
        self._t_prev = 0.0
        self.last = None
        if not self.finished:
            c = self.condition
            self.targets = ring_positions(self.center, c.amplitude, self.n_targets)

    @property
    def finished(self) -> bool:
        return self.index >= len(self.plan)

    @property
    def condition(self) -> Condition:
        return self.plan[self.index]

    @property
    def waiting(self) -> bool:
        """True until the start target of the current sequence is selected."""
        return not self.finished and self._step == 0

    @property
    def trial_number(self) -> int:
        """1-based trial within the sequence; 0 while waiting at the start target."""
        return self._step

    @property
    def current_target(self) -> Point | None:
        return None if self.finished else self.targets[self._visits[self._step]]

    def select(self, x: float, y: float, t: float) -> TrialRecord | None:
        """A button press at window position (x, y) at time ``t`` (seconds).
        Returns the logged trial, or None if this press logged nothing."""
        if self.finished:
            return None
        c = self.condition
        target = self.current_target
        hit = hit_test(x, y, target, c.width)
        if self.waiting:
            if hit:
                self._from, self._t_prev, self._step = (float(x), float(y)), t, 1
            return None
        record = TrialRecord(
            participant=self.participant,
            input=self.input,
            block=c.block,
            trial=self._step,
            amplitude=c.amplitude,
            width=c.width,
            target=target,
            start=self._from,
            click=(float(x), float(y)),
            hit=hit,
            movement_time_ms=(t - self._t_prev) * 1000.0,
            timestamp=self.now().isoformat(timespec="milliseconds"),
        )
        self.trials_done += 1
        self.hits += hit
        if self.on_trial is not None:
            self.on_trial(record)
        self.last = record
        self._from, self._t_prev = record.click, t
        self._step += 1
        if self._step >= len(self._visits):
            self.index += 1
            self._begin_sequence()
        return record


# ------------------------------------------------------------------- logging
def session_csv_path(participant: str, input_name: str, results_dir=RESULTS_DIR, when=None) -> Path:
    when = when or datetime.now()
    return Path(results_dir) / f"{clean_participant(participant)}_{input_name}_{when:%Y%m%d-%H%M%S}.csv"


class TrialLogger:
    """One CSV per session, flushed after every trial so a crash loses nothing.
    Use it as a session's ``on_trial``. Never overwrites an existing file, and a
    session that recorded nothing leaves no file behind."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self.path.open("x", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._file, CSV_FIELDS)
        self._writer.writeheader()
        self._file.flush()
        self.count = 0

    def __call__(self, record: TrialRecord) -> None:
        self._writer.writerow(record.as_row())
        self._file.flush()
        self.count += 1

    def close(self) -> None:
        if self._file.closed:
            return
        self._file.close()
        if self.count == 0:
            self.path.unlink(missing_ok=True)

    def __enter__(self) -> "TrialLogger":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


# --------------------------------------------------------------- input sources
class WindowBackend:
    """Mouse backend whose "screen" is the task window.

    It has the interface of the real backends in ``airtouch.controller``, so the
    GestureEngine works unchanged, but it only remembers a position and a list of
    presses. The real mouse is never touched: safe for the participant's desktop,
    and precise because the cursor lives in the same pixels as the targets.
    """

    def __init__(self, width: int = WINDOW_SIZE[0], height: int = WINDOW_SIZE[1]):
        self.size = (int(width), int(height))
        self.pos = (self.size[0] // 2, self.size[1] // 2)
        self.down: set[str] = set()
        self._presses: list[tuple[str, int, int]] = []

    def screen_size(self) -> tuple[int, int]:
        return self.size

    def move(self, x: int, y: int) -> None:
        self.pos = (min(max(int(x), 0), self.size[0] - 1), min(max(int(y), 0), self.size[1] - 1))

    def press(self, button: str) -> None:
        self.down.add(button)
        self._presses.append((button, *self.pos))

    def release(self, button: str) -> None:
        self.down.discard(button)

    def scroll(self, notches: float) -> None:
        pass        # the task has nothing to scroll

    def zoom(self, steps: float) -> None:
        pass

    def pop_presses(self) -> list[tuple[str, int, int]]:
        """(button, x, y) for every press since the last call."""
        presses, self._presses = self._presses, []
        return presses


class MouseInput:
    """Left-button presses from cv2.setMouseCallback, time-stamped as they arrive.
    A second quick press on one spot arrives as a double-click event: still a press."""

    def __init__(self, clock=time.perf_counter):
        self.clock = clock
        self._presses: list[tuple[int, int, float]] = []

    def on_event(self, event: int, x: int, y: int, flags: int = 0, param=None) -> None:
        if event in (cv2.EVENT_LBUTTONDOWN, cv2.EVENT_LBUTTONDBLCLK):
            self._presses.append((x, y, self.clock()))

    def drain(self) -> list[tuple[int, int, float]]:
        presses, self._presses = self._presses, []
        return presses


# ------------------------------------------------------------------- drawing
def _px(p: Point) -> tuple[int, int]:
    return int(round(p[0])), int(round(p[1]))


def _label(img: np.ndarray, text: str, org: tuple[int, int], scale: float = 0.55, color=TEXT_COLOR) -> None:
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


def _draw_preview(img: np.ndarray, session: FittsSession, frame: np.ndarray, hands, box) -> None:
    """Small camera view beside the ring, so people can keep their hand in the box."""
    width, _height = session.size
    pw = min(240, (width - session.widest_ring) // 2 - 2 * MARGIN)
    if pw < 100:
        return
    ph = pw * frame.shape[0] // frame.shape[1]
    small = cv2.resize(frame, (pw, ph), interpolation=cv2.INTER_AREA)
    if box is not None:
        x0, y0, bw, bh = box
        cv2.rectangle(small, (int(x0 * pw), int(y0 * ph)), (int((x0 + bw) * pw), int((y0 + bh) * ph)),
                      TEXT_COLOR, 1, cv2.LINE_AA)
    for hand in hands:
        draw_landmarks(small, hand, TARGET_COLOR)
    x, y = width - pw - MARGIN, TOP_BAR + MARGIN
    img[y:y + ph, x:x + pw] = small


def render(session: FittsSession, cursor=None, pressed: bool = False, frame=None, hands=(), box=None,
           status: str = "") -> np.ndarray:
    """One picture of the task. Text is English: cv2.putText cannot draw Chinese."""
    width, height = session.size
    img = blank_canvas(width, height)
    cv2.rectangle(img, (0, 0), (width, TOP_BAR), BAR_COLOR, -1)
    if session.finished:
        _label(img, "All done - thank you!", (width // 2 - 140, height // 2), 0.9)
    else:
        c = session.condition
        radius = max(1, int(round(c.width / 2)))
        for p in session.targets:
            cv2.circle(img, _px(p), radius, RING_COLOR, 1, cv2.LINE_AA)
        cv2.circle(img, _px(session.current_target), radius, TARGET_COLOR, -1, cv2.LINE_AA)
        last = session.last
        if last is not None and not last.hit:
            x, y = _px(last.click)
            cv2.line(img, (x - 7, y - 7), (x + 7, y + 7), MISS_COLOR, 2, cv2.LINE_AA)
            cv2.line(img, (x - 7, y + 7), (x + 7, y - 7), MISS_COLOR, 2, cv2.LINE_AA)
        per_block = len(session.plan) // session.blocks
        where = (f"Block {c.block}/{session.blocks}   "
                 f"Set {session.index % per_block + 1}/{per_block}   "
                 f"Target {session.trial_number}/{session.n_targets}")
        _label(img, where, (MARGIN, 26))
        if session.waiting:
            how = ("touch thumb to index finger" if session.input == "gesture" else "click")
            _label(img, f"Rest if you need to. To start: {how} on the green target.",
                   (MARGIN, height - 12), 0.6)
        else:
            _label(img, "Be fast and accurate.", (MARGIN, height - 12), 0.6)
    _label(img, "Q or Esc = stop", (width - 150, 26), 0.5)
    if status:
        _label(img, status, (MARGIN, TOP_BAR + 24))
    if frame is not None:
        _draw_preview(img, session, frame, hands, box)
    if cursor is not None:
        x, y = _px(cursor)
        color = PRESSED_COLOR if pressed else CURSOR_COLOR
        for a, b in (((x - 12, y), (x + 12, y)), ((x, y - 12), (x, y + 12))):
            cv2.line(img, a, b, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.line(img, a, b, color, 2, cv2.LINE_AA)
    return img


class CvDisplay:
    """The real window. Everything that needs a desktop is in here, so the loops
    below can be driven by a stand-in during tests."""

    def __init__(self, title: str = WINDOW_TITLE):
        self.title = title
        cv2.namedWindow(title, cv2.WINDOW_AUTOSIZE)

    def show(self, img: np.ndarray) -> None:
        cv2.imshow(self.title, img)

    def wait_key(self, ms: int = 1) -> int:
        return cv2.waitKey(ms) & 0xFF

    def on_mouse(self, callback) -> None:
        cv2.setMouseCallback(self.title, callback)

    def closed(self) -> bool:
        try:
            return cv2.getWindowProperty(self.title, cv2.WND_PROP_VISIBLE) < 1
        except cv2.error:
            return True

    def close(self) -> None:
        try:
            cv2.destroyWindow(self.title)
        except cv2.error:
            pass


# ----------------------------------------------------------------- main loops
def mouse_loop(session: FittsSession, display, clock=time.perf_counter) -> bool:
    """Run with the real mouse until done (True) or stopped early (False)."""
    mouse = MouseInput(clock)
    display.on_mouse(mouse.on_event)
    display.show(render(session))
    while not session.finished:
        key = display.wait_key(5)
        presses = mouse.drain()
        for x, y, t in presses:
            session.select(x, y, t)
        if presses:
            display.show(render(session))
        if key in QUIT_KEYS or display.closed():
            return False
    return True


def gesture_loop(session: FittsSession, engine: GestureEngine, backend: WindowBackend, camera, tracker, display,
                 clock=time.perf_counter, mirror: bool = True) -> bool:
    """Camera -> tracker -> engine -> in-window cursor, until done (True) or stopped
    early (False). A press is timed when the system registers it, which is what the
    participant experiences; constant camera delay cancels out, because movement
    time is the gap between two presses."""
    box = engine.controller.active_box
    display.show(render(session, cursor=backend.pos, box=box))
    seq = 0
    origin = clock()
    last_ms = -1
    while not session.finished:
        seq, frame = camera.read_if_new(seq)
        if frame is not None:
            if mirror:
                frame = cv2.flip(frame, 1)
            now = clock()
            last_ms = max(int((now - origin) * 1000), last_ms + 1)     # must strictly increase
            hands = tracker.detect(frame, last_ms)
            result = engine.process_hands(hands, clock())
            for button, x, y in backend.pop_presses():
                if button == "left":
                    session.select(x, y, clock())
            status = result.label if result.hands else "show your hand to the camera"
            display.show(render(session, cursor=backend.pos, pressed="left" in backend.down,
                                frame=frame, hands=hands, box=box, status=status))
        key = display.wait_key(1 if frame is not None else 5)
        if key in QUIT_KEYS or display.closed():
            return False
    return True


def run_mouse(session: FittsSession, display) -> bool:
    return mouse_loop(session, display)


def run_gesture(session: FittsSession, display, cfg: Config | None = None) -> bool:
    # Imported here, not at the top: importing this module (tests, the mouse
    # condition) must never load MediaPipe or touch the camera.
    from airtouch.camera import ThreadedCamera
    from airtouch.hand_tracker import HandTracker

    # One hand and no dwell-click: selection is by pinch for every participant.
    cfg = replace(cfg or Config.load(), max_hands=1, dwell_enabled=False)
    recognizer, _note = load_recognizer(cfg)
    backend = WindowBackend(*session.size)
    engine = GestureEngine(cfg, backend, recognizer)
    camera = ThreadedCamera(cfg.camera_index, cfg.frame_width, cfg.frame_height, cfg.camera_fps).start()
    try:
        tracker = HandTracker(cfg)
        try:
            return gesture_loop(session, engine, backend, camera, tracker, display, mirror=cfg.mirror)
        finally:
            tracker.close()
    finally:
        camera.release()


# ------------------------------------------------------------------------ CLI
def _parse_size(text: str) -> tuple[int, int]:
    match = re.fullmatch(r"(\d{3,4})[xX](\d{3,4})", text.strip())
    if not match:
        raise argparse.ArgumentTypeError(f"窗口大小要写成 1280x720 这样，现在是“{text}”。")
    return int(match.group(1)), int(match.group(2))


def main(argv=None, display_factory=CvDisplay) -> int:
    use_utf8_console()
    ap = argparse.ArgumentParser(description="目标点击测试：用手势或鼠标依次点圈上的目标，并记录用时和准确度。")
    ap.add_argument("--participant", required=True, help="参与者编号，比如 P01（不要用真实姓名）")
    ap.add_argument("--input", choices=INPUTS, required=True, help="gesture = 用手势，mouse = 用鼠标")
    ap.add_argument("--blocks", type=int, default=2, help="做几轮（默认 2）")
    ap.add_argument("--targets", type=int, default=NUM_TARGETS, help=f"每一圈有几个目标，必须是单数（默认 {NUM_TARGETS}）")
    ap.add_argument("--size", type=_parse_size, default=WINDOW_SIZE, help="窗口大小，比如 1280x720")
    ap.add_argument("--seed", type=int, default=None, help="固定随机顺序（默认按参与者编号决定）")
    ap.add_argument("--practice", action="store_true", help="练习：照常做，但不保存结果")
    ap.add_argument("--results", default=str(RESULTS_DIR), help="结果保存到哪个文件夹")
    args = ap.parse_args(argv)

    try:
        session = FittsSession(args.participant, args.input, blocks=args.blocks, size=args.size,
                               n_targets=args.targets, seed=args.seed)
    except ValueError as e:
        print(e, file=sys.stderr)
        return 2

    logger = None
    if not args.practice:
        try:
            logger = TrialLogger(session_csv_path(session.participant, session.input, args.results))
        except OSError as e:
            print(f"没法保存结果文件：{e}", file=sys.stderr)
            return 1
        session.on_trial = logger

    print(f"参与者 {session.participant}，操作方式 {session.input}，"
          f"一共 {len(session.plan)} 组，每组 {session.n_targets} 次。按 Q 或 Esc 可以随时停止。")
    try:
        display = display_factory()
        try:
            finished = run_gesture(session, display) if session.input == "gesture" else run_mouse(session, display)
            if finished:
                display.show(render(session))
                display.wait_key(1500)
        finally:
            display.close()
    except KeyboardInterrupt:
        finished = False
    except (RuntimeError, OSError, cv2.error) as e:
        print(f"测试没能继续：{e}", file=sys.stderr)
        return 1
    finally:
        if logger is not None:
            logger.close()

    print(("已全部做完。" if finished else "已提前停止。") + f"一共 {session.trials_done} 次，点中 {session.hits} 次。")
    if logger is None:
        print("这是练习，结果没有保存。")
    elif logger.count:
        print(f"结果已保存：{logger.path}")
    else:
        print("还没有做完任何一次，所以没有保存结果。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
