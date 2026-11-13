"""The live loop end to end, with a fake camera and FakeBackend.

No webcam is opened, the real mouse is never touched and no window ever appears:
the fixture below makes any slip fail the test, and the window tests replace
OpenCV's GUI calls with a fake.
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pytest

import diagnose_speed
import hand_mouse
from airtouch import app
from airtouch.config import HAND_MODEL_PATH, PROJECT_ROOT, Config
from airtouch.controller import FakeBackend
from airtouch.engine import GestureEngine
from airtouch.gestures_rule import RuleRecognizer
from ml.synthetic_data import HandPose, make_hand

POINT_CURLS = {"index": 0.05, "middle": 0.9, "ring": 0.9, "pinky": 0.9}
PINCH_CURLS = {"index": 0.45, "middle": 0.7, "ring": 0.7, "pinky": 0.7}


@pytest.fixture(autouse=True)
def nothing_real(monkeypatch):
    """A slip in a test must fail, not move the mouse or switch on the webcam."""

    def forbidden(*args, **kwargs):
        raise AssertionError("a test tried to use the real mouse or camera")

    monkeypatch.setattr(app, "make_backend", forbidden)
    monkeypatch.setattr(app, "ThreadedCamera", forbidden)
    monkeypatch.setattr(diagnose_speed, "ThreadedCamera", forbidden)
    monkeypatch.setattr(cv2, "VideoCapture", forbidden)
    for name in ("namedWindow", "imshow", "waitKey", "getWindowProperty", "destroyAllWindows"):
        monkeypatch.setattr(cv2, name, forbidden)


# -- fakes -----------------------------------------------------------------------
class FakeCamera:
    """Duck-typed ThreadedCamera: numbered frames every ``period`` seconds, and a
    silent gap of ``gap_s`` seconds after frame number ``gap_after``.

    on_frame(seq) runs just before a frame is handed over, letting a test look at
    the world as it was when the camera came back.
    """

    def __init__(self, period=0.001, gap_after=None, gap_s=0.0, on_frame=None):
        self.period = period
        self.gap_after = gap_after
        self.gap_s = gap_s
        self.on_frame = on_frame
        self.seq = 0
        self.polls = 0
        self.started = 0
        self.released = 0
        self.frame_times = {}
        self.next_t = 0.0

    def start(self):
        self.started += 1
        return self

    def read_if_new(self, last_seq):
        self.polls += 1
        now = time.perf_counter()
        if now < self.next_t:
            return self.seq, None
        self.seq += 1
        self.frame_times[self.seq] = now
        self.next_t = now + (self.gap_s if self.seq == self.gap_after else self.period)
        if self.on_frame is not None:
            self.on_frame(self.seq)
        return self.seq, np.zeros((48, 64, 3), dtype=np.uint8)

    def release(self):
        self.released += 1


class FakeTracker:
    """Plays back a script, one entry per frame: a hand, a list of hands, or None."""

    def __init__(self, script):
        self.script = list(script)
        self.calls = 0
        self.closed = 0
        self.stamps = []
        self.seen_frames = []

    def detect(self, frame, t_ms=None):
        self.seen_frames.append(frame)
        self.stamps.append(t_ms)
        entry = self.script[min(self.calls, len(self.script) - 1)]
        self.calls += 1
        if entry is None:
            return []
        return entry if isinstance(entry, list) else [entry]

    def close(self):
        self.closed += 1


class FakeGui:
    """Stands in for OpenCV's window calls.

    keys:         {frame number: key} pressed right after that frame was shown
    closed_after: the window's X button is clicked after this many frames
    """

    def __init__(self, monkeypatch, keys=None, closed_after=None):
        self.keys = dict(keys or {})
        self.closed_after = closed_after
        self.created = 0
        self.shown = 0
        self.destroyed = 0
        self.waits = 0
        for name in ("namedWindow", "imshow", "waitKey", "getWindowProperty", "destroyAllWindows"):
            monkeypatch.setattr(cv2, name, getattr(self, name))

    def namedWindow(self, name, flags=None):
        assert name == app.WINDOW
        self.created += 1

    def imshow(self, name, img):
        assert name == app.WINDOW
        self.shown += 1

    def waitKey(self, ms):
        self.waits += 1
        return self.keys.pop(self.shown, -1)

    def getWindowProperty(self, name, prop):
        assert prop == cv2.WND_PROP_VISIBLE
        return 0.0 if self.closed_after is not None and self.shown >= self.closed_after else 1.0

    def destroyAllWindows(self):
        self.destroyed += 1


def point_hand(rng, cx=0.5):
    return make_hand("point", rng, pose=HandPose(cx=cx, size=0.2), curls=POINT_CURLS, noise=0.0015)


def pinch_hand(rng, cx=0.5, gap=0.02):
    return make_hand("pinch_index", rng, pose=HandPose(cx=cx, size=0.2), curls=PINCH_CURLS,
                     thumb_gap=gap, noise=0.0015)


def run(tracker, camera=None, cfg=None, backend=None, show_window=False, max_frames=None):
    camera = camera or FakeCamera()
    backend = backend or FakeBackend()
    frames = len(tracker.script) if max_frames is None else max_frames
    code = app.main(cfg or Config(), show_window=show_window, camera=camera, tracker=tracker,
                    backend=backend, max_frames=frames)
    return code, camera, backend


def drag_script(rng):
    """20 frames pointing, 15 pinching while moving, 25 pointing: 60 in all."""
    script = [point_hand(rng, 0.35 + i * 0.01) for i in range(20)]
    script += [pinch_hand(rng, 0.55 + i * 0.005) for i in range(15)]
    script += [point_hand(rng, 0.62 - i * 0.005) for i in range(25)]
    return script


# -- the whole loop ----------------------------------------------------------------
def test_scripted_session_moves_clicks_and_exits_clean():
    rng = np.random.default_rng(0)
    tracker = FakeTracker(drag_script(rng))
    code, camera, backend = run(tracker)
    assert code == 0
    assert tracker.calls == 60

    moves = [e for e in backend.events if e[0] == "move"]
    assert len({e[1] for e in moves}) > 10      # the cursor really travelled

    assert backend.count("down", "left") == 1
    assert backend.count("up", "left") == 1
    assert backend.count("down", "right") == 0
    down = backend.events.index(("down", "left"))
    up = backend.events.index(("up", "left"))
    assert down < up
    assert any(e[0] == "move" for e in backend.events[down:up])     # touch + move = drag

    assert backend.down == set()
    assert camera.started == 1 and camera.released == 1
    assert tracker.closed == 1


def test_exit_during_a_pinch_leaves_no_button_held():
    rng = np.random.default_rng(1)
    tracker = FakeTracker([point_hand(rng) for _ in range(5)] + [pinch_hand(rng) for _ in range(10)])
    code, _, backend = run(tracker)
    assert code == 0
    assert backend.count("down", "left") == 1
    assert backend.down == set()
    assert backend.events[-1] == ("up", "left")


def test_two_hands_zoom_without_clicks():
    rng = np.random.default_rng(2)
    script = []
    for i in range(12):
        spread = 0.1 + i * 0.012
        script.append([point_hand(rng, 0.5 - spread), point_hand(rng, 0.5 + spread)])
    code, _, backend = run(FakeTracker(script), cfg=Config(max_hands=2))
    assert code == 0
    zooms = [e[1] for e in backend.events if e[0] == "zoom"]
    assert zooms and all(steps > 0 for steps in zooms)      # hands apart = zoom in
    assert backend.count("down") == 0


def test_timestamps_handed_to_the_tracker_increase():
    rng = np.random.default_rng(3)
    tracker = FakeTracker([point_hand(rng) for _ in range(10)])
    run(tracker, FakeCamera(period=0.002))
    assert all(isinstance(ts, int) for ts in tracker.stamps)
    assert tracker.stamps == sorted(tracker.stamps)


def test_process_frame_flips_before_tracking():
    frame = np.zeros((2, 4, 3), dtype=np.uint8)
    frame[:, 0] = 255       # left column white
    tracker = FakeTracker([None])
    engine = GestureEngine(Config(), FakeBackend())

    shown, hands, result = app.process_frame(frame, tracker, engine, 1.0, mirror=True)
    assert shown[0, 3].tolist() == [255, 255, 255] and shown[0, 0].tolist() == [0, 0, 0]
    assert tracker.seen_frames[-1] is shown         # what is tracked is what is shown
    assert hands == [] and result.hands == 0

    shown, _, _ = app.process_frame(frame, tracker, engine, 1.1, mirror=False)
    assert shown is frame


def test_process_frame_returns_engine_result_and_passes_time_in_ms():
    rng = np.random.default_rng(4)
    hand = point_hand(rng)
    tracker = FakeTracker([hand])
    engine = GestureEngine(Config(), FakeBackend())
    _, hands, result = app.process_frame(np.zeros((48, 64, 3), np.uint8), tracker, engine, 2.5)
    assert tracker.stamps == [2500]
    assert hands[0] is hand
    assert result.t == 2.5 and result.hands == 1 and result.raw_label == "point"


def test_only_each_new_frame_is_processed_once():
    rng = np.random.default_rng(5)
    tracker = FakeTracker([point_hand(rng) for _ in range(5)])
    camera = FakeCamera(period=0.02)
    run(tracker, camera)
    assert tracker.calls == 5
    assert camera.polls > 5     # the loop polled between frames, and did no work on those polls


# -- camera trouble ---------------------------------------------------------------------
class TimedBackend(FakeBackend):
    """Remembers when buttons were let go."""

    def __init__(self):
        super().__init__()
        self.release_times = []

    def release(self, button):
        super().release(button)
        self.release_times.append(time.perf_counter())


def test_stalled_camera_releases_a_held_button():
    rng = np.random.default_rng(6)
    tracker = FakeTracker([pinch_hand(rng) for _ in range(10)] + [None] * 5)
    backend = TimedBackend()
    down_at = {}

    camera = FakeCamera(gap_after=10, gap_s=0.7,
                        on_frame=lambda seq: down_at.__setitem__(seq, set(backend.down)))
    code, _, _ = run(tracker, camera, backend=backend, max_frames=15)
    assert code == 0
    assert down_at[10] == {"left"}      # held when the camera went silent
    assert down_at[11] == set()         # let go before any new frame arrived
    silent_for = backend.release_times[0] - camera.frame_times[10]
    assert silent_for >= app.STALL_RELEASE_S
    assert backend.count("up", "left") == 1


def test_button_is_let_go_the_moment_the_silence_passes_the_limit():
    rng = np.random.default_rng(31)
    backend = FakeBackend()
    engine = GestureEngine(Config(), backend)
    hand = pinch_hand(rng)
    for t in (99.8, 99.9, 100.0):
        engine.process_hands([hand], t)
    assert backend.down == {"left"}
    session = app.LiveSession(Config(), FakeCamera(), FakeTracker([None]), engine)
    session.last_frame_t = 100.0
    session._on_no_frame(100.0 + app.STALL_RELEASE_S - 0.01)
    assert backend.down == {"left"}
    session._on_no_frame(100.0 + app.STALL_RELEASE_S + 0.01)
    assert backend.down == set()
    session._on_no_frame(100.0 + app.STALL_RELEASE_S + 0.02)    # and on later passes nothing more happens
    assert backend.count("up", "left") == 1


def test_short_gap_keeps_a_drag_alive():
    rng = np.random.default_rng(7)
    tracker = FakeTracker([pinch_hand(rng) for _ in range(20)])
    backend = FakeBackend()
    down_at = {}
    camera = FakeCamera(gap_after=10, gap_s=0.1,
                        on_frame=lambda seq: down_at.__setitem__(seq, set(backend.down)))
    run(tracker, camera, backend=backend)
    assert down_at[11] == {"left"}      # not released during the gap...
    assert backend.count("down", "left") == 1
    assert backend.count("up", "left") == 1     # ...only at exit


def test_long_stall_says_so_once_and_says_when_it_is_over(monkeypatch, capsys):
    monkeypatch.setattr(app, "STALL_WARN_S", 0.2)
    rng = np.random.default_rng(8)
    tracker = FakeTracker([point_hand(rng) for _ in range(6)])
    run(tracker, FakeCamera(gap_after=3, gap_s=0.5))
    out = capsys.readouterr().out
    assert out.count("摄像头好像没有画面了") == 1
    assert out.count("摄像头又有画面了") == 1


def test_short_gap_prints_nothing_about_the_camera(capsys):
    rng = np.random.default_rng(9)
    run(FakeTracker([point_hand(rng) for _ in range(6)]), FakeCamera(gap_after=3, gap_s=0.05))
    assert "摄像头" not in capsys.readouterr().out


def test_idle_loop_does_not_spin():
    rng = np.random.default_rng(10)
    tracker = FakeTracker([point_hand(rng) for _ in range(3)])
    camera = FakeCamera(gap_after=2, gap_s=0.3)
    t0 = time.perf_counter()
    run(tracker, camera)
    elapsed = time.perf_counter() - t0
    # About 2 ms per pass, so no more than ~500 polls a second. A spinning loop does 100x that.
    assert camera.polls / elapsed < 800


# -- failures stay polite --------------------------------------------------------------
class NoCamera(FakeCamera):
    def start(self):
        raise RuntimeError("打不开摄像头。")


def test_camera_failure_is_a_plain_message_and_exit_code_1(capsys):
    tracker = FakeTracker([None])
    backend = FakeBackend()
    code = app.main(Config(), show_window=False, camera=NoCamera(), tracker=tracker, backend=backend)
    assert code == 1
    captured = capsys.readouterr()
    assert "打不开摄像头" in captured.out
    assert "Traceback" not in captured.out + captured.err
    assert tracker.closed == 1
    assert backend.events == []


def test_model_download_failure_is_a_plain_message(monkeypatch, capsys):
    def fail():
        raise RuntimeError("手部识别模型下载失败。请检查网络连接后再试一次。")

    monkeypatch.setattr(app, "ensure_model", fail)
    camera = FakeCamera()
    code = app.main(Config(), show_window=False, camera=camera, backend=FakeBackend())
    assert code == 1
    assert "下载失败" in capsys.readouterr().out
    assert camera.started == 0      # no point switching the camera on


def test_missing_model_file_is_a_plain_message(monkeypatch, capsys):
    monkeypatch.setattr(app, "ensure_model", lambda: None)

    def missing(cfg):
        raise FileNotFoundError("找不到手部识别模型")

    monkeypatch.setattr(app, "HandTracker", missing)
    camera = FakeCamera()
    code = app.main(Config(), show_window=False, camera=camera, backend=FakeBackend())
    assert code == 1
    assert "找不到手部识别模型" in capsys.readouterr().out
    assert camera.released == 1     # the camera that was already on is switched off again


def test_tracker_is_built_after_the_model_check(monkeypatch):
    order = []
    monkeypatch.setattr(app, "ensure_model", lambda: order.append("model"))

    def build(cfg):
        order.append("tracker")
        return FakeTracker([None])

    monkeypatch.setattr(app, "HandTracker", build)

    class Camera(FakeCamera):
        def start(self):
            order.append("camera")
            return super().start()

    code = app.main(Config(), show_window=False, camera=Camera(), backend=FakeBackend(), max_frames=2)
    assert code == 0
    assert order == ["model", "camera", "tracker"]


def test_error_in_the_loop_lets_go_of_the_mouse_and_exits_plainly(capsys):
    rng = np.random.default_rng(11)

    class BreaksLater(FakeTracker):
        def detect(self, frame, t_ms=None):
            if self.calls == 5:
                raise ValueError("kaput")
            return super().detect(frame, t_ms)

    tracker = BreaksLater([pinch_hand(rng) for _ in range(10)])
    code, camera, backend = run(tracker)
    assert code == 1
    assert backend.count("down", "left") == 1 and backend.down == set()
    assert camera.released == 1 and tracker.closed == 1
    captured = capsys.readouterr()
    assert "程序遇到问题" in captured.out and "kaput" in captured.out
    assert "Traceback" not in captured.out + captured.err


def test_ctrl_c_is_a_normal_exit_that_lets_go_of_the_mouse():
    rng = np.random.default_rng(12)

    class Interrupted(FakeTracker):
        def detect(self, frame, t_ms=None):
            if self.calls == 5:
                raise KeyboardInterrupt
            return super().detect(frame, t_ms)

    tracker = Interrupted([pinch_hand(rng) for _ in range(10)])
    code, camera, backend = run(tracker)
    assert code == 0
    assert backend.down == set()
    assert camera.released == 1 and tracker.closed == 1


def test_failure_while_cleaning_up_does_not_stop_the_rest():
    rng = np.random.default_rng(13)

    class BadClose(FakeTracker):
        def close(self):
            raise RuntimeError("native close failed")

    tracker = BadClose([pinch_hand(rng) for _ in range(5)])
    code, camera, backend = run(tracker)
    assert code == 0
    assert backend.down == set()
    assert camera.released == 1


# -- lite mode ----------------------------------------------------------------------------
def test_lite_mode_has_no_window_and_prints_speed_in_plain_words(capsys):
    # The fixture makes any cv2 window call raise, so this also proves none is made.
    rng = np.random.default_rng(14)
    tracker = FakeTracker([point_hand(rng) for _ in range(70)])
    code, _, _ = run(tracker, FakeCamera(period=0.02))     # 70 frames: about 1.4 s
    assert code == 0
    out = capsys.readouterr().out
    assert "Ctrl+C" in out
    assert "速度：每秒处理" in out
    assert "FPS" not in out


def test_lite_mode_does_not_draw(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("lite mode must not draw")

    monkeypatch.setattr(app, "draw_hud", forbidden)
    rng = np.random.default_rng(15)
    assert run(FakeTracker([point_hand(rng) for _ in range(5)]))[0] == 0


# -- window mode --------------------------------------------------------------------------
@pytest.fixture
def hud_calls(monkeypatch):
    """Records what the HUD is told on every frame, instead of drawing it."""
    calls = []

    def spy(img, result=None, hands=(), fps=0.0, engine_name="rule", active_box=None,
            enabled=True, dwell=False):
        calls.append(dict(result=result, hands=hands, engine_name=engine_name,
                          active_box=active_box, enabled=enabled, dwell=dwell))
        return img

    monkeypatch.setattr(app, "draw_hud", spy)
    return calls


def test_window_shows_every_frame_and_is_closed_at_the_end(monkeypatch, hud_calls):
    gui = FakeGui(monkeypatch)
    rng = np.random.default_rng(16)
    tracker = FakeTracker([point_hand(rng) for _ in range(8)])
    code, _, _ = run(tracker, show_window=True)
    assert code == 0
    assert gui.created == 1 and gui.shown == 8 and gui.destroyed == 1
    assert len(hud_calls) == 8
    assert hud_calls[0]["engine_name"] == "rule"
    assert hud_calls[0]["active_box"] is not None
    assert hud_calls[-1]["hands"] and hud_calls[-1]["enabled"] is True


def test_the_real_hud_can_draw_on_a_real_frame(monkeypatch):
    # No spy here: exercises draw_hud with what the loop really passes it.
    gui = FakeGui(monkeypatch)
    rng = np.random.default_rng(17)
    code, _, _ = run(FakeTracker([pinch_hand(rng) for _ in range(5)]), show_window=True)
    assert code == 0 and gui.shown == 5


def test_hud_can_be_switched_off_in_config(monkeypatch, hud_calls):
    gui = FakeGui(monkeypatch)
    rng = np.random.default_rng(18)
    run(FakeTracker([point_hand(rng) for _ in range(3)]), cfg=Config(show_hud=False), show_window=True)
    assert gui.shown == 3 and hud_calls == []


@pytest.mark.parametrize("key", [ord("q"), ord("Q"), 27])
def test_quit_keys(monkeypatch, hud_calls, key):
    gui = FakeGui(monkeypatch, keys={3: key})
    rng = np.random.default_rng(19)
    tracker = FakeTracker([pinch_hand(rng) for _ in range(50)])
    code, _, backend = run(tracker, show_window=True)
    assert code == 0
    assert tracker.calls == 3
    assert backend.down == set()
    assert gui.destroyed == 1


def test_closing_the_window_with_its_x_button_quits(monkeypatch, hud_calls):
    # Otherwise imshow would silently re-create the window and the app could never be closed.
    gui = FakeGui(monkeypatch, closed_after=4)
    rng = np.random.default_rng(20)
    tracker = FakeTracker([pinch_hand(rng) for _ in range(50)])
    code, _, backend = run(tracker, show_window=True)
    assert code == 0
    assert gui.shown == 4
    assert tracker.calls == 4
    assert backend.down == set()


def test_window_closed_between_frames_is_not_recreated(monkeypatch, hud_calls):
    class NativeWindow(FakeGui):
        visible = True
        recreated = 0

        def imshow(self, name, img):
            if not self.visible:
                self.recreated += 1
                self.visible = True
            super().imshow(name, img)

        def getWindowProperty(self, name, prop):
            return float(self.visible)

    gui = NativeWindow(monkeypatch)

    def camera_frame(seq):
        if seq == 4:
            gui.visible = False

    tracker = FakeTracker([None] * 8)
    camera = FakeCamera(on_frame=camera_frame)
    code, _, _ = run(tracker, camera, show_window=True)
    assert code == 0
    assert gui.recreated == 0
    assert tracker.calls == gui.shown == 3
    assert camera.released == tracker.closed == 1


def test_window_stays_responsive_and_closable_while_the_camera_is_silent(monkeypatch, hud_calls):
    gui = FakeGui(monkeypatch)
    rng = np.random.default_rng(21)
    tracker = FakeTracker([point_hand(rng) for _ in range(4)])
    camera = FakeCamera(gap_after=2, gap_s=0.2)
    run(tracker, camera, show_window=True)
    assert gui.waits > 20       # far more waitKey calls than the 4 frames shown


def test_error_reading_the_window_state_counts_as_closed(monkeypatch, hud_calls):
    gui = FakeGui(monkeypatch)

    def gone(name, prop):
        raise cv2.error("no such window")

    monkeypatch.setattr(cv2, "getWindowProperty", gone)
    rng = np.random.default_rng(22)
    tracker = FakeTracker([point_hand(rng) for _ in range(10)])
    assert run(tracker, show_window=True)[0] == 0
    assert tracker.calls == 1
    assert gui.destroyed == 1


def test_no_window_support_falls_back_to_lite_mode(monkeypatch, capsys):
    gui = FakeGui(monkeypatch)

    def unsupported(name, flags=None):
        raise cv2.error("The function is not implemented")

    monkeypatch.setattr(cv2, "namedWindow", unsupported)
    rng = np.random.default_rng(23)
    code, _, _ = run(FakeTracker([point_hand(rng) for _ in range(3)]), show_window=True)
    assert code == 0
    assert gui.shown == 0 and gui.destroyed == 0
    assert "Ctrl+C" in capsys.readouterr().out


def test_p_pauses_and_lets_go_then_resumes(monkeypatch, hud_calls, capsys):
    FakeGui(monkeypatch, keys={5: ord("p"), 10: ord("P")})
    rng = np.random.default_rng(24)
    code, _, backend = run(FakeTracker([pinch_hand(rng) for _ in range(15)]), show_window=True)
    assert code == 0
    # pressed on frame 1, let go by the pause, pressed again after it, let go at exit
    assert backend.count("down", "left") == 2
    assert backend.count("up", "left") == 2
    assert backend.down == set()
    assert [c["enabled"] for c in hud_calls] == [True] * 5 + [False] * 5 + [True] * 5
    out = capsys.readouterr().out
    assert "已暂停" in out and "继续" in out


def test_d_toggles_dwell_without_touching_the_callers_config(monkeypatch, hud_calls, capsys):
    FakeGui(monkeypatch, keys={2: ord("d"), 4: ord("D")})
    rng = np.random.default_rng(25)
    cfg = Config()
    run(FakeTracker([point_hand(rng) for _ in range(6)]), cfg=cfg, show_window=True)
    assert [c["dwell"] for c in hud_calls] == [False, False, True, True, False, False]
    assert cfg.dwell_enabled is False
    out = capsys.readouterr().out
    assert "已打开停留点击" in out and "已关闭停留点击" in out


class StubMlRecognizer:
    name = "ml"

    def predict(self, lm):
        return "none", 0.0


def test_m_without_a_trained_model_says_so_and_stays_on_rules(monkeypatch, hud_calls, capsys):
    FakeGui(monkeypatch, keys={2: ord("m")})
    wanted = []

    def fake_load(cfg):
        wanted.append(cfg.engine)
        return RuleRecognizer(cfg.aspect, cfg.pinch_on), "no trained model yet; using rules"

    monkeypatch.setattr(app, "load_recognizer", fake_load)
    rng = np.random.default_rng(26)
    run(FakeTracker([point_hand(rng) for _ in range(4)]), show_window=True)
    assert wanted == ["rule", "ml"]
    assert {c["engine_name"] for c in hud_calls} == {"rule"}
    assert "手势模型现在用不了" in capsys.readouterr().out


def test_m_switches_to_the_trained_model_and_back(monkeypatch, hud_calls, capsys):
    FakeGui(monkeypatch, keys={2: ord("m"), 4: ord("M")})
    rule = RuleRecognizer()

    def fake_load(cfg):
        return (StubMlRecognizer(), "ml") if cfg.engine == "ml" else (rule, "rule")

    monkeypatch.setattr(app, "load_recognizer", fake_load)
    rng = np.random.default_rng(27)
    run(FakeTracker([point_hand(rng) for _ in range(6)]), show_window=True)
    assert [c["engine_name"] for c in hud_calls] == ["rule", "rule", "ml", "ml", "rule", "rule"]
    assert "手势模型现在用不了" not in capsys.readouterr().out


def test_m_lets_go_of_the_mouse_before_loading_a_model(monkeypatch, hud_calls):
    FakeGui(monkeypatch, keys={3: ord("m")})
    backend = FakeBackend()
    down_while_loading = []

    def fake_load(cfg):
        if cfg.engine == "ml":
            down_while_loading.append(set(backend.down))
        return RuleRecognizer(cfg.aspect, cfg.pinch_on), "rule"

    monkeypatch.setattr(app, "load_recognizer", fake_load)
    rng = np.random.default_rng(28)
    run(FakeTracker([pinch_hand(rng) for _ in range(5)]), backend=backend, show_window=True)
    assert down_while_loading == [set()]


def test_asking_for_ml_at_start_without_a_model_explains_the_fallback(monkeypatch, capsys):
    monkeypatch.setattr(app, "load_recognizer",
                        lambda cfg: (RuleRecognizer(cfg.aspect, cfg.pinch_on), "no trained model yet; using rules"))
    rng = np.random.default_rng(29)
    code, _, _ = run(FakeTracker([point_hand(rng) for _ in range(2)]), cfg=Config(engine="ml"))
    assert code == 0
    out = capsys.readouterr().out
    assert "手势模型现在用不了" in out and "规则识别" in out


def test_startup_message_is_plain_chinese(monkeypatch, hud_calls, capsys):
    FakeGui(monkeypatch)
    rng = np.random.default_rng(30)
    run(FakeTracker([point_hand(rng) for _ in range(2)]), show_window=True)
    out = capsys.readouterr().out
    assert "AirTouch 已启动" in out and "规则识别" in out
    assert "按 Q 或 Esc" in out and "已退出" in out
    for jargon in ("FPS", "debounce", "landmark", "One-Euro", "MediaPipe"):
        assert jargon not in out


# -- picture shape -----------------------------------------------------------------------
@pytest.mark.parametrize("width,height,aspect", [
    (1280, 720, 16 / 9), (1920, 1200, 1.6), (3840, 2160, 16 / 9),
])
def test_config_follows_the_shape_the_camera_really_delivers(width, height, aspect):
    camera = FakeCamera()
    camera.frame_size = (width, height)
    cfg = app._match_camera_shape(Config(), camera)
    assert (cfg.frame_width, cfg.frame_height) == (width, height)
    assert cfg.aspect == pytest.approx(aspect)


def test_config_is_unchanged_when_the_camera_delivers_what_was_asked():
    camera = FakeCamera()
    camera.frame_size = (640, 480)
    cfg = Config()
    assert app._match_camera_shape(cfg, camera) is cfg
    assert app._match_camera_shape(cfg, FakeCamera()) is cfg     # a camera that doesn't say


# -- the real tracker, on made-up frames ------------------------------------------------------
@pytest.mark.skipif(not HAND_MODEL_PATH.is_file(), reason="hand model not downloaded yet")
def test_whole_app_with_the_real_tracker_and_no_hand():
    camera = FakeCamera(period=0.005)
    backend = FakeBackend()
    code = app.main(Config(), show_window=False, camera=camera, backend=backend, max_frames=5)
    assert code == 0
    assert camera.released == 1
    assert backend.count("down") == 0


# -- hand_mouse.py ---------------------------------------------------------------------------
@pytest.fixture
def launched(monkeypatch):
    calls = []

    def fake_main(cfg=None, show_window=True, **kwargs):
        calls.append((cfg, show_window))
        return 7

    monkeypatch.setattr(app, "main", fake_main)
    return calls


def test_hand_mouse_defaults(launched, tmp_path):
    assert hand_mouse.main(["--config", str(tmp_path / "none.json")]) == 7
    cfg, show_window = launched[0]
    assert show_window is True
    assert cfg == Config()


def test_hand_mouse_flags_override_config(launched, tmp_path):
    settings = tmp_path / "设置.json"
    settings.write_text('{"camera_index": 1, "engine": "ml", "smoothing": 2.0}', encoding="utf-8")
    hand_mouse.main(["--config", str(settings), "--lite", "--engine", "rule", "--camera", "2",
                     "--two-hands", "--dwell"])
    cfg, show_window = launched[0]
    assert show_window is False
    assert cfg.camera_index == 2 and cfg.engine == "rule" and cfg.max_hands == 2
    assert cfg.dwell_enabled is True
    assert cfg.smoothing == pytest.approx(2.0)      # what the flags don't mention stays


def test_hand_mouse_keeps_config_values_when_flags_are_absent(launched, tmp_path):
    settings = tmp_path / "config.json"
    settings.write_text('{"camera_index": 1, "engine": "ml"}', encoding="utf-8")
    hand_mouse.main(["--config", str(settings)])
    cfg, _ = launched[0]
    assert cfg.camera_index == 1 and cfg.engine == "ml" and cfg.max_hands == 1


def test_hand_mouse_camera_zero_is_respected(launched, tmp_path):
    settings = tmp_path / "config.json"
    settings.write_text('{"camera_index": 1}', encoding="utf-8")
    hand_mouse.main(["--config", str(settings), "--camera", "0"])
    assert launched[0][0].camera_index == 0


def test_hand_mouse_help_runs_as_a_script_and_is_quick():
    t0 = time.perf_counter()
    done = subprocess.run([sys.executable, str(PROJECT_ROOT / "hand_mouse.py"), "--help"],
                          capture_output=True, encoding="utf-8", cwd=PROJECT_ROOT, timeout=60)
    assert done.returncode == 0, done.stderr
    for flag in ("--lite", "--engine", "--camera", "--two-hands", "--dwell", "--config"):
        assert flag in done.stdout
    assert "摄像头" in done.stdout
    assert time.perf_counter() - t0 < 30


# -- diagnose_speed.py -------------------------------------------------------------------------
class TimingTracker:
    def __init__(self):
        self.calls = 0
        self.closed = 0

    def detect(self, frame, t_ms=None):
        self.calls += 1
        return []

    def close(self):
        self.closed += 1


class BrightCamera(FakeCamera):
    def __init__(self, level=128, **kwargs):
        super().__init__(**kwargs)
        self.level = level

    def read_if_new(self, last_seq):
        seq, frame = super().read_if_new(last_seq)
        return seq, None if frame is None else np.full_like(frame, self.level)


def test_diagnose_no_camera_measures_the_computer_only(capsys, tmp_path):
    tracker = TimingTracker()
    assert diagnose_speed.main(["--no-camera"], tracker=tracker) == 0
    out = capsys.readouterr().out
    assert tracker.calls > 10 and tracker.closed == 1
    assert "这次没有打开摄像头" in out
    assert "电脑生成的测试画面" in out
    assert "每秒大约能处理" in out


def test_diagnose_measures_a_camera(capsys):
    camera = BrightCamera(level=128, period=0.005)
    tracker = TimingTracker()
    code = diagnose_speed.main(["--seconds", "0.3"], camera=camera, tracker=tracker)
    assert code == 0
    assert camera.released == 1 and camera.started == 1
    out = capsys.readouterr().out
    assert "每秒送来" in out and "摄像头拍到的画面" in out
    assert "画面亮度：128" in out
    assert "全黑" not in out.replace("0 是全黑", "") and "比较暗" not in out


def test_diagnose_without_a_camera_still_measures_the_computer(capsys):
    tracker = TimingTracker()
    code = diagnose_speed.main([], camera=NoCamera(), tracker=tracker)
    assert code == 0
    out = capsys.readouterr().out
    assert "打不开摄像头" in out
    assert tracker.calls > 10
    assert "电脑生成的测试画面" in out
    assert "Traceback" not in out


def test_diagnose_camera_that_sends_nothing_does_not_crash(capsys):
    class Silent(FakeCamera):
        def read_if_new(self, last_seq):
            return 0, None

    code = diagnose_speed.main(["--seconds", "0.05"], camera=Silent(), tracker=TimingTracker())
    assert code == 0
    assert "一张画面都没有送来" in capsys.readouterr().out


def test_diagnose_missing_model_does_not_crash(monkeypatch, capsys):
    def missing(cfg):
        raise FileNotFoundError("找不到手部识别模型")

    monkeypatch.setattr(diagnose_speed, "HandTracker", missing)
    assert diagnose_speed.main(["--no-camera"]) == 0
    assert "找不到手部识别模型" in capsys.readouterr().out


def test_diagnose_broken_tracker_does_not_crash(capsys):
    class Broken(TimingTracker):
        def detect(self, frame, t_ms=None):
            raise RuntimeError("native failure")

    tracker = Broken()
    assert diagnose_speed.main(["--no-camera"], tracker=tracker) == 0
    assert "native failure" in capsys.readouterr().out
    assert tracker.closed == 1


def test_diagnose_report_is_saved_in_a_non_ascii_folder(tmp_path, capsys):
    report = tmp_path / "【 报告 】" / "结果.txt"
    assert diagnose_speed.main(["--no-camera", "--report", str(report)], tracker=TimingTracker()) == 0
    text = report.read_text(encoding="utf-8")
    assert "测量结果" in text and "结论" in text


def test_diagnose_saves_nothing_without_report(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    diagnose_speed.main(["--no-camera"], tracker=TimingTracker())
    assert list(tmp_path.iterdir()) == []


def test_diagnose_unwritable_report_does_not_crash(tmp_path, capsys):
    blocked = tmp_path / "file"
    blocked.write_text("x")
    code = diagnose_speed.main(["--no-camera", "--report", str(blocked / "sub" / "r.txt")],
                               tracker=TimingTracker())
    assert code == 0
    assert "没能保存" in capsys.readouterr().out


def measured(**kwargs):
    defaults = dict(camera_tested=True, camera_fps=30.0, brightness=120.0, convert_ms=0.3, tracker_ms=15.0)
    return diagnose_speed.Measurements(**{**defaults, **kwargs})


def verdict_text(**kwargs) -> str:
    return "\n".join(diagnose_speed.verdict(measured(**kwargs)))


def test_verdict_all_good():
    text = verdict_text()
    assert "摄像头的速度正常" in text and "完全够用" in text and "很流畅" in text
    assert "run.bat" not in text


def test_verdict_black_picture_points_at_the_lens_cover():
    text = verdict_text(brightness=2.0, camera_fps=30.0)
    assert "全黑" in text and "镜头盖" in text and "隐私开关" in text


def test_verdict_dim_room_says_light_is_the_biggest_lever():
    text = verdict_text(brightness=30.0, camera_fps=9.0)
    assert "开灯" in text and "最有效" in text


def test_verdict_slow_camera_in_good_light_still_mentions_light():
    text = verdict_text(camera_fps=10.0)
    assert "摄像头偏慢" in text and "光线" in text


def test_verdict_slow_computer_names_tracking_as_the_limit():
    text = verdict_text(tracker_ms=120.0)
    assert "最慢的一步" in text and "run.bat --lite" in text and "关掉不用的程序" in text
    assert "卡顿" in text


def test_verdict_middling_computer_gets_the_tips_too():
    assert "run.bat --lite" in verdict_text(tracker_ms=50.0)


def test_expected_speed_is_set_by_the_slowest_step():
    assert diagnose_speed.expected_speed(measured(camera_fps=10.0)) == pytest.approx(10.0)
    assert diagnose_speed.expected_speed(measured(tracker_ms=99.7, convert_ms=0.3)) == pytest.approx(10.0)
    assert diagnose_speed.expected_speed(measured(camera_fps=None)) == pytest.approx(1000 / 15.3)
    assert diagnose_speed.expected_speed(measured(camera_fps=None, tracker_ms=None)) is None


def test_verdicts_have_no_jargon():
    cases = [dict(), dict(brightness=1.0), dict(brightness=30.0), dict(camera_fps=8.0),
             dict(tracker_ms=150.0), dict(camera_tested=False, camera_fps=None, brightness=None),
             dict(camera_error="打不开摄像头。", camera_fps=None, brightness=None)]
    for case in cases:
        m = measured(**case)
        text = "\n".join(diagnose_speed.summary(m) + diagnose_speed.verdict(m))
        for jargon in ("FPS", "fps", "landmark", "MediaPipe", "debounce", "RGB", "BGR"):
            assert jargon not in text


def test_project_root_is_where_the_entry_points_live():
    assert (Path(PROJECT_ROOT) / "hand_mouse.py").is_file()
    assert (Path(PROJECT_ROOT) / "diagnose_speed.py").is_file()
