"""Turns (debounced gesture, landmarks) into real mouse actions.

Relative mode distinguishes a short tap from a sustained drag. A confirmed
pinch waits 250 ms without holding an OS button; opening sooner clicks once,
while keeping it closed starts a drag. The click target stays still while
the pinch is pending. Absolute mode retains physical mouse-button behavior:

    fingers touch  -> button DOWN on that very frame
    fingers part   -> button UP on that very frame
    touch + move   -> the OS sees a drag

The hysteresis gap between ``pinch_on`` and ``pinch_off`` stops a
half-closed pinch from chattering. A finger tucked into a fist can never
press (see ``gestures_rule.PRESS_REACH``).

Backends
--------
Win32Backend   ctypes SendInput - fastest, no extra dependency (Windows)
PynputBackend  cross-platform fallback
FakeBackend    records events; used by tests, the simulator and research
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass, field

import numpy as np

from . import gestures_rule as gr
from .config import Config
from .landmarks import INDEX_TIP, MIDDLE_TIP, anchor_point, is_valid, to_metric
from .smoothing import OneEuroFilter

MOVE_LABELS = ("point", "pinch_index", "pinch_middle")
BUTTON_BLOCK_LABELS = ("fist", "scroll")   # these poses may never start a click


# ----------------------------------------------------------------- backends
class FakeBackend:
    """In-memory mouse. ``events`` is a list of tuples, newest last."""

    def __init__(self, width: int = 1920, height: int = 1080):
        self.size = (width, height)
        self.pos = (width // 2, height // 2)
        self.events: list[tuple] = []
        self.down: set[str] = set()

    def screen_size(self) -> tuple[int, int]:
        return self.size

    def screen_bounds(self) -> tuple[int, int, int, int]:
        return 0, 0, *self.screen_size()

    def cursor_position(self) -> tuple[int, int]:
        return self.pos

    def move(self, x: int, y: int) -> None:
        self.pos = (x, y)
        self.events.append(("move", x, y))

    def press(self, button: str) -> None:
        self.down.add(button)
        self.events.append(("down", button))

    def release(self, button: str) -> None:
        self.down.discard(button)
        self.events.append(("up", button))

    def scroll(self, notches: float) -> None:
        self.events.append(("scroll", notches))

    def zoom(self, steps: float) -> None:
        self.events.append(("zoom", steps))

    def count(self, kind: str, arg=None) -> int:
        return sum(1 for e in self.events if e[0] == kind and (arg is None or e[1] == arg))


class Win32Backend:
    """Direct Win32 input via ctypes. Makes the process DPI-aware so cursor
    coordinates are real pixels even with 125%/150% display scaling."""

    def __init__(self):
        import ctypes
        from ctypes import wintypes

        self._ct = ctypes
        user32 = ctypes.WinDLL("user32", use_last_error=True)
        self._user32 = user32
        try:
            ctypes.windll.shcore.SetProcessDpiAwareness(2)   # per-monitor aware
        except Exception:
            try:
                user32.SetProcessDPIAware()
            except Exception:
                pass

        ULONG_PTR = ctypes.c_size_t

        class MOUSEINPUT(ctypes.Structure):
            _fields_ = [("dx", wintypes.LONG), ("dy", wintypes.LONG), ("mouseData", wintypes.DWORD),
                        ("dwFlags", wintypes.DWORD), ("time", wintypes.DWORD), ("dwExtraInfo", ULONG_PTR)]

        class KEYBDINPUT(ctypes.Structure):
            _fields_ = [("wVk", wintypes.WORD), ("wScan", wintypes.WORD), ("dwFlags", wintypes.DWORD),
                        ("time", wintypes.DWORD), ("dwExtraInfo", ULONG_PTR)]

        class HARDWAREINPUT(ctypes.Structure):
            _fields_ = [("uMsg", wintypes.DWORD), ("wParamL", wintypes.WORD), ("wParamH", wintypes.WORD)]

        class _U(ctypes.Union):
            _fields_ = [("mi", MOUSEINPUT), ("ki", KEYBDINPUT), ("hi", HARDWAREINPUT)]

        class INPUT(ctypes.Structure):
            _fields_ = [("type", wintypes.DWORD), ("u", _U)]

        self._INPUT, self._MOUSEINPUT, self._KEYBDINPUT = INPUT, MOUSEINPUT, KEYBDINPUT
        user32.SendInput.argtypes = (wintypes.UINT, ctypes.POINTER(INPUT), ctypes.c_int)
        user32.SendInput.restype = wintypes.UINT
        user32.SetCursorPos.argtypes = (ctypes.c_int, ctypes.c_int)
        user32.SetCursorPos.restype = wintypes.BOOL
        self._POINT = wintypes.POINT
        user32.GetCursorPos.argtypes = (ctypes.POINTER(wintypes.POINT),)
        user32.GetCursorPos.restype = wintypes.BOOL
        self._wheel_rest = 0.0
        self._ctrl_down = False

    _FLAGS = {"left": (0x0002, 0x0004), "right": (0x0008, 0x0010)}

    def _send_mouse(self, flags: int, data: int = 0) -> None:
        inp = self._INPUT(type=0)
        inp.u.mi = self._MOUSEINPUT(0, 0, data & 0xFFFFFFFF, flags, 0, 0)
        if self._user32.SendInput(1, self._ct.byref(inp), self._ct.sizeof(inp)) != 1:
            raise OSError("Windows rejected mouse input (SendInput)")

    def _send_key(self, vk: int, up: bool) -> None:
        inp = self._INPUT(type=1)
        inp.u.ki = self._KEYBDINPUT(vk, 0, 0x0002 if up else 0, 0, 0)
        if self._user32.SendInput(1, self._ct.byref(inp), self._ct.sizeof(inp)) != 1:
            # UIPI can block this API without setting a useful last-error code.
            raise OSError("Windows rejected keyboard input (SendInput)")
        if vk == 0x11:
            self._ctrl_down = not up

    def screen_size(self) -> tuple[int, int]:
        return self._user32.GetSystemMetrics(0), self._user32.GetSystemMetrics(1)

    def screen_bounds(self) -> tuple[int, int, int, int]:
        # The virtual desktop can start left of or above the primary screen.
        return tuple(self._user32.GetSystemMetrics(index) for index in (76, 77, 78, 79))

    def cursor_position(self) -> tuple[int, int]:
        point = self._POINT()
        if not self._user32.GetCursorPos(self._ct.byref(point)):
            raise self._ct.WinError()
        return int(point.x), int(point.y)

    def move(self, x: int, y: int) -> None:
        if not self._user32.SetCursorPos(int(x), int(y)):
            raise OSError("Windows rejected cursor movement (SetCursorPos)")

    def press(self, button: str) -> None:
        self._send_mouse(self._FLAGS[button][0])

    def release(self, button: str) -> None:
        self._send_mouse(self._FLAGS[button][1])

    def _wheel(self, notches: float) -> None:
        total = notches * 120 + self._wheel_rest
        units = int(total)
        self._wheel_rest = total - units
        if units:
            self._send_mouse(0x0800, units)

    def scroll(self, notches: float) -> None:
        self._wheel(notches)

    def zoom(self, steps: float) -> None:
        self._send_key(0x11, up=False)        # Ctrl down
        try:
            self._wheel(steps)
        finally:
            self._send_key(0x11, up=True)

    def release_modifiers(self) -> None:
        """Retry only a modifier this backend pressed but could not release."""
        if self._ctrl_down:
            self._send_key(0x11, up=True)


class PynputBackend:
    def __init__(self):
        from pynput.keyboard import Controller as Kb, Key
        from pynput.mouse import Button, Controller

        self._m, self._kb, self._key = Controller(), Kb(), Key
        self._buttons = {"left": Button.left, "right": Button.right}
        self._rest = 0.0

    def screen_size(self) -> tuple[int, int]:
        try:
            import tkinter

            root = tkinter.Tk()
            root.withdraw()
            size = root.winfo_screenwidth(), root.winfo_screenheight()
            root.destroy()
            return size
        except Exception:
            return 1920, 1080

    def move(self, x: int, y: int) -> None:
        self._m.position = (int(x), int(y))

    def cursor_position(self) -> tuple[int, int]:
        x, y = self._m.position
        return int(x), int(y)

    def screen_bounds(self) -> tuple[int, int, int, int]:
        return 0, 0, *self.screen_size()

    def press(self, button: str) -> None:
        self._m.press(self._buttons[button])

    def release(self, button: str) -> None:
        self._m.release(self._buttons[button])

    def scroll(self, notches: float) -> None:
        total = notches + self._rest
        whole = int(total)
        self._rest = total - whole
        if whole:
            self._m.scroll(0, whole)

    def zoom(self, steps: float) -> None:
        with self._kb.pressed(self._key.ctrl):
            self.scroll(steps)


def make_backend():
    """Best real backend for this OS."""
    if sys.platform == "win32":
        try:
            return Win32Backend()
        except Exception:
            pass
    return PynputBackend()


# --------------------------------------------------------------- controller
@dataclass
class ControlResult:
    cursor: tuple[int, int] | None = None
    events: list[str] = field(default_factory=list)
    button: str | None = None


class MouseController:
    DISPLAY_REFRESH_S = 0.25
    PINCH_DRAG_HOLD_S = 0.25
    DISPLAY_UNAVAILABLE = "显示器尺寸暂时不可用，请恢复显示器后重新开启。"

    def __init__(self, cfg: Config, backend):
        self.cfg = cfg
        self.backend = backend
        self.screen = tuple(backend.screen_size())
        if not self._valid_geometry(self.screen, 2, 0):
            raise RuntimeError(self.DISPLAY_UNAVAILABLE)
        self._display_checked_t: float | None = None
        self._display_bounds: tuple[int, int, int, int] | None = None
        self._display_failed = False
        self.filter = OneEuroFilter(cfg.smoothing, cfg.beta, cfg.d_cutoff)
        self.button: str | None = None
        self._pending_button: str | None = None
        self._pending_t0 = 0.0
        self._pending_target: tuple[int, int] | None = None
        self._loose_frames = 0
        self.cursor: tuple[int, int] | None = None
        self._moving = False
        self._relative_hand: np.ndarray | None = None
        self._relative_pos: np.ndarray | None = None
        self._relative_bounds: tuple[int, int, int, int] | None = None
        self._relative_anchor_kind: str | None = None
        self._scroll_y: float | None = None
        self._scroll_ref: float | None = None
        self._dwell_anchor: tuple[int, int] | None = None
        self._dwell_t0 = 0.0
        self._dwell_armed = True
        self._box = self._active_box()

    @property
    def active_box(self) -> tuple[float, float, float, float]:
        """(x0, y0, w, h), normalised: the camera region mapped onto the screen."""
        return self._box

    def halt(self, events: list[str] | None = None) -> None:
        """Let go of everything: release buttons, stop the cursor, forget scrolling."""
        try:
            self.release_all(events)
        finally:
            try:
                release_modifiers = getattr(self.backend, "release_modifiers", None)
                if release_modifiers is not None:
                    release_modifiers()
            finally:
                self._stop_motion()
                self._scroll_y = None

    def suspend_motion(self) -> None:
        """Forget the cursor stroke while retaining a button through tracker flicker."""
        self._stop_motion()

    # -- mapping -------------------------------------------------------------
    @staticmethod
    def _valid_geometry(values, length: int, positive_from: int) -> bool:
        return (len(values) == length
                and all(isinstance(value, (int, np.integer))
                        and not isinstance(value, (bool, np.bool_)) for value in values)
                and all(value > 0 for value in values[positive_from:]))

    def _refresh_display(self, t: float, force: bool = False) -> bool:
        # Win32 geometry queries are cheap, but the fallback backend uses Tk.
        # Poll at a bounded rate rather than creating a toolkit root each frame.
        if self._display_checked_t == t:
            return False              # already checked for this frame's action
        if (not force and self._display_checked_t is not None
                and 0 <= t - self._display_checked_t < self.DISPLAY_REFRESH_S):
            return False
        try:
            screen = tuple(self.backend.screen_size())
            bounds = (tuple(self.backend.screen_bounds())
                      if self.cfg.cursor_mode == "relative" else None)
            if (not self._valid_geometry(screen, 2, 0)
                    or bounds is not None and not self._valid_geometry(bounds, 4, 2)):
                raise ValueError("invalid display geometry")
        except Exception as error:
            self._display_checked_t = None
            self._display_failed = True
            self.halt()
            raise RuntimeError(self.DISPLAY_UNAVAILABLE) from error
        changed = (self._display_failed or screen != self.screen
                   or self._display_bounds is not None and bounds != self._display_bounds)
        self._display_checked_t = t
        self._display_bounds = bounds
        self._display_failed = False
        if changed:
            self.screen = screen
            self._box = self._active_box()
            self._stop_motion()
            self._scroll_y = None
        return changed

    def _active_box(self) -> tuple[float, float, float, float]:
        """Region of the camera image (x0, y0, w, h, normalised) mapped to the screen.

        Its shape matches the SCREEN's shape in physical camera units, so the
        same hand movement moves the cursor the same distance horizontally and
        vertically (a 4:3 camera on a 16:9 screen would otherwise feel ~1.3x
        faster sideways).
        """
        sw, sh = self.screen
        cam_aspect = self.cfg.aspect
        scr_aspect = sw / max(sh, 1)
        h = 1.0 - 2 * self.cfg.frame_margin
        w = h * scr_aspect / cam_aspect
        if w > 0.96:
            w = 0.96
            h = w * cam_aspect / scr_aspect
        return 0.5 - w / 2, 0.5 - h / 2, w, h

    def to_screen(self, nx: float, ny: float) -> tuple[int, int]:
        x0, y0, w, h = self._box
        sw, sh = self.screen
        u = min(max((nx - x0) / w, 0.0), 1.0)
        v = min(max((ny - y0) / h, 0.0), 1.0)
        return int(round(u * (sw - 1))), int(round(v * (sh - 1)))

    # -- buttons ---------------------------------------------------------------
    def release_all(self, events: list[str] | None = None) -> None:
        self._cancel_pending_pinch()
        if self.button is not None:
            self.backend.release(self.button)
            if events is not None:
                events.append(f"up:{self.button}")
            self.button = None

    def _cancel_pending_pinch(self) -> None:
        self._pending_button = None
        self._pending_t0 = 0.0
        self._pending_target = None

    def _update_relative_pinch(self, label: str, lm: np.ndarray, m: np.ndarray,
                               t: float, events: list[str], allow_press: bool) -> bool:
        if self.button is not None:
            return False
        if self._pending_button is not None:
            button = self._pending_button
            finger = "index" if button == "left" else "middle"
            elapsed = t - self._pending_t0
            reach = gr.finger_reach(m, finger)
            tap_off = (self.cfg.pinch_on + self.cfg.pinch_off) / 2
            opened = gr.pinch_gap(m, button) >= tap_off
            # Opening a bent index finger can transiently be labelled a fist
            # by the 2-D recognizer. An opened tap must also complete on the
            # first camera frame past the hold deadline. A prolonged closed
            # ambiguous pose still cancels rather than starting a drag.
            tap_fist = (label == "fist" and elapsed >= 0
                        and reach > max(1.0, gr.PRESS_REACH[button])
                        and (opened or elapsed < self.PINCH_DRAG_HOLD_S))
            # Losing the target or an explicit pause cancels, rather than
            # completing a tap at an unrelated position.
            if (label in ("none", "scroll") or label == "fist" and not tap_fist
                    or (label in ("pinch_index", "pinch_middle")
                        and label != ("pinch_index" if button == "left" else "pinch_middle"))
                    or elapsed < 0
                    or tuple(self.backend.cursor_position()) != self._pending_target
                    or reach <= gr.HOLD_REACH[button]):
                self._refresh_display(t, force=True)
                self._cancel_pending_pinch()
                self._stop_motion()
                return False
            # A tap needs less opening than an established drag. This still
            # leaves hysteresis above pinch_on, so small gap noise does not
            # split one tap. Held drags keep the full pinch_off threshold.
            if opened or elapsed >= self.PINCH_DRAG_HOLD_S:
                if self._refresh_display(t, force=True):
                    return False              # the remap cancelled the old target
                self._cancel_pending_pinch()
                self.backend.press(button)
                self.button = button          # retain if the following release fails
                self._loose_frames = 0
                events.append(f"down:{button}")
                if opened:
                    self.release_all(events)
                    self._stop_motion()       # do not apply tap-time hand travel afterwards
                # For a drag, the caller now moves only AFTER the button down,
                # applying the palm travel since the original click target.
            return False
        if allow_press and label not in BUTTON_BLOCK_LABELS:
            button = gr.which_pinch(m, self.cfg.pinch_on)
            if button is not None:
                if self._refresh_display(t, force=True):
                    return False
                self._dwell_anchor = None
                self._stop_motion()
                self._move(lm, t, events, dwell=False, anchor_kind="palm")
                self._pending_button = button
                self._pending_t0 = t
                self._pending_target = self.cursor
        return False

    def _update_buttons(self, label: str, lm: np.ndarray, t: float, events: list[str],
                        allow_press: bool = True) -> bool:
        """Update live pinches; return whether a new press already moved the cursor."""
        m = to_metric(lm, self.cfg.aspect)
        if self.button is not None:
            # Release at once when the fingers clearly part (> pinch_off) or the
            # finger tucks into a fist. Relative drags retain the full hysteresis
            # gap; absolute mode keeps its legacy loose-pinch timeout.
            if self.cfg.cursor_mode == "absolute":
                gap = gr.pinch_gap(m, self.button)
                self._loose_frames = self._loose_frames + 1 if gap > self.cfg.pinch_on else 0
            if (not gr.pinch_held(m, self.button, self.cfg.pinch_off)
                    or (self.cfg.cursor_mode == "absolute"
                        and self._loose_frames >= self.cfg.debounce_frames)):
                self.release_all(events)
        if self.cfg.cursor_mode == "relative":
            return self._update_relative_pinch(label, lm, m, t, events, allow_press)
        if allow_press and self.button is None and label not in BUTTON_BLOCK_LABELS:
            btn = gr.which_pinch(m, self.cfg.pinch_on)
            if btn is not None:
                if self._refresh_display(t, force=True):
                    return False       # don't click using a remapped old target
                self._dwell_anchor = None   # manual input cancels the pending dwell
                # Absolute mode targets this frame's hand position. Relative
                # mode resumes at the actual pointer before pressing.
                self._move(lm, t, events, dwell=False, anchor_kind="palm")
                self.backend.press(btn)
                self.button = btn
                self._loose_frames = 0
                events.append(f"down:{btn}")
                return True
        return False

    # -- main entry --------------------------------------------------------------
    def update(self, label: str, lm: np.ndarray | None, t: float,
               allow_press: bool = True, allow_dwell: bool = True,
               preserve_anchor: bool = False) -> ControlResult:
        """One frame; new pinches/dwell can be gated without breaking a held drag."""
        res = ControlResult()
        if lm is not None and not is_valid(lm):
            lm = None
        kind = "palm" if self.button is not None or self._pending_button is not None or label != "point" else "tip"
        if preserve_anchor and self._relative_anchor_kind is not None:
            kind = self._relative_anchor_kind
        new_stroke = ((label in MOVE_LABELS or self.button is not None)
                      and (not self._moving or self.cfg.cursor_mode == "relative"
                           and self._relative_anchor_kind != kind))
        dwell_due = (allow_dwell and self.cfg.dwell_enabled and self.button is None and self._pending_button is None
                     and self._dwell_armed and self._dwell_anchor is not None
                     and label in MOVE_LABELS
                     and (t - self._dwell_t0) * 1000 >= self.cfg.dwell_ms)
        if lm is not None and self._refresh_display(t, force=new_stroke or dwell_due):
            # A remap must not turn an old target/dwell timer into a new click.
            # Existing buttons still follow their live release geometry.
            allow_press = allow_dwell = False
        if lm is None:
            # hand gone: let go of everything and freeze the cursor
            self.halt(res.events)
        elif (self.button is None
              and (label in ("none", "fist")
                   or (self.cfg.cursor_mode == "relative" and label == "open_palm"))
              and not (self.cfg.cursor_mode == "relative" and self._pending_button is not None
                       and label == "fist")):
            # pause (fist) / unknown pose: nothing happens
            self._stop_motion()
            self._scroll_y = None
        else:
            # A held button is governed only by the live finger distance, so a
            # slightly loose pinch during a drag never drops the button even if
            # the gesture label wobbles.
            new_press = self._update_buttons(label, lm, t, res.events, allow_press)
            if self._pending_button is not None:
                self._scroll_y = None          # a pending tap keeps its original target
            elif self.button is not None:
                self._scroll_y = None
                if not new_press:
                    self._move(lm, t, res.events, anchor_kind="palm")
            elif label == "scroll":
                self._stop_motion()
                self._scroll(lm, res.events)
            else:
                self._scroll_y = None
                if label in MOVE_LABELS:
                    kind = "tip" if label == "point" else "palm"
                    if preserve_anchor and self._relative_anchor_kind is not None:
                        kind = self._relative_anchor_kind
                    self._move(lm, t, res.events, dwell=allow_dwell, anchor_kind=kind)
                else:                       # open_palm = idle, cursor stays put
                    self._stop_motion()
        res.cursor, res.button = self.cursor, self.button
        return res

    def _stop_motion(self) -> None:
        # A new stroke has no filter history. Absolute mode maps the new hand
        # position; relative mode anchors it at the current OS pointer.
        self._cancel_pending_pinch()
        if self._moving:
            self.filter.reset()
            self._moving = False
            self._dwell_anchor = None
        self._relative_hand = None
        self._relative_pos = None
        self._relative_bounds = None
        self._relative_anchor_kind = None

    def _move(self, lm: np.ndarray, t: float, events: list[str], dwell: bool = True,
              anchor_kind: str = "palm") -> None:
        relative = self.cfg.cursor_mode == "relative"
        if relative and self._relative_anchor_kind != anchor_kind:
            # Tip and palm coordinates are different reference frames. Start a
            # fresh stroke at the actual pointer without releasing a held button.
            self._stop_motion()
            self._relative_anchor_kind = anchor_kind
        resuming = not self._moving
        self._moving = True
        anchor = lm[INDEX_TIP, :2] if relative and anchor_kind == "tip" else anchor_point(lm)
        p = self.filter(anchor, t)
        if relative:
            self._move_relative(p, resuming)
        else:
            x, y = self.to_screen(float(p[0]), float(p[1]))
            if resuming or (x, y) != self.cursor:
                self.backend.move(x, y)
                self.cursor = (x, y)
        if dwell and self.cfg.dwell_enabled and self.button is None:
            self._dwell(t, events)

    def _move_relative(self, p: np.ndarray, resuming: bool) -> None:
        if resuming:
            self.cursor = tuple(self.backend.cursor_position())
            self._relative_pos = np.array(self.cursor, dtype=np.float64)
            self._relative_bounds = (self._display_bounds if self._display_bounds is not None
                                     else tuple(self.backend.screen_bounds()))
        else:
            _, _, w, h = self._box
            sw, sh = self.screen
            scale = np.array([(sw - 1) / w, (sh - 1) / h]) * self.cfg.cursor_gain
            left, top, width, height = self._relative_bounds
            self._relative_pos = np.clip(self._relative_pos + (p - self._relative_hand) * scale,
                                         (left, top), (left + width - 1, top + height - 1))
            target = tuple(int(round(v)) for v in self._relative_pos)
            if target != self.cursor:
                self.backend.move(*target)
                self.cursor = target
        # Always advance the hand reference, even at a screen edge; discard
        # clipped travel while retaining fractional pixels inside the screen.
        self._relative_hand = p.copy()

    def reset_dwell(self) -> None:
        """Start a fresh dwell hold without interrupting cursor motion or dragging."""
        self._dwell_anchor = None
        self._dwell_t0 = 0.0
        self._dwell_armed = True

    def _dwell(self, t: float, events: list[str]) -> None:
        """Hold the cursor still for ``dwell_ms`` -> one left click (accessibility)."""
        cur = self.cursor
        if self._dwell_anchor is None or math.dist(cur, self._dwell_anchor) > self.cfg.dwell_radius_px:
            self._dwell_anchor, self._dwell_t0, self._dwell_armed = cur, t, True
            return
        if self._dwell_armed and (t - self._dwell_t0) * 1000 >= self.cfg.dwell_ms:
            self.backend.press("left")
            self.button = "left"          # halt can retry if the release fails
            self.release_all()
            events.append("click:dwell")
            self._dwell_armed = False       # re-arms once the cursor moves away

    def _scroll(self, lm: np.ndarray, events: list[str]) -> None:
        y = float((lm[INDEX_TIP, 1] + lm[MIDDLE_TIP, 1]) / 2)
        if self._scroll_y is None:
            self._scroll_y = self._scroll_ref = y
            return
        a = self.cfg.scroll_smoothing
        self._scroll_y = a * y + (1 - a) * self._scroll_y
        delta = self._scroll_ref - self._scroll_y          # > 0: hand moved up
        if abs(delta) > self.cfg.scroll_deadzone:
            notches = delta * self.cfg.scroll_gain
            self.backend.scroll(notches)
            events.append(f"scroll:{notches:+.2f}")
            self._scroll_ref = self._scroll_y

    def zoom(self, steps: float, events: list[str]) -> None:
        self.backend.zoom(steps)
        events.append(f"zoom:{'in' if steps > 0 else 'out'}")
