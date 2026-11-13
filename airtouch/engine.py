"""GestureEngine - the per-frame pipeline shared by the live app, the tray
app, the headless simulator and the research scripts.

    landmarks -> recogniser (rule or ML) -> debouncer -> MouseController

Feed it one frame at a time with ``process(hand, t)`` (one hand) or
``process_hands(list_of_hands, t)`` (0, 1 or 2 hands; 2 hands = zoom).
Every call returns a ``FrameResult`` describing exactly what happened.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field

import numpy as np

from .config import Config
from .controller import FakeBackend, MouseController
from .debounce import Debouncer
from .gestures_rule import GESTURES, RuleRecognizer
from .landmarks import is_valid, palm_center, to_metric

KNOWN_LABELS = set(GESTURES) | {"none"}


@dataclass
class FrameResult:
    t: float
    hands: int = 0
    raw_label: str = "none"      # what the recogniser saw this frame
    label: str = "none"          # gesture actually in effect after movement safety
    confidence: float = 0.0
    cursor: tuple[int, int] | None = None
    button: str | None = None
    events: list[str] = field(default_factory=list)
    enabled: bool = True

    def as_dict(self) -> dict:
        return asdict(self)


class GestureEngine:
    # A held button survives tracker flicker for this long. Relative cursor
    # strokes still pause immediately so reacquisition cannot jump the mouse.
    LOST_GRACE_S = 0.15
    UNKNOWN_GRACE_S = 0.12
    PINCH_CONFIRM_FRAMES = 2

    def __init__(self, cfg: Config | None = None, backend=None, recognizer=None):
        self.cfg = cfg or Config()
        self.backend = backend if backend is not None else FakeBackend()
        self.recognizer = recognizer or RuleRecognizer(self.cfg.aspect, self.cfg.pinch_on)
        self.debouncer = Debouncer(self.cfg.debounce_frames)
        self.controller = MouseController(self.cfg, self.backend)
        self.enabled = True
        self._zoom_ref: float | None = None
        self._lost_since: float | None = None
        self._last_motion_t: float | None = None
        self._pinch_candidate: str | None = None
        self._pinch_frames = 0

    # -- control -------------------------------------------------------------
    def set_recognizer(self, recognizer) -> None:
        self.recognizer = recognizer
        self.debouncer.reset()
        self._reset_relative_confirmation()

    def set_enabled(self, on: bool) -> None:
        """Pause/resume everything (keyboard 'p', tray switch)."""
        if not on:
            self.reset()
        self.enabled = on

    def reset(self) -> None:
        self.controller.halt()
        self.debouncer.reset()
        self._zoom_ref = None
        self._lost_since = None
        self._reset_relative_confirmation()

    # -- per frame -------------------------------------------------------------
    def process(self, lm, t: float) -> FrameResult:
        # This public entry point represents one/no hand, so it also ends a
        # preceding two-hand span when callers mix the two processing APIs.
        self._zoom_ref = None
        if not self.enabled:
            return FrameResult(t=t, enabled=False)
        if not is_valid(lm):
            return self._no_hand(t)
        events = []
        if self._lost_since is not None and t - self._lost_since >= self.LOST_GRACE_S:
            # Tracking may skip directly from a lost frame to a valid frame.
            # Enforce the elapsed release deadline before forgetting that gap;
            # the returned hand must pass the normal fresh confirmation again.
            events = self._no_hand(t).events
        self._lost_since = None
        lm = np.asarray(lm, dtype=np.float64)
        raw, conf = self.recognizer.predict(lm)
        safe_raw = raw if raw in KNOWN_LABELS else "none"
        label = self.debouncer.update(safe_raw)
        if self.cfg.cursor_mode == "relative":
            label, allow_press = self._relative_gesture(safe_raw, label, t)
            ctrl = self.controller.update(label, lm, t, allow_press=allow_press,
                                          allow_dwell=safe_raw == "point",
                                          preserve_anchor=safe_raw == "none" and label == "point")
        else:
            ctrl = self.controller.update(label, lm, t)
        return FrameResult(t=t, hands=1, raw_label=raw, label=label, confidence=conf,
                           cursor=ctrl.cursor, button=ctrl.button, events=events + ctrl.events)

    def process_hands(self, hands, t: float) -> FrameResult:
        valid = [np.asarray(h, dtype=np.float64) for h in (hands or []) if is_valid(h)]
        if len(valid) < 2:
            self._zoom_ref = None
            return self.process(valid[0] if valid else None, t)
        if not self.enabled:
            return FrameResult(t=t, hands=len(valid), enabled=False)
        return self._zoom(valid[0], valid[1], t)

    # -- internals ---------------------------------------------------------------
    def _reset_relative_confirmation(self) -> None:
        self._last_motion_t = None
        self._pinch_candidate = None
        self._pinch_frames = 0

    def _relative_gesture(self, raw: str, stable: str, t: float) -> tuple[str, bool]:
        if raw in ("pinch_index", "pinch_middle"):
            self._last_motion_t = t
            if raw == self._pinch_candidate:
                self._pinch_frames += 1
            else:
                self._pinch_candidate, self._pinch_frames = raw, 1
            # The first candidate already uses the pinch's stable palm anchor,
            # but only the next matching raw frame may start a button press.
            return raw, self._pinch_frames >= self.PINCH_CONFIRM_FRAMES

        self._pinch_candidate, self._pinch_frames = None, 0
        if raw in ("point", "open_palm"):
            # Both poses move by the same index tip. Opening the other fingers
            # must not clutch or re-anchor an otherwise continuous stroke.
            self._last_motion_t = t
            return "point", False
        if raw == "none":
            recent_motion = (self._last_motion_t is not None
                             and 0 <= t - self._last_motion_t < self.UNKNOWN_GRACE_S)
            if self.controller.button is None and recent_motion:
                # A brief ambiguous pose keeps the same stroke/filter history.
                # It never starts a click and never extends its own deadline.
                return "point", False
            if not recent_motion:
                self._reset_relative_confirmation()
            # Held pinches remain governed by live geometry. If they release
            # on this unknown frame, the controller also stops immediately.
            return "none", False

        self._reset_relative_confirmation()
        # A fist immediately clutches even before label debounce. Other modes,
        # including scrolling, retain their existing label debounce.
        return (raw if raw == "fist" else stable), False

    def _no_hand(self, t: float) -> FrameResult:
        if self._lost_since is None:
            self._lost_since = t
        c = self.controller
        if self.cfg.cursor_mode == "relative":
            self._reset_relative_confirmation()
            c.suspend_motion()
        if t - self._lost_since < self.LOST_GRACE_S:
            return FrameResult(t=t, label=self.debouncer.current, cursor=c.cursor, button=c.button)
        self.debouncer.reset("none")
        ctrl = c.update("none", None, t)
        return FrameResult(t=t, cursor=ctrl.cursor, button=ctrl.button, events=ctrl.events)

    def _zoom(self, a: np.ndarray, b: np.ndarray, t: float) -> FrameResult:
        self._lost_since = None
        events: list[str] = []
        c = self.controller
        c.halt(events)
        self.debouncer.reset("none")
        self._reset_relative_confirmation()
        asp = self.cfg.aspect
        spread = float(np.linalg.norm(palm_center(to_metric(a, asp)) - palm_center(to_metric(b, asp))))
        if self._zoom_ref is None or self._zoom_ref <= 1e-6:
            self._zoom_ref = spread
        else:
            ratio = spread / self._zoom_ref
            if ratio > 0 and abs(ratio - 1.0) > self.cfg.zoom_deadzone:
                c.zoom(math.log(ratio) * self.cfg.zoom_gain, events)
                self._zoom_ref = spread
        return FrameResult(t=t, hands=2, raw_label="zoom", label="zoom", confidence=1.0,
                           cursor=c.cursor, button=c.button, events=events)
