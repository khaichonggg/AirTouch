"""Rule-based gesture recogniser.

Every test is a *ratio* of distances measured in square (metric) units, so
it does not care where the hand is, how big it looks or how it is rotated.

Gestures
--------
point         index finger up                   -> move the cursor
pinch_index   thumb touches index tip            -> left button (hold = drag)
pinch_middle  thumb touches middle tip           -> right button
scroll        index + middle up ("V")            -> scroll by moving the hand
fist          all fingers curled                 -> pause (cursor freezes)
open_palm     all fingers up                     -> idle / hands-off
none          anything else
"""

from __future__ import annotations

import numpy as np

from .landmarks import (
    FINGERS,
    INDEX_TIP,
    MIDDLE_TIP,
    THUMB_TIP,
    WRIST,
    hand_size,
    to_metric,
)

GESTURES = ("point", "pinch_index", "pinch_middle", "scroll", "fist", "open_palm")
NONE = "none"

# A finger is "extended" when its tip is clearly further from the wrist than its
# middle joint (PIP). Straight ~1.3, half-bent ~1.0, curled ~0.6.
EXTENDED_RATIO = 1.12
# A finger may only START a pinch if its tip is NOT tucked into the palm:
# tip-to-wrist / knuckle-to-wrist. Fist ~0.75 (max ~0.88), real pinch ~0.95-1.25.
# The middle finger needs more: in the pointing pose the thumb rests on the
# curled middle finger, close to its tip, so a looser test would fire false
# right-clicks. Once a button is held the lower HOLD value applies
# (hysteresis), so small wobbles during a drag never drop the button.
PRESS_REACH = {"left": 0.86, "right": 0.92}
HOLD_REACH = {"left": 0.76, "right": 0.82}


def _d(m: np.ndarray, a: int, b: int) -> float:
    return float(np.linalg.norm(m[a, :2] - m[b, :2]))


def finger_extension(m: np.ndarray) -> dict[str, float]:
    """tip-to-wrist / PIP-to-wrist for each long finger (metric landmarks)."""
    out = {}
    for name, (_mcp, pip, _dip, tip) in FINGERS.items():
        out[name] = _d(m, tip, WRIST) / max(_d(m, pip, WRIST), 1e-6)
    return out


def finger_reach(m: np.ndarray, name: str) -> float:
    """tip-to-wrist / knuckle-to-wrist: < ~0.88 means the tip is tucked into the palm."""
    mcp, _pip, _dip, tip = FINGERS[name]
    return _d(m, tip, WRIST) / max(_d(m, mcp, WRIST), 1e-6)


def pinch_gaps(m: np.ndarray) -> tuple[float, float]:
    """(thumb-index gap, thumb-middle gap) in hand-size units."""
    size = hand_size(m)
    return _d(m, THUMB_TIP, INDEX_TIP) / size, _d(m, THUMB_TIP, MIDDLE_TIP) / size


def which_pinch(m: np.ndarray, threshold: float) -> str | None:
    """'left' / 'right' / None: which pinch is closed enough to START a press."""
    d_i, d_m = pinch_gaps(m)
    ok_i = finger_reach(m, "index") > PRESS_REACH["left"]
    ok_m = finger_reach(m, "middle") > PRESS_REACH["right"]
    if ok_i and d_i < threshold and (d_i <= d_m or not ok_m):
        return "left"
    if ok_m and d_m < threshold and (d_m < d_i or not ok_i):
        return "right"
    return None


def pinch_gap(m: np.ndarray, button: str) -> float:
    d_i, d_m = pinch_gaps(m)
    return d_i if button == "left" else d_m


def pinch_held(m: np.ndarray, button: str, release_threshold: float) -> bool:
    """True while an already-pressed ``button`` should stay down (looser test)."""
    finger = "index" if button == "left" else "middle"
    return (pinch_gap(m, button) < release_threshold
            and finger_reach(m, finger) > HOLD_REACH[button])


def classify_metric(m: np.ndarray, pinch_on: float = 0.20) -> str:
    pinch = which_pinch(m, pinch_on)
    if pinch == "left":
        return "pinch_index"
    if pinch == "right":
        return "pinch_middle"
    ext = {k: v > EXTENDED_RATIO for k, v in finger_extension(m).items()}
    up = (ext["index"], ext["middle"], ext["ring"], ext["pinky"])
    if up == (True, False, False, False):
        return "point"
    if up == (True, True, False, False):
        return "scroll"
    if all(up):
        return "open_palm"
    if not any(up):
        return "fist"
    return NONE


def classify(lm: np.ndarray, aspect: float = 4 / 3, pinch_on: float = 0.20) -> str:
    return classify_metric(to_metric(lm, aspect), pinch_on)


class RuleRecognizer:
    """Same interface as the ML ``GestureClassifier``: predict(lm) -> (label, confidence)."""

    name = "rule"

    def __init__(self, aspect: float = 4 / 3, pinch_on: float = 0.20):
        self.aspect = aspect
        self.pinch_on = pinch_on

    def predict(self, lm: np.ndarray) -> tuple[str, float]:
        return classify(lm, self.aspect, self.pinch_on), 1.0
