"""Heads-up display drawn on top of the camera preview (OpenCV, BGR images).

Used by the live app and by the headless simulator, which draws the same
HUD on a blank canvas and saves PNGs with ``save_png`` (``cv2.imwrite``
fails on non-ASCII Windows paths, so PNGs are encoded in memory and written
with ``Path.write_bytes``).
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .landmarks import CONNECTIONS, anchor_point

# BGR colours per gesture
COLORS = {
    "point": (255, 200, 0),
    "pinch_index": (0, 220, 0),
    "pinch_middle": (0, 140, 255),
    "scroll": (255, 0, 200),
    "fist": (60, 60, 255),
    "open_palm": (200, 200, 200),
    "zoom": (0, 255, 255),
    "none": (140, 140, 140),
}

ACTIONS = {
    "point": "MOVE",
    "pinch_index": "LEFT BUTTON",
    "pinch_middle": "RIGHT BUTTON",
    "scroll": "SCROLL",
    "fist": "PAUSED",
    "open_palm": "IDLE",
    "zoom": "ZOOM",
    "none": "-",
}

LEGEND = (
    "point=move  thumb+index=left/drag  thumb+middle=right",
    "V=scroll  fist=pause  open palm=idle  2 hands=zoom",
)


def blank_canvas(width: int = 640, height: int = 480) -> np.ndarray:
    img = np.zeros((height, width, 3), dtype=np.uint8)
    img[:] = (40, 34, 30)
    return img


def draw_landmarks(img: np.ndarray, lm: np.ndarray, color=(0, 255, 0)) -> None:
    h, w = img.shape[:2]
    pts = [(int(x * w), int(y * h)) for x, y, _ in np.asarray(lm)]
    for a, b in CONNECTIONS:
        cv2.line(img, pts[a], pts[b], color, 2, cv2.LINE_AA)
    for i, p in enumerate(pts):
        cv2.circle(img, p, 5 if i in (4, 8, 12) else 3, (255, 255, 255), -1, cv2.LINE_AA)
    ax, ay = anchor_point(np.asarray(lm))
    cv2.circle(img, (int(ax * w), int(ay * h)), 8, color, 2, cv2.LINE_AA)


def _text(img, s, org, scale=0.6, color=(255, 255, 255), thick=1):
    cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thick + 3, cv2.LINE_AA)
    cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


def draw_hud(
    img: np.ndarray,
    result=None,
    hands=(),
    fps: float = 0.0,
    engine_name: str = "rule",
    active_box: tuple[float, float, float, float] | None = None,
    enabled: bool = True,
    dwell: bool = False,
) -> np.ndarray:
    """Draw everything in place and return ``img``.

    result:     engine FrameResult (or None)
    hands:      list of (21, 3) landmark arrays to draw
    active_box: (x0, y0, w, h) normalised region mapped to the screen
    """
    h, w = img.shape[:2]
    label = getattr(result, "label", "none") if result is not None else "none"
    color = COLORS.get(label, COLORS["none"])

    if active_box is not None:
        x0, y0, bw, bh = active_box
        cv2.rectangle(img, (int(x0 * w), int(y0 * h)), (int((x0 + bw) * w), int((y0 + bh) * h)),
                      (90, 90, 90), 1, cv2.LINE_AA)
    for lm in hands:
        draw_landmarks(img, lm, color)

    # top bar
    cv2.rectangle(img, (0, 0), (w, 34), (25, 25, 25), -1)
    state = ACTIONS.get(label, label.upper()) if enabled else "OFF (press P)"
    _text(img, f"{label}  ->  {state}", (10, 23), 0.65, color if enabled else (60, 60, 255), 2)
    _text(img, f"{fps:4.1f} FPS  {engine_name}{'  dwell' if dwell else ''}", (w - 220, 23), 0.5)

    if result is not None:
        btn = getattr(result, "button", None)
        if btn:
            _text(img, f"[{btn.upper()} DOWN]", (10, 60), 0.7, (0, 255, 0), 2)
        ev = getattr(result, "events", None)
        if ev:
            _text(img, " ".join(ev)[:60], (10, 88), 0.55, (0, 255, 255))
        conf = getattr(result, "confidence", 0.0)
        raw = getattr(result, "raw_label", "")
        if raw and raw != label:
            _text(img, f"seen: {raw} ({conf:.2f})", (10, h - 54), 0.5, (200, 200, 200))

    # legend
    for i, line in enumerate(LEGEND):
        _text(img, line, (10, h - 30 + i * 18), 0.42, (220, 220, 220))
    return img


def save_png(path: str | Path, img: np.ndarray) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode(".png", img)
    if not ok:
        raise RuntimeError("PNG encoding failed")
    path.write_bytes(buf.tobytes())
    return path
