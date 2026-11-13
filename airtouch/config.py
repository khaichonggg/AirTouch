"""All tunable settings in one place.

Values can be overridden with a ``config.json`` next to ``hand_mouse.py``
(see ``config.example.json``). ``__post_init__`` clamps every value into a
safe range so a bad edit can never break the app.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, fields
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
HAND_MODEL_PATH = MODELS_DIR / "hand_landmarker.task"
GESTURE_MODEL_PATH = MODELS_DIR / "gesture_model.joblib"
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config.json"

# (min, max) for every numeric field; used by __post_init__.
_RANGES = {
    "camera_index": (0, 10),
    "frame_width": (160, 1920),
    "frame_height": (120, 1080),
    "camera_fps": (5, 60),
    "max_hands": (1, 2),
    "min_detection_confidence": (0.1, 0.95),
    "min_tracking_confidence": (0.1, 0.95),
    "frame_margin": (0.0, 0.4),
    "smoothing": (0.05, 10.0),
    "beta": (0.0, 10.0),
    "d_cutoff": (0.1, 10.0),
    "cursor_gain": (0.1, 5.0),
    "pinch_on": (0.05, 0.6),
    "pinch_off": (0.06, 0.8),
    "debounce_frames": (1, 10),
    "scroll_gain": (1.0, 200.0),
    "scroll_deadzone": (0.001, 0.1),
    "scroll_smoothing": (0.05, 1.0),
    "zoom_gain": (0.5, 50.0),
    "zoom_deadzone": (0.01, 0.5),
    "dwell_ms": (200, 5000),
    "dwell_radius_px": (2, 200),
    "ml_confidence": (0.0, 1.0),
}


@dataclass
class Config:
    # --- camera -----------------------------------------------------------
    camera_index: int = 0
    frame_width: int = 640
    frame_height: int = 480
    camera_fps: int = 30
    mirror: bool = True              # selfie view: move hand right -> cursor right

    # --- hand tracker -----------------------------------------------------
    max_hands: int = 1               # 2 enables two-hand zoom (costs a bit more CPU)
    min_detection_confidence: float = 0.6
    min_tracking_confidence: float = 0.5

    # --- cursor -----------------------------------------------------------
    frame_margin: float = 0.15       # ignore this border of the camera image
    smoothing: float = 1.2           # One-Euro min_cutoff: lower = steadier, higher = snappier
    beta: float = 1.0                # One-Euro speed coefficient: higher = less lag on fast moves
    d_cutoff: float = 1.0
    cursor_mode: str = "absolute"    # relative acts like a touchpad; absolute supports research
    cursor_gain: float = 1.5

    # --- clicking (thumb-to-finger distance, in hand-size units) -----------
    pinch_on: float = 0.20           # fingers closer than this = button down
    pinch_off: float = 0.32          # fingers further than this = button up (hysteresis)

    # --- gesture stability -------------------------------------------------
    debounce_frames: int = 3         # a new gesture must hold this many frames

    # --- scrolling ----------------------------------------------------------
    scroll_gain: float = 40.0        # wheel notches per full-frame-height of hand travel
    scroll_deadzone: float = 0.010   # ignore smaller (smoothed) hand movements
    scroll_smoothing: float = 0.4    # EMA factor on the scroll signal (kills tremor)

    # --- two-hand zoom ------------------------------------------------------
    zoom_gain: float = 6.0
    zoom_deadzone: float = 0.08      # relative change in hand spread before zooming

    # --- dwell-to-click (accessibility) --------------------------------------
    dwell_enabled: bool = False
    dwell_ms: int = 900
    dwell_radius_px: int = 18

    # --- recogniser -----------------------------------------------------------
    engine: str = "rule"             # "rule" or "ml"
    ml_confidence: float = 0.55      # ML predictions below this become "none"

    # --- display ---------------------------------------------------------------
    show_hud: bool = True
    cursor_halo: bool = False

    def __post_init__(self) -> None:
        for f in fields(self):
            value = getattr(self, f.name)
            if f.name in _RANGES:
                lo, hi = _RANGES[f.name]
                default = f.default
                try:
                    num = float(value)
                except (TypeError, ValueError, OverflowError):
                    num = float(default)
                if not math.isfinite(num):
                    num = float(default)
                num = min(max(num, lo), hi)
                setattr(self, f.name, int(round(num)) if isinstance(default, int)
                        and not isinstance(default, bool) else num)
            elif isinstance(f.default, bool):
                if isinstance(value, str):
                    text = value.strip().lower()
                    if text in ("false", "0", "off", "no", ""):
                        value = False
                    elif text in ("true", "1", "on", "yes"):
                        value = True
                    else:
                        value = f.default
                elif value is not None and (not isinstance(value, (bool, int, float))
                                            or value not in (0, 1)):
                    # Only canonical numeric flags (0/1) are supported. JSON
                    # containers, other numbers and non-finite values must not
                    # silently enable dwell clicks or native overlays.
                    value = f.default
                setattr(self, f.name, bool(value))
        if self.engine not in ("rule", "ml"):
            self.engine = "rule"
        if self.cursor_mode not in ("absolute", "relative"):
            self.cursor_mode = "absolute"
        # hysteresis must open wider than it closes
        if self.pinch_off <= self.pinch_on:
            self.pinch_off = self.pinch_on + 0.08

    @property
    def aspect(self) -> float:
        """Camera width / height. Landmark x is scaled by this to get square units."""
        return self.frame_width / self.frame_height

    @classmethod
    def load(cls, path: str | Path | None = None) -> "Config":
        path = Path(path) if path else DEFAULT_CONFIG_PATH
        try:
            data = json.loads(path.read_text(encoding="utf-8-sig"))
        except (OSError, ValueError, RecursionError):
            return cls()
        if not isinstance(data, dict):
            return cls()
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in known})

    def save(self, path: str | Path | None = None) -> None:
        path = Path(path) if path else DEFAULT_CONFIG_PATH
        path.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")
