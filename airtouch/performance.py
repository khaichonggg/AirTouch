"""Bounded, aggregate diagnostics. Never saves frames or hand coordinates."""

from __future__ import annotations

import json
import logging
import math
import uuid
from collections import Counter, deque
from dataclasses import asdict
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
from pathlib import Path

LOG = logging.getLogger("airtouch.performance")


def configure_performance_log(folder: str | Path, max_bytes: int = 1_000_000,
                              backups: int = 2):
    """Configured only by the real tray entry point; tests need no disk sink."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    target = folder / "performance.jsonl"
    for handler in LOG.handlers:
        if isinstance(handler, RotatingFileHandler) and Path(handler.baseFilename) == target.resolve():
            return handler
    handler = RotatingFileHandler(target, maxBytes=max_bytes, backupCount=backups,
                                  encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(message)s"))
    LOG.addHandler(handler)
    LOG.setLevel(logging.INFO)
    LOG.propagate = False
    return handler


def _finite(value):
    try:
        number = float(value)
        return round(number, 3) if math.isfinite(number) else None
    except (TypeError, ValueError, OverflowError):
        return None


def _summary(values):
    if not values:
        return None
    ordered = sorted(values)
    return {"mean": round(sum(ordered) / len(ordered), 3),
            "p95": round(ordered[max(0, math.ceil(.95 * len(ordered)) - 1)], 3),
            "max": round(ordered[-1], 3)}


class PerformanceMonitor:
    def __init__(self, cfg, camera, engine_name: str, interval: float = 5.0, logger=None):
        self.cfg, self.camera, self.engine_name = cfg, camera, engine_name
        self.interval = interval
        self.logger = logger if logger is not None else LOG
        self.session_id = uuid.uuid4().hex[:12]
        self.started = self.window_start = 0.0
        self.total_frames = 0
        self.last_seq = None
        self.last_raw = None
        self.had_hand = False
        self.last_cursor = None
        self._quality_last_at = None
        self._reset_window()

    def _reset_window(self):
        self.frames = self.hand_frames = self.hand_losses = self.skipped = self.raw_switches = 0
        self.idle_polls = 0
        self.tracker_observed_frames = self.tracker_hand_frames = self.invalid_hand_frames = 0
        self.mouse_actions = Counter()
        self.cursor_changed_frames = self.unknown_continuation_frames = 0
        self.cursor_path = 0.0
        self.geometry = {}
        self.quality_samples = 0
        self.quality = {}
        self.raw, self.stable = Counter(), Counter()
        self.tracking, self.control, self.age = (deque(maxlen=2048) for _ in range(3))
        self.confidence = deque(maxlen=2048)

    def _emit(self, event: str, **values):
        try:
            payload = {"event": event, "session": self.session_id,
                       "time": datetime.now(timezone.utc).isoformat(), **values}
            self.logger.info("%s", json.dumps(payload, ensure_ascii=False, allow_nan=False))
        except Exception:
            pass  # diagnostics must never interrupt mouse release or the capture loop

    def start(self, now: float):
        self.started = self.window_start = now
        self._emit("session_start", engine=self.engine_name, config=asdict(self.cfg),
                   frame_size=getattr(self.camera, "frame_size", None))

    def frame(self, now: float, seq: int, hands: int, raw_label: str, label: str,
              confidence: float, tracking_ms: float, control_ms: float,
              frame_age_ms: float | None, *, cursor=None, events=(), landmarks=None,
              tracker_hands: int | None = None, frame=None):
        self.frames += 1
        self.total_frames += 1
        self.hand_frames += bool(hands)
        # Optional raw counts distinguish no detection from rejected landmarks.
        # Count frames with at least one rejected hand, including partial rejects.
        if isinstance(tracker_hands, int) and tracker_hands >= 0:
            self.tracker_observed_frames += 1
            self.tracker_hand_frames += tracker_hands > 0
            self.invalid_hand_frames += tracker_hands > hands
        self.hand_losses += self.had_hand and not hands
        self.had_hand = bool(hands)
        if self.last_seq is not None:
            self.skipped += max(0, seq - self.last_seq - 1)
        self.last_seq = seq
        self.raw_switches += self.last_raw is not None and raw_label != self.last_raw
        self.last_raw = raw_label
        self.raw[raw_label] += 1
        self.stable[label] += 1
        self.unknown_continuation_frames += raw_label == "none" and label == "point"
        for event in events:
            parts = event.split(":", 1)
            key = event if parts[0] in ("down", "up", "click") else parts[0]
            self.mouse_actions[key] += 1
        if cursor is not None:
            if self.last_cursor is not None and cursor != self.last_cursor:
                self.cursor_changed_frames += 1
                self.cursor_path += math.dist(cursor, self.last_cursor)
            self.last_cursor = tuple(cursor)
        if hands and landmarks is not None and self.frames % 3 == 0:
            self._sample_geometry(landmarks)
        if frame is not None and (self._quality_last_at is None
                                  or now - self._quality_last_at >= .5):
            self._quality_last_at = now
            self._sample_frame_quality(frame)
        self.tracking.append(max(0.0, tracking_ms))
        self.control.append(max(0.0, control_ms))
        confidence = _finite(confidence)
        if hands and confidence is not None:
            self.confidence.append(confidence)
        age = _finite(frame_age_ms)
        if age is not None:
            self.age.append(max(0.0, age))
        if now - self.window_start >= self.interval:
            self._flush(now, "performance")

    def _sample_geometry(self, landmarks):
        # Ten small geometry samples/s at 30FPS; only aggregate ratios are kept.
        try:
            from . import gestures_rule as gr
            from .landmarks import hand_size, to_metric

            hand = to_metric(landmarks, self.cfg.aspect)
            left, right = gr.pinch_gaps(hand)
            extension = gr.finger_extension(hand)
            values = {"left_gap": left, "right_gap": right,
                      "index_extension": extension["index"],
                      "middle_extension": extension["middle"],
                      "ring_extension": extension["ring"],
                      "pinky_extension": extension["pinky"],
                      "index_reach": gr.finger_reach(hand, "index"),
                      "middle_reach": gr.finger_reach(hand, "middle"),
                      "hand_size": hand_size(hand)}
            for key, value in values.items():
                value = _finite(value)
                if value is not None:
                    self.geometry.setdefault(key, deque(maxlen=1024)).append(value)
        except (TypeError, ValueError, IndexError, OverflowError):
            pass

    def _sample_frame_quality(self, frame):
        """2Hz, at most 32x32 existing pixels; retain aggregate values only.

        Luminance and its population standard deviation use the 0..255 scale.
        Dark fraction is the share of sampled pixels with luminance below 32.
        This describes overall exposure/contrast, not hand visibility or blur.
        """
        try:
            import numpy as np

            if not isinstance(frame, np.ndarray) or frame.dtype != np.uint8:
                return
            if frame.ndim not in (2, 3) or not all(frame.shape[:2]):
                return
            if frame.ndim == 3 and frame.shape[2] not in (3, 4):
                return
            height, width = frame.shape[:2]
            pixels = frame[::max(1, (height + 31) // 32),
                           ::max(1, (width + 31) // 32)]
            if frame.ndim == 3:
                # OpenCV supplies BGR; ignore any optional alpha channel.
                luma = pixels[..., 0] * .114 + pixels[..., 1] * .587 + pixels[..., 2] * .299
            else:
                luma = pixels
            values = {"luma_mean": float(luma.mean()), "luma_std": float(luma.std()),
                      "dark_fraction": float((luma < 32).mean())}
            for key, value in values.items():
                self.quality.setdefault(key, deque(maxlen=1024)).append(value)
            self.quality_samples += 1
        except (TypeError, ValueError, IndexError, OverflowError):
            pass

    def idle(self, now: float):
        self.idle_polls += 1
        if now - self.window_start >= self.interval:
            self._flush(now, "performance")

    def _flush(self, now: float, event: str):
        elapsed = max(0.0, now - self.window_start)
        self._emit(event, window_s=round(elapsed, 3), frames=self.frames,
                   session_frames=self.total_frames, session_s=round(now - self.started, 3),
                   processing_fps=round(self.frames / elapsed, 3) if elapsed else 0.0,
                   capture_fps=_finite(getattr(self.camera, "measured_fps", None)),
                   tracking_ms=_summary(self.tracking), control_ms=_summary(self.control),
                   confidence=_summary(self.confidence),
                   frame_age_ms=_summary(self.age), skipped_frames=self.skipped,
                   hand_frames=self.hand_frames, hand_loss_events=self.hand_losses,
                   tracker_observed_frames=self.tracker_observed_frames,
                   tracker_hand_frames=self.tracker_hand_frames if self.tracker_observed_frames else None,
                   invalid_hand_frames=self.invalid_hand_frames if self.tracker_observed_frames else None,
                   raw_switches=self.raw_switches, raw_gestures=dict(self.raw),
                   stable_gestures=dict(self.stable), idle_polls=self.idle_polls,
                   mouse_actions=dict(self.mouse_actions), cursor_changed_frames=self.cursor_changed_frames,
                   cursor_path_px=round(self.cursor_path, 3),
                   unknown_continuation_frames=self.unknown_continuation_frames,
                   frame_quality={"samples": self.quality_samples,
                                  **{key: {**_summary(values), "min": round(min(values), 3)}
                                     for key, values in self.quality.items() if values}},
                   hand_geometry={key: {**_summary(values), "min": round(min(values), 3)}
                                  for key, values in self.geometry.items() if values})
        self._reset_window()
        self.window_start = now

    def finish(self, now: float):
        self._flush(now, "session_end")
