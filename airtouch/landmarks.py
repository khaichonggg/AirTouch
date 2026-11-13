"""MediaPipe hand-landmark indices and small geometry helpers.

A hand is a ``(21, 3)`` float array of MediaPipe *normalised* coordinates:
x in [0, 1] across the image width, y in [0, 1] down the image height,
z = relative depth (smaller = closer to the camera).

Because x and y are normalised by different image dimensions, distances
and angles must be measured in *metric* space: ``to_metric`` multiplies x
by the frame aspect ratio so one unit is the same physical length on both
axes.
"""

from __future__ import annotations

import numpy as np

WRIST = 0
THUMB_CMC, THUMB_MCP, THUMB_IP, THUMB_TIP = 1, 2, 3, 4
INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP = 5, 6, 7, 8
MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP = 9, 10, 11, 12
RING_MCP, RING_PIP, RING_DIP, RING_TIP = 13, 14, 15, 16
PINKY_MCP, PINKY_PIP, PINKY_DIP, PINKY_TIP = 17, 18, 19, 20

NUM_LANDMARKS = 21

# (mcp, pip, dip, tip) for the four long fingers
FINGERS = {
    "index": (INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP),
    "middle": (MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP),
    "ring": (RING_MCP, RING_PIP, RING_DIP, RING_TIP),
    "pinky": (PINKY_MCP, PINKY_PIP, PINKY_DIP, PINKY_TIP),
}
TIPS = (THUMB_TIP, INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP)

# Skeleton edges, for drawing
CONNECTIONS = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
)


def is_valid(lm) -> bool:
    """True if ``lm`` is a finite (21, 3) array."""
    if lm is None:
        return False
    try:
        arr = np.asarray(lm)
        return arr.shape == (NUM_LANDMARKS, 3) and bool(np.all(np.isfinite(arr)))
    except (TypeError, ValueError, OverflowError):
        return False


def to_metric(lm: np.ndarray, aspect: float) -> np.ndarray:
    """Copy of ``lm`` with x scaled by the frame aspect so axes share one unit."""
    out = np.array(lm, dtype=np.float64, copy=True)
    out[:, 0] *= aspect
    return out


def hand_size(m: np.ndarray) -> float:
    """Wrist -> middle-finger knuckle distance (2-D, metric). Never zero."""
    return max(float(np.linalg.norm(m[MIDDLE_MCP, :2] - m[WRIST, :2])), 1e-6)


def dist(m: np.ndarray, a: int, b: int) -> float:
    """2-D distance between two landmarks (metric units)."""
    return float(np.linalg.norm(m[a, :2] - m[b, :2]))


def palm_center(m: np.ndarray) -> np.ndarray:
    """Mean of wrist + four knuckles; stable while fingers move (2-D)."""
    return m[[WRIST, INDEX_MCP, MIDDLE_MCP, RING_MCP, PINKY_MCP], :2].mean(axis=0)


def anchor_point(lm: np.ndarray) -> np.ndarray:
    """The point that drives the cursor, in *normalised* image coords.

    Uses the knuckle line (index + middle MCP) blended with the wrist rather
    than a fingertip: pinching moves the fingertips but barely moves the
    knuckles, so a click does not drag the cursor off its target.
    """
    pts = lm[[INDEX_MCP, MIDDLE_MCP, WRIST], :2]
    return pts[0] * 0.4 + pts[1] * 0.4 + pts[2] * 0.2
