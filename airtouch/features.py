"""Turn raw landmarks into a pose description that ignores *where* the hand
is, *how big* it looks and *how it is rotated* - so a gesture looks the same
to the classifier anywhere in the camera image.

``normalize_landmarks`` exposes each invariance as a flag so the research
ablation study can switch them off one at a time.
"""

from __future__ import annotations

import numpy as np

from .landmarks import (
    FINGERS,
    INDEX_MCP,
    MIDDLE_MCP,
    PINKY_MCP,
    THUMB_CMC,
    THUMB_IP,
    THUMB_MCP,
    THUMB_TIP,
    TIPS,
    WRIST,
    to_metric,
)

DEFAULT_ASPECT = 4 / 3


def normalize_landmarks(
    lm: np.ndarray,
    aspect: float = DEFAULT_ASPECT,
    translate: bool = True,
    rotate: bool = True,
    scale: bool = True,
) -> np.ndarray:
    """Return a normalised (21, 3) copy of ``lm``.

    translate: wrist moved to the origin.
    rotate:    wrist->middle-knuckle vector turned to point straight up, and
               the hand mirrored (if needed) so the index knuckle is always on
               the left - left and right hands then look identical.
    scale:     divided by hand size (wrist->middle-knuckle length).
    """
    m = to_metric(lm, aspect)
    if translate:
        m = m - m[WRIST]
    if rotate:
        v = m[MIDDLE_MCP, :2] - m[WRIST, :2]
        # angle that rotates v onto (0, -1) = "up" in image coordinates
        ang = -np.arctan2(v[0], -v[1])
        c, s = np.cos(ang), np.sin(ang)
        x, y = m[:, 0].copy(), m[:, 1].copy()
        m[:, 0] = c * x - s * y
        m[:, 1] = s * x + c * y
        if m[INDEX_MCP, 0] > m[PINKY_MCP, 0]:
            m[:, 0] = -m[:, 0]
    if scale:
        size = float(np.linalg.norm(m[MIDDLE_MCP, :2] - m[WRIST, :2]))
        m = m / max(size, 1e-6)
    return m


def _straightness(m: np.ndarray, a: int, b: int, c: int) -> float:
    """Cosine of the bend at joint ``b`` (1 = perfectly straight)."""
    u = m[b, :2] - m[a, :2]
    v = m[c, :2] - m[b, :2]
    nu, nv = np.linalg.norm(u), np.linalg.norm(v)
    if nu < 1e-9 or nv < 1e-9:
        return 1.0
    return float(np.dot(u, v) / (nu * nv))


def extra_features(n: np.ndarray) -> np.ndarray:
    """Hand-crafted distances/angles computed on normalised landmarks."""
    tips_to_wrist = [np.linalg.norm(n[t, :2] - n[WRIST, :2]) for t in TIPS]
    thumb_to_tips = [np.linalg.norm(n[THUMB_TIP, :2] - n[t, :2]) for t in TIPS[1:]]
    neighbours = [np.linalg.norm(n[TIPS[i], :2] - n[TIPS[i + 1], :2]) for i in range(1, 4)]
    straight = [_straightness(n, THUMB_CMC, THUMB_MCP, THUMB_IP)]
    straight += [_straightness(n, mcp, pip, tip) for mcp, pip, _dip, tip in FINGERS.values()]
    return np.asarray(tips_to_wrist + thumb_to_tips + neighbours + straight, dtype=np.float64)


def feature_vector(
    lm: np.ndarray,
    aspect: float = DEFAULT_ASPECT,
    translate: bool = True,
    rotate: bool = True,
    scale: bool = True,
) -> np.ndarray:
    """63 normalised coordinates + 17 extra features = 80 numbers."""
    n = normalize_landmarks(lm, aspect, translate=translate, rotate=rotate, scale=scale)
    return np.concatenate([n.reshape(-1), extra_features(n)])


def feature_matrix(hands, aspect: float = DEFAULT_ASPECT, **flags) -> np.ndarray:
    return np.stack([feature_vector(h, aspect, **flags) for h in hands])


NUM_FEATURES = 63 + 17
