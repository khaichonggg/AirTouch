"""Cursor smoothing filters.

The One-Euro filter (Casiez, Roussel & Vogel, CHI 2012) adapts to speed:
when the hand is still it smooths heavily (no jitter), when the hand moves
fast it barely smooths (no lag). EMA and moving-average are the classic
fixed baselines used in the research comparison.
"""

from __future__ import annotations

import math
from collections import deque

import numpy as np


def _alpha(cutoff: float, dt: float) -> float:
    tau = 1.0 / (2.0 * math.pi * cutoff)
    return 1.0 / (1.0 + tau / dt)


class OneEuroFilter:
    """Works on scalars or numpy vectors; for vectors the speed is the vector norm,
    so x and y always get the same amount of smoothing."""

    def __init__(self, min_cutoff: float = 1.0, beta: float = 0.0, d_cutoff: float = 1.0):
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        self.reset()

    def reset(self) -> None:
        self._x = None
        self._dx = None
        self._t = None

    def __call__(self, x, t: float):
        x = np.asarray(x, dtype=np.float64)
        if self._x is None or self._t is None:
            self._x, self._dx, self._t = x, np.zeros_like(x), t
            return x.copy()
        dt = t - self._t
        if dt <= 0:
            # duplicate / out-of-order timestamp: keep the previous estimate
            return self._x.copy()
        dx = (x - self._x) / dt
        a_d = _alpha(self.d_cutoff, dt)
        dx_hat = a_d * dx + (1.0 - a_d) * self._dx
        speed = float(np.linalg.norm(dx_hat))
        cutoff = self.min_cutoff + self.beta * speed
        a = _alpha(cutoff, dt)
        x_hat = a * x + (1.0 - a) * self._x
        self._x, self._dx, self._t = x_hat, dx_hat, t
        return x_hat.copy()


class EMAFilter:
    """Exponential moving average: y = a*x + (1-a)*y_prev."""

    def __init__(self, alpha: float = 0.5):
        self.alpha = float(alpha)
        self.reset()

    def reset(self) -> None:
        self._y = None

    def __call__(self, x, t: float = 0.0):
        x = np.asarray(x, dtype=np.float64)
        self._y = x if self._y is None else self.alpha * x + (1 - self.alpha) * self._y
        return self._y.copy()


class MovingAverageFilter:
    """Mean of the last ``n`` samples."""

    def __init__(self, n: int = 5):
        self.n = max(1, int(n))
        self.reset()

    def reset(self) -> None:
        self._buf: deque = deque(maxlen=self.n)

    def __call__(self, x, t: float = 0.0):
        self._buf.append(np.asarray(x, dtype=np.float64))
        return np.mean(self._buf, axis=0)
