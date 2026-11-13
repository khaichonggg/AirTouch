"""Gesture debouncing: a new gesture only "counts" after it has been seen for
``frames`` consecutive frames. This stops one bad frame from flipping the
mode (research study 3 shows N=3 removes almost all false triggers)."""

from __future__ import annotations


class Debouncer:
    def __init__(self, frames: int = 3, initial: str = "none"):
        self.frames = max(1, int(frames))
        self.reset(initial)

    def reset(self, label: str = "none") -> None:
        self.current = label
        self._candidate = None
        self._count = 0

    def update(self, label: str) -> str:
        if label == self.current:
            self._candidate, self._count = None, 0
            return self.current
        if label == self._candidate:
            self._count += 1
        else:
            self._candidate, self._count = label, 1
        if self._count >= self.frames:
            self.current = label
            self._candidate, self._count = None, 0
        return self.current
