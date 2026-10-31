"""Small rolling-window helpers shared across sensing/estimation modules."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field


@dataclass
class TimedSample:
    t: float
    value: object


class RollingWindow:
    """Fixed-duration (not fixed-length) buffer of (timestamp, value) samples.

    Old samples fall off automatically once they exceed ``duration_s`` relative
    to the most recently pushed timestamp. This makes every consumer robust to
    variable camera frame rate instead of assuming a fixed sample count.
    """

    def __init__(self, duration_s: float, max_len: int = 4096):
        self.duration_s = duration_s
        self._buf: deque[TimedSample] = deque(maxlen=max_len)

    def push(self, t: float, value: object) -> None:
        self._buf.append(TimedSample(t, value))
        self._evict(t)

    def _evict(self, now: float) -> None:
        while self._buf and (now - self._buf[0].t) > self.duration_s:
            self._buf.popleft()

    def values(self) -> list:
        return [s.value for s in self._buf]

    def timestamps(self) -> list[float]:
        return [s.t for s in self._buf]

    def items(self) -> list[tuple[float, object]]:
        return [(s.t, s.value) for s in self._buf]

    def __len__(self) -> int:
        return len(self._buf)

    def clear(self) -> None:
        self._buf.clear()

    def latest(self):
        return self._buf[-1].value if self._buf else None


class EMA:
    """Exponential moving average, used to smooth confidence/fatigue signals."""

    def __init__(self, alpha: float, initial: float | None = None):
        self.alpha = alpha
        self._value = initial

    @property
    def value(self) -> float | None:
        return self._value

    def update(self, x: float) -> float:
        if self._value is None:
            self._value = x
        else:
            self._value = self.alpha * x + (1 - self.alpha) * self._value
        return self._value
