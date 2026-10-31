"""Adaptive dwell-time click, used as a fallback where smooth pursuit is
impractical (dense text, small targets that cannot host a visible orbit
without overlapping neighbours).

Fixation detection: the gaze point is considered "fixating" a widget while
it stays within `fixation_radius_px` of the widget's center. We track how
long the *current* fixation has lasted and compare it against a threshold
tau(t) that grows with estimated user fatigue f_hat(t) in [0, 1]:

    tau(t) = tau_0 * (1 + lambda * f_hat(t))

Rationale: a tired user blinks more, holds fixations less steadily, and is
more likely to trigger accidental dwell-clicks while merely resting their
eyes on a widget. Raising tau proportionally to f_hat trades a bit of speed
for a lower error rate exactly when the user is least able to self-correct a
mis-click. tau_0 is the baseline (alert-user) threshold and lambda controls
how aggressively fatigue lengthens it; both are configurable per user.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class DwellState:
    target_id: str | None
    fixation_start: float | None
    progress: float               # 0..1, current dwell fraction of tau(t)
    threshold_s: float
    triggered: bool


class AdaptiveDwellSelector:
    def __init__(self, base_threshold_s: float, fatigue_sensitivity: float,
                 max_threshold_s: float, fixation_radius_px: float):
        self.tau0 = base_threshold_s
        self.lam = fatigue_sensitivity
        self.tau_max = max_threshold_s
        self.fixation_radius_px = fixation_radius_px

        self._current_target: str | None = None
        self._fixation_start: float | None = None
        self._last_confirmed_threshold: float = base_threshold_s

    def threshold(self, f_hat: float) -> float:
        """tau(t) = tau_0 * (1 + lambda * f_hat), clamped to tau_max."""
        f_hat = float(np.clip(f_hat, 0.0, 1.0))
        tau = self.tau0 * (1.0 + self.lam * f_hat)
        return float(min(tau, self.tau_max))

    def update(self, t: float, gaze_xy: tuple[float, float] | None,
               widgets: dict[str, tuple[float, float]], f_hat: float) -> DwellState:
        tau = self.threshold(f_hat)
        self._last_confirmed_threshold = tau

        if gaze_xy is None:
            self._current_target = None
            self._fixation_start = None
            return DwellState(None, None, 0.0, tau, False)

        hit_id = None
        best_d = None
        for wid, center in widgets.items():
            d = float(np.hypot(gaze_xy[0] - center[0], gaze_xy[1] - center[1]))
            if d <= self.fixation_radius_px and (best_d is None or d < best_d):
                best_d = d
                hit_id = wid

        if hit_id is None:
            self._current_target = None
            self._fixation_start = None
            return DwellState(None, None, 0.0, tau, False)

        if hit_id != self._current_target:
            self._current_target = hit_id
            self._fixation_start = t

        elapsed = t - self._fixation_start
        progress = float(np.clip(elapsed / tau, 0.0, 1.0))
        triggered = elapsed >= tau

        state = DwellState(hit_id, self._fixation_start, progress, tau, triggered)
        if triggered:
            # Reset so the same widget needs a fresh fixation to fire again.
            self._current_target = None
            self._fixation_start = None
        return state
