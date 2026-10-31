"""Smooth-pursuit target selection.

This is the primary selection mechanism and the direct answer to the Midas
touch problem: instead of asking "is the user fixating this widget for X
milliseconds?" (which cannot distinguish *looking* from *wanting to
activate*), we ask "does the user's eye movement *pattern* match this
widget's unique motion pattern?". Passive looking at a moving target does not
require any decision from the interface until the correlation is
sufficiently exclusive to one candidate.

Mathematics
-----------
Let G(t) = (Gx(t), Gy(t)) be the calibrated gaze position sampled over a
sliding window of duration W (default 1.5 s), and T_i(t) = (Tix(t), Tiy(t))
the trajectory of orbiting target i over the same window.

We work in the *velocity domain* (first differences) rather than raw
position, for two reasons: (1) it removes the constant offset between the
gaze point and the orbit center (a user's gaze estimate is rarely pixel-
perfect after calibration, but a well-tracking eye still reproduces the
*shape* of the motion), and (2) it directly captures the smooth-pursuit
oculomotor response, which is a velocity-matching reflex.

For a window of N synchronized samples:

    dGx[k] = Gx[k] - Gx[k-1],  dGy[k] = Gy[k] - Gy[k-1]
    dTix[k] = Tix[k] - Tix[k-1],  dTiy[k] = Tiy[k] - Tiy[k-1]

Pearson correlation on each axis:

    r_x(i) = corr(dGx, dTix),   r_y(i) = corr(dGy, dTiy)
    score_i = (r_x(i) + r_y(i)) / 2                      in [-1, 1]

Oculomotor smooth pursuit lags the visual stimulus by roughly 100-200 ms, so
we do not correlate at a fixed zero lag: we shift the target trajectory by a
small set of candidate lags tau in [0, max_lag] and keep the maximum
correlation:

    score_i = max_{tau in [0, max_lag]} ( r_x(i, tau) + r_y(i, tau) ) / 2

The winning target is the arg max over i, and it is *confirmed* (i.e. an
actual click/activation event fires) only if:
  1. score_winner >= correlation_threshold, AND
  2. score_winner - score_runner_up >= margin_over_runner_up, AND
  3. condition (1)+(2) holds continuously for >= confirm_seconds.

Requiring a sustained, exclusive winner (not just a single-frame peak) is
what suppresses accidental activation from noisy tracking, which is the
smooth-pursuit analogue of debouncing.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.selection_engine.targets import OrbitTarget
from src.utils.buffers import RollingWindow


@dataclass
class PursuitScore:
    target_id: str
    score: float
    best_lag_s: float


@dataclass
class PursuitResult:
    winner: PursuitScore | None
    all_scores: list[PursuitScore]
    confirmed: bool


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 3 or np.std(a) < 1e-9 or np.std(b) < 1e-9:
        return 0.0
    r = np.corrcoef(a, b)[0, 1]
    return float(r) if np.isfinite(r) else 0.0


class SmoothPursuitSelector:
    def __init__(self, window_seconds: float, min_samples: int,
                 correlation_threshold: float, margin_over_runner_up: float,
                 confirm_seconds: float, max_lag_seconds: float,
                 lag_step_seconds: float):
        self.window_seconds = window_seconds
        self.min_samples = min_samples
        self.correlation_threshold = correlation_threshold
        self.margin_over_runner_up = margin_over_runner_up
        self.confirm_seconds = confirm_seconds
        self.max_lag_seconds = max_lag_seconds
        self.lag_step_seconds = lag_step_seconds

        self._gaze_window = RollingWindow(window_seconds)
        self._winner_streak_id: str | None = None
        self._winner_streak_start: float | None = None

    def push_gaze(self, t: float, xy: tuple[float, float]) -> None:
        self._gaze_window.push(t, xy)

    def reset_streak(self) -> None:
        self._winner_streak_id = None
        self._winner_streak_start = None

    def evaluate(self, targets: list[OrbitTarget], now: float) -> PursuitResult:
        items = self._gaze_window.items()
        if len(items) < self.min_samples:
            self.reset_streak()
            return PursuitResult(winner=None, all_scores=[], confirmed=False)

        ts = np.array([t for t, _ in items])
        gx = np.array([v[0] for _, v in items])
        gy = np.array([v[1] for _, v in items])
        dgx = np.diff(gx)
        dgy = np.diff(gy)

        lags = np.arange(0.0, self.max_lag_seconds + 1e-9, self.lag_step_seconds)
        scores: list[PursuitScore] = []
        for target in targets:
            best = -1.0
            best_lag = 0.0
            for lag in lags:
                tx = np.array([target.position(t - lag)[0] for t in ts])
                ty = np.array([target.position(t - lag)[1] for t in ts])
                dtx = np.diff(tx)
                dty = np.diff(ty)
                r_x = _pearson(dgx, dtx)
                r_y = _pearson(dgy, dty)
                s = (r_x + r_y) / 2.0
                if s > best:
                    best = s
                    best_lag = lag
            scores.append(PursuitScore(target_id=target.id, score=best, best_lag_s=best_lag))

        scores.sort(key=lambda s: s.score, reverse=True)
        winner = scores[0] if scores else None
        runner_up_score = scores[1].score if len(scores) > 1 else -1.0

        confirmed = False
        if winner is not None:
            passes = (winner.score >= self.correlation_threshold and
                      (winner.score - runner_up_score) >= self.margin_over_runner_up)
            if passes:
                if self._winner_streak_id != winner.target_id:
                    self._winner_streak_id = winner.target_id
                    self._winner_streak_start = now
                elif (now - self._winner_streak_start) >= self.confirm_seconds:
                    confirmed = True
            else:
                self.reset_streak()

        return PursuitResult(winner=winner, all_scores=scores, confirmed=confirmed)
