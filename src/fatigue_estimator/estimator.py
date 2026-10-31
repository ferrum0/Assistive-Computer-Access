"""Fatigue estimation f_hat(t) in [0, 1], used to drive the adaptive dwell
threshold tau(t) = tau_0 * (1 + lambda * f_hat(t)).

Three evidence-backed signals are combined, each computed over a rolling
window (default 60 s):

1. PERCLOS (PERcentage of eye CLOSure): the fraction of frames in the window
   where EAR falls below `perclos_eye_closed_threshold`. This is the most
   widely validated drowsiness/fatigue proxy in the driver-monitoring
   literature; we reuse it here directly since EAR is already computed by
   the face tracker for blink detection.

       f_perclos = clip(PERCLOS / 0.5, 0, 1)     # 50% closure -> fully saturated

2. Blink duration: fatigued users tend to produce longer, slower blinks than
   their own alert baseline. We track mean blink duration in the window and
   normalize against a saturation point:

       f_duration = clip((mean_dur - baseline_dur) /
                          (duration_saturation - baseline_dur), 0, 1)

3. Fixation jitter: the position variance of the gaze signal *during
   putative fixations* (i.e. excluding smooth-pursuit/saccade motion,
   approximated here by looking at low-velocity segments) increases as
   oculomotor control degrades with fatigue.

       f_jitter = clip((std_px - baseline_px) /
                        (jitter_saturation_px - baseline_px), 0, 1)

Final estimate is a weighted sum (weights sum to 1), then exponentially
smoothed to avoid jumpy threshold changes:

    f_raw(t)  = w1*f_perclos + w2*f_duration + w3*f_jitter
    f_hat(t)  = alpha * f_raw(t) + (1 - alpha) * f_hat(t - 1)
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.utils.buffers import EMA, RollingWindow


@dataclass
class FatigueSnapshot:
    f_hat: float
    perclos: float
    mean_blink_duration_s: float
    fixation_jitter_px: float
    n_blinks_in_window: int


class FatigueEstimator:
    def __init__(self, window_seconds: float, ema_alpha: float,
                 perclos_weight: float, blink_duration_weight: float,
                 jitter_weight: float, perclos_eye_closed_threshold: float,
                 baseline_blink_duration_s: float, baseline_jitter_px: float,
                 blink_duration_saturation_s: float, jitter_saturation_px: float):
        self.w_perclos = perclos_weight
        self.w_duration = blink_duration_weight
        self.w_jitter = jitter_weight
        self.ear_closed_threshold = perclos_eye_closed_threshold
        self.baseline_blink_duration_s = baseline_blink_duration_s
        self.baseline_jitter_px = baseline_jitter_px
        self.duration_saturation_s = blink_duration_saturation_s
        self.jitter_saturation_px = jitter_saturation_px

        self._ear_window = RollingWindow(window_seconds)       # (t, ear)
        self._gaze_window = RollingWindow(window_seconds)      # (t, (x, y))
        self._ema = EMA(ema_alpha, initial=0.0)

        self._blink_active = False
        self._blink_start_t: float | None = None
        self._recent_blink_durations = RollingWindow(window_seconds)  # (t, duration)

    def push_ear(self, t: float, ear: float | None, is_blink: bool) -> None:
        if ear is not None:
            self._ear_window.push(t, ear)

        if is_blink and not self._blink_active:
            self._blink_active = True
            self._blink_start_t = t
        elif not is_blink and self._blink_active:
            duration = t - self._blink_start_t
            self._recent_blink_durations.push(t, duration)
            self._blink_active = False
            self._blink_start_t = None

    def push_gaze_px(self, t: float, xy: tuple[float, float] | None) -> None:
        if xy is not None:
            self._gaze_window.push(t, xy)

    def _perclos(self) -> float:
        ears = self._ear_window.values()
        if not ears:
            return 0.0
        closed = sum(1 for e in ears if e < self.ear_closed_threshold)
        return closed / len(ears)

    def _mean_blink_duration(self) -> tuple[float, int]:
        durations = self._recent_blink_durations.values()
        if not durations:
            return self.baseline_blink_duration_s, 0
        return float(np.mean(durations)), len(durations)

    def _fixation_jitter_px(self) -> float:
        """Std-dev of gaze position restricted to low-velocity samples, as a
        cheap proxy for "fixation-only" variance without a full fixation
        classifier.
        """
        items = self._gaze_window.items()
        if len(items) < 5:
            return self.baseline_jitter_px
        pts = np.array([v for _, v in items])
        vel = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        vel_threshold = np.percentile(vel, 40) if len(vel) else 0.0
        low_vel_idx = np.where(vel <= vel_threshold)[0]
        if len(low_vel_idx) < 3:
            return self.baseline_jitter_px
        still_pts = pts[low_vel_idx]
        std_xy = np.std(still_pts, axis=0)
        return float(np.mean(std_xy))

    def update(self, t: float) -> FatigueSnapshot:
        perclos = self._perclos()
        mean_dur, n_blinks = self._mean_blink_duration()
        jitter_px = self._fixation_jitter_px()

        f_perclos = np.clip(perclos / 0.5, 0.0, 1.0)
        f_duration = np.clip(
            (mean_dur - self.baseline_blink_duration_s) /
            max(self.duration_saturation_s - self.baseline_blink_duration_s, 1e-6),
            0.0, 1.0)
        f_jitter = np.clip(
            (jitter_px - self.baseline_jitter_px) /
            max(self.jitter_saturation_px - self.baseline_jitter_px, 1e-6),
            0.0, 1.0)

        f_raw = (self.w_perclos * f_perclos +
                 self.w_duration * f_duration +
                 self.w_jitter * f_jitter)
        f_hat = self._ema.update(float(np.clip(f_raw, 0.0, 1.0)))

        return FatigueSnapshot(
            f_hat=f_hat,
            perclos=perclos,
            mean_blink_duration_s=mean_dur,
            fixation_jitter_px=jitter_px,
            n_blinks_in_window=n_blinks,
        )
