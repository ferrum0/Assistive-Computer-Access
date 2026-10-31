"""Fast (<30s) gaze calibration via polynomial regression.

The user fixates a sequence of on-screen points (a 3x3 grid by default).
For each point we collect raw gaze vectors g = (g_x, g_y) from
`FaceTracker` and the corresponding known screen coordinate (X, Y).

We fit two independent polynomial regressions (degree 2 by default):

    X = a0 + a1*gx + a2*gy + a3*gx^2 + a4*gy^2 + a5*gx*gy
    Y = b0 + b1*gx + b2*gy + b3*gx^2 + b4*gy^2 + b5*gx*gy

i.e. we build the monomial design matrix Phi(g) and solve the linear
least-squares problem  min_a || Phi @ a - X ||^2  (and similarly for Y) via
numpy's lstsq. A quadratic surface is enough to absorb the mild nonlinearity
of iris-offset-to-gaze-angle without needing many calibration points, which
is what keeps the whole procedure under 30 seconds (9 points x ~2.5s + a
short settle time between points).
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


def _design_matrix(g: np.ndarray, degree: int) -> np.ndarray:
    """Build monomial features [1, gx, gy, gx^2, gy^2, gx*gy, ...] up to `degree`.

    g: (N, 2) array of raw gaze vectors.
    """
    gx, gy = g[:, 0], g[:, 1]
    cols = [np.ones_like(gx)]
    if degree >= 1:
        cols += [gx, gy]
    if degree >= 2:
        cols += [gx ** 2, gy ** 2, gx * gy]
    if degree >= 3:
        cols += [gx ** 3, gy ** 3, (gx ** 2) * gy, gx * (gy ** 2)]
    return np.stack(cols, axis=1)


@dataclass
class CalibrationPoint:
    screen_xy: tuple[float, float]
    raw_gaze_samples: list[tuple[float, float]] = field(default_factory=list)


class GazeCalibrator:
    def __init__(self, screen_w: int, screen_h: int, grid: tuple[int, int] = (3, 3),
                 degree: int = 2, margin_frac: float = 0.12):
        self.screen_w = screen_w
        self.screen_h = screen_h
        self.degree = degree
        self.points: list[CalibrationPoint] = self._build_grid(grid, margin_frac)
        self._coef_x: np.ndarray | None = None
        self._coef_y: np.ndarray | None = None
        self._rmse: tuple[float, float] | None = None

    def _build_grid(self, grid: tuple[int, int], margin_frac: float) -> list[CalibrationPoint]:
        cols, rows = grid
        mx, my = self.screen_w * margin_frac, self.screen_h * margin_frac
        xs = np.linspace(mx, self.screen_w - mx, cols)
        ys = np.linspace(my, self.screen_h - my, rows)
        pts = []
        for y in ys:
            for x in xs:
                pts.append(CalibrationPoint(screen_xy=(float(x), float(y))))
        return pts

    def add_sample(self, point_index: int, raw_gaze: tuple[float, float]) -> None:
        self.points[point_index].raw_gaze_samples.append(raw_gaze)

    def fit(self) -> dict:
        """Fit the polynomial regression from collected samples.

        Per-point samples are averaged first (robust to per-frame noise/
        outliers within a fixation) before the regression is solved, so each
        calibration point contributes one clean (g, X, Y) observation
        weighted implicitly by how many raw samples were captured for it.
        """
        g_list, x_list, y_list = [], [], []
        for p in self.points:
            if not p.raw_gaze_samples:
                continue
            arr = np.array(p.raw_gaze_samples)
            # Trim to the middle 70% of samples to drop settle-in transients.
            n = len(arr)
            lo, hi = int(n * 0.15), max(int(n * 0.85), int(n * 0.15) + 1)
            trimmed = arr[lo:hi] if hi > lo else arr
            g_mean = trimmed.mean(axis=0)
            g_list.append(g_mean)
            x_list.append(p.screen_xy[0])
            y_list.append(p.screen_xy[1])

        if len(g_list) < 4:
            raise RuntimeError("Not enough calibration points with data to fit "
                                "a regression (need >= 4).")

        g = np.array(g_list)
        X = np.array(x_list)
        Y = np.array(y_list)
        Phi = _design_matrix(g, self.degree)

        coef_x, res_x, _, _ = np.linalg.lstsq(Phi, X, rcond=None)
        coef_y, res_y, _, _ = np.linalg.lstsq(Phi, Y, rcond=None)
        self._coef_x, self._coef_y = coef_x, coef_y

        pred_x = Phi @ coef_x
        pred_y = Phi @ coef_y
        rmse_x = float(np.sqrt(np.mean((pred_x - X) ** 2)))
        rmse_y = float(np.sqrt(np.mean((pred_y - Y) ** 2)))
        self._rmse = (rmse_x, rmse_y)
        return {"rmse_x_px": rmse_x, "rmse_y_px": rmse_y, "n_points": len(g_list)}

    def predict(self, raw_gaze: tuple[float, float]) -> tuple[float, float]:
        if self._coef_x is None or self._coef_y is None:
            raise RuntimeError("Calibrator not fitted yet.")
        g = np.array([raw_gaze])
        Phi = _design_matrix(g, self.degree)
        x = float(Phi @ self._coef_x)
        y = float(Phi @ self._coef_y)
        x = float(np.clip(x, 0, self.screen_w))
        y = float(np.clip(y, 0, self.screen_h))
        return x, y

    @property
    def is_fitted(self) -> bool:
        return self._coef_x is not None

    def save(self, path: Path) -> None:
        payload = {
            "screen_w": self.screen_w,
            "screen_h": self.screen_h,
            "degree": self.degree,
            "coef_x": self._coef_x.tolist() if self._coef_x is not None else None,
            "coef_y": self._coef_y.tolist() if self._coef_y is not None else None,
            "rmse": self._rmse,
            "saved_at": time.time(),
        }
        path.write_text(json.dumps(payload, indent=2))

    @classmethod
    def load(cls, path: Path) -> "GazeCalibrator":
        payload = json.loads(path.read_text())
        obj = cls(payload["screen_w"], payload["screen_h"], degree=payload["degree"])
        obj._coef_x = np.array(payload["coef_x"])
        obj._coef_y = np.array(payload["coef_y"])
        obj._rmse = tuple(payload["rmse"]) if payload.get("rmse") else None
        return obj

    def estimated_duration_s(self, seconds_per_point: float, settle_s: float) -> float:
        return len(self.points) * (seconds_per_point + settle_s)
