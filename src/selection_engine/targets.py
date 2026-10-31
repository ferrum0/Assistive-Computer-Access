"""Orbiting target trajectories used by the smooth-pursuit selection engine.

Each clickable widget is assigned a unique closed orbit around its resting
position:

    T_i(t) = c_i + r_i * [cos(omega_i * t + phi_i), sin(omega_i * t + phi_i)]

Distinct targets get distinct phase offsets phi_i (and optionally slightly
different angular speeds omega_i / radii r_i). This is what makes the
trajectories mutually near-uncorrelated so that correlating the user's gaze
path against all of them simultaneously yields one clear winner rather than
several plausible ones (the same principle as Vidal et al.'s "Pursuits"
gaze-interaction technique).
"""
from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class OrbitTarget:
    id: str
    label: str
    center: tuple[float, float]
    radius_px: float
    period_s: float
    phase_rad: float
    action: object = None  # callback invoked on selection

    def position(self, t: float) -> tuple[float, float]:
        omega = 2 * math.pi / self.period_s
        cx, cy = self.center
        x = cx + self.radius_px * math.cos(omega * t + self.phase_rad)
        y = cy + self.radius_px * math.sin(omega * t + self.phase_rad)
        return x, y

    def velocity(self, t: float) -> tuple[float, float]:
        """Analytic derivative dT/dt; used for velocity-domain correlation."""
        omega = 2 * math.pi / self.period_s
        vx = -self.radius_px * omega * math.sin(omega * t + self.phase_rad)
        vy = self.radius_px * omega * math.cos(omega * t + self.phase_rad)
        return vx, vy


def make_orbit_targets(specs: list[dict], base_radius_px: float,
                        base_period_s: float) -> list[OrbitTarget]:
    """Build targets with maximally-separated phases (and slight period/
    radius jitter per target) so trajectories stay decorrelated even for
    many simultaneous targets.

    specs: [{"id": ..., "label": ..., "center": (x, y), "action": callable}, ...]
    """
    n = len(specs)
    targets = []
    for i, spec in enumerate(specs):
        phase = 2 * math.pi * i / max(n, 1)
        # +/-15% period jitter and +/-10% radius jitter, deterministic per index,
        # further reduces cross-correlation between nearby-phase targets.
        period = base_period_s * (1.0 + 0.15 * math.sin(1.7 * i))
        radius = base_radius_px * (1.0 + 0.10 * math.cos(2.3 * i))
        targets.append(OrbitTarget(
            id=spec["id"],
            label=spec.get("label", spec["id"]),
            center=spec["center"],
            radius_px=radius,
            period_s=period,
            phase_rad=phase,
            action=spec.get("action"),
        ))
    return targets
