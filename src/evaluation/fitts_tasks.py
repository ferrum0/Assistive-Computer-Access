"""ISO 9241-9 style target-acquisition task generator for Fitts' law analysis.

Fitts' law (Shannon formulation) relates movement time MT to an index of
difficulty ID derived from the amplitude A (distance to the target) and its
width W (tolerance/size):

    ID = log2(A / W + 1)                      [bits]
    MT = a + b * ID                            (linear regression fit per mode)
    TP = ID / MT                               [bits/s], throughput

We generate a standard multi-directional "circle of targets" layout (as in
ISO 9241-9): N targets evenly spaced on a circle of diameter A around a
screen center; a trial requires acquiring the target diametrically opposite
the previous one, so consecutive movements each cover the full amplitude A.
Combining several (A, W) pairs yields a spread of ID values, which is what
the regression needs to be meaningful (a single ID gives a line through one
point).

`generate_tasks` below picks a fixed set of (A, W) combinations, spanning a
target ID range, and caps the total trial count to the requested 10-15 range
by default.
"""
from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class FittsTrialSpec:
    trial_index: int
    amplitude_px: float
    width_px: float
    index_of_difficulty: float
    start_xy: tuple[float, float]
    target_xy: tuple[float, float]


def index_of_difficulty(amplitude_px: float, width_px: float) -> float:
    return math.log2(amplitude_px / width_px + 1.0)


def generate_tasks(screen_w: int, screen_h: int,
                    amplitudes_px: tuple[float, ...],
                    widths_px: tuple[float, ...],
                    n_directions: int = 8,
                    max_trials: int = 13) -> list[FittsTrialSpec]:
    """Build a trial list spanning a range of Fitts' ID values.

    We select (A, W) pairs by sorting all combinations by ID and taking an
    evenly-spaced subsample across the sorted list, which guarantees good
    coverage of easy/medium/hard conditions rather than clustering near one
    difficulty level.
    """
    combos = []
    for a in amplitudes_px:
        for w in widths_px:
            combos.append((a, w, index_of_difficulty(a, w)))
    combos.sort(key=lambda c: c[2])

    if len(combos) > max_trials:
        idxs = [round(i * (len(combos) - 1) / (max_trials - 1)) for i in range(max_trials)]
        idxs = sorted(set(idxs))
        combos = [combos[i] for i in idxs]

    cx, cy = screen_w / 2.0, screen_h / 2.0
    specs: list[FittsTrialSpec] = []
    for i, (a, w, ID) in enumerate(combos):
        angle_prev = 2 * math.pi * (i % n_directions) / n_directions
        angle_next = angle_prev + math.pi  # diametrically opposite, per ISO 9241-9
        start = (cx + (a / 2) * math.cos(angle_prev), cy + (a / 2) * math.sin(angle_prev))
        target = (cx + (a / 2) * math.cos(angle_next), cy + (a / 2) * math.sin(angle_next))
        specs.append(FittsTrialSpec(
            trial_index=i, amplitude_px=a, width_px=w,
            index_of_difficulty=ID, start_xy=start, target_xy=target,
        ))
    return specs
