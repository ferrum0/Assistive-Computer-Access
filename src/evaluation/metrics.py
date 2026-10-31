"""Trial logging and cross-mode comparison analysis for the evaluation study.

Per trial we log: selection mode, Fitts (A, W, ID), movement/selection time,
whether the correct target was acquired (error rate), and which widget was
actually selected (for confusion analysis).

Cross-mode comparison fits Fitts' law per mode:

    MT = a + b * ID          (ordinary least squares, numpy.polyfit degree 1)
    TP = ID / MT              throughput per trial, then averaged (bits/s)

and reports R^2 of the linear fit as a goodness-of-fit check for the Fitts
model itself, plus the overall error rate per mode.
"""
from __future__ import annotations

import csv
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np


@dataclass
class TrialResult:
    participant_id: str
    mode: str                     # "fixed_dwell" | "adaptive_dwell" | "smooth_pursuit"
    trial_index: int
    amplitude_px: float
    width_px: float
    index_of_difficulty: float
    movement_time_s: float
    correct: bool
    selected_widget_id: str
    target_widget_id: str
    timestamp: float


class TrialLogger:
    _FIELDS = list(TrialResult.__dataclass_fields__.keys())

    def __init__(self, csv_path: Path):
        self.csv_path = csv_path
        is_new = not csv_path.exists()
        self._fh = open(csv_path, "a", newline="")
        self._writer = csv.DictWriter(self._fh, fieldnames=self._FIELDS)
        if is_new:
            self._writer.writeheader()

    def log(self, result: TrialResult) -> None:
        self._writer.writerow(asdict(result))
        self._fh.flush()

    def close(self) -> None:
        self._fh.close()


def load_trials(csv_path: Path) -> list[TrialResult]:
    rows = []
    with open(csv_path, newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            row["trial_index"] = int(row["trial_index"])
            row["amplitude_px"] = float(row["amplitude_px"])
            row["width_px"] = float(row["width_px"])
            row["index_of_difficulty"] = float(row["index_of_difficulty"])
            row["movement_time_s"] = float(row["movement_time_s"])
            row["correct"] = row["correct"] in ("True", "true", "1")
            row["timestamp"] = float(row["timestamp"])
            rows.append(TrialResult(**row))
    return rows


@dataclass
class ModeSummary:
    mode: str
    n_trials: int
    error_rate: float
    fitts_a: float                # intercept (s)
    fitts_b: float                 # slope (s/bit)
    fitts_r2: float
    mean_throughput_bps: float     # mean of ID/MT per correct trial


def summarize_mode(trials: list[TrialResult]) -> ModeSummary:
    mode = trials[0].mode
    n = len(trials)
    errors = sum(1 for t in trials if not t.correct)
    correct_trials = [t for t in trials if t.correct]

    ids = np.array([t.index_of_difficulty for t in correct_trials])
    mts = np.array([t.movement_time_s for t in correct_trials])

    if len(ids) >= 2 and np.std(ids) > 1e-9:
        b, a = np.polyfit(ids, mts, 1)
        pred = a + b * ids
        ss_res = np.sum((mts - pred) ** 2)
        ss_tot = np.sum((mts - np.mean(mts)) ** 2)
        r2 = float(1 - ss_res / ss_tot) if ss_tot > 1e-9 else 0.0
    else:
        a, b, r2 = float("nan"), float("nan"), float("nan")

    # Guard against near-zero movement times (e.g. a selection confirmed on
    # the same frame a trial began) blowing throughput up to inf.
    safe_mts = np.maximum(mts, 1e-3)
    throughput = ids / safe_mts if len(ids) else np.array([])
    mean_tp = float(np.mean(throughput)) if len(throughput) else float("nan")

    return ModeSummary(
        mode=mode, n_trials=n, error_rate=errors / n if n else float("nan"),
        fitts_a=float(a), fitts_b=float(b), fitts_r2=r2,
        mean_throughput_bps=mean_tp,
    )


def compare_modes(all_trials: list[TrialResult]) -> dict[str, ModeSummary]:
    by_mode: dict[str, list[TrialResult]] = {}
    for t in all_trials:
        by_mode.setdefault(t.mode, []).append(t)
    return {mode: summarize_mode(ts) for mode, ts in by_mode.items()}


def print_comparison_table(summaries: dict[str, ModeSummary]) -> str:
    lines = [
        f"{'mode':<16}{'n':>5}{'err%':>8}{'a (s)':>9}{'b (s/bit)':>11}"
        f"{'R2':>7}{'TP (bit/s)':>12}"
    ]
    for mode, s in summaries.items():
        lines.append(
            f"{mode:<16}{s.n_trials:>5}{s.error_rate * 100:>7.1f}%"
            f"{s.fitts_a:>9.3f}{s.fitts_b:>11.3f}{s.fitts_r2:>7.2f}"
            f"{s.mean_throughput_bps:>12.2f}"
        )
    table = "\n".join(lines)
    print(table)
    return table


def default_log_path(logs_dir: Path, participant_id: str) -> Path:
    ts = time.strftime("%Y%m%d_%H%M%S")
    return logs_dir / f"trials_{participant_id}_{ts}.csv"
