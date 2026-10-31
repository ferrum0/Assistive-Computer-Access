"""Orchestrates the three-way comparison study: fixed dwell / adaptive dwell
/ smooth pursuit, run over the same set of Fitts' law target-acquisition
tasks (see `fitts_tasks.py`), logging one `TrialResult` per trial.

Each trial presents two candidates: the correct target and one distractor
(placed at the trial's start position), so that "error rate" is meaningful
for all three modes -- a wrong dwell-click or a pursuit lock onto the
distractor both count as an error.

"fixed dwell" and "adaptive dwell" reuse the exact same
`AdaptiveDwellSelector` implementation; fixed dwell simply sets
`fatigue_sensitivity = 0` so tau(t) = tau_0 stays constant regardless of
f_hat(t), which is what makes the comparison a controlled test of the
adaptive-threshold idea itself rather than of two different algorithms.

This module only implements the trial *state machine*; it does not touch
the camera or the selection engine directly. The UI layer (`ui/app.py`)
drives it once per frame: it asks the runner for the current pursuit
targets / dwell widgets to render and feed into `SelectionEngine.update`,
then reports the resulting `SelectionEvent` (if any) back via `on_frame`.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

from src.evaluation.fitts_tasks import FittsTrialSpec
from src.evaluation.metrics import TrialLogger, TrialResult
from src.selection_engine.engine import SelectionEvent
from src.selection_engine.targets import OrbitTarget, make_orbit_targets

MODES = ("fixed_dwell", "adaptive_dwell", "smooth_pursuit")

TARGET_ID = "target"
DISTRACTOR_ID = "distractor"


@dataclass
class TrialOutcome:
    spec: FittsTrialSpec
    correct: bool
    selected_widget_id: str
    movement_time_s: float


class StudyRunner:
    def __init__(self, participant_id: str, mode: str, specs: list[FittsTrialSpec],
                 logger: TrialLogger, trial_timeout_s: float = 8.0,
                 orbit_period_s: float = 2.2):
        assert mode in MODES, f"unknown mode {mode}"
        self.participant_id = participant_id
        self.mode = mode
        self.specs = specs
        self.logger = logger
        self.trial_timeout_s = trial_timeout_s
        self.orbit_period_s = orbit_period_s

        self._trial_i = 0
        self._trial_start_t: float | None = None
        self._finished = False
        self.outcomes: list[TrialOutcome] = []

    @property
    def is_finished(self) -> bool:
        return self._finished

    @property
    def trial_number(self) -> int:
        """1-indexed position of the current (or just-finished) trial."""
        return min(self._trial_i + 1, len(self.specs))

    @property
    def current_spec(self) -> FittsTrialSpec | None:
        if self._finished or self._trial_i >= len(self.specs):
            return None
        return self.specs[self._trial_i]

    def begin_trial(self, t: float) -> None:
        self._trial_start_t = t

    def pursuit_targets(self) -> list[OrbitTarget]:
        spec = self.current_spec
        if spec is None or self.mode != "smooth_pursuit":
            return []
        return make_orbit_targets(
            [
                {"id": TARGET_ID, "label": "target", "center": spec.target_xy},
                {"id": DISTRACTOR_ID, "label": "distractor", "center": spec.start_xy},
            ],
            base_radius_px=max(spec.width_px * 0.6, 15.0),
            base_period_s=self.orbit_period_s,
        )

    def dwell_widgets(self) -> dict[str, tuple[float, float]]:
        spec = self.current_spec
        if spec is None or self.mode not in ("fixed_dwell", "adaptive_dwell"):
            return {}
        return {TARGET_ID: spec.target_xy, DISTRACTOR_ID: spec.start_xy}

    def on_selection_event(self, t: float, event: SelectionEvent | None) -> TrialOutcome | None:
        """Call once per frame with the SelectionEngine's output (may be None).

        Returns a TrialOutcome and advances to the next trial when the
        current trial concludes (either a selection was made or it timed
        out).
        """
        spec = self.current_spec
        if spec is None:
            return None
        if self._trial_start_t is None:
            self.begin_trial(t)

        timed_out = (t - self._trial_start_t) >= self.trial_timeout_s
        made_selection = event is not None and event.widget_id in (TARGET_ID, DISTRACTOR_ID)

        if not made_selection and not timed_out:
            return None

        mt = t - self._trial_start_t
        if made_selection:
            selected = event.widget_id
            correct = selected == TARGET_ID
        else:
            selected = "timeout"
            correct = False

        outcome = TrialOutcome(spec=spec, correct=correct,
                                selected_widget_id=selected, movement_time_s=mt)
        self.outcomes.append(outcome)
        self.logger.log(TrialResult(
            participant_id=self.participant_id, mode=self.mode,
            trial_index=spec.trial_index, amplitude_px=spec.amplitude_px,
            width_px=spec.width_px, index_of_difficulty=spec.index_of_difficulty,
            movement_time_s=mt, correct=correct, selected_widget_id=selected,
            target_widget_id=TARGET_ID, timestamp=time.time(),
        ))

        self._trial_i += 1
        self._trial_start_t = None
        if self._trial_i >= len(self.specs):
            self._finished = True
        return outcome


def dwell_config_for_mode(mode: str, base_cfg) -> "AdaptiveDwellConfig":  # noqa: F821
    """Return a copy of the dwell config with fatigue_sensitivity zeroed out
    for the 'fixed_dwell' baseline condition, unchanged otherwise.
    """
    from dataclasses import replace
    if mode == "fixed_dwell":
        return replace(base_cfg, fatigue_sensitivity=0.0)
    return base_cfg


def default_trial_log_path(logs_dir: Path, participant_id: str) -> Path:
    from src.evaluation.metrics import default_log_path
    return default_log_path(logs_dir, participant_id)
