"""Central configuration for the gaze/gesture assistive interface.

All tunable constants live here so behaviour can be adjusted without
hunting through the module tree.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT_DIR / "models"
DATA_DIR = ROOT_DIR / "data"
LOGS_DIR = DATA_DIR / "logs"
CALIBRATION_FILE = DATA_DIR / "calibration.json"

# MediaPipe Tasks model bundles (download separately, see README).
FACE_LANDMARKER_MODEL = MODELS_DIR / "face_landmarker.task"
HAND_LANDMARKER_MODEL = MODELS_DIR / "hand_landmarker.task"


@dataclass
class CameraConfig:
    device_index: int = 0
    width: int = 1280
    height: int = 720
    target_fps: int = 30


@dataclass
class ScreenConfig:
    width: int = 1920
    height: int = 1080


@dataclass
class CalibrationConfig:
    grid: tuple[int, int] = (3, 3)          # 9-point calibration grid
    seconds_per_point: float = 2.5          # < 30s total for a 3x3 grid
    settle_seconds: float = 0.8             # discard samples right after a jump
    poly_degree: int = 2                    # quadratic feature expansion


@dataclass
class SmoothPursuitConfig:
    window_seconds: float = 1.5             # correlation window length
    min_samples: int = 20                   # min samples in window to score
    correlation_threshold: float = 0.75     # min score to arm a selection
    margin_over_runner_up: float = 0.12     # winner must beat 2nd place by this
    confirm_seconds: float = 0.5            # sustained-winner time to confirm
    max_lag_seconds: float = 0.20           # search window for oculomotor latency
    lag_step_seconds: float = 0.02
    orbit_radius_px: float = 45.0
    orbit_period_seconds: float = 2.2       # base period; phase differs per target


@dataclass
class AdaptiveDwellConfig:
    base_threshold_seconds: float = 0.9     # tau_0
    fatigue_sensitivity: float = 1.2        # lambda
    max_threshold_seconds: float = 2.5
    fixation_radius_px: float = 35.0        # spatial tolerance for "still looking"


@dataclass
class FatigueConfig:
    window_seconds: float = 60.0
    ema_alpha: float = 0.15                 # smoothing of the final f_hat signal
    perclos_weight: float = 0.45
    blink_duration_weight: float = 0.25
    jitter_weight: float = 0.30
    perclos_eye_closed_threshold: float = 0.20   # EAR below this = "closed"
    baseline_blink_duration_s: float = 0.12
    baseline_jitter_px: float = 8.0
    blink_duration_saturation_s: float = 0.45
    jitter_saturation_px: float = 40.0


@dataclass
class GazeFilterConfig:
    # EMA smoothing applied to the calibrated gaze point before it drives the
    # cursor/selection logic. Webcam iris tracking is much noisier frame-to-
    # frame than dedicated eye-tracking hardware, so the raw calibrated point
    # visibly jitters on screen; this trades a small amount of responsiveness
    # (lower alpha = more smoothing = more lag) for a much steadier cursor.
    smoothing_alpha: float = 0.35


@dataclass
class ArbiterConfig:
    gaze_confidence_threshold: float = 0.55
    hand_confidence_threshold: float = 0.55
    ema_alpha: float = 0.3                  # smoothing applied to raw confidences
    low_light_brightness_threshold: float = 45.0  # mean pixel intensity 0-255


@dataclass
class FittsTaskConfig:
    amplitudes_px: tuple[float, ...] = (150, 250, 400, 550, 700)
    widths_px: tuple[float, ...] = (20, 35, 55, 80)
    n_directions: int = 8                    # ISO 9241-9 circular layout
    trials_per_condition: int = 1


@dataclass
class AppConfig:
    camera: CameraConfig = field(default_factory=CameraConfig)
    screen: ScreenConfig = field(default_factory=ScreenConfig)
    calibration: CalibrationConfig = field(default_factory=CalibrationConfig)
    gaze_filter: GazeFilterConfig = field(default_factory=GazeFilterConfig)
    pursuit: SmoothPursuitConfig = field(default_factory=SmoothPursuitConfig)
    dwell: AdaptiveDwellConfig = field(default_factory=AdaptiveDwellConfig)
    fatigue: FatigueConfig = field(default_factory=FatigueConfig)
    arbiter: ArbiterConfig = field(default_factory=ArbiterConfig)
    fitts: FittsTaskConfig = field(default_factory=FittsTaskConfig)


CONFIG = AppConfig()

for _dir in (MODELS_DIR, DATA_DIR, LOGS_DIR):
    _dir.mkdir(parents=True, exist_ok=True)
