"""Main application: wires sensing -> arbiter -> fatigue estimator ->
selection engine -> overlay rendering into one per-frame pipeline, then
exposes two entry points: an interactive assistive-desktop demo, and the
evaluation-study runner.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import cv2
import numpy as np

from src.arbiter.modality_arbiter import ArbiterDecision, Modality, ModalityArbiter
from src.calibration.calibrator import GazeCalibrator
from src.config import CONFIG, LOGS_DIR
from src.evaluation.fitts_tasks import generate_tasks
from src.evaluation.metrics import TrialLogger, compare_modes, default_log_path, load_trials, print_comparison_table
from src.evaluation.nasa_tlx import TLXLogger, collect_tlx_cli
from src.evaluation.study_runner import StudyRunner, dwell_config_for_mode
from src.fatigue_estimator.estimator import FatigueSnapshot, FatigueEstimator
from src.sensing.face_tracker import FaceTracker
from src.sensing.hand_tracker import HandTracker
from src.selection_engine.adaptive_dwell import AdaptiveDwellSelector
from src.selection_engine.engine import SelectionEngine, SelectionEvent
from src.selection_engine.smooth_pursuit import SmoothPursuitSelector
from src.selection_engine.targets import make_orbit_targets
from src.utils.buffers import EMA
from src.ui.overlay import (blank_canvas, draw_dwell_widget, draw_gaze_cursor,
                             draw_hand_cursor, draw_hud, draw_orbit_target,
                             fatigue_sparkline, make_fullscreen_window)


@dataclass
class FramePerception:
    t: float
    gaze_screen_xy: tuple[float, float] | None
    hand_pointer_screen_xy: tuple[float, float] | None
    is_pinch: bool
    pinch_edge: bool                                 # True only on the frame a pinch starts
    effective_pointer_screen_xy: tuple[float, float] | None  # gaze or hand, per current modality
    decision: ArbiterDecision
    fatigue: FatigueSnapshot


class Pipeline:
    """Owns the sensors + arbiter + fatigue estimator and produces one
    `FramePerception` per camera frame. Shared by both the demo UI and the
    evaluation runner so the two modes are measured under identical sensing
    conditions.
    """

    def __init__(self, calibrator: GazeCalibrator):
        cfg = CONFIG
        self.calibrator = calibrator
        # The calibrator's own screen_w/screen_h (detected from the OS at
        # calibration time, see calibration_screen.py) is the source of
        # truth for the coordinate space gaze predictions are expressed in
        # -- using CONFIG.screen here instead could silently disagree with
        # it (e.g. on Retina displays, or a different monitor).
        self.screen_w = calibrator.screen_w
        self.screen_h = calibrator.screen_h
        self.face_tracker = FaceTracker(str(cfg_model_path("face")))
        self.hand_tracker = HandTracker(str(cfg_model_path("hand")))
        self.arbiter = ModalityArbiter(
            gaze_threshold=cfg.arbiter.gaze_confidence_threshold,
            hand_threshold=cfg.arbiter.hand_confidence_threshold,
            ema_alpha=cfg.arbiter.ema_alpha,
            low_light_brightness_threshold=cfg.arbiter.low_light_brightness_threshold,
        )
        self.fatigue = FatigueEstimator(
            window_seconds=cfg.fatigue.window_seconds,
            ema_alpha=cfg.fatigue.ema_alpha,
            perclos_weight=cfg.fatigue.perclos_weight,
            blink_duration_weight=cfg.fatigue.blink_duration_weight,
            jitter_weight=cfg.fatigue.jitter_weight,
            perclos_eye_closed_threshold=cfg.fatigue.perclos_eye_closed_threshold,
            baseline_blink_duration_s=cfg.fatigue.baseline_blink_duration_s,
            baseline_jitter_px=cfg.fatigue.baseline_jitter_px,
            blink_duration_saturation_s=cfg.fatigue.blink_duration_saturation_s,
            jitter_saturation_px=cfg.fatigue.jitter_saturation_px,
        )
        self._frame_ts_ms = 0
        self.f_hat_history: list[float] = []
        self._gaze_x_ema = EMA(cfg.gaze_filter.smoothing_alpha)
        self._gaze_y_ema = EMA(cfg.gaze_filter.smoothing_alpha)
        self._prev_is_pinch = False

    def close(self) -> None:
        self.face_tracker.close()
        self.hand_tracker.close()

    def step(self, bgr_frame: np.ndarray) -> FramePerception:
        self._frame_ts_ms += 33
        t = time.time()

        face_sample = self.face_tracker.process(bgr_frame, self._frame_ts_ms)
        hand_sample = self.hand_tracker.process(bgr_frame, self._frame_ts_ms)
        brightness = float(np.mean(cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2GRAY)))

        decision = self.arbiter.update(face_sample.confidence, hand_sample.confidence,
                                        brightness)

        gaze_screen_xy = None
        if decision.modality in (Modality.FUSED, Modality.GAZE_ONLY):
            if face_sample.raw_gaze is not None:
                raw_screen_xy = self.calibrator.predict(face_sample.raw_gaze)
                gaze_screen_xy = (self._gaze_x_ema.update(raw_screen_xy[0]),
                                  self._gaze_y_ema.update(raw_screen_xy[1]))
            elif self._gaze_x_ema.value is not None:
                # No reliable raw gaze this frame (most commonly a blink):
                # hold the last smoothed position instead of letting the
                # cursor jump to a noisy estimate or flicker away entirely.
                gaze_screen_xy = (self._gaze_x_ema.value, self._gaze_y_ema.value)

        hand_screen_xy = None
        if hand_sample.pointer is not None and decision.modality in (
                Modality.FUSED, Modality.HAND_ONLY):
            # The camera frame is not mirrored, but the user faces the
            # camera like a mirror -- without flipping x, moving a hand to
            # the user's left moves right across the raw frame, so the
            # cursor would move the opposite way from what feels natural.
            # Gaze doesn't show this because its regression silently learns
            # whatever raw-frame convention is consistent at calibration
            # time; the hand pointer has no such calibration, so it must be
            # flipped explicitly.
            hand_screen_xy = ((1.0 - hand_sample.pointer[0]) * self.screen_w,
                               hand_sample.pointer[1] * self.screen_h)

        pinch_edge = hand_sample.is_pinch and not self._prev_is_pinch
        self._prev_is_pinch = hand_sample.is_pinch

        # Which channel actually drives the cursor/selection this frame:
        # gaze when it's trusted (FUSED or GAZE_ONLY), hand position as the
        # fallback when gaze isn't (HAND_ONLY) -- this is what makes the
        # arbiter's modality switch actually change interface behavior
        # instead of only being reported in the HUD.
        if decision.modality in (Modality.FUSED, Modality.GAZE_ONLY):
            effective_pointer_screen_xy = gaze_screen_xy
        elif decision.modality is Modality.HAND_ONLY:
            effective_pointer_screen_xy = hand_screen_xy
        else:
            effective_pointer_screen_xy = None

        self.fatigue.push_ear(t, face_sample.ear, face_sample.is_blink)
        self.fatigue.push_gaze_px(t, gaze_screen_xy)
        fatigue_snapshot = self.fatigue.update(t)
        self.f_hat_history.append(fatigue_snapshot.f_hat)
        if len(self.f_hat_history) > 400:
            self.f_hat_history.pop(0)

        return FramePerception(
            t=t, gaze_screen_xy=gaze_screen_xy,
            hand_pointer_screen_xy=hand_screen_xy,
            is_pinch=hand_sample.is_pinch, pinch_edge=pinch_edge,
            effective_pointer_screen_xy=effective_pointer_screen_xy,
            decision=decision, fatigue=fatigue_snapshot,
        )


def cfg_model_path(kind: str):
    from src.config import FACE_LANDMARKER_MODEL, HAND_LANDMARKER_MODEL
    return FACE_LANDMARKER_MODEL if kind == "face" else HAND_LANDMARKER_MODEL


def open_camera() -> cv2.VideoCapture:
    cam_cfg = CONFIG.camera
    cap = cv2.VideoCapture(cam_cfg.device_index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, cam_cfg.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cam_cfg.height)
    cap.set(cv2.CAP_PROP_FPS, cam_cfg.target_fps)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam.")
    return cap


def _demo_widgets(screen_w: int, screen_h: int):
    pursuit_specs = [
        {"id": "open_app", "label": "Open", "center": (screen_w * 0.2, screen_h * 0.3)},
        {"id": "call", "label": "Call", "center": (screen_w * 0.5, screen_h * 0.3)},
        {"id": "settings", "label": "Settings", "center": (screen_w * 0.8, screen_h * 0.3)},
    ]
    dwell_widgets = {
        "yes": (screen_w * 0.3, screen_h * 0.7),
        "no": (screen_w * 0.5, screen_h * 0.7),
        "help": (screen_w * 0.7, screen_h * 0.7),
    }
    return pursuit_specs, dwell_widgets


def _pinch_hit_test(pointer_xy: tuple[float, float], t: float,
                     pursuit_targets: list, pursuit_hit_radius: float,
                     dwell_widgets: dict[str, tuple[float, float]],
                     dwell_hit_radius: float) -> SelectionEvent | None:
    """A pinch is already an unambiguous, deliberate gesture -- it doesn't
    need pursuit's correlation window or dwell's timing threshold to confirm
    intent, so it bypasses both and fires immediately against whatever the
    active pointer is currently over. Pursuit targets are hit-tested at
    their current animated position (not their resting center), since that
    is where the target visually is right now.
    """
    px, py = pointer_xy
    for target in pursuit_targets:
        tx, ty = target.position(t)
        if (px - tx) ** 2 + (py - ty) ** 2 <= pursuit_hit_radius ** 2:
            return SelectionEvent(target.id, "pinch", None)
    for widget_id, (wx, wy) in dwell_widgets.items():
        if (px - wx) ** 2 + (py - wy) ** 2 <= dwell_hit_radius ** 2:
            return SelectionEvent(widget_id, "pinch", None)
    return None


def run_assistive_demo(calibrator: GazeCalibrator) -> None:
    cfg = CONFIG
    cap = open_camera()
    pipeline = Pipeline(calibrator)
    screen_w, screen_h = pipeline.screen_w, pipeline.screen_h

    pursuit_selector = SmoothPursuitSelector(
        window_seconds=cfg.pursuit.window_seconds, min_samples=cfg.pursuit.min_samples,
        correlation_threshold=cfg.pursuit.correlation_threshold,
        margin_over_runner_up=cfg.pursuit.margin_over_runner_up,
        confirm_seconds=cfg.pursuit.confirm_seconds,
        max_lag_seconds=cfg.pursuit.max_lag_seconds,
        lag_step_seconds=cfg.pursuit.lag_step_seconds,
    )
    dwell_selector = AdaptiveDwellSelector(
        base_threshold_s=cfg.dwell.base_threshold_seconds,
        fatigue_sensitivity=cfg.dwell.fatigue_sensitivity,
        max_threshold_s=cfg.dwell.max_threshold_seconds,
        fixation_radius_px=cfg.dwell.fixation_radius_px,
    )
    engine = SelectionEngine(pursuit_selector, dwell_selector)

    pursuit_specs, dwell_widgets = _demo_widgets(screen_w, screen_h)
    pursuit_targets = make_orbit_targets(pursuit_specs, cfg.pursuit.orbit_radius_px,
                                          cfg.pursuit.orbit_period_seconds)

    window = "Assistive Gaze/Gesture Interface"
    make_fullscreen_window(window)
    last_event: SelectionEvent | None = None
    last_event_t = 0.0
    fps_ema = 15.0
    prev_t = time.time()

    while True:
        ok, frame = cap.read()
        if not ok:
            continue
        perception = pipeline.step(frame)

        event, pursuit_result, dwell_state = engine.update(
            t=perception.t, gaze_xy=perception.effective_pointer_screen_xy,
            pursuit_targets=pursuit_targets, dwell_widgets=dwell_widgets,
            f_hat=perception.fatigue.f_hat,
        )

        # A pinch is already a deliberate, unambiguous gesture -- let it
        # confirm a selection immediately rather than waiting for pursuit's
        # correlation window or dwell's timeout, whenever the hand channel
        # is actually trusted this frame.
        if (perception.pinch_edge
                and perception.decision.modality in (Modality.FUSED, Modality.HAND_ONLY)
                and perception.effective_pointer_screen_xy is not None):
            pinch_event = _pinch_hit_test(
                perception.effective_pointer_screen_xy, perception.t,
                pursuit_targets, 30.0, dwell_widgets, cfg.dwell.fixation_radius_px)
            if pinch_event is not None:
                event = pinch_event

        if event is not None:
            last_event, last_event_t = event, perception.t

        canvas = blank_canvas(screen_w, screen_h)
        score_by_id = {s.target_id: s.score for s in pursuit_result.all_scores}
        winner_id = pursuit_result.winner.target_id if pursuit_result.winner else None
        for target in pursuit_targets:
            draw_orbit_target(canvas, target, perception.t,
                               score_by_id.get(target.id), target.id == winner_id)

        for wid, center in dwell_widgets.items():
            is_active = dwell_state.target_id == wid
            draw_dwell_widget(canvas, wid, center, cfg.dwell.fixation_radius_px,
                               dwell_state.progress if is_active else 0.0, is_active)

        if perception.decision.modality is Modality.HAND_ONLY:
            draw_hand_cursor(canvas, perception.effective_pointer_screen_xy, perception.is_pinch)
        else:
            draw_gaze_cursor(canvas, perception.effective_pointer_screen_xy, perception.is_pinch)
        draw_hud(canvas, perception.decision, perception.fatigue,
                 dwell_selector.threshold(perception.fatigue.f_hat), fps_ema)
        fatigue_sparkline(canvas, pipeline.f_hat_history)

        if last_event is not None and (perception.t - last_event_t) < 1.5:
            cv2.putText(canvas, f"SELECTED: {last_event.widget_id} ({last_event.mode})",
                        (12, screen_h - 20), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (90, 220, 90), 2, cv2.LINE_AA)

        cv2.imshow(window, canvas)
        now = time.time()
        fps_ema = 0.9 * fps_ema + 0.1 * (1.0 / max(now - prev_t, 1e-3))
        prev_t = now
        if cv2.waitKey(1) & 0xFF == 27:
            break

    pipeline.close()
    cap.release()
    cv2.destroyAllWindows()


def run_evaluation(calibrator: GazeCalibrator, participant_id: str,
                    modes: list[str] | None = None, with_tlx: bool = False) -> None:
    cfg = CONFIG
    modes = modes or ["fixed_dwell", "adaptive_dwell", "smooth_pursuit"]

    cap = open_camera()
    pipeline = Pipeline(calibrator)
    screen_w, screen_h = pipeline.screen_w, pipeline.screen_h

    log_path = default_log_path(LOGS_DIR, participant_id)
    trial_logger = TrialLogger(log_path)
    tlx_logger = TLXLogger(log_path.with_name(log_path.stem + "_tlx.csv")) if with_tlx else None

    specs = generate_tasks(screen_w, screen_h,
                            cfg.fitts.amplitudes_px, cfg.fitts.widths_px,
                            n_directions=cfg.fitts.n_directions)

    window = "Evaluation"
    make_fullscreen_window(window)

    for mode in modes:
        print(f"\n=== Running mode: {mode} ({len(specs)} trials) ===")
        dwell_cfg = dwell_config_for_mode(mode, cfg.dwell)
        pursuit_selector = SmoothPursuitSelector(
            window_seconds=cfg.pursuit.window_seconds, min_samples=cfg.pursuit.min_samples,
            correlation_threshold=cfg.pursuit.correlation_threshold,
            margin_over_runner_up=cfg.pursuit.margin_over_runner_up,
            confirm_seconds=cfg.pursuit.confirm_seconds,
            max_lag_seconds=cfg.pursuit.max_lag_seconds,
            lag_step_seconds=cfg.pursuit.lag_step_seconds,
        )
        dwell_selector = AdaptiveDwellSelector(
            base_threshold_s=dwell_cfg.base_threshold_seconds,
            fatigue_sensitivity=dwell_cfg.fatigue_sensitivity,
            max_threshold_s=dwell_cfg.max_threshold_seconds,
            fixation_radius_px=dwell_cfg.fixation_radius_px,
        )
        engine = SelectionEngine(pursuit_selector, dwell_selector)
        runner = StudyRunner(participant_id, mode, specs, trial_logger,
                              orbit_period_s=cfg.pursuit.orbit_period_seconds)

        while not runner.is_finished:
            ok, frame = cap.read()
            if not ok:
                continue
            perception = pipeline.step(frame)

            event, pursuit_result, dwell_state = engine.update(
                t=perception.t, gaze_xy=perception.gaze_screen_xy,
                pursuit_targets=runner.pursuit_targets(),
                dwell_widgets=runner.dwell_widgets(),
                f_hat=perception.fatigue.f_hat,
            )
            outcome = runner.on_selection_event(perception.t, event)

            canvas = blank_canvas(screen_w, screen_h)
            spec = runner.current_spec
            if spec is not None:
                cv2.putText(canvas, f"mode={mode}  trial {runner.trial_number}/{len(specs)}  "
                                     f"ID={spec.index_of_difficulty:.2f} bits",
                            (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                score_by_id = {s.target_id: s.score for s in pursuit_result.all_scores}
                winner_id = pursuit_result.winner.target_id if pursuit_result.winner else None
                for target in runner.pursuit_targets():
                    draw_orbit_target(canvas, target, perception.t,
                                       score_by_id.get(target.id), target.id == winner_id)
                for wid, center in runner.dwell_widgets().items():
                    is_active = dwell_state.target_id == wid
                    draw_dwell_widget(canvas, wid, center, spec.width_px / 2,
                                       dwell_state.progress if is_active else 0.0, is_active)
            draw_gaze_cursor(canvas, perception.gaze_screen_xy)
            if outcome is not None:
                msg = "CORRECT" if outcome.correct else "ERROR"
                cv2.putText(canvas, msg, (screen_w // 2 - 60, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                            (90, 220, 90) if outcome.correct else (60, 60, 220), 2)

            cv2.imshow(window, canvas)
            if cv2.waitKey(1) & 0xFF == 27:
                raise KeyboardInterrupt("Evaluation aborted by user")

        if tlx_logger is not None:
            cv2.destroyWindow(window)
            response = collect_tlx_cli(participant_id, mode)
            tlx_logger.log(response)
            make_fullscreen_window(window)

    trial_logger.close()
    if tlx_logger is not None:
        tlx_logger.close()
    pipeline.close()
    cap.release()
    cv2.destroyAllWindows()

    all_trials = load_trials(log_path)
    summaries = compare_modes(all_trials)
    print("\n=== Cross-mode comparison (Fitts' law fit MT = a + b*ID) ===")
    print_comparison_table(summaries)
    print(f"\nRaw trial log: {log_path}")
