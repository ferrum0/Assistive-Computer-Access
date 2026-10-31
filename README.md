# Gaze/Gesture Assistive Interface

A webcam-only (no dedicated eye-tracking hardware) assistive control interface for
people with limited hand mobility (ALS, spinal cord injury, late-stage Parkinson's).
The design goal is to avoid the **Midas touch problem** — the classic failure mode of
gaze interfaces where the system cannot tell "the user is just looking" from "the user
wants to click," which is why most existing tools (Tobii Dynavox, GazePointer, etc.)
fall back to slow, fatiguing fixed dwell-time clicking.

## Why smooth pursuit solves Midas touch

Dwell-time click infers *intent* from *fixation duration alone*, which is ambiguous:
resting your eyes on a button for a second is indistinguishable from wanting to
activate it. **Smooth pursuit selection** sidesteps the ambiguity entirely by changing
what evidence it looks for: each clickable element moves along its own small, unique
trajectory, and the interface asks "whose motion is the user's gaze *tracking*?" A
static gaze position can accidentally overlap a widget; a gaze *velocity pattern* that
matches one specific moving target for a sustained window essentially cannot happen by
accident. This is the same principle behind Vidal et al.'s "Pursuits" technique — the
system implemented here is a from-scratch reimplementation of that idea, purpose-built
around a dynamic dwell fallback and a fatigue/confidence-aware control loop.

## Architecture

```
main.py                         CLI entry point (calibrate / run / evaluate)
src/
  config.py                     All tunable constants in one place
  sensing/
    face_tracker.py             MediaPipe FaceLandmarker -> iris-offset raw gaze
                                 vector, blink EAR, per-frame tracking confidence
    hand_tracker.py              MediaPipe HandLandmarker -> fingertip pointer,
                                 pinch gesture, per-frame confidence
  calibration/
    calibrator.py                 9-point, <30s calibration; polynomial regression
                                 raw gaze -> screen (x, y)
  selection_engine/
    targets.py                    Orbiting-target trajectory model
    smooth_pursuit.py             Sliding-window cross-correlation selector (primary)
    adaptive_dwell.py             Fatigue-scaled dwell click (fallback)
    engine.py                     Combines both behind one interface
  fatigue_estimator/
    estimator.py                  f_hat(t) from PERCLOS + blink duration + gaze jitter
  arbiter/
    modality_arbiter.py           Confidence-based gaze/hand/fused/degraded switch
  evaluation/
    fitts_tasks.py                ISO 9241-9 style target-acquisition task generator
    metrics.py                    Trial logging, Fitts' law regression, throughput
    nasa_tlx.py                   Raw NASA-TLX (RTLX) workload questionnaire
    study_runner.py               Orchestrates the 3-mode comparison study
  ui/
    overlay.py                    OpenCV drawing helpers
    calibration_screen.py         Calibration UI loop
    app.py                        Sensing->arbiter->fatigue->selection pipeline +
                                   the interactive demo and evaluation UI loops
```

Everything renders through OpenCV directly onto one canvas per frame rather than a
separate widget toolkit, because smooth pursuit needs the target-trajectory clock, the
gaze-sampling clock, and the render clock to be the exact same clock — routing frames
through a second event loop (Tk/Qt) would add uncontrolled jitter to exactly the signal
the correlation depends on.

## Libraries

- **MediaPipe Tasks (FaceLandmarker with `refine_landmarks`, HandLandmarker)** — the
  sensing backbone. `refine_landmarks=True` is what exposes the 10 iris landmarks
  (indices 468-477) needed for gaze estimation; without it Face Mesh only gives eyelid
  contours, not iris position.
- **OpenCV** — camera capture, all UI rendering, color-space conversion.
- **NumPy** — polynomial regression (calibration), correlation (smooth pursuit),
  rolling statistics (fatigue), linear regression (Fitts' law analysis).
- No PyQt/Tkinter: an all-OpenCV canvas keeps sensing and rendering on a single clock
  (important for smooth pursuit) and keeps the dependency footprint minimal, which
  matters for an assistive tool that should be simple to install for a caregiver.

## Math

### 1. Calibration (polynomial regression)

Raw gaze is the iris center's position normalized against each eye's corner-to-corner
axis, `g = (g_x, g_y)`, averaged across both eyes. During calibration the user fixates
9 known screen points; per point we collect raw gaze samples, trim the first/last 15%
(settle-in transients), and average. We then fit two independent degree-2 polynomial
regressions by ordinary least squares:

```
X = a0 + a1*gx + a2*gy + a3*gx^2 + a4*gy^2 + a5*gx*gy
Y = b0 + b1*gx + b2*gy + b3*gx^2 + b4*gy^2 + b5*gx*gy
```

solved with `numpy.linalg.lstsq` on the monomial design matrix. A quadratic surface
absorbs the mild nonlinearity between iris offset and gaze angle without requiring many
calibration points — 9 points at ~2.5s each keeps the whole procedure under 30s.

### 2. Smooth pursuit selection

Each target `i` moves on `T_i(t) = c_i + r_i * (cos(w_i t + phi_i), sin(w_i t + phi_i))`,
with a unique phase `phi_i` (and slightly jittered period/radius) per target so
trajectories stay mutually decorrelated. Over a sliding window of gaze samples we
compute, in the **velocity domain** (first differences, to remove the constant
calibration-offset between gaze estimate and orbit and to capture the pursuit reflex
directly):

```
score_i = max_{tau in [0, 200ms]} ( corr(dGx, dTix(t - tau)) + corr(dGy, dTiy(t - tau)) ) / 2
```

The `tau` search accounts for the ~100-200ms physiological lag of the oculomotor
smooth-pursuit response. A selection is confirmed only when the winning target's score
exceeds a threshold **and** beats the runner-up by a margin **and** stays the winner for
a sustained confirm window — this debounces the decision so a single noisy frame cannot
trigger a false activation. Full derivation and code: `selection_engine/smooth_pursuit.py`.

### 3. Adaptive dwell fallback

For dense/small targets that cannot host a legible orbit, dwell click remains, but its
threshold is not fixed:

```
tau(t) = tau_0 * (1 + lambda * f_hat(t))
```

`tau_0` is the alert-user baseline threshold, `lambda` controls sensitivity, `f_hat(t)`
is the fatigue estimate (below). As the user tires, `tau(t)` grows, trading a bit of
speed for fewer accidental dwell-triggers exactly when the user is least able to
self-correct a mis-click. Setting `lambda = 0` reproduces classic *fixed* dwell click,
which is how the evaluation study derives its "fixed dwell" baseline condition from the
same code path as "adaptive dwell" (see `evaluation/study_runner.py`).

### 4. Fatigue estimator

`f_hat(t) in [0, 1]` combines three evidence-backed signals over a 60s rolling window:

- **PERCLOS** — fraction of frames with eye-aspect-ratio (EAR) below a closure
  threshold; the standard drowsiness proxy in driver-monitoring research.
- **Blink duration** — fatigued users produce longer blinks than their own alert
  baseline.
- **Fixation jitter** — position variance during low-velocity (putative fixation)
  gaze segments increases as oculomotor control degrades.

```
f_raw(t) = w1*f_perclos + w2*f_duration + w3*f_jitter      (w1+w2+w3 = 1)
f_hat(t) = alpha * f_raw(t) + (1 - alpha) * f_hat(t-1)      (EMA smoothing)
```

The live HUD (`ui/overlay.py: draw_hud`, `fatigue_sparkline`) plots `f_hat(t)` and the
resulting `tau(t)` throughout the session.

### 5. Modality arbiter

Per-frame confidences `c_gaze` (from face tracking, discounted by frame brightness as an
early low-light warning independent of MediaPipe's own score) and `c_hand` (hand
detector's handedness score) are each EMA-smoothed, then thresholded independently:

```
gaze_ok & hand_ok   -> FUSED          (gaze points, hand pinch confirms)
gaze_ok & !hand_ok  -> GAZE_ONLY
!gaze_ok & hand_ok  -> HAND_ONLY
!gaze_ok & !hand_ok -> DEGRADED       (freeze cursor, max out dwell threshold,
                                        surface a low-confidence indicator)
```

This is what lets the interface degrade smoothly under poor lighting or user movement
instead of failing outright.

### 6. Fitts' law analysis (evaluation)

Tasks follow the ISO 9241-9 multidirectional tapping layout: `n` targets evenly spaced
on a circle of diameter `A` (amplitude), each with tolerance/width `W`; a trial acquires
the target diametrically opposite the previous one. Index of difficulty (Shannon
formulation):

```
ID = log2(A / W + 1)                    [bits]
```

`evaluation/fitts_tasks.py` spans a set of `(A, W)` pairs sorted by `ID` and subsampled
to 10-15 trials covering an easy-to-hard range. Per selection mode, `evaluation/metrics.py`
fits `MT = a + b * ID` by OLS and reports throughput `TP = ID / MT` (bits/s) and `R^2` as
a goodness-of-fit check for the Fitts model itself, alongside the error rate (wrong
target selected, or trial timeout).

## Setup

```bash
pip install -r requirements.txt
```

Download the two MediaPipe Tasks model bundles into `models/`:

```bash
curl -L -o models/face_landmarker.task \
  https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task
curl -L -o models/hand_landmarker.task \
  https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task
```

Set your actual screen resolution in `src/config.py: ScreenConfig` (`CONFIG.screen`).

## Running

```bash
python main.py calibrate                 # <30s calibration, saved to data/calibration.json
python main.py run                       # interactive assistive demo (orbiting + dwell widgets)
python main.py evaluate --participant P01 --tlx   # 3-mode comparison study + NASA-TLX (RTLX)
```

`evaluate` runs all three conditions (`fixed_dwell`, `adaptive_dwell`, `smooth_pursuit`)
back-to-back over the same Fitts task set, logs every trial to
`data/logs/trials_<participant>_<timestamp>.csv`, and prints a Fitts-law comparison
table at the end. Pass `--modes` to restrict to a subset. For a real user study, run
this with 3-5 participants and aggregate the per-participant CSVs; `--tlx` collects a
Raw NASA-TLX workload score per condition per participant, logged alongside the trial
data (`..._tlx.csv`).

## Notes and limitations

- Gaze mapping does not compensate for head pose; it assumes the head stays roughly
  centered in the camera frame during use (reasonable for a seated assistive-tech
  setup, but a head-pose correction term — e.g. from `solvePnP` on stable facial
  landmarks — would improve robustness for users who cannot hold their head still).
- The hand pointer maps normalized fingertip position directly to screen space; unlike
  gaze it is not calibrated per-user. For precision hand pointing, apply the same
  polynomial-regression calibration used for gaze.
- Smooth pursuit orbit radius/period and the dwell/pursuit correlation thresholds in
  `src/config.py` are reasonable starting points, not tuned defaults — they should be
  adjusted per user, and the adaptive dwell `lambda` in particular should be set from a
  short baseline-fatigue observation period for each individual.
