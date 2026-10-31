"""Runs the <30s calibration sequence: show each grid point, collect raw
gaze samples while the user fixates it, fit the polynomial regression, and
save the result to disk for reuse by the main app / evaluation runner.
"""
from __future__ import annotations

import time

import cv2

from src.calibration.calibrator import GazeCalibrator
from src.config import CALIBRATION_FILE, CONFIG
from src.sensing.face_tracker import FaceTracker
from src.ui.overlay import (WHITE, YELLOW, blank_canvas, detect_fullscreen_size,
                             make_fullscreen_window)


_BLACK_FRAME_BRIGHTNESS = 2.0


def _preflight_camera_check(cap: cv2.VideoCapture, face_tracker: FaceTracker,
                             warmup_timeout_s: float = 5.0) -> None:
    """Poll frames before calibration starts (up to `warmup_timeout_s`) and
    fail fast with a diagnosable message instead of silently collecting zero
    samples for every point.

    Many webcams -- especially over macOS's AVFoundation backend right after
    permission is granted or the process is (re)launched -- deliver several
    hundred ms to a few seconds of all-black frames while auto-exposure
    ramps up, so this keeps polling for a non-black frame rather than
    judging camera health from a handful of frames read immediately after
    `VideoCapture` opens.
    """
    start = time.time()
    last_frame = None
    last_brightness = 0.0
    while time.time() - start < warmup_timeout_s:
        ok, frame = cap.read()
        if ok and frame is not None:
            last_frame = frame
            last_brightness = float(frame.mean())
            if last_brightness >= _BLACK_FRAME_BRIGHTNESS:
                break
        time.sleep(0.03)

    if last_frame is None:
        raise RuntimeError(
            "Camera preflight failed: cap.read() never returned a frame "
            f"within {warmup_timeout_s:.0f}s. Check that no other app is "
            "using the webcam and that CONFIG.camera.device_index is correct."
        )

    if last_brightness < _BLACK_FRAME_BRIGHTNESS:
        raise RuntimeError(
            f"Camera preflight failed: frames are still essentially black "
            f"after {warmup_timeout_s:.0f}s (mean brightness "
            f"{last_brightness:.2f}/255). This almost always means camera "
            f"access has not been granted to the actual process running "
            f"this script. If you're running from an IDE's integrated "
            f"terminal (VS Code, PyCharm, ...), macOS tracks camera "
            f"permission per parent app -- check System Settings -> "
            f"Privacy & Security -> Camera for the IDE itself (e.g. "
            f"'Visual Studio Code'), not just 'Terminal', enable it, then "
            f"fully quit and relaunch the IDE (permission changes often "
            f"don't apply to an already-running process). If it still "
            f"fails, run this script from Terminal.app directly instead."
        )

    sample = face_tracker.process(last_frame, 0)
    print(f"[calibration] preflight: camera warmed up after "
          f"{time.time() - start:.1f}s, mean brightness={last_brightness:.1f}/255, "
          f"face_found={sample.face_found}")
    if not sample.face_found:
        print("[calibration] WARNING: no face detected during preflight. "
              "Make sure your face is well-lit and inside the camera frame "
              "before the calibration dots appear.")


def run_calibration(face_tracker: FaceTracker, cap: cv2.VideoCapture) -> GazeCalibrator:
    cfg = CONFIG.calibration

    _preflight_camera_check(cap, face_tracker)

    window = "Calibration"
    make_fullscreen_window(window)
    screen_w, screen_h = detect_fullscreen_size(
        window, fallback=(CONFIG.screen.width, CONFIG.screen.height))
    print(f"[calibration] detected on-screen canvas: {screen_w}x{screen_h}")

    calibrator = GazeCalibrator(screen_w, screen_h, grid=cfg.grid,
                                 degree=cfg.poly_degree)

    total_est = calibrator.estimated_duration_s(cfg.seconds_per_point, cfg.settle_seconds)
    print(f"[calibration] {len(calibrator.points)} points, "
          f"~{total_est:.1f}s total")

    frame_ts_ms = 0
    for idx, point in enumerate(calibrator.points):
        point_start = time.time()
        frames_seen = frames_face_found = frames_blink = samples_added = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                continue
            frame_ts_ms += 33

            elapsed = time.time() - point_start
            canvas = blank_canvas(screen_w, screen_h)
            settled = elapsed >= cfg.settle_seconds
            color = YELLOW if settled else WHITE
            radius = 14 if settled else 10
            cx, cy = int(point.screen_xy[0]), int(point.screen_xy[1])
            cv2.circle(canvas, (cx, cy), radius, color, -1, cv2.LINE_AA)
            cv2.putText(canvas, f"Look at the dot  ({idx + 1}/{len(calibrator.points)})",
                        (40, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, WHITE, 2, cv2.LINE_AA)
            cv2.imshow(window, canvas)
            if cv2.waitKey(1) & 0xFF == 27:
                raise KeyboardInterrupt("Calibration aborted by user")

            if settled:
                frames_seen += 1
                sample = face_tracker.process(frame, frame_ts_ms)
                frames_face_found += int(sample.face_found)
                frames_blink += int(sample.is_blink)
                if sample.face_found and sample.raw_gaze is not None and not sample.is_blink:
                    calibrator.add_sample(idx, sample.raw_gaze)
                    samples_added += 1

            if elapsed >= cfg.seconds_per_point:
                break

        print(f"[calibration] point {idx + 1}/{len(calibrator.points)}: "
              f"{samples_added} samples added "
              f"(frames={frames_seen}, face_found={frames_face_found}, "
              f"blink={frames_blink})")

    cv2.destroyWindow(window)
    report = calibrator.fit()
    print(f"[calibration] done: rmse_x={report['rmse_x_px']:.1f}px "
          f"rmse_y={report['rmse_y_px']:.1f}px n_points={report['n_points']}")
    calibrator.save(CALIBRATION_FILE)
    return calibrator
