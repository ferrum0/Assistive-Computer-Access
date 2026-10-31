"""OpenCV drawing helpers for the assistive interface overlay.

The whole UI is rendered directly onto a canvas with OpenCV rather than a
separate widget toolkit: it keeps the render loop and the gaze/selection
loop in the same thread/frame-tick, which matters for smooth pursuit (the
target trajectories and the correlation window must share one clock).
"""
from __future__ import annotations

import cv2
import numpy as np

from src.arbiter.modality_arbiter import ArbiterDecision, Modality
from src.fatigue_estimator.estimator import FatigueSnapshot
from src.selection_engine.smooth_pursuit import PursuitResult
from src.selection_engine.targets import OrbitTarget

WHITE = (255, 255, 255)
GRAY = (150, 150, 150)
GREEN = (90, 200, 90)
BLUE = (230, 160, 60)
RED = (60, 60, 220)
YELLOW = (40, 220, 230)
BLACK = (20, 20, 20)


def blank_canvas(w: int, h: int, bg=(30, 30, 30)) -> np.ndarray:
    canvas = np.full((h, w, 3), bg, dtype=np.uint8)
    return canvas


def make_fullscreen_window(name: str) -> None:
    """Create (or reuse) a window and put it into true OS-level fullscreen.

    `cv2.WND_PROP_FULLSCREEN` is a *property id* for `setWindowProperty`, not
    a window-creation flag -- passing it directly to `namedWindow` silently
    creates an ordinary small window instead (it happens to equal 0, same as
    `cv2.WINDOW_NORMAL`). Fullscreen must be requested via the two-call
    sequence below.
    """
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)


def detect_fullscreen_size(name: str, fallback: tuple[int, int]) -> tuple[int, int]:
    """Query the OS for the actual on-screen size (in the same units the
    window manager positions windows in) of an already-fullscreened window,
    instead of trusting a hardcoded config value.

    This matters a lot on Retina/HiDPI displays: `system_profiler` reports
    the panel's native *physical* pixel count (e.g. 3024x1964 on many MacBook
    Pros), but OpenCV's Cocoa-backed window operates in *logical points*
    (typically exactly half that at 2x scaling, e.g. 1512x982) -- and the two
    numbers also commonly have different aspect ratios than a configured
    default like 1920x1080, which would otherwise stretch gaze-to-cursor
    mapping unevenly between axes. A couple of `waitKey` pumps give the
    window manager a moment to apply the fullscreen transition before the
    size is queried; if the query still comes back degenerate for any reason,
    `fallback` (typically `CONFIG.screen`) is used instead.
    """
    for _ in range(5):
        cv2.waitKey(30)
    _x, _y, w, h = cv2.getWindowImageRect(name)
    if w <= 0 or h <= 0:
        return fallback
    return w, h


def draw_orbit_target(canvas: np.ndarray, target: OrbitTarget, t: float,
                       score: float | None, is_winner: bool) -> None:
    x, y = target.position(t)
    color = YELLOW if is_winner else BLUE
    radius = 16 if is_winner else 13
    cv2.circle(canvas, (int(x), int(y)), radius, color, -1, lineType=cv2.LINE_AA)
    cv2.circle(canvas, (int(target.center[0]), int(target.center[1])), 2, GRAY, -1)
    cv2.putText(canvas, target.label, (int(x) - 20, int(y) - 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 1, cv2.LINE_AA)
    if score is not None:
        cv2.putText(canvas, f"{score:+.2f}", (int(x) - 18, int(y) + 30),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.45, GRAY, 1, cv2.LINE_AA)


def draw_dwell_widget(canvas: np.ndarray, widget_id: str, center: tuple[float, float],
                       radius_px: float, progress: float, is_active: bool) -> None:
    cx, cy = int(center[0]), int(center[1])
    cv2.circle(canvas, (cx, cy), int(radius_px), GRAY, 2, cv2.LINE_AA)
    if is_active and progress > 0:
        angle = int(360 * progress)
        cv2.ellipse(canvas, (cx, cy), (int(radius_px), int(radius_px)), -90, 0,
                    angle, GREEN, 4, cv2.LINE_AA)
    cv2.putText(canvas, widget_id, (cx - 25, cy + int(radius_px) + 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 1, cv2.LINE_AA)


def draw_gaze_cursor(canvas: np.ndarray, xy: tuple[float, float] | None,
                      is_pinching: bool = False) -> None:
    """Draws the active pointer. Color signals which channel is driving it
    (red = gaze, orange = hand) so the modality arbiter's behavior is
    legible to the user as it switches, not just reported in the HUD text.
    A filled ring is added while a pinch gesture is held, as feedback that
    the gesture is being recognized.
    """
    if xy is None:
        return
    x, y = int(xy[0]), int(xy[1])
    cv2.drawMarker(canvas, (x, y), RED, markerType=cv2.MARKER_CROSS,
                    markerSize=16, thickness=2)
    if is_pinching:
        cv2.circle(canvas, (x, y), 22, GREEN, 3, cv2.LINE_AA)


def draw_hand_cursor(canvas: np.ndarray, xy: tuple[float, float] | None,
                      is_pinching: bool = False) -> None:
    if xy is None:
        return
    x, y = int(xy[0]), int(xy[1])
    cv2.drawMarker(canvas, (x, y), BLUE, markerType=cv2.MARKER_CROSS,
                    markerSize=16, thickness=2)
    if is_pinching:
        cv2.circle(canvas, (x, y), 22, GREEN, 3, cv2.LINE_AA)


def draw_hud(canvas: np.ndarray, decision: ArbiterDecision,
             fatigue: FatigueSnapshot, dwell_threshold_s: float, fps: float) -> None:
    lines = [
        f"modality: {decision.modality.value}  "
        f"(gaze={decision.gaze_confidence:.2f} hand={decision.hand_confidence:.2f}"
        f"{'  LOW LIGHT' if not decision.lighting_ok else ''})",
        f"fatigue f_hat={fatigue.f_hat:.2f}  PERCLOS={fatigue.perclos:.2f}  "
        f"blink_dur={fatigue.mean_blink_duration_s * 1000:.0f}ms  "
        f"jitter={fatigue.fixation_jitter_px:.1f}px",
        f"dwell threshold tau(t) = {dwell_threshold_s:.2f}s   fps={fps:.0f}",
    ]
    y = 24
    for line in lines:
        cv2.putText(canvas, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    WHITE, 1, cv2.LINE_AA)
        y += 24

    color = {
        Modality.FUSED: GREEN,
        Modality.GAZE_ONLY: BLUE,
        Modality.HAND_ONLY: YELLOW,
        Modality.DEGRADED: RED,
    }[decision.modality]
    cv2.circle(canvas, (canvas.shape[1] - 24, 24), 10, color, -1)


def fatigue_sparkline(canvas: np.ndarray, history: list[float], origin=(12, 100),
                       size=(240, 50)) -> None:
    """Small inline plot of f_hat(t) over the session, drawn with raw OpenCV
    primitives (keeps this module dependency-free besides cv2/numpy).
    """
    ox, oy = origin
    w, h = size
    cv2.rectangle(canvas, (ox, oy), (ox + w, oy + h), GRAY, 1)
    if len(history) < 2:
        return
    n = len(history)
    step = w / max(n - 1, 1)
    pts = []
    for i, val in enumerate(history):
        x = ox + i * step
        y = oy + h - val * h
        pts.append((int(x), int(y)))
    for p0, p1 in zip(pts[:-1], pts[1:]):
        cv2.line(canvas, p0, p1, YELLOW, 2, cv2.LINE_AA)
    cv2.putText(canvas, "f_hat(t)", (ox, oy - 6), cv2.FONT_HERSHEY_SIMPLEX,
                0.45, GRAY, 1, cv2.LINE_AA)
