"""Face/eye sensing: iris-based raw gaze vector, blink (EAR) and a per-frame
tracking-confidence score, built on MediaPipe's FaceLandmarker (Tasks API,
refine_landmarks=True gives the 10 iris landmarks in addition to the 468
face mesh points).

Landmark index reference (MediaPipe Face Mesh, 468+iris topology):
  Left eye corners:   33 (outer), 133 (inner)
  Left eye lids:       159 (upper), 145 (lower)
  Left iris:            468-472 (center = 468)
  Right eye corners:  362 (inner), 263 (outer)
  Right eye lids:      386 (upper), 374 (lower)
  Right iris:           473-477 (center = 473)
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

LEFT_EYE_CORNERS = (33, 133)
LEFT_EYE_LIDS = (159, 145)
LEFT_IRIS_CENTER = 468

RIGHT_EYE_CORNERS = (362, 263)
RIGHT_EYE_LIDS = (386, 374)
RIGHT_IRIS_CENTER = 473


@dataclass
class FaceSample:
    t: float
    raw_gaze: tuple[float, float] | None   # normalized iris offset, eye-frame
    ear: float | None                      # eye-aspect-ratio (blink signal)
    is_blink: bool
    confidence: float                      # 0..1 tracking reliability
    face_found: bool


def _landmark_xy(landmarks, idx: int) -> np.ndarray:
    lm = landmarks[idx]
    return np.array([lm.x, lm.y], dtype=np.float64)


def _eye_aspect_ratio(landmarks, corners, lids) -> float:
    p_outer = _landmark_xy(landmarks, corners[0])
    p_inner = _landmark_xy(landmarks, corners[1])
    p_upper = _landmark_xy(landmarks, lids[0])
    p_lower = _landmark_xy(landmarks, lids[1])
    horiz = np.linalg.norm(p_outer - p_inner) + 1e-8
    vert = np.linalg.norm(p_upper - p_lower)
    return float(vert / horiz)


def _iris_offset(landmarks, corners, iris_center_idx) -> np.ndarray:
    """Iris center position normalized to [-1, 1] within the eye corners.

    0 = iris centered between the corners; sign follows screen-space x/y so
    e.g. -1 means the iris is pinned to the first corner.
    """
    p_a = _landmark_xy(landmarks, corners[0])
    p_b = _landmark_xy(landmarks, corners[1])
    iris = _landmark_xy(landmarks, iris_center_idx)
    span = p_b - p_a
    span_len = np.linalg.norm(span) + 1e-8
    # Project iris onto the corner-to-corner axis and its perpendicular.
    axis = span / span_len
    rel = iris - p_a
    along = np.dot(rel, axis) / span_len           # 0..1 along the eye
    perp_axis = np.array([-axis[1], axis[0]])
    eye_height_proxy = span_len * 0.5
    across = np.dot(rel, perp_axis) / (eye_height_proxy + 1e-8)
    return np.array([along * 2 - 1, across], dtype=np.float64)


class FaceTracker:
    """Wraps mediapipe.tasks.vision.FaceLandmarker in VIDEO mode.

    Call `process(bgr_frame, timestamp_ms)` once per camera frame.
    """

    def __init__(self, model_path: str, ear_blink_threshold: float = 0.20,
                 min_face_confidence: float = 0.5):
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision as mp_vision

        self._mp_vision = mp_vision
        # Force the CPU delegate: the default GPU/Metal delegate depends on a
        # graphics service context that isn't available outside an
        # interactive GUI session (e.g. run from a plain terminal or a
        # sandboxed process) and crashes the whole process with
        # "Check failed: service_ Service is unavailable" rather than
        # raising a catchable Python exception.
        base_options = mp_python.BaseOptions(
            model_asset_path=model_path,
            delegate=mp_python.BaseOptions.Delegate.CPU,
        )
        options = mp_vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=mp_vision.RunningMode.VIDEO,
            num_faces=1,
            min_face_detection_confidence=min_face_confidence,
            min_face_presence_confidence=min_face_confidence,
            min_tracking_confidence=min_face_confidence,
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=False,
        )
        self._landmarker = mp_vision.FaceLandmarker.create_from_options(options)
        self._ear_blink_threshold = ear_blink_threshold
        self._mp_image_cls = mp.Image
        self._mp_image_format = mp.ImageFormat.SRGB

    def process(self, bgr_frame: np.ndarray, timestamp_ms: int) -> FaceSample:
        import cv2
        rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        mp_image = self._mp_image_cls(image_format=self._mp_image_format, data=rgb)
        result = self._landmarker.detect_for_video(mp_image, timestamp_ms)

        now = time.time()
        if not result.face_landmarks:
            return FaceSample(t=now, raw_gaze=None, ear=None, is_blink=False,
                               confidence=0.0, face_found=False)

        landmarks = result.face_landmarks[0]
        if len(landmarks) < 478:
            # Iris refinement not available -> cannot estimate gaze reliably.
            return FaceSample(t=now, raw_gaze=None, ear=None, is_blink=False,
                               confidence=0.2, face_found=True)

        ear_l = _eye_aspect_ratio(landmarks, LEFT_EYE_CORNERS, LEFT_EYE_LIDS)
        ear_r = _eye_aspect_ratio(landmarks, RIGHT_EYE_CORNERS, RIGHT_EYE_LIDS)
        ear = (ear_l + ear_r) / 2.0
        is_blink = ear < self._ear_blink_threshold

        # During a blink the eyelid is closing/closed, so the iris landmarks
        # it would be computed from are unreliable -- a half- or fully-closed
        # eye produces a garbage iris-offset estimate, not a slightly-noisy
        # one. Withholding raw_gaze entirely (rather than returning that
        # garbage estimate) lets downstream consumers hold their last good
        # position instead of visibly jumping on every blink.
        if is_blink:
            raw_gaze = None
        else:
            gaze_l = _iris_offset(landmarks, LEFT_EYE_CORNERS, LEFT_IRIS_CENTER)
            gaze_r = _iris_offset(landmarks, RIGHT_EYE_CORNERS, RIGHT_IRIS_CENTER)
            raw_gaze = tuple(((gaze_l + gaze_r) / 2.0).tolist())

        # Confidence combines blendshape-derived face tracking quality proxy
        # (mean landmark presence is not directly exposed, so we fall back to
        # a heuristic: blink frames and extreme eye closure are unreliable
        # for gaze, everything else is scaled by detector presence score 1.0).
        confidence = 0.15 if is_blink else 0.95

        return FaceSample(t=now, raw_gaze=raw_gaze, ear=ear, is_blink=is_blink,
                           confidence=confidence, face_found=True)

    def close(self) -> None:
        self._landmarker.close()
