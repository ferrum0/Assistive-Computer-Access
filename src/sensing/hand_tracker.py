"""Secondary input channel: hand landmarks via MediaPipe HandLandmarker.

Exposes an index-fingertip pointer position (screen-normalized) and a simple
pinch-gesture detector (thumb tip to index tip distance) that acts as a
discrete "click" event for users who retain some hand/finger mobility but not
enough for precise mouse control.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

WRIST = 0
THUMB_TIP = 4
INDEX_TIP = 8


@dataclass
class HandSample:
    t: float
    pointer: tuple[float, float] | None   # normalized (x, y) in [0, 1], image space
    is_pinch: bool
    confidence: float                     # 0..1, from handedness score
    hand_found: bool


class HandTracker:
    """Wraps mediapipe.tasks.vision.HandLandmarker in VIDEO mode."""

    def __init__(self, model_path: str, pinch_threshold: float = 0.06,
                 min_hand_confidence: float = 0.5):
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision as mp_vision

        # CPU delegate: see the matching note in face_tracker.py -- the GPU/
        # Metal delegate crashes the process outside an interactive GUI
        # session instead of raising a catchable exception.
        base_options = mp_python.BaseOptions(
            model_asset_path=model_path,
            delegate=mp_python.BaseOptions.Delegate.CPU,
        )
        options = mp_vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=mp_vision.RunningMode.VIDEO,
            num_hands=1,
            min_hand_detection_confidence=min_hand_confidence,
            min_hand_presence_confidence=min_hand_confidence,
            min_tracking_confidence=min_hand_confidence,
        )
        self._landmarker = mp_vision.HandLandmarker.create_from_options(options)
        self._pinch_threshold = pinch_threshold
        self._mp_image_cls = mp.Image
        self._mp_image_format = mp.ImageFormat.SRGB

    def process(self, bgr_frame: np.ndarray, timestamp_ms: int) -> HandSample:
        import cv2
        rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        mp_image = self._mp_image_cls(image_format=self._mp_image_format, data=rgb)
        result = self._landmarker.detect_for_video(mp_image, timestamp_ms)

        now = time.time()
        if not result.hand_landmarks:
            return HandSample(t=now, pointer=None, is_pinch=False,
                               confidence=0.0, hand_found=False)

        landmarks = result.hand_landmarks[0]
        index_tip = landmarks[INDEX_TIP]
        thumb_tip = landmarks[THUMB_TIP]
        pointer = (index_tip.x, index_tip.y)

        pinch_dist = float(np.hypot(index_tip.x - thumb_tip.x,
                                     index_tip.y - thumb_tip.y))
        is_pinch = pinch_dist < self._pinch_threshold

        confidence = 0.0
        if result.handedness and result.handedness[0]:
            confidence = float(result.handedness[0][0].score)

        return HandSample(t=now, pointer=pointer, is_pinch=is_pinch,
                           confidence=confidence, hand_found=True)

    def close(self) -> None:
        self._landmarker.close()
