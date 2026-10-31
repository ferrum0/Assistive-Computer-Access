"""Modality arbiter: decides, frame by frame, whether gaze, hand, both, or
neither channel should drive the cursor/selection, so the interface degrades
gracefully (slower, more conservative) rather than failing outright when
lighting is poor or the user moves out of the ideal tracking envelope.

Inputs per frame:
  - c_gaze_raw: FaceTracker confidence (0 = no face / blink / lost tracking)
  - c_hand_raw: HandTracker confidence (handedness score, 0 = no hand)
  - frame_brightness: mean pixel intensity, used as a lighting-quality signal
    that discounts gaze confidence independently of MediaPipe's own score
    (MediaPipe can report a confident landmark fit on a poorly-lit frame
    right up until it suddenly loses tracking; brightness is an earlier,
    smoother warning signal).

Both confidences are exponentially smoothed (EMA) before thresholding, to
avoid flickering between modes on single noisy frames.

Decision:
  gaze_ok = c_gaze_smooth >= gaze_threshold
  hand_ok = c_hand_smooth >= hand_threshold

  gaze_ok & hand_ok   -> FUSED   (gaze for pointing, hand pinch for click:
                                   the two channels are complementary, not
                                   redundant, so both stay active)
  gaze_ok & !hand_ok  -> GAZE_ONLY
  !gaze_ok & hand_ok  -> HAND_ONLY
  !gaze_ok & !hand_ok -> DEGRADED (freeze cursor, widen dwell threshold to
                                   max, surface a "low confidence" indicator
                                   instead of guessing)
"""
from __future__ import annotations

import enum
from dataclasses import dataclass

from src.utils.buffers import EMA


class Modality(enum.Enum):
    FUSED = "fused"
    GAZE_ONLY = "gaze_only"
    HAND_ONLY = "hand_only"
    DEGRADED = "degraded"


@dataclass
class ArbiterDecision:
    modality: Modality
    gaze_confidence: float
    hand_confidence: float
    lighting_ok: bool


class ModalityArbiter:
    def __init__(self, gaze_threshold: float, hand_threshold: float,
                 ema_alpha: float, low_light_brightness_threshold: float):
        self.gaze_threshold = gaze_threshold
        self.hand_threshold = hand_threshold
        self.low_light_threshold = low_light_brightness_threshold

        # No `initial=` -> the first real sample sets the EMA directly instead
        # of ramping up from 0, so a confident signal is recognized on the
        # first frame rather than needing several frames to climb out of a
        # spurious DEGRADED start state.
        self._gaze_ema = EMA(ema_alpha)
        self._hand_ema = EMA(ema_alpha)

    def update(self, c_gaze_raw: float, c_hand_raw: float,
               frame_brightness: float) -> ArbiterDecision:
        lighting_ok = frame_brightness >= self.low_light_threshold
        # Poor lighting discounts the raw gaze confidence before smoothing,
        # since low-light frames make iris localisation unreliable even when
        # the face detector itself still reports a confident box.
        lighting_factor = 1.0 if lighting_ok else max(
            frame_brightness / self.low_light_threshold, 0.0)
        c_gaze_adj = c_gaze_raw * lighting_factor

        c_gaze = self._gaze_ema.update(c_gaze_adj)
        c_hand = self._hand_ema.update(c_hand_raw)

        gaze_ok = c_gaze >= self.gaze_threshold
        hand_ok = c_hand >= self.hand_threshold

        if gaze_ok and hand_ok:
            modality = Modality.FUSED
        elif gaze_ok:
            modality = Modality.GAZE_ONLY
        elif hand_ok:
            modality = Modality.HAND_ONLY
        else:
            modality = Modality.DEGRADED

        return ArbiterDecision(modality=modality, gaze_confidence=c_gaze,
                                hand_confidence=c_hand, lighting_ok=lighting_ok)
