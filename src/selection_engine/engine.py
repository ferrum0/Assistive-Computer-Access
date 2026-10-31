"""Unifies smooth-pursuit and adaptive-dwell selection behind one interface.

Widgets are tagged with a `mode` ("pursuit" or "dwell") at creation time by
the UI layer (e.g. large icon buttons use pursuit; dense text/list rows too
small to host a legible orbit fall back to dwell). The engine evaluates
whichever mechanisms are in play this frame and returns at most one
selection event.
"""
from __future__ import annotations

from dataclasses import dataclass

from src.selection_engine.adaptive_dwell import AdaptiveDwellSelector, DwellState
from src.selection_engine.smooth_pursuit import PursuitResult, SmoothPursuitSelector
from src.selection_engine.targets import OrbitTarget


@dataclass
class SelectionEvent:
    widget_id: str
    mode: str          # "pursuit" | "dwell"
    detail: object      # PursuitResult or DwellState, for logging/debugging


class SelectionEngine:
    def __init__(self, pursuit: SmoothPursuitSelector, dwell: AdaptiveDwellSelector):
        self.pursuit = pursuit
        self.dwell = dwell

    def update(self, t: float, gaze_xy: tuple[float, float] | None,
               pursuit_targets: list[OrbitTarget],
               dwell_widgets: dict[str, tuple[float, float]],
               f_hat: float) -> tuple[SelectionEvent | None, PursuitResult, DwellState]:
        if gaze_xy is not None:
            self.pursuit.push_gaze(t, gaze_xy)

        pursuit_result = self.pursuit.evaluate(pursuit_targets, t)
        dwell_state = self.dwell.update(t, gaze_xy, dwell_widgets, f_hat)

        event = None
        if pursuit_result.confirmed and pursuit_result.winner is not None:
            event = SelectionEvent(pursuit_result.winner.target_id, "pursuit", pursuit_result)
        elif dwell_state.triggered:
            event = SelectionEvent(dwell_state.target_id, "dwell", dwell_state)

        return event, pursuit_result, dwell_state
