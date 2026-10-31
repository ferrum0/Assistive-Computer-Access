"""Raw NASA-TLX (RTLX) subjective workload questionnaire.

Six subscales, each rated 0-100 by the participant immediately after a
condition (selection mode) is completed:
  Mental Demand, Physical Demand, Temporal Demand, Performance,
  Effort, Frustration.

We use the unweighted "Raw TLX" variant (mean of the six subscales) rather
than the original pairwise-comparison-weighted TLX: RTLX is shown in the
literature to correlate very closely with weighted TLX while being much
faster to administer, which matters when running back-to-back conditions
with participants who have limited hand mobility.
"""
from __future__ import annotations

import csv
from dataclasses import asdict, dataclass, fields
from pathlib import Path

SUBSCALES = [
    "mental_demand", "physical_demand", "temporal_demand",
    "performance", "effort", "frustration",
]


@dataclass
class TLXResponse:
    participant_id: str
    mode: str
    mental_demand: int
    physical_demand: int
    temporal_demand: int
    performance: int
    effort: int
    frustration: int

    @property
    def rtlx_score(self) -> float:
        return sum(getattr(self, f) for f in SUBSCALES) / len(SUBSCALES)


def collect_tlx_cli(participant_id: str, mode: str) -> TLXResponse:
    """Prompt the participant on stdin for each subscale (0-100).

    Kept deliberately simple (CLI) so it can run headless in the same
    process as the evaluation study without extra UI dependencies; swap in
    a Tkinter slider dialog for in-person studies if preferred.
    """
    print(f"\nNASA-TLX for participant={participant_id}, mode={mode}")
    print("Rate each item from 0 (very low) to 100 (very high).")
    prompts = {
        "mental_demand": "Mental Demand (how mentally demanding was the task?)",
        "physical_demand": "Physical Demand (how physically demanding?)",
        "temporal_demand": "Temporal Demand (how hurried/rushed was the pace?)",
        "performance": "Performance (how successful were you? 0=perfect,100=failure)",
        "effort": "Effort (how hard did you have to work?)",
        "frustration": "Frustration (how insecure/irritated/stressed?)",
    }
    values = {}
    for key, prompt in prompts.items():
        while True:
            raw = input(f"  {prompt} [0-100]: ").strip()
            try:
                v = int(raw)
                if 0 <= v <= 100:
                    values[key] = v
                    break
            except ValueError:
                pass
            print("    Please enter an integer between 0 and 100.")
    return TLXResponse(participant_id=participant_id, mode=mode, **values)


class TLXLogger:
    _FIELDS = [f.name for f in fields(TLXResponse)] + ["rtlx_score"]

    def __init__(self, csv_path: Path):
        self.csv_path = csv_path
        is_new = not csv_path.exists()
        self._fh = open(csv_path, "a", newline="")
        self._writer = csv.DictWriter(self._fh, fieldnames=self._FIELDS)
        if is_new:
            self._writer.writeheader()

    def log(self, response: TLXResponse) -> None:
        row = asdict(response)
        row["rtlx_score"] = response.rtlx_score
        self._writer.writerow(row)
        self._fh.flush()

    def close(self) -> None:
        self._fh.close()
