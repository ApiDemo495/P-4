"""FORMULA 26 - Run-Age Reversal (RAR).

A monotone run on the tape has a lifetime.  RAR measures how old the current
run is against the tape's own distribution of run lengths (survival
function): a run already past its median age is more likely to end inside
the next window than to continue, so RAR leans *against* it; a young run
leans *with* it.  value = direction × (1 − 2·F(age)), F the empirical CDF of
run lengths over the tape - a hazard reading at the window horizon.

Different from MPS (variance-ratio persistence) and MCPE (cycle phase): RAR
is about *this* run's age, not the tape's average persistence.
"""

from __future__ import annotations

import numpy as np

from backend.formulas._util import finite, trace
from backend.formulas.category_i_window import _window as W

NAME = "RAR"
CATEGORY = "I"
TITLE = "Run-Age Reversal"
BRAIN_NODE = ""
DIRECTIONAL = True
LATENCY_MS = 0.3
DESCRIPTION = "Hazard of the current price run ending inside the window, from the tape's own run lengths."


class State:
    __slots__ = ("last",)

    def __init__(self) -> None:
        self.last = 0.0

    def to_dict(self) -> dict:
        return {"last": self.last}

    @classmethod
    def from_dict(cls, payload: dict) -> "State":
        obj = cls()
        obj.last = float(payload.get("last", 0.0))
        return obj


def compute(snapshot, asset: str, state: State, params: dict, ctx: dict | None = None) -> float:
    tr = W.trades_only(snapshot.ticks(asset))
    if tr.shape[0] < 40:
        return 0.0
    all_runs = W.runs(tr[:, 1])
    if len(all_runs) < 8:
        trace(ctx, "note", len(all_runs), "fewer than 8 runs - RAR abstains")
        return 0.0
    direction, age = all_runs[-1]
    lengths = np.array([n for _, n in all_runs[:-1]], dtype=np.float64)
    cdf = float(np.mean(lengths <= age))
    value = direction * (1.0 - 2.0 * cdf)
    state.last = value
    trace(ctx, "runs on the tape", len(all_runs), "monotone price runs")
    trace(ctx, "current run", direction, f"direction, age {age} moves")
    trace(ctx, "median run", float(np.median(lengths)), "moves")
    trace(ctx, "age percentile", cdf, "F(age) over past runs")
    trace(ctx, "hazard lean", value, "direction × (1 − 2·F(age))")
    return finite(value)


DOUBLE_CHECK = "value = run direction × (1 − 2 × age percentile)"


def double_check(t: dict, asset: str) -> float:
    if "current run" not in t or "age percentile" not in t:
        return 0.0
    return float(t["current run"]) * (1.0 - 2.0 * float(t["age percentile"]))
