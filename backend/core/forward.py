"""Round BB - Forward Expectancy: every number the engine acts on is a
statement about the NEXT window, measured, never an extrapolation of the
present.

The user's diagnosis, verbatim: "everything in the app calculates on the
basis of present and thinks the future would be like present".  He is right
about the recipe's grammar.  A formula reading of +0.42 was treated as
"+0.42 bullish for the next window" - which silently assumes the next 53 s
look like the last 53 s.  Nobody can observe the future, but it *can* be
measured in the only honest way: by conditioning on it.  For every formula
and every value it has taken, this layer keeps what the price did in the
window that FOLLOWED - the forward conditional

    P(next window up | formula f is in bin b, last window went d)

decayed so it follows the regime, over the real tape (the warm start's
hours of Binance history, then every live window).  A formula's *forward
value* is then 2·P(up) − 1 - a number that can and does come out with the
opposite sign to the reading itself ("when TAI is this bullish after an up
window, the next window went DOWN 56 % of the time").  That is what
"calculating from the future" means in practice: the input to the decision
is the measured distribution of what came next, not the present reading.

Where it is used (all three stages the user named):

* **the formulas' vote** - ``composite()`` combines the forward values of
  every covered formula (naive-Bayes log-odds, shrunk by sample size,
  divided by sqrt(k) for correlation) into P(up) for the next window; it is
  a weighted voter in the fusion (``weight_forward``) and a source in the
  evidence ledger, so it is itself scored on the future;
* **the brain** - when the layer has coverage, the projection-neuron vector
  injected into the real mushroom body is the forward values, so the fly
  smells "what followed readings like these", not the readings;
* **the UI** - every formula row prints its forward line: "now +0.42 →
  next window up 44 % (n 212, after an up window)".

Honesty: with few samples a bin says 50 % (Beta prior), and ``covered``
counts only formulas with a real record; the composite abstains below five
covered formulas.  Ties are counted and reported, excluded from P(up).
"""
from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import dataclass

import numpy as np

#: value bins - quantile edges once a formula has a record, fixed edges before
N_BINS = 7
FIXED_EDGES = (-0.6, -0.3, -0.1, 0.1, 0.3, 0.6)
PRIOR = 6.0               # Beta(3, 3): six pseudo-observations at 50 %
MIN_N_CONTEXT = 12.0      # decayed samples before the (bin, last-window) cell is trusted
MIN_N_POOLED = 8.0        # ... before the pooled bin is trusted at all
HALF_LIFE = 400.0         # windows (~6 h): follows the regime, remembers the day
MIN_COVERED = 5           # formulas with a record before the composite votes
RESERVOIR = 3000
IGNORED = frozenset({"DRG", "KCAE", "CCSv2", "HSI", "ERC", "PIN", "_hsi"})   # non-directional / outputs


@dataclass
class Cell:
    up: float = 0.0
    down: float = 0.0
    tie: float = 0.0
    move: float = 0.0        # decayed sum of forward move (bps)

    @property
    def n(self) -> float:
        return self.up + self.down

    def p_up(self) -> float:
        return (self.up + PRIOR / 2.0) / (self.n + PRIOR)

    def decay(self, k: float) -> None:
        self.up *= k
        self.down *= k
        self.tie *= k
        self.move *= k


@dataclass
class Expectancy:
    name: str
    value: float
    bin: int
    p_up: float
    n: float
    p_tie: float
    exp_move_bps: float
    context: str          # "after up" / "after down" / "pooled"
    edge: float           # 2 p_up - 1

    def as_dict(self) -> dict:
        return {"name": self.name, "value": round(self.value, 4), "bin": self.bin, "p_up": round(self.p_up, 4),
                "n": round(self.n, 1), "p_tie": round(self.p_tie, 4), "exp_move_bps": round(self.exp_move_bps, 2),
                "context": self.context, "edge": round(self.edge, 4),
                "reads": (f"now {self.value:+.3f} → next window up {self.p_up:.0%} (n {self.n:.0f}, {self.context})"
                          if self.n >= MIN_N_POOLED else f"now {self.value:+.3f} → no forward record yet (n {self.n:.0f})")}


class ForwardExpectancy:
    """Per-formula forward conditional tables, learned on settled windows."""

    def __init__(self, half_life: float = HALF_LIFE) -> None:
        self.decay = 0.5 ** (1.0 / max(1.0, float(half_life)))
        self.cells: dict[tuple[str, int, int], Cell] = {}
        self.reservoir: dict[str, deque] = {}
        self.pending: dict[int, tuple[dict[str, tuple[float, int]], int]] = {}
        self.settled = 0
        self.ties = 0
        self.updated_at = 0.0
        self.last_composite: dict = {}

    # -------------------------------------------------------------- binning
    def edges(self, name: str) -> np.ndarray:
        res = self.reservoir.get(name)
        if res is not None and len(res) >= 60:
            arr = np.fromiter(res, dtype=float, count=len(res))
            q = np.quantile(arr, np.linspace(0, 1, N_BINS + 1)[1:-1])
            if np.all(np.diff(q) > 1e-12):
                return q
        return np.asarray(FIXED_EDGES, dtype=float)

    def bin_of(self, name: str, value: float) -> int:
        return int(np.searchsorted(self.edges(name), float(value), side="right"))

    @staticmethod
    def _usable(name: str, value) -> bool:
        if name in IGNORED or name.startswith("_"):
            return False
        try:
            v = float(value)
        except (TypeError, ValueError):
            return False
        return math.isfinite(v)

    # ------------------------------------------------------------- learning
    def observe(self, cycle: int, values: dict, last_dir: int) -> None:
        """Remember the bins of this window's readings; settle later."""
        bins: dict[str, tuple[float, int]] = {}
        for name, value in (values or {}).items():
            if not self._usable(name, value):
                continue
            v = float(value)
            bins[name] = (v, self.bin_of(name, v))
            self.reservoir.setdefault(name, deque(maxlen=RESERVOIR)).append(v)
        self.pending[int(cycle)] = (bins, int(np.sign(last_dir)))
        if len(self.pending) > 64:
            for old in sorted(self.pending)[:-40]:
                self.pending.pop(old, None)

    def settle(self, cycle: int, went_up: bool | None, move_bps: float = 0.0) -> bool:
        row = self.pending.pop(int(cycle), None)
        if row is None:
            return False
        bins, last_dir = row
        for cell in self.cells.values():
            cell.decay(self.decay)
        for name, (_v, b) in bins.items():
            for ctx in (last_dir, 0):            # the context cell and the pooled cell
                cell = self.cells.setdefault((name, b, ctx), Cell())
                if went_up is None:
                    cell.tie += 1.0
                elif went_up:
                    cell.up += 1.0
                else:
                    cell.down += 1.0
                cell.move += float(move_bps or 0.0)
        self.settled += 1
        if went_up is None:
            self.ties += 1
        self.updated_at = time.time()
        return True

    # -------------------------------------------------------------- reading
    def expect_one(self, name: str, value: float, last_dir: int) -> Expectancy:
        b = self.bin_of(name, value)
        ctx_cell = self.cells.get((name, b, int(np.sign(last_dir)))) if last_dir else None
        pooled = self.cells.get((name, b, 0))
        if ctx_cell is not None and ctx_cell.n >= MIN_N_CONTEXT:
            cell, context = ctx_cell, ("after an up window" if last_dir > 0 else "after a down window")
        elif pooled is not None:
            cell, context = pooled, "pooled over last-window direction"
        else:
            cell, context = Cell(), "no record"
        total = cell.n + cell.tie
        p_tie = cell.tie / total if total > 0 else 0.0
        exp_move = cell.move / total if total > 0 else 0.0
        p = cell.p_up()
        return Expectancy(name, float(value), b, p, cell.n, p_tie, exp_move, context, 2.0 * p - 1.0)

    def expect(self, values: dict, last_dir: int) -> dict[str, Expectancy]:
        return {name: self.expect_one(name, float(v), last_dir)
                for name, v in (values or {}).items() if self._usable(name, v)}

    def edges_vector(self, values: dict, last_dir: int) -> dict[str, float]:
        """name -> forward value (2 P(up) - 1) for every formula with a record;
        the brain's projection neurons are fed these instead of the readings."""
        out = {}
        for name, e in self.expect(values, last_dir).items():
            if e.n >= MIN_N_POOLED:
                out[name] = e.edge
        return out

    def composite(self, values: dict, last_dir: int) -> dict:
        """P(next window up) from every covered formula's forward conditional."""
        ex = self.expect(values, last_dir)
        rows = []
        total = 0.0
        for e in ex.values():
            if e.n < MIN_N_POOLED:
                continue
            shrink = e.n / (e.n + PRIOR)                  # trust grows with the record
            lo = math.log(e.p_up / (1.0 - e.p_up)) * shrink
            total += lo
            rows.append((e, lo))
        k = len(rows)
        covered = k
        if k:
            z = total / math.sqrt(k)                       # correlated witnesses
            p_up = 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, z))))
        else:
            z, p_up = 0.0, 0.5
        rows.sort(key=lambda r: -abs(r[1]))
        side = "BUY" if p_up >= 0.5 else "SELL"
        p_tie = float(np.mean([e.p_tie for e, _ in rows])) if rows else 0.0
        exp_move = float(np.mean([e.exp_move_bps for e, _ in rows])) if rows else 0.0
        # how many covered formulas point the OTHER way than their own reading
        # - the measured size of "right about the past, wrong about the future"
        inverted = sum(1 for e, _ in rows if e.value != 0 and (e.edge > 0) != (e.value > 0) and abs(e.edge) > 0.04)
        out = {
            "available": covered >= MIN_COVERED,
            "side": side if covered >= MIN_COVERED else None,
            "p_up": round(p_up, 4),
            "p_side": round(max(p_up, 1.0 - p_up), 4),
            "edge": round(2.0 * max(p_up, 1.0 - p_up) - 1.0, 4),
            "value": round(2.0 * p_up - 1.0, 4),
            "log_odds": round(z, 4),
            "covered": covered,
            "total": len(ex),
            "inverted_vs_reading": inverted,
            "p_tie": round(p_tie, 4),
            "exp_move_bps": round(exp_move, 2),
            "last_window": "up" if last_dir > 0 else "down" if last_dir < 0 else "unknown",
            "settled": self.settled,
            "drivers": [e.as_dict() for e, _ in rows[:6]],
            "note": (f"P(next window up) {p_up:.0%} from {covered} formulas' forward records "
                     f"({inverted} of them point against their own present reading) · "
                     f"last window {('up' if last_dir > 0 else 'down' if last_dir < 0 else 'unknown')} · "
                     f"{self.settled} settled windows in the tables"
                     if covered >= MIN_COVERED else
                     f"forward tables cover {covered}/{len(ex)} formulas - the composite abstains until {MIN_COVERED} have a record"),
            "reads": "p_up = sigma( sum_f shrink(n_f) * logit P(up | f in its bin, last window dir) / sqrt(k) ); every term is the measured frequency of what FOLLOWED readings like this one",
        }
        self.last_composite = out
        return out

    def to_dict(self) -> dict:
        return {"settled": self.settled, "ties": self.ties, "cells": len(self.cells), "formulas": len(self.reservoir),
                "half_life_windows": round(math.log(0.5) / math.log(self.decay), 0), "updated_at": self.updated_at,
                "pending": len(self.pending), "last_composite": self.last_composite}

    def table(self, name: str) -> dict:
        """The full forward table of one formula, for the explorer."""
        edges = self.edges(name).tolist()
        rows = []
        for b in range(N_BINS):
            for ctx, label in ((1, "after up"), (-1, "after down"), (0, "pooled")):
                cell = self.cells.get((name, b, ctx))
                if cell is None:
                    continue
                total = cell.n + cell.tie
                rows.append({"bin": b, "context": label, "n": round(cell.n, 1), "p_up": round(cell.p_up(), 3),
                             "p_tie": round(cell.tie / total, 3) if total else 0.0,
                             "exp_move_bps": round(cell.move / total, 2) if total else 0.0})
        return {"name": name, "edges": [round(e, 4) for e in edges], "rows": rows}
