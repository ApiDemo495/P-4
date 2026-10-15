"""Round BC - Superposition: every formula, every physics mechanism and every
emotion is carried as a state of the NEXT window, not a reading of this one.

The user's instruction: "change each formula, each physics, everything, so
they are already in the future predicting their next future - basically the
superposition concept".  Implemented literally, within what can be measured:

* A source's *measured* value ``v`` (what the tape says now) is first
  **projected** to where the source itself will be at the next boundary,
  ``v̂ = v + φ·(v − v_prev)`` with ``φ`` fitted by least squares on the
  source's own recent history (damped, |φ| ≤ 0.9).  A formula that is rising
  is read where it is going, not where it is.
* Both the measured and the projected reading are looked up in the forward
  tables (:mod:`backend.core.forward`): the measured frequency of what the
  NEXT window did after readings like these, conditioned on the last
  window's direction.
* The source's state is then a **superposition** of the three settlement
  outcomes - |up⟩, |down⟩, |tie⟩ - with weights that blend the two branches
  by their sample sizes.  The *collapsed value* is ``P(up) − P(down)``; the
  *coherence* is how far the state is from a uniform mixture
  (1 − H/ln 3).  A coherent state is a source that, for this reading, has
  a sharp forward record; an incoherent one is a coin.

The collapsed value is what the rest of the engine consumes wherever it
used to consume the reading: the formula consensus, the ledger votes, the
brain's projection neurons, the physics vote and the emotion layer's
intensities.  The measured value is kept, labelled "measured now", so the
logic stays visible.  A source with no forward record collapses to its
measured value with coherence 0 - it is never invented.
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass

import numpy as np

from backend.core.forward import MIN_N_POOLED, ForwardExpectancy

HISTORY = 24          # own-history points used for the projection
PHI_MAX = 0.9


@dataclass
class FutureState:
    name: str
    measured: float
    projected: float
    phi: float
    p_up: float
    p_down: float
    p_tie: float
    n: float
    collapsed: float
    coherence: float
    context: str
    covered: bool

    def as_dict(self) -> dict:
        return {
            "name": self.name, "measured": round(self.measured, 4), "projected": round(self.projected, 4),
            "phi": round(self.phi, 3), "p_up": round(self.p_up, 4), "p_down": round(self.p_down, 4),
            "p_tie": round(self.p_tie, 4), "n": round(self.n, 1), "collapsed": round(self.collapsed, 4),
            "coherence": round(self.coherence, 4), "context": self.context, "covered": self.covered,
            "reads": (f"measured {self.measured:+.3f} → projected {self.projected:+.3f} at the boundary · "
                      f"next window |up⟩ {self.p_up:.0%} |down⟩ {self.p_down:.0%} |tie⟩ {self.p_tie:.0%} "
                      f"(n {self.n:.0f}, {self.context}) → collapses to {self.collapsed:+.3f}, coherence {self.coherence:.2f}"
                      if self.covered else
                      f"measured {self.measured:+.3f} → projected {self.projected:+.3f} · no forward record for this reading yet"),
        }


def project(history: list[float] | deque, value: float) -> tuple[float, float]:
    """Damped AR(1) projection of a source's own next value from its history."""
    h = np.asarray(list(history)[-HISTORY:], dtype=float)
    h = h[np.isfinite(h)]
    if h.size < 6:
        return float(value), 0.0
    d = np.diff(h)
    x, y = d[:-1], d[1:]
    denom = float(np.dot(x, x))
    phi = float(np.dot(x, y) / denom) if denom > 1e-12 else 0.0
    phi = max(-PHI_MAX, min(PHI_MAX, phi))
    step = float(value) - float(h[-1]) if abs(float(value) - float(h[-1])) > 0 else float(d[-1])
    return float(value) + phi * step, phi


class Superposition:
    """Builds the next-window state of every source from the forward tables."""

    def __init__(self, forward: ForwardExpectancy) -> None:
        self.forward = forward
        self.history: dict[str, deque] = {}
        self.last: dict[str, FutureState] = {}

    def remember(self, values: dict) -> None:
        for name, v in (values or {}).items():
            try:
                fv = float(v)
            except (TypeError, ValueError):
                continue
            if math.isfinite(fv):
                self.history.setdefault(name, deque(maxlen=HISTORY)).append(fv)

    def state(self, name: str, value: float, last_dir: int) -> FutureState:
        projected, phi = project(self.history.get(name, ()), value)
        now = self.forward.expect_one(name, float(value), last_dir)
        nxt = self.forward.expect_one(name, projected, last_dir)
        w_now, w_nxt = now.n, nxt.n
        total_n = w_now + w_nxt
        covered = max(now.n, nxt.n) >= MIN_N_POOLED
        if covered and total_n > 0:
            p_up = (now.p_up * w_now + nxt.p_up * w_nxt) / total_n
            p_tie = (now.p_tie * w_now + nxt.p_tie * w_nxt) / total_n
            p_up = p_up * (1.0 - p_tie)
            p_down = max(0.0, 1.0 - p_up - p_tie)
            n = total_n / 2.0
            ps = np.asarray([p_up, p_down, p_tie], dtype=float)
            ps = ps[ps > 1e-12]
            coherence = float(1.0 - (-(ps * np.log(ps)).sum()) / math.log(3.0))
            collapsed = float(p_up - p_down)
            context = now.context if now.n >= nxt.n else nxt.context
        else:
            p_up = p_down = 0.5
            p_tie, n, coherence, collapsed, context = 0.0, max(now.n, nxt.n), 0.0, float(value), "no record"
        st = FutureState(name, float(value), projected, phi, p_up, p_down, p_tie, n, collapsed, coherence, context, covered)
        self.last[name] = st
        return st

    def states(self, values: dict, last_dir: int) -> dict[str, FutureState]:
        out = {}
        for name, v in (values or {}).items():
            if not ForwardExpectancy._usable(name, v):
                continue
            out[name] = self.state(name, float(v), last_dir)
        return out

    def collapse(self, values: dict, last_dir: int) -> dict[str, float]:
        """name -> the value the engine acts on: collapsed when covered, else measured."""
        out = dict(values or {})
        for name, st in self.states(values, last_dir).items():
            if st.covered:
                out[name] = st.collapsed
        return out

    def summary(self) -> dict:
        covered = [s for s in self.last.values() if s.covered]
        against = [s.name for s in covered if s.measured != 0 and (s.collapsed > 0) != (s.measured > 0) and abs(s.collapsed) > 0.04]
        return {
            "sources": len(self.last), "covered": len(covered),
            "mean_coherence": round(float(np.mean([s.coherence for s in covered])), 4) if covered else 0.0,
            "against_their_reading": against,
            "reads": ("every source is carried as |up⟩/|down⟩/|tie⟩ for the NEXT window, from its measured and its "
                      "projected reading; the engine acts on P(up) − P(down), never on the reading itself"),
        }
