"""Part 6 - formula lifecycle and autopsy.

    BIRTH ──► CANDIDATE ──► ACTIVE ──► DECAYING ──► DEAD ──► AUTOPSY
                 ▲                                            │
                 └──────────── resurrected with regime gates ◄┘

* BIRTH      exists but has never been scored
* CANDIDATE  scored at least once, fitness above the candidate floor
* ACTIVE     in the current top-200 (fitness ordered, correlation pruned)
* DECAYING   was active, fitness below the active floor for two consecutive
             re-evaluations (or dropped out of the top-200 twice)
* DEAD       decayed for three more re-evaluations - stops being evaluated
* AUTOPSY    the post-mortem: which regimes it lost in, whether it was
             overfit, whether a gated version deserves a second life

An autopsy that finds the formula profitable in some regimes and ruinous in
others writes those regimes into ``spec.gates`` and resurrects it as a
CANDIDATE that only fires inside the regimes it survived.  A formula that
failed everywhere is archived (kept in ``graveyard`` for the dashboard).
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from backend.genesis.fitness import Score, _spearman
from backend.genesis.spec import FormulaSpec, State

CANDIDATE_FLOOR = 0.40
ACTIVE_FLOOR = 0.45
DECAY_STRIKES = 2
DEATH_STRIKES = 3
MAX_LIVES = 2


@dataclass
class Formula:
    spec: FormulaSpec
    state: State = State.BIRTH
    score: Score = field(default_factory=Score)
    history: list = field(default_factory=list)   # (ts, fitness, state)
    strikes: int = 0
    lives: int = 0
    born_at: float = field(default_factory=time.time)
    last_eval: float = 0.0
    last_signal: float = 0.0
    last_raw: float = 0.0
    autopsy: dict = field(default_factory=dict)
    error: str = ""

    @property
    def fid(self) -> str:
        return self.spec.fid

    @property
    def fitness(self) -> float:
        return float(self.score.fitness)

    def record(self, score: Score) -> None:
        self.score = score
        self.last_eval = time.time()
        self.history.append((self.last_eval, round(score.fitness, 4), self.state.value))
        if len(self.history) > 200:
            del self.history[: len(self.history) - 200]

    def as_dict(self, verbose: bool = False) -> dict:
        d = {"id": self.fid, "name": self.spec.name, "domain": self.spec.domain,
             "domain_name": self.spec.domain_name, "subcategory": self.spec.subcategory,
             "layer": self.spec.layer, "state": self.state.value, "fitness": round(self.fitness, 4),
             "signal": round(float(self.last_signal), 4), "raw": round(float(self.last_raw), 6),
             "origin": self.spec.origin, "lives": self.lives, "strikes": self.strikes,
             "gates": self.spec.gates, "note": self.score.note or self.error}
        if verbose:
            d.update({"params": self.spec.params, "definition": self.spec.definition,
                      "interpretation": self.spec.interpretation, "expression": self.spec.expression,
                      "score": self.score.as_dict(), "history": self.history[-40:], "autopsy": self.autopsy})
        return d


def transition(f: Formula, in_active_set: bool) -> State:
    """Apply the lifecycle rules after a re-evaluation; returns the new state."""
    fit = f.fitness
    s = f.state
    if s == State.BIRTH:
        if in_active_set:
            f.state, f.strikes = State.ACTIVE, 0
        elif fit >= CANDIDATE_FLOOR:
            f.state, f.strikes = State.CANDIDATE, 0
        else:
            f.strikes += 1            # a newborn gets three looks before it dies
            if f.strikes >= DEATH_STRIKES:
                f.state = State.DEAD
    elif s == State.CANDIDATE:
        if in_active_set:
            f.state, f.strikes = State.ACTIVE, 0
        elif fit < CANDIDATE_FLOOR:
            f.strikes += 1
            if f.strikes >= DEATH_STRIKES:
                f.state = State.DEAD
        else:
            f.strikes = 0
    elif s == State.ACTIVE:
        if in_active_set and fit >= ACTIVE_FLOOR:
            f.strikes = 0
        else:
            f.strikes += 1
            if f.strikes >= DECAY_STRIKES:
                f.state, f.strikes = State.DECAYING, 0
    elif s == State.DECAYING:
        if in_active_set and fit >= ACTIVE_FLOOR:
            f.state, f.strikes = State.ACTIVE, 0
        else:
            f.strikes += 1
            if f.strikes >= DEATH_STRIKES:
                f.state = State.DEAD
    return f.state


def autopsy(f: Formula, signal: np.ndarray, ret: np.ndarray, labels: np.ndarray) -> dict:
    """Post-mortem of a DEAD formula: per-regime IC and P&L, overfit verdict,
    and the resurrection decision."""
    n = min(len(signal), len(ret), len(labels)) - 1
    report = {"at": time.time(), "verdict": "archived", "regimes": {}, "overfit": round(f.score.overfit, 4)}
    if n < 100:
        report["cause"] = "too little history to diagnose"
        f.autopsy = report
        f.state = State.AUTOPSY
        return report
    sig = np.nan_to_num(np.asarray(signal[:n], dtype=np.float64))
    nxt = np.nan_to_num(np.asarray(ret[1:n + 1], dtype=np.float64))
    lab = np.asarray(labels[:n])
    live = sig != 0
    good, bad = [], []
    for reg in np.unique(lab):
        m = live & (lab == reg)
        if m.sum() < 20:
            continue
        pnl = float(np.sum(sig[m] * nxt[m]))
        ic = _spearman(sig[m], nxt[m])
        report["regimes"][str(reg)] = {"n": int(m.sum()), "ic": round(ic, 4), "pnl": round(pnl, 6)}
        (good if (ic > 0.03 and pnl > 0) else bad).append(str(reg))
    if f.score.overfit > 0.10:
        report["cause"] = "overfit: in-sample fitness far above out-of-sample"
    elif not live.any():
        report["cause"] = "never produced a signal"
    elif bad and not good:
        report["cause"] = "lost in every regime it traded"
    elif good and bad:
        report["cause"] = "regime-specific: profitable in " + ", ".join(good) + "; ruinous in " + ", ".join(bad)
    else:
        report["cause"] = "fitness decayed below the active floor"
    if good and bad and f.lives < MAX_LIVES and f.score.overfit <= 0.10:
        f.spec.gates = {"regimes": sorted(good)}
        f.lives += 1
        f.strikes = 0
        f.state = State.CANDIDATE
        report["verdict"] = "resurrected with regime gate " + "/".join(sorted(good))
    else:
        f.state = State.AUTOPSY
    f.autopsy = report
    return report


def gated_signal(f: Formula, value: float, regime: str) -> float:
    """Apply an autopsy-derived regime gate to a live reading."""
    allowed = f.spec.gates.get("regimes")
    if allowed and regime not in allowed:
        return 0.0
    return value
