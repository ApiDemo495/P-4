"""Part 3 - fitness.  Seven metrics per formula, an overfit penalty, and the
greedy correlation-pruned selection that picks the active 200.

A formula's ``signal[t]`` is its reading at the close of candle ``t`` for the
return of candle ``t+1``.  Every metric is computed on the *out-of-sample*
tail (the last 30 % of the evaluated history) and on the in-sample head; the
gap between the two is the overfit penalty.

The seven metrics (all mapped to [0, 1] before weighting):

    hit        signed directional accuracy on the next return
    ic         Spearman rank correlation of signal with next return
    sharpe     annualised Sharpe of the signal-weighted return stream
    pf         profit factor (gross up / gross down) after 1 bp per unit turnover
    dd         1 - max drawdown of the cumulative stream, relative to gross gain
    regime     worst-regime IC over the regimes the formula actually traded
    steady     share of 50-candle blocks with positive signal-weighted return

    fitness = Σ wᵢ · mᵢ  -  overfit  -  dead-time penalty
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


COST_PER_TURN = 1e-4          # 1 bp per unit change in position
BLOCK = 50
OOS_FRACTION = 0.30
MIN_SIGNALS = 40
WEIGHTS = {"hit": 0.15, "ic": 0.25, "sharpe": 0.20, "pf": 0.10, "dd": 0.10, "regime": 0.10, "steady": 0.10}


@dataclass
class Score:
    fitness: float = -1.0
    metrics: dict = field(default_factory=dict)      # out-of-sample metrics
    insample: dict = field(default_factory=dict)
    overfit: float = 0.0
    coverage: float = 0.0        # share of candles with a non-zero signal
    n_eval: int = 0
    note: str = ""

    def as_dict(self) -> dict:
        return {"fitness": round(float(self.fitness), 4), "overfit": round(float(self.overfit), 4),
                "coverage": round(float(self.coverage), 3), "n_eval": int(self.n_eval), "note": self.note,
                "metrics": {k: round(float(v), 4) for k, v in self.metrics.items()},
                "insample": {k: round(float(v), 4) for k, v in self.insample.items()}}


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    if a.size < 8:
        return 0.0
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    ra -= ra.mean()
    rb -= rb.mean()
    d = np.sqrt((ra * ra).sum() * (rb * rb).sum())
    return float((ra * rb).sum() / d) if d > 0 else 0.0


def _metrics(sig: np.ndarray, nxt: np.ndarray, labels: np.ndarray | None) -> dict:
    """The seven metrics on one aligned (signal, next-return) slice."""
    m = {"hit": 0.5, "ic": 0.0, "sharpe": 0.0, "pf": 1.0, "dd": 0.0, "regime": 0.0, "steady": 0.5}
    live = np.isfinite(sig) & np.isfinite(nxt) & (sig != 0)
    if live.sum() < MIN_SIGNALS:
        return m
    s = np.where(np.isfinite(sig), sig, 0.0)
    r = np.where(np.isfinite(nxt), nxt, 0.0)
    turn = np.abs(np.diff(np.concatenate(([0.0], s))))
    pnl = s * r - COST_PER_TURN * turn
    m["hit"] = float(np.mean(np.sign(s[live]) == np.sign(r[live])))
    m["ic"] = _spearman(s[live], r[live])
    sd = pnl[live].std()
    m["sharpe"] = float(pnl[live].mean() / sd * np.sqrt(525_600)) if sd > 0 else 0.0
    up = pnl[pnl > 0].sum()
    dn = -pnl[pnl < 0].sum()
    m["pf"] = float(up / dn) if dn > 0 else (3.0 if up > 0 else 1.0)
    cum = np.cumsum(pnl)
    peak = np.maximum.accumulate(cum)
    mdd = float((peak - cum).max()) if cum.size else 0.0
    m["dd"] = float(1.0 - min(1.0, mdd / (up + 1e-12)))
    nb = len(pnl) // BLOCK
    if nb >= 2:
        blocks = pnl[: nb * BLOCK].reshape(nb, BLOCK).sum(axis=1)
        m["steady"] = float(np.mean(blocks > 0))
    if labels is not None and labels.size == sig.size:
        ics = []
        for lab in np.unique(labels[live]):
            mask = live & (labels == lab)
            if mask.sum() >= MIN_SIGNALS // 2:
                ics.append(_spearman(s[mask], r[mask]))
        m["regime"] = float(min(ics)) if ics else m["ic"]
    else:
        m["regime"] = m["ic"]
    return m


def _unit(m: dict) -> dict:
    """Map each metric onto [0, 1]."""
    return {"hit": np.clip((m["hit"] - 0.45) / 0.15, 0, 1),
            "ic": np.clip(m["ic"] / 0.12, -1, 1) * 0.5 + 0.5,
            "sharpe": np.clip(m["sharpe"] / 6.0, -1, 1) * 0.5 + 0.5,
            "pf": np.clip((m["pf"] - 0.9) / 0.5, 0, 1),
            "dd": np.clip(m["dd"], 0, 1),
            "regime": np.clip(m["regime"] / 0.08, -1, 1) * 0.5 + 0.5,
            "steady": np.clip((m["steady"] - 0.4) / 0.3, 0, 1)}


def score(signal: np.ndarray, ret: np.ndarray, labels: np.ndarray | None = None,
          eval_tail: int | None = None) -> Score:
    """Score ``signal`` against ``ret`` (both candle-aligned, ``ret`` is the
    candle's own log return so the target of ``signal[t]`` is ``ret[t+1]``)."""
    sig = np.asarray(signal, dtype=np.float64)
    n = min(len(sig), len(ret))
    if n < 2 * MIN_SIGNALS:
        return Score(note="too little history")
    sig = sig[:n - 1]
    nxt = np.asarray(ret[1:n], dtype=np.float64)
    lab = None if labels is None else np.asarray(labels)[:n - 1]
    if eval_tail and eval_tail < len(sig):
        sig, nxt = sig[-eval_tail:], nxt[-eval_tail:]
        lab = None if lab is None else lab[-eval_tail:]
    cut = int(len(sig) * (1 - OOS_FRACTION))
    oos = _metrics(sig[cut:], nxt[cut:], None if lab is None else lab[cut:])
    ins = _metrics(sig[:cut], nxt[:cut], None if lab is None else lab[:cut])
    live = np.isfinite(sig) & (sig != 0)
    coverage = float(live.mean()) if live.size else 0.0
    if live.sum() < MIN_SIGNALS:
        return Score(fitness=-1.0, metrics=oos, insample=ins, coverage=coverage, n_eval=int(len(sig)), note="no signal")
    u_oos, u_ins = _unit(oos), _unit(ins)
    f_oos = sum(WEIGHTS[k] * float(u_oos[k]) for k in WEIGHTS)
    f_ins = sum(WEIGHTS[k] * float(u_ins[k]) for k in WEIGHTS)
    overfit = max(0.0, f_ins - f_oos)
    dead = 0.15 * max(0.0, 0.25 - coverage) / 0.25
    fitness = 0.6 * f_oos + 0.4 * f_ins - 1.5 * overfit - dead
    note = "healthy" if oos["ic"] > 0.02 and oos["pf"] > 1.0 else ("overfit" if overfit > 0.08 else "weak")
    return Score(fitness=float(fitness), metrics=oos, insample=ins, overfit=overfit, coverage=coverage,
                 n_eval=int(len(sig)), note=note)


def select(candidates: list, signals: dict, k: int = 200, max_corr: float = 0.70, tail: int = 500) -> list:
    """Greedy correlation-pruned selection.  ``candidates`` are formulas
    ordered by fitness (best first); ``signals[fid]`` their signal arrays.
    A formula joins the active set only if its recent signal correlates below
    ``max_corr`` with every formula already chosen."""
    chosen: list = []
    kept: list[np.ndarray] = []
    for f in candidates:
        if len(chosen) >= k:
            break
        s = signals.get(f.fid)
        if s is None:
            continue
        v = np.nan_to_num(np.asarray(s[-tail:], dtype=np.float64))
        v = v - v.mean()
        nv = np.linalg.norm(v)
        if nv == 0:
            continue
        v = v / nv
        if kept:
            m = min(len(v), min(len(w) for w in kept))
            corr = np.array([abs(float(np.dot(v[-m:], w[-m:]))) for w in kept])
            if corr.max() > max_corr:
                continue
        chosen.append(f)
        kept.append(v)
    return chosen


def correlation_matrix(signals: dict, fids: list, tail: int = 500) -> np.ndarray:
    rows = []
    for fid in fids:
        v = np.nan_to_num(np.asarray(signals[fid][-tail:], dtype=np.float64))
        v = v - v.mean()
        nv = np.linalg.norm(v)
        rows.append(v / nv if nv else v)
    m = min(len(r) for r in rows) if rows else 0
    if not m:
        return np.zeros((0, 0))
    A = np.array([r[-m:] for r in rows])
    return A @ A.T


def rank_fitness(metrics: dict) -> float:
    """Convenience for tests / UI: fitness of already-computed unit metrics."""
    u = _unit(metrics)
    return float(sum(WEIGHTS[k] * float(u[k]) for k in WEIGHTS))


__all__ = ["Score", "score", "select", "correlation_matrix", "rank_fitness", "WEIGHTS"]
