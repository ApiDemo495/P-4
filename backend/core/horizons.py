"""Round BD - four horizons at once: 55 s, 60 s, 70 s and 120 s.

Every window the engine locks ONE contest pick (the 70 s window, the only
one the contest settles).  This module issues, at the same boundary and from
the same frozen tape, a pick for each of the four horizons, so the dashboard
shows "if the window were 55 / 60 / 70 / 120 s, this is the side" - and then
settles every one of them against the last trade ``h`` seconds later, so each
horizon carries its own measured strict-sign win rate and streak.

The per-horizon mathematics is the Category I window maths re-run at the
horizon (Round AX), on the trade tape frozen at the boundary:

* ``drift``       - WDR at h: t-statistic of the recency-weighted mean return
                    of overlapping h-second windows (stride 5 s), tanh(t/2)
* ``persistence`` - PRT at h: lag-one autocorrelation of non-overlapping
                    h-second returns, applied to the sign of the last one
* ``p_tie``       - PIN at h: the larger of the share of h-windows that closed
                    at their open and 2·Φ(tick / 2σ_h) − 1
* ``sigma_bps``   - the h-second return dispersion

and the horizon score blends the locked contest pick with them::

    score_h = 0.50 · lock + 0.35 · drift_h + 0.15 · persistence_h

For the contest horizon (70 s) the pick IS the lock - the module never
disagrees with the Signal Lock.  P(win) is reported honestly as
P(side) · (1 − P(tie)) with P(side) = ½ + 0.45·|score|.
"""

from __future__ import annotations

from backend.core.numfmt import fx, fxs

import math
import time
from dataclasses import dataclass, field

import numpy as np

from backend.formulas.category_i_window import _window as W

HORIZONS: tuple[int, ...] = (55, 60, 70, 120)
HISTORY = 240
W_LOCK, W_DRIFT, W_PERSIST = 0.50, 0.35, 0.15


def _phi(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def horizon_stats(ticks: np.ndarray, horizon_s: float) -> dict:
    """The window maths at one horizon on one frozen tape (never raises)."""
    out = {"horizon": int(horizon_s), "n": 0, "drift": 0.0, "t": 0.0, "persistence": 0.0, "rho": 0.0,
           "p_tie": 0.0, "sigma_bps": 0.0, "last_bps": 0.0, "covered": False}
    try:
        opens, closes = W.horizon_returns(ticks, horizon_s)
        if opens.size < 6:
            return out
        r = (closes / opens - 1.0) * 1e4
        n = int(r.size)
        per_window = horizon_s / W.STRIDE_S
        # recency weights, half-life two horizons
        lam = math.log(2.0) / max(1.0, 2.0 * per_window)
        w = np.exp(-lam * (n - 1 - np.arange(n)))
        w /= w.sum()
        mean = float(np.dot(w, r))
        var = float(np.dot(w, (r - mean) ** 2))
        sd = math.sqrt(max(var, 1e-12))
        n_ind = max(2.0, n / per_window)                  # independent windows
        t = mean / (sd / math.sqrt(n_ind))
        drift = math.tanh(t / 2.0)
        # non-overlapping lag-one autocorrelation
        step = max(1, int(round(per_window)))
        nr = r[::-1][::step][::-1]
        rho = 0.0
        if nr.size >= 6:
            a, b = nr[:-1] - nr.mean(), nr[1:] - nr.mean()
            den = float(np.sqrt(np.dot(a, a) * np.dot(b, b)))
            rho = float(np.dot(a, b) / den) if den > 0 else 0.0
        last = float(nr[-1]) if nr.size else float(r[-1])
        persistence = float(np.clip(rho, -1.0, 1.0)) * (1.0 if last > 0 else -1.0 if last < 0 else 0.0)
        # tie risk at this horizon
        prices = W.trades_only(ticks)[:, 1]
        tick = W.tick_size(prices)
        px = float(prices[-1]) if prices.size else 0.0
        tick_bps = (tick / px) * 1e4 if px > 0 else 0.0
        p_flat = float(np.mean(r == 0.0))
        p_tie = max(p_flat, 2.0 * _phi(tick_bps / (2.0 * sd)) - 1.0 if sd > 0 else 0.0)
        out.update({"n": n, "drift": round(drift, 4), "t": round(t, 3), "persistence": round(persistence, 4),
                    "rho": round(rho, 4), "p_tie": round(float(np.clip(p_tie, 0.0, 1.0)), 4),
                    "sigma_bps": round(sd, 2), "last_bps": round(last, 2), "covered": True})
    except Exception:  # noqa: BLE001 - a horizon that cannot be measured abstains
        return out
    return out


@dataclass
class HorizonPick:
    cycle: int
    horizon: int
    side: str
    score: float
    confidence: float
    p_win: float
    p_tie: float
    entry: float
    issued_at: float
    due_at: float
    stats: dict
    logic: list[str]
    exit: float | None = None
    won: bool | None = None
    tie: bool = False
    settled_at: float | None = None

    def to_dict(self) -> dict:
        return {
            "cycle": self.cycle, "horizon": self.horizon, "side": self.side, "score": round(self.score, 4),
            "confidence": round(self.confidence, 4), "p_win": round(self.p_win, 4), "p_tie": round(self.p_tie, 4),
            "entry": self.entry, "issued_at": self.issued_at, "due_at": self.due_at,
            "exit": self.exit, "won": self.won, "tie": self.tie, "settled_at": self.settled_at,
            "stats": self.stats, "logic": list(self.logic),
        }


@dataclass
class HorizonLane:
    horizon: int
    picks: list[HorizonPick] = field(default_factory=list)
    wins: int = 0
    losses: int = 0
    ties: int = 0
    streak: int = 0
    best_streak: int = 0

    @property
    def settled(self) -> int:
        return self.wins + self.losses

    @property
    def win_rate(self) -> float | None:
        return self.wins / self.settled if self.settled else None

    def to_dict(self) -> dict:
        current = self.picks[-1] if self.picks else None
        return {
            "horizon": self.horizon, "wins": self.wins, "losses": self.losses, "ties": self.ties,
            "settled": self.settled, "win_rate": None if self.win_rate is None else round(self.win_rate, 4),
            "streak": self.streak, "best_streak": self.best_streak,
            "current": current.to_dict() if current else None,
            "recent": [p.to_dict() for p in self.picks[-12:]],
        }


class HorizonBook:
    """Issues the four picks at the boundary and keeps each lane's record."""

    def __init__(self, horizons: tuple[int, ...] = HORIZONS, contest_horizon: int = 70) -> None:
        self.horizons = tuple(int(h) for h in horizons)
        self.contest_horizon = int(contest_horizon)
        self.lanes: dict[int, HorizonLane] = {h: HorizonLane(h) for h in self.horizons}
        self.last_issue: dict = {}

    # ------------------------------------------------------------------
    def issue(self, cycle: int, entry: float, lock_side: str, lock_confidence: float,
              ticks: np.ndarray, now: float | None = None) -> list[HorizonPick]:
        now = float(now if now is not None else time.time())
        lock_sign = 1.0 if str(lock_side).upper() == "BUY" else -1.0
        lock = lock_sign * float(np.clip(lock_confidence, 0.0, 1.0))
        picks: list[HorizonPick] = []
        for h in self.horizons:
            st = horizon_stats(ticks, float(h))
            if h == self.contest_horizon or not st["covered"]:
                score = lock if h == self.contest_horizon else lock_sign * max(0.05, abs(lock))
                side = lock_side
            else:
                score = W_LOCK * lock + W_DRIFT * st["drift"] + W_PERSIST * st["persistence"]
                side = "BUY" if score > 0 else "SELL" if score < 0 else lock_side
            conf = float(min(1.0, abs(score)))
            p_side = 0.5 + 0.45 * conf
            p_win = p_side * (1.0 - st["p_tie"])
            logic = [
                f"locked contest pick {lock_side} at {abs(lock):.0%} (weight {fx(W_LOCK, 2)})",
                (f"WDR@{h}s: t = {fxs(st['t'], 2)} over {st['n']} overlapping windows -> drift {fxs(st['drift'], 3)} (weight {fx(W_DRIFT, 2)})"
                 if st["covered"] else f"tape shorter than 2 x {h}s - horizon maths abstains, the lock carries"),
                (f"PRT@{h}s: rho1 = {fxs(st['rho'], 2)}, last {h}s return {st['last_bps']:+.1f} bp -> persistence {fxs(st['persistence'], 3)} (weight {fx(W_PERSIST, 2)})"
                 if st["covered"] else ""),
                f"PIN@{h}s: P(tie) = {st['p_tie']:.1%}, sigma_{h}s = {st['sigma_bps']:.1f} bp",
                (f"score {fxs(score, 3)} -> {side}; P(win) = ({p_side:.0%} side) x (1 - {st['p_tie']:.0%} tie) = {p_win:.0%}"
                 if h != self.contest_horizon else
                 f"the contest horizon: the pick IS the Signal Lock ({lock_side}); P(win) = {p_win:.0%}"),
            ]
            pick = HorizonPick(cycle=cycle, horizon=h, side=side, score=float(score), confidence=conf,
                               p_win=float(p_win), p_tie=float(st["p_tie"]), entry=float(entry),
                               issued_at=now, due_at=now + h, stats=st, logic=[x for x in logic if x])
            lane = self.lanes[h]
            lane.picks.append(pick)
            if len(lane.picks) > HISTORY:
                del lane.picks[:-HISTORY]
            picks.append(pick)
        self.last_issue = {"cycle": cycle, "at": now, "entry": entry, "lock": lock_side}
        return picks

    def settle(self, pick: HorizonPick, exit_price: float, now: float | None = None) -> HorizonPick:
        """Strict sign: tie = loss (the contest's rule, applied to every lane)."""
        lane = self.lanes[pick.horizon]
        pick.exit = float(exit_price)
        pick.settled_at = float(now if now is not None else time.time())
        pick.tie = pick.exit == pick.entry
        up = pick.exit > pick.entry
        pick.won = (not pick.tie) and ((pick.side == "BUY") == up)
        if pick.tie:
            lane.ties += 1
        if pick.won:
            lane.wins += 1
            lane.streak += 1
            lane.best_streak = max(lane.best_streak, lane.streak)
        else:
            lane.losses += 1
            lane.streak = 0
        return pick

    def to_dict(self) -> dict:
        lanes = [self.lanes[h].to_dict() for h in self.horizons]
        best = max((l for l in lanes if l["settled"] >= 10), key=lambda l: l["win_rate"] or 0.0, default=None)
        return {
            "horizons": list(self.horizons),
            "contest_horizon": self.contest_horizon,
            "lanes": lanes,
            "last_issue": dict(self.last_issue),
            "best_measured": best["horizon"] if best else None,
            "reads": ("one boundary, four horizons: each lane is the side the same frozen tape gives for a 55 / 60 / 70 / 120 s "
                      "window, settled strictly on the last trade h seconds later (tie = loss); only the 70 s lane is the contest pick"),
        }

    # persistence ------------------------------------------------------
    def export_state(self) -> dict:
        return {str(h): {"wins": l.wins, "losses": l.losses, "ties": l.ties, "streak": l.streak,
                         "best_streak": l.best_streak} for h, l in self.lanes.items()}

    def import_state(self, payload: dict) -> None:
        for h, row in (payload or {}).items():
            lane = self.lanes.get(int(h))
            if lane is None or not isinstance(row, dict):
                continue
            for key in ("wins", "losses", "ties", "streak", "best_streak"):
                setattr(lane, key, int(row.get(key, 0) or 0))
