"""Round BL - the Temporal Arbitrage Engine.

Every window the contest sells ONE structure: *continuous risk*, the sign of
the last trade over 70 straight seconds.  This module prices, at the same
boundary and from the same frozen tape, its **synthetic twin**: *segmented
risk* - a 60 s leg that rolls at second 60 into a 10 s leg, so that the two
legs together cover the same 70 seconds.

The two structures are NOT the same bet and the market does not price them
the same.  The 70 s variance decomposes exactly as

    sigma_70^2 = sigma_60^2 + sigma_10^2 + 2 * rho_roll * sigma_60 * sigma_10

where ``rho_roll`` is the correlation between the first 60 s of a window and
the 10 s that follow it, measured on the tape.  The cross term is the
**volatility decay spread**: it is what the continuous window carries and the
segmented twin does not.  When ``rho_roll`` is negative the 70 s window loses
variance to mean reversion at the roll and the twin owns more sign-risk
(per leg) than the single window; when positive, the reverse.  The twin's
second leg is also *re-priced at the roll* with the first leg's realised
return in hand - information the continuous bet can never use - and the
Gaussian identity ``P(sign r_10 = sign r_60) = 1/2 + arcsin(rho)/pi`` prices
that leg before the window opens.

Everything is measured, never assumed:

* at the boundary the engine prices both structures (``price``) - P(win)
  per leg, expected win-units, tie risk per leg and the spread;
* at second 60 it records the roll print and decides the second leg's side
  (``roll``) - from ``rho_roll`` and the realised first leg, or the lock side
  when the roll correlation is too weak to act on;
* at second 70 it settles both structures on the last trade (``settle``)
  and books the *realised* spread: twin P&L minus continuous P&L in bps,
  plus how often the twin out-earned the single window, and - the case the
  structure exists for - what the twin earned in windows the 70 s print
  closed flat (a tie, a loss for the continuous bet by contest rule).

The contest pick is never touched: the lock still decides the 70 s side and
the twin's first leg inherits it.  This is a pricing engine and a ledger, so
the dashboard can say in numbers whether segmented risk pays here - and the
realised ledger, not the model, is the verdict.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

import numpy as np

from backend.core.numfmt import fxs
from backend.formulas.category_i_window import _window as W

LEG_A_S = 60.0          # first leg
LEG_B_S = 10.0          # rolled leg
TOTAL_S = 70.0          # the contest window
HISTORY = 240
MIN_PAIRS = 12          # roll pairs needed before rho_roll is trusted
ACT_RHO = 0.10          # |rho_roll| below this: the roll leg keeps the lock side


def _phi(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _probit(p: float) -> float:
    """Inverse normal CDF (Acklam's rational approximation; |err| < 1e-9)."""
    p = min(max(p, 1e-9), 1.0 - 1e-9)
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00, 3.754408661907416e+00]
    if p < 0.02425:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    if p > 1 - 0.02425:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    q = p - 0.5
    r = q * q
    return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)


def _sigma(ticks: np.ndarray, horizon_s: float, stride_s: float) -> tuple[float, float, int]:
    """(sigma_bps, P(flat), n) of the horizon's returns over the tape."""
    opens, closes = W.horizon_returns(ticks, horizon_s, stride_s=stride_s)
    if opens.size < 6:
        return 0.0, 0.0, 0
    r = (closes / opens - 1.0) * 1e4
    sd = float(np.sqrt(np.mean((r - r.mean()) ** 2)))
    return max(sd, 1e-6), float(np.mean(r == 0.0)), int(r.size)


def roll_pairs(ticks: np.ndarray, stride_s: float = 2.0) -> tuple[np.ndarray, np.ndarray]:
    """(r_60, r_10) in bps for every window on the tape: the first leg's
    return and the return of the 10 s that roll out of it."""
    tr = W.trades_only(ticks)
    if tr.shape[0] < 20:
        return np.zeros(0), np.zeros(0)
    t = tr[:, 0] / 1000.0
    p = tr[:, 1]
    if t[-1] - t[0] < 2.0 * TOTAL_S:
        return np.zeros(0), np.zeros(0)
    starts = np.arange(t[0], t[-1] - TOTAL_S, stride_s)
    i0 = np.searchsorted(t, starts, side="right") - 1
    i1 = np.searchsorted(t, starts + LEG_A_S, side="right") - 1
    i2 = np.searchsorted(t, starts + TOTAL_S, side="right") - 1
    keep = (i0 >= 0) & (i1 > i0) & (i2 > i1)
    a = (p[i1[keep]] / p[i0[keep]] - 1.0) * 1e4
    b = (p[i2[keep]] / p[i1[keep]] - 1.0) * 1e4
    return a, b


def price_structures(ticks: np.ndarray, lock_side: str, lock_confidence: float) -> dict:
    """Price continuous (70 s) risk against its segmented twin (60 s -> 10 s)
    on one frozen tape.  Never raises; ``covered`` says whether the tape was
    long enough to measure."""
    out = {"covered": False, "sigma_60": 0.0, "sigma_10": 0.0, "sigma_70": 0.0, "rho_roll": 0.0, "pairs": 0,
           "spread_bps2": 0.0, "spread_share": 0.0, "p_tie_70": 0.0, "p_tie_60": 0.0, "p_tie_10": 0.0,
           "p_win_70": 0.0, "p_win_60": 0.0, "p_win_10": 0.0, "ev_continuous": 0.0, "ev_twin": 0.0,
           "twin_edge": 0.0, "roll_rule": "lock side", "logic": []}
    try:
        side = str(lock_side).upper()
        conf = float(np.clip(lock_confidence, 0.0, 1.0))
        s60, flat60, n60 = _sigma(ticks, LEG_A_S, 5.0)
        s10, flat10, n10 = _sigma(ticks, LEG_B_S, 2.0)
        s70, flat70, n70 = _sigma(ticks, TOTAL_S, 5.0)
        a, b = roll_pairs(ticks)
        if not (n60 and n10 and n70) or a.size < MIN_PAIRS:
            out["logic"] = ["tape shorter than two windows - the twin cannot be priced yet"]
            return out
        da, db = a - a.mean(), b - b.mean()
        den = float(np.sqrt(np.dot(da, da) * np.dot(db, db)))
        rho = float(np.dot(da, db) / den) if den > 0 else 0.0
        rho = float(np.clip(rho, -0.95, 0.95))
        # the exact decomposition of the 70 s variance
        cross = 2.0 * rho * s60 * s10
        seg = s60 ** 2 + s10 ** 2
        s70_model = math.sqrt(max(seg + cross, 1e-9))
        spread_share = cross / max(s70 ** 2, 1e-9)
        # the lock's edge expressed as a drift-to-vol ratio at 70 s, then
        # re-scaled to each leg (drift grows with t, vol with sqrt t)
        p_side = 0.5 + 0.45 * conf
        z70 = _probit(p_side)
        mu_rate = z70 * s70 / TOTAL_S                      # bps per second
        z60 = mu_rate * LEG_A_S / s60
        z10 = mu_rate * LEG_B_S / s10
        p60 = _phi(z60)
        p70 = _phi(z70)
        # the roll leg: side chosen at second 60 from the first leg's sign
        # when the roll correlation is actionable, else the lock side
        p_cond = 0.5 + abs(math.asin(rho)) / math.pi
        if abs(rho) >= ACT_RHO and a.size >= MIN_PAIRS:
            p10 = max(p_cond, _phi(z10))
            rule = ("continue the first leg's sign" if rho > 0 else "fade the first leg's sign")
        else:
            p10 = _phi(z10)
            rule = "lock side"
        # tie risk per structure: a flat print is a loss by contest rule
        prices = W.trades_only(ticks)[:, 1]
        tick_bps = (W.tick_size(prices) / float(prices[-1])) * 1e4 if prices.size and prices[-1] > 0 else 0.0
        tie = lambda sd, flat: float(np.clip(max(flat, 2.0 * _phi(tick_bps / (2.0 * sd)) - 1.0), 0.0, 1.0))  # noqa: E731
        t70, t60, t10 = tie(s70, flat70), tie(s60, flat60), tie(s10, flat10)
        # expected win-units (+1 / -1 per structure, tie = -1), legs weighted
        # by the share of 70 s risk each one carries
        ev_c = (2.0 * p70 * (1.0 - t70) - 1.0)
        w60, w10 = s60 / (s60 + s10), s10 / (s60 + s10)
        ev_t = w60 * (2.0 * p60 * (1.0 - t60) - 1.0) + w10 * (2.0 * p10 * (1.0 - t10) - 1.0)
        edge = ev_t - ev_c
        out.update({
            "covered": True, "sigma_60": round(s60, 2), "sigma_10": round(s10, 2), "sigma_70": round(s70, 2),
            "sigma_70_model": round(s70_model, 2), "rho_roll": round(rho, 4), "pairs": int(a.size),
            "spread_bps2": round(cross, 3), "spread_share": round(spread_share, 4),
            "p_tie_70": round(t70, 4), "p_tie_60": round(t60, 4), "p_tie_10": round(t10, 4),
            "p_win_70": round(p70 * (1 - t70), 4), "p_win_60": round(p60 * (1 - t60), 4),
            "p_win_10": round(p10 * (1 - t10), 4), "ev_continuous": round(ev_c, 4), "ev_twin": round(ev_t, 4),
            "twin_edge": round(edge, 4), "roll_rule": rule, "leg_a_side": side,
        })
        out["logic"] = [
            f"sigma_60 = {s60:.1f} bp, sigma_10 = {s10:.1f} bp, sigma_70 = {s70:.1f} bp (tape {n70} windows)",
            f"rho_roll = {fxs(rho, 3)} over {a.size} roll pairs -> cross term 2·rho·s60·s10 = {fxs(cross, 1)} bp² "
            f"= {spread_share:+.1%} of the 70 s variance (volatility decay spread)",
            f"check: sqrt(s60² + s10² + cross) = {s70_model:.1f} bp vs measured sigma_70 {s70:.1f} bp",
            f"continuous: {side} 70 s, P(side) {p70:.1%} x (1 - tie {t70:.1%}) -> EV {fxs(ev_c, 3)} win-units",
            f"twin leg A: {side} 60 s, P(side) {p60:.1%}, tie {t60:.1%}; "
            f"leg B: 10 s rolled at :60, rule '{rule}', P(side) {p10:.1%} (arcsin law {p_cond:.1%}), tie {t10:.1%}",
            f"twin EV {fxs(ev_t, 3)} -> twin edge {fxs(edge, 3)} win-units per window ({'twin' if edge > 0 else 'continuous'} is the richer structure here)",
        ]
    except Exception as exc:  # noqa: BLE001
        out["logic"] = [f"pricing abstained: {exc}"]
    return out


@dataclass
class TwinTrade:
    cycle: int
    entry: float
    issued_at: float
    roll_at: float
    due_at: float
    lock_side: str
    pricing: dict
    roll_price: float | None = None
    leg_b_side: str | None = None
    roll_note: str = ""
    exit: float | None = None
    r60_bps: float | None = None
    r10_bps: float | None = None
    r70_bps: float | None = None
    continuous_pnl: float | None = None
    twin_pnl: float | None = None
    continuous_won: bool | None = None
    leg_a_won: bool | None = None
    leg_b_won: bool | None = None
    flat_70: bool = False
    settled_at: float | None = None

    def to_dict(self) -> dict:
        return {k: (round(v, 3) if isinstance(v, float) else v) for k, v in self.__dict__.items()}


class TemporalArbEngine:
    """Prices the synthetic twin at every boundary, rolls it at :60, settles
    both structures at :70 and keeps the realised ledger."""

    def __init__(self) -> None:
        self.trades: list[TwinTrade] = []
        self.windows = 0
        self.twin_better = 0
        self.flat_windows = 0
        self.flat_twin_bps = 0.0
        self.sum_spread_bps = 0.0
        self.sum_continuous_bps = 0.0
        self.sum_twin_bps = 0.0
        self.leg_a_wins = 0
        self.leg_b_wins = 0
        self.continuous_wins = 0
        self.last_pricing: dict = {}

    # ------------------------------------------------------------------
    def price(self, cycle: int, entry: float, lock_side: str, lock_confidence: float,
              ticks: np.ndarray, now: float | None = None) -> TwinTrade:
        now = float(now if now is not None else time.time())
        pricing = price_structures(ticks, lock_side, lock_confidence)
        trade = TwinTrade(cycle=int(cycle), entry=float(entry), issued_at=now, roll_at=now + LEG_A_S,
                          due_at=now + TOTAL_S, lock_side=str(lock_side).upper(), pricing=pricing)
        self.trades.append(trade)
        if len(self.trades) > HISTORY:
            del self.trades[:-HISTORY]
        self.last_pricing = dict(pricing, cycle=int(cycle))
        return trade

    def roll(self, trade: TwinTrade, roll_price: float, now: float | None = None) -> TwinTrade:
        """Second 60: book the roll print and pick the second leg's side."""
        trade.roll_price = float(roll_price)
        r60 = (trade.roll_price / trade.entry - 1.0) * 1e4 if trade.entry > 0 else 0.0
        trade.r60_bps = r60
        rho = float(trade.pricing.get("rho_roll", 0.0) or 0.0)
        rule = trade.pricing.get("roll_rule", "lock side")
        if rule != "lock side" and r60 != 0.0:
            sign = (1.0 if r60 > 0 else -1.0) * (1.0 if rho > 0 else -1.0)
            trade.leg_b_side = "BUY" if sign > 0 else "SELL"
            trade.roll_note = f"leg A closed {r60:+.1f} bp; rho_roll {rho:+.2f} -> {rule} -> {trade.leg_b_side}"
        else:
            trade.leg_b_side = trade.lock_side
            trade.roll_note = (f"leg A closed {r60:+.1f} bp; roll correlation too weak to act on -> keep the lock side"
                               if r60 != 0.0 else "leg A closed flat - nothing to roll on -> keep the lock side")
        return trade

    def settle(self, trade: TwinTrade, exit_price: float, now: float | None = None) -> TwinTrade:
        trade.exit = float(exit_price)
        trade.settled_at = float(now if now is not None else time.time())
        if trade.roll_price is None:            # roll print missed - it rolls at the exit
            self.roll(trade, trade.exit)
        e, m, x = trade.entry, float(trade.roll_price), trade.exit
        r70 = (x / e - 1.0) * 1e4
        r10 = (x / m - 1.0) * 1e4 if m > 0 else 0.0
        sa = 1.0 if trade.lock_side == "BUY" else -1.0
        sb = 1.0 if trade.leg_b_side == "BUY" else -1.0
        trade.r70_bps, trade.r10_bps = r70, r10
        trade.continuous_pnl = sa * r70
        trade.twin_pnl = sa * float(trade.r60_bps or 0.0) + sb * r10
        trade.flat_70 = x == e
        trade.continuous_won = (not trade.flat_70) and ((x > e) == (sa > 0))
        trade.leg_a_won = (m != e) and ((m > e) == (sa > 0))
        trade.leg_b_won = (x != m) and ((x > m) == (sb > 0))
        # ledger
        self.windows += 1
        self.sum_continuous_bps += trade.continuous_pnl
        self.sum_twin_bps += trade.twin_pnl
        self.sum_spread_bps += trade.twin_pnl - trade.continuous_pnl
        if trade.twin_pnl > trade.continuous_pnl + 0.05:
            self.twin_better += 1
        if trade.flat_70:
            self.flat_windows += 1
            self.flat_twin_bps += trade.twin_pnl
        self.continuous_wins += int(bool(trade.continuous_won))
        self.leg_a_wins += int(bool(trade.leg_a_won))
        self.leg_b_wins += int(bool(trade.leg_b_won))
        return trade

    # ------------------------------------------------------------------
    def verdict(self) -> str:
        if self.windows < 10:
            return f"ledger too short ({self.windows} settled) - the pricing is a model until the realised spread confirms it"
        avg = self.sum_spread_bps / self.windows
        share = self.twin_better / self.windows
        if avg > 0.5 and share > 0.5:
            return (f"segmented risk pays here: the twin out-earned the single 70 s window by {avg:+.2f} bp per window "
                    f"and in {share:.0%} of windows")
        if avg < -0.5 and share < 0.5:
            return (f"continuous risk pays here: the twin under-earned the single window by {avg:+.2f} bp per window; "
                    f"the roll gives information away rather than capturing it")
        return f"no priced spread in this tape yet: {avg:+.2f} bp per window, twin ahead in {share:.0%} of windows"

    def to_dict(self) -> dict:
        cur = self.trades[-1] if self.trades else None
        n = max(1, self.windows)
        return {
            "structure": {"continuous": "70 s single window", "twin": "60 s leg rolled at :60 into a 10 s leg"},
            "current": cur.to_dict() if cur else None,
            "pricing": dict(self.last_pricing),
            "ledger": {
                "windows": self.windows, "twin_better": self.twin_better,
                "twin_better_share": round(self.twin_better / n, 4),
                "avg_spread_bps": round(self.sum_spread_bps / n, 3),
                "sum_spread_bps": round(self.sum_spread_bps, 2),
                "sum_continuous_bps": round(self.sum_continuous_bps, 2),
                "sum_twin_bps": round(self.sum_twin_bps, 2),
                "flat_windows": self.flat_windows, "flat_twin_bps": round(self.flat_twin_bps, 2),
                "continuous_wins": self.continuous_wins, "leg_a_wins": self.leg_a_wins, "leg_b_wins": self.leg_b_wins,
                "continuous_rate": round(self.continuous_wins / n, 4),
                "leg_a_rate": round(self.leg_a_wins / n, 4), "leg_b_rate": round(self.leg_b_wins / n, 4),
            },
            "recent": [t.to_dict() for t in self.trades[-12:]],
            "verdict": self.verdict(),
            "reads": ("continuous 70 s risk priced against its segmented twin (60 s -> roll -> 10 s) on the same tape; "
                      "the cross term 2·rho·sigma_60·sigma_10 is the volatility decay spread; the ledger is the realised "
                      "twin minus continuous P&L, settled on the last trade"),
        }

    # persistence ------------------------------------------------------
    def export_state(self) -> dict:
        return {k: getattr(self, k) for k in ("windows", "twin_better", "flat_windows", "flat_twin_bps", "sum_spread_bps",
                                              "sum_continuous_bps", "sum_twin_bps", "leg_a_wins", "leg_b_wins",
                                              "continuous_wins")}

    def import_state(self, payload: dict) -> None:
        for k, v in (payload or {}).items():
            if hasattr(self, k) and isinstance(v, (int, float)):
                setattr(self, k, type(getattr(self, k))(v))
