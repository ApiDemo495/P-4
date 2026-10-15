"""Round AS - the competition the app is really played in.

The rules, as the user plays them:

* every 70 s window must be called BUY or SELL - no skipping, no sizing;
* entry = the exchange's **last trade** at the opening second, settlement =
  the last trade at the closing second;
* strictly higher wins a BUY, strictly lower wins a SELL, **a tie loses**;
* the prize ladder pays on consecutive wins - 5 (coupon), 20 (coupon),
  50 (cash), 72 (cash) - and a loss resets the streak to zero.

That changes two things about the engine.  First, the number that matters is
the per-pick win rate *under exactly that rule* - not bps, not "flat inside
the spread is void".  :class:`CompetitionLedger` scores every window that way
and keeps the streak, the ladder and the probability arithmetic (a streak of
k at per-pick rate p happens with probability p**k; the expected number of
windows until the first such streak is (1 - p**k) / ((1 - p) p**k)).

Second, the side must be chosen against the actual entry print, at the
boundary.  :class:`BoundaryPicker` takes the locked recipe's probability and
adds the three things that are only known in the last second and that decide
the one-tick windows where ties and coin-flips live:

* **microprice gap** - where the size-weighted quote sits relative to the
  entry print.  A print at the ask with a heavy bid is *above* fair value and
  the next prints lean down (Stoikov 2018);
* **last-print reversal** - the aggressor side of the entry print itself
  (bid-ask bounce: Roll 1984), and the move of the last 3 s, which at a 70 s
  horizon reverts more often than it continues;
* **short order-flow imbalance** - signed volume of the last 5 s (Cont,
  Kukanov & Stoikov 2014), the only short-horizon feature that *continues*.

The combination is a logistic model whose weights start at literature-sized
priors and are then learned online from the competition outcomes (plain SGD
on the log-loss), so the picker gets better at *this* feed and never claims
more than it has measured.  Nothing here touches a window after it opens:
the pick is made once, at the boundary, and then locked like everything
else.
"""

from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import dataclass, field

import numpy as np

from backend.core import config as cfg

#: The ladder, as the user described it.
PRIZE_LADDER: tuple[tuple[int, str], ...] = (
    (5, "$1 fee coupon - only usable on a $50 trade, expires"),
    (20, "$10 coupon - only usable on a $100 trade, expires"),
    (50, "$30 real cash - withdrawable (the goal)"),
    (72, "$500 (designed to be unreachable)"),
)


def streak_probability(p: float, k: int) -> float:
    """P(k consecutive wins) at per-pick rate p."""
    p = min(max(float(p), 0.0), 1.0)
    return p ** int(k)


def expected_windows_to_streak(p: float, k: int) -> float | None:
    """Expected number of windows until the first run of k wins (Feller)."""
    p = float(p)
    if p <= 0.0:
        return None
    if p >= 1.0:
        return float(k)
    pk = p ** k
    return (1.0 - pk) / ((1.0 - p) * pk)


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, x))))


def _logit(p: float) -> float:
    p = min(max(float(p), 1e-4), 1.0 - 1e-4)
    return math.log(p / (1.0 - p))


# ----------------------------------------------------------------- the pick
@dataclass
class BoundaryFeatures:
    entry: float
    entry_side: float          # +1 buyer-aggressor print, -1 seller, 0 unknown
    microprice: float
    gap_ticks: float           # (microprice - entry) / tick
    r3_ticks: float            # entry vs price 3 s before, in ticks
    ofi5: float                # signed volume share of the last 5 s, [-1, 1]
    book_imbalance: float      # (bid_qty - ask_qty)/(bid_qty + ask_qty) at L1
    pin_rate: float            # share of recent windows that closed exactly flat
    tick: float
    prints_5s: int

    def as_dict(self) -> dict:
        return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in self.__dict__.items()}


@dataclass
class BoundaryPick:
    side: str
    p_up: float
    recipe_side: str
    recipe_p_up: float
    flipped: bool
    features: BoundaryFeatures
    contributions: dict
    note: str
    at: float = field(default_factory=time.time)

    def as_dict(self) -> dict:
        return {
            "side": self.side,
            "p_up": round(self.p_up, 4),
            "p_side": round(self.p_up if self.side == "BUY" else 1.0 - self.p_up, 4),
            "recipe_side": self.recipe_side,
            "recipe_p_up": round(self.recipe_p_up, 4),
            "flipped": self.flipped,
            "features": self.features.as_dict(),
            "contributions": {k: round(v, 4) for k, v in self.contributions.items()},
            "note": self.note,
            "at": self.at,
        }


def boundary_features(ticks: np.ndarray, bid: float, ask: float, bid_qty: float, ask_qty: float,
                      now_ms: float, pin_rate: float, tick: float | None = None) -> BoundaryFeatures | None:
    """Read the last-second state of the tape.  ``ticks`` = rows of
    ``[time_ms, price, qty, side]``; quote prints carry qty = 0 / side = 0."""
    if ticks is None or ticks.shape[0] == 0:
        return None
    trades = ticks[ticks[:, 2] > 0]
    source = trades if trades.shape[0] else ticks
    entry = float(source[-1, 1])
    if entry <= 0:
        return None
    entry_side = float(source[-1, 3]) if trades.shape[0] else 0.0
    if tick is None:
        # a tick size estimated from the tape: a low quantile of the non-zero
        # price steps (the exact tick on a real exchange tape, a sane step on
        # a continuous simulator), floored by half the spread and bounded so
        # a thin tape cannot make a $5 "tick"
        steps = np.abs(np.diff(source[-400:, 1]))
        steps = steps[steps > 0]
        tick = float(np.percentile(steps, 20)) if steps.size else entry * 1e-5
        if bid > 0 and ask > bid:
            tick = max(tick, (ask - bid) * 0.5)
        tick = min(max(tick, entry * 1e-7), entry * 1e-3)
    if bid > 0 and ask > bid and (bid_qty + ask_qty) > 0:
        microprice = (bid * ask_qty + ask * bid_qty) / (bid_qty + ask_qty)
        book_imbalance = (bid_qty - ask_qty) / (bid_qty + ask_qty)
    else:
        microprice, book_imbalance = entry, 0.0
    gap_ticks = (microprice - entry) / tick
    # last 3 s move
    t3 = source[source[:, 0] <= now_ms - 3000.0]
    ref3 = float(t3[-1, 1]) if t3.shape[0] else float(source[0, 1])
    r3_ticks = (entry - ref3) / tick
    # last 5 s signed volume
    recent = trades[trades[:, 0] >= now_ms - 5000.0] if trades.shape[0] else trades
    if recent.shape[0]:
        vol = float(np.sum(recent[:, 2]))
        ofi5 = float(np.sum(recent[:, 2] * recent[:, 3]) / vol) if vol > 0 else 0.0
    else:
        ofi5 = 0.0
    return BoundaryFeatures(
        entry=entry, entry_side=entry_side, microprice=float(microprice),
        gap_ticks=float(np.clip(gap_ticks, -10.0, 10.0)), r3_ticks=float(np.clip(r3_ticks, -50.0, 50.0)),
        ofi5=ofi5, book_imbalance=float(book_imbalance), pin_rate=float(pin_rate),
        tick=float(tick), prints_5s=int(recent.shape[0]),
    )


class BoundaryPicker:
    """Logistic side-picker, learned online on competition outcomes."""

    #: literature-sized priors (log-odds per unit feature)
    PRIOR = {
        "recipe": 1.0,        # × logit(recipe p_up)
        "gap": 0.35,          # × tanh(gap_ticks / 2)
        "bounce": -0.20,      # × entry_side  (a buy print leans down next)
        "r3": -0.12,          # × tanh(r3_ticks / 5)   (3 s reversal)
        "ofi": 0.30,          # × ofi5
        "imb": 0.25,          # × book imbalance
        "bias": 0.0,
    }
    LR = 0.03
    L2 = 0.002

    def __init__(self) -> None:
        self.w = dict(self.PRIOR)
        self.updates = 0
        self.log_loss = deque(maxlen=200)
        self._pending: dict[int, tuple[dict, float]] = {}

    @staticmethod
    def _x(f: BoundaryFeatures, recipe_p_up: float) -> dict:
        return {
            "recipe": _logit(recipe_p_up),
            "gap": math.tanh(f.gap_ticks / 2.0),
            "bounce": f.entry_side,
            "r3": math.tanh(f.r3_ticks / 5.0),
            "ofi": f.ofi5,
            "imb": f.book_imbalance,
            "bias": 1.0,
        }

    def pick(self, cycle: int, f: BoundaryFeatures, recipe_side: str, recipe_p_up: float) -> BoundaryPick:
        x = self._x(f, recipe_p_up)
        contributions = {k: self.w[k] * x[k] for k in x}
        z = sum(contributions.values())
        p_full = _sigmoid(z)
        # Round AX - the tie problem.  ``pin_rate`` is the tie probability
        # (the larger of the ledger's flat-close share and the PIN formula).
        # When the window is likely to end within a tick of where it started,
        # the close is decided by the microstructure of the entry print - the
        # queue the microprice leans to, the bounce off the print side, the
        # book's own flow - not by a 70 s trend recipe that is indifferent to
        # a one-tick move.  So the microstructure terms (everything but the
        # recipe) are re-read on their own, amplified, and blended in with the
        # tie probability as the weight.
        micro_z = 2.0 * sum(v for k, v in contributions.items() if k not in ("recipe", "bias"))
        p_micro = _sigmoid(micro_z)
        pin = float(max(0.0, min(1.0, f.pin_rate)))
        p_up = (1.0 - pin) * p_full + pin * p_micro
        if pin >= 0.25:
            tie_note = (f" · tie risk {pin:.0%}: microstructure weighted {pin:.0%}"
                        f" (micro P(up) {p_micro:.3f}, full {p_full:.3f})")
            # never leave the pick on the knife edge in a pinned tape: break
            # toward the microprice if the blend is still indifferent
            if abs(p_up - 0.5) < 0.02 and abs(f.gap_ticks) > 0.05:
                p_up = 0.5 + 0.04 * (1 if f.gap_ticks > 0 else -1)
                tie_note += " · tie broken toward the microprice"
        else:
            tie_note = ""
        side = "BUY" if p_up >= 0.5 else "SELL"
        flipped = side != recipe_side
        lead = max(contributions, key=lambda k: abs(contributions[k]) if k != "bias" else -1)
        note = (
            f"entry {f.entry:.2f} ({'buy' if f.entry_side > 0 else 'sell' if f.entry_side < 0 else 'quote'} print)"
            f" · microprice {f.gap_ticks:+.2f} ticks away · last 3 s {f.r3_ticks:+.1f} ticks · OFI 5 s {f.ofi5:+.2f}"
            f" · P(up) {p_up:.3f} vs recipe {recipe_p_up:.3f}"
            f"{' · FLIPPED to ' + side if flipped else ' · recipe kept'} (largest term: {lead}){tie_note}"
        )
        self._pending[int(cycle)] = (x, p_up)
        if len(self._pending) > 50:
            for old in sorted(self._pending)[:-30]:
                self._pending.pop(old, None)
        return BoundaryPick(side, p_up, recipe_side, recipe_p_up, flipped, f, contributions, note)

    def learn(self, cycle: int, went_up: bool | None) -> None:
        """One SGD step on the log-loss; a tie teaches nothing about the side."""
        row = self._pending.pop(int(cycle), None)
        if row is None or went_up is None:
            return
        x, p = row
        y = 1.0 if went_up else 0.0
        err = p - y
        for k in self.w:
            self.w[k] -= self.LR * (err * x[k] + self.L2 * (self.w[k] - self.PRIOR[k]))
        self.w["recipe"] = max(0.2, self.w["recipe"])          # the recipe never turns into its own inverse here
        self.updates += 1
        self.log_loss.append(-(y * math.log(max(p, 1e-6)) + (1 - y) * math.log(max(1 - p, 1e-6))))

    def to_dict(self) -> dict:
        return {
            "weights": {k: round(v, 4) for k, v in self.w.items()},
            "priors": dict(self.PRIOR),
            "updates": self.updates,
            "log_loss_200": round(float(np.mean(self.log_loss)), 4) if self.log_loss else None,
            "reads": "p_up = sigma(w_recipe*logit(p_recipe) + w_gap*tanh(gap/2) + w_bounce*entry_side + w_r3*tanh(r3/5) + w_ofi*OFI5 + w_imb*imbalance + bias); learned by SGD on the competition outcomes",
        }


# --------------------------------------------------------------- the ledger
@dataclass
class CompetitionRow:
    cycle: int
    side: str
    entry: float
    exit: float
    won: bool
    tie: bool
    at: float

    def as_dict(self) -> dict:
        return {"cycle": self.cycle, "side": self.side, "entry": self.entry, "exit": self.exit,
                "won": self.won, "tie": self.tie, "at": self.at}


class CompetitionLedger:
    """Scores every window by the competition's rule and keeps the ladder."""

    def __init__(self, ladder=PRIZE_LADDER, keep: int = 2000) -> None:
        self.ladder = tuple(ladder)
        self.rows: deque[CompetitionRow] = deque(maxlen=keep)
        self.streak = 0
        self.best_streak = 0
        self.ties = 0
        self.milestones_hit: dict[int, int] = {k: 0 for k, _ in self.ladder}
        self.streak_history: deque[int] = deque(maxlen=200)   # length of every finished streak
        self.open: dict[int, tuple[str, float, float]] = {}   # cycle -> (side, entry, at)

    # -- recording -------------------------------------------------------
    def enter(self, cycle: int, side: str, entry: float) -> None:
        if entry > 0 and side in ("BUY", "SELL"):
            self.open[int(cycle)] = (side, float(entry), time.time())
            if len(self.open) > 20:
                for old in sorted(self.open)[:-10]:
                    self.open.pop(old, None)

    def settle(self, cycle: int, exit_price: float) -> CompetitionRow | None:
        spec = self.open.pop(int(cycle), None)
        if spec is None or exit_price <= 0:
            return None
        side, entry, at = spec
        tie = exit_price == entry
        won = (not tie) and ((exit_price > entry) if side == "BUY" else (exit_price < entry))
        row = CompetitionRow(int(cycle), side, entry, float(exit_price), won, tie, time.time())
        self.rows.append(row)
        if tie:
            self.ties += 1
        if won:
            self.streak += 1
            self.best_streak = max(self.best_streak, self.streak)
            if self.streak in self.milestones_hit:
                self.milestones_hit[self.streak] += 1
        else:
            if self.streak:
                self.streak_history.append(self.streak)
            self.streak = 0
        return row

    # -- reading ---------------------------------------------------------
    def lag_audit(self, window: int = 200) -> dict:
        """Round BA: is the pick a forecast or a description of the window
        that just closed?  Over consecutive settled windows: how often the
        side equalled the PREVIOUS window's direction (following), how often
        the NEXT one's (winning), and the win rate when following vs fading
        the last move.  A picker that is "right about the past" shows a high
        follow rate with a sub-50 % win-when-following."""
        rows = list(self.rows)[-window:]
        follow = follow_n = 0
        win_follow = n_follow = win_fade = n_fade = 0
        for prev, cur in zip(rows, rows[1:]):
            if prev.tie or cur.cycle != prev.cycle + 1:
                continue
            last_dir = "BUY" if prev.exit > prev.entry else "SELL"
            follow_n += 1
            if cur.side == last_dir:
                follow += 1
                n_follow += 1
                win_follow += 1 if cur.won else 0
            else:
                n_fade += 1
                win_fade += 1 if cur.won else 0
        return {
            "pairs": follow_n,
            "follows_last_window": round(follow / follow_n, 4) if follow_n else None,
            "win_when_following": round(win_follow / n_follow, 4) if n_follow else None,
            "win_when_fading": round(win_fade / n_fade, 4) if n_fade else None,
            "reads": "share of picks equal to the previous window's direction, and the strict-sign win rate of those vs the others",
        }

    def win_rate(self, window: int | None = None) -> float | None:
        rows = list(self.rows)[-window:] if window else list(self.rows)
        if not rows:
            return None
        return sum(1 for r in rows if r.won) / len(rows)

    def to_dict(self) -> dict:
        n = len(self.rows)
        p_all = self.win_rate()
        p_100 = self.win_rate(100)
        # the rate used for the arithmetic: the last 100 picks, shrunk toward
        # 0.5 while the sample is small so a lucky first hour does not print
        # "a 50-streak every afternoon"
        k = min(n, 100)
        p_est = ((p_100 or 0.5) * k + 0.5 * 10) / (k + 10) if n else 0.5
        next_rung = next((rung for rung, _ in self.ladder if rung > self.streak), None)
        ladder = []
        for rung, prize in self.ladder:
            ew = expected_windows_to_streak(p_est, rung)
            ladder.append({
                "streak": rung, "prize": prize, "hit": self.milestones_hit[rung],
                "probability_per_attempt": streak_probability(p_est, rung),
                "expected_windows": None if ew is None or ew > 1e12 else round(ew, 0),
                "expected_minutes": None if ew is None or ew > 1e12 else round(ew * cfg.window_seconds() / 60.0, 0),
                "expected_hours": None if ew is None or ew > 1e12 else round(ew * cfg.window_seconds() / 3600.0, 1),
                "reached": self.streak >= rung,
            })
        recent = [r.as_dict() for r in list(self.rows)[-30:]]
        return {
            "rule": f"BUY wins if the last trade at the end of the {cfg.window_seconds():g} s window is strictly above the last trade at its start; SELL strictly below; a tie is a loss; every window must be called",
            "scored": n,
            "wins": sum(1 for r in self.rows if r.won),
            "ties": self.ties,
            "win_rate": None if p_all is None else round(p_all, 4),
            "win_rate_100": None if p_100 is None else round(p_100, 4),
            "p_estimate": round(p_est, 4),
            "streak": self.streak,
            "best_streak": self.best_streak,
            "next_rung": next_rung,
            "to_next_rung": None if next_rung is None else next_rung - self.streak,
            "ladder": ladder,
            "streaks_finished": list(self.streak_history)[-20:],
            "mean_streak": round(float(np.mean(self.streak_history)), 2) if self.streak_history else None,
            "recent": recent,
            "lag_audit": self.lag_audit(),
            "reading": self._reading(p_est, n),
        }

    def _reading(self, p: float, n: int) -> str:
        if not n:
            return "no window settled yet - the scoreboard fills one window after the first lock"
        nxt = next((rung for rung, _ in self.ladder if rung > self.streak), None)
        parts = [f"{self.streak} in a row (best {self.best_streak})"]
        if nxt:
            parts.append(f"{nxt - self.streak} more to the {nxt}-rung")
        parts.append(f"per-pick {p:.0%} over the last {min(n, 100)} → a 5-streak {streak_probability(p, 5):.0%}, "
                     f"a 20-streak {streak_probability(p, 20):.2%}, a 50-streak {streak_probability(p, 50):.4%} per attempt")
        return " · ".join(parts)
