"""Hindsight engine - the last hour of the market re-grades every input.

Round BI.  The user asked for a system that "analyses the already-happened
last 1 hour of market and changes all formulas according to them: analyse
the current formula results and change them according to where they had
loss - each feature, each agent, each system".

What it does, every settled window
----------------------------------
1. At every lock the engine records the *signed reading* of every input that
   went into the window: each directional formula, each AI agent (side x
   confidence), the brain (CCSv2), the physics layer, the genesis vote, the
   news index, the crowd tone, the forward table, the spec's fused score,
   and the signed micro-structure features the emotions are built from.
2. When the window settles the record is scored against the real move (last
   trade to last trade, in bps).  Only the windows of the **trailing hour**
   count - older windows roll off, so the review always describes the
   market that just happened, not last month's.
3. For every input the review computes, over that hour: how many times it
   was on the right side, how many on the wrong side, the bps it earned
   when right, the bps it *lost* when wrong, the windows it lost in, its
   mean reading (bias) and the slope of the move on its reading.
4. From that it derives a bounded correction that is applied to the input
   from the next window on - printed, never silent:

   * **re-centre** - if the input leaned one way all hour and that side
     lost, its hourly mean is subtracted (clipped to half its own spread);
   * **damp / boost** - the multiplier ``clip(1 + 0.25 z, 0.25, 1.5)`` where
     ``z = (hits - misses) / sqrt(hits + misses)`` - an input that lost 9 of
     12 windows now carries a quarter of its weight; one that won 10 of 12
     carries 1.5x;
   * **invert** - only when the hour is overwhelming (``z <= -3`` on at
     least 16 decided windows): a consistently inverted indicator is
     information, so it is read the other way round, and the panel says so.

No input is removed and no formula's mathematics is edited: what changes is
how much, and in which direction, each reading counts, as taught by the
hour that just happened.  Corrections expire with the rows that justified
them.  State survives a restart through the vault.
"""
from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import dataclass, field

import numpy as np

HOUR_SECONDS = 3600.0
#: decided windows an input needs in the hour before any correction applies
MIN_DECIDED = 8
#: the strongest damping / boost the hour may apply
MULT_FLOOR, MULT_CEIL = 0.25, 1.5
#: inversion needs an overwhelming hour
INVERT_Z, INVERT_MIN = -3.0, 16
#: a reading smaller than this has no side
EPS = 1e-9
#: the hedge formulas that carry a side (HSI is a dampener and never votes)
HEDGE_FORMULAS = ("HRDD", "SHRP", "GCDV")
#: the signed features (from the crowd-emotion feature set) that are reviewed
SIGNED_FEATURES = (
    "return_1s_bps", "return_win_bps", "z_win", "aggression_1s", "aggression_5s", "aggression_window",
    "aggression_micro", "depth_imbalance", "candle_trend_bps", "news_sentiment", "social_sentiment",
    "herding",
)


@dataclass
class WindowRecord:
    cycle: int
    at: float
    side: str
    confidence: float
    inputs: dict[str, float]
    groups: dict[str, str]
    move_bps: float | None = None
    cost_bps: float = 0.0
    settled: bool = False

    @property
    def decided(self) -> bool:
        return self.settled and self.move_bps is not None and abs(self.move_bps) > self.cost_bps

    def to_dict(self) -> dict:
        return {"cycle": self.cycle, "at": self.at, "side": self.side, "confidence": self.confidence,
                "inputs": self.inputs, "groups": self.groups, "move_bps": self.move_bps,
                "cost_bps": self.cost_bps, "settled": self.settled}

    @classmethod
    def from_dict(cls, raw: dict) -> "WindowRecord":
        return cls(cycle=int(raw["cycle"]), at=float(raw["at"]), side=str(raw.get("side") or ""),
                   confidence=float(raw.get("confidence") or 0.0),
                   inputs={str(k): float(v) for k, v in (raw.get("inputs") or {}).items()},
                   groups={str(k): str(v) for k, v in (raw.get("groups") or {}).items()},
                   move_bps=None if raw.get("move_bps") is None else float(raw["move_bps"]),
                   cost_bps=float(raw.get("cost_bps") or 0.0), settled=bool(raw.get("settled")))


@dataclass
class ComponentReview:
    name: str
    group: str
    n: int = 0
    hits: int = 0
    misses: int = 0
    silent: int = 0
    won_bps: float = 0.0
    lost_bps: float = 0.0
    bias: float = 0.0
    spread: float = 0.0
    slope_bps: float = 0.0
    z: float = 0.0
    losses: list[dict] = field(default_factory=list)
    offset: float = 0.0
    multiplier: float = 1.0
    inverted: bool = False
    action: str = "watching"
    reason: str = ""

    @property
    def decided(self) -> int:
        return self.hits + self.misses

    @property
    def hit_rate(self) -> float | None:
        return self.hits / self.decided if self.decided else None

    @property
    def net_bps(self) -> float:
        return self.won_bps - self.lost_bps

    def apply(self, value: float) -> float:
        out = (float(value) - self.offset) * self.multiplier
        return -out if self.inverted else out

    def to_dict(self) -> dict:
        return {
            "name": self.name, "group": self.group, "windows": self.n, "decided": self.decided,
            "hits": self.hits, "misses": self.misses, "silent": self.silent,
            "hit_rate": None if self.hit_rate is None else round(self.hit_rate, 4),
            "won_bps": round(self.won_bps, 2), "lost_bps": round(self.lost_bps, 2), "net_bps": round(self.net_bps, 2),
            "bias": round(self.bias, 6), "spread": round(self.spread, 6), "slope_bps": round(self.slope_bps, 4),
            "z": round(self.z, 3), "losses": self.losses,
            "offset": round(self.offset, 6), "multiplier": round(self.multiplier, 4), "inverted": self.inverted,
            "action": self.action, "reason": self.reason,
        }


class HindsightReviewer:
    """Keeps the trailing-hour record and the per-input corrections."""

    def __init__(self, hour_seconds: float = HOUR_SECONDS, max_rows: int = 3000) -> None:
        self.hour_seconds = float(hour_seconds)
        self.rows: deque[WindowRecord] = deque(maxlen=max_rows)
        self.reviews: dict[str, ComponentReview] = {}
        self.last_review_at: float = 0.0
        self.reviewed_rows: int = 0
        self.reviewed_decided: int = 0
        self.last_settled: dict | None = None

    # ------------------------------------------------------------------
    # recording
    # ------------------------------------------------------------------
    @staticmethod
    def inputs_from(*, formula_values: dict, directional: dict, agents: dict, ccs_value: float,
                    physics_vote: float, genesis_vote: float, niv: float, crowd_tone: float,
                    forward_vote: float, spec_score: float, features: dict | None,
                    emotions: list | None = None) -> tuple[dict, dict]:
        """Every signed reading that entered this window, with its layer
        (formula · hedge · agent · reasoning · physics · genesis · emotion · feature)."""
        inputs: dict[str, float] = {}
        groups: dict[str, str] = {}

        def put(name: str, value, group: str) -> None:
            try:
                v = float(value)
            except (TypeError, ValueError):
                return
            if math.isfinite(v):
                inputs[name] = v
                groups[name] = group

        for name in directional:
            if name in formula_values:
                put(name, formula_values[name], "hedge" if name in HEDGE_FORMULAS else "formula")
        for name, res in (agents or {}).items():
            decision = getattr(res, "decision", None)
            if decision not in ("BUY", "SELL"):
                continue
            conf = float(getattr(res, "confidence", None) or 0.5)
            put(f"agent:{name}", conf if decision == "BUY" else -conf, "agent")
        put("brain:CCSv2", ccs_value, "reasoning")
        put("physics", physics_vote, "physics")
        put("genesis", genesis_vote, "genesis")
        put("news:NIV", niv, "reasoning")
        put("crowd:tone", crowd_tone, "emotion")
        put("forward", forward_vote, "reasoning")
        put("fusion:recipe", spec_score, "reasoning")
        for e in emotions or []:
            try:
                name, tone, inten = e.get("name"), str(e.get("tone") or ""), float(e.get("intensity") or 0.0)
            except (AttributeError, TypeError, ValueError):
                continue
            sign = 1.0 if tone.lower() in ("bullish", "greed", "up", "positive") else -1.0 if tone.lower() in ("bearish", "fear", "down", "negative") else 0.0
            if name and sign:
                put(f"emotion:{name}", sign * inten, "emotion")
        for key in SIGNED_FEATURES:
            if features and key in features:
                put(f"feature:{key}", features.get(key), "feature")
        return inputs, groups

    def record(self, cycle: int, side: str, confidence: float, inputs: dict, groups: dict,
               at: float | None = None) -> None:
        cycle = int(cycle)
        for row in self.rows:
            if row.cycle == cycle:
                row.inputs, row.groups, row.side, row.confidence = dict(inputs), dict(groups), side, float(confidence)
                return
        self.rows.append(WindowRecord(cycle=cycle, at=float(at if at is not None else time.time()), side=side,
                                      confidence=float(confidence), inputs=dict(inputs), groups=dict(groups)))

    def settle(self, cycle: int, move_bps: float, cost_bps: float = 0.0) -> bool:
        for row in self.rows:
            if row.cycle == int(cycle):
                row.move_bps = float(move_bps)
                row.cost_bps = float(cost_bps)
                row.settled = True
                self.last_settled = {"cycle": row.cycle, "move_bps": round(row.move_bps, 2), "side": row.side}
                self.review()
                return True
        return False

    # ------------------------------------------------------------------
    # the review
    # ------------------------------------------------------------------
    def hour_rows(self, now: float | None = None) -> list[WindowRecord]:
        now = time.time() if now is None else now
        return [r for r in self.rows if r.settled and now - r.at <= self.hour_seconds]

    def review(self, now: float | None = None) -> dict[str, ComponentReview]:
        now = time.time() if now is None else now
        rows = self.hour_rows(now)
        reviews: dict[str, ComponentReview] = {}
        names: dict[str, str] = {}
        for r in rows:
            names.update(r.groups)
        for name, group in names.items():
            cr = ComponentReview(name=name, group=group)
            vals: list[float] = []
            moves: list[float] = []
            losses: list[dict] = []
            for r in rows:
                if name not in r.inputs:
                    continue
                v = r.inputs[name]
                cr.n += 1
                if not r.decided:
                    continue
                m = float(r.move_bps or 0.0)
                vals.append(v)
                moves.append(m)
                if abs(v) <= EPS:
                    cr.silent += 1
                    continue
                if (v > 0) == (m > 0):
                    cr.hits += 1
                    cr.won_bps += abs(m)
                else:
                    cr.misses += 1
                    cr.lost_bps += abs(m)
                    losses.append({"cycle": r.cycle, "value": round(v, 4), "move_bps": round(m, 2), "at": r.at})
            if vals:
                a = np.asarray(vals, dtype=np.float64)
                b = np.asarray(moves, dtype=np.float64)
                cr.bias = float(a.mean())
                cr.spread = float(a.std())
                den = float(np.dot(a, a))
                cr.slope_bps = float(np.dot(a, b) / den) if den > 0 else 0.0
            d = cr.decided
            cr.z = (cr.hits - cr.misses) / math.sqrt(d) if d else 0.0
            losses.sort(key=lambda x: -abs(x["move_bps"]))
            cr.losses = losses[:6]
            self._decide(cr, rows)
            reviews[name] = cr
        self.reviews = reviews
        self.last_review_at = now
        self.reviewed_rows = len(rows)
        self.reviewed_decided = sum(1 for r in rows if r.decided)
        return reviews

    def _decide(self, cr: ComponentReview, rows: list[WindowRecord]) -> None:
        d = cr.decided
        if d < MIN_DECIDED:
            cr.action = "watching"
            cr.reason = f"{d} decided windows in the hour - needs {MIN_DECIDED} before the hour may correct it"
            return
        # multiplier from the hour's edge z-score (bounded)
        cr.multiplier = float(min(MULT_CEIL, max(MULT_FLOOR, 1.0 + 0.25 * cr.z)))
        # re-centre when the side it leaned to lost
        cr.offset = 0.0
        if abs(cr.bias) > EPS and cr.spread > 0:
            lean = 1.0 if cr.bias > 0 else -1.0
            lean_hits = lean_miss = 0
            for r in rows:
                if not r.decided or cr.name not in r.inputs:
                    continue
                v = r.inputs[cr.name]
                if abs(v) <= EPS or (v > 0) != (lean > 0):
                    continue
                if (v > 0) == (float(r.move_bps) > 0):
                    lean_hits += 1
                else:
                    lean_miss += 1
            if lean_miss >= lean_hits:   # the lean earned nothing (or lost): take it out
                cr.offset = float(max(-0.5 * cr.spread, min(0.5 * cr.spread, cr.bias)))
        cr.inverted = bool(cr.z <= INVERT_Z and d >= INVERT_MIN)
        parts = []
        if cr.inverted:
            parts.append(f"INVERTED - wrong {cr.misses}/{d} times this hour (z {cr.z:+.1f}), read the other way round")
            cr.multiplier = float(max(MULT_FLOOR, min(1.0, abs(cr.z) * 0.15)))
        elif cr.multiplier < 0.999:
            parts.append(f"damped x{cr.multiplier:.2f} - lost {cr.misses}/{d} windows costing {cr.lost_bps:.0f} bp")
        elif cr.multiplier > 1.001:
            parts.append(f"boosted x{cr.multiplier:.2f} - right {cr.hits}/{d} windows earning {cr.won_bps:.0f} bp")
        else:
            parts.append(f"kept x1.00 - {cr.hits}/{d} this hour, no edge either way")
        if cr.offset != 0.0:
            parts.append(f"re-centred by {cr.offset:+.4f} (leaned {'up' if cr.bias > 0 else 'down'} all hour and that side lost)")
        cr.action = "inverted" if cr.inverted else "damped" if cr.multiplier < 0.999 else "boosted" if cr.multiplier > 1.001 else "kept"
        if cr.offset != 0.0 and cr.action == "kept":
            cr.action = "re-centred"
        cr.reason = "; ".join(parts)

    # ------------------------------------------------------------------
    # applying
    # ------------------------------------------------------------------
    def active(self, name: str) -> ComponentReview | None:
        cr = self.reviews.get(name)
        return cr if cr is not None and cr.decided >= MIN_DECIDED else None

    def apply(self, name: str, value: float) -> float:
        cr = self.active(name)
        return float(value) if cr is None else cr.apply(value)

    def multiplier(self, name: str) -> float:
        cr = self.active(name)
        if cr is None:
            return 1.0
        return -cr.multiplier if cr.inverted else cr.multiplier

    def apply_map(self, values: dict) -> tuple[dict, dict]:
        """Corrected copy of ``values`` plus the per-name change that was made."""
        out = dict(values)
        changed: dict[str, dict] = {}
        for name, v in values.items():
            cr = self.active(name)
            if cr is None:
                continue
            try:
                nv = cr.apply(float(v))
            except (TypeError, ValueError):
                continue
            if nv != v:
                out[name] = nv
                changed[name] = {"before": round(float(v), 6), "after": round(nv, 6), "action": cr.action}
        return out, changed

    # ------------------------------------------------------------------
    # reporting / persistence
    # ------------------------------------------------------------------
    def to_dict(self, now: float | None = None) -> dict:
        now = time.time() if now is None else now
        table = sorted((cr.to_dict() for cr in self.reviews.values()), key=lambda r: (r["net_bps"], -r["lost_bps"]))
        by_group: dict[str, dict] = {}
        for r in table:
            g = by_group.setdefault(r["group"], {"inputs": 0, "corrected": 0, "lost_bps": 0.0, "won_bps": 0.0})
            g["inputs"] += 1
            g["corrected"] += 1 if r["action"] not in ("watching", "kept") else 0
            g["lost_bps"] = round(g["lost_bps"] + r["lost_bps"], 2)
            g["won_bps"] = round(g["won_bps"] + r["won_bps"], 2)
        rows = self.hour_rows(now)
        wins = sum(1 for r in rows if r.decided and ((r.side == "BUY") == (float(r.move_bps) > 0)))
        decided = sum(1 for r in rows if r.decided)
        lost_windows = sorted(
            ({"cycle": r.cycle, "side": r.side, "move_bps": round(float(r.move_bps), 2), "at": r.at,
              "wrong_inputs": sum(1 for k, v in r.inputs.items() if abs(v) > EPS and (v > 0) != (float(r.move_bps) > 0)),
              "inputs": len(r.inputs)}
             for r in rows if r.decided and ((r.side == "BUY") != (float(r.move_bps) > 0))),
            key=lambda x: -abs(x["move_bps"]))
        return {
            "available": bool(self.reviews),
            "hour_seconds": self.hour_seconds,
            "windows_in_hour": len(rows),
            "decided_in_hour": decided,
            "engine_hits_in_hour": wins,
            "engine_hit_rate_hour": None if not decided else round(wins / decided, 4),
            "reviewed_at": self.last_review_at,
            "age_seconds": round(now - self.last_review_at, 1) if self.last_review_at else None,
            "min_decided": MIN_DECIDED,
            "bounds": {"multiplier": [MULT_FLOOR, MULT_CEIL], "offset": "±0.5 x own spread",
                       "invert": f"z <= {INVERT_Z:g} on >= {INVERT_MIN} decided windows"},
            "corrected": sum(1 for r in table if r["action"] not in ("watching", "kept")),
            "inputs": len(table),
            "by_group": by_group,
            "table": table,
            "lost_windows": lost_windows[:12],
            "last_settled": self.last_settled,
            "method": ("every settled window of the trailing hour grades every input on the real move; "
                       "multiplier = clip(1 + 0.25 z, 0.25, 1.5) with z = (hits − misses)/√decided; "
                       "re-centre by the hourly mean when the leaned side lost; invert only at z ≤ −3 on ≥ 16 windows"),
        }

    def export_state(self) -> dict:
        return {"rows": [r.to_dict() for r in self.rows], "saved_at": time.time()}

    def import_state(self, payload: dict) -> int:
        rows = (payload or {}).get("rows") or []
        n = 0
        for raw in rows:
            try:
                self.rows.append(WindowRecord.from_dict(raw))
                n += 1
            except (KeyError, TypeError, ValueError):
                continue
        if n:
            self.review()
        return n
