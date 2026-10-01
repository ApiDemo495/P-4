"""Evidence ledger - the engine learns which of its own inputs predict.

Why this exists (Round N, "predictions are not accurate").  Until now the
*side* of a window came from the spec's fixed recipe: the brain's CCSv2 read
of the 22 formulas (weight 0.40) plus whichever AI agents answered, with the
formulas, news and crowd only ever moving the *confidence*.  Nothing in that
chain had ever been checked against what the price then did.  A source that
is right 45 % of the time was weighted exactly like one that is right 58 %.

The ledger fixes that with the oldest honest tool there is: it keeps score.

* Every window, each **source** casts a directional vote (+1 up, -1 down, 0
  abstain): each of the 22 formulas that carries a direction, the brain's
  CCSv2, each AI agent, the news index, the crowd tone, the spec's own fused
  score, and three micro-structure priors that are known to matter at the
  one-minute horizon - the last minute's return (bid/ask bounce makes it
  mean-revert), the five-minute drift, and the signed order flow.
* When the window is scored, every source that voted is credited a hit or a
  miss.  Counts decay (half-life ``CALIBRATION_HALF_LIFE`` windows) so the
  ledger follows the regime instead of averaging over last month.
* A source's **reliability** is the posterior mean of a Beta(a, a) prior
  updated with its decayed hits/misses; its **weight** is the log-odds of
  that reliability.  A source at 50 % weighs nothing; one at 60 % pulls; one
  that is reliably *wrong* pulls the other way - a consistently inverted
  indicator is information, not noise.
* The window's side is the sign of the summed weighted votes, and its
  probability ``sigma(sum)`` is a *calibrated* number: the ledger also keeps
  the realised hit rate per probability bucket, so the UI can print "when the
  engine said 62 %, it was right 58 % of the time".

The ledger only takes over once it has ``CALIBRATION_MIN_SAMPLES`` scored
windows for the asset (default 30); before that the spec recipe decides and
the ledger just watches.  ``CALIBRATION_ENABLED=0`` keeps it in watch-only
mode for ever.  State is persisted to ``.run/calibration.json`` after every
update, so a restart does not forget.

This is not a promise of profit.  One-minute direction is close to a coin
toss for everyone; what the ledger guarantees is that the engine stops
trusting inputs that its own record says do not work, and that the
confidence it prints is a number it has actually earned.
"""

from __future__ import annotations

import json
import logging
import math
import os
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

log = logging.getLogger("drosophila.calibration")

#: Crowd-emotion votes count at half weight (Round P: "remove weightage of emotions to half").
EMOTION_VOTE_SCALE = 0.5
PRIOR_STRENGTH = 4.0          # Beta(a, a): four pseudo-observations at 50 %
MAX_ABS_WEIGHT = 1.5          # log-odds clip: ~82 % reliability saturates
BUCKETS = (0.5, 0.55, 0.6, 0.65, 0.7, 0.8, 1.01)


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, "") or default)
    except ValueError:
        return default


@dataclass
class SourceStat:
    hits: float = 0.0
    misses: float = 0.0
    votes: int = 0            # undecayed count, for the UI's "n"
    last_vote: int = 0

    @property
    def n(self) -> float:
        return self.hits + self.misses

    def reliability(self, prior: float = PRIOR_STRENGTH) -> float:
        return (self.hits + prior) / (self.n + 2.0 * prior)

    def weight(self) -> float:
        p = min(1.0 - 1e-6, max(1e-6, self.reliability()))
        return max(-MAX_ABS_WEIGHT, min(MAX_ABS_WEIGHT, math.log(p / (1.0 - p))))

    def verdict(self) -> str:
        """follow / fade / noise - only when the edge is statistically real
        (two standard errors away from a coin toss on the decayed sample)."""
        if self.n < 8:
            return "noise"
        p = self.reliability()
        se = math.sqrt(0.25 / self.n)
        if p - 0.5 > 2.0 * se and self.weight() > 0.1:
            return "follow"
        if 0.5 - p > 2.0 * se and self.weight() < -0.1:
            return "fade"
        return "noise"

    def to_dict(self) -> dict:
        return {"hits": round(self.hits, 3), "misses": round(self.misses, 3),
                "votes": self.votes, "last_vote": self.last_vote}

    @classmethod
    def from_dict(cls, raw: dict) -> "SourceStat":
        return cls(hits=float(raw.get("hits", 0.0)), misses=float(raw.get("misses", 0.0)),
                   votes=int(raw.get("votes", 0)), last_vote=int(raw.get("last_vote", 0)))


@dataclass
class AssetLedger:
    sources: dict[str, SourceStat] = field(default_factory=dict)
    scored: int = 0
    # Calibration buckets over the *ledger's* probability: [hits, total] each.
    buckets: list[list[float]] = field(default_factory=lambda: [[0.0, 0.0] for _ in range(len(BUCKETS) - 1)])
    # The last 200 (p_up, actual_up) pairs, for the Brier score.
    recent: list[tuple[float, int]] = field(default_factory=list)


def _bucket_index(p_side: float) -> int:
    for i in range(len(BUCKETS) - 1):
        if BUCKETS[i] <= p_side < BUCKETS[i + 1]:
            return i
    return len(BUCKETS) - 2


class EvidenceLedger:
    """Per-asset reliability of every directional source (thread-safe)."""

    def __init__(self, path: Path | None = None, *, enabled: bool | None = None,
                 min_samples: int | None = None, half_life: float | None = None) -> None:
        self.path = path
        self.enabled = (os.environ.get("CALIBRATION_ENABLED", "1").strip() != "0") if enabled is None else enabled
        self.min_samples = int(_env_float("CALIBRATION_MIN_SAMPLES", 30)) if min_samples is None else int(min_samples)
        self.half_life = _env_float("CALIBRATION_HALF_LIFE", 120.0) if half_life is None else float(half_life)
        self.decay = 0.5 ** (1.0 / max(1.0, self.half_life))
        self._assets: dict[str, AssetLedger] = {}
        self._pending: dict[tuple[str, int], tuple[dict[str, int], float]] = {}
        self._lock = threading.Lock()
        self.updated_at = 0.0
        if path is not None:
            self._load()

    # ------------------------------------------------------------------ votes
    @staticmethod
    def votes_from(
        *,
        formula_values: dict,
        directional: dict[str, str],
        ccs_value: float,
        agents: dict,
        spec_score: float,
        niv: float = 0.0,
        crowd_tone: float = 0.0,
        candles=None,
        ticks=None,
    ) -> dict[str, int]:
        """Collect this window's directional votes, one per source."""
        votes: dict[str, int] = {}

        def sign(x) -> int:
            try:
                v = float(x)
            except (TypeError, ValueError):
                return 0
            if not math.isfinite(v) or v == 0.0:
                return 0
            return 1 if v > 0 else -1

        for name, category in (directional or {}).items():
            s = sign((formula_values or {}).get(name))
            if s:
                votes[f"f:{name}"] = s
        s = sign(ccs_value)
        if s:
            votes["brain:CCSv2"] = s
        for name, result in (agents or {}).items():
            decision = getattr(result, "decision", None)
            available = getattr(result, "available", True)
            if available and decision in ("BUY", "SELL"):
                votes[f"agent:{name}"] = 1 if decision == "BUY" else -1
        s = sign(spec_score)
        if s:
            votes["spec:fusion"] = s
        s = sign(niv)
        if s:
            votes["news:NIV"] = s
        s = sign(crowd_tone)
        if s:
            votes["crowd:tone"] = s

        # Micro-structure priors.  Signs are the *raw* direction of the
        # quantity; the ledger learns whether to follow or fade each of them.
        try:
            c = np.asarray(candles, dtype=float).reshape(-1) if candles is not None else np.zeros(0)
            c = c[c > 0]
            if c.size >= 2:
                s = sign(c[-1] - c[-2])
                if s:
                    votes["micro:ret_1m"] = s
            if c.size >= 6:
                s = sign(c[-1] - c[-6])
                if s:
                    votes["micro:ret_5m"] = s
        except Exception:  # noqa: BLE001 - priors are optional
            pass
        try:
            t = np.asarray(ticks, dtype=float) if ticks is not None else np.zeros((0, 4))
            if t.ndim == 2 and t.shape[0] >= 20 and t.shape[1] >= 4:
                # side column: +1 buyer-initiated, -1 seller-initiated (0 unknown)
                flow = float(np.sum(t[:, 2] * np.sign(t[:, 3])))
                s = sign(flow)
                if s:
                    votes["micro:order_flow"] = s
        except Exception:  # noqa: BLE001
            pass
        return votes

    # --------------------------------------------------------------- decision
    def evaluate(self, asset: str, votes: dict[str, int]) -> dict:
        """Weighted verdict of the ledger for these votes (never raises)."""
        with self._lock:
            ledger = self._assets.get(asset) or AssetLedger()
            total = 0.0
            rows = []
            significant = 0
            for name, vote in votes.items():
                stat = ledger.sources.get(name)
                if stat is None or stat.n < 10.0:   # a source needs a record before it may pull
                    continue
                w = stat.weight()
                if name.startswith("crowd:"):
                    # Round P: emotions carry half the weight of any other source.
                    w *= EMOTION_VOTE_SCALE
                # Sources whose edge is not statistically real pull at a
                # quarter weight: 20 correlated formulas drifting the same way
                # by chance must not add up to a confident call.
                if stat.verdict() == "noise":
                    w *= 0.25
                else:
                    significant += 1
                contribution = w * vote
                total += contribution
                rows.append((name, vote, stat.reliability(), stat.n, w, contribution))
            # Correlated evidence: shrink the sum by the square root of the
            # number of pulling sources (they are not independent witnesses).
            pulling = sum(1 for r in rows if abs(r[5]) > 1e-9)
            total_shrunk = total / math.sqrt(max(1.0, float(pulling))) if rows else 0.0
            p_raw = 1.0 / (1.0 + math.exp(-total_shrunk)) if rows else 0.5
            side = "BUY" if total > 0 else "SELL" if total < 0 else None
            p_side_raw = max(p_raw, 1.0 - p_raw)
            b = ledger.buckets[_bucket_index(p_side_raw)]
            realised = (b[0] / b[1]) if b[1] >= 5 else None
            # The probability the engine *prints* is what it has earned in
            # this confidence bucket, once the bucket has a record.
            p_side = float(realised) if (realised is not None and b[1] >= 15) else p_side_raw
            p_side = max(0.5, min(0.95, p_side))
            p_up = p_side if p_raw >= 0.5 else 1.0 - p_side
            # Guardrail: the ledger only decides while its own record is at
            # least as good as the spec recipe it replaces.
            ledger_rate, spec_rate, handed_back = self._ledger_vs_spec(ledger)
            active = bool(self.enabled and ledger.scored >= self.min_samples and rows and not handed_back)
            rows.sort(key=lambda r: -abs(r[5]))
            return {
                "active": active,
                "handed_back": handed_back,
                "ledger_hit_rate": None if ledger_rate is None else round(ledger_rate, 4),
                "spec_hit_rate": None if spec_rate is None else round(spec_rate, 4),
                "significant_sources": significant,
                "p_raw": round(p_raw, 4),
                "enabled": self.enabled,
                "scored": ledger.scored,
                "min_samples": self.min_samples,
                "side": side,
                "p_up": round(p_up, 4),
                "p_side": round(p_side, 4),
                "log_odds": round(total_shrunk, 4),
                "score": round(math.tanh(total_shrunk), 4),
                "realised_at_this_confidence": None if realised is None else round(realised, 4),
                "bucket_samples": int(b[1]),
                "for": [self._row(r) for r in rows if r[5] > 0][:6],
                "against": [self._row(r) for r in rows if r[5] < 0][:6],
                "voters": len(rows),
                "rule": (
                    f"side = sign of the reliability-weighted votes once {self.min_samples} windows are scored; "
                    "weight = log-odds of a source's decayed hit rate (a 50 % source weighs 0, a reliably "
                    "wrong one votes against itself)"
                ),
            }

    @staticmethod
    def _ledger_vs_spec(ledger: AssetLedger) -> tuple[float | None, float | None, bool]:
        """Hit rate of the ledger's own calls vs the spec recipe's, both over
        the same recent windows; ``handed_back`` when the ledger is worse."""
        if len(ledger.recent) < 20:
            return None, None, False
        arr = np.asarray(ledger.recent[-120:], dtype=float)
        ledger_rate = float(np.mean((arr[:, 0] >= 0.5) == (arr[:, 1] > 0.5)))
        spec = ledger.sources.get("spec:fusion")
        spec_rate = spec.reliability() if spec is not None and spec.n >= 10 else None
        handed_back = spec_rate is not None and ledger_rate + 0.02 < spec_rate
        return ledger_rate, spec_rate, handed_back

    @staticmethod
    def _row(r) -> dict:
        name, vote, rel, n, w, contribution = r
        return {"source": name, "vote": "up" if vote > 0 else "down", "reliability": round(rel, 3),
                "n": round(n, 1), "weight": round(w, 3), "contribution": round(contribution, 3)}

    # --------------------------------------------------------------- learning
    def remember(self, asset: str, cycle_number: int, votes: dict[str, int], p_up: float) -> None:
        with self._lock:
            self._pending[(asset, int(cycle_number))] = (dict(votes), float(p_up))
            if len(self._pending) > 64:
                oldest = sorted(self._pending)[0]
                self._pending.pop(oldest, None)

    def score(self, asset: str, cycle_number: int, actual_up: bool | None) -> dict | None:
        """Credit every source that voted on this window.  ``None`` = flat."""
        with self._lock:
            entry = self._pending.pop((asset, int(cycle_number)), None)
            if entry is None or actual_up is None:
                return None
            votes, p_up = entry
            ledger = self._assets.setdefault(asset, AssetLedger())
            truth = 1 if actual_up else -1
            for stat in ledger.sources.values():
                stat.hits *= self.decay
                stat.misses *= self.decay
            hits = misses = 0
            for name, vote in votes.items():
                if not vote:
                    continue
                stat = ledger.sources.setdefault(name, SourceStat())
                stat.votes += 1
                stat.last_vote = vote
                if vote == truth:
                    stat.hits += 1.0
                    hits += 1
                else:
                    stat.misses += 1.0
                    misses += 1
            ledger.scored += 1
            p_side = max(p_up, 1.0 - p_up)
            ledger_side_up = p_up >= 0.5
            b = ledger.buckets[_bucket_index(p_side)]
            b[1] += 1.0
            if ledger_side_up == actual_up:
                b[0] += 1.0
            ledger.recent.append((p_up, 1 if actual_up else 0))
            del ledger.recent[:-200]
            self.updated_at = time.time()
            snapshot = {"asset": asset, "cycle_number": cycle_number, "actual": "up" if actual_up else "down",
                        "sources_right": hits, "sources_wrong": misses, "scored": ledger.scored}
        self._save()
        return snapshot

    # ---------------------------------------------------------------- reports
    def report(self, asset: str | None = None) -> dict:
        with self._lock:
            assets = [asset] if asset else sorted(self._assets)
            out = {"enabled": self.enabled, "min_samples": self.min_samples, "half_life_windows": self.half_life,
                   "path": str(self.path) if self.path else None, "assets": {}}
            for a in assets:
                ledger = self._assets.get(a) or AssetLedger()
                rows = []
                for name, stat in ledger.sources.items():
                    rows.append({"source": name, "reliability": round(stat.reliability(), 3),
                                 "weight": round(stat.weight(), 3), "n": round(stat.n, 1), "votes": stat.votes,
                                 "last_vote": stat.last_vote, "verdict": stat.verdict()})
                rows.sort(key=lambda r: -abs(r["weight"]))
                brier = None
                if ledger.recent:
                    arr = np.asarray(ledger.recent, dtype=float)
                    brier = float(np.mean((arr[:, 0] - arr[:, 1]) ** 2))
                cal = []
                for i in range(len(BUCKETS) - 1):
                    h, n = ledger.buckets[i]
                    cal.append({"claimed": f"{int(BUCKETS[i]*100)}-{min(100, int(BUCKETS[i+1]*100))}%",
                                "windows": int(n), "realised": None if n < 1 else round(h / n, 3)})
                ledger_rate, spec_rate, handed_back = self._ledger_vs_spec(ledger)
                out["assets"][a] = {"scored": ledger.scored,
                                    "active": bool(self.enabled and ledger.scored >= self.min_samples and not handed_back),
                                    "handed_back_to_spec": handed_back,
                                    "ledger_hit_rate": None if ledger_rate is None else round(ledger_rate, 4),
                                    "spec_hit_rate": None if spec_rate is None else round(spec_rate, 4),
                                    "sources": rows, "calibration": cal,
                                    "brier": None if brier is None else round(brier, 4),
                                    "pending": sum(1 for k in self._pending if k[0] == a)}
            return out

    # ------------------------------------------------------------ persistence
    def _save(self) -> None:
        if self.path is None:
            return
        try:
            with self._lock:
                payload = {
                    "version": 1, "updated_at": self.updated_at,
                    "assets": {
                        a: {"scored": l.scored, "buckets": l.buckets, "recent": l.recent[-200:],
                            "sources": {n: s.to_dict() for n, s in l.sources.items()}}
                        for a, l in self._assets.items()
                    },
                }
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(".tmp")
            tmp.write_text(json.dumps(payload), encoding="utf-8")
            os.replace(tmp, self.path)
        except Exception as exc:  # noqa: BLE001 - persistence must never break a cycle
            log.debug("calibration save failed: %s", exc)

    def _load(self) -> None:
        try:
            if self.path is None or not self.path.is_file():
                return
            raw = json.loads(self.path.read_text(encoding="utf-8"))
            for a, l in (raw.get("assets") or {}).items():
                ledger = AssetLedger(scored=int(l.get("scored", 0)))
                buckets = l.get("buckets")
                if isinstance(buckets, list) and len(buckets) == len(BUCKETS) - 1:
                    ledger.buckets = [[float(x[0]), float(x[1])] for x in buckets]
                ledger.recent = [(float(p), int(y)) for p, y in (l.get("recent") or [])][-200:]
                ledger.sources = {n: SourceStat.from_dict(s) for n, s in (l.get("sources") or {}).items()}
                self._assets[a] = ledger
            self.updated_at = float(raw.get("updated_at", 0.0))
            log.info("calibration ledger loaded: %s", {a: l.scored for a, l in self._assets.items()})
        except Exception as exc:  # noqa: BLE001
            log.warning("calibration ledger unreadable (%s) - starting fresh", exc)
