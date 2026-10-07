"""Three minds, one analyst (Round AR).

Every window three very different intelligences give an opinion:

* **HUMAN** - the crowd: the emotion layer's tone (fear / greed on the tape),
  the social posts and the news sentiment.  This is what human traders feel.
* **AI** - the language-model agents (Gemini, GitHub Models, the local model):
  what an AI reasons from the same evidence.
* **DATA** - the fly brain and the formulas: the mushroom-body verdict (CCSv2),
  the formula consensus and the physics layer.  Pure measurement.

A good human analyst does not add these up with fixed weights.  She keeps
score of *who has been right lately*, notices *when they agree and when they
split*, remembers that the crowd is often right in a trend and wrong at the
extremes, and knows that when all three say the same thing the call is
better than any one of them.  That is this module:

* per mind: rolling hit rate (shrunk to 0.5, Wilson-style), correlation of
  its vote with the realised move, and hit rate *conditional on the crowd's
  dominant emotion* (regime-aware trust);
* per pair and for the triad: how often they agree and how good the call is
  when they do;
* a contrarian switch: a mind whose record is significantly *below* a coin
  flip (lower Wilson bound < 0.5, n >= 15) is read inverted - the fading of a
  crowd that is reliably wrong at extremes;
* the composite: reliability-weighted vote, corroboration bonus, and a
  confidence multiplier the fusion applies - plus the sentence that explains it.

The ledger (``calibration.EvidenceLedger``) learns individual sources; this
layer learns the *relationship between the three kinds of mind*.
"""

from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import dataclass, field

import numpy as np

MINDS = ("human", "ai", "data")
LABELS = {"human": "HUMAN (crowd)", "ai": "AI (agents)", "data": "DATA (fly brain + formulas)"}
WINDOW = 120
MIN_N = 8            # before this a mind's weight is its prior
CONTRARIAN_N = 15
SHRINK = 12.0
Z = 1.96


def _wilson_low(k: float, n: float) -> float:
    if n <= 0:
        return 0.0
    p = k / n
    den = 1.0 + Z * Z / n
    centre = p + Z * Z / (2 * n)
    adj = Z * math.sqrt(p * (1 - p) / n + Z * Z / (4 * n * n))
    return (centre - adj) / den


@dataclass
class Record:
    cycle: int
    asset: str
    votes: dict                   # mind -> vote in [-1, 1]
    emotion: str
    published: str
    at: float
    actual: float | None = None   # +1 up / -1 down / 0 flat
    move_bps: float = 0.0


@dataclass
class TriuneAnalyst:
    records: deque = field(default_factory=lambda: deque(maxlen=WINDOW))
    pending: dict = field(default_factory=dict)      # cycle -> Record
    scored: int = 0

    # ------------------------------------------------------------------
    @staticmethod
    def minds_from(*, crowd_tone: float, social_sentiment: float, news_niv: float,
                   agents: dict, ccs_value: float, formula_consensus: float, physics_vote: float) -> dict:
        """The three opinions for this window, each in [-1, 1]."""
        human_parts = [(0.6, crowd_tone), (0.2, social_sentiment), (0.2, news_niv)]
        human = sum(w * float(np.clip(v, -1, 1)) for w, v in human_parts)
        ai_votes = []
        for res in (agents or {}).values():
            try:
                if getattr(res, "ok", None) is False or (isinstance(res, dict) and res.get("ok") is False):
                    continue
                dec = getattr(res, "decision", None) if not isinstance(res, dict) else res.get("decision")
                conf = getattr(res, "confidence", None) if not isinstance(res, dict) else res.get("confidence")
                if dec in ("BUY", "SELL"):
                    ai_votes.append((1.0 if dec == "BUY" else -1.0) * float(conf if conf is not None else 0.5))
            except Exception:  # noqa: BLE001
                continue
        ai = float(np.mean(ai_votes)) if ai_votes else 0.0
        data = 0.5 * float(np.clip(ccs_value, -1, 1)) + 0.3 * float(np.clip(formula_consensus, -1, 1)) \
            + 0.2 * float(np.clip(physics_vote, -1, 1))
        return {"human": round(float(np.clip(human, -1, 1)), 4), "ai": round(float(np.clip(ai, -1, 1)), 4),
                "data": round(float(np.clip(data, -1, 1)), 4), "ai_answered": len(ai_votes)}

    def remember(self, cycle: int, asset: str, votes: dict, emotion: str, published: str) -> None:
        self.pending[int(cycle)] = Record(int(cycle), asset, {m: float(votes.get(m, 0.0)) for m in MINDS},
                                          str(emotion or ""), str(published or ""), time.time())
        if len(self.pending) > 200:
            for key in sorted(self.pending)[:-100]:
                self.pending.pop(key, None)

    def score(self, cycle: int, actual_up: bool | None, move_bps: float) -> dict | None:
        rec = self.pending.pop(int(cycle), None)
        if rec is None:
            return None
        rec.actual = 0.0 if actual_up is None else (1.0 if actual_up else -1.0)
        rec.move_bps = float(move_bps)
        self.records.append(rec)
        self.scored += 1
        return {"cycle": rec.cycle, "actual": rec.actual, "votes": rec.votes}

    # ------------------------------------------------------------------
    def _decided(self) -> list[Record]:
        return [r for r in self.records if r.actual not in (None, 0.0)]

    def _mind_stats(self, rows: list[Record]) -> dict:
        out = {}
        for m in MINDS:
            voted = [r for r in rows if abs(r.votes.get(m, 0.0)) > 0.02]
            n = len(voted)
            hits = sum(1 for r in voted if math.copysign(1, r.votes[m]) == r.actual)
            hit_rate = hits / n if n else None
            shrunk = 0.5 + (n / (n + SHRINK)) * ((hit_rate or 0.5) - 0.5) if n else 0.5
            corr = None
            if n >= 5:
                v = np.array([r.votes[m] for r in voted])
                mv = np.array([r.move_bps for r in voted])
                if v.std() > 1e-9 and mv.std() > 1e-9:
                    corr = float(np.corrcoef(v, mv)[0, 1])
            low = _wilson_low(hits, n) if n else 0.0
            contrarian = n >= CONTRARIAN_N and hit_rate is not None and _wilson_low(n - hits, n) > 0.5
            by_emotion: dict[str, dict] = {}
            for r in voted:
                e = r.emotion or "—"
                d = by_emotion.setdefault(e, {"n": 0, "hits": 0})
                d["n"] += 1
                d["hits"] += int(math.copysign(1, r.votes[m]) == r.actual)
            out[m] = {"n": n, "hits": hits, "hit_rate": None if hit_rate is None else round(hit_rate, 3),
                      "shrunk": round(shrunk, 3), "wilson_low": round(low, 3), "corr_with_move": None if corr is None else round(corr, 3),
                      "contrarian": bool(contrarian),
                      "by_emotion": {e: {"n": d["n"], "hit_rate": round(d["hits"] / d["n"], 2)} for e, d in by_emotion.items() if d["n"] >= 3}}
        return out

    def _pair_stats(self, rows: list[Record]) -> dict:
        out = {}
        pairs = [("human", "ai"), ("human", "data"), ("ai", "data"), ("human", "ai", "data")]
        for pair in pairs:
            agree = [r for r in rows if all(abs(r.votes.get(m, 0.0)) > 0.02 for m in pair)
                     and len({math.copysign(1, r.votes[m]) for m in pair}) == 1]
            n = len(agree)
            hits = sum(1 for r in agree if math.copysign(1, r.votes[pair[0]]) == r.actual)
            both = [r for r in rows if all(abs(r.votes.get(m, 0.0)) > 0.02 for m in pair)]
            out["+".join(pair)] = {"agreements": n, "of": len(both), "hit_rate_when_agree": round(hits / n, 3) if n else None}
        return out

    def analyse(self, votes: dict, emotion: str = "") -> dict:
        """The analyst's verdict for the current window."""
        rows = self._decided()
        stats = self._mind_stats(rows)
        pairs = self._pair_stats(rows)
        weights, used, notes = {}, {}, []
        for m in MINDS:
            s = stats[m]
            v = float(votes.get(m, 0.0))
            base = s["shrunk"] if s["n"] >= MIN_N else 0.5
            # regime-aware: trust in this crowd state, if there is a record of it
            reg = (s["by_emotion"] or {}).get(emotion or "—")
            if reg and reg["n"] >= 5:
                base = 0.5 * base + 0.5 * (0.5 + (reg["n"] / (reg["n"] + SHRINK)) * (reg["hit_rate"] - 0.5))
            w = max(0.0, 2.0 * (base - 0.5))     # 0 at coin flip, 1 at a perfect record
            w = 0.15 + 0.85 * w                  # every mind keeps a voice
            if s["contrarian"]:
                v = -v
                notes.append(f"{LABELS[m]} is read inverted: {s['hit_rate']:.0%} over {s['n']} windows (significantly below a coin flip)")
            weights[m] = round(w, 3)
            used[m] = round(v, 4)
        total_w = sum(weights.values()) or 1.0
        composite = sum(weights[m] * used[m] for m in MINDS) / total_w
        sides = {math.copysign(1, used[m]) for m in MINDS if abs(used[m]) > 0.05}
        speaking = [m for m in MINDS if abs(used[m]) > 0.05]
        corroboration = 1.0 if len(speaking) >= 2 and len(sides) == 1 else 0.0 if len(sides) > 1 else 0.5
        triad = pairs.get("human+ai+data") or {}
        triad_hr = triad.get("hit_rate_when_agree")
        # confidence multiplier: agreement with a good record lifts, a split lowers
        if corroboration == 1.0:
            mult = 1.10 if (triad_hr is None or triad_hr >= 0.5) else 0.95
        elif corroboration == 0.0:
            mult = 0.85
        else:
            mult = 1.0
        maturity = min(1.0, len(rows) / 30.0)
        side = "BUY" if composite > 0 else "SELL" if composite < 0 else "—"
        best = max(MINDS, key=lambda m: stats[m]["shrunk"])
        story = (f"{LABELS['human']} {used['human']:+.2f} · {LABELS['ai']} {used['ai']:+.2f} · {LABELS['data']} {used['data']:+.2f} → "
                 + ("all three agree" if corroboration == 1.0 and len(speaking) == 3 else
                    "two agree" if corroboration == 1.0 else "they split" if corroboration == 0.0 else "only one is speaking")
                 + f"; composite {composite:+.2f} ({side})"
                 + (f"; most reliable lately: {LABELS[best]} {stats[best]['hit_rate']:.0%} of {stats[best]['n']}" if stats[best]["n"] >= MIN_N else
                    f"; {len(rows)} decided windows scored so far - weights are still priors")
                 + (f"; when all three agreed the call was right {triad_hr:.0%} of {triad['agreements']}" if triad_hr is not None else ""))
        return {
            "votes": {m: round(float(votes.get(m, 0.0)), 4) for m in MINDS}, "used": used, "weights": weights,
            "composite": round(float(composite), 4), "side": side, "corroboration": corroboration,
            "confidence_multiplier": mult, "maturity": round(maturity, 3), "scored": len(rows),
            "minds": stats, "pairs": pairs, "notes": notes, "story": story, "emotion": emotion,
            "ai_answered": int(votes.get("ai_answered", 0) or 0),
        }

    def report(self) -> dict:
        rows = self._decided()
        return {"scored": len(rows), "pending": len(self.pending), "minds": self._mind_stats(rows), "pairs": self._pair_stats(rows),
                "recent": [{"cycle": r.cycle, "votes": r.votes, "actual": r.actual, "move_bps": round(r.move_bps, 2), "emotion": r.emotion}
                           for r in list(self.records)[-15:]]}
