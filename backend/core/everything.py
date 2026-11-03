"""Round BU - the Theory of Nearly Everything: one evidence ledger for every
source the app has.

Honest framing first: no physical theory predicts a market.  What *can* be
done rigorously is to treat every source the app already computes - the 31
formulas, the physics gauges, the emotions, the AI agents, the brain, the
regime features, the four physics estimators - as **evidence** about the
same binary question (does Binance BTC/USDT print higher at :70 than at :00?)
and combine that evidence the way evidence is combined: additively, in
log-odds (nats), each source weighted by how much it has actually told us,
with the double counting of correlated sources removed.

    e_i  = a_i · z_i                     per-source evidence in nats
    a_i  = 1-D ridge-logistic slope of outcome on the standardised source z_i,
           fitted walk-forward on the settled windows (exp-weighted, half-life
           120 windows); a source that has told us nothing gets a_i ≈ 0
    L    = (Σ_i e_i) · √(k_eff / k)       correlation shrinkage
    k_eff = k / (1 + (k − 1) ρ̄)           effective number of independent
                                          sources, ρ̄ = mean |pairwise corr|
    P(up) = σ(b · L)                      b = Platt slope on the graded record
    predictability = 1 − H₂(P)            bits the forecast carries about the
                                          sign (0 = coin flip, 1 = certain)
    information_i = I(sign z_i ; outcome) bits each source has actually
                                          carried, from decayed sign counts

It takes the side only while its own strict-sign walk-forward record earns
it - the same rule the meta-combiner lives under (Round BK): trailing ≥ 53 %
over 40 graded windows, not below the recipe and not below the
meta-combiner.  Everything is printed either way, with the per-source
evidence, the shrinkage, the predictability gauge and the record.
"""
from __future__ import annotations

import math
import time
from collections import deque

import numpy as np

HALF_LIFE = 120.0
LAMBDA = 2.0
MIN_FIT = 40
WINDOW = 40
TAKEOVER_RATE = 0.53
MIN_GRADED = 30
BLOCK = 20
CALIB_MIN = 40
MIN_PRESENCE = 0.6
INFO_DECAY = 0.99          # per settled window, for the per-source sign counts
ROUTING_PREFIX = "regime:"


def _sigmoid(z: float) -> float:
    z = max(-40.0, min(40.0, float(z)))
    return 1.0 / (1.0 + math.exp(-z))


def _h2(p: float) -> float:
    p = min(1.0 - 1e-12, max(1e-12, float(p)))
    return float(-(p * math.log2(p) + (1 - p) * math.log2(1 - p)))


class Fit:
    __slots__ = ("names", "slopes", "centre", "scale", "rho_bar", "rows", "at", "t_stats")

    def __init__(self, names, slopes, centre, scale, rho_bar, rows, t_stats):
        self.names = list(names)
        self.slopes = np.asarray(slopes, dtype=float)
        self.centre = np.asarray(centre, dtype=float)
        self.scale = np.asarray(scale, dtype=float)
        self.rho_bar = float(rho_bar)
        self.rows = int(rows)
        self.at = time.time()
        self.t_stats = np.asarray(t_stats, dtype=float)

    def to_dict(self) -> dict:
        return {"names": self.names, "slopes": self.slopes.tolist(), "centre": self.centre.tolist(),
                "scale": self.scale.tolist(), "rho_bar": self.rho_bar, "rows": self.rows, "at": self.at,
                "t_stats": self.t_stats.tolist()}

    @classmethod
    def from_dict(cls, d: dict) -> "Fit":
        obj = cls(d["names"], d["slopes"], d["centre"], d["scale"], d.get("rho_bar", 0.0), d.get("rows", 0),
                  d.get("t_stats", [0.0] * len(d["names"])))
        obj.at = float(d.get("at", time.time()))
        return obj


class UnifiedEvidence:
    def __init__(self) -> None:
        self.fit: Fit | None = None
        self.pending: dict[int, dict] = {}
        self.graded: deque = deque(maxlen=600)
        self.recipe_graded: deque = deque(maxlen=600)
        self.calib_slope = 1.0
        self.info_counts: dict[str, np.ndarray] = {}     # name -> 2x2 decayed counts [sign][outcome]
        self.last: dict = {}

    # ------------------------------------------------------------------
    # fitting
    # ------------------------------------------------------------------
    @staticmethod
    def _slope(z: np.ndarray, y: np.ndarray, w: np.ndarray, lam: float = LAMBDA) -> tuple[float, float]:
        """1-D ridge logistic slope (no intercept: the question is symmetric) by Newton."""
        a = 0.0
        n = float(w.sum()) or 1.0
        for _ in range(12):
            p = 1.0 / (1.0 + np.exp(-np.clip(a * z, -40, 40)))
            t = (y + 1.0) / 2.0
            grad = float(np.sum(w * (t - p) * z)) - lam * a
            hess = float(np.sum(w * p * (1 - p) * z * z)) + lam
            step = grad / max(hess, 1e-9)
            a += step
            if abs(step) < 1e-7:
                break
        p = 1.0 / (1.0 + np.exp(-np.clip(a * z, -40, 40)))
        se = 1.0 / math.sqrt(max(float(np.sum(w * p * (1 - p) * z * z)) * (n / n), 1e-9))
        return float(a), float(a / se)

    def refit(self, rows: list, now: float | None = None) -> Fit | None:
        rows = [r for r in rows if getattr(r, "decided", False)]
        if len(rows) < MIN_FIT:
            return self.fit
        rows.sort(key=lambda r: r.at)
        now = now if now is not None else rows[-1].at
        presence: dict[str, int] = {}
        for r in rows:
            for k, v in r.inputs.items():
                if k.startswith(ROUTING_PREFIX) or v is None:
                    continue
                presence[k] = presence.get(k, 0) + 1
        names = sorted(k for k, c in presence.items() if c >= MIN_PRESENCE * len(rows))
        if not names:
            return self.fit
        X = np.array([[float(r.inputs.get(k, 0.0) or 0.0) for k in names] for r in rows])
        y = np.array([1.0 if r.move_bps > 0 else -1.0 for r in rows])
        age = np.array([max(0.0, (now - r.at)) for r in rows])
        # ages are in seconds; convert to windows via the median spacing
        spacing = float(np.median(np.diff([r.at for r in rows]))) if len(rows) > 2 else 70.0
        w = 0.5 ** (age / max(1e-9, HALF_LIFE * max(spacing, 1e-9)))
        centre = np.median(X, axis=0)
        scale = np.maximum(1.4826 * np.median(np.abs(X - centre), axis=0), 1e-9)
        Z = np.clip((X - centre) / scale, -4, 4)
        slopes = np.zeros(len(names))
        t_stats = np.zeros(len(names))
        for j in range(len(names)):
            slopes[j], t_stats[j] = self._slope(Z[:, j], y, w)
        # correlation among the sources that carry evidence
        keep = np.abs(slopes) > 1e-6
        rho_bar = 0.0
        if keep.sum() > 1:
            C = np.corrcoef(Z[:, keep].T)
            C = np.nan_to_num(C)
            k = C.shape[0]
            rho_bar = float((np.abs(C).sum() - k) / (k * (k - 1)))
        self.fit = Fit(names, slopes, centre, scale, rho_bar, len(rows), t_stats)
        self._fit_calibration()
        return self.fit

    def _fit_calibration(self) -> None:
        rows = [g for g in self.graded if "L" in g]
        if len(rows) < CALIB_MIN:
            self.calib_slope = 1.0
            return
        L = np.array([g["L"] for g in rows[-300:]])
        y = np.array([1.0 if g["up"] else -1.0 for g in rows[-300:]])
        b, _ = self._slope(L, y, np.ones_like(L), lam=0.5)
        self.calib_slope = float(min(5.0, max(0.05, b)))

    # ------------------------------------------------------------------
    # prediction
    # ------------------------------------------------------------------
    def information(self, name: str) -> float:
        c = self.info_counts.get(name)
        if c is None or c.sum() < 8:
            return 0.0
        p = c / c.sum()
        ps = p.sum(axis=1, keepdims=True)
        po = p.sum(axis=0, keepdims=True)
        with np.errstate(divide="ignore", invalid="ignore"):
            terms = np.where(p > 0, p * np.log2(p / (ps @ po)), 0.0)
        return float(max(0.0, terms.sum()))

    def predict(self, inputs: dict, fit: Fit | None = None) -> dict:
        fit = fit or self.fit
        if fit is None:
            return {"available": False, "why": f"needs {MIN_FIT} decided windows before the first fit",
                    "sources": [], "p_up": 0.5, "predictability": 0.0}
        contributions = []
        total = 0.0
        k = 0
        for j, name in enumerate(fit.names):
            v = inputs.get(name)
            if v is None or fit.slopes[j] == 0.0:
                continue
            z = float(np.clip((float(v) - fit.centre[j]) / fit.scale[j], -4, 4))
            e = float(fit.slopes[j] * z)
            total += e
            k += 1
            contributions.append({"name": name, "z": round(z, 3), "slope": round(float(fit.slopes[j]), 4),
                                  "t": round(float(fit.t_stats[j]), 2), "evidence_nats": round(e, 4),
                                  "bits": round(self.information(name), 4)})
        if k == 0:
            return {"available": False, "why": "none of the fitted sources is present", "sources": [], "p_up": 0.5,
                    "predictability": 0.0}
        k_eff = k / (1.0 + (k - 1) * fit.rho_bar) if k > 1 else 1.0
        shrink = math.sqrt(k_eff / k)
        L = total * shrink
        p_up = _sigmoid(self.calib_slope * L)
        side = "BUY" if p_up >= 0.5 else "SELL"
        contributions.sort(key=lambda c: -abs(c["evidence_nats"]))
        bits_total = float(sum(c["bits"] for c in contributions))
        return {
            "available": True, "side": side, "p_up": round(p_up, 4), "confidence": round(abs(2 * p_up - 1), 4),
            "L": round(L, 4), "raw_nats": round(total, 4), "sources_used": k, "k_eff": round(k_eff, 2),
            "rho_bar": round(fit.rho_bar, 3), "shrink": round(shrink, 3), "calib_slope": round(self.calib_slope, 3),
            "predictability": round(1.0 - _h2(p_up), 4), "entropy_bits": round(_h2(p_up), 4),
            "information_bits": round(bits_total, 4), "sources": contributions[:40],
            "for_nats": round(sum(c["evidence_nats"] for c in contributions if c["evidence_nats"] > 0), 3),
            "against_nats": round(sum(c["evidence_nats"] for c in contributions if c["evidence_nats"] < 0), 3),
        }

    # ------------------------------------------------------------------
    # record keeping
    # ------------------------------------------------------------------
    def remember(self, cycle: int, prediction: dict, recipe_side: str | None, inputs: dict | None = None) -> None:
        signs = {}
        for k, v in (inputs or {}).items():
            if k.startswith(ROUTING_PREFIX) or v is None:
                continue
            try:
                fv = float(v)
            except (TypeError, ValueError):
                continue
            if fv != 0.0:
                signs[k] = 1 if fv > 0 else 0
        rec = {"recipe_side": recipe_side, "at": time.time(), "signs": signs}
        if prediction.get("available"):
            rec.update({"p_up": prediction["p_up"], "side": prediction["side"], "L": prediction["L"]})
        self.pending[int(cycle)] = rec
        if len(self.pending) > 200:
            for k in sorted(self.pending)[:-200]:
                self.pending.pop(k, None)

    def settle(self, cycle: int, move_bps: float, cost_bps: float = 0.0) -> dict | None:
        rec = self.pending.pop(int(cycle), None)
        if rec is None or abs(move_bps) <= cost_bps:
            return None
        up = move_bps > 0
        for name, c in self.info_counts.items():
            c *= INFO_DECAY
        for name, sgn in rec.get("signs", {}).items():
            c = self.info_counts.setdefault(name, np.zeros((2, 2)))
            c[sgn, 1 if up else 0] += 1.0
        if len(self.info_counts) > 400:
            for name in sorted(self.info_counts, key=lambda n: self.info_counts[n].sum())[:50]:
                self.info_counts.pop(name, None)
        if rec.get("recipe_side") in ("BUY", "SELL"):
            self.recipe_graded.append(1 if (rec["recipe_side"] == "BUY") == up else 0)
        if "side" not in rec:
            return None
        hit = (rec["side"] == "BUY") == up
        self.graded.append({"cycle": int(cycle), "hit": bool(hit), "up": bool(up), "p_up": rec["p_up"], "L": rec["L"],
                            "move_bps": round(float(move_bps), 2), "at": time.time()})
        return self.graded[-1]

    def build_record(self, rows: list) -> int:
        rows = sorted((r for r in rows if r.decided), key=lambda r: r.at)
        if len(rows) < MIN_FIT + BLOCK:
            self.refit(rows)
            return 0
        graded = 0
        for start in range(MIN_FIT, len(rows), BLOCK):
            fit = self.refit(rows[:start])
            if fit is None:
                continue
            for r in rows[start:start + BLOCK]:
                pred = self.predict(r.inputs, fit)
                recipe = r.inputs.get("fusion:recipe")
                recipe_side = None if recipe is None or recipe == 0 else ("BUY" if recipe > 0 else "SELL")
                self.remember(r.cycle, pred, recipe_side, r.inputs)
                self.pending[r.cycle]["at"] = r.at
                if self.settle(r.cycle, float(r.move_bps), float(r.cost_bps)) is not None:
                    graded += 1
        self.refit(rows)
        return graded

    # ------------------------------------------------------------------
    # verdict
    # ------------------------------------------------------------------
    def trailing(self, n: int = WINDOW) -> tuple[float | None, int]:
        rows = list(self.graded)[-n:]
        if not rows:
            return None, 0
        return sum(1 for r in rows if r["hit"]) / len(rows), len(rows)

    def recipe_trailing(self, n: int = WINDOW) -> tuple[float | None, int]:
        rows = list(self.recipe_graded)[-n:]
        if not rows:
            return None, 0
        return sum(rows) / len(rows), len(rows)

    def may_decide(self, meta_rate: float | None = None) -> tuple[bool, str]:
        rate, n = self.trailing()
        r_rate, _ = self.recipe_trailing()
        if self.fit is None:
            return False, f"no fit yet (needs {MIN_FIT} decided windows)"
        if n < MIN_GRADED:
            return False, f"walk-forward record {n}/{MIN_GRADED} graded windows - watching"
        if rate is None or rate < TAKEOVER_RATE:
            return False, f"trailing {rate:.0%} over {n} < {TAKEOVER_RATE:.0%} - watching"
        if r_rate is not None and rate < r_rate:
            return False, f"trailing {rate:.0%} < recipe {r_rate:.0%} over the same windows - the recipe keeps the side"
        if meta_rate is not None and rate < meta_rate:
            return False, f"trailing {rate:.0%} < meta-combiner {meta_rate:.0%} - the meta-combiner keeps the side"
        return True, f"trailing {rate:.0%} over {n} graded windows - the unified ledger picks the side"

    def reliability(self, bins: int = 5) -> list[dict]:
        rows = list(self.graded)[-300:]
        out = []
        for i in range(bins):
            lo, hi = i / bins, (i + 1) / bins
            sel = [g for g in rows if lo <= g["p_up"] < hi or (i == bins - 1 and g["p_up"] == 1.0)]
            if sel:
                out.append({"bin": f"{lo:.1f}–{hi:.1f}", "n": len(sel),
                            "forecast": round(sum(g["p_up"] for g in sel) / len(sel), 3),
                            "observed": round(sum(1 for g in sel if g["up"]) / len(sel), 3)})
        return out

    def to_dict(self) -> dict:
        rate, n = self.trailing()
        r_rate, _ = self.recipe_trailing()
        decide, why = self.may_decide()
        last = dict(self.last or {})
        top_info = sorted(((k, self.information(k)) for k in self.info_counts), key=lambda kv: -kv[1])[:12]
        return {
            "name": "Theory of Nearly Everything - unified evidence ledger",
            "fitted": self.fit is not None,
            "fit_rows": self.fit.rows if self.fit else 0,
            "sources_fitted": len(self.fit.names) if self.fit else 0,
            "rho_bar": round(self.fit.rho_bar, 3) if self.fit else None,
            "calib_slope": round(self.calib_slope, 3),
            "trailing_rate": None if rate is None else round(rate, 4), "trailing_n": n,
            "recipe_rate": None if r_rate is None else round(r_rate, 4),
            "graded": len(self.graded), "decides": decide, "why": why,
            "last": last,
            "information": [{"name": k, "bits": round(b, 4)} for k, b in top_info],
            "reliability": self.reliability(),
            "rule": (f"evidence e_i = a_i z_i (nats) · L = Σe_i · √(k_eff/k), k_eff = k/(1+(k−1)ρ̄) · P(up) = σ(b·L) · "
                     f"decides only with a {WINDOW}-window strict-sign record ≥ {TAKEOVER_RATE:.0%}, ≥ recipe and ≥ meta-combiner"),
            "honest": "no physical theory predicts a market; this is every source the app has, weighed as evidence by its own record",
        }

    def export_state(self) -> dict:
        return {"fit": self.fit.to_dict() if self.fit else None, "graded": list(self.graded),
                "recipe_graded": list(self.recipe_graded), "calib_slope": self.calib_slope,
                "info_counts": {k: v.tolist() for k, v in self.info_counts.items()}}

    def import_state(self, payload: dict) -> int:
        try:
            if payload.get("fit"):
                self.fit = Fit.from_dict(payload["fit"])
            self.graded = deque(payload.get("graded") or [], maxlen=600)
            self.recipe_graded = deque(payload.get("recipe_graded") or [], maxlen=600)
            self.calib_slope = float(payload.get("calib_slope", 1.0))
            self.info_counts = {k: np.asarray(v, dtype=float) for k, v in (payload.get("info_counts") or {}).items()}
            return len(self.graded)
        except Exception:  # noqa: BLE001
            return 0
