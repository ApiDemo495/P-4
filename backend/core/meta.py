"""Meta-combiner - one model learns, from the record, how to weigh everything.

Round BK ("still lacking" - the win rate).  Every layer so far is combined
by a hand-written recipe (0.40 / 0.25 / 0.20 / 0.15, dampeners, vetoes) that
the ledger and the hindsight review then *adjust*.  This module replaces the
hand-writing with the oldest honest tool for the job: a regularised
logistic regression over **all** the recorded inputs of every settled
window - formulas, hedge, physics, emotions, features, reasoning, agents,
genesis - fitted on the engine's own record (opening audit + warm-start
replay + live windows), with recent windows weighted more.

It is scored **walk-forward**: the prediction for a window is stored at the
lock, graded at settlement, and only that graded record decides whether the
model may pick the side.  It takes the side only while its trailing record
(last ``WINDOW`` decided windows) is at least as good as the recipe's and
above ``TAKEOVER_RATE``; otherwise it watches and prints what it would have
said.  When it decides, the panel shows the inputs that drove the call, with
their learned weights.

Mathematics
-----------
    x  = standardised inputs (trailing mean / std per input, missing -> 0)
    p  = sigma(w . x + b)
    w  = argmin  sum_i a_i * logloss(y_i, p_i) + lambda * |w|^2
         (a_i = 0.5 ** (age_i / HALF_LIFE windows), IRLS / Newton steps)
    side = BUY if p > 0.5 else SELL,  confidence = |2p - 1|
"""
from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import dataclass

import numpy as np

HALF_LIFE = 120.0          # windows
LAMBDA = 3.0               # ridge strength (inputs are standardised)
MIN_FIT = 40               # decided windows before the first fit
WINDOW = 40                # trailing windows the takeover test looks at
TAKEOVER_RATE = 0.53       # must beat this - and the recipe - to decide
MIN_GRADED = 30            # graded walk-forward predictions before it may decide
REFIT_EVERY = 1            # windows between refits (live)
BLOCK = 20                 # walk-forward block size when building the start record
#: Round BN: regime experts - one extra fit per change-point regime, used
#: only where its own walk-forward record in that regime beats the global fit
REGIME_KEY = "regime:bocpd"
REGIMES = ("fresh", "young", "mature", "old")
#: Round BO: every routing key the experts may specialise on.  A label is
#: "<key>=<name>"; an input named ``regime:*`` is a routing key, never a feature.
ROUTES: dict[str, tuple[str, ...]] = {
    REGIME_KEY: REGIMES,
    "regime:session": ("asia", "europe", "us", "late-us"),
    "regime:vol": ("calm", "normal", "violent"),
}
EXPERT_MIN_FIT = 40
EXPERT_MIN_GRADED = 15
CALIB_MIN = 40             # graded windows before the probability is calibrated


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30.0, 30.0)))


@dataclass
class Fit:
    names: list
    mean: np.ndarray
    std: np.ndarray
    w: np.ndarray
    b: float
    n: int
    fitted_at: float
    train_hit_rate: float


class MetaCombiner:
    def __init__(self) -> None:
        self.fit: Fit | None = None
        self.pending: dict[int, dict] = {}         # cycle -> prediction made at lock
        self.graded: deque[dict] = deque(maxlen=600)
        self.recipe_graded: deque[int] = deque(maxlen=600)
        self.last: dict = {}
        self.last_error: str = ""
        self.settled_since_fit: int = 0
        self.experts: dict[str, Fit] = {}
        self.expert_graded: dict[str, deque] = {}
        self.calib: tuple[float, float] | None = None       # Platt (a, b) on the walk-forward record

    # ------------------------------------------------------------------
    # fitting
    # ------------------------------------------------------------------
    @staticmethod
    def _matrix(rows: list, names: list | None = None) -> tuple[list, np.ndarray, np.ndarray, np.ndarray]:
        if names is None:
            seen: dict[str, int] = {}
            for r in rows:
                for k in r.inputs:
                    seen[k] = seen.get(k, 0) + 1
            # an input has to be present on at least a third of the rows;
            # regime labels are routing keys, never features
            names = sorted(k for k, c in seen.items() if c >= max(5, len(rows) // 3) and not k.startswith("regime:"))
        idx = {k: j for j, k in enumerate(names)}
        X = np.zeros((len(rows), len(names)), dtype=np.float64)
        M = np.zeros_like(X, dtype=bool)
        y = np.zeros(len(rows), dtype=np.float64)
        t = np.zeros(len(rows), dtype=np.float64)
        for i, r in enumerate(rows):
            for k, v in r.inputs.items():
                j = idx.get(k)
                if j is not None and math.isfinite(v):
                    X[i, j] = v
                    M[i, j] = True
            y[i] = 1.0 if float(r.move_bps) > 0 else 0.0
            t[i] = r.at
        return names, X, M, y, t  # type: ignore[return-value]

    @staticmethod
    def regime_of(inputs: dict) -> str | None:
        try:
            code = int(round(float(inputs.get(REGIME_KEY))))
        except (TypeError, ValueError):
            return None
        return REGIMES[code] if 0 <= code < len(REGIMES) else None

    @staticmethod
    def routes_of(inputs: dict) -> list[str]:
        """Every "<key>=<name>" label this window falls in."""
        out = []
        for key, names in ROUTES.items():
            try:
                code = int(round(float(inputs.get(key))))
            except (TypeError, ValueError):
                continue
            if 0 <= code < len(names):
                out.append(f"{key}={names[code]}")
        return out

    def refit(self, rows: list, now: float | None = None) -> Fit | None:
        """The global fit, then one expert per regime that has enough rows."""
        rows = [r for r in rows if r.decided]
        fit = self._fit_rows(rows, now, MIN_FIT)
        if fit is None:
            return None
        self.fit = fit
        self.settled_since_fit = 0
        experts: dict[str, Fit] = {}
        buckets: dict[str, list] = {}
        for r in rows:
            for label in self.routes_of(r.inputs):
                buckets.setdefault(label, []).append(r)
        for label, sub in buckets.items():
            ef = self._fit_rows(sub, now, EXPERT_MIN_FIT)
            if ef is not None:
                experts[label] = ef
        self.experts = experts
        self._fit_calibration()
        return fit

    # ------------------------------------------------------------------
    # Round BO: Platt calibration of P(up) on the walk-forward record
    # ------------------------------------------------------------------
    def _fit_calibration(self) -> None:
        rows = [g for g in list(self.graded)[-400:] if "p_up" in g]
        if len(rows) < CALIB_MIN:
            self.calib = None
            return
        z = np.array([math.log(max(1e-4, g["p_up"]) / max(1e-4, 1 - g["p_up"])) for g in rows])
        y = np.array([1.0 if ((g["p_up"] > 0.5) == bool(g["hit"])) else 0.0 for g in rows])   # realised "up"
        # one-parameter Platt (slope only): the side is never changed by
        # calibration, only how much the probability is trusted
        a = 1.0
        for _ in range(25):
            p = 1.0 / (1.0 + np.exp(-(a * z)))
            g = float(np.dot(z, p - y)) + 0.1 * (a - 1.0)
            h = float(np.dot(z * z, p * (1 - p))) + 0.1
            da = g / h
            a -= da
            if abs(da) < 1e-7:
                break
        self.calib = (float(np.clip(a, 0.05, 5.0)), 0.0)

    def calibrate(self, p_up: float) -> float:
        if self.calib is None:
            return float(p_up)
        a, b = self.calib
        z = math.log(max(1e-4, p_up) / max(1e-4, 1 - p_up))
        return float(1.0 / (1.0 + math.exp(-(a * z + b))))

    def reliability(self, bins: int = 5) -> list[dict]:
        """Predicted confidence vs realised hit rate, in confidence bins."""
        rows = [g for g in list(self.graded)[-400:] if "p_up" in g]
        out = []
        edges = np.linspace(0.0, 1.0, bins + 1)
        for lo, hi in zip(edges[:-1], edges[1:]):
            sub = [g for g in rows if lo <= abs(2 * g["p_up"] - 1) < hi or (hi == 1.0 and abs(2 * g["p_up"] - 1) == 1.0)]
            if not sub:
                out.append({"lo": round(float(lo), 2), "hi": round(float(hi), 2), "n": 0, "predicted": None, "realised": None})
                continue
            pred = float(np.mean([0.5 + abs(2 * g["p_up"] - 1) / 2 for g in sub]))
            real = float(np.mean([1.0 if g["hit"] else 0.0 for g in sub]))
            out.append({"lo": round(float(lo), 2), "hi": round(float(hi), 2), "n": len(sub), "predicted": round(pred, 4), "realised": round(real, 4)})
        return out

    def _fit_rows(self, rows: list, now: float | None, min_rows: int) -> Fit | None:
        if len(rows) < min_rows:
            return None
        now = time.time() if now is None else now
        names, X, M, y, t = self._matrix(rows)
        if not names:
            return None
        mean = np.where(M.sum(0) > 0, (X * M).sum(0) / np.maximum(1, M.sum(0)), 0.0)
        var = ((X - mean) ** 2 * M).sum(0) / np.maximum(1, M.sum(0))
        std = np.sqrt(np.maximum(var, 1e-12))
        Z = np.where(M, (X - mean) / std, 0.0)
        Z = np.clip(Z, -4.0, 4.0)
        order = np.argsort(t)
        age = (len(rows) - 1 - np.argsort(order)).astype(np.float64)   # 0 = newest
        a = 0.5 ** (age / HALF_LIFE)
        a *= len(rows) / a.sum()
        n, d = Z.shape
        w = np.zeros(d)
        b = float(np.log(max(1e-6, y.mean()) / max(1e-6, 1 - y.mean())))
        for _ in range(12):
            p = _sigmoid(Z @ w + b)
            g = Z.T @ (a * (p - y)) + 2.0 * LAMBDA * w
            gb = float(np.sum(a * (p - y)))
            s = a * p * (1 - p) + 1e-9
            H = (Z * s[:, None]).T @ Z + 2.0 * LAMBDA * np.eye(d)
            Hb = float(np.sum(s))
            try:
                step = np.linalg.solve(H, g)
            except np.linalg.LinAlgError:
                step = g / (np.diag(H) + 1e-9)
            w -= step
            b -= gb / Hb
            if float(np.abs(step).max()) < 1e-6:
                break
        p = _sigmoid(Z @ w + b)
        train = float(np.mean((p > 0.5) == (y > 0.5)))
        return Fit(names=list(names), mean=mean, std=std, w=w, b=b, n=n, fitted_at=now, train_hit_rate=train)

    # ------------------------------------------------------------------
    # predicting / grading
    # ------------------------------------------------------------------
    def expert_record(self, label: str, n: int = WINDOW) -> tuple[float | None, float | None, int]:
        """(expert hit rate, global hit rate, count) over the last n graded
        windows that fell in this route."""
        rows = list(self.expert_graded.get(label, ()))[-n:]
        if not rows:
            return None, None, 0
        return (sum(e for e, _ in rows) / len(rows), sum(g for _, g in rows) / len(rows), len(rows))

    def predict(self, inputs: dict, fit: Fit | None = None) -> dict:
        """The global fit's call, routed to the expert (change-point regime,
        session, volatility) that has earned it best walk-forward; P(up) is
        then Platt-calibrated on the walk-forward record."""
        base = self._predict_fit(inputs, fit)
        if not base.get("available"):
            return base
        base["regime"] = self.regime_of(inputs)
        routes = self.routes_of(inputs)
        base["routes"] = routes
        expert_sides: dict[str, str] = {}
        best = None                                   # (margin, label, prediction, e_rate, g_rate, n)
        watching = []
        for label in routes:
            ef = self.experts.get(label)
            if ef is None:
                continue
            ex = self._predict_fit(inputs, ef)
            expert_sides[label] = ex.get("side")
            e_rate, g_rate, n = self.expert_record(label)
            if n >= EXPERT_MIN_GRADED and e_rate is not None and g_rate is not None and e_rate > g_rate:
                margin = e_rate - g_rate
                if best is None or margin > best[0]:
                    best = (margin, label, ex, e_rate, g_rate, n)
            else:
                watching.append(f"{label} {n}/{EXPERT_MIN_GRADED}" if n < EXPERT_MIN_GRADED else f"{label} {e_rate:.0%} vs {g_rate:.0%}")
        base["expert_sides"] = expert_sides
        base["expert_side"] = expert_sides.get(f"{REGIME_KEY}={base['regime']}") if base["regime"] else None
        if best is not None:
            _, label, ex, e_rate, g_rate, n = best
            routed = dict(ex)
            routed.update({"regime": base["regime"], "routes": routes, "expert_sides": expert_sides,
                           "expert_side": ex.get("side"), "global_side": base["side"], "expert_used": True, "route": label,
                           "routing": f"{label} expert ({e_rate:.0%} vs global {g_rate:.0%} over {n}) picks the side"})
            return self._calibrated(routed)
        base["expert_used"] = False
        base["route"] = None
        base["routing"] = ("no expert fitted yet" if not expert_sides else "experts watching: " + "; ".join(watching)) if routes else "no routing labels"
        return self._calibrated(base)

    def _calibrated(self, pred: dict) -> dict:
        p = float(pred.get("p_up", 0.5))
        pc = self.calibrate(p)
        pred["p_up_raw"] = round(p, 4)
        pred["p_up"] = round(pc, 4)
        # slope-only calibration keeps the side; it only resizes the confidence
        pred["confidence"] = round(abs(2 * pc - 1), 4)
        pred["calibrated"] = self.calib is not None
        return pred

    def _predict_fit(self, inputs: dict, fit: Fit | None = None) -> dict:
        f = fit or self.fit
        if f is None:
            return {"available": False, "reason": f"needs {MIN_FIT} decided windows for the first fit"}
        z = np.zeros(len(f.names))
        present = 0
        for j, k in enumerate(f.names):
            v = inputs.get(k)
            if v is None or not math.isfinite(float(v)):
                continue
            z[j] = float(np.clip((float(v) - f.mean[j]) / f.std[j], -4.0, 4.0))
            present += 1
        contrib = z * f.w
        logit = float(contrib.sum() + f.b)
        p = float(_sigmoid(np.array([logit]))[0])
        top = np.argsort(-np.abs(contrib))[:6]
        return {
            "available": True, "p_up": round(p, 4), "side": "BUY" if p > 0.5 else "SELL",
            "confidence": round(abs(2 * p - 1), 4), "logit": round(logit, 4), "inputs_present": present,
            "inputs_total": len(f.names),
            "drivers": [{"name": f.names[j], "z": round(float(z[j]), 3), "weight": round(float(f.w[j]), 4),
                         "contribution": round(float(contrib[j]), 4)} for j in top if contrib[j] != 0.0],
        }

    def remember(self, cycle: int, prediction: dict, recipe_side: str | None) -> None:
        if prediction.get("available"):
            self.pending[int(cycle)] = {"p_up": prediction["p_up"], "side": prediction["side"],
                                        "recipe_side": recipe_side, "at": time.time(),
                                        "regime": prediction.get("regime"), "expert_side": prediction.get("expert_side"),
                                        "expert_sides": dict(prediction.get("expert_sides") or {}),
                                        "global_side": prediction.get("global_side") or prediction["side"]}
            if len(self.pending) > 200:
                for k in sorted(self.pending)[:-200]:
                    self.pending.pop(k, None)

    def settle(self, cycle: int, move_bps: float, cost_bps: float = 0.0) -> dict | None:
        rec = self.pending.pop(int(cycle), None)
        self.settled_since_fit += 1
        if rec is None or abs(move_bps) <= cost_bps:
            return None
        up = move_bps > 0
        hit = (rec["side"] == "BUY") == up
        self.graded.append({"cycle": int(cycle), "hit": bool(hit), "p_up": rec["p_up"], "move_bps": round(move_bps, 2), "at": time.time()})
        if rec.get("recipe_side") in ("BUY", "SELL"):
            self.recipe_graded.append(1 if (rec["recipe_side"] == "BUY") == up else 0)
        g_hit = int((rec.get("global_side") == "BUY") == up)
        for label, side in (rec.get("expert_sides") or {}).items():
            if side in ("BUY", "SELL"):
                self.expert_graded.setdefault(label, deque(maxlen=300)).append((int((side == "BUY") == up), g_hit))
        return self.graded[-1]

    # ------------------------------------------------------------------
    # start-up: honest walk-forward over the recorded history
    # ------------------------------------------------------------------
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
                if not pred.get("available"):
                    continue
                recipe = r.inputs.get("fusion:recipe")
                recipe_side = None if recipe is None or recipe == 0 else ("BUY" if recipe > 0 else "SELL")
                self.pending[r.cycle] = {"p_up": pred["p_up"], "side": pred["side"], "recipe_side": recipe_side, "at": r.at,
                                         "regime": pred.get("regime"), "expert_side": pred.get("expert_side"),
                                         "expert_sides": dict(pred.get("expert_sides") or {}),
                                         "global_side": pred.get("global_side") or pred["side"]}
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

    def may_decide(self) -> tuple[bool, str]:
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
        return True, f"trailing {rate:.0%} over {n} graded windows (recipe {r_rate:.0%}) - the meta-combiner picks the side" if r_rate is not None \
            else f"trailing {rate:.0%} over {n} graded windows - the meta-combiner picks the side"

    def to_dict(self) -> dict:
        rate, n = self.trailing()
        r_rate, rn = self.recipe_trailing()
        all_rate = (sum(1 for r in self.graded if r["hit"]) / len(self.graded)) if self.graded else None
        decide, why = self.may_decide()
        f = self.fit
        weights = []
        if f is not None:
            order = np.argsort(-np.abs(f.w))[:15]
            weights = [{"name": f.names[j], "weight": round(float(f.w[j]), 4)} for j in order]
        return {
            "available": f is not None, "decides": decide, "why": why,
            "trailing_hit_rate": None if rate is None else round(rate, 4), "trailing_n": n,
            "recipe_trailing_hit_rate": None if r_rate is None else round(r_rate, 4), "recipe_trailing_n": rn,
            "all_time_hit_rate": None if all_rate is None else round(all_rate, 4), "graded": len(self.graded),
            "fit": None if f is None else {"inputs": len(f.names), "rows": f.n, "train_hit_rate": round(f.train_hit_rate, 4),
                                           "fitted_at": f.fitted_at, "bias": round(float(f.b), 4), "top_weights": weights},
            "last": self.last, "error": self.last_error,
            "experts": [
                {"route": lbl, "key": lbl.split("=")[0], "regime": lbl.split("=")[1],
                 "fitted": lbl in self.experts, "rows": self.experts[lbl].n if lbl in self.experts else 0,
                 "expert_hit_rate": None if er is None else round(er, 4), "global_hit_rate": None if gr is None else round(gr, 4),
                 "graded": n, "in_use": bool(lbl in self.experts and n >= EXPERT_MIN_GRADED and er is not None and gr is not None and er > gr)}
                for lbl in [f"{k}={nm}" for k, names in ROUTES.items() for nm in names] for (er, gr, n) in [self.expert_record(lbl)]
            ],
            "calibration": {"fitted": self.calib is not None,
                            "a": None if self.calib is None else round(self.calib[0], 3),
                            "b": None if self.calib is None else round(self.calib[1], 3),
                            "reliability": self.reliability()},
            "method": ("ridge logistic regression over every recorded input (standardised), recency half-life "
                       f"{HALF_LIFE:.0f} windows, λ={LAMBDA:g}; scored walk-forward; takes the side only while its trailing "
                       f"{WINDOW}-window record ≥ {TAKEOVER_RATE:.0%} and ≥ the recipe's"),
        }

    def export_state(self) -> dict:
        return {"graded": list(self.graded), "recipe_graded": list(self.recipe_graded),
                "expert_graded": {k: [list(x) for x in v] for k, v in self.expert_graded.items()}, "saved_at": time.time()}

    def import_state(self, payload: dict) -> int:
        g = (payload or {}).get("graded") or []
        for row in g:
            self.graded.append(dict(row))
        for v in (payload or {}).get("recipe_graded") or []:
            self.recipe_graded.append(int(v))
        for k, rows in ((payload or {}).get("expert_graded") or {}).items():
            label = k if "=" in k else f"{REGIME_KEY}={k}"
            dq = self.expert_graded.setdefault(label, deque(maxlen=300))
            for e, gl in rows:
                dq.append((int(e), int(gl)))
        self._fit_calibration()
        return len(g)
