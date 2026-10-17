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
            # an input has to be present on at least a third of the rows
            names = sorted(k for k, c in seen.items() if c >= max(5, len(rows) // 3))
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

    def refit(self, rows: list, now: float | None = None) -> Fit | None:
        rows = [r for r in rows if r.decided]
        if len(rows) < MIN_FIT:
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
        self.fit = Fit(names=list(names), mean=mean, std=std, w=w, b=b, n=n, fitted_at=now, train_hit_rate=train)
        self.settled_since_fit = 0
        return self.fit

    # ------------------------------------------------------------------
    # predicting / grading
    # ------------------------------------------------------------------
    def predict(self, inputs: dict, fit: Fit | None = None) -> dict:
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
                                        "recipe_side": recipe_side, "at": time.time()}
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
                self.pending[r.cycle] = {"p_up": pred["p_up"], "side": pred["side"], "recipe_side": recipe_side, "at": r.at}
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
            "method": ("ridge logistic regression over every recorded input (standardised), recency half-life "
                       f"{HALF_LIFE:.0f} windows, λ={LAMBDA:g}; scored walk-forward; takes the side only while its trailing "
                       f"{WINDOW}-window record ≥ {TAKEOVER_RATE:.0%} and ≥ the recipe's"),
        }

    def export_state(self) -> dict:
        return {"graded": list(self.graded), "recipe_graded": list(self.recipe_graded), "saved_at": time.time()}

    def import_state(self, payload: dict) -> int:
        g = (payload or {}).get("graded") or []
        for row in g:
            self.graded.append(dict(row))
        for v in (payload or {}).get("recipe_graded") or []:
            self.recipe_graded.append(int(v))
        return len(g)
