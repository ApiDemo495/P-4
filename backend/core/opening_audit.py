"""Opening audit - the last hour, candle by candle, through every layer.

Round BJ.  The user: "as soon as I open the app it has already analysed all
market and all results … tell me where my app is lacking and why.  Analyse
the last hour market every 60 second candle with all formulas, physics,
genesis, hedge, reasoning and emotions, each candle; if those things were in
favour keep them, if not change them (based on majority) and make them win
more."

At start (right after the warm-start history is fetched) this module walks
the **last 60 one-minute candles of the real tape** and, at every candle
close, runs the same code the live windows run on the tape as it stood at
that moment:

* the 31 formulas (microstructure, order book, **hedge** HRDD/HSI/SHRP/GCDV,
  regime, temporal, Kalman, news, brain CCSv2/KCAE, window formulas),
* the thermodynamic **physics** layer (O-U, VPIN, Hawkes, flux, entropy,
  diffusion, pendulum, temperature, Avellaneda-Stoikov),
* the crowd **emotions** (nine intensities with their tone, the crowd tone,
  the signed features),
* the **hedge outcomes** (best pair action's expected bps),
* the **reasoning** (brain CCSv2 - the spec's recipe without agents - and
  the forward table),
* **genesis** (the active set's fitness-weighted vote at each candle, from
  the engine's own candle store).

Each reading is graded on the candle that followed (strict sign, bps) and
handed to the hindsight reviewer - the same object the live windows feed -
so corrections (damp / boost / re-centre / invert, decided by the majority
of candles) are in force **before the first live window**, and the panel can
say, per layer, where the app was lacking in that hour and why.
"""
from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field

import numpy as np

from backend.core.hindsight import HEDGE_FORMULAS, SIGNED_FEATURES
from backend.core.warm_start import LOOKBACK_TICKS, bars_to_ticks, snapshot_at

log = logging.getLogger("drosophila.audit")

CANDLE_S = 60.0
HOUR_CANDLES = 60
LAYERS = ("formula", "hedge", "physics", "emotion", "reasoning", "genesis", "feature")
LAYER_LABEL = {
    "formula": "formulas", "hedge": "hedge", "physics": "physics", "emotion": "emotions",
    "reasoning": "reasoning (brain · forward · recipe)", "genesis": "genesis", "feature": "crowd features",
}


@dataclass
class AuditReport:
    phase: str = "idle"          # idle | running | done | failed | skipped
    detail: str = ""
    error: str = ""
    asset: str = "BTC"
    candles: int = 0
    graded: int = 0
    started_at: float = 0.0
    finished_at: float = 0.0
    elapsed_s: float = 0.0
    market: dict = field(default_factory=dict)
    layers: dict = field(default_factory=dict)
    lacking: list = field(default_factory=list)
    in_favour: list = field(default_factory=list)
    changed: dict = field(default_factory=dict)
    layer_errors: dict = field(default_factory=dict)
    summary: str = ""

    def to_dict(self) -> dict:
        return dict(self.__dict__)


def _tone_sign(tone: str) -> float:
    t = (tone or "").lower()
    if t in ("bullish", "greed", "up", "positive"):
        return 1.0
    if t in ("bearish", "fear", "down", "negative"):
        return -1.0
    return 0.0


def _market_facts(bars_1m: np.ndarray) -> dict:
    """What the hour did, in the competition's own terms."""
    if bars_1m.shape[0] < 2:
        return {}
    close = bars_1m[:, 4]
    open_ = bars_1m[:, 1]
    rets = (close / open_ - 1.0) * 1e4
    signs = np.sign(close - open_)
    decided = signs[signs != 0]
    cont = float(np.mean(decided[1:] == decided[:-1])) if decided.size > 1 else None
    return {
        "candles": int(bars_1m.shape[0]),
        "net_bps": round(float((close[-1] / open_[0] - 1.0) * 1e4), 1),
        "up_candles": int(np.sum(signs > 0)), "down_candles": int(np.sum(signs < 0)), "flat_candles": int(np.sum(signs == 0)),
        "up_rate": round(float(np.mean(decided > 0)), 3) if decided.size else None,
        "continuation_rate": None if cont is None else round(cont, 3),
        "sigma_bps": round(float(np.std(rets)), 1),
        "largest_bps": round(float(rets[np.argmax(np.abs(rets))]), 1),
        "range_bps": round(float((bars_1m[:, 2].max() / bars_1m[:, 3].min() - 1.0) * 1e4), 1),
        "character": ("trending" if cont is not None and cont >= 0.58 else "mean-reverting" if cont is not None and cont <= 0.42 else "choppy"),
    }


def run_audit(asset: str, bars_1s: np.ndarray, hedge_bars: np.ndarray | None, hindsight, *,
              brain_mb=None, physics=None, genesis=None, forward=None, directional: dict | None = None,
              report: AuditReport | None = None, candles: int = HOUR_CANDLES) -> AuditReport:
    """Walk the last ``candles`` one-minute closes and grade every layer."""
    from backend.core import emotions as emotions_module
    from backend.core import hedge_outcomes
    from backend.core.warm_start import _ReplayBrain
    from backend.formulas.engine import ALL_FORMULAS, FormulaEngine

    rep = report or AuditReport(asset=asset)
    rep.phase, rep.started_at, rep.error = "running", time.time(), ""
    directional = directional or {s.name: s.category for s in ALL_FORMULAS if getattr(s, "directional", True)
                                  and s.name not in ("HSI", "ERC")}
    try:
        ticks = bars_to_ticks(bars_1s)
        hedge_ticks = bars_to_ticks(hedge_bars) if hedge_bars is not None and hedge_bars.size else np.zeros((0, 4))
        if ticks.shape[0] < LOOKBACK_TICKS + 3 * int(CANDLE_S):
            raise RuntimeError(f"only {ticks.shape[0]} seconds of history")
        times = ticks[:, 0]
        last_ms = float(times[-1])
        # closed-minute boundaries: the last ``candles`` minute marks before the tape's end
        last_close = math.floor(last_ms / 60_000.0) * 60_000.0
        bounds = np.array([last_close - k * 60_000.0 for k in range(candles, 0, -1)], dtype=np.float64)
        bounds = bounds[bounds >= float(times[min(LOOKBACK_TICKS, times.size - 1)])]
        rep.candles = int(bounds.size)
        engine = FormulaEngine(brain=_ReplayBrain(brain_mb) if brain_mb is not None else None)
        last_dir = 0
        if forward is not None:
            engine.forward_hook = lambda values: forward.edges_vector(values, last_dir)
        drg_hist = np.zeros((0, 2))
        genesis_votes = {}
        if genesis is not None:
            try:
                for row in genesis.candle_votes(asset, last_n=candles + 2):
                    genesis_votes[int(round(row["ts_ms"] / 60_000.0))] = float(row["vote"])
            except Exception as exc:  # noqa: BLE001
                rep.layer_errors["genesis"] = f"{type(exc).__name__}: {exc}"[:200]
        minute_bars = []
        for b in bounds:
            end = int(np.searchsorted(times, b, side="right"))
            start = max(0, end - LOOKBACK_TICKS)
            window = ticks[start:end]
            nxt_end = int(np.searchsorted(times, b + CANDLE_S * 1000.0, side="right"))
            if window.shape[0] < 50 or nxt_end <= end or nxt_end > ticks.shape[0]:
                continue
            entry = float(window[-1, 1])
            exit_px = float(ticks[nxt_end - 1, 1])
            seg = ticks[end:nxt_end, 1]
            minute_bars.append([b, entry, float(seg.max()), float(seg.min()), exit_px])
            if hedge_ticks.shape[0]:
                h_end = int(np.searchsorted(hedge_ticks[:, 0], b, side="right"))
                h_window = hedge_ticks[max(0, h_end - LOOKBACK_TICKS):h_end]
            else:
                h_window = np.zeros((0, 4))
            snap = snapshot_at(asset, b, window, h_window, bars_1s, hedge_bars, drg_hist,
                               source="opening audit (Binance 1 s klines, 60 s candles)")
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

            # --- formulas (incl. hedge and brain) --------------------------
            try:
                result = engine.run(snap, asset)
                values = dict(result.values)
            except Exception as exc:  # noqa: BLE001
                rep.layer_errors["formula"] = f"{type(exc).__name__}: {exc}"[:200]
                continue
            for name in directional:
                if name in values:
                    put(name, values[name], "hedge" if name in HEDGE_FORMULAS else "formula")
            ccs = float(values.get("CCSv2", 0.0) or 0.0)
            put("brain:CCSv2", ccs, "reasoning")
            # the recipe without agents: the brain's read, else the formulas' mean vote
            dir_vals = [float(values[n]) for n in directional if n in values and math.isfinite(float(values[n]))]
            recipe_score = ccs if ccs != 0.0 else (float(np.mean(dir_vals)) if dir_vals else 0.0)
            put("fusion:recipe", recipe_score, "reasoning")
            if forward is not None:
                try:
                    fwd = forward.composite(values, last_dir)
                    if fwd.get("available"):
                        put("forward", fwd.get("value"), "reasoning")
                except Exception as exc:  # noqa: BLE001
                    rep.layer_errors["forward"] = f"{type(exc).__name__}: {exc}"[:200]
            # --- physics -------------------------------------------------
            if physics is not None:
                try:
                    ph = physics.compute(snap, asset, seconds_left=CANDLE_S)
                    put("physics", ph.get("vote"), "physics")
                    for m in ph.get("mechanisms") or []:
                        if m.get("active"):
                            put(f"physics:{m.get('key') or m.get('name')}", float(m.get("direction") or 0) * float(m.get("edge_bps") or 0.0), "physics")
                except Exception as exc:  # noqa: BLE001
                    rep.layer_errors["physics"] = f"{type(exc).__name__}: {exc}"[:200]
            # --- emotions + features --------------------------------------
            side_guess = "BUY" if recipe_score >= 0 else "SELL"
            try:
                emo = emotions_module.analyze(snap, asset, formulas=values, directional=directional)
                if emo.available:
                    agreement = getattr(emo, "formula_agreement", None) or {}
                    put("crowd:tone", agreement.get("crowd_tone", 0.0) if isinstance(agreement, dict) else 0.0, "emotion")
                    for sc in emo.scores:
                        put(f"emotion:{sc.name}", _tone_sign(sc.tone) * float(sc.intensity), "emotion")
                    feats = getattr(emo, "features", None) or {}
                    for key in SIGNED_FEATURES:
                        if key in feats:
                            put(f"feature:{key}", feats[key], "feature")
            except Exception as exc:  # noqa: BLE001
                rep.layer_errors["emotion"] = f"{type(exc).__name__}: {exc}"[:200]
            # --- hedge outcomes -------------------------------------------
            try:
                ho = hedge_outcomes.build(snap, values, side_guess, 0.55, asset, horizon_s=CANDLE_S)
                if ho.get("valid") and ho.get("best"):
                    best = ho["best"]
                    put("hedge:best_action", float(best.get("btc_weight") or 0.0) * float(best.get("expected_bps") or 0.0), "hedge")
            except Exception as exc:  # noqa: BLE001
                rep.layer_errors["hedge"] = f"{type(exc).__name__}: {exc}"[:200]
            # --- genesis (from its own candle store) ----------------------
            gv = genesis_votes.get(int(round(b / 60_000.0)))
            if gv is not None:
                put("genesis", gv, "genesis")
            # --- grade on the candle that followed -------------------------
            move_bps = (exit_px / entry - 1.0) * 1e4 if entry > 0 else 0.0
            cycle = -(int(b // 1000))
            hindsight.record(cycle, side_guess, 0.55, inputs, groups, at=b / 1000.0 + CANDLE_S)
            hindsight.settle(cycle, move_bps, cost_bps=1.0)
            sign_next = 1 if move_bps > 0 else -1 if move_bps < 0 else 0
            recipe = 1 if recipe_score > 0 else -1 if recipe_score < 0 else 0
            pnl = move_bps if recipe >= 0 else -move_bps
            drg_hist = np.vstack([drg_hist, [[1.0 if (recipe and recipe == sign_next) else -1.0 if recipe else 0.0, pnl]]])
            last_dir = sign_next
            rep.graded += 1
        rep.market = _market_facts(np.asarray(minute_bars, dtype=np.float64)) if minute_bars else {}
        _diagnose(rep, hindsight)
        rep.phase = "done"
    except Exception as exc:  # noqa: BLE001
        rep.phase, rep.error = "failed", f"{type(exc).__name__}: {exc}"[:300]
        rep.detail = "the opening audit could not run; the hour is learned live instead"
        log.warning("opening audit failed: %s", rep.error)
    finally:
        rep.finished_at = time.time()
        rep.elapsed_s = round(rep.finished_at - rep.started_at, 1)
    return rep


def _why(row: dict, market: dict) -> str:
    """One honest sentence on why an input lost the hour."""
    hr = row.get("hit_rate")
    bias = float(row.get("bias") or 0.0)
    spread = float(row.get("spread") or 0.0)
    net = float(market.get("net_bps") or 0.0)
    cont = market.get("continuation_rate")
    silent = int(row.get("silent") or 0)
    decided = int(row.get("decided") or 0)
    if decided and silent > decided:
        return "mostly silent - gave no side on more candles than it did"
    if hr is None:
        return "no decided candle"
    leaned = "up" if bias > 0.15 * spread else "down" if bias < -0.15 * spread else ""
    if leaned and ((leaned == "up") != (net > 0)) and hr < 0.5:
        return f"leaned {leaned} through an hour that went {'up' if net > 0 else 'down'} {abs(net):.0f} bp"
    if cont is not None and cont <= 0.42 and hr < 0.5:
        return "followed the last move in a mean-reverting hour (continuation {:.0%})".format(cont)
    if cont is not None and cont >= 0.58 and hr < 0.5:
        return "faded the move in a trending hour (continuation {:.0%})".format(cont)
    if hr < 0.5:
        return "wrong more often than right with no systematic lean - noise this hour"
    return "on the right side more often than not"


def _diagnose(rep: AuditReport, hindsight) -> None:
    payload = hindsight.to_dict()
    table = payload.get("table") or []
    market = rep.market or {}
    layers: dict[str, dict] = {}
    for row in table:
        g = row["group"]
        L = layers.setdefault(g, {"label": LAYER_LABEL.get(g, g), "inputs": 0, "decided_inputs": 0, "hits": 0, "misses": 0,
                                  "won_bps": 0.0, "lost_bps": 0.0, "kept": 0, "boosted": 0, "damped": 0,
                                  "re_centred": 0, "inverted": 0, "watching": 0, "worst": None, "best": None})
        L["inputs"] += 1
        if row["decided"]:
            L["decided_inputs"] += 1
        L["hits"] += row["hits"]
        L["misses"] += row["misses"]
        L["won_bps"] = round(L["won_bps"] + row["won_bps"], 1)
        L["lost_bps"] = round(L["lost_bps"] + row["lost_bps"], 1)
        key = {"kept": "kept", "boosted": "boosted", "damped": "damped", "re-centred": "re_centred",
               "inverted": "inverted", "watching": "watching"}.get(row["action"], "watching")
        L[key] += 1
        if L["worst"] is None or row["net_bps"] < L["worst"]["net_bps"]:
            L["worst"] = {"name": row["name"], "net_bps": row["net_bps"], "hits": row["hits"], "decided": row["decided"],
                          "action": row["action"], "why": _why(row, market)}
        if L["best"] is None or row["net_bps"] > L["best"]["net_bps"]:
            L["best"] = {"name": row["name"], "net_bps": row["net_bps"], "hits": row["hits"], "decided": row["decided"],
                         "action": row["action"]}
    for g, L in layers.items():
        d = L["hits"] + L["misses"]
        L["hit_rate"] = round(L["hits"] / d, 4) if d else None
        L["net_bps"] = round(L["won_bps"] - L["lost_bps"], 1)
        L["corrected"] = L["boosted"] + L["damped"] + L["re_centred"] + L["inverted"]
        if g in rep.layer_errors:
            L["verdict"] = f"could not be audited: {rep.layer_errors[g]}"
        elif L["hit_rate"] is None:
            L["verdict"] = "no decided candle this hour"
        elif L["hit_rate"] < 0.47:
            L["verdict"] = (f"LACKING - {L['hits']}/{d} right ({L['hit_rate']:.0%}), lost {L['lost_bps']:.0f} bp; "
                            f"worst {L['worst']['name']}: {L['worst']['why']}; {L['damped'] + L['inverted']} inputs damped/inverted, "
                            f"{L['re_centred']} re-centred")
        elif L["hit_rate"] > 0.53:
            L["verdict"] = (f"IN FAVOUR - {L['hits']}/{d} right ({L['hit_rate']:.0%}), net {L['net_bps']:+.0f} bp; "
                            f"{L['boosted']} inputs boosted, {L['kept']} kept")
        else:
            L["verdict"] = (f"COIN-FLIP - {L['hits']}/{d} right ({L['hit_rate']:.0%}); no edge either way this hour; "
                            f"{L['corrected']} inputs corrected on their own record")
    rep.layers = {g: layers[g] for g in LAYERS if g in layers}
    rep.lacking = [r["name"] for r in table if r["decided"] >= payload.get("min_decided", 8) and (r["hit_rate"] or 0) < 0.45][:15]
    rep.in_favour = [r["name"] for r in reversed(table) if r["decided"] >= payload.get("min_decided", 8) and (r["hit_rate"] or 0) > 0.55][:15]
    rep.changed = {k: sum(1 for r in table if r["action"] == k) for k in ("boosted", "damped", "re-centred", "inverted", "kept", "watching")}
    engine_hr = payload.get("engine_hit_rate_hour")
    mk = market
    rep.summary = (
        f"Last hour on {rep.asset}: {mk.get('net_bps', 0):+.0f} bp net, {mk.get('up_candles', 0)} up / {mk.get('down_candles', 0)} down candles, "
        f"{mk.get('character', '?')} (continuation {mk.get('continuation_rate') if mk.get('continuation_rate') is not None else '—'}), σ {mk.get('sigma_bps', 0)} bp/candle. "
        f"The recipe alone was right {engine_hr:.0%} of decided candles. " if engine_hr is not None else
        f"Last hour on {rep.asset}: {mk.get('net_bps', 0):+.0f} bp net. ") + (
        f"Lacking: {', '.join(LAYER_LABEL[g] for g, L in rep.layers.items() if str(L.get('verdict', '')).startswith('LACKING')) or 'no layer'}. "
        f"In favour: {', '.join(LAYER_LABEL[g] for g, L in rep.layers.items() if str(L.get('verdict', '')).startswith('IN FAVOUR')) or 'no layer'}. "
        f"Changed by majority: {rep.changed.get('damped', 0)} damped, {rep.changed.get('boosted', 0)} boosted, "
        f"{rep.changed.get('re-centred', 0)} re-centred, {rep.changed.get('inverted', 0)} inverted; {rep.changed.get('kept', 0)} kept as they were."
    )
    rep.detail = f"{rep.graded} candles graded through formulas · hedge · physics · emotions · reasoning · genesis in {rep.elapsed_s or (time.time() - rep.started_at):.1f} s"
