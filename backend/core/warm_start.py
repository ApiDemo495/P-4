"""Round BA - the warm start: the engine proves its formulas on the *future*
before it is allowed to call one.

The user's words: "they were right but in the past, not future".  That is the
exact failure mode of a recipe built from backward-looking measurements: a
formula that says "buyers were aggressive for the last 40 s" is *true* about
the window that just closed and is read as a forecast of the next one.  At a
53 s horizon on BTC/USDT last-trade settlement the next window reverts about
as often as it continues (bid-ask bounce, Roll 1984; one-minute
autocorrelation is small and often negative), so a momentum reading that is
right about the past is wrong about the future roughly half the time - and
the user, who watches the arrow against the move that produced it, sees "it
was right, just late".

The evidence ledger (``backend.core.calibration``) is the cure: it scores
every source on what the price did *afterwards* and fades the ones that are
reliably late.  But it needed 40 live windows (35 min) before it could act,
and the user starts a fresh Codespace every time - so every session began
with the blind recipe.

The warm start closes that gap in seconds:

1. fetch the last ``hours`` (default 24) of **real** 1-second klines for the asset from
   Binance (public REST, no key; paginated, 1000 bars per call);
2. turn them into the same tape the live engine sees (one trade per second:
   time, close, volume, aggressor side from the taker-buy share; one-minute
   closes for the candle inputs);
3. cut the history on the **same epoch-aligned 53 s grid** the live clock
   uses, build a ``FrozenMarketSnapshot`` at every boundary from the 600 s
   before it, run the full :class:`FormulaEngine` (all 31 formulas + the real
   mushroom body, no side effects on the live brain) and collect the window's
   votes exactly as the live cycle does;
4. score every vote against the **next** window's close - strict sign, the
   competition's rule - through ``EvidenceLedger.remember/score``.

The ledger therefore starts the session with a few hundred *forward* scored
windows: it is active from the first live boundary, every formula already
has a follow / fade / noise verdict, and the dashboard can print the lag
audit - "the formulas agree with the window that just closed X %, with the
next one Y %" - as a measured number instead of a feeling.

Honesty notes.  Historical bars carry no order book, so book-only formulas
(DGW, LCS, BAR, ...) see an empty book and mostly abstain; the agents and the
news do not replay; ``spec:fusion`` in the replay is the brain's CCSv2 sign
(the recipe's 40 % anchor), not the full live fusion.  The report says which
sources were replayed and which were not.  Nothing here runs in the sandbox
that cannot reach Binance; the module fails soft and reports why.
"""
from __future__ import annotations

import json
import logging
import math
import os
import ssl
import threading
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass, field

import numpy as np

from backend.core import config as cfg
from backend.core.frozen_snapshot import FrozenMarketSnapshot

log = logging.getLogger("drosophila.warm_start")

SYMBOLS = {"BTC": "BTCUSDT", "PAXG": "PAXGUSDT"}
LOOKBACK_TICKS = 600          # the live tape buffer the formulas see
BARS_PER_CALL = 1000


@dataclass
class WarmStartReport:
    phase: str = "idle"           # idle | fetching | replaying | done | failed | disabled
    detail: str = ""
    error: str = ""
    asset: str = "BTC"
    hours: float = 0.0
    bars: int = 0
    windows: int = 0
    scored: int = 0
    ties: int = 0
    started_at: float = 0.0
    finished_at: float = 0.0
    elapsed_s: float = 0.0
    # market facts on the real tape, under the competition's rule
    continuation_rate: float | None = None     # P(sign(next) == sign(last)) over non-tie windows
    tie_rate: float | None = None
    up_rate: float | None = None
    # the lag audit of the recipe (brain CCSv2 sign)
    recipe_agrees_with_past: float | None = None
    recipe_agrees_with_future: float | None = None
    recipe_windows: int = 0
    # per-source forward hit rates in the replay (not decayed)
    sources: dict = field(default_factory=dict)
    replayed_sources: list = field(default_factory=list)
    not_replayed: list = field(default_factory=lambda: ["agents (Gemini / GitHub / local)", "news (NIV, themes)",
                                                         "crowd emotions", "physics", "genesis", "order book (L2) formulas"])

    def to_dict(self) -> dict:
        out = dict(self.__dict__)
        out["sources"] = dict(sorted(self.sources.items(), key=lambda kv: -abs(kv[1].get("hit_rate", 0.5) - 0.5))[:40])
        return out


# ------------------------------------------------------------------ fetching
def _get_json(url: str, timeout: float = 20.0):
    req = urllib.request.Request(url, headers={"User-Agent": "drosophila-trader/2.0 warm-start"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except ssl.SSLError:
        ctx = ssl.create_default_context()
        with urllib.request.urlopen(req, timeout=timeout, context=ctx) as resp:
            return json.loads(resp.read().decode("utf-8"))


def fetch_klines(symbol: str, interval: str, start_ms: int, end_ms: int, base: str | None = None) -> np.ndarray:
    """All klines in [start_ms, end_ms] as rows
    ``[close_ms, open, high, low, close, volume, taker_buy_base]`` (oldest first)."""
    base = (base or cfg.SETTINGS.binance_rest_base or "https://api.binance.com").rstrip("/")
    rows: list[list[float]] = []
    cursor = int(start_ms)
    while cursor < end_ms:
        q = urllib.parse.urlencode({"symbol": symbol, "interval": interval, "startTime": cursor,
                                    "endTime": int(end_ms), "limit": BARS_PER_CALL})
        data = _get_json(f"{base}/api/v3/klines?{q}")
        if not isinstance(data, list) or not data:
            break
        for k in data:
            rows.append([float(k[6]), float(k[1]), float(k[2]), float(k[3]), float(k[4]), float(k[5]), float(k[9])])
        last_open = int(data[-1][0])
        step = int(data[-1][6]) - int(data[-1][0]) + 1
        nxt = last_open + step
        if nxt <= cursor:
            break
        cursor = nxt
        if len(data) < BARS_PER_CALL:
            break
    return np.asarray(rows, dtype=np.float64).reshape(-1, 7)


def bars_to_ticks(bars: np.ndarray) -> np.ndarray:
    """1 s bars -> the live tape shape ``[time_ms, price, qty, side]``: one
    print per second at the close, the second's volume, the aggressor side
    from the taker-buy share (+1 buyers, -1 sellers, 0 no trades)."""
    if bars.size == 0:
        return np.zeros((0, 4))
    vol = bars[:, 5]
    buy = bars[:, 6]
    side = np.where(vol <= 0, 0.0, np.where(buy > vol * 0.5, 1.0, np.where(buy < vol * 0.5, -1.0, 0.0)))
    return np.column_stack([bars[:, 0], bars[:, 4], vol, side])


def minute_closes(bars: np.ndarray, end_ms: float, count: int = 60) -> np.ndarray:
    """The last ``count`` one-minute closes at or before ``end_ms``."""
    if bars.size == 0:
        return np.zeros(0)
    sel = bars[bars[:, 0] <= end_ms]
    if sel.shape[0] == 0:
        return np.zeros(0)
    minute = (sel[:, 0] // 60000).astype(np.int64)
    _, last_idx = np.unique(minute[::-1], return_index=True)
    idx = sel.shape[0] - 1 - last_idx
    closes = sel[np.sort(idx), 4]
    return closes[-count:]


# ------------------------------------------------------------------- replay
class _ReplayBrain:
    """The real mushroom body, with no side effects on the live brain."""

    def __init__(self, mb) -> None:
        self.mb = mb
        self.conv = None

    def mushroom_body_pass(self, vector, drg: float, hsi: float):
        return self.mb.propagate(vector, drg=float(drg), hsi=float(hsi), detail=True)


def _empty_book() -> np.ndarray:
    return np.zeros((2, 20, 2))


def window_boundaries(first_ms: float, last_ms: float, period_s: float) -> np.ndarray:
    period_ms = float(period_s) * 1000.0
    first_b = math.ceil(first_ms / period_ms) * period_ms
    return np.arange(first_b, last_ms + 1, period_ms)


def replay(asset: str, bars: np.ndarray, hedge_bars: np.ndarray | None, ledger, *, brain_mb=None,
           period_s: float | None = None, directional: dict | None = None,
           report: WarmStartReport | None = None, max_windows: int = 2000, forward=None) -> WarmStartReport:
    """Run the formulas at every historical boundary and score them on the
    next window (strict sign).  ``ledger`` is the live EvidenceLedger."""
    from backend.core.calibration import EvidenceLedger
    from backend.formulas.engine import ALL_FORMULAS, FormulaEngine

    rep = report or WarmStartReport(asset=asset)
    period = float(period_s or cfg.SETTINGS.cycle_period_seconds)
    directional = directional or {s.name: s.category for s in ALL_FORMULAS if getattr(s, "directional", True)
                                  and s.name not in ("HSI", "ERC")}
    ticks = bars_to_ticks(bars)
    hedge_ticks = bars_to_ticks(hedge_bars) if hedge_bars is not None and hedge_bars.size else np.zeros((0, 4))
    if ticks.shape[0] < LOOKBACK_TICKS + 2 * int(period) + 5:
        rep.phase, rep.error = "failed", f"only {ticks.shape[0]} seconds of history"
        return rep
    engine = FormulaEngine(brain=_ReplayBrain(brain_mb) if brain_mb is not None else None)
    last_dir = [0]
    if forward is not None:          # the replayed brain smells forward values too, as live
        engine.forward_hook = lambda values: forward.edges_vector(values, last_dir[0])
    bounds = window_boundaries(ticks[LOOKBACK_TICKS, 0], ticks[-1, 0] - period * 1000.0, period)
    if bounds.size > max_windows:
        bounds = bounds[-max_windows:]
    rep.phase, rep.windows = "replaying", int(bounds.size)
    times = ticks[:, 0]
    saver = getattr(ledger, "_save", None)
    if saver is not None:           # one disk write at the end, not one per window
        ledger._save = lambda: None  # type: ignore[method-assign]
    hits: dict[str, list[int]] = {}
    ups = ties = 0
    cont = cont_n = 0
    past_agree = future_agree = recipe_n = 0
    prev_sign = 0
    drg_hist = np.zeros((0, 2))
    try:
        for i, b in enumerate(bounds):
            end = int(np.searchsorted(times, b, side="right"))
            start = max(0, end - LOOKBACK_TICKS)
            window = ticks[start:end]
            if window.shape[0] < 50:
                continue
            nxt_end = int(np.searchsorted(times, b + period * 1000.0, side="right"))
            if nxt_end <= end or nxt_end > ticks.shape[0]:
                continue
            entry = float(window[-1, 1])
            exit_px = float(ticks[nxt_end - 1, 1])
            if hedge_ticks.shape[0]:
                h_end = int(np.searchsorted(hedge_ticks[:, 0], b, side="right"))
                h_window = hedge_ticks[max(0, h_end - LOOKBACK_TICKS):h_end]
            else:
                h_window = np.zeros((0, 4))
            b_candles = minute_closes(bars, b)
            h_candles = minute_closes(hedge_bars, b) if hedge_bars is not None and hedge_bars.size else np.zeros(0)
            payload = {
                "timestamp": b / 1000.0,
                "btc_ticks": window if asset == "BTC" else h_window,
                "paxg_ticks": h_window if asset == "BTC" else window,
                "btc_book": _empty_book(), "paxg_book": _empty_book(),
                "btc_book_prev": _empty_book(), "paxg_book_prev": _empty_book(),
                "btc_candles": b_candles if asset == "BTC" else h_candles,
                "paxg_candles": h_candles if asset == "BTC" else b_candles,
                "news_items": (), "drg_outcomes": drg_hist[-20:] if drg_hist.size else np.zeros((0, 2)),
                "btc_tick_count": int(window.shape[0]), "paxg_tick_count": int(h_window.shape[0]),
                "btc_spreads": None, "paxg_spreads": None, "synced": None, "warnings": (),
                "source": "warm-start replay (Binance 1 s klines)",
            }
            snap = FrozenMarketSnapshot(**payload)
            try:
                result = engine.run(snap, asset)
            except Exception as exc:  # noqa: BLE001
                log.debug("replay window %d failed: %s", i, exc)
                continue
            ccs = float(result.values.get("CCSv2", 0.0) or 0.0)
            votes = EvidenceLedger.votes_from(
                formula_values=result.values, directional=directional, ccs_value=ccs, agents={},
                spec_score=ccs, candles=b_candles, ticks=window,
            )
            if not votes:
                continue
            cycle = -(int(b // 1000))            # negative cycle ids: never collide with live windows
            move_bps = (exit_px / entry - 1.0) * 1e4 if entry > 0 else 0.0
            actual_up = None if exit_px == entry else exit_px > entry
            # Round BB: the forward tables learn here too, and the composite
            # votes in the ledger once it has coverage - so it is scored on
            # the future like every other source
            if forward is not None:
                fwd = forward.composite(result.values, prev_sign)
                if fwd.get("available"):
                    votes["forward:composite"] = 1 if float(fwd.get("value") or 0.0) > 0 else -1
                forward.observe(cycle, result.values, prev_sign)
            learned = ledger.evaluate(asset, votes)
            p_up = float(learned.get("p_up") or 0.5)
            ledger.remember(asset, cycle, votes, p_up)
            ledger.score(asset, cycle, actual_up, move_bps=move_bps, cost_bps=1.0)
            if forward is not None:
                forward.settle(cycle, actual_up, move_bps)
            # facts for the report (un-decayed, strict sign)
            sign_next = 0 if actual_up is None else (1 if actual_up else -1)
            if sign_next == 0:
                ties += 1
            else:
                ups += 1 if sign_next > 0 else 0
                if prev_sign:
                    cont_n += 1
                    cont += 1 if prev_sign == sign_next else 0
                for name, vote in votes.items():
                    hits.setdefault(name, []).append(1 if vote == sign_next else 0)
            recipe = 1 if ccs > 0 else -1 if ccs < 0 else 0
            if recipe and prev_sign and sign_next:
                recipe_n += 1
                past_agree += 1 if recipe == prev_sign else 0
                future_agree += 1 if recipe == sign_next else 0
            pnl = move_bps if (recipe >= 0) else -move_bps
            drg_hist = np.vstack([drg_hist, [[1.0 if (recipe and recipe == sign_next) else -1.0 if recipe else 0.0, pnl]]])
            prev_sign = sign_next
            last_dir[0] = sign_next
            rep.scored += 1
    finally:
        if saver is not None:
            ledger._save = saver  # type: ignore[method-assign]
            try:
                saver()
            except Exception:  # noqa: BLE001
                pass
    rep.ties = ties
    n_dir = rep.scored - ties
    rep.tie_rate = round(ties / rep.scored, 4) if rep.scored else None
    rep.up_rate = round(ups / n_dir, 4) if n_dir else None
    rep.continuation_rate = round(cont / cont_n, 4) if cont_n else None
    rep.recipe_windows = recipe_n
    rep.recipe_agrees_with_past = round(past_agree / recipe_n, 4) if recipe_n else None
    rep.recipe_agrees_with_future = round(future_agree / recipe_n, 4) if recipe_n else None
    rep.sources = {name: {"n": len(v), "hit_rate": round(float(np.mean(v)), 4),
                          "verdict": _verdict(float(np.mean(v)), len(v))}
                   for name, v in hits.items() if len(v) >= 10}
    rep.replayed_sources = sorted(hits)
    rep.phase = "done"
    return rep


def _verdict(rate: float, n: int) -> str:
    se = math.sqrt(0.25 / max(1, n))
    if rate - 0.5 > 2.0 * se:
        return "follow"
    if 0.5 - rate > 2.0 * se:
        return "fade (right about the past, wrong about the future)"
    return "noise"


# ------------------------------------------------------------- orchestration
class WarmStarter:
    """Runs the fetch + replay on a worker thread; never blocks the engine."""

    def __init__(self, ledger, *, asset: str = "BTC", hours: float | None = None, brain=None, forward=None) -> None:
        self.ledger = ledger
        self.forward = forward
        self.asset = asset
        env_hours = os.environ.get("WARM_START_HOURS", "").strip()
        self.hours = float(hours if hours is not None else (env_hours or 24.0))
        self.enabled = os.environ.get("WARM_START", "1").strip().lower() not in ("0", "false", "no")
        self.brain = brain
        self.report = WarmStartReport(asset=asset, hours=self.hours,
                                      phase="idle" if self.enabled else "disabled",
                                      detail="" if self.enabled else "WARM_START=0")
        self._thread: threading.Thread | None = None

    def start(self) -> bool:
        if not self.enabled or (self._thread and self._thread.is_alive()):
            return False
        self._thread = threading.Thread(target=self.run, name="warm-start", daemon=True)
        self._thread.start()
        return True

    def run(self) -> WarmStartReport:
        rep = self.report
        rep.phase, rep.started_at, rep.error = "fetching", time.time(), ""
        try:
            end_ms = int(time.time() * 1000)
            start_ms = end_ms - int(self.hours * 3600 * 1000) - LOOKBACK_TICKS * 1000
            rep.detail = f"Binance 1 s klines, last {self.hours:g} h"
            bars = fetch_klines(SYMBOLS[self.asset], "1s", start_ms, end_ms)
            rep.bars = int(bars.shape[0])
            if bars.shape[0] < LOOKBACK_TICKS + 200:
                raise RuntimeError(f"Binance returned {bars.shape[0]} bars")
            hedge = None
            try:
                other = "PAXG" if self.asset == "BTC" else "BTC"
                hedge = fetch_klines(SYMBOLS[other], "1m", start_ms, end_ms)
                # one print per minute is what a thin PAXG tape looks like anyway
            except Exception as exc:  # noqa: BLE001
                log.debug("hedge history skipped: %s", exc)
            # the real mushroom body is loaded by the brain's own loop; give it
            # up to 90 s (0.2 s from the image, a few seconds from disk) so the
            # replay runs the same circuit the live windows run
            mb = None
            for _ in range(450):
                mb = getattr(self.brain, "mb", None) if self.brain is not None else None
                if mb is not None or self.brain is None:
                    break
                time.sleep(0.2)
            rep.detail = f"replaying {bars.shape[0]:,} seconds of real tape through the 31 formulas"
            replay(self.asset, bars, hedge, self.ledger, brain_mb=mb, report=rep, forward=self.forward)
            rep.detail = (f"{rep.scored} real windows scored on the NEXT window · recipe agreed with the past "
                          f"{(rep.recipe_agrees_with_past or 0) * 100:.0f} %, with the future "
                          f"{(rep.recipe_agrees_with_future or 0) * 100:.0f} %")
            log.info("warm start: %s", rep.detail)
        except Exception as exc:  # noqa: BLE001
            rep.phase = "failed"
            rep.error = f"{type(exc).__name__}: {exc}"[:300]
            rep.detail = "no history: the ledger starts blind and learns live (40 windows)"
            log.warning("warm start failed: %s", rep.error)
        finally:
            rep.finished_at = time.time()
            rep.elapsed_s = round(rep.finished_at - rep.started_at, 1)
        return rep
