"""Deterministic BTC + PAXG market simulator.

Purpose
-------
Binance, CoinGecko and the news APIs are all reachable from a normal machine but
may be blocked in restricted sandboxes / CI.  When that happens the app must
still be demonstrable end-to-end rather than sitting in "NO DATA -> forced
HOLD" (degradation level 6).  ``MarketSimulator`` therefore synthesises a
*statistically plausible* BTC/PAXG tape: two correlated assets with a
risk-on / risk-off regime that rotates capital between them, an order book with
liquidity walls and cliffs, and occasional flash moves.

It is deliberately *not* a random walk: the regime process is what makes the
hedge formulas (HRDD, SHRP, GCDV, HSI) produce genuinely varying output, which
is what you want when validating the pipeline.

The simulator is never used while a real feed is healthy - see
``source_manager.py``.
"""

from __future__ import annotations

import asyncio
import logging
import math
import time
from dataclasses import dataclass, field

import numpy as np

from backend.core import config as cfg

log = logging.getLogger("drosophila.simulator")


@dataclass
class AssetSimState:
    symbol: str
    price: float
    sigma_1s: float
    tick_rate: float  # trades per second
    lot_scale: float  # typical trade size
    spread_bps: float
    drift: float = 0.0
    rng: np.random.Generator | None = None
    candles: list[float] = field(default_factory=list)
    book_centres: list[float] = field(default_factory=list)


class MarketSimulator:
    """Generates ticks, L2 snapshots and candles for BTC/USDT and PAXG/USDT."""

    def __init__(self, seed: int | None = None) -> None:
        seed = cfg.SETTINGS.simulator_seed if seed is None else seed
        master = np.random.default_rng(seed)
        self._regime = 0.0  # -1 = risk-off (gold bid), +1 = risk-on (BTC bid)
        self._regime_target = 0.15
        self._regime_clock = 0.0
        self.assets: dict[str, AssetSimState] = {
            "BTC": AssetSimState(
                symbol="BTCUSDT",
                price=64_000.0 + float(master.uniform(-2000, 2000)),
                sigma_1s=0.00022,
                tick_rate=9.0,
                lot_scale=0.012,
                spread_bps=0.35,
                rng=np.random.default_rng(seed + 1),
            ),
            "PAXG": AssetSimState(
                symbol="PAXGUSDT",
                price=2_420.0 + float(master.uniform(-40, 40)),
                sigma_1s=0.00013,
                tick_rate=1.6,
                lot_scale=0.35,
                spread_bps=1.6,
                rng=np.random.default_rng(seed + 2),
            ),
        }
        self._last_book: dict[str, np.ndarray] = {}

    # ------------------------------------------------------------------
    # Regime process — drives hedge rotation
    # ------------------------------------------------------------------
    def _step_regime(self, dt: float) -> None:
        self._regime_clock += dt
        if self._regime_clock > 18.0:  # re-aim every ~18 s
            self._regime_clock = 0.0
            new_target = float(np.clip(self._regime_target + np.random.default_rng().normal(0, 0.9), -1, 1))
            self._regime_target = new_target
        # OU pull toward the target
        self._regime += (self._regime_target - self._regime) * min(1.0, dt / 6.0)
        self._regime = float(np.clip(self._regime, -1.0, 1.0))

    # ------------------------------------------------------------------
    # Tick generation
    # ------------------------------------------------------------------
    def step(self, dt: float, now: float | None = None) -> dict[str, list[tuple[float, float, float, float]]]:
        """Advance the simulation by ``dt`` seconds.

        Returns ``{asset: [(time_ms, price, qty, side), ...]}``.
        """
        now = now if now is not None else time.time()
        self._step_regime(dt)

        # Common shock so the two assets are not independent
        common = float(np.random.default_rng().normal(0, 1)) * math.sqrt(dt) * 0.00012
        out: dict[str, list[tuple[float, float, float, float]]] = {}

        for asset, state in self.assets.items():
            rng = state.rng
            assert rng is not None

            # Expected return: risk-on lifts BTC and pressures gold, and vice versa.
            beta_regime = 0.00035 if asset == "BTC" else -0.00022
            drift = beta_regime * self._regime * dt
            state.drift = drift

            shock = rng.normal(0.0, state.sigma_1s * math.sqrt(dt))
            ret = drift + shock + (common * (1.0 if asset == "BTC" else 0.18))
            state.price = max(1.0, state.price * math.exp(ret))

            # Occasional liquidity burst: volume spike + momentum kick
            burst = rng.random() < (0.035 * dt) * 10
            if burst:
                kick = rng.normal(0, 1) * state.sigma_1s * 6.0
                state.price = max(1.0, state.price * math.exp(kick))

            n_ticks = rng.poisson(state.tick_rate * dt)
            if n_ticks == 0:
                out[asset] = []
                continue

            rows: list[tuple[float, float, float, float]] = []
            span_ms = dt * 1000.0
            for k in range(n_ticks):
                ts = now * 1000.0 - span_ms + (k + 0.5) * span_ms / n_ticks
                px = state.price * (1.0 + rng.normal(0, 1) * state.spread_bps * 1e-4)
                qty = float(abs(rng.lognormal(mean=math.log(state.lot_scale), sigma=1.1)))
                if burst:
                    qty *= 4.0
                # Trades lean with the regime (aggressor imbalance)
                p_buy = 0.5 + (0.09 if asset == "BTC" else -0.06) * self._regime
                side = 1.0 if rng.random() < float(np.clip(p_buy, 0.15, 0.85)) else -1.0
                rows.append((ts, px, qty, side))
            if burst:
                state.price = max(1.0, state.price * math.exp(rng.normal(0, 1) * state.sigma_1s * 3.0))
            out[asset] = rows

        return out

    # ------------------------------------------------------------------
    # Order book
    # ------------------------------------------------------------------
    def book(self, asset: str, now: float | None = None) -> np.ndarray:
        """Build a (2, 20, 2) L2 snapshot around the current mid."""
        state = self.assets[asset]
        rng = state.rng
        assert rng is not None
        now = now if now is not None else time.time()

        mid = state.price
        half_spread = mid * state.spread_bps * 1e-4
        book = np.zeros((2, cfg.L2_DEPTH_LEVELS, 2), dtype=np.float64)

        # Regime skews resting depth: risk-on = more bids, risk-off = more asks
        skew = 1.0 + 0.35 * self._regime * (1.0 if asset == "BTC" else -1.0)
        base_qty = 1.0 if asset == "BTC" else 0.06

        for side in (0, 1):  # 0 = bids, 1 = asks
            depth_scale = skew if side == 0 else (2.0 - skew)
            for level in range(cfg.L2_DEPTH_LEVELS):
                step = mid * (1.5e-5 * (level + 1))
                price = mid - half_spread - step if side == 0 else mid + half_spread + step
                # Liquidity generally thins with distance, but walls and cliffs appear
                decay = math.exp(-level / 9.0)
                noise = float(rng.lognormal(0.0, 0.45))
                qty = base_qty * depth_scale * decay * noise * (1.0 + 0.05 * level)
                if rng.random() < 0.08:  # liquidity wall
                    qty *= float(rng.uniform(3.0, 9.0))
                if rng.random() < 0.05:  # liquidity cliff
                    qty *= 0.05
                book[side, level] = (price, max(1e-8, qty))

        self._last_book[asset] = book
        return book

    # ------------------------------------------------------------------
    # Warm-up
    # ------------------------------------------------------------------
    def warm_up(self, seconds: int = 240, now: float | None = None) -> dict:
        """Pre-fill ring buffers so the first cycle already has 600 ticks.

        Returns ``{'BTC': ticks, 'PAXG': ticks, 'candles': {...}, 'book': {...}}``.
        """
        now = now if now is not None else time.time()
        ticks: dict[str, list[tuple[float, float, float, float]]] = {"BTC": [], "PAXG": []}
        candles: dict[str, list[float]] = {"BTC": [], "PAXG": []}
        dt = 0.25
        steps = int(seconds / dt)
        for i in range(steps):
            t = now - seconds + i * dt
            generated = self.step(dt, now=t)
            for asset, rows in generated.items():
                ticks[asset].extend(rows)
            if i % 4 == 3:  # every simulated second -> 1-minute candle every 240 steps
                if i % 240 == 239:
                    candles["BTC"].append(self.assets["BTC"].price)
                    candles["PAXG"].append(self.assets["PAXG"].price)

        books = {a: self.book(a, now=now) for a in self.assets}
        return {
            "BTC": np.asarray(ticks["BTC"][-cfg.TICK_BUFFER_SIZE * 2 :], dtype=np.float64),
            "PAXG": np.asarray(ticks["PAXG"][-cfg.TICK_BUFFER_SIZE * 2 :], dtype=np.float64),
            "candles": {a: np.asarray(c, dtype=np.float64) for a, c in candles.items()},
            "book": books,
        }

    def synthetic_flash_move(self, asset: str, pct: float = -6.0) -> None:
        """Inject a flash move - drives the price-based emergency detector."""
        state = self.assets[asset]
        state.price = max(1.0, state.price * (1.0 + pct / 100.0))
        log.info("Simulator injected %.1f%% flash move on %s", pct, asset)


async def run_simulator(sim: MarketSimulator, on_step) -> None:
    """Drive the simulator in real time, calling ``on_step(asset, ticks, book)``."""
    tick_interval = 0.05  # 20 Hz wire cadence
    book_every = 0.5  # depth20@100ms is throttled to 500 ms here
    last_book = time.time()
    while True:
        t0 = time.time()
        generated = sim.step(tick_interval, now=t0)
        for asset, rows in generated.items():
            if rows:
                await on_step(asset, rows, None)
        if t0 - last_book >= book_every:
            last_book = t0
            for asset in sim.assets:
                await on_step(asset, None, sim.book(asset, now=t0))
        elapsed = time.time() - t0
        await asyncio.sleep(max(0.0, tick_interval - elapsed))
