"""Synthetic market tapes with **known character**, for the formula self-test.

The point of this module is to make a formula's logic falsifiable.  A formula
that claims "positive output when buyers are aggressive" can be checked by
handing it a tape where buyers are unambiguously aggressive and looking at the
sign.  A random walk cannot do that: everything looks like noise, which is
exactly how a broken normaliser stays hidden for months.

Each scenario is a *stream* of windows (not a single snapshot) because a third
of the formulas carry state - Kalman filters, EMA baselines, Hurst histories,
self-calibrating scales.  A one-shot snapshot makes those formulas return their
"not warmed up yet" value, which is correct behaviour but tells you nothing.

Every tape is deterministic: same prices, same volumes, same books.  A PASS is
reproducible and a FAIL is a real regression.

Feature map (BULL tape; the BEAR tape is the exact mirror):

===================  ==========================================================
formula              which feature of the tape it is supposed to detect
===================  ==========================================================
TAI / AFPR           upward drift with aggressive buying
SED                  spread widening right after buy prints
VSD                  a volume shock in the final window, price up
DGW / LCS / BAR      bid-heavy book, ask ladder thinner, deeper bid cliff
HRDD / SHRP / GCDV   PAXG tracking BTC (intact hedge, ratio stable)
HSI                  correlated legs -> hedge works (value well under 0.8)
RSV                  chop in the early windows, trend in the last ones
VSS                  volatility expansion in the final window
ERC                  trend, not chop (sample entropy low)
MCPE                 the last window ends just after an up-crossing
MPS                  positively autocorrelated returns with an upward drift
TWRS                 upside-skewed return tails
DSKD                 fast Kalman estimate above the slow one
NIV / SMD            bullish headlines
KCAE / CCSv2         the whole ensemble pointing the same way
===================  ==========================================================
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from backend.core.frozen_snapshot import FrozenMarketSnapshot

LEVELS = 20
TICKS_PER_WINDOW = 600
WINDOWS = 48
#: Micro-cycle period, in ticks: 30 ticks = 3 s at 10 ticks/s.
CYCLE_PERIOD_TICKS = 30
WINDOW_SECONDS = 60.0
END_MS = 1_700_000_000_000.0
SEED = 20240924

SCENARIOS = (
    "BULL",
    "BEAR",
    "FLAT",
    "STRESS",
    "IMPULSE_UP",
    "IMPULSE_DOWN",
    "CYCLE_UP",
    "CYCLE_DOWN",
)

#: Fixed seeds per scenario: ``hash()`` is salted per process in Python 3, so
#: using it would make the tapes different on every run.
SCENARIO_SEEDS = {
    "BULL": 11,
    "BEAR": 23,
    "FLAT": 37,
    "STRESS": 53,
    "IMPULSE_UP": 71,
    "IMPULSE_DOWN": 89,
    "CYCLE_UP": 103,
    "CYCLE_DOWN": 121,
}


@dataclass
class NewsItem:
    """Minimal news item - matches the fields NIV / SMD read."""

    headline: str
    source: str = "self-test"
    published_at: float = 1_700_000_000.0
    sentiment: float = 0.0
    credibility: float = 1.0
    url: str = ""
    tier: int = 1

    def age_seconds(self, now: float | None = None) -> float:
        return 0.0


# ---------------------------------------------------------------------------
# tape construction
# ---------------------------------------------------------------------------
def _correlated_noise(rng: np.random.Generator, n: int, rho: float) -> np.ndarray:
    """AR(1) noise: rho > 0 gives trending returns, rho < 0 gives chop."""
    shocks = rng.normal(0.0, 1.0, size=n)
    out = np.empty(n, dtype=np.float64)
    acc = 0.0
    for i, shock in enumerate(shocks):
        acc = rho * acc + np.sqrt(max(1.0 - rho * rho, 1e-9)) * shock
        out[i] = acc
    return out


def _price_path(
    *,
    direction: int,
    start: float,
    n: int,
    drift_bps_per_window: float,
    vol_bps_per_window: float,
    rho: float,
    rho_trending: float | None,
    cycle_amplitude: float,
    cycle_period: int,
    cycle_phase: float,
    skew: float,
    vol_expansion: float,
    impulse_bps: float,
    final_drift_bps: float,
    final_window_indices: tuple[int, int],
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Tick prices with a known drift, regime, cycle, skew and volatility profile.

    * the drift is constant per window (the trend) and arrives *already*
      signed - do not multiply it by ``direction`` again, or every tape in
      the fixture turns into a bull tape;
    * ``rho`` sets the return autocorrelation; if ``rho_trending`` is given, the
      tape switches from ``rho`` to ``rho_trending`` at 60 % of the stream - a
      deliberate regime change for RSV to find;
    * ``cycle_phase`` puts the micro-cycle in a known place, which is what MCPE
      measures (0 = the final window ends just after an up-crossing);
    * ``impulse_bps`` is an accelerating move inside the final window: real
      aggression stepping on the tape, for TAI / VSD / DSKD to detect.
    """
    rng = np.random.default_rng(seed)
    per_tick_drift = start * (drift_bps_per_window / 1e4) / TICKS_PER_WINDOW
    per_tick_vol = start * (vol_bps_per_window / 1e4) / np.sqrt(TICKS_PER_WINDOW)

    switch = int(n * 0.6)
    noise = np.concatenate(
        [
            _correlated_noise(rng, switch, rho),
            _correlated_noise(rng, n - switch, rho_trending if rho_trending is not None else rho),
        ]
    )
    if skew:
        # A few outsized moves in the direction of the skew (squeeze / stop run).
        # 0.5 % of ticks, five sigma each: enough to make the third moment
        # visible over a 3000-tick horizon without swamping the trend.
        jumps = np.where(rng.random(n) < 0.005, skew * 5.0, 0.0)
        noise = noise + jumps

    vol_profile = np.ones(n, dtype=np.float64)
    if vol_expansion != 1.0:
        lo, hi = final_window_indices
        ramp = np.linspace(1.0, vol_expansion, max(1, hi - lo))
        vol_profile[lo:hi] = ramp[: hi - lo]

    steps = per_tick_drift + noise * per_tick_vol * vol_profile
    prices = start + np.cumsum(steps)

    # The impulse: the last 30 % of the final window, cubic so that the
    # acceleration (third derivative) is genuinely positive rather than a step.
    lo, hi = final_window_indices
    if impulse_bps:
        start_i = lo + int((hi - lo) * 0.7)
        t = np.arange(start_i, hi, dtype=np.float64)
        progress = (t - start_i) / max(1.0, (hi - start_i))
        prices[start_i:hi] += direction * start * (impulse_bps / 1e4) * progress**3

    # The settlement ramp: a linear, direction-signed move across the *scored*
    # window.  Without it the final 600 ticks are dominated by the tape's own
    # random walk (one bar's move is about one sigma of that walk, so a bull
    # tape's last window lands negative a third of the time and every
    # direction-sensitive formula reads a coin flip instead of the tape).  A
    # straight line adds no acceleration, so the jerk formulas are unaffected.
    if final_drift_bps:
        ramp = np.linspace(0.0, 1.0, max(1, hi - lo))
        prices[lo:hi] += direction * start * (final_drift_bps / 1e4) * ramp[: hi - lo]

    # A deterministic micro-cycle with a controlled phase.  Only the tape built
    # for the phase estimator carries one: a market-wide oscillation would show
    # up as a hedge breakdown in HSI, and a BTC-only one is a genuine feature.
    if cycle_amplitude:
        i = np.arange(n, dtype=np.float64)
        prices = prices + start * cycle_amplitude * np.sin(
            2.0 * np.pi * i / cycle_period + cycle_phase
        )
    return prices, vol_profile


def _volumes_and_sides(
    *,
    direction: int,
    n: int,
    aggression: float,
    shock_window: tuple[int, int] | None,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    base = 0.35 * (1.0 + 0.5 * np.abs(np.sin(np.arange(n) / 11.0)))
    volumes = base * np.where(rng.random(n) < 0.05, 5.0, 1.0)

    buy_probability = 0.5 + 0.5 * aggression * direction
    sides = np.where(rng.random(n) < buy_probability, 1.0, -1.0)

    if shock_window is not None and direction != 0:
        lo, hi = shock_window
        volumes[lo:hi] *= 8.0
        sides[lo:hi] = float(np.sign(direction))  # the shock is one-sided
    return volumes, sides


def _book(
    mid: float,
    tick: float,
    direction: int,
    cliff: bool,
    erosion: float = 0.0,
) -> np.ndarray:
    """Two-sided ladder with a known liquidity story.

    Bullish tape: bids are stacked *at the touch* (real support), the offers are
    thin near the market with their size parked 8-12 levels up, and the ask
    ladder ends in a cliff.  That is the configuration DGW ("the ask side is the
    empty side -> upward pull") and LCS ("the ask ladder is the broken one") are
    supposed to describe.  Bearish: the exact mirror.  Balanced: two identical
    ladders, which must read 0 on both.
    """
    idx = np.arange(LEVELS, dtype=np.float64)

    def touch_heavy() -> np.ndarray:
        return 0.78**idx

    def parked_far(with_cliff: bool) -> np.ndarray:
        shape = np.full(LEVELS, 0.06)
        shape[7:12] = 1.0                       # the wall sits 8-12 levels up
        shape[:7] = 0.10                        # thin right at the touch
        if with_cliff:
            shape[12:] *= 0.05                  # ... and then it disappears
            shape[12] = 0.02
        return shape

    if direction > 0:
        bid_shape, ask_shape = touch_heavy(), parked_far(cliff)
    elif direction < 0:
        ask_shape, bid_shape = touch_heavy(), parked_far(cliff)
    else:
        bid_shape = ask_shape = 1.0 / (1.0 + idx)

    # Erosion: the *thin* side of a directional book is the side being eaten, so
    # it shrinks over the stream while the heavy side is refilled.  A balanced
    # book erodes on both sides at once, which is why it must read 0 for BAR.
    if direction == 0:
        erode_both = max(0.15, 1.0 - 0.85 * erosion)
        bid_scale = ask_scale = erode_both
    elif direction > 0:
        bid_scale, ask_scale = 1.0, max(0.15, 1.0 - 0.85 * erosion)
    else:
        bid_scale, ask_scale = max(0.15, 1.0 - 0.85 * erosion), 1.0

    bid_qty = (6.0 * bid_shape + 0.05) * bid_scale
    ask_qty = (6.0 * ask_shape + 0.05) * ask_scale
    bids = np.column_stack([mid - tick * (idx + 1.0), bid_qty])
    asks = np.column_stack([mid + tick * (idx + 1.0), ask_qty])
    return np.stack([bids, asks])


def _spreads(
    mid: float,
    tick: float,
    n: int,
    widening: float,
    volumes: np.ndarray,
    sides: np.ndarray,
    direction: int,
) -> np.ndarray:
    """(time_ms, spread) history with a *signed* maker-pressure response.

    This is the relationship SED exists to find: after a buy print the makers
    pull their asks and the spread widens; after a sell print they pull their
    bids and it widens too *on the other side of the book*.  Signing the
    pressure by the tape's own direction is what makes a bearish tape show the
    mirror image rather than the same positive reading.

    ``pressure`` is a decayed sum of signed aggressor volume, so the spread is
    a function of the *flow* - not of its magnitude, which would give the same
    positive response on every tape.
    """
    times = _tick_times(n)
    ramp = np.linspace(1.0, widening, n)
    half_spread = tick * 0.5 * ramp
    spread = np.empty(n, dtype=np.float64)
    pressure = 0.0
    decay = 0.90
    mean_vol = max(float(np.mean(volumes)), 1e-9)
    normalised = volumes * sides * direction / mean_vol
    # Centre the pressure: a persistently one-sided tape (88 % buys) would
    # otherwise pin it at the clamp within a few ticks, the spread would be a
    # constant, and SED's regression would have nothing to fit.  Physically this
    # is "makers widen when flow is heavier than *usual*", not when the tape is
    # simply directional.
    centre = float(np.mean(normalised))
    for i in range(n):
        pressure = decay * pressure + (float(normalised[i]) - centre)
        # Linear response, clamped so a burst cannot explode the spread:
        # 25 % of the base spread is the full maker-pull effect.
        spread[i] = half_spread[i] * (1.0 + 0.25 * max(-2.0, min(2.0, pressure)))
    return np.column_stack([times, spread])


def _tick_times(n: int, end_ms: float = END_MS, step_ms: float = 100.0) -> np.ndarray:
    """Ascending tick timestamps ending exactly at ``end_ms``."""
    return end_ms - (n - 1 - np.arange(n)) * step_ms


def _candles(prices: np.ndarray, count: int = 60) -> np.ndarray:
    idx = np.linspace(0, prices.size - 1, count).astype(int)
    return prices[idx].astype(np.float64)


def _clip_around(mid: float, offset: float) -> float:
    return float(mid + offset)


# ---------------------------------------------------------------------------
# scenario definition + snapshot stream
# ---------------------------------------------------------------------------
@dataclass
class Scenario:
    key: str
    label: str
    description: str
    direction: int  # +1 bullish, -1 bearish, 0 balanced
    snapshots: list[FrozenMarketSnapshot] = field(default_factory=list)
    hedged: bool = True  # PAXG moves with BTC (the hedge is intact)

    @property
    def last(self) -> FrozenMarketSnapshot:
        return self.snapshots[-1]


def _build_scenario(key: str) -> Scenario:
    spec = {
        "BULL": dict(
            direction=1,
            label="Bullish tape",
            description=(
                "48 windows rising ~6 bps each, mean-reverting early and trending after the "
                "regime switch, with an upside skew and an upward micro-cycle placed so the "
                "last window ends just after an up-crossing. 82 % of the volume hits the offer, "
                "the spread widens when the tape is busy, bids are stacked at the touch while "
                "the ask ladder breaks above level 6, and the final window carries an "
                "accelerating buy impulse on 6x volume. PAXG is negatively correlated (the "
                "hedge works)."
            ),
            rho=-0.40,
            rho_trending=0.45,
            skew=1.0,
            aggression=0.64,
            vol_expansion=1.8,
            widening=1.6,
            impulse_bps=0.0,
            cycle_phase=0.62,
            cycle_amp_bps=0.0,
            hedge_beta=-0.40,
            own_vol_bps=0.5,
            final_drift_bps=60.0,
            news=(NewsItem("Bitcoin ETF sees record inflows as institutions buy", sentiment=0.95),
                  NewsItem("Analysts raise BTC targets after breakout", sentiment=0.8)),
        ),
        "BEAR": dict(
            direction=-1,
            label="Bearish tape",
            description=(
                "The exact mirror: falling ~6 bps per window with a downside skew, a downward "
                "micro-cycle, 82 % of the volume hitting the bid, offers stacked above while the "
                "bid ladder breaks, and an accelerating sell impulse on 6x volume in the final "
                "window. PAXG is negatively correlated (the hedge works)."
            ),
            rho=-0.40,
            rho_trending=0.45,
            skew=-1.0,
            aggression=0.64,
            vol_expansion=1.8,
            widening=1.6,
            impulse_bps=0.0,
            cycle_phase=3.76,
            cycle_amp_bps=0.0,
            hedge_beta=-0.40,
            own_vol_bps=0.5,
            final_drift_bps=60.0,
            news=(NewsItem("Exchange halts withdrawals as Bitcoin plunges", sentiment=-0.95),
                  NewsItem("Regulators announce crackdown on crypto trading", sentiment=-0.8)),
        ),
        "FLAT": dict(
            direction=0,
            label="Balanced / choppy tape",
            description=(
                "No drift at all, mean-reverting returns (negative autocorrelation) the whole "
                "way through, symmetric book with identical ladders on both sides, no volume "
                "shock, no skew, stable volatility, PAXG independent of BTC. This is the tape "
                "that must produce values near zero."
            ),
            rho=-0.45,
            rho_trending=None,
            skew=0.0,
            aggression=0.0,
            vol_expansion=1.0,
            widening=1.0,
            impulse_bps=0.0,
            cycle_phase=1.80,
            cycle_amp_bps=0.0,
            hedge_beta=0.0,
            news=(NewsItem("Markets quiet ahead of the weekend", sentiment=0.0),),
        ),
        "IMPULSE_UP": dict(
            direction=1,
            label="Buy impulse",
            description=(
                "A quiet, hedged tape whose final window carries one thing: an accelerating "
                "buy impulse over the last 30 % of the window, on 8x volume. This is the tape "
                "for the acceleration / volume-shock / state-estimation formulas."
            ),
            rho=-0.40,
            rho_trending=None,
            skew=0.0,
            aggression=0.70,
            vol_expansion=1.2,
            widening=1.4,
            impulse_bps=22.0,
            cycle_phase=0.0,
            cycle_amp_bps=0.0,
            hedge_beta=-0.40,
            own_vol_bps=0.5,
            final_drift_bps=45.0,
            news=(NewsItem("Large buyer steps in, lifting every offer", sentiment=0.6),),
        ),
        "IMPULSE_DOWN": dict(
            direction=-1,
            label="Sell impulse",
            description="The mirror of the buy-impulse tape: an accelerating sell burst.",
            rho=-0.40,
            rho_trending=None,
            skew=0.0,
            aggression=0.70,
            vol_expansion=1.2,
            widening=1.4,
            impulse_bps=22.0,
            cycle_phase=0.0,
            cycle_amp_bps=0.0,
            hedge_beta=-0.40,
            own_vol_bps=0.5,
            final_drift_bps=45.0,
            news=(NewsItem("Large seller hits every bid", sentiment=-0.6),),
        ),
        "CYCLE_UP": dict(
            direction=0,
            label="Micro-cycle, rising phase",
            description=(
                "Flat tape with exactly one feature: a 3 bps oscillation with a 3-second "
                "period, placed so the last window ends three ticks after an up-crossing. "
                "The phase estimator must read the rising half of the cycle."
            ),
            rho=-0.40,
            rho_trending=None,
            skew=0.0,
            aggression=0.0,
            vol_expansion=1.0,
            widening=1.0,
            impulse_bps=0.0,
            cycle_phase=0.62,
            cycle_amp_bps=3.0,
            hedge_beta=-0.40,
            own_vol_bps=0.5,
            news=(),
        ),
        "CYCLE_DOWN": dict(
            direction=0,
            label="Micro-cycle, falling phase",
            description=(
                "The mirror: the same oscillation with the last window ending three ticks "
                "after a down-crossing."
            ),
            rho=-0.40,
            rho_trending=None,
            skew=0.0,
            aggression=0.0,
            vol_expansion=1.0,
            widening=1.0,
            impulse_bps=0.0,
            cycle_phase=3.76,
            cycle_amp_bps=3.0,
            hedge_beta=-0.40,
            own_vol_bps=0.5,
            news=(),
        ),
        "STRESS": dict(
            direction=1,
            label="Hedge breakdown",
            description=(
                "BTC trends up while PAXG amplifies the same moves (correlation 1.0): the "
                "blend is as wild as either leg, which is exactly what the HSI dampening "
                "threshold exists for. The spread widens 3x and the hedge ratio drifts."
            ),
            rho=-0.40,
            rho_trending=0.45,
            skew=0.6,
            aggression=0.5,
            vol_expansion=2.2,
            widening=3.0,
            impulse_bps=10.0,
            cycle_phase=0.62,
            cycle_amp_bps=0.0,
            hedge_beta_ramp=(2.4, 1.0),
            own_vol_bps=0.35,
            final_drift_bps=60.0,
            news=(NewsItem("Gold now trades like risk: correlation with crypto hits 1.0",
                           sentiment=0.1),),
        ),
    }[key]

    direction = spec["direction"]
    n = TICKS_PER_WINDOW * WINDOWS
    btc_start, paxg_start = 68_000.0, 2_400.0

    seed = SEED + SCENARIO_SEEDS.get(key, 5)
    del seed  # the per-block generators below carry their own fixed seeds

    final_lo = TICKS_PER_WINDOW * (WINDOWS - 1)
    # The micro-cycle is loud inside the final window only: it is there for the
    # phase estimator and must not drown the other formulas.
    cycle_amp = spec["cycle_amp_bps"] / 1e4
    btc_prices, _ = _price_path(
        direction=direction,
        start=btc_start,
        n=n,
        drift_bps_per_window=0.0 if direction == 0 else 6.0 * direction,
        vol_bps_per_window=5.0,
        rho=spec["rho"],
        rho_trending=spec["rho_trending"],
        cycle_amplitude=cycle_amp,
        cycle_period=CYCLE_PERIOD_TICKS,
        cycle_phase=spec["cycle_phase"],
        skew=spec["skew"],
        vol_expansion=spec["vol_expansion"],
        impulse_bps=spec["impulse_bps"],
        final_drift_bps=spec.get("final_drift_bps", 0.0),
        final_window_indices=(final_lo, n),
        seed=SEED,
    )

    # PAXG is built from the *returns* of BTC: a negative beta means the pair
    # actually hedges (HSI low), beta ~ +1.3 means gold just amplifies crypto
    # (no diversification at all: HSI high).  That is the physical difference
    # between the calm tapes and the breakdown tape.
    btc_returns = np.diff(np.log(btc_prices))
    own = _correlated_noise(np.random.default_rng(SEED + 31), btc_returns.size, -0.2)
    own_noise = spec.get("own_vol_bps", 0.6) * 1e-4
    ramp = spec.get("hedge_beta_ramp")
    if ramp:
        # A hedge ratio that moves: it starts by amplifying every BTC move and
        # ends much closer to neutral, so the pair's relationship is *drifting*
        # (what HRDD scores) while still being tightly coupled (what HSI sees).
        betas = np.linspace(float(ramp[0]), float(ramp[1]), btc_returns.size)
        paxg_returns = betas * btc_returns + own * own_noise
    else:
        paxg_returns = spec["hedge_beta"] * btc_returns + own * own_noise
    paxg_prices = paxg_start * np.exp(np.concatenate([[0.0], np.cumsum(paxg_returns)]))



    btc_volumes, btc_sides = _volumes_and_sides(
        direction=direction,
        n=n,
        aggression=spec["aggression"],
        # A burst in the last 15 ticks: long enough to be a shock, short enough
        # that the baseline sample stays normal.
        shock_window=((n - 15, n) if spec["impulse_bps"] else None),
        rng=np.random.default_rng(SEED + 3 + SCENARIO_SEEDS.get(key, 5)),
    )
    paxg_volumes, paxg_sides = _volumes_and_sides(
        # PAXG's aggressor side follows the sign of its own move, which is what a
        # negatively correlated hedge leg would look like.
        direction=(
            -direction
            if (spec.get("hedge_beta", spec.get("hedge_beta_ramp", (1.0,))[0])) < 0
            else direction
        ),
        n=n, aggression=spec["aggression"] * 0.6,
        shock_window=None, rng=np.random.default_rng(SEED + 5),
    )

    btc_tick = btc_start * 0.00004
    paxg_tick = paxg_start * 0.00008
    times = _tick_times(n)
    btc_ticks = np.column_stack([times, btc_prices, btc_volumes, btc_sides])
    paxg_ticks = np.column_stack([times, paxg_prices, paxg_volumes, paxg_sides])

    from backend.data import cross_asset_sync as sync

    snapshots: list[FrozenMarketSnapshot] = []
    for w in range(WINDOWS):
        lo = w * TICKS_PER_WINDOW
        hi = lo + TICKS_PER_WINDOW
        btc_window = btc_ticks[lo:hi]
        paxg_window = paxg_ticks[lo:hi]
        mid = float(btc_window[-1, 1])
        paxg_mid = float(paxg_window[-1, 1])

        # The previous book is a slightly different ladder, so absorption and
        # cliff formulas see a real change between snapshots.
        # Erosion grows monotonically over the stream, so "now" is always a
        # little thinner than "15 seconds ago" on the side being eaten.
        erosion = 0.05 + 0.9 * (w / max(1, WINDOWS - 1))
        book = _book(mid, btc_tick, direction, cliff=True, erosion=erosion)
        book_prev = _book(
            _clip_around(mid, -btc_tick), btc_tick, direction, cliff=True,
            erosion=max(0.0, erosion - 0.06),
        )
        payload = {
            "timestamp": END_MS / 1000.0 - (WINDOWS - 1 - w) * WINDOW_SECONDS,
            "btc_ticks": btc_window,
            "paxg_ticks": paxg_window,
            "btc_book": book,
            "paxg_book": _book(paxg_mid, paxg_tick, direction, cliff=False),
            "btc_book_prev": book_prev,
            "paxg_book_prev": _book(_clip_around(paxg_mid, -paxg_tick), paxg_tick, direction, cliff=False),
            "btc_candles": _candles(btc_prices[max(0, hi - TICKS_PER_WINDOW * 12) : hi]),
            "paxg_candles": _candles(paxg_prices[max(0, hi - TICKS_PER_WINDOW * 12) : hi]),
            "news_items": tuple(spec["news"]),
            "drg_outcomes": np.asarray([[1.0, 12.0]] * 10, dtype=np.float64),
            "btc_tick_count": TICKS_PER_WINDOW,
            "paxg_tick_count": TICKS_PER_WINDOW,
            "btc_spreads": _spreads(
                mid, btc_tick, TICKS_PER_WINDOW, spec["widening"],
                btc_window[:, 2], btc_window[:, 3], direction,
            ),
            "paxg_spreads": _spreads(
                paxg_mid, paxg_tick, TICKS_PER_WINDOW, spec["widening"],
                paxg_window[:, 2], paxg_window[:, 3], direction,
            ),
            "warnings": (),
            "source": "self-test",
        }
        payload["synced"] = sync.synchronise(
            btc_window, paxg_window, now=payload["timestamp"]
        )
        snapshots.append(FrozenMarketSnapshot(**payload))

    return Scenario(
        key=key,
        label=spec["label"],
        description=spec["description"],
        direction=direction,
        snapshots=snapshots,
        hedged=bool(spec.get("hedged", True)),
    )


_cache: dict[str, Scenario] = {}


def scenario(key: str) -> Scenario:
    key = key.upper()
    if key not in _cache:
        if key not in SCENARIOS:
            raise KeyError(f"unknown scenario {key!r}; expected one of {SCENARIOS}")
        _cache[key] = _build_scenario(key)
    return _cache[key]


def all_scenarios() -> dict[str, Scenario]:
    return {key: scenario(key) for key in SCENARIOS}
