"""FORMULA 13 - Volatility Surprise Score (VSS).

Compares realised volatility over the last window (53 s) with the 5-window baseline
that the market has "priced in".  A positive surprise means the tape is waking
up - large moves are becoming likely.  The magnitude is then multiplied by the
sign of the recent mean return, so a volatility spike *with* upward drift reads
bullish while a spike with downward drift reads bearish.

Brain mapping: giant fibre (GF) - the fly's startle neuron, which amplifies the
brain's response to everything else when it fires.

Latency budget: < 0.2 ms over <= 300 returns.
"""

from __future__ import annotations

import numpy as np

from backend.core import config as cfg

from backend.formulas._util import EPS, finite, tanh, trace

NAME = "VSS"
CATEGORY = "D"
TITLE = "Volatility Surprise Score"
BRAIN_NODE = "Giant Fiber (GF)"
DIRECTIONAL = True
LATENCY_MS = 0.2
DESCRIPTION = "Realised one-window volatility vs. the 5-window baseline, signed by drift."

SHORT_WINDOW = 60
LONG_WINDOW = 300
#: The direction is only reported when the drift inside the baseline window is
#: statistically visible (|t| >= 1.5).  The 1-tick volatility is ~5x the 1-tick
#: drift, so a 60-tick mean return flips sign at random - that is where v2.0.0's
#: "volatility surprise" got its random +/- readings from.
MIN_T_STAT = 1.5


class State:
    __slots__ = ("last_sigma_short", "last_sigma_long")

    def __init__(self) -> None:
        self.last_sigma_short = 0.0
        self.last_sigma_long = 0.0

    def to_dict(self) -> dict:
        return {"last_sigma_short": self.last_sigma_short, "last_sigma_long": self.last_sigma_long}

    @classmethod
    def from_dict(cls, payload: dict) -> "State":
        obj = cls()
        obj.last_sigma_short = float(payload.get("last_sigma_short", 0.0))
        obj.last_sigma_long = float(payload.get("last_sigma_long", 0.0))
        return obj


def _returns(prices: np.ndarray) -> np.ndarray:
    if prices.size < 2:
        return np.zeros(0)
    safe = np.maximum(prices, EPS)
    return np.diff(np.log(safe))


def compute(snapshot, asset: str, state: State, params: dict, ctx: dict | None = None) -> float:
    prices = snapshot.prices(asset)
    if prices.size < 10:
        return 0.0

    returns = _returns(prices)
    if returns.size < 10:
        return 0.0

    # Round AW: the windows are *time*, not tick counts.  "60 ticks" was 6 s
    # on a 10 Hz tape and a minute on a thin one - the surprise then compared
    # whatever the feed rate happened to be.  Short = the last window (53 s),
    # long = the last five windows, both on the tape's own timestamps; the
    # tick counts are only the fallback when the tape carries no clock.
    n_short, n_long = min(SHORT_WINDOW, returns.size), min(LONG_WINDOW, returns.size)
    try:
        times_s = np.asarray(snapshot.ticks(asset))[:, 0] / 1000.0
        if times_s.size == prices.size and times_s.size > 2:
            now_s = float(times_s[-1])
            w = float(cfg.window_seconds())
            n_short = int(np.count_nonzero(times_s[1:] > now_s - w))
            n_long = int(np.count_nonzero(times_s[1:] > now_s - 5.0 * w))
    except Exception:  # noqa: BLE001 - a tape without a time column keeps the tick counts
        pass
    n_long = max(10, min(n_long, returns.size))
    # a tape shorter than two windows cannot hold a window *and* a baseline:
    # keep the 1:5 ratio inside whatever span there is (and say so)
    if n_short > 0.6 * n_long:
        n_short = max(10, n_long // 5)
        trace(ctx, "baseline note", 1, "tape shorter than two windows - short window = a fifth of the span")
    n_short = max(10, min(n_short, returns.size))
    n_long = max(n_short, n_long)
    short = returns[-n_short:]
    long = returns[-n_long:]
    trace(ctx, "short window", n_short, f"returns inside the last {cfg.window_seconds():.0f} s")
    trace(ctx, "baseline window", n_long, f"returns inside the last {5 * cfg.window_seconds():.0f} s")

    sigma_real = float(np.sqrt(np.mean(short**2)))
    sigma_exp = float(np.sqrt(np.mean(long**2)))
    state.last_sigma_short, state.last_sigma_long = sigma_real, sigma_exp

    if sigma_exp <= EPS:
        return 0.0

    surprise = (sigma_real - sigma_exp) / (sigma_exp + EPS)
    drift = float(np.mean(long))
    t_stat = abs(drift) * float(np.sqrt(long.size)) / (sigma_exp + EPS)
    direction = float(np.sign(drift)) if t_stat >= MIN_T_STAT else 0.0

    trace(ctx, "realised vol (window)", sigma_real, "per tick")
    trace(ctx, "baseline vol (5 windows)", sigma_exp, "per tick")
    trace(ctx, "vol surprise", surprise, "sigma_short / sigma_long - 1")
    trace(ctx, "drift (baseline window)", drift, "mean log return per tick")
    trace(ctx, "drift t-statistic", t_stat, "|mean| / (sigma / sqrt(n))")
    trace(ctx, "direction used", direction, "+1 / 0 / -1")

    if direction == 0.0:
        return 0.0
    return finite(tanh(surprise) * direction)


DOUBLE_CHECK = "value = tanh(vol surprise) x drift direction"


def double_check(t: dict, asset: str) -> float:
    """Independent re-derivation of the output from the traced intermediates."""
    if "vol surprise" not in t:
        return 0.0
    return tanh(float(t["vol surprise"])) * float(t["direction used"])
