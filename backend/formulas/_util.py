"""Shared numeric helpers for the formula engine.

Nothing here is formula-specific; it exists so that every formula module can be
read as a direct transcription of its specification.
"""

from __future__ import annotations

import numpy as np

EPS = 1e-12
"""Universal epsilon - prevents division by zero without masking real values."""


def tanh(x: float) -> float:
    """Saturating normaliser used by (almost) every formula output."""
    return float(np.tanh(float(x)))


def clip(x: float, lo: float = -1.0, hi: float = 1.0) -> float:
    return float(np.clip(float(x), lo, hi))


def safe_div(num: float, den: float, eps: float = EPS) -> float:
    return float(num) / (float(den) + eps)


def finite(x: float, default: float = 0.0) -> float:
    """Replace NaN/Inf with a neutral value - a broken formula must not poison CCSv2."""
    return float(x) if np.isfinite(x) else float(default)


class RunningVariance:
    """Exponentially weighted variance with a decay factor (TAI, DSKD).

    ``RunningVariance(0.999)`` implements

        sigma^2 <- 0.999 * sigma^2 + 0.001 * x^2
    """

    __slots__ = ("decay", "_var", "count")

    def __init__(self, decay: float = 0.999, initial: float = 0.0) -> None:
        self.decay = float(decay)
        self._var = float(initial)
        self.count = 0

    def update(self, x: float) -> float:
        self._var = self.decay * self._var + (1.0 - self.decay) * float(x) ** 2
        self.count += 1
        return self._var

    @property
    def variance(self) -> float:
        return self._var

    def std(self, floor: float = 1e-18) -> float:
        return float(np.sqrt(max(self._var, floor)))

    def warm(self) -> bool:
        """Has the estimator seen enough samples to be trusted?"""
        return self.count >= 30

    def to_dict(self) -> dict:
        return {"decay": self.decay, "var": self._var, "count": self.count}

    @classmethod
    def from_dict(cls, payload: dict) -> "RunningVariance":
        obj = cls(payload.get("decay", 0.999), payload.get("var", 0.0))
        obj.count = int(payload.get("count", 0))
        return obj


class Ema:
    """Plain exponential moving average with an explicit span (HRDD)."""

    __slots__ = ("alpha", "value", "count")

    def __init__(self, span: float) -> None:
        self.alpha = 2.0 / (float(span) + 1.0)
        self.value = 0.0
        self.count = 0

    def update(self, x: float) -> float:
        x = float(x)
        self.count += 1
        if self.count == 1:
            self.value = x
        else:
            self.value = self.alpha * x + (1.0 - self.alpha) * self.value
        return self.value

    @property
    def ready(self) -> bool:
        return self.count >= 5

    def to_dict(self) -> dict:
        return {"alpha": self.alpha, "value": self.value, "count": self.count}

    @classmethod
    def from_dict(cls, payload: dict) -> "Ema":
        obj = cls(span=max(1.0, 2.0 / max(payload.get("alpha", 0.5), 1e-9) - 1.0))
        obj.value = float(payload.get("value", 0.0))
        obj.count = int(payload.get("count", 0))
        return obj


class RollingWindow:
    """Fixed-length float window with running mean/std (DSKD spread scaling)."""

    __slots__ = ("_buf", "_n", "_i", "_sum", "_sumsq")

    def __init__(self, size: int) -> None:
        self._buf = np.zeros(size, dtype=np.float64)
        self._n = 0
        self._i = 0
        self._sum = 0.0
        self._sumsq = 0.0

    def push(self, x: float) -> None:
        x = float(x)
        if self._n == self._buf.size:
            old = self._buf[self._i]
            self._sum -= old
            self._sumsq -= old * old
            self._n -= 1
        self._buf[self._i] = x
        self._i = (self._i + 1) % self._buf.size
        self._n += 1
        self._sum += x
        self._sumsq += x * x

    @property
    def count(self) -> int:
        return self._n

    def mean(self) -> float:
        return self._sum / self._n if self._n else 0.0

    def std(self, floor: float = 1e-15) -> float:
        if self._n < 2:
            return floor
        var = max(self._sumsq / self._n - (self._sum / self._n) ** 2, 0.0)
        return float(np.sqrt(max(var, floor)))


def ols_slope(x: np.ndarray, y: np.ndarray) -> float:
    """Least-squares slope of y on x (used by HRDD and MCPE detrending)."""
    n = min(x.size, y.size)
    if n < 2:
        return 0.0
    x = x[:n].astype(np.float64)
    y = y[:n].astype(np.float64)
    xm, ym = x.mean(), y.mean()
    xc = x - xm
    denom = float(np.dot(xc, xc))
    if denom <= EPS:
        return 0.0
    return float(np.dot(xc, y - ym) / denom)


def mad(values: np.ndarray) -> float:
    """Median absolute deviation, scaled to be a consistent sigma estimator."""
    if values.size == 0:
        return 0.0
    med = float(np.median(values))
    return 1.4826 * float(np.median(np.abs(values - med)))


def hurst_rs(series: np.ndarray, min_chunk: int = 2) -> float:
    """Rescaled-range estimate of the Hurst exponent for one window (RSV).

    Splits the window into two halves and averages the rescaled range, which is
    the fast O(n) variant the specification asks for (8 windows x 20 points).
    """
    n = series.size
    if n < 2 * min_chunk:
        return 0.5
    chunk = max(min_chunk, n // 2)
    ratios: list[float] = []
    for start in range(0, n - chunk + 1, chunk):
        seg = series[start : start + chunk].astype(np.float64)
        if seg.size < 2:
            continue
        mean = seg.mean()
        dev = np.cumsum(seg - mean)
        spread = float(dev.max() - dev.min())
        std = float(seg.std(ddof=1)) if seg.size > 1 else 0.0
        if std <= EPS or spread <= EPS:
            continue
        ratios.append(spread / std)
    if not ratios:
        return 0.5
    rs = float(np.mean(ratios))
    if rs <= 0:
        return 0.5
    # H = ln(R/S) / ln(chunk); clamp to a sane band - tiny windows are noisy.
    h = float(np.log(rs) / np.log(chunk))
    return float(np.clip(h, 0.0, 1.0))
