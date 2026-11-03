"""Round BH - adaptive number formatting for every human-facing string.

``fx(v, d)`` prints ``v`` with ``d`` decimals, but a value that would print as
0.00 is shown with as many decimals as reveal its first significant digit (up
to 6; exponent form below that).  A true zero prints as 0.00.  ``fxs`` is the
signed form.  The dashboard has the same helper in app.js.
"""
from __future__ import annotations

import math


def fx(v, d: int = 2) -> str:
    try:
        n = float(v)
    except (TypeError, ValueError):
        return "—"
    if not math.isfinite(n):
        return "—"
    if n == 0.0:
        return f"{0.0:.{d}f}"
    k = d
    while k < 6 and abs(n) < 0.5 * 10 ** (-k):
        k += 1
    if abs(n) < 0.5 * 10 ** (-k):
        return f"{n:.1e}"
    return f"{n:.{k}f}"


def fxs(v, d: int = 2) -> str:
    out = fx(v, d)
    return out if out == "—" or out.startswith("-") else "+" + out
