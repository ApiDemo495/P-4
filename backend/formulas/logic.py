"""The logic registry: what each formula computes, in words, with its numbers.

The Formula Explorer used to show a name, a bar and a one-line description.  That
is not enough to trust a number - or to debug one.  Every formula now publishes:

``expression``
    The formula itself, in readable maths.
``reads``
    Which market inputs it consumes (ticks, book, synced grid, news...).
``steps``
    The computation, in order, exactly the way the module implements it.
``bands``
    How to read the output: this value means this.  ``bands`` is a list of
    ``(lo, hi, text)`` ranges over the output, checked in order.
``sign``
    What a positive output means (or "not directional").
``trace``
    The names of the intermediate values the module records while it runs - the
    live numbers that turn ``steps`` into an audit trail.  The engine fills these
    in on every pass, so the UI can show the real arithmetic of the current
    window, not an example.
``why``
    The one-line reason this formula exists in the ensemble.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Logic:
    name: str
    expression: str
    reads: tuple[str, ...]
    steps: tuple[str, ...]
    bands: tuple[tuple[float, float, str], ...]
    sign: str
    trace: tuple[str, ...] = ()
    why: str = ""

    def reading(self, value: float) -> str:
        """Turn the current value into a sentence using the band table."""
        if value is None:
            return "no value yet"
        for lo, hi, text in self.bands:
            if lo <= value < hi:
                return text
        return "outside the expected range"

    def to_dict(self, value: float | None = None, trace: list[dict] | None = None) -> dict:
        payload = {
            "name": self.name,
            "expression": self.expression,
            "reads": list(self.reads),
            "steps": list(self.steps),
            "bands": [
                {"lo": lo, "hi": hi, "text": text} for lo, hi, text in self.bands
            ],
            "sign": self.sign,
            "trace_names": list(self.trace),
            "why": self.why,
        }
        if value is not None:
            payload["value"] = round(float(value), 6)
            payload["reading"] = self.reading(float(value))
        if trace is not None:
            payload["trace"] = trace
        return payload


def readings(values: dict | None) -> dict[str, str]:
    """One sentence per formula for the values of the current window."""
    if not values:
        return {}
    out: dict[str, str] = {}
    for name, value in values.items():
        entry = get(name)
        if entry is None:
            continue
        try:
            out[name] = entry.reading(float(value))
        except (TypeError, ValueError):
            continue
    return out


def _bands(*pairs: tuple[float, float, str]) -> tuple[tuple[float, float, str], ...]:
    return tuple(pairs)


#: Reading tables shared by the directional formulas so the words stay consistent.
DIRECTIONAL_SYMMETRIC = _bands(
    (0.85, 1.01, "very strong upward pressure"),
    (0.55, 0.85, "strong upward pressure"),
    (0.30, 0.55, "clear upward pressure"),
    (0.12, 0.30, "mild upward pressure"),
    (-0.12, 0.12, "balanced - no directional information"),
    (-0.30, -0.12, "mild downward pressure"),
    (-0.55, -0.30, "clear downward pressure"),
    (-0.85, -0.55, "strong downward pressure"),
    (-1.01, -0.85, "very strong downward pressure"),
)

_UNIT = " [-1, +1]"

LOGIC: dict[str, Logic] = {}


def register(logic: Logic) -> Logic:
    LOGIC[logic.name] = logic
    return logic


# ---------------------------------------------------------------------------
# Category A - micro-structure
# ---------------------------------------------------------------------------
register(
    Logic(
        name="TAI",
        expression=(
            "TAI = tanh( jerk_rel / JERK_FLOOR ),   "
            "jerk = 6 * c3 from a recency- and volume-weighted cubic fit"
        ),
        reads=("last 30 ticks: price, timestamp, volume, aggressor side", "running sigma of jerk (decay 0.999)"),
        steps=(
            "Keep a rolling 600-tick buffer of (time, price, volume, side) - the fit needs a continuous path, not just this window.",
            "EMA-smooth the price path (span 5): the literal 3rd difference of a tick series is tick noise (its magnitude scales as sigma/dt^3), a fitted curvature is not.",
            "Fit price(t) = c0 + c1*t + c2*t^2 + c3*t^3 by weighted least squares with weight = sqrt(volume) * exp(-(t_now - t)/10 s): recent, heavy prints shape the fit.",
            "jerk = 6*c3 is d^3(price)/dt^3; divide by the mean price to make it relative.",
            "Score against the materiality floor 2.6e-6 /s^3 (~4 bps of displacement in 10 s); tanh saturates into [-1, 1].",
        ),
        bands=DIRECTIONAL_SYMMETRIC,
        sign="positive = price is accelerating upward",
        trace=("ticks for this window", "jerk_recent", "sigma_jerk", "raw"),
        why="It is the earliest detectable onset of a move: curvature shows up before price.",
    )
)

register(
    Logic(
        name="AFPR",
        expression="AFPR = tanh( (sum w_i * v_i * s_i) / (sum w_i * v_i + eps) * power )",
        reads=("tick volumes and aggressor sides", "per-asset power (BTC 3.0, PAXG 2.0)"),
        steps=(
            "Weight each tick by w_i = (1 - i/n)^power so the newest trades count most.",
            "Aggressor side s_i = +1 (buy) or -1 (sell).",
            "Sum the signed weighted volume and divide by total weighted volume.",
            "Multiply by power and squash with tanh.",
        ),
        bands=DIRECTIONAL_SYMMETRIC,
        sign="positive = aggressive buyers dominate",
        trace=("ticks for this window", "signed_volume", "total_volume", "raw"),
        why="It separates *who crossed the spread* from price, which lags the crossing.",
    )
)

register(
    Logic(
        name="SED",
        expression="SED = tanh( k * beta * (mean|flow| / mean(spread)) ) * sign(net flow)",
        reads=("L2 spread history interpolated onto tick timestamps", "signed aggressor volume"),
        steps=(
            "Interpolate the (time, spread) history onto the tick timestamps of the last 30 ticks.",
            "Regress d(spread) on signed trade volume (OLS) -> beta, the spread elasticity.",
            "Scale beta by a typical flow / a typical spread so it is dimensionless.",
            "Multiply by the sign of the net flow: spread widening under buying is bullish.",
        ),
        bands=DIRECTIONAL_SYMMETRIC,
        sign="positive = makers widen offers into buying (they expect higher prices)",
        trace=("ticks for this window", "beta", "net_flow", "relative_effect"),
        why="It reads market-maker *intention*: how liquidity responds to being hit.",
    )
)

register(
    Logic(
        name="VSD",
        expression="VSD = tanh( z / 3 ) * sign(d(price)),   z = (median(recent) - median(base)) / (1.4826 * MAD)",
        reads=("tick volumes (20 recent vs 100 baseline)", "tick prices"),
        steps=(
            "Split the window into a 100-tick baseline and a 20-tick recent sample.",
            "z = (median(recent) - median(baseline)) / robust sigma (1.4826 * MAD).",
            "Take the sign of the price change over the recent sample.",
            "Multiply: a volume shock carries the direction the tape is moving.",
        ),
        bands=DIRECTIONAL_SYMMETRIC,
        sign="positive = upward price move on unusual volume",
        trace=("ticks for this window", "baseline_median", "recent_median", "mad", "z", "price_sign"),
        why="Volume without direction is noise; the sign says whether the shock is real demand.",
    )
)

# ---------------------------------------------------------------------------
# Category B - order book
# ---------------------------------------------------------------------------
register(
    Logic(
        name="DGW",
        expression="DGW = tanh( ((ask_com - mid) - (mid - bid_com)) / (mid * s) )",
        reads=("current L2 book: 20 bid levels, 20 ask levels", "volume-squared weighting"),
        steps=(
            "Compute the centre of mass of each side, weighted by quantity^2 (size dominates distance).",
            "Measure each centre's distance from the mid: bid below, ask above.",
            "If the ask centre sits further away than the bid centre, offers are thin -> upward gravity.",
            "Normalise by the mid and squash: positive = price is being pulled up.",
        ),
        bands=DIRECTIONAL_SYMMETRIC,
        sign="positive = the ask side is the emptier side (upward pull)",
        trace=(
            "ticks for this window",
            "mid",
            "bid_center_of_mass",
            "ask_center_of_mass",
            "distance_bid",
            "distance_ask",
            "relative_gravity",
        ),
        why="It measures where resting liquidity actually is, not just how much.",
    )
)

register(
    Logic(
        name="LCS",
        expression="LCS = tanh( (w_ask - w_bid) / (w_ask + w_bid + eps) ),  w = worst relative drop-off",
        reads=("current L2 ladder, both sides", "top-of-book quantity as the yardstick"),
        steps=(
            "For each side, walk the ladder outward and find the worst level-to-level drop-off.",
            "Express each drop-off as a fraction of that side's top-of-book size.",
            "Compare: a deeper cliff on the ask side means offers are withdrawing.",
            "tanh the asymmetry: positive = the ask ladder is the broken one.",
        ),
        bands=DIRECTIONAL_SYMMETRIC,
        sign="positive = the ask side has the deeper liquidity cliff (bullish)",
        trace=("ticks for this window", "ask_cliff", "bid_cliff", "ask_dropoff", "bid_dropoff"),
        why="A ladder that stops supporting itself is where the next fast move comes from.",
    )
)

register(
    Logic(
        name="BAR",
        expression="BAR = (a_ask - a_bid) / (a_ask + a_bid + eps),  a = resting qty consumed since the last snapshot",
        reads=("current top-10 book", "previous L2 snapshot"),
        steps=(
            "Total the resting quantity in the top 10 levels of each side, now and one snapshot ago.",
            "Consumption a_bid = max(0, prev_bid - now_bid), a_ask = max(0, prev_ask - now_ask).",
            "Take the signed ratio: which side got eaten.",
            "The result is already in [-1, 1]; +1 means only asks were absorbed.",
        ),
        bands=DIRECTIONAL_SYMMETRIC,
        sign="positive = offers are being absorbed (buyers are clearing the ask)",
        trace=("ticks for this window", "ask_consumed", "bid_consumed"),
        why="Absorption is what turns a bid/ask imbalance into an actual price move.",
    )
)

# ---------------------------------------------------------------------------
# Category C - hedge (BTC x PAXG)
# ---------------------------------------------------------------------------
register(
    Logic(
        name="HRDD",
        expression="d_beta = (beta_now - EMA_60(beta)) / scale;  HRDD = -tanh(d_beta) for BTC",
        reads=("synchronised 1-second BTC and PAXG return grid", "60-cycle EMA of the hedge ratio"),
        steps=(
            "Regress PAXG returns on BTC returns over the synchronised grid -> instantaneous beta.",
            "Compare beta with its 60-minute EMA baseline.",
            "Normalise the drift by the historical dispersion of beta.",
            "Negate for BTC: if the hedge ratio is drifting, BTC is the leg that is out of line.",
        ),
        bands=DIRECTIONAL_SYMMETRIC,
        sign="positive = the BTC leg is cheap relative to the pair's own history",
        trace=("ticks for this window", "beta_now", "beta_baseline", "beta_drift", "score"),
        why="It is the only formula that tells you when the hedge itself has moved.",
    )
)

register(
    Logic(
        name="SHRP",
        expression=(
            "SHRP = -R for BTC, +R for PAXG;   "
            "R = 0.5 * ( flow_PAXG / typical_PAXG - flow_BTC / typical_BTC )"
        ),
        reads=("synchronised BTC and PAXG returns", "tick flow per leg"),
        steps=(
            "Dollar flow per leg = sum of (i/T)^2 * volume * side * price over the last 60 s, a recency-weighted net flow in dollars.",
            "Normalise each leg by the flow *that leg* normally does (EMA of its own |flow|, floored at half of the pair's typical flow) - dividing a window by itself would always give +/-1.",
            "Rotation R = half the difference of the two normalised flows: +1 is a complete rotation into PAXG, -1 a complete rotation into BTC.",
            "Sign it for the asset under analysis: money leaving BTC for gold is bearish BTC.",
        ),
        bands=DIRECTIONAL_SYMMETRIC,
        sign="positive = flow rotating toward the asset being traded",
        trace=("ticks for this window", "rotation", "score"),
        why="Rotation between the two legs is what the hedge pair is for.",
    )
)

register(
    Logic(
        name="GCDV",
        expression="GCDV = tanh( mean(d(last 10)) / scale ),  d = BTC_norm - PAXG_norm",
        reads=("synchronised price paths, both legs normalised to their first grid point"),
        steps=(
            "Normalise both legs to their first synchronised value.",
            "d = BTC_norm - PAXG_norm: how far apart the two paths are right now.",
            "velocity = mean of the last 10 first-differences of d.",
            "Scale by a meaningful divergence speed (with a self-scaling floor) and squash.",
        ),
        bands=DIRECTIONAL_SYMMETRIC,
        sign="positive = BTC is outpacing PAXG (bullish rotation into risk)",
        trace=("ticks for this window", "divergence", "velocity", "scale"),
        why="Correlation is static; the *speed* of separation is what a scalper can trade.",
    )
)

register(
    Logic(
        name="HSI",
        expression="HSI = flip((sigma_hedge / sigma_avg - base) / (1 - base))  in [0, 1]",
        reads=("synchronised BTC and PAXG return grid", "hedge weight w = 0.5"),
        steps=(
            "Build the hedged return series r_h = w * r_btc + (1-w) * r_paxg.",
            "Compare sigma(r_h) with the average of the two legs' sigmas.",
            "Anchor against the *uncorrelated* baseline: 0.5 means the pair behaves like two "
            "independent assets, > 0.5 means the hedge is worse than useless, < 0.5 means it works.",
            "Clip into [0, 1] - HSI is a regime dial, never a direction.",
        ),
        bands=_bands(
            (0.80, 1.01, "hedge broken - confidence dampened, size cut"),
            (0.60, 0.80, "hedge under stress"),
            (0.45, 0.60, "pair behaving like two independent assets (neutral)"),
            (0.25, 0.45, "hedge working - diversification benefit"),
            (-0.01, 0.25, "strong hedge - legs move together"),
        ),
        sign="not directional: it scales conviction (Section 8.2), it never picks a side",
        trace=("ticks for this window", "sigma_btc", "sigma_paxg", "sigma_hedge", "neutral_baseline", "ratio"),
        why="A signal is only as trustworthy as the pair it is measured in.",
    )
)

# ---------------------------------------------------------------------------
# Category D - volatility & regime
# ---------------------------------------------------------------------------
register(
    Logic(
        name="RSV",
        expression="RSV = tanh( (H_now - H_prev) / scale ) * sign(recent drift)",
        reads=("tick prices over the last 160 ticks", "history of the Hurst exponent from previous windows"),
        steps=(
            "Split the window into 8 overlapping 20-tick sub-windows and compute R/S Hurst for each.",
            "Average them into this window's Hurst exponent H_now.",
            "Velocity = H_now - H_previous window (how fast the market's memory is changing).",
            "Sign it with the prevailing drift: rising persistence inside an up-move is bullish.",
        ),
        bands=DIRECTIONAL_SYMMETRIC,
        sign="positive = persistence is rising while price rises",
        trace=("ticks for this window", "hurst_now", "hurst_prev", "velocity", "scale", "drift"),
        why="It flags the moment a market switches between trending and mean-reverting.",
    )
)

register(
    Logic(
        name="VSS",
        expression="VSS = tanh( (sigma_1m - sigma_5m) / sigma_5m ) * sign(recent drift)",
        reads=("tick prices of the current window", "returns distribution of the window"),
        steps=(
            "sigma_1m = root-mean-square of the most recent short window of log returns.",
            "sigma_5m = the same over the long window.",
            "surprise = (sigma_1m - sigma_5m) / sigma_5m.",
            "Multiply by the sign of the recent drift: a volatility expansion means the move is real.",
        ),
        bands=DIRECTIONAL_SYMMETRIC,
        sign="positive = volatility expanding while price drifts up",
        trace=("ticks for this window", "sigma_short", "sigma_long", "surprise", "drift_sign"),
        why="It separates a genuine repricing from a quiet drift that is about to be reversed.",
    )
)

register(
    Logic(
        name="ERC",
        expression="ERC = tanh( SampEn - 1.0 )",
        reads=("log return series of the window", "embedding tolerance"),
        steps=(
            "Build template vectors of the return series (embedded, tolerance = fraction of sd).",
            "Count matching templates for length m and m+1 -> sample entropy.",
            "Compare with the neutral value 1.0: below it the series is structured, above it is noisy.",
            "tanh the difference: negative = trending, ~0 = transitional, positive = choppy.",
        ),
        bands=_bands(
            (0.30, 1.01, "choppy: patterns break down, fade moves"),
            (0.05, 0.30, "mildly noisy"),
            (-0.15, 0.05, "transitional regime"),
            (-0.40, -0.15, "structured: trend-following works"),
            (-1.01, -0.40, "highly structured: strong persistent regime"),
        ),
        sign="not directional: it describes the regime, it does not pick a side",
        trace=("ticks for this window", "sample_entropy", "tolerance", "templates"),
        why="The same formula output means different things in a trend and in chop.",
    )
)

# ---------------------------------------------------------------------------
# Category E - temporal pattern
# ---------------------------------------------------------------------------
register(
    Logic(
        name="MCPE",
        expression=(
            "MCPE = cos(2*pi*frac) * dir * damping,   "
            "frac = phase since the last crossing / 2*pi"
        ),
        reads=("last 60 tick prices", "detrended crossing structure"),
        steps=(
            "EMA-smooth the 60-tick window (span 5) and OLS-detrend it, so the cycle sits on a flat line and tick noise does not create crossings of its own.",
            "Count hysteresis crossings: the series must travel +/-35 % of its own spread before a crossing is accepted.",
            "Dominant period = 2 x the median gap between crossings, smoothed by an EMA across windows (a cycle's period does not change every minute).",
            "Fit one Hann-windowed DFT bin at that period -> amplitude A and phase psi; read the phase at 'now' as frac = the fraction of a period since the last crossing, with dir = +1 after an up-crossing and -1 after a down-crossing.",
            "cos(2*pi*frac) is +1 at the trough and -1 at the peak. damping = min(1, crossings/3) x crossing regularity x cycle coherence, so a random walk (no dominant frequency) is damped toward zero.",
        ),
        bands=DIRECTIONAL_SYMMETRIC,
        sign="positive = the micro-cycle is in its rising half",
        trace=("ticks for this window", "crossings", "period", "last_crossing_index", "frac", "up_sign"),
        why="Scalping is a cycle game: buying the trough half beats buying the peak half.",
    )
)

register(
    Logic(
        name="MPS",
        expression=(
            "MPS = tanh( 3 * max(0, rho_w) ) * sign(drift) * min(1, 2 * ER),   "
            "rho_w = (3*rho_1 + 2*rho_2 + rho_3)/6,   "
            "ER = |net move| / total travel"
        ),
        reads=("log returns of the window", "lag-1/2/3 autocorrelation"),
        steps=(
            "Compute the lag-1, lag-2 and lag-3 autocorrelation of tick returns.",
            "Weight them 3 : 2 : 1 (nearest lag matters most) -> rho_w.",
            "Clamp at zero: only positive autocorrelation is trend persistence. A mean-reverting tape (rho < 0) has no trend to continue, so its persistence is 0, not negative.",
            "Persistence strength = tanh(3 * max(0, rho_w)).",
            "Sign it with the recent drift and scale by the window's efficiency ratio (2 * min(1, ER)) so chop that happens to drift cannot read as a trend.",
        ),
        bands=DIRECTIONAL_SYMMETRIC,
        sign="positive = a persistent up-move that tends to continue",
        trace=("ticks for this window", "rho1", "rho2", "rho3", "weighted", "persistence", "drift_sign"),
        why="It tells you whether to buy the breakout or fade it.",
    )
)

register(
    Logic(
        name="TWRS",
        expression="TWRS = tanh( m3_w / (m2_w^1.5 + eps) ),  moments weighted by recency",
        reads=("tick log returns", "recency weights"),
        steps=(
            "Weight each return by its recency (exponential decay).",
            "Compute the weighted second and third central moments.",
            "Standardise: skew = m3 / m2^1.5.",
            "Scale by a self-calibrating skew scale so quiet tapes do not read as extreme.",
        ),
        bands=DIRECTIONAL_SYMMETRIC,
        sign="positive = the recent tail is skewed to the upside",
        trace=("ticks for this window", "mean", "m2", "m3", "skew", "scale"),
        why="Tail asymmetry is where stop-hunts and squeezes show up.",
    )
)

# ---------------------------------------------------------------------------
# Category F - Kalman
# ---------------------------------------------------------------------------
register(
    Logic(
        name="DSKD",
        expression=(
            "DSKD = tanh( fresh / scale ),   "
            "fresh = (p_fast - p_slow) - baseline,   "
            "scale = max(sigma_div, 3e-5, 0.75 * realised window move)"
        ),
        reads=("tick prices streamed into two random-walk Kalman filters", "running spread of the two estimates"),
        steps=(
            "Normalise the price to the window's first value, then run a fast Kalman random-walk filter (q = 1.0) and a slow one (q = 0.01) over the same ticks.",
            "divergence = p_fast - p_slow: how far the fast estimate has run ahead of the slow one, in relative price units (1e-5 = 1 bp).",
            "Track the divergence's own slow baseline inside the window and score the fresh part: a steady trend leaves a constant gap (fresh ~ 0), a real level shift moves it.",
            "Divide by the largest of the running spread of the divergence, the 3e-5 (3 bp) floor, and three quarters of the tape's realised move this window, so an ordinary wiggle on a quiet tape cannot read as a shift.",
        ),
        bands=DIRECTIONAL_SYMMETRIC,
        sign="positive = a genuine upward level shift, not measurement noise",
        trace=("ticks for this window", "p_fast", "p_slow", "divergence", "sigma_resid", "scale"),
        why="It is the arbiter between a real move and noise, at 0.01 ms a call.",
    )
)

# ---------------------------------------------------------------------------
# Category G - news
# ---------------------------------------------------------------------------
register(
    Logic(
        name="NIV",
        expression="NIV = sum(s_j * c_j * e^(-dt/300)) / (sum(c_j * e^(-dt/300)) + eps)",
        reads=("the five latest headlines", "sentiment score per headline", "source credibility"),
        steps=(
            "Score each headline's sentiment (lexicon or provider score).",
            "Weight it by source credibility and by e^(-age/300 s).",
            "Take the weighted mean: a fresh, credible headline dominates an old one.",
        ),
        bands=DIRECTIONAL_SYMMETRIC,
        sign="positive = the news flow is bullish right now",
        trace=("headlines considered", "weights", "sentiment", "score"),
        why="News moves the first 30 seconds of a minute; the tape moves the rest.",
    )
)

register(
    Logic(
        name="SMD",
        expression="SMD = tanh( NIV - TAI )",
        reads=("NIV (news sentiment)", "TAI (price acceleration)"),
        steps=(
            "Take the news read (NIV) and the tape read (TAI).",
            "Subtract: the divergence between what is being said and what is being traded.",
            "tanh keeps it in range: positive = news is more bullish than the tape.",
        ),
        bands=DIRECTIONAL_SYMMETRIC,
        sign="positive = the news is ahead of the price (potential continuation)",
        trace=("headlines considered", "niv", "tai", "divergence"),
        why="Divergence between narrative and tape is the classic scalp setup.",
    )
)

# ---------------------------------------------------------------------------
# Category H - brain
# ---------------------------------------------------------------------------
register(
    Logic(
        name="KCAE",
        expression="p_j = |a_j|^2 / sum|a_k|^2;  H = -sum p_j ln p_j;  KCAE = 1 - H/ln(50)",
        reads=("Kenyon Cell activations from the graph convolution (pre-ReLU drive)"),
        steps=(
            "Normalise the 50 KC activations into a probability distribution.",
            "Compute the Shannon entropy of that distribution.",
            "Divide by ln(50), the entropy of a uniform code, and invert.",
            "1.0 = a single KC cluster fires (a very specific pattern), 0 = the code is mush.",
        ),
        bands=_bands(
            (0.80, 1.01, "the brain recognised a very specific pattern"),
            (0.55, 0.80, "clear pattern recognition"),
            (0.35, 0.55, "weak pattern"),
            (0.15, 0.35, "little structure in the code"),
            (-0.01, 0.15, "the KC code is effectively noise"),
        ),
        sign="not directional: it is the brain's own confidence term",
        trace=("ticks for this window", "kc_count", "entropy", "max_entropy", "sparsity"),
        why="Confidence has to come from the circuit itself, not from a hard-coded number.",
    )
)

register(
    Logic(
        name="CCSv2",
        expression=(
            "CCSv2 = tanh( (LH_approach - LH_avoid) - resting_balance ),   "
            "resting_balance = the same read-out for a zero input vector"
        ),
        reads=("all 20 input formulas mapped to the 20 projection neurons", "80-node connectome adjacency"),
        steps=(
            "Write the 20 formula values onto PN 0-19 (each input has its own neuron).",
            "Propagate the zero vector through the same circuit: the Kenyon Cells drive the neutral MBON on the rectified (bullish) side only, so the circuit has a positive resting read-out. Subtracting it makes the score sign-symmetric.",
            "Propagate the real vector through 3 graph-convolution layers: PN -> KC (sparse code) -> MBON, gated by dopamine (DRG) and octopamine (HSI).",
            "Read the lateral horn: (approach - avoid) - resting_balance, squashed by tanh.",
            "The whole brain has exactly one vote, in [-1, 1].",
        ),
        bands=DIRECTIONAL_SYMMETRIC,
        sign="positive = the circuit voted for approach (buy)",
        trace=("ticks for this window", "pn_vector_top", "kc_active", "lh_approach", "lh_avoid", "raw"),
        why="It is the only aggregation that respects the connectome instead of averaging numbers.",
    )
)

register(
    Logic(
        name="DRG",
        expression="DA = delta^0.8 if delta>0 else -|delta|^1.2;  DRG = tanh(0.1 * DA)",
        reads=("outcomes of the last 20 windows: win/loss and pnl in bps"),
        steps=(
            "V = sum gamma^(R-i) * m_i * o_i / sum gamma^(R-i)  with gamma = 0.95 (reward trace).",
            "delta = (m_R * o_R + gamma * V) - V_prev  (temporal-difference error).",
            "Apply loss aversion: losses count 1.2x, wins 0.8x.",
            "tanh(0.1 * DA): positive = the engine is being rewarded, negative = punished.",
        ),
        bands=_bands(
            (0.60, 1.01, "strongly rewarded - recent signals are paying"),
            (0.20, 0.60, "rewarded"),
            (-0.20, 0.20, "neutral reward history"),
            (-0.60, -0.20, "being punished - reduce size"),
            (-1.01, -0.60, "heavy losses recently - the dopamine gate is closing"),
        ),
        sign="not directional: it gates the dopamine nodes and reports recent performance",
        trace=("ticks for this window", "reward_value", "delta", "da", "gamma", "outcomes"),
        why="The brain has to learn from what actually happened, or it is just arithmetic.",
    )
)


def get(name: str) -> Logic | None:
    return LOGIC.get(name)


def readings(values: dict[str, float]) -> dict[str, str]:
    """``{name: one-line reading}`` for every formula that has a value."""
    out: dict[str, str] = {}
    for name, value in values.items():
        logic = LOGIC.get(name)
        if logic is not None:
            out[name] = logic.reading(float(value))
    return out
