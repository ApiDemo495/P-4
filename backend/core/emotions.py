"""The crowd's emotions, measured from microseconds to minutes.

The user's observation is the premise of this module: *"60 second timeframe
market easily get manipulated by retailers emotions."*  A one-minute window is
short enough that the tape is often nothing but a crowd acting on a feeling -
fear, a chase, a surrender - rather than on new information.

So this module reads the same tape the formulas read and asks one question:
**what is the crowd feeling right now, and on which timescale?**  It answers with
eight named emotions, each an intensity in [0, 1], each built from features that
are measured at microsecond resolution but aggregated over five bands:

===========  ==========================================================
``micro``    µs - ms: inter-tick intervals, jitter, quote lifetime, burst
``seconds``  1 - 15 s: velocity, aggression, spread blow-out
``window``   the 60-second window: range position, drawdown, volume climax
``minutes``  1-minute candles: trend, run-up, drawdown from the high
``news``     the headlines: sentiment and how much of it there is
===========  ==========================================================

The eight emotions
------------------
``FEAR``          sellers in control, volatility rising, spread widening.
``PANIC``         the acute form: tick-rate surge + a violent 1-3 s drop.
``CAPITULATION``  surrender: extreme drawdown, climax volume, then calm -
                  the crowd has given up rather than turned bullish.
``DENIAL``        price falling while buys keep absorbing it (buying the dip).
``HOPE``          a constructive drift off the lows, still below the mean.
``EUPHORIA``      "happy": price pinned at the top, buys aggressive, calm.
``FOMO``          chasing: fast up-move, volume surge, the ask side thinning.
``COMPLACENCY``   nothing happening: tight spread, low volatility, mid-range.

Alongside the emotions the module measures the *manipulation* signature the user
described - herding, whipsaw, book imbalance and stop hunts - and returns a
``manipulation`` block with a 0-1 score, the dominant kind and the evidence.
That score is what the fusion uses to dampen confidence: an emotional,
manipulated minute is a minute to trade smaller, never a reason to flip a locked
side.

Everything here is a pure function of a frozen snapshot (or a compacted live
view of the same shape), so the Signal Lock Protocol still holds: the emotions
that shaped a signal cannot change afterwards.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from backend.core import deep_micro
from backend.core.prediction import consensus
from backend.core.timebase import format_us, now_us, us_to_iso

# ---------------------------------------------------------------------------
# The eight emotions
# ---------------------------------------------------------------------------

#: ``name -> (label, tone)``.  Tone is what the UI colours the bar with:
#: red for a fear-family reading, green for a chase, grey for flat calm.
EMOTIONS: dict[str, tuple[str, str]] = {
    "FEAR": ("Fear", "negative"),
    "PANIC": ("Panic", "negative"),
    "CAPITULATION": ("Capitulation", "negative"),
    "DENIAL": ("Denial", "negative"),
    "HOPE": ("Hope", "positive"),
    "EUPHORIA": ("Euphoria", "positive"),
    "FOMO": ("FOMO", "positive"),
    "COMPLACENCY": ("Complacency", "neutral"),
}

#: The five bands every emotion is decomposed into.
TIMESCALES: tuple[str, ...] = ("micro", "seconds", "window", "minutes", "news")

#: The emotional families, for the one-line summary the UI prints.
FAMILIES: dict[str, str] = {
    "FEAR": "fear",
    "PANIC": "fear",
    "CAPITULATION": "withdrawal",
    "DENIAL": "denial",
    "HOPE": "hope",
    "EUPHORIA": "happiness",
    "FOMO": "chase",
    "COMPLACENCY": "calm",
}

#: How the crowd's feeling translates into trade size (never into direction).
CONVICTION_HINT: dict[str, str] = {
    "PANIC": "the crowd is panicking - size down, the tape is emotional",
    "FEAR": "fear is driving the tape - sell into strength, do not chase it",
    "CAPITULATION": "sellers are surrendering - expect reflex bounces",
    "DENIAL": "the tape and the headlines disagree - wait for confirmation",
    "FOMO": "the crowd is chasing - entries are late here",
    "EUPHORIA": "euphoria at the highs - the last buyers are arriving",
    "HOPE": "a tentative turn up - the crowd is not committed yet",
    "COMPLACENCY": "nothing is happening - the range is the story",
}


# ---------------------------------------------------------------------------
# Small numeric helpers - every one of them is a legible transform, because a
# number a reader cannot reproduce is a number a reader cannot trust.
# ---------------------------------------------------------------------------
def _num(value: Any, fallback: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return fallback
    return out if np.isfinite(out) else fallback


def _ramp(value: float, low: float, high: float) -> float:
    """0 at ``low``, 1 at ``high``, linear between - the unit of every score."""
    if high <= low:
        return 0.0
    return float(min(1.0, max(0.0, (value - low) / (high - low))))


def _weighted(parts: list[tuple[float, float]]) -> float:
    """Weighted mean of ``(weight, value)`` pairs, ignoring zero weights."""
    total = sum(w for w, _ in parts)
    if total <= 0:
        return 0.0
    return float(sum(w * v for w, v in parts) / total)


def _terms(parts: list[tuple[float, str, float]]) -> tuple[float, list[dict]]:
    """``_weighted`` that also returns every term, so the panel can print the
    emotion's formula with the live numbers substituted (the user's "show me
    the emotion formulas" ask)."""
    total = sum(w for w, _, _ in parts) or 1.0
    value = float(sum(w * v for w, _, v in parts) / total)
    rows = [
        {
            "weight": w,
            "term": name,
            "value": round(float(v), 4),
            "contribution": round(float(w * v / total), 4),
        }
        for w, name, v in parts
    ]
    return value, rows


#: The symbolic formula of each emotion - the same expression ``score_emotions``
#: evaluates, printed.  ``ramp(x; a, b)`` is the linear ramp clipped to [0, 1]
#: between ``a`` and ``b``; every ``z_*`` is a move in multiples of the tape's
#: own typical move, so the emotions are surprise measures, not volatility.
EMOTION_FORMULAS: dict[str, str] = {
    "FEAR": (
        "FEAR = [0.30·sell + 0.25·vol_rise + 0.20·max(down_5s, down_60s) "
        "+ 0.15·bid_thin + 0.10·blowout] × (1 − 0.50·top)"
    ),
    "PANIC": "PANIC = 0.30·down_1s + 0.25·surge + 0.20·blowout + 0.15·jitter + 0.10·climax",
    "CAPITULATION": (
        "CAPITULATION = fall_gate × [0.35·drawdown_extreme + 0.25·climax "
        "+ 0.25·settling + 0.15·vol_calm],  fall_gate = ramp(z_drawdown; 0.8, 2.0)"
    ),
    "DENIAL": (
        "DENIAL = 0.40·divergence + 0.35·dip_buying + 0.25·bottom,  "
        "divergence = news_pos·down_5s,  dip_buying = min(buy, down_5s)"
    ),
    "HOPE": (
        "HOPE = [0.35·recovery + 0.25·max(up_1s, up_60s) + 0.20·buy "
        "+ 0.20·max(news_pos, vol_calm)] × (1 − 0.60·top)"
    ),
    "EUPHORIA": (
        "EUPHORIA = [0.30·top + 0.25·up_1s + 0.20·buy + 0.15·vol_calm "
        "+ 0.10·news_pos] × (1 − 0.60·bottom)"
    ),
    "FOMO": (
        "FOMO = [0.30·up_5s + 0.25·climax + 0.20·buy + 0.15·ask_thin "
        "+ 0.10·surge] × (1 − 0.30·bottom)"
    ),
    "COMPLACENCY": (
        "COMPLACENCY = [0.35·calm_moves + 0.20·no_aggression + 0.20·middle "
        "+ 0.15·tight + 0.10·max(slow_tape, vol_calm)] "
        "× (1 − 0.70·max(fear, panic, fomo, ½·up_1s, ½·down_1s))"
    ),
}

#: What each term in the formulas above is, in the same notation.
TERM_DEFINITIONS: dict[str, str] = {
    "sell": "ramp(−aggression_5s; 0.15, 0.65) - the share of prints crossing to the bid",
    "buy": "ramp(+aggression_5s; 0.15, 0.65)",
    "down_1s": "ramp(−z_1s; 1.2, 3.5) - the 1 s move in typical 1 s moves",
    "up_1s": "ramp(+z_1s; 1.2, 3.5)",
    "down_5s": "ramp(−z_5s; 1.0, 3.0)",
    "up_5s": "ramp(+z_5s; 1.0, 3.0)",
    "down_60s": "ramp(−z_60s; 0.8, 2.5)",
    "up_60s": "ramp(+z_60s; 0.8, 2.5)",
    "vol_rise": "ramp(σ_now/σ_baseline; 1.15, 1.80)",
    "vol_calm": "1 − ramp(σ_now/σ_baseline; 0.85, 1.30)",
    "climax": "ramp(volume / its own pace; 2.0, 8.0)",
    "surge": "ramp(tick rate / its own pace; 1.5, 5.0)",
    "slow_tape": "1 − ramp(tick rate / its own pace; 0.6, 1.6)",
    "jitter": "ramp(inter-arrival jitter; 5 ms, 200 ms)",
    "blowout": "ramp(spread / its baseline; 1.6, 4.0) × min(1, movement) × (½ if spread < 1 bp)",
    "tight": "max(1 − ramp(spread/baseline; 0.9, 1.8), 1 − ramp(spread_bps; 1, 8))",
    "calm_moves": "1 − ramp(max|z_1s, z_5s, z_60s|; 0.8, 2.5)",
    "no_aggression": "1 − ramp(|aggression_5s|; 0.10, 0.50)",
    "top": "ramp(range_position; 0.70, 0.98)",
    "bottom": "1 − ramp(range_position; 0.05, 0.35)",
    "middle": "max(0, 1 − 4·|range_position − ½|)",
    "drawdown_extreme": "ramp(z_drawdown; 2.0, 5.0)",
    "run_up": "ramp(z_run_up; 1.5, 4.0)",
    "settling": "1 − ramp(|z_1s|; 0.8, 2.5)",
    "recovery": "min(up_5s, 1 − ramp(range_position; 0.45, 0.80))",
    "news_pos": "ramp(news_sentiment; 0.15, 0.70)",
    "news_neg": "ramp(−news_sentiment; 0.15, 0.70)",
    "bid_thin": "ramp(+depth_imbalance; 0.10, 0.55)",
    "ask_thin": "ramp(−depth_imbalance; 0.10, 0.55)",
    "divergence": "news_pos × down_5s",
    "dip_buying": "min(buy, down_5s)",
}


@dataclass
class EmotionScore:
    """One emotion, on one tape, at one instant."""

    name: str = ""
    label: str = ""
    tone: str = "neutral"
    family: str = ""
    intensity: float = 0.0
    by_timescale: dict[str, float] = field(default_factory=dict)
    dominant_timescale: str = ""
    drivers: tuple[str, ...] = ()
    #: the Bayesian filter's belief in this emotion (0-1), from ``deep_micro``
    belief: float = 0.0
    #: the ramp reading before the belief was blended in
    ramp: float = 0.0
    #: the symbolic formula and its live terms (weight, term, value, contribution)
    formula: str = ""
    terms: tuple[dict, ...] = ()
    #: the multiplicative gate / contrast factor applied after the weighted sum
    gate: float = 1.0

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "belief": round(self.belief, 4),
            "ramp": round(self.ramp, 4),
            "formula": self.formula,
            "terms": list(self.terms),
            "gate": round(self.gate, 4),
            "label": self.label,
            "tone": self.tone,
            "family": self.family,
            "intensity": round(self.intensity, 4),
            "percent": round(self.intensity * 100.0, 1),
            "by_timescale": {k: round(v, 4) for k, v in self.by_timescale.items()},
            "dominant_timescale": self.dominant_timescale,
            "drivers": list(self.drivers),
        }


@dataclass
class EmotionTracker:
    """Rolling emotion state: smoothing, history and persistence.

    The engine is asked for a reading once per second, but a crowd does not
    change its mind every second.  The tracker keeps an EMA of every emotion so
    the panel shows a *state* rather than a flicker, records a 300-sample history
    (five minutes at 1 Hz) and answers two questions the UI needs: how long the
    dominant emotion has held, and how fast the emotional temperature is moving.
    """

    asset: str = "BTC"
    #: EMA weight per sample (0.5 s cadence -> a ~2 s time constant): fast
    #: enough to catch a panic spike, slow enough not to flicker on one burst.
    alpha: float = 0.25
    #: Hysteresis for the *dominant* emotion: a challenger must lead the
    #: incumbent by this much, for this many consecutive samples, before the
    #: panel switches.  The bars stay honest (they show the smoothed values);
    #: only the headline waits for the crowd to mean it.
    switch_margin: float = 0.04
    switch_dwell: int = 2
    history_size: int = 300
    emotions: dict[str, float] = field(default_factory=dict)
    #: EMA of the 22-formula consensus the live reading is checked against
    #: (the formulas are refreshed every 2 s; the verdict must not flicker on a
    #: single pass that crosses zero).  ``None`` until the first vote.
    consensus_ema: float | None = None
    samples: int = 0
    history: list[dict] = field(default_factory=list)
    last_dominant: str = ""
    last_switch_us: int = 0
    last_intensity: float = 0.0
    last_change_us: int = 0
    volatility_per_s: float = 0.0
    challenger: str = ""
    challenger_count: int = 0
    #: the Bayesian filter's posterior, carried between samples
    deep_state: deep_micro.DeepState = field(default_factory=deep_micro.DeepState)

    def _elect(self, ranked: list[tuple[str, float]]) -> str:
        """The dominant emotion, with hysteresis (see the class docstring)."""
        if not ranked:
            return ""
        leader, lead_value = ranked[0]
        if not self.last_dominant or self.last_dominant not in self.emotions:
            return leader
        if leader == self.last_dominant:
            self.challenger, self.challenger_count = "", 0
            return leader
        incumbent_value = self.emotions[self.last_dominant]
        if lead_value - incumbent_value < self.switch_margin:
            self.challenger, self.challenger_count = "", 0
            return self.last_dominant
        if self.challenger == leader:
            self.challenger_count += 1
        else:
            self.challenger, self.challenger_count = leader, 1
        if self.challenger_count >= self.switch_dwell:
            self.challenger, self.challenger_count = "", 0
            return leader
        return self.last_dominant

    def observe(self, report: "EmotionReport") -> dict:
        """Fold one raw reading into the state and return the smoothed view."""
        scores = {s.name: s.intensity for s in report.scores}
        if not self.emotions:
            self.emotions = dict(scores)
        else:
            for name, value in scores.items():
                previous = self.emotions.get(name, value)
                # Fast attack, slow release: a reading that jumps a long way
                # from the state (a panic arriving) is adopted at double speed,
                # while ordinary sample-to-sample noise is smoothed normally.
                step = self.alpha if abs(value - previous) < 0.2 else min(1.0, self.alpha * 2.0)
                self.emotions[name] = (1 - step) * previous + step * value
        self.samples += 1

        ranked = sorted(self.emotions.items(), key=lambda kv: kv[1], reverse=True)
        dominant = self._elect(ranked)
        intensity = self.emotions.get(dominant, 0.0)
        now = report.at_us or now_us()
        if dominant != self.last_dominant:
            self.last_dominant = dominant
            self.last_switch_us = now
        if self.last_intensity:
            # per-second rate of change of the dominant intensity
            elapsed = max(1e-6, (now - self.last_change_us) / 1e6)
            self.volatility_per_s = abs(intensity - self.last_intensity) / elapsed
        else:
            self.volatility_per_s = 0.0
        self.last_intensity = intensity
        self.last_change_us = now

        sample = {
            "at_us": now,
            "dominant": dominant,
            "intensity": round(intensity, 4),
            "tone_bias": round(_tone_bias(self.emotions), 4),
            "manipulation": round(report.manipulation_score, 4),
        }
        self.history.append(sample)
        if len(self.history) > self.history_size:
            del self.history[: -self.history_size]

        window = [s for s in self.history if now - s["at_us"] <= 60_000_000]
        switches = sum(
            1 for a, b in zip(window, window[1:]) if a["dominant"] != b["dominant"]
        )
        return {
            "dominant": dominant,
            "intensity": intensity,
            "smoothed": dict(self.emotions),
            "held_us": max(0, now - self.last_switch_us),
            "held_seconds": round(max(0.0, (now - self.last_switch_us) / 1e6), 3),
            "emotional_volatility_per_s": round(self.volatility_per_s, 5),
            "churn_per_minute": switches,
            "samples": self.samples,
            "history": list(self.history[-60:]),
        }


def _tone_bias(emotions: dict[str, float]) -> float:
    """Signed temperature of the crowd: -1 terrified, +1 euphoric, 0 asleep.

    The denominator includes complacency (and a small floor), so a tape where
    nothing is happening reads 0 rather than "maximally sad because the only
    emotion with any weight happens to be fear at 13%".
    """
    negative = sum(
        emotions.get(name, 0.0) for name in ("FEAR", "PANIC", "CAPITULATION", "DENIAL")
    )
    positive = sum(emotions.get(name, 0.0) for name in ("HOPE", "EUPHORIA", "FOMO"))
    calm = emotions.get("COMPLACENCY", 0.0)
    total = negative + positive + calm + 0.15
    if total <= 1e-9:
        return 0.0
    return float((positive - negative) / total)


@dataclass
class EmotionReport:
    """The crowd's state on one tape: eight emotions plus the manipulation read."""

    asset: str = ""
    available: bool = False
    reason: str = ""
    at_us: int = 0
    ticks: int = 0
    span_us: int = 0
    resolution_us: float = 0.0
    measured_over_us: int = 0
    scores: list[EmotionScore] = field(default_factory=list)
    features: dict[str, float] = field(default_factory=dict)
    manipulation: dict = field(default_factory=dict)
    tracker: dict = field(default_factory=dict)
    deep: dict = field(default_factory=dict)
    #: how the crowd reading sits against the 22 formulas' vote (Round L)
    formula_agreement: dict = field(default_factory=dict)

    # -- convenience ----------------------------------------------------
    @property
    def intensity(self) -> dict[str, float]:
        return {s.name: s.intensity for s in self.scores}

    @property
    def manipulation_score(self) -> float:
        return _num(self.manipulation.get("score"))

    def ranked(self) -> list[EmotionScore]:
        """Strongest first - with the tracker's (hysteresis) choice in front."""
        ordered = sorted(self.scores, key=lambda s: s.intensity, reverse=True)
        elected = (self.tracker or {}).get("dominant")
        if elected:
            for index, score in enumerate(ordered):
                if score.name == elected and index:
                    ordered.insert(0, ordered.pop(index))
                    break
        return ordered

    def dominant(self) -> EmotionScore | None:
        if not self.scores:
            return None
        top = self.ranked()[0]
        return top if top.intensity > 0 else None

    def read(self) -> str:
        """One sentence a person can act on - the live "what is it feeling"."""
        top = self.dominant()
        if top is None:
            return "no emotion reading yet - the tape has not moved"
        second = self.ranked()[:2]
        text = (
            f"{top.label} is dominant ({top.intensity * 100:.0f}% on the "
            f"{top.dominant_timescale} timescale)"
        )
        if len(second) > 1 and second[1].intensity > 0.25:
            text += f", with {second[1].label.lower()} behind it ({second[1].intensity * 100:.0f}%)"
        hint = CONVICTION_HINT.get(top.name)
        if hint:
            text += f" - {hint}"
        post = (self.deep or {}).get("posterior") or {}
        if post.get("argmax"):
            belief = post.get("argmax_probability", 0.0) * 100.0
            if post["argmax"] == top.name:
                text += f"; the Bayesian filter agrees ({belief:.0f}% belief)"
            else:
                text += (
                    f"; the Bayesian filter leans {post['argmax'].capitalize()} "
                    f"({belief:.0f}% belief)"
                )
        agree = self.formula_agreement or {}
        if agree.get("note"):
            text += f"; {agree['note']}"
        return text + "."

    def to_dict(self) -> dict:
        ranked = self.ranked()
        top = ranked[0] if ranked else None
        second = ranked[1] if len(ranked) > 1 else None
        tracker = dict(self.tracker or {})
        payload = {
            "asset": self.asset,
            "available": self.available,
            "reason": self.reason,
            "at_us": self.at_us,
            "at": us_to_iso(self.at_us) if self.at_us else "",
            "ticks": self.ticks,
            "span_us": self.span_us,
            "span_label": format_us(self.span_us) if self.span_us else "—",
            "resolution_us": round(float(self.resolution_us), 3),
            "resolution_label": (
                format_us(self.resolution_us) if self.resolution_us else "—"
            ),
            "measured_over_us": self.measured_over_us,
            "measured_over_label": (
                format_us(self.measured_over_us) if self.measured_over_us else "—"
            ),
            "emotions": [s.to_dict() for s in ranked],
            "dominant": (
                {
                    **{
                        k: v
                        for k, v in top.to_dict().items()
                        if k in ("name", "label", "tone", "family", "intensity", "percent")
                    },
                    "dominant_timescale": top.dominant_timescale,
                    "held_seconds": tracker.get("held_seconds", 0.0),
                    "held_us": tracker.get("held_us", 0),
                    **({"drivers": list(top.drivers)} if top.drivers else {}),
                }
                if top is not None
                else None
            ),
            "runner_up": (
                {
                    "name": second.name,
                    "label": second.label,
                    "intensity": round(second.intensity, 4),
                    "percent": round(second.intensity * 100.0, 1),
                }
                if second is not None
                else None
            ),
            "tone_bias": round(_tone_bias(self.intensity), 4),
            "emotional_volatility_per_s": tracker.get("emotional_volatility_per_s", 0.0),
            "churn_per_minute": tracker.get("churn_per_minute", 0),
            "samples": tracker.get("samples", 0),
            "held_seconds": tracker.get("held_seconds", 0.0),
            "features": {k: round(v, 6) for k, v in self.features.items()},
            "manipulation": dict(self.manipulation),
            "deep": dict(self.deep),
            "formula_agreement": dict(self.formula_agreement),
            "formula_glossary": dict(TERM_DEFINITIONS),
            "history": tracker.get("history", []),
        }
        if top is not None:
            payload["read"] = self.read()
            payload["hint"] = CONVICTION_HINT.get(top.name, "")
        return payload


def compact(payload: dict) -> dict:
    """The streamed form of a reading: everything the panel draws, nothing else.

    The full payload carries ~60 raw features and a five-minute history, which
    is right for REST and for the per-mark PULSE but too heavy to push twice a
    second.  The stream keeps the eight bars, the dominant emotion with its
    drivers and hold time, the runner-up, the tone, the manipulation verdict
    and the one-line read.
    """
    keep = (
        "asset", "available", "reason", "at_us", "at", "ticks", "resolution_us",
        "resolution_label", "measured_over_label", "emotions", "dominant", "runner_up",
        "tone_bias", "emotional_volatility_per_s", "churn_per_minute", "samples",
        "held_seconds", "read", "hint", "interval_seconds", "measured_at_us",
        "formula_agreement",
    )
    out = {key: payload[key] for key in keep if key in payload}
    out["deep"] = deep_micro.compact(payload.get("deep") or {})
    manipulation = payload.get("manipulation") or {}
    out["manipulation"] = {
        key: manipulation[key]
        for key in ("score", "percent", "kind", "note", "components")
        if key in manipulation
    }
    return out


# ---------------------------------------------------------------------------
# Live view: the same accessors the frozen snapshot has, over the live buffers
# ---------------------------------------------------------------------------


class LiveTapeView:
    """The market hub's live buffers, wearing the frozen snapshot's accessors.

    ``analyze()`` only ever calls ``ticks``/``book``/``candles``/
    ``spread_history``/``news_items``, so the exact same code that reads the
    immutable per-window snapshot can read the moving tape between windows.
    That is what makes the live panel and the locked reading comparable: they
    are the same measurement, one second apart.
    """

    def __init__(self, hub: Any, news_items: tuple = (), timestamp: float | None = None) -> None:
        self.hub = hub
        self.news_items = tuple(news_items or ())
        self.timestamp = float(timestamp if timestamp is not None else time.time())

    def _buffers(self, asset: str):
        return self.hub.buffers[str(asset).upper()]

    def ticks(self, asset: str):
        return self._buffers(asset).ticks.view()

    def book(self, asset: str):
        return self._buffers(asset).book.current

    def candles(self, asset: str):
        return self._buffers(asset).candles.closes()

    def spread_history(self, asset: str):
        return self._buffers(asset).book.spreads()

    def last_price(self, asset: str) -> float:
        return float(self.hub.last_price(str(asset).upper()))


@dataclass
class EmotionMonitor:
    """Samples the live tape on a timer and remembers what it found.

    The tracker smooths the raw reading (a crowd does not change its mind every
    second), keeps a five-minute history and answers the two questions the panel
    needs - *which emotion is dominant right now* and *how long has it held*.
    """

    hub: Any
    asset: str = "BTC"
    news_provider: Any = None  # callable -> tuple[NewsItem, ...]
    #: callable -> (formula values, directional map): the live 22-formula pass
    #: the crowd reading is cross-checked against
    formula_provider: Any = None
    interval_seconds: float = 0.5
    history_size: int = 720
    tracker: EmotionTracker = field(default_factory=EmotionTracker)
    last: dict = field(default_factory=dict)
    last_report: "EmotionReport | None" = None
    samples: int = 0
    errors: int = 0

    def __post_init__(self) -> None:
        self.asset = str(self.asset).upper()
        self.tracker.asset = self.asset
        self.tracker.history_size = max(60, int(self.history_size))

    # -- the loop ------------------------------------------------------
    def sample(self, at_us_value: int | None = None) -> dict:
        """Take one reading of the live tape and fold it into the tracker."""
        news = ()
        if callable(self.news_provider):
            try:
                news = tuple(self.news_provider() or ())
            except Exception:  # noqa: BLE001 - news must never break the panel
                news = ()
        formulas: dict = {}
        directional: dict = {}
        if callable(self.formula_provider):
            try:
                formulas, directional = self.formula_provider()
            except Exception:  # noqa: BLE001 - the formulas must never break the panel
                formulas, directional = {}, {}
        tape = LiveTapeView(self.hub, news_items=news)
        report = analyze(
            tape, self.asset, tracker=self.tracker, at_us=at_us_value,
            formulas=formulas, directional=directional,
        )
        self.last_report = report
        payload = report.to_dict()
        # The panel needs to know how fresh the reading is and how often it is
        # taken; both are part of the message so the UI never guesses.
        payload["interval_seconds"] = self.interval_seconds
        payload["measured_at_us"] = payload.get("at_us", 0)
        self.last = payload
        self.samples += 1
        return payload

    def payload(self) -> dict:
        if not self.last:
            return {
                "asset": self.asset,
                "available": False,
                "reason": "the emotion engine has not sampled the tape yet",
                "emotions": [],
                "interval_seconds": self.interval_seconds,
            }
        return self.last


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------
def _price_at(times_ms: np.ndarray, prices: np.ndarray, seconds_ago: float, now_ms: float) -> float:
    """The last price at or before ``now_ms - seconds_ago`` (0 when unknown)."""
    if prices.size == 0:
        return 0.0
    cutoff = now_ms - seconds_ago * 1000.0
    index = int(np.searchsorted(times_ms, cutoff, side="right")) - 1
    if index < 0:
        return float(prices[0])
    return float(prices[min(index, prices.size - 1)])


def _side_skew(ticks: np.ndarray) -> float:
    """Signed share of volume that crossed on the buy side (-1 .. +1)."""
    if ticks.size == 0:
        return 0.0
    qty = np.abs(ticks[:, 2])
    if qty.sum() <= 0:
        return 0.0
    return float(np.sum(qty * np.sign(ticks[:, 3])) / np.sum(qty))


def _herding(ticks: np.ndarray) -> float:
    """How one-sided the crowd is: side skew and how long its runs are.

    Retail emotional moves are *runs* of same-side prints: a skew of +0.4 with
    runs of 9 prints in a row is a crowd, not a market.
    """
    if ticks.shape[0] < 4:
        return 0.0
    sides = np.sign(ticks[:, 3])
    runs: list[int] = []
    current = 1
    for a, b in zip(sides, sides[1:]):
        if b == a and b != 0:
            current += 1
        else:
            runs.append(current)
            current = 1
    runs.append(current)
    longest = max(runs) if runs else 1
    run_score = _ramp(longest, 4, 20)
    skew_score = _ramp(abs(_side_skew(ticks)), 0.15, 0.75)
    return _weighted([(0.5, skew_score), (0.5, run_score)])


def _reversals(prices: np.ndarray, bucket: int = 5) -> int:
    """Direction flips of a bucketed price path - the whipsaw count."""
    if prices.size < bucket * 3:
        return 0
    buckets = prices[-(prices.size // bucket) * bucket :].reshape(-1, bucket).mean(axis=1)
    deltas = np.diff(buckets)
    signs = np.sign(deltas)
    signs = signs[signs != 0]
    return int(np.sum(np.diff(signs) != 0)) if signs.size > 1 else 0


def _candle_trend(candles: np.ndarray) -> tuple[float, float, float]:
    """(trend in bps over the last 10 candles, run-up, drawdown) from closes."""
    closes = np.asarray(candles, dtype=np.float64).reshape(-1)
    if closes.size < 3:
        return 0.0, 0.0, 0.0
    last = float(closes[-1])
    span = closes[-min(closes.size, 10) :]
    base = float(span[0]) or last
    trend_bps = (last - base) / base * 10_000.0 if base else 0.0
    high = float(np.max(closes)); low = float(np.min(closes))
    run_up = (last - low) / low * 10_000.0 if low else 0.0
    drawdown = (high - last) / high * 10_000.0 if high else 0.0
    return trend_bps, run_up, drawdown


def _depth_imbalance(book: Any, levels: int = 20) -> float:
    """Signed depth imbalance of the top ``levels`` (+1 all bids, -1 all asks)."""
    try:
        bids = np.asarray(book[0][:levels, 1], dtype=np.float64)
        asks = np.asarray(book[1][:levels, 1], dtype=np.float64)
    except (IndexError, TypeError, ValueError):
        return 0.0
    total = float(bids.sum() + asks.sum())
    if total <= 0:
        return 0.0
    return float((bids.sum() - asks.sum()) / total)


def _typical_move_bps(times_ms: np.ndarray, prices: np.ndarray, seconds: float) -> float:
    """The tape's *own* typical move over ``seconds``, in bps.

    Every ramp in this module is expressed in units of this number rather than
    in absolute bps, because "40 bps in a second" is a panic on a dead tape and
    a rounding error on a violent one.  Measuring a crowd with an absolute scale
    would make the reading a function of volatility instead of a function of
    *surprise*.

    The moves are measured over *time* (each tick against the last price at or
    before ``seconds`` earlier), exactly as the live return is, so an irregular
    tape - bursts of prints, then a gap - cannot make the scale and the return
    disagree about what "one second" means.
    """
    if prices.size < 4:
        return 0.0
    horizon_ms = seconds * 1000.0
    earlier = np.searchsorted(times_ms, times_ms - horizon_ms, side="right") - 1
    valid = earlier >= 0
    if not np.any(valid):
        return 0.0
    base = prices[earlier[valid]]
    moves = np.abs(prices[valid] / base - 1.0) * 10_000.0
    moves = moves[np.isfinite(moves)]
    # A horizon the tape barely covers (a 60 s lookback on a 65 s tape) has
    # too few samples for a median to mean anything; report "no scale" and
    # let the caller derive it from the next-shortest horizon.
    if moves.size < 30:
        return 0.0
    return float(np.median(moves))


def features(
    tape: Any,
    asset: str,
    *,
    now_us_value: int | None = None,
    recent: int = 600,
) -> dict[str, float]:
    """Everything the eight emotions are built from, in one flat dictionary.

    The keys are the vocabulary of the panel's tooltips: velocity per horizon
    (in units of the tape's own typical move), aggression, spread blow-out,
    range position, volume climax, herding, whipsaw, depth imbalance and news.
    """
    at = now_us_value if now_us_value is not None else now_us()
    out: dict[str, float] = {"at_us": float(at)}
    ticks = np.asarray(tape.ticks(asset), dtype=np.float64)
    if ticks.ndim != 2 or ticks.shape[0] == 0:
        return out
    if ticks.shape[0] > recent:
        ticks = ticks[-recent:]
    times_ms = ticks[:, 0]
    prices = np.asarray(ticks[:, 1], dtype=np.float64)
    now_ms = float(times_ms[-1])
    out["ticks"] = float(ticks.shape[0])
    out["span_us"] = float(max(0.0, (times_ms[-1] - times_ms[0]) * 1000.0))
    gaps_us = np.diff(times_ms) * 1000.0 if ticks.shape[0] > 1 else np.zeros(0)
    if gaps_us.size:
        out["resolution_us"] = float(np.median(gaps_us))
        out["mean_interval_us"] = float(np.mean(gaps_us))
        out["jitter_us"] = float(1.4826 * np.median(np.abs(gaps_us - np.median(gaps_us))))
        out["tick_rate_hz"] = float(1e6 / max(1.0, float(np.mean(gaps_us))))
    else:
        out["resolution_us"] = out["mean_interval_us"] = out["jitter_us"] = 0.0
        out["tick_rate_hz"] = 0.0
    last = float(prices[-1])
    out["last"] = last

    # --- velocity on five horizons, each in its own surprise units --------
    scale = {
        seconds: _typical_move_bps(times_ms, prices, seconds)
        for seconds in (0.25, 1.0, 5.0, 15.0, 60.0)
    }
    for seconds, typical in scale.items():
        base = _price_at(times_ms, prices, seconds, now_ms)
        bps = (last - base) / base * 10_000.0 if base else 0.0
        out[f"return_{seconds:g}s_bps"] = bps
        out[f"scale_{seconds:g}s_bps"] = typical
        # A move counts from ~1x the tape's own typical move for that horizon,
        # and is extreme at ~3x.
        out[f"z_{seconds:g}s"] = bps / typical if typical > 1e-6 else 0.0
    # A tape shorter than a horizon has no scale of its own; estimate it from the
    # next-shortest one by random-walk scaling (sqrt of the horizon ratio).
    if scale.get(1.0, 0.0) <= 1e-6 and scale.get(0.25, 0.0) > 1e-6:
        scale[1.0] = scale[0.25] * 2.0
    if scale.get(5.0, 0.0) <= 1e-6 and scale.get(1.0, 0.0) > 1e-6:
        scale[5.0] = scale[1.0] * 2.236
    if scale.get(15.0, 0.0) <= 1e-6 and scale.get(5.0, 0.0) > 1e-6:
        scale[15.0] = scale[5.0] * 1.732
    if scale.get(60.0, 0.0) <= 1e-6 and scale.get(15.0, 0.0) > 1e-6:
        scale[60.0] = scale[15.0] * 2.0
    for seconds, typical in scale.items():
        out[f"scale_{seconds:g}s_bps"] = typical
        base = _price_at(times_ms, prices, seconds, now_ms)
        bps = (last - base) / base * 10_000.0 if base else 0.0
        out[f"z_{seconds:g}s"] = bps / typical if typical > 1e-6 else 0.0

    out["velocity_bps_per_s"] = out["return_1s_bps"]
    out["acceleration_bps"] = out["return_5s_bps"] - out["return_1s_bps"]

    # --- aggression and volume -------------------------------------------
    one_second = int(np.searchsorted(times_ms, now_ms - 1000.0, side="left"))
    five_seconds = int(np.searchsorted(times_ms, now_ms - 5000.0, side="left"))
    out["aggression_micro"] = _side_skew(ticks[-min(30, ticks.shape[0]) :])
    out["aggression_1s"] = _side_skew(ticks[one_second:]) if one_second < ticks.shape[0] else 0.0
    out["aggression_5s"] = _side_skew(ticks[five_seconds:]) if five_seconds < ticks.shape[0] else 0.0
    out["aggression_window"] = _side_skew(ticks)
    span_seconds = max(1e-3, out["span_us"] / 1e6)
    qty_all = float(np.abs(ticks[:, 2]).sum())
    per_second = qty_all / span_seconds
    out["volume_per_second"] = per_second
    qty_recent = float(np.abs(ticks[one_second:, 2]).sum()) if one_second < ticks.shape[0] else 0.0
    # Climax and surge are both relative to the tape's own pace: a panic is
    # "five times the usual number of prints", not "40 prints".
    out["volume_climax"] = (qty_recent / per_second) if per_second > 0 else 0.0
    ticks_recent = float(max(0, ticks.shape[0] - one_second))
    ticks_per_second = ticks.shape[0] / span_seconds
    out["tick_surge"] = (ticks_recent / ticks_per_second) if ticks_per_second > 0 else 0.0

    # --- volatility and range --------------------------------------------
    window = prices[-min(prices.size, 120) :]
    out["vol_bps"] = (
        float(np.std(np.diff(window) / window[:-1]) * 10_000.0 * np.sqrt(window.size))
        if window.size > 2
        else 0.0
    )
    short = prices[-min(prices.size, 30) :]
    long_ = prices[-min(prices.size, 240) :]
    if short.size > 2 and long_.size > 4:
        vol_short = float(np.std(np.diff(short) / short[:-1]))
        vol_long = float(np.std(np.diff(long_) / long_[:-1]))
        out["vol_ratio"] = vol_short / vol_long if vol_long > 1e-12 else 1.0
    else:
        out["vol_ratio"] = 1.0
    recent_slice = prices[-min(prices.size, 240) :]
    low = float(np.min(recent_slice)); high = float(np.max(recent_slice))
    out["range_position"] = (last - low) / (high - low) if high > low else 0.5
    out["drawdown_bps"] = (high - last) / high * 10_000.0 if high else 0.0
    out["run_up_bps"] = (last - low) / low * 10_000.0 if low else 0.0
    scale_60 = scale.get(60.0, 0.0)
    out["z_drawdown"] = out["drawdown_bps"] / scale_60 if scale_60 > 1e-6 else 0.0
    out["z_run_up"] = out["run_up_bps"] / scale_60 if scale_60 > 1e-6 else 0.0
    out["reversals"] = float(_reversals(recent_slice[-min(recent_slice.size, 120) :]))

    # --- the order book ---------------------------------------------------
    book = tape.book(asset)
    out["depth_imbalance"] = _depth_imbalance(book)
    try:
        best_bid = float(book[0, 0, 0]); best_ask = float(book[1, 0, 0])
    except (IndexError, TypeError, ValueError):
        best_bid = best_ask = 0.0
    spread = best_ask - best_bid if best_ask > best_bid > 0 else 0.0
    mid = (best_ask + best_bid) / 2.0 if best_bid and best_ask else last
    out["spread_bps"] = spread / mid * 10_000.0 if mid else 0.0
    spreads = np.asarray(tape.spread_history(asset), dtype=np.float64)
    # The blow-out ratio needs a real series; on a single-snapshot tape (the
    # synthetic ones, and the first second of a live one) it stays neutral.
    if spreads.ndim == 2 and spreads.shape[0] >= 8:
        prior_bps = spreads[:-1, 1] / mid * 10_000.0 if mid else spreads[:-1, 1]
        baseline = float(np.median(prior_bps)) if prior_bps.size else 0.0
        out["spread_baseline_bps"] = baseline
        out["spread_blowout"] = out["spread_bps"] / baseline if baseline > 1e-9 else 1.0
    else:
        out["spread_baseline_bps"] = 0.0
        out["spread_blowout"] = 1.0
    out["quote_lifetime_us"] = float(out.get("mean_interval_us", 0.0))

    # --- the slow bands ---------------------------------------------------
    trend, run_up_c, drawdown_c = _candle_trend(tape.candles(asset))
    out["candle_trend_bps"] = trend
    out["candle_run_up_bps"] = run_up_c
    out["candle_drawdown_bps"] = drawdown_c

    # --- news -------------------------------------------------------------
    items = tuple(getattr(tape, "news_items", ()) or ())
    if items:
        sentiments = [_num(getattr(item, "sentiment", 0.0)) for item in items[-10:]]
        out["news_sentiment"] = float(np.mean(sentiments)) if sentiments else 0.0
        out["news_items"] = float(len(items))
        ages = [
            max(0.0, now_ms / 1000.0 - _num(getattr(item, "published_at", 0.0)))
            for item in items[-10:]
        ]
        out["news_freshest_seconds"] = float(min(ages)) if ages else 0.0
        out["news_volume"] = _ramp(len(items), 6.0, 20.0)
    else:
        out["news_sentiment"] = 0.0
        out["news_items"] = 0.0
        out["news_freshest_seconds"] = 0.0
        out["news_volume"] = 0.0

    # --- crowding / manipulation signature -------------------------------
    out["herding"] = _herding(ticks[-min(ticks.shape[0], 300) :])
    out["stop_hunt"] = _stop_hunt(recent_slice)
    return out


def _stop_hunt(prices: np.ndarray, lookback: int = 90) -> float:
    """A wick beyond the recent extreme that price immediately recovers from.

    The classic one-minute manipulation: run the stops above the high (or below
    the low), then snap back inside.  Scored 0-1 by how far price went beyond the
    extreme and how fully it came back.
    """
    if prices.size < 20:
        return 0.0
    path = prices[-min(prices.size, lookback) :]
    body = path[:-5]
    tail = path[-5:]
    if body.size < 5:
        return 0.0
    high, low = float(np.max(body)), float(np.min(body))
    overshoot = max(float(np.max(tail)) - high, low - float(np.min(tail)))
    if overshoot <= 0:
        return 0.0
    span = (high - low) or (abs(high) * 1e-4) or 1e-9
    return _ramp(overshoot / span, 0.25, 1.5)


# ---------------------------------------------------------------------------
# The eight emotions, as legible formulas over those features
# ---------------------------------------------------------------------------
def score_emotions(f: dict[str, float]) -> list[EmotionScore]:
    """Turn the feature dictionary into the eight intensities.

    Each emotion is a weighted mean of ramps over *surprise units* - multiples of
    the tape's own typical move, of its own tick pace, of its own spread - so the
    reading says how unusual the crowd's behaviour is rather than how volatile
    the asset happens to be.  The drivers are the same terms, printed.
    """
    # --- signed surprise on each horizon ---------------------------------
    down_1s = _ramp(-f.get("z_1s", 0.0), 1.2, 3.5)
    up_1s = _ramp(f.get("z_1s", 0.0), 1.2, 3.5)
    down_5s = _ramp(-f.get("z_5s", 0.0), 1.0, 3.0)
    up_5s = _ramp(f.get("z_5s", 0.0), 1.0, 3.0)
    down_60s = _ramp(-f.get("z_60s", 0.0), 0.8, 2.5)
    up_60s = _ramp(f.get("z_60s", 0.0), 0.8, 2.5)

    # --- who is crossing the spread --------------------------------------
    sell = _ramp(-f.get("aggression_5s", 0.0), 0.15, 0.65)
    buy = _ramp(f.get("aggression_5s", 0.0), 0.15, 0.65)

    # --- pace, volatility and spread, all relative ------------------------
    vol_rise = _ramp(f.get("vol_ratio", 1.0), 1.15, 1.80)
    vol_calm = 1.0 - _ramp(f.get("vol_ratio", 1.0), 0.85, 1.30)
    # A wider spread on its own is not fear - it is often just a quiet book
    # being re-quoted.  It only counts as stress next to a move that is already
    # happening, so the term is multiplied by the strongest moving evidence.
    climax = _ramp(f.get("volume_climax", 0.0), 2.0, 8.0)
    surge = _ramp(f.get("tick_surge", 0.0), 1.5, 5.0)
    slow_tape = 1.0 - _ramp(f.get("tick_surge", 0.0), 0.6, 1.6)
    jitter = _ramp(f.get("jitter_us", 0.0), 5_000.0, 200_000.0)
    # A wider spread on its own is not fear - it is often just a quiet book being
    # re-quoted.  It only counts as stress next to movement that is already
    # happening, so the term is multiplied by the strongest moving evidence.
    blowout_raw = _ramp(f.get("spread_blowout", 1.0), 1.6, 4.0)
    if f.get("spread_bps", 0.0) < 1.0:
        blowout_raw *= 0.5  # a sub-bps spread cannot be a fear signal
    movement = max(
        sell, down_1s, down_5s,
        0.5 * surge, 0.5 * vol_rise,
        _ramp(f.get("z_drawdown", 0.0), 0.5, 2.0),
    )
    blowout = blowout_raw * min(1.0, movement)
    # "Tight" is both relative (this book against its own history) and absolute
    # (a sub-basis-point spread is calm on any market), so a degenerate spread
    # series can never fake a quiet tape on its own.
    tight = max(
        1.0 - _ramp(f.get("spread_blowout", 1.0), 0.9, 1.8),
        1.0 - _ramp(f.get("spread_bps", 0.0), 1.0, 8.0),
    )
    # The clearest evidence of "nothing is happening" is that nothing surprised
    # the tape: every horizon is inside its own normal range, and neither side is
    # crossing the spread with conviction.
    surprise = max(
        abs(f.get("z_1s", 0.0)), abs(f.get("z_5s", 0.0)), abs(f.get("z_60s", 0.0))
    )
    calm_moves = 1.0 - _ramp(surprise, 0.8, 2.5)
    no_aggression = 1.0 - _ramp(abs(f.get("aggression_5s", 0.0)), 0.10, 0.50)

    # --- where price sits, and how far it has come ------------------------
    pos = f.get("range_position", 0.5)
    top = _ramp(pos, 0.70, 0.98)
    bottom = 1.0 - _ramp(pos, 0.05, 0.35)
    middle = max(0.0, 1.0 - abs(pos - 0.5) * 4.0)
    drawdown_extreme = _ramp(f.get("z_drawdown", 0.0), 2.0, 5.0)
    run_up = _ramp(f.get("z_run_up", 0.0), 1.5, 4.0)

    # --- the last second going quiet (what a surrender looks like) --------
    settling = 1.0 - _ramp(abs(f.get("z_1s", 0.0)), 0.8, 2.5)

    # --- news -------------------------------------------------------------
    news = f.get("news_sentiment", 0.0)
    news_pos = _ramp(news, 0.15, 0.70)
    news_neg = _ramp(-news, 0.15, 0.70)
    news_flow = f.get("news_volume", 0.0)

    # --- the book ---------------------------------------------------------
    depth = f.get("depth_imbalance", 0.0)
    ask_thin = _ramp(-depth, 0.10, 0.55)
    bid_thin = _ramp(depth, 0.10, 0.55)

    candle_up = _ramp(f.get("candle_trend_bps", 0.0), 20.0, 250.0)
    candle_down = _ramp(-f.get("candle_trend_bps", 0.0), 20.0, 250.0)

    # --- FEAR: sellers in control, volatility rising, the bid thinning ----
    fear, fear_terms = _terms([
        (0.30, "sell", sell), (0.25, "vol_rise", vol_rise),
        (0.20, "max(down_5s, down_60s)", max(down_5s, down_60s)),
        (0.15, "bid_thin", bid_thin), (0.10, "blowout", blowout),
    ])
    # --- PANIC: the acute form, on the fast bands -------------------------
    panic, panic_terms = _terms([
        (0.30, "down_1s", down_1s), (0.25, "surge", surge), (0.20, "blowout", blowout),
        (0.15, "jitter", jitter), (0.10, "climax", climax),
    ])
    # --- CAPITULATION: the fall is over - surrender, then quiet -----------
    # Gated on the fall actually having happened: a quiet second on a flat tape
    # is not surrender, it is a quiet second.
    fall_gate = _ramp(f.get("z_drawdown", 0.0), 0.8, 2.0)
    capitulation, capitulation_terms = _terms([
        (0.35, "drawdown_extreme", drawdown_extreme), (0.25, "climax", climax),
        (0.25, "settling", settling), (0.15, "vol_calm", vol_calm),
    ])
    capitulation *= fall_gate
    # --- DENIAL: buys absorbing a falling tape, or good news vs bad price --
    dip_buying = min(buy, down_5s)
    divergence = news_pos * down_5s
    denial, denial_terms = _terms([
        (0.40, "divergence", divergence), (0.35, "dip_buying", dip_buying), (0.25, "bottom", bottom),
    ])
    # --- HOPE: a constructive drift, still under the mean -----------------
    recovery = min(up_5s, 1.0 - _ramp(pos, 0.45, 0.80))
    hope, hope_terms = _terms([
        (0.35, "recovery", recovery), (0.25, "max(up_1s, up_60s)", max(up_1s, up_60s)),
        (0.20, "buy", buy), (0.20, "max(news_pos, vol_calm)", max(news_pos, vol_calm)),
    ])
    # --- EUPHORIA: "happy" - pinned at the highs, calm, still buying ------
    euphoria, euphoria_terms = _terms([
        (0.30, "top", top), (0.25, "up_1s", up_1s), (0.20, "buy", buy),
        (0.15, "vol_calm", vol_calm), (0.10, "news_pos", news_pos),
    ])
    # --- FOMO: chasing - fast up-move, volume surge, thin asks ------------
    fomo, fomo_terms = _terms([
        (0.30, "up_5s", up_5s), (0.25, "climax", climax), (0.20, "buy", buy),
        (0.15, "ask_thin", ask_thin), (0.10, "surge", surge),
    ])
    # --- COMPLACENCY: nothing is happening --------------------------------
    complacency, complacency_terms = _terms([
        (0.35, "calm_moves", calm_moves), (0.20, "no_aggression", no_aggression),
        (0.20, "middle", middle), (0.15, "tight", tight),
        (0.10, "max(slow_tape, vol_calm)", max(slow_tape, vol_calm)),
    ])

    # Contrast: a crowd cannot be terrified and euphoric at once, and a tape
    # that is genuinely dead is not "afraid" - it is asleep.  These are
    # continuous transforms, not thresholds.
    gates = {
        "FEAR": 1.0 - 0.50 * top,
        "PANIC": 1.0,
        "CAPITULATION": fall_gate,
        "DENIAL": 1.0,
        "HOPE": 1.0 - 0.60 * top,
        "EUPHORIA": 1.0 - 0.60 * bottom,
        "FOMO": 1.0 - 0.30 * bottom,
    }
    fear *= gates["FEAR"]
    euphoria *= gates["EUPHORIA"]
    hope *= gates["HOPE"]
    fomo *= gates["FOMO"]
    gates["COMPLACENCY"] = 1.0 - 0.70 * max(fear, panic, fomo, up_1s * 0.5, down_1s * 0.5)
    complacency *= gates["COMPLACENCY"]
    terms = {
        "FEAR": fear_terms, "PANIC": panic_terms, "CAPITULATION": capitulation_terms,
        "DENIAL": denial_terms, "HOPE": hope_terms, "EUPHORIA": euphoria_terms,
        "FOMO": fomo_terms, "COMPLACENCY": complacency_terms,
    }

    by_band = {
        "micro": {
            "FEAR": _weighted([(0.5, bid_thin), (0.3, jitter), (0.2, surge)]),
            "PANIC": _weighted([(0.5, surge), (0.3, jitter), (0.2, blowout)]),
            "CAPITULATION": _weighted([(0.6, climax), (0.4, settling)]),
            "DENIAL": _weighted([(0.5, buy), (0.5, bottom)]),
            "HOPE": _weighted([(0.6, buy), (0.4, vol_calm)]),
            "EUPHORIA": _weighted([(0.6, buy), (0.4, tight)]),
            "FOMO": _weighted([(0.5, surge), (0.5, ask_thin)]),
            "COMPLACENCY": _weighted([(0.5, max(slow_tape, tight)), (0.5, tight)]),
        },
        "seconds": {
            "FEAR": _weighted([(0.5, sell), (0.3, down_1s), (0.2, vol_rise)]),
            "PANIC": _weighted([(0.5, down_1s), (0.3, surge), (0.2, blowout)]),
            "CAPITULATION": _weighted([(0.5, settling), (0.3, climax), (0.2, vol_calm)]),
            "DENIAL": _weighted([(0.6, dip_buying), (0.4, down_5s)]),
            "HOPE": _weighted([(0.6, up_1s), (0.4, recovery)]),
            "EUPHORIA": _weighted([(0.6, up_1s), (0.4, top)]),
            "FOMO": _weighted([(0.6, up_5s), (0.4, climax)]),
            "COMPLACENCY": _weighted([(0.6, vol_calm), (0.4, middle)]),
        },
        "window": {
            "FEAR": _weighted([(0.5, down_5s), (0.3, bottom), (0.2, vol_rise)]),
            "PANIC": _weighted([(0.6, down_1s), (0.4, blowout)]),
            "CAPITULATION": _weighted([(0.6, drawdown_extreme), (0.4, climax)]),
            "DENIAL": _weighted([(0.5, divergence), (0.5, bottom)]),
            "HOPE": _weighted([(0.6, recovery), (0.4, up_5s)]),
            "EUPHORIA": _weighted([(0.7, top), (0.3, up_5s)]),
            "FOMO": _weighted([(0.6, up_5s), (0.4, climax)]),
            "COMPLACENCY": _weighted([(0.6, middle), (0.4, vol_calm)]),
        },
        "minutes": {
            "FEAR": _weighted([(0.6, candle_down), (0.4, vol_rise)]),
            "PANIC": _weighted([(0.7, candle_down), (0.3, vol_rise)]),
            "CAPITULATION": _weighted([(0.7, drawdown_extreme), (0.3, settling)]),
            "DENIAL": _weighted([(0.7, bottom), (0.3, divergence)]),
            "HOPE": _weighted([(0.6, candle_up), (0.4, recovery)]),
            "EUPHORIA": _weighted([(0.6, top), (0.4, candle_up)]),
            "FOMO": _weighted([(0.6, candle_up), (0.4, run_up)]),
            "COMPLACENCY": _weighted([(0.6, vol_calm), (0.4, middle)]),
        },
        "news": {
            "FEAR": news_neg * (0.6 + 0.4 * news_flow),
            "PANIC": news_neg * (0.8 + 0.2 * news_flow),
            "CAPITULATION": news_neg * (0.5 + 0.5 * news_flow),
            "DENIAL": news_pos * down_5s,
            "HOPE": news_pos * (0.5 + 0.5 * news_flow),
            "EUPHORIA": news_pos * (0.6 + 0.4 * news_flow),
            "FOMO": news_pos * (0.6 + 0.4 * news_flow),
            # Silence in the headlines is not a feeling, so complacency has no
            # news band - otherwise "no news" would fake a calm reading.
            "COMPLACENCY": 0.0,
        },
    }

    drivers = {
        "FEAR": [
            f"sell aggression {f.get('aggression_5s', 0.0):+.2f}",
            f"volatility x{f.get('vol_ratio', 1.0):.2f} its own baseline",
            f"the 5 s move is {f.get('z_5s', 0.0):+.1f}x a typical one",
        ],
        "PANIC": [
            f"{f.get('tick_surge', 0.0):.1f}x the usual tick pace",
            f"1 s move {f.get('z_1s', 0.0):+.1f}x typical",
            f"jitter {format_us(f.get('jitter_us', 0.0))}",
        ],
        "CAPITULATION": [
            f"drawdown {f.get('z_drawdown', 0.0):.1f}x a typical minute",
            f"volume {f.get('volume_climax', 0.0):.1f}x the tape's pace",
            "the last second has gone quiet",
        ],
        "DENIAL": [
            f"buys absorbing a {f.get('z_5s', 0.0):+.1f}x move",
            f"news sentiment {f.get('news_sentiment', 0.0):+.2f}",
            f"range position {f.get('range_position', 0.5) * 100:.0f}%",
        ],
        "HOPE": [
            f"5 s move {f.get('z_5s', 0.0):+.1f}x typical, off the low",
            f"buy aggression {f.get('aggression_5s', 0.0):+.2f}",
            f"still at {f.get('range_position', 0.5) * 100:.0f}% of the range",
        ],
        "EUPHORIA": [
            f"price at {f.get('range_position', 0.5) * 100:.0f}% of the range",
            f"buy aggression {f.get('aggression_5s', 0.0):+.2f}",
            f"volatility x{f.get('vol_ratio', 1.0):.2f} - a calm melt-up",
        ],
        "FOMO": [
            f"5 s move {f.get('z_5s', 0.0):+.1f}x typical",
            f"volume {f.get('volume_climax', 0.0):.1f}x the tape's pace",
            f"ask side {'thin' if f.get('depth_imbalance', 0.0) < 0 else 'heavy'} ({f.get('depth_imbalance', 0.0):+.2f})",
        ],
        "COMPLACENCY": [
            f"volatility x{f.get('vol_ratio', 1.0):.2f}",
            f"spread x{f.get('spread_blowout', 1.0):.2f} of normal",
            f"{f.get('tick_rate_hz', 0.0):.1f} ticks/s",
        ],
    }

    order = [
        ("FEAR", fear), ("PANIC", panic), ("CAPITULATION", capitulation),
        ("DENIAL", denial), ("HOPE", hope), ("EUPHORIA", euphoria),
        ("FOMO", fomo), ("COMPLACENCY", complacency),
    ]
    out: list[EmotionScore] = []
    for name, value in order:
        label, tone = EMOTIONS[name]
        bands = {band: float(by_band[band][name]) for band in TIMESCALES}
        # An emotion that is loud on *one* band (a µs panic spike on an otherwise
        # quiet minute) must not be averaged away by the quiet ones, so the
        # strongest band lifts the score - but only part of the way.
        peak_band = max(bands, key=lambda b: bands[b])
        blended = _num(value) + 0.25 * max(0.0, bands[peak_band] - _num(value))
        out.append(
            EmotionScore(
                name=name,
                label=label,
                tone=tone,
                family=FAMILIES[name],
                intensity=float(min(1.0, max(0.0, blended))),
                by_timescale={k: round(min(1.0, max(0.0, v)), 4) for k, v in bands.items()},
                dominant_timescale=peak_band,
                drivers=tuple(drivers[name]),
                formula=EMOTION_FORMULAS[name],
                terms=tuple(terms[name]),
                gate=float(gates[name]),
            )
        )
    return out


def manipulation_read(f: dict[str, float]) -> dict:
    """The crowding / manipulation signature of the minute.

    ``score`` is what the fusion dampens confidence with.  ``kind`` names the
    dominant mechanism so the panel can say *how* the minute is being pushed,
    and ``evidence`` repeats the numbers behind the verdict.
    """
    herding = _num(f.get("herding"))
    whipsaw = _ramp(_num(f.get("reversals")), 3.0, 12.0)
    thin = _ramp(abs(_num(f.get("depth_imbalance"))), 0.15, 0.65)
    spoof = _weighted([
        (0.5, thin),
        (0.5, _ramp(_num(f.get("spread_blowout"), 1.0), 1.3, 3.0)),
    ])
    hunt = _num(f.get("stop_hunt"))
    # The climax arrives as a multiple of the tape's own pace (3.4x, 10.2x...);
    # the score needs it on the same 0-1 scale as every other component.
    volume = _ramp(_num(f.get("volume_climax")), 2.0, 8.0)
    # The deep detectors (backend.core.deep_micro): a burst that faded, prints
    # without price, resting size nobody trades against, one-sided flow and a
    # tape whose price the flow itself is moving.
    ignition = _num(f.get("deep_ignition"))
    stuffing = _num(f.get("deep_stuffing"))
    spoofing = _num(f.get("deep_spoofing"))
    toxicity = _num(f.get("deep_toxicity"))
    pushable = _num(f.get("deep_pushable"))
    score = _weighted([
        (0.22, herding), (0.13, whipsaw), (0.15, hunt), (0.08, spoof), (0.08, volume),
        (0.12, ignition), (0.08, toxicity), (0.06, stuffing), (0.04, spoofing), (0.04, pushable),
    ])
    kinds = {
        "retail chase": herding,
        "stop hunt": hunt,
        "whipsaw": whipsaw,
        "book imbalance": spoof,
        "momentum ignition": ignition,
        "toxic flow": toxicity * 0.8,
        "quote stuffing": stuffing,
        "spoofing": spoofing,
    }
    kind = max(kinds, key=lambda k: kinds[k])
    if score < 0.25:
        kind = "none"
    evidence = [
        f"herding {herding * 100:.0f}% (side skew and same-side runs)",
        f"{int(_num(f.get('reversals')))} direction flips in the last ~120 ticks",
        f"depth imbalance {_num(f.get('depth_imbalance')):+.2f}",
        f"stop-hunt wick score {hunt * 100:.0f}%",
        f"volume {_num(f.get('volume_climax')):.1f}x the tape's pace",
        f"ignition {ignition * 100:.0f}% (burst-then-fade), toxicity {toxicity * 100:.0f}% (VPIN)",
        f"stuffing {stuffing * 100:.0f}%, spoofing {spoofing * 100:.0f}%, pushable {pushable * 100:.0f}% (Kyle)",
    ]
    return {
        "score": round(float(score), 4),
        "percent": round(float(score) * 100.0, 1),
        "kind": kind,
        "components": {
            "herding": round(herding, 4),
            "whipsaw": round(whipsaw, 4),
            "stop_hunt": round(hunt, 4),
            "thin_book": round(spoof, 4),
            "volume_climax": round(volume, 4),
            "ignition": round(ignition, 4),
            "toxicity": round(toxicity, 4),
            "stuffing": round(stuffing, 4),
            "spoofing": round(spoofing, 4),
            "pushable": round(pushable, 4),
        },
        "evidence": evidence,
        "note": (
            "emotion-driven tape: the crowd, not new information, is moving it"
            if score >= 0.45
            else "some emotional pressure, still within normal microstructure"
            if score >= 0.25
            else "no crowding signature in this window"
        ),
    }


def analyze(
    tape: Any,
    asset: str,
    *,
    tracker: EmotionTracker | None = None,
    at_us: int | None = None,
    recent: int = 600,
    formulas: dict | None = None,
    directional: dict | None = None,
) -> EmotionReport:
    """Measure the crowd's emotions on one tape.

    ``tape`` is anything with the frozen-snapshot accessors (``ticks``,
    ``book``, ``candles``, ``spread_history``, ``news_items``), which is why the
    live 1 Hz loop can hand it a compacted live view of the same class.
    """
    report = EmotionReport(asset=asset.upper(), at_us=at_us or now_us())
    if tape is None:
        report.reason = "no tape"
        return report
    f = features(tape, asset, now_us_value=report.at_us, recent=recent)
    if not f.get("ticks"):
        report.reason = "no ticks on the tape yet"
        return report

    report.available = True
    report.ticks = int(f.get("ticks", 0))
    report.span_us = int(f.get("span_us", 0))
    report.resolution_us = float(f.get("resolution_us", 0.0))
    report.measured_over_us = int(f.get("span_us", 0.0))
    report.features = {
        k: v for k, v in f.items() if k not in ("at_us",)
    }
    report.scores = score_emotions(f)
    # The deep layer: microstructure formulas + the Bayesian filter.  Its
    # posterior is blended into the ramp intensities (the ramps say how *big*
    # the behaviour is, the filter how *consistent* the whole tape is with the
    # emotion), and its detectors feed the manipulation read.
    agreement = consensus(formulas or {}, directional or {})
    if tracker is not None and agreement["voters"] >= 3:
        # Smooth the live vote with the same time constant as the emotions,
        # so "aligned" / "conflict" is a state, not a flicker.
        raw = float(agreement["score"])
        tracker.consensus_ema = (
            raw if tracker.consensus_ema is None
            else tracker.consensus_ema + tracker.alpha * (raw - tracker.consensus_ema)
        )
        agreement = {**agreement, "score": tracker.consensus_ema, "raw_score": raw}
    report.deep = deep_micro.analyze(
        tape, asset, f,
        state=tracker.deep_state if tracker is not None else None,
        formula_consensus=agreement["score"],
        formula_voters=agreement["voters"],
    )
    posterior = (report.deep.get("posterior") or {}).get("posterior") or {}
    for score in report.scores:
        score.ramp = score.intensity
        if posterior:
            score.belief = float(posterior.get(score.name, 0.0))
            score.intensity = min(
                1.0, 0.65 * score.ramp + 0.35 * min(1.0, 2.5 * score.belief)
            )
    for key, value in (report.deep.get("manipulation") or {}).items():
        f[f"deep_{key}"] = float(value)
    report.manipulation = manipulation_read(f)
    if tracker is not None:
        report.tracker = tracker.observe(report)
        # The panel shows the smoothed *state*, not the raw half-second flicker:
        # the intensities are replaced by the tracker's EMA so the dominant
        # emotion, its hold time and the bars all describe the same thing.
        smoothed = report.tracker.get("smoothed") or {}
        for score in report.scores:
            if score.name in smoothed:
                score.intensity = float(min(1.0, max(0.0, smoothed[score.name])))
    report.formula_agreement = formula_agreement(agreement, _tone_bias(report.intensity))
    return report


def formula_agreement(agreement: dict, tone_bias: float) -> dict:
    """How the crowd's tone sits against the 22 formulas' weighted vote.

    ``alignment`` is in [-1, 1]: the product of the two signs times the
    smaller magnitude, so a strong vote with a flat crowd is *not* a conflict
    and a flat vote with an excited crowd is not agreement.  The note is the
    sentence the panel prints, and it always says who has the vote: the
    formulas carry 40% of the fusion and the crowd is a bounded confidence
    modifier - it never picks the side.
    """
    score = float(agreement.get("score", 0.0))
    voters = int(agreement.get("voters", 0))
    tone = float(tone_bias)
    aligned = score * tone
    alignment = (1.0 if aligned > 0 else -1.0 if aligned < 0 else 0.0) * min(abs(score), abs(tone))
    side = "BUY" if score > 0 else "SELL" if score < 0 else "split"
    crowd = "buying" if tone > 0.05 else "selling" if tone < -0.05 else "flat"
    up = int(agreement.get("up", 0))
    down = int(agreement.get("down", 0))
    vote = f"the weighted vote of {voters} formulas leans {side}, {up} up / {down} down"
    if voters < 3:
        verdict, note = "formulas silent", "fewer than 3 formulas voted - the crowd stands alone"
    elif abs(score) < 0.08:
        verdict, note = "formulas split", f"{voters} formulas are split; the crowd is {crowd}"
    elif abs(tone) < 0.08:
        verdict = "crowd flat"
        note = f"{vote}, while the crowd is flat"
    elif aligned > 0:
        verdict = "aligned"
        note = f"the crowd is {crowd} with the formulas ({vote}) - confidence stands"
    else:
        verdict = "conflict"
        note = (
            f"the crowd is {crowd} against the formulas ({vote}) - "
            "the formulas keep the vote, the crowd only trims confidence"
        )
    return {
        "available": voters >= 3,
        "consensus": round(score, 4),
        "consensus_raw": round(float(agreement.get("raw_score", score)), 4),
        "voters": voters,
        "up": up,
        "down": down,
        "up_names": list(agreement.get("up_names", []))[:4],
        "down_names": list(agreement.get("down_names", []))[:4],
        "formula_side": side,
        "crowd_tone": round(tone, 4),
        "crowd_side": crowd,
        "alignment": round(alignment, 4),
        "verdict": verdict,
        "note": note,
        "weights": {
            "formulas": 0.40,
            "agents": 0.25,
            "brain": 0.20,
            "news": 0.15,
            "crowd_max_confidence_cut": 0.125,
            "crowd_vote_scale": 0.5,
        },
        "rule": (
            "the 22 formulas carry 40% of the direction vote; the crowd never votes - "
            "it can only cut confidence by at most 12.5% when it disagrees (halved in Round P), "
            "and its tone counts at half weight in the learned evidence ledger"
        ),
    }
