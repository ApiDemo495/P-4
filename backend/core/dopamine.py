"""The brain's own dopamine - how it feels wins and losses (Round AR).

Human (mammalian) dopamine does not report reward, it reports **reward
prediction error** (Schultz, Dayan & Montague 1997): a burst when the outcome
beats what was expected, nothing when it merely meets it, a dip when it falls
short.  The expectation itself is learned (Rescorla-Wagner), the learning rate
rises after surprises (Pearce-Hall), losses are felt about 2.25x as hard as
gains of the same size (Kahneman & Tversky's loss aversion), and the slow
average of those errors is *mood* - tonic dopamine.

Humans then do two well-documented things with that chemistry that lose money:
after a winning streak they become over-confident and size up (the hot hand);
after losses they chase (tilt: risk-seeking in the loss domain).  This module
measures both in itself and does the opposite: the appetite multiplier can
only *reduce* size while the brain is elated or frustrated.

    u        = pnl               if pnl >= 0          subjective value (bps)
             = LAMBDA * pnl      if pnl <  0          loss aversion, LAMBDA = 2.25
    delta    = u - V                                   reward prediction error
    alpha    = clip(0.10 + 0.40 * |delta| / scale, 0.10, 0.50)   Pearce-Hall
    V       <- V + alpha * delta                       Rescorla-Wagner
    phasic   = tanh(delta / scale) * (0.6 + 0.4 * confidence_of_the_call)
    tonic   <- 0.85 * tonic + 0.15 * phasic            mood
    appetite = clip(1 - 0.35 * overconfidence - 0.35 * tilt, 0.5, 1.0)

``scale`` is the running mean |delta| (so a 5 bp surprise on a 3 bp tape is
a big one, on a 30 bp tape a small one).  Between outcomes the phasic burst
decays by 0.6 per window - dopamine transients are short.

The phasic value is what the fly brain's PAM / PPL1 dopamine neurons receive
on top of the DRG formula (``Brain.dopamine_phasic``); the mood and the two
bias indices are shown on the emotion card as the brain's own feeling.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field

import numpy as np

LAMBDA = 2.25            # loss aversion
ALPHA_MIN, ALPHA_MAX = 0.10, 0.50
TONIC_DECAY = 0.85
PHASIC_DECAY = 0.60
SCALE_DECAY = 0.90
HISTORY = 60

MOODS = {
    "ELATED": ("elated", "a run of better-than-expected windows - the classic over-confidence moment"),
    "CONTENT": ("content", "outcomes slightly above expectation"),
    "NEUTRAL": ("neutral", "outcomes about as expected - the healthy state"),
    "DISAPPOINTED": ("disappointed", "recent windows fell short of expectation"),
    "FRUSTRATED": ("frustrated", "a losing run below expectation - the tilt / loss-chasing moment"),
}


def _ramp(x: float, lo: float, hi: float) -> float:
    if hi <= lo:
        return 1.0 if x >= hi else 0.0
    return float(min(1.0, max(0.0, (x - lo) / (hi - lo))))


@dataclass
class DopamineBrain:
    expectation: float = 0.0        # V, in subjective bps
    scale: float = 5.0              # running mean |delta|, bps
    phasic: float = 0.0             # last burst / dip in [-1, 1], decaying
    tonic: float = 0.0              # mood in [-1, 1]
    win_streak: int = 0
    loss_streak: int = 0
    outcomes_seen: int = 0
    last_delta: float = 0.0
    last_alpha: float = 0.0
    last_u: float = 0.0
    last_pnl: float = 0.0
    last_confidence: float = 0.0
    last_outcome_at: float = 0.0
    history: deque = field(default_factory=lambda: deque(maxlen=HISTORY))

    # ------------------------------------------------------------------
    def observe(self, pnl_bps: float, outcome: float, confidence: float = 0.5) -> dict:
        """One scored window.  ``outcome`` is +1 / -1 / 0 (flat); flats teach
        the expectation (the window paid nothing) but cause no burst."""
        pnl = float(pnl_bps or 0.0)
        confidence = float(min(1.0, max(0.0, confidence or 0.0)))
        u = pnl if pnl >= 0 else LAMBDA * pnl
        delta = u - self.expectation
        scale = max(self.scale, 1e-6)
        alpha = float(np.clip(ALPHA_MIN + 0.40 * abs(delta) / scale, ALPHA_MIN, ALPHA_MAX))
        self.expectation += alpha * delta
        self.scale = SCALE_DECAY * self.scale + (1.0 - SCALE_DECAY) * abs(delta)
        # a confident call that fails hurts more; a confident call that wins
        # was expected to - humans discount it (and so does the burst)
        if outcome > 0:
            weight = 1.0 - 0.4 * confidence
        elif outcome < 0:
            weight = 0.6 + 0.4 * confidence
        else:
            weight = 0.0
        phasic = float(np.tanh(delta / scale)) * weight
        self.phasic = phasic
        self.tonic = TONIC_DECAY * self.tonic + (1.0 - TONIC_DECAY) * phasic
        if outcome > 0:
            self.win_streak += 1
            self.loss_streak = 0
        elif outcome < 0:
            self.loss_streak += 1
            self.win_streak = 0
        self.outcomes_seen += 1
        self.last_delta, self.last_alpha, self.last_u = delta, alpha, u
        self.last_pnl, self.last_confidence = pnl, confidence
        self.last_outcome_at = time.time()
        row = {"pnl_bps": round(pnl, 2), "u": round(u, 2), "expectation_before": round(self.expectation - alpha * delta, 2),
               "delta": round(delta, 2), "phasic": round(phasic, 4), "tonic": round(self.tonic, 4), "outcome": float(outcome),
               "at": self.last_outcome_at}
        self.history.append(row)
        return row

    def tick_window(self) -> None:
        """A window passed without a new outcome: the transient fades."""
        self.phasic *= PHASIC_DECAY
        if abs(self.phasic) < 1e-4:
            self.phasic = 0.0

    # ------------------------------------------------------------------
    @property
    def overconfidence(self) -> float:
        """Hot-hand index: a winning streak while elated."""
        return _ramp(self.win_streak, 2.0, 5.0) * _ramp(self.tonic, 0.15, 0.55)

    @property
    def tilt(self) -> float:
        """Loss-chasing index: a losing streak while frustrated."""
        return _ramp(self.loss_streak, 1.0, 4.0) * _ramp(-self.tonic, 0.15, 0.55)

    @property
    def appetite(self) -> float:
        """Size multiplier.  Only ever <= 1: the brain protects itself from its
        own chemistry instead of acting on it."""
        return float(np.clip(1.0 - 0.35 * self.overconfidence - 0.35 * self.tilt, 0.5, 1.0))

    @property
    def mood(self) -> str:
        t = self.tonic
        if t > 0.35:
            return "ELATED"
        if t > 0.10:
            return "CONTENT"
        if t < -0.35:
            return "FRUSTRATED"
        if t < -0.10:
            return "DISAPPOINTED"
        return "NEUTRAL"

    def brain_gain(self) -> float:
        """What the PAM / PPL1 neurons receive from the brain's own chemistry."""
        return float(np.clip(self.phasic, -1.0, 1.0))

    # ------------------------------------------------------------------
    def to_dict(self) -> dict:
        mood = self.mood
        label, meaning = MOODS[mood]
        oc, tilt = self.overconfidence, self.tilt
        guard = []
        if oc > 0.05:
            guard.append(f"over-confidence guard {oc:.0%} after {self.win_streak} wins in a row")
        if tilt > 0.05:
            guard.append(f"tilt guard {tilt:.0%} after {self.loss_streak} losses in a row")
        return {
            "available": self.outcomes_seen > 0,
            "mood": mood, "mood_label": label, "mood_meaning": meaning,
            "phasic": round(self.phasic, 4), "tonic": round(self.tonic, 4),
            "expectation_bps": round(self.expectation, 2), "scale_bps": round(self.scale, 2),
            "last": {"pnl_bps": round(self.last_pnl, 2), "subjective_bps": round(self.last_u, 2),
                     "delta_bps": round(self.last_delta, 2), "alpha": round(self.last_alpha, 3),
                     "confidence": round(self.last_confidence, 3), "at": self.last_outcome_at},
            "win_streak": self.win_streak, "loss_streak": self.loss_streak, "outcomes": self.outcomes_seen,
            "overconfidence": round(oc, 3), "tilt": round(tilt, 3), "appetite": round(self.appetite, 3),
            "brain_gain": round(self.brain_gain(), 4),
            "guard": guard,
            "reading": self._reading(),
            "equations": [
                "u = pnl (gain) | 2.25·pnl (loss)  — loss aversion",
                "δ = u − V  — reward prediction error (Schultz)",
                "α = clip(0.10 + 0.40·|δ|/scale, 0.10, 0.50)  — Pearce-Hall",
                "V ← V + α·δ  — Rescorla-Wagner expectation",
                "phasic = tanh(δ/scale) × (1 − 0.4·conf | 0.6 + 0.4·conf)",
                "tonic ← 0.85·tonic + 0.15·phasic  — mood",
                "appetite = clip(1 − 0.35·overconfidence − 0.35·tilt, 0.5, 1)",
            ],
            "history": list(self.history)[-12:],
        }

    def _reading(self) -> str:
        if self.outcomes_seen == 0:
            return "no scored window yet - expectation V = 0, the first outcome is pure surprise"
        d = self.last_delta
        kind = "burst" if self.phasic > 0.05 else "dip" if self.phasic < -0.05 else "no transient"
        s = (f"last window {self.last_pnl:+.1f} bp felt like {self.last_u:+.1f} bp against an expectation of "
             f"{self.expectation - self.last_alpha * d:+.1f} bp → prediction error {d:+.1f} bp → dopamine {kind} "
             f"{self.phasic:+.2f}; mood {self.mood.lower()} ({self.tonic:+.2f})")
        if self.appetite < 0.999:
            s += f"; size held at {self.appetite:.2f}× (a human would be sizing up here)"
        return s
