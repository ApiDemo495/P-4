"""Binary direction policy - **every window resolves to BUY or SELL**.

The specification's decision layer had a third state, ``HOLD`` ("no trade").
It was removed at the user's request: a signal panel that says HOLD is not
actionable, and every HOLD branch was a place where the app could sit silent for
minutes at a time.  What replaces it is not "trade always" - it is *forced
direction with a declared conviction*:

* the engine still measures the edge exactly as before (fused score, CCSv2,
  confidence, HSI dampening, evidence quality);
* it always names the side that the evidence points to, and says *why* that side
  won (``source``), how big the edge is (``edge``) and how much it trusts it
  (``conviction``);
* when the edge is inside the neutral band the side is chosen by an explicit,
  documented tie-break ladder instead of a silent HOLD, and the result is marked
  ``weak`` so the UI can tell the user to reduce size.

Tie-break ladder (first rule that has a non-zero value wins):

    1. sign of the fused score            -> "fused score"
    2. sign of the brain's CCSv2 read-out -> "brain CCSv2"
    3. the direction of the previous window ("sticky") -> "previous window"
    4. BUY as the last-resort default (the ladder is total on purpose)

An emergency override is *also* binary: "exit any open position" is expressed by
signalling the opposite side of the direction that is currently open, which is
the only way to flatten a position in a two-state system.  If nothing is open,
the emergency keeps the direction the engine had and simply flags itself.
"""

from __future__ import annotations

from dataclasses import dataclass

BUY = "BUY"
SELL = "SELL"
DIRECTIONS = (BUY, SELL)

#: Below this |score| the fused ensemble has no usable lean and the ladder moves
#: on to the brain.  Well under ``signal_threshold`` (0.25) on purpose: the
#: ladder is about *which side*, the threshold is about *how strongly*.
NEUTRAL_BAND = 0.02

#: Round BR: the brain's CCSv2 may break a tie only when the circuit is at
#: least this confident in the pattern it recognised - a 6 % read-out is noise.
BRAIN_TIE_MIN_CONFIDENCE = 0.25

#: |score| at which the edge is considered full size for the conviction label.
FULL_EDGE = 0.50


def opposite(signal: str | None) -> str | None:
    if signal == BUY:
        return SELL
    if signal == SELL:
        return BUY
    return None


def is_direction(value: str | None) -> bool:
    return value in DIRECTIONS


@dataclass(frozen=True)
class DirectionDecision:
    """The resolved side plus everything the UI needs to justify it."""

    decision: str
    edge: float
    conviction: str  # "HIGH" | "MEDIUM" | "LOW"
    source: str  # why this side won
    weak: bool  # True when the side came from a fallback, not from a real edge
    tie_break: str = ""  # the ladder rule that fired, for the reasoning line
    emergency_exit: bool = False
    closed_signal: str | None = None

    def to_dict(self) -> dict:
        return {
            "decision": self.decision,
            "edge": round(self.edge, 4),
            "conviction": self.conviction,
            "direction_source": self.source,
            "weak": self.weak,
            "tie_break": self.tie_break,
            "emergency_exit": self.emergency_exit,
            "closed_signal": self.closed_signal,
        }


def conviction_label(edge: float, confidence: float, settings) -> str:
    """HIGH / MEDIUM / LOW from the edge size and the fused confidence."""
    strong_edge = edge >= settings.signal_threshold
    strong_conf = confidence >= settings.min_fusion_confidence
    if strong_edge and strong_conf:
        return "HIGH"
    if strong_edge or strong_conf:
        return "MEDIUM"
    return "LOW"


def resolve(
    *,
    score: float,
    ccs_value: float,
    confidence: float,
    previous: str | None = None,
    settings,
    emergency: bool = False,
    degraded: bool = False,
    window_lean: float | None = None,
    ccs_confidence: float = 1.0,
    neutral_band: float | None = None,
) -> DirectionDecision:
    """Return the BUY/SELL side for this window (never a third state).

    Round BR: ``neutral_band`` is the measured noise floor of the fused score
    (cycle manager: half the median |score| of recent windows, never below
    ``NEUTRAL_BAND``); inside it the sign of the sum is noise and the ladder
    moves to the horizon-matched ``window_lean`` (the nine WINDOW formulas,
    designed for exactly this horizon), then to the brain when it is
    confident, then to the previous window.
    """
    edge = min(1.0, abs(float(score)) / max(float(settings.signal_threshold), 1e-9) * 0.5)
    band = max(NEUTRAL_BAND, float(neutral_band)) if neutral_band is not None else NEUTRAL_BAND

    # ---- emergency: binary encoding of "exit any open position" ----------
    if emergency:
        closed = previous if is_direction(previous) else None
        exit_side = opposite(closed)
        if exit_side is not None:
            return DirectionDecision(
                decision=exit_side,
                edge=max(edge, 0.0),
                conviction="HIGH",
                source=f"emergency exit of the open {closed}",
                weak=False,
                tie_break="emergency",
                emergency_exit=True,
                closed_signal=closed,
            )
        decision, tie_break, ladder_source = _ladder(score, ccs_value, previous, window_lean, ccs_confidence, band)
        return DirectionDecision(
            decision=decision,
            edge=edge,
            conviction="LOW",
            source=f"emergency fired with nothing open; side unchanged ({ladder_source})",
            weak=True,
            tie_break="emergency-no-position",
        )

    # ---- normal path: which side does the evidence point to? -------------
    decision, tie_break, source = _ladder(score, ccs_value, previous, window_lean, ccs_confidence, band)

    weak = abs(float(score)) < band or degraded
    if degraded:
        source = f"{source} (degraded evidence - 11+ formulas returned zero)"
    return DirectionDecision(
        decision=decision,
        edge=edge,
        conviction="LOW" if degraded else conviction_label(edge, confidence, settings),
        source=source,
        weak=weak,
        tie_break=tie_break,
    )


def _ladder(
    score: float, ccs_value: float, previous: str | None,
    window_lean: float | None = None, ccs_confidence: float = 1.0, band: float = NEUTRAL_BAND,
) -> tuple[str, str, str]:
    """The tie-break ladder.  Always returns a side.

    1. fused score outside the noise floor ``band``;
    2. the horizon-matched window lean (nine WINDOW formulas) when it has a sign;
    3. the brain's CCSv2 when the circuit is confident (>= 25 %);
    4. the previous window's side; 5. BUY.
    """
    score = float(score)
    ccs_value = float(ccs_value)

    if abs(score) > band:
        return (
            BUY if score > 0 else SELL,
            "fused score",
            f"fused score {score:+.3f} leans "
            f"{'up' if score > 0 else 'down'} (|score| > noise floor {band:.3f})",
        )
    if window_lean is not None and abs(float(window_lean)) > 1e-6:
        wl = float(window_lean)
        return (
            BUY if wl > 0 else SELL,
            "window formulas",
            f"fused score {score:+.3f} is inside the noise floor ({band:.3f}); the {int(round(_window_seconds()))} s "
            f"window formulas lean {wl:+.3f} and decide the side",
        )
    if abs(ccs_value) > 1e-9 and float(ccs_confidence) >= BRAIN_TIE_MIN_CONFIDENCE:
        return (
            BUY if ccs_value > 0 else SELL,
            "brain CCSv2",
            f"score and window formulas are flat; brain CCSv2 {ccs_value:+.3f} "
            f"({float(ccs_confidence):.0%} confident) decides the side",
        )
    if is_direction(previous):
        return (
            previous,
            "previous window",
            "score and brain are both flat; holding the previous direction",
        )
    return BUY, "default", "no usable evidence at all; defaulting to BUY"


def _window_seconds() -> float:
    try:
        from backend.core import config as cfg

        return float(cfg.window_seconds())
    except Exception:  # noqa: BLE001
        return 70.0


def describe(decision: DirectionDecision, confidence: float) -> str:
    """One sentence the panel can print next to the signal."""
    size = {"HIGH": "full size", "MEDIUM": "half size", "LOW": "quarter size"}[
        decision.conviction
    ]
    if decision.emergency_exit:
        return (
            f"{decision.decision} closes the open {decision.closed_signal}: emergency exit, "
            f"flat is the only safe state."
        )
    return (
        f"{decision.decision} at {decision.conviction.lower()} conviction "
        f"({size}, confidence {confidence:.0%}) - {decision.source}."
    )
