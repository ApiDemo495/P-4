"""Agent decision fusion (Section 8).

    drosophila 0.40 | gemini 0.25 | local 0.20 | github 0.15

Weights are renormalised over the agents that actually answered this cycle, so
losing Gemini does not silently halve the signal.

Two v2.0 additions:

* **HSI override** - when the Hedge Stress Index exceeds 0.80 the final
  confidence is multiplied by ``max(0.2, 1 - HSI)``.  During extreme hedge
  stress even a strong signal is dampened; the 0.2 floor stops it being zeroed.
* **Binary direction** - the third state (HOLD) was removed at the user's
  request.  Every gate that used to return HOLD now returns a side plus the
  reason it won; see :mod:`backend.core.direction` for the ladder.  The gates
  therefore degrade *conviction* and *size*, never the existence of a signal.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

from backend.agents.base import AgentResult
from backend.core import config as cfg
from backend.core.direction import (
    BUY,
    SELL,
    DirectionDecision,
    describe as describe_direction,
    opposite,
    resolve as resolve_direction,
)

log = logging.getLogger("drosophila.fusion")


@dataclass
class FusionResult:
    decision: str
    confidence: float
    raw_confidence: float
    score: float
    reasoning: str = ""
    contributions: dict = field(default_factory=dict)
    forced_reason: str = ""
    hsi_adjustment: float = 1.0
    weights_used: dict = field(default_factory=dict)
    direction: DirectionDecision | None = None
    formula_consensus: float | None = None
    confirmation: float = 1.0
    calibration: float = 1.0

    # -- binary direction fields (HOLD was removed) ----------------------
    @property
    def edge(self) -> float:
        return self.direction.edge if self.direction else 0.0

    @property
    def conviction(self) -> str:
        return self.direction.conviction if self.direction else "LOW"

    @property
    def weak(self) -> bool:
        return bool(self.direction.weak) if self.direction else True

    @property
    def direction_source(self) -> str:
        return self.direction.source if self.direction else ""

    def to_dict(self) -> dict:
        payload = {
            "decision": self.decision,
            "confidence": round(self.confidence, 4),
            "raw_confidence": round(self.raw_confidence, 4),
            "score": round(self.score, 4),
            "reasoning": self.reasoning,
            "contributions": self.contributions,
            "forced_reason": self.forced_reason,
            "hsi_adjustment": round(self.hsi_adjustment, 4),
            "weights_used": {k: round(v, 4) for k, v in self.weights_used.items()},
            "formula_consensus": None
            if self.formula_consensus is None
            else round(float(self.formula_consensus), 4),
            "confirmation": self.confirmation,
            "calibration": self.calibration,
        }
        # Kept for the two clients: "lean" now always equals the decision,
        # because there is no third state to lean away from.
        payload["lean"] = self.decision
        if self.direction is not None:
            payload.update(self.direction.to_dict())
        return payload


def _direction_value(decision: str | None) -> float:
    return {"BUY": 1.0, "SELL": -1.0}.get(decision or "", 0.0)


def fuse(
    agents: dict[str, AgentResult],
    ccs_value: float,
    ccs_confidence: float,
    hsi: float,
    settings=None,
    warnings: list[str] | None = None,
    insufficient_evidence: bool = False,
    emergency: bool = False,
    previous_signal: str | None = None,
    formula_consensus: float | None = None,
    consensus_voters: int = 0,
    recent_accuracy: dict | None = None,
) -> FusionResult:
    """Combine the Drosophila brain with the available AI agents.

    Always returns BUY or SELL: see :mod:`backend.core.direction`.

    ``formula_consensus`` is the weighted agreement of the directional formulas
    (-1..+1) and ``recent_accuracy`` the measured hit rate of recent windows.
    Neither moves the *side* - the direction ladder stays the spec's - but both
    move the confidence, because a side that the tape's own formulas contradict,
    or that has been losing recently, is not a side to size up on.
    """
    settings = settings or cfg.SETTINGS
    warnings = warnings or []

    # ------------------------------------------------------------------
    # Weighted score
    # ------------------------------------------------------------------
    default_weights = {
        "drosophila": settings.weight_drosophila,
        "gemini": settings.weight_gemini,
        "local": settings.weight_local,
        "github": settings.weight_github,
    }

    contributions: dict[str, dict] = {}
    active: dict[str, float] = {}

    # The Drosophila brain is always available: it is computed synchronously.
    active["drosophila"] = default_weights["drosophila"]
    contributions["drosophila"] = {
        "decision": BUY if ccs_value >= 0 else SELL,
        "confidence": round(ccs_confidence, 4),
        "value": round(ccs_value, 4),
        "weight": default_weights["drosophila"],
        "status": "LIVE",
        # Where the brain's vote lands in the final number - the UI shows this
        # so "the fly is 40 % of the decision" is verifiable, not a claim.
        "weighted_value": round(default_weights["drosophila"] * max(-1.0, min(1.0, ccs_value)), 4),
        "source": "80-node mushroom body, 3-layer graph convolution",
    }

    for name in ("gemini", "local", "github"):
        result = agents.get(name)
        if result is not None and result.available:
            active[name] = default_weights[name]
            contributions[name] = {
                "decision": result.decision,
                "confidence": round(result.confidence or 0.0, 4),
                "value": round(_direction_value(result.decision) * (result.confidence or 0.0), 4),
                "weight": default_weights[name],
                "status": result.status.value,
                "latency_ms": round(result.latency_ms, 1),
                "model": result.model,
            }

    total_weight = sum(active.values()) or 1e-9
    weights_used = {name: weight / total_weight for name, weight in active.items()}

    score = 0.0
    for name, weight in active.items():
        if name == "drosophila":
            score += weight * max(-1.0, min(1.0, ccs_value))
        else:
            result = agents[name]
            score += weight * _direction_value(result.decision) * float(result.confidence or 0.0)
    score /= total_weight

    # ------------------------------------------------------------------
    # HSI override (Section 8.2)
    # ------------------------------------------------------------------
    hsi_adjustment = 1.0
    if hsi > settings.hsi_dampen_threshold:
        hsi_adjustment = max(settings.hsi_confidence_floor, 1.0 - hsi)

    # --- confidence -----------------------------------------------------
    # Two things make a signal trustworthy: how far the ensemble is from
    # neutral (magnitude), and how confident the brain is in the pattern it
    # recognised (ccs_confidence).  Magnitude is normalised against twice the
    # decision threshold, i.e. |score| = 0.5 reads as full conviction.
    directions = [
        1.0 if value["value"] > 0 else -1.0 if value["value"] < 0 else 0.0
        for value in contributions.values()
    ]
    directional = [d for d in directions if d != 0.0]
    agreement = abs(float(np.mean(directional))) if len(directional) > 1 else (1.0 if directional else 0.0)

    magnitude = min(1.0, abs(score) / max(2.0 * settings.signal_threshold, 1e-9))

    agent_confidences = [
        float(contributions[name]["confidence"])
        for name in ("gemini", "local", "github")
        if name in contributions
    ]
    brain_conf = max(0.0, min(1.0, ccs_confidence))
    if agent_confidences:
        agent_conf = float(np.mean(agent_confidences))
        brain_term = 0.6 * brain_conf + 0.4 * agent_conf
    else:
        # No AI agent answered: the Drosophila brain is the sole decision maker
        # (degradation level 3), so its confidence carries the whole weight.
        brain_term = brain_conf

    raw_confidence = 0.65 * magnitude + 0.35 * brain_term
    raw_confidence *= 0.85 + 0.15 * agreement

    # --- formula confirmation ------------------------------------------
    # The 22 formulas are the only inputs that are verified against known
    # tapes (backend/formulas/self_test.py).  When they disagree with the side
    # the fusion picked, the honest move is to trust the side less rather than
    # to flip it: the ladder has already accounted for the same inputs.
    confirmation = 1.0
    consensus_note = ""
    if formula_consensus is not None and consensus_voters >= 3:
        side_value = 1.0 if score >= 0 else -1.0
        alignment = side_value * float(formula_consensus)
        if alignment >= 0:
            confirmation = 0.90 + 0.10 * min(1.0, alignment)
            consensus_note = f"formulas agree ({formula_consensus:+.2f})"
        else:
            confirmation = 1.0 - 0.35 * min(1.0, abs(alignment))
            consensus_note = f"formulas disagree ({formula_consensus:+.2f})"
    raw_confidence *= confirmation

    # --- realised-form calibration --------------------------------------
    # If the last windows of this kind have been losing, a high confidence is a
    # claim the engine has not been earning.  Damp it and say so.
    calibration = 1.0
    accuracy_note = ""
    accuracy = recent_accuracy or {}
    evaluated = int(accuracy.get("evaluated") or 0)
    if evaluated >= 8:
        win_rate = float(accuracy.get("win_rate") or 0.0)
        if win_rate < 0.50:
            calibration = 0.80 + 0.20 * (win_rate / 0.50)
            accuracy_note = f"recent form {win_rate:.0%} of {evaluated}"
    raw_confidence *= calibration

    confidence = raw_confidence * hsi_adjustment
    confidence = max(0.0, min(0.95, confidence))

    # ------------------------------------------------------------------
    # Decision: always a side (Section 10.1, amended to binary)
    # ------------------------------------------------------------------
    degraded = insufficient_evidence or any(
        "Insufficient" in w for w in warnings
    )
    direction = resolve_direction(
        score=score,
        ccs_value=ccs_value,
        confidence=confidence,
        previous=previous_signal,
        settings=settings,
        emergency=emergency,
        degraded=degraded,
    )

    forced_reason = ""
    if emergency:
        forced_reason = "Emergency override active"
    elif degraded:
        forced_reason = "Insufficient formula evidence"

    reasoning = _explain(
        direction,
        score,
        contributions,
        ccs_value,
        ccs_confidence,
        hsi,
        hsi_adjustment,
        settings,
        confidence,
        consensus_note=consensus_note,
        accuracy_note=accuracy_note,
    )

    return FusionResult(
        decision=direction.decision,
        confidence=confidence,
        raw_confidence=raw_confidence,
        score=score,
        reasoning=reasoning,
        contributions=contributions,
        forced_reason=forced_reason,
        hsi_adjustment=hsi_adjustment,
        weights_used=weights_used,
        direction=direction,
        formula_consensus=formula_consensus,
        confirmation=round(confirmation, 4),
        calibration=round(calibration, 4),
    )


def _explain(
    direction: DirectionDecision,
    score: float,
    contributions: dict,
    ccs_value: float,
    ccs_confidence: float,
    hsi: float,
    hsi_adjustment: float,
    settings,
    confidence: float,
    consensus_note: str = "",
    accuracy_note: str = "",
) -> str:
    parts: list[str] = []

    ranked = sorted(
        contributions.items(),
        key=lambda kv: abs(float(kv[1].get("value", 0.0))),
        reverse=True,
    )
    top = [f"{name} {value['decision']} ({value['confidence']:.0%})" for name, value in ranked[:2]]
    if top:
        parts.append("Strongest inputs: " + ", ".join(top))

    if hsi_adjustment < 1.0:
        parts.append(
            f"HSI={hsi:.2f} above the {settings.hsi_dampen_threshold:.2f} stress threshold - "
            f"confidence multiplied by {hsi_adjustment:.2f}"
        )
    else:
        parts.append(f"Hedge stress low (HSI={hsi:.2f})")

    parts.append(f"brain CCSv2={ccs_value:+.2f} at {ccs_confidence:.0%} confidence")
    if consensus_note:
        parts.append(consensus_note)
    if accuracy_note:
        parts.append(f"confidence dampened on {accuracy_note}")
    parts.append(describe_direction(direction, confidence))
    if direction.tie_break and direction.tie_break not in ("emergency", "fused score"):
        parts.append(f"tie-break: {direction.tie_break}")
    return ". ".join(parts) + "."


def conviction_note(direction: DirectionDecision, confidence: float) -> dict | None:
    """Replaces the Section 10.2 HOLD box.

    The box existed because a HOLD leaves the user with no instruction.  Now
    every window carries a direction, so the equivalent message is about *size*
    rather than about the absence of a signal: it appears when the side came
    from the tie-break ladder (``weak``) or when the engine's own conviction is
    below HIGH - which is exactly the case the 0.80 HSI dampening produces.
    Ordinary high-conviction windows stay quiet.
    """
    if not direction.weak and direction.conviction == "HIGH":
        return None

    if direction.emergency_exit:
        text = (
            f"Emergency exit: {direction.decision} closes the open "
            f"{direction.closed_signal}. Get flat first, decide after."
        )
    elif direction.weak:
        text = (
            f"Thin edge on this window: the ensemble is inside the neutral band, so the "
            f"{direction.decision} side was chosen by {direction.tie_break} rather than by a "
            f"strong score. Take it at reduced size - half or less - and keep the stop tight."
        )
    else:
        text = (
            f"{direction.decision} at {direction.conviction.lower()} conviction "
            f"(confidence {confidence:.0%}): the side is clear but the evidence behind it is "
            f"thin. Take it at reduced size."
        )
    return {
        "visible": True,
        "conviction": direction.conviction,
        "text": text,
        "direction": direction.decision,
        "source": direction.source,
        "edge": round(direction.edge, 4),
        "confidence": round(confidence, 4),
    }
