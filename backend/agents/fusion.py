"""Agent decision fusion (Section 8).

    drosophila 0.40 | gemini 0.25 | local 0.20 | github 0.15

Weights are renormalised over the agents that actually answered this cycle, so
losing Gemini does not silently halve the signal.

Two v2.0 additions:

* **HSI override** - when the Hedge Stress Index exceeds 0.80 the final
  confidence is multiplied by ``max(0.2, 1 - HSI)``.  During extreme hedge
  stress even a strong signal is dampened; the 0.2 floor stops it being zeroed.
* **Hard gates** - emergency override, insufficient data and the "11 of 20
  formulas are zero" rule all force HOLD regardless of what the agents say.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

from backend.agents.base import AgentResult
from backend.core import config as cfg

log = logging.getLogger("drosophila.fusion")


@dataclass
class FusionResult:
    decision: str
    confidence: float
    raw_confidence: float
    score: float
    lean: str | None = None
    reasoning: str = ""
    contributions: dict = field(default_factory=dict)
    forced_reason: str = ""
    hsi_adjustment: float = 1.0
    weights_used: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "decision": self.decision,
            "confidence": round(self.confidence, 4),
            "raw_confidence": round(self.raw_confidence, 4),
            "score": round(self.score, 4),
            "lean": self.lean,
            "reasoning": self.reasoning,
            "contributions": self.contributions,
            "forced_reason": self.forced_reason,
            "hsi_adjustment": round(self.hsi_adjustment, 4),
            "weights_used": {k: round(v, 4) for k, v in self.weights_used.items()},
        }


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
) -> FusionResult:
    """Combine the Drosophila brain with the available AI agents."""
    settings = settings or cfg.SETTINGS
    warnings = warnings or []

    # ------------------------------------------------------------------
    # Hard gates first (Section 10.1)
    # ------------------------------------------------------------------
    if emergency:
        return FusionResult(
            decision="HOLD",
            confidence=1.0,
            raw_confidence=1.0,
            score=0.0,
            forced_reason="Emergency override active",
            reasoning="Critical news event: the only safe action is to stop.",
        )
    if insufficient_evidence:
        return FusionResult(
            decision="HOLD",
            confidence=0.5,
            raw_confidence=0.5,
            score=0.0,
            forced_reason="Insufficient formula evidence",
            reasoning=(
                "11 or more of the 20 input formulas returned zero, which means the data "
                "feed is degraded. Forcing HOLD."
            ),
        )

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
        "decision": "BUY" if ccs_value > 0.25 else "SELL" if ccs_value < -0.25 else "HOLD",
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
    confidence = raw_confidence * hsi_adjustment
    confidence = max(0.0, min(0.95, confidence))

    # ------------------------------------------------------------------
    # Decision (Section 10.1)
    # ------------------------------------------------------------------
    threshold = settings.signal_threshold
    if abs(ccs_value) <= threshold or confidence <= settings.min_fusion_confidence:
        decision = "HOLD"
    elif score > 0:
        decision = "BUY"
    elif score < 0:
        decision = "SELL"
    else:
        decision = "HOLD"

    lean = None
    if decision == "HOLD":
        lean = "BUY" if score > 0.05 else "SELL" if score < -0.05 else None

    if any(w.startswith("Insufficient") for w in warnings) or warnings:
        if any("Insufficient" in w for w in warnings):
            decision = "HOLD"
            lean = "BUY" if score > 0 else "SELL" if score < 0 else None

    reasoning = _explain(
        decision, score, contributions, ccs_value, ccs_confidence, hsi, hsi_adjustment, settings
    )

    return FusionResult(
        decision=decision,
        confidence=confidence,
        raw_confidence=raw_confidence,
        score=score,
        lean=lean,
        reasoning=reasoning,
        contributions=contributions,
        hsi_adjustment=hsi_adjustment,
        weights_used=weights_used,
    )


def _explain(
    decision: str,
    score: float,
    contributions: dict,
    ccs_value: float,
    ccs_confidence: float,
    hsi: float,
    hsi_adjustment: float,
    settings,
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

    if decision == "HOLD":
        parts.append("No trade: score or confidence did not clear the entry threshold")
    return ". ".join(parts) + "."


def hold_warning(score: float, lean: str | None, confidence: float) -> dict:
    """The verbatim HOLD warning box from Section 10.2."""
    return {
        "visible": True,
        "text": (
            "This prediction is not as powerful as it should be because we are receiving HOLD "
            "signals. However, if you urgently need to take a trade, you may follow the BUY or "
            "SELL prediction shown - but proceed with caution."
        ),
        "lean": lean,
        "lean_score": round(score, 4),
        "confidence": round(confidence, 4),
    }
