"""Agent decision fusion (Section 8).

    drosophila 0.40 | gemini 0.25 | local 0.20 | github 0.15

Weights are renormalised over the agents that actually answered this cycle, so
losing Gemini does not silently halve the signal.

Two v2.0 additions:

* **HSI override** - when the Hedge Stress Index exceeds 0.80 the final
  confidence is multiplied by ``max(0.2, 1 - HSI)``.  During extreme hedge
  stress even a strong signal is dampened; the 0.2 floor stops it being zeroed.
* **Crowd-emotion dampener** - the emotion engine measures how crowded and
  emotional the minute is (herding, whipsaw, stop hunts, book imbalance, a
  volume climax).  When that manipulation score passes
  ``emotion_dampen_threshold`` the confidence is multiplied by
  ``1 - emotion_dampen_max * score`` (max cut 12.5 %, halved in Round P): an emotional tape is a tape to trade
  smaller, never a reason to flip a locked side.
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
from backend.core import lock_weights
from backend.core.direction import (
    BUY,
    SELL,
    DirectionDecision,
    describe as describe_direction,
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
    lock_weights: dict = field(default_factory=dict)
    direction: DirectionDecision | None = None
    formula_consensus: float | None = None
    confirmation: float = 1.0
    calibration: float = 1.0
    crowd_adjustment: float = 1.0
    crowd_note: str = ""
    learned: dict = field(default_factory=dict)
    spec_score: float = 0.0
    physics: dict = field(default_factory=dict)
    genesis: dict = field(default_factory=dict)

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
            "lock_weights": self.lock_weights,
            "formula_consensus": None
            if self.formula_consensus is None
            else round(float(self.formula_consensus), 4),
            "confirmation": self.confirmation,
            "calibration": self.calibration,
            "crowd_adjustment": round(self.crowd_adjustment, 4),
            "crowd_note": self.crowd_note,
            "learned": self.learned,
            "spec_score": round(self.spec_score, 4),
            "physics": self.physics,
            "genesis": self.genesis,
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
    crowd: dict | None = None,
    learned: dict | None = None,
    physics: dict | None = None,
    genesis: dict | None = None,
    news_impact: dict | None = None,
    asset: str = "BTC",
    ledger_sources: list | None = None,
) -> FusionResult:
    """Combine the Drosophila brain with the available AI agents.

    Always returns BUY or SELL: see :mod:`backend.core.direction`.

    ``formula_consensus`` is the weighted agreement of the directional formulas
    (-1..+1), ``recent_accuracy`` the measured hit rate of recent windows and
    ``crowd`` the emotion engine's reading of the tape (see
    :mod:`backend.core.emotions`).
    Neither moves the *side* - the direction ladder stays the spec's - but both
    move the confidence, because a side that the tape's own formulas contradict,
    or that has been losing recently, is not a side to size up on.
    """
    settings = settings or cfg.SETTINGS
    warnings = warnings or []

    # ------------------------------------------------------------------
    # Weighted score
    # ------------------------------------------------------------------
    # Round AA - ONE lock-weight table (backend/core/lock_weights.py): base
    # prior x ledger reliability x availability for every voter.
    table = lock_weights.build(settings, ledger_sources)

    contributions: dict[str, dict] = {}
    active: dict[str, float] = {}

    # The Drosophila brain is always available: it is computed synchronously.
    active["drosophila"] = table.mark("drosophila")
    contributions["drosophila"] = {
        "decision": BUY if ccs_value >= 0 else SELL,
        "confidence": round(ccs_confidence, 4),
        "value": round(ccs_value, 4),
        "weight": round(active["drosophila"], 4),
        "status": "LIVE",
        # Where the brain's vote lands in the final number - the UI shows this
        # so "the fly is 40 % of the decision" is verifiable, not a claim.
        "weighted_value": round(active["drosophila"] * max(-1.0, min(1.0, ccs_value)), 4),
        "source": "80-node mushroom body, 3-layer graph convolution",
    }

    # Round V: the formulas' weighted consensus is a voter in its own right.
    # The user's standing rule is that the formulas outrank every soft input;
    # before this they reached the side only through the brain's read-out, so
    # a modelled physics vote at 0.10 could pick the side against twenty live
    # formulas that leaned the other way.
    formulas_weight = float(getattr(settings, "weight_formulas", 0.0) or 0.0)
    if formula_consensus is not None and consensus_voters >= 3 and formulas_weight > 0:
        f_value = max(-1.0, min(1.0, float(formula_consensus)))
        formulas_weight = table.mark("formulas")
        active["formulas"] = formulas_weight
        contributions["formulas"] = {
            "decision": BUY if f_value >= 0 else SELL,
            "confidence": round(min(1.0, abs(f_value)), 4),
            "value": round(f_value, 4),
            "weight": formulas_weight,
            "status": "LIVE",
            "weighted_value": round(formulas_weight * f_value, 4),
            "source": f"weighted consensus of {int(consensus_voters)} directional formulas",
        }

    for name in ("gemini", "local", "github"):
        result = agents.get(name)
        if result is not None and result.available:
            active[name] = table.mark(name)
            contributions[name] = {
                "decision": result.decision,
                "confidence": round(result.confidence or 0.0, 4),
                "value": round(_direction_value(result.decision) * (result.confidence or 0.0), 4),
                "weight": round(active[name], 4),
                "status": result.status.value,
                "latency_ms": round(result.latency_ms, 1),
                "model": result.model,
            }

    # Round T/AI: the physics layer (O-U / VPIN / Hawkes / momentum flux /
    # entropy / diffusion / pendulum / peg) - one more weighted voter, never a veto.
    physics = physics or {}
    physics_weight = float(getattr(settings, "weight_physics", 0.0) or 0.0)
    if physics and physics_weight > 0:
        p_value = max(-1.0, min(1.0, float(physics.get("vote") or 0.0)))
        # Round AJ: every active mechanism on a real tape is a live input; a
        # layer on the simulator counts half.
        liveness = min(1.0, int(physics.get("live_inputs") or 0) / 3.0)
        physics_weight = table.mark("physics", 0.5 + 0.5 * liveness)
        active["physics"] = physics_weight
        contributions["physics"] = {
            "decision": BUY if p_value >= 0 else SELL,
            "confidence": round(float(physics.get("confidence") or 0.0), 4),
            "value": round(p_value, 4),
            "weight": physics_weight,
            "status": "LIVE" if int(physics.get("live_inputs") or 0) > 0 else "MODEL",
            "weighted_value": round(physics_weight * p_value, 4),
            "source": "physics layer: O-U·VPIN·Hawkes·momentum flux·entropy·diffusion·pendulum·temperature (Kelly, cost-gated)",
            "w_final": (physics.get("weights") or {}).get("w_final"),
        }

    # Round AL: the Formula Genesis Engine's composite - the fitness-weighted
    # vote of the regime-gated active set (of 200, of 2,100+).  Weight scales
    # with how many gated formulas actually fired; while the pool is still
    # warming up it does not vote at all (never an imputed 0).
    genesis = genesis or {}
    genesis_weight = float(getattr(settings, "weight_genesis", 0.0) or 0.0)
    if genesis.get("status") == "live" and genesis_weight > 0 and int(genesis.get("firing") or 0) >= 5:
        g_value = max(-1.0, min(1.0, float(genesis.get("vote") or 0.0)))
        firing_share = min(1.0, int(genesis.get("firing") or 0) / max(1, int(genesis.get("gated") or 1)))
        genesis_weight = table.mark("genesis", 0.5 + 0.5 * firing_share)
        active["genesis"] = genesis_weight
        regime = (genesis.get("regime") or {}).get("regime", "")
        contributions["genesis"] = {
            "decision": BUY if g_value >= 0 else SELL,
            "confidence": round(float(genesis.get("confidence") or 0.0), 4),
            "value": round(g_value, 4),
            "weight": genesis_weight,
            "status": "LIVE",
            "weighted_value": round(genesis_weight * g_value, 4),
            "source": f"genesis engine: {genesis.get('firing')}/{genesis.get('gated')} gated formulas of "
                      f"{genesis.get('active')} active firing in {regime} · generation {genesis.get('generation')}",
            "agreement": genesis.get("agreement"),
            "regime": regime,
        }

    # Round Z: the news wire votes - signed impact on THIS asset (a war is
    # bearish BTC and bullish PAXG), weight scaled by how much fresh,
    # classified news there is (no news -> no vote, never an imputed 0).
    news_impact = news_impact or {}
    news_weight = float(getattr(settings, "weight_news", 0.0) or 0.0)
    n_value = float(news_impact.get(asset.upper()) or 0.0)
    n_mass = float(news_impact.get("weight") or 0.0)
    if news_weight > 0 and n_mass > 0 and abs(n_value) > 1e-6:
        news_weight = table.mark("news", min(1.0, n_mass / 1.5))
        active["news"] = news_weight
        top = (news_impact.get("drivers") or {}).get(asset.upper()) or []
        contributions["news"] = {
            "decision": BUY if n_value >= 0 else SELL,
            "confidence": round(min(1.0, abs(n_value)), 4),
            "value": round(n_value, 4),
            "weight": news_weight,
            "status": "LIVE",
            "weighted_value": round(news_weight * n_value, 4),
            "source": "news impact: " + "; ".join(f"{d['theme']} ({d['impact']:+.2f})" for d in top[:2]),
            "drivers": top[:3],
        }

    total_weight = sum(active.values()) or 1e-9
    weights_used = {name: weight / total_weight for name, weight in active.items()}

    score = 0.0
    for name, weight in active.items():
        if name == "drosophila":
            score += weight * max(-1.0, min(1.0, ccs_value))
        elif name in ("physics", "formulas", "news", "genesis"):
            score += weight * float(contributions[name]["value"])
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
    if "formulas" in contributions:
        # A coherent formula consensus is earned confidence too.
        brain_term = 0.5 * brain_term + 0.5 * float(contributions["formulas"]["confidence"])

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

    # --- crowd emotion -----------------------------------------------
    # The user's premise is the whole reason this term exists: a 60-second
    # window is easy to push when the crowd is emotional.  The manipulation
    # score therefore scales the confidence down - it never touches the side.
    crowd_adjustment = 1.0
    crowd_note = ""
    crowd = crowd or {}
    manipulation = float((crowd.get("manipulation") or {}).get("score") or 0.0)
    threshold = float(getattr(settings, "emotion_dampen_threshold", 0.45))
    if manipulation >= threshold:
        span = max(1e-6, 1.0 - threshold)
        excess = min(1.0, (manipulation - threshold) / span)
        crowd_adjustment = 1.0 - float(getattr(settings, "emotion_dampen_max", 0.125)) * excess
        dominant = (crowd.get("dominant") or {}).get("label") or "an emotional crowd"
        kind = (crowd.get("manipulation") or {}).get("kind") or "crowding"
        crowd_note = (
            f"{dominant.lower()} tape ({manipulation:.0%} {kind}) - "
            f"confidence x{crowd_adjustment:.2f}"
        )
    raw_confidence *= crowd_adjustment

    confidence = raw_confidence * hsi_adjustment
    confidence = max(0.0, min(0.95, confidence))

    # --- learned evidence (Round N) -------------------------------------
    # Once the evidence ledger has scored enough windows it knows which of
    # the inputs above actually predict the next minute.  Its verdict then
    # decides the side, and its calibrated probability bounds the confidence:
    # the engine may not claim more certainty than its own record supports.
    spec_score = score
    learned_note = ""
    learned = learned or {}
    # Round Y - ONE logic, not two recipes with a hand-over: the prediction
    # history re-weights every source continuously.  ``lam`` is how much of
    # the decision the record has earned: 0 with no scored windows, 1 once
    # ``min_samples`` windows are scored (and 0 again if the ledger's own hit
    # rate falls below the spec recipe's - it never gets to be worse).
    scored = int(learned.get("scored") or 0)
    min_samples = max(1, int(learned.get("min_samples") or 30))
    learned_side_ok = learned.get("side") in (BUY, SELL) and bool(learned.get("enabled", True))
    lam = 0.0
    if learned_side_ok and not learned.get("handed_back") and int(learned.get("voters", 3) or 0) >= 3:
        lam = max(0.0, min(1.0, scored / float(min_samples)))
    if lam > 0:
        learned_score = float(learned.get("score") or 0.0)
        score = (1.0 - lam) * spec_score + lam * learned_score
        p_side = float(learned.get("p_side") or 0.5)
        realised = learned.get("realised_at_this_confidence")
        # The printed confidence is the earned probability of this side
        # (already replaced by the realised hit rate of its bucket once the
        # bucket has a record) - neither the spec recipe's optimism nor its
        # pessimism about inputs the ledger has shown to be noise.
        earned = float(realised) if realised is not None else p_side
        # ONE confidence scale everywhere: confidence is the EDGE of the call,
        # 2*P(side)-1 (0 = coin flip, 0.95 = near-certain).  The ledger speaks
        # in probabilities, so its earned P(side) is converted before blending.
        earned_edge = max(0.0, min(0.95, 2.0 * earned - 1.0))
        confidence = (1.0 - lam) * confidence + lam * earned_edge
        flipped = (spec_score > 0) != (score > 0) and spec_score != 0
        top = ", ".join(f"{r['source']} {r['reliability']:.0%}" for r in (learned.get("for") or [])[:3])
        learned_note = (
            f"prediction history weighs {lam:.0%} of this call ({scored} of {min_samples} windows scored): "
            f"P({learned['side']}) {p_side:.0%}"
            + (f", realised {float(realised):.0%} at this confidence" if realised is not None else "")
            + (f"; it overrides the spec recipe ({spec_score:+.2f})" if flipped else "")
            + (f"; most reliable: {top}" if top else "")
        )

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
        crowd_note=crowd_note,
        learned_note=learned_note,
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
        crowd_adjustment=round(crowd_adjustment, 4),
        crowd_note=crowd_note,
        learned=learned if learned.get("active") else {"active": False, "scored": learned.get("scored", 0),
                                                       "min_samples": learned.get("min_samples")},
        spec_score=spec_score,
        physics=physics,
        genesis=genesis,
        lock_weights=table.as_dict(),
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
    crowd_note: str = "",
    learned_note: str = "",
) -> str:
    parts: list[str] = []
    if learned_note:
        parts.append(learned_note)

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
    physics = contributions.get("physics")
    if physics:
        parts.append(
            f"physics layer {physics['decision']} ({physics['value']:+.2f}, "
            f"Kelly BTC weight {float(physics.get('w_final') or 0.5):.0%})"
        )
    genesis = contributions.get("genesis")
    if genesis:
        parts.append(
            f"genesis engine {genesis['decision']} ({genesis['value']:+.2f}, "
            f"{float(genesis.get('agreement') or 0.0):.0%} of firing formulas agree, regime {genesis.get('regime')})"
        )
    news = contributions.get("news")
    if news:
        parts.append(
            f"news impact {news['decision']} ({news['value']:+.2f} on this asset: "
            + "; ".join(f"{d['theme']}" for d in (news.get("drivers") or [])[:2]) + ")"
        )
    if consensus_note:
        parts.append(consensus_note)
    if accuracy_note:
        parts.append(f"confidence dampened on {accuracy_note}")
    if crowd_note:
        parts.append(f"crowd: {crowd_note}")
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
