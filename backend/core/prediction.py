"""The prediction block: side, levels, freshness and the reasoning behind it.

Every payload the dashboard and the Flutter client render carries a
``prediction`` object.  It exists because a number on its own ("SELL") is not
actionable: the user asked for three things and this module is all three:

1.  **Freshness.**  ``age_seconds`` / ``stale`` / ``expires_at`` say exactly how
    old the prediction is.  A prediction may never be older than
    ``PREDICTION_MAX_AGE_SECONDS`` (15 s by default): the engine recomputes it
    every window, and the API refuses to serve a stale one without saying so.
2.  **Reasoning.**  ``reasoning.summary`` plus a list of bullets - which
    formulas back the side, which argue against it, what the brain read out,
    how the hedge and the news look, where the levels come from.
3.  **Risk.**  The 1:1 take-profit / stop-loss pair, in price and in bps.

The module is deliberately pure: it takes plain dictionaries and returns plain
dictionaries, so it can be unit-tested without an engine.
"""

from __future__ import annotations

import time

#: How a prediction is labelled once its age passes the contract.
LIVE = "LIVE"
STALE = "STALE"

#: Category weights for the formula consensus.  Microstructure and order-book
#: formulas see the tape itself, so they lead; news and the brain read-out carry
#: less on their own because both are already part of the fusion score.
CATEGORY_WEIGHTS = {
    "A": 1.0,
    "B": 1.0,
    "C": 0.9,
    "D": 0.8,
    "E": 0.8,
    "F": 0.9,
    "G": 0.7,
    "H": 1.0,
}


def _iso(timestamp: float) -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(timestamp))


def consensus(formula_values: dict, directional: dict[str, str] | None = None) -> dict:
    """Weighted agreement of the directional formulas with each other.

    Returns ``score`` (in [-1, 1]), the number of formulas that voted, how many
    of them lean each way, and the names of the strongest supporters of each
    side - everything the reasoning bullets need.
    """
    directional = directional or {}
    total = 0.0
    weighted = 0.0
    up: list[tuple[float, str]] = []
    down: list[tuple[float, str]] = []
    for name, raw in (formula_values or {}).items():
        category = directional.get(name)
        if not category:
            continue
        try:
            value = float(raw)
        except (TypeError, ValueError):
            continue
        if value == 0.0:
            # A formula that produced nothing this window (or is exactly
            # neutral) is not evidence; scoring it as a zero vote would dilute
            # the consensus toward the noise floor.
            continue
        weight = CATEGORY_WEIGHTS.get(category, 0.8)
        total += weight
        weighted += weight * max(-1.0, min(1.0, value))
        (up if value > 0 else down).append((abs(value), name))
    if total <= 0:
        return {
            "score": 0.0,
            "voters": 0,
            "up": 0,
            "down": 0,
            "up_names": [],
            "down_names": [],
        }
    up.sort(reverse=True)
    down.sort(reverse=True)
    return {
        "score": weighted / total,
        "voters": len(up) + len(down),
        "up": len(up),
        "down": len(down),
        "up_names": [name for _, name in up[:4]],
        "down_names": [name for _, name in down[:4]],
    }


def freshness(computed_wall: float, max_age: float, now: float | None = None) -> dict:
    """Age of the prediction against the staleness contract."""
    now = time.time() if now is None else now
    computed = float(computed_wall or 0.0)
    if computed <= 0:
        return {
            "state": STALE,
            "computed_at": "",
            "age_seconds": None,
            "max_age_seconds": round(float(max_age), 1),
            "seconds_until_stale": 0.0,
            "expires_at": "",
            "on_time": False,
        }
    age = max(0.0, now - computed)
    remaining = max(0.0, float(max_age) - age)
    return {
        "state": LIVE if age <= float(max_age) else STALE,
        "computed_at": _iso(computed),
        "age_seconds": round(age, 1),
        "max_age_seconds": round(float(max_age), 1),
        "seconds_until_stale": round(remaining, 1),
        "expires_at": _iso(computed + float(max_age)),
        "on_time": age <= float(max_age),
    }


def _fmt(value: float, digits: int = 2) -> str:
    try:
        return f"{float(value):+.{digits}f}"
    except (TypeError, ValueError):
        return "—"


def build_reasoning(
    side: str,
    confidence: float,
    conviction: str,
    fusion: dict | None = None,
    formula_values: dict | None = None,
    directional: dict[str, str] | None = None,
    hedge: dict | None = None,
    news: dict | None = None,
    risk: dict | None = None,
    brain: dict | None = None,
    accuracy: dict | None = None,
    window_seconds: float = 15.0,
    extra_against: list[str] | None = None,
) -> dict:
    """Compose the plain-English case for (and against) the side.

    The bullets are ordered the way the decision was actually made: the brain
    read-out, the formula consensus, the contributors on each side, the hedge,
    the news, the risk levels, then the measured accuracy of recent windows.
    """
    fusion = fusion or {}
    formula_values = formula_values or {}
    hedge = hedge or {}
    news = news or {}
    risk = risk or {}
    brain = brain or {}
    accuracy = accuracy or {}

    agreement = consensus(formula_values, directional)
    score = agreement["score"]
    supporters = agreement["up_names"] if side == "BUY" else agreement["down_names"]
    dissenters = agreement["down_names"] if side == "BUY" else agreement["up_names"]

    bullets: list[dict] = []

    # 1. The brain -----------------------------------------------------------------
    ccs = fusion.get("contributions", {}).get("drosophila", {}).get("value")
    if ccs is None:
        ccs = brain.get("ccs")
    if ccs is not None:
        weight = float(fusion.get("weights_used", {}).get("drosophila", 0.40))
        lean = "up" if float(ccs) >= 0 else "down"
        solo = weight >= 0.99
        bullets.append({
            "kind": "brain",
            "weight": round(weight, 2),
            "supports": (lean == "up") == (side == "BUY"),
            "text": (
                f"Drosophila read-out: CCSv2 {_fmt(ccs)} — the mushroom-body output "
                f"leans {lean}"
                + (
                    " and was the whole fusion this window (no AI agent answered)."
                    if solo
                    else f" and carries {weight:.0%} of the fusion weight."
                )
            ),
        })

    # 2. Formula consensus ---------------------------------------------------------
    if agreement["voters"]:
        same = agreement["up"] if side == "BUY" else agreement["down"]
        other = agreement["down"] if side == "BUY" else agreement["up"]
        bullets.append({
            "kind": "formulas",
            "weight": None,
            "supports": same >= other,
            "text": (
                f"Formula consensus {_fmt(score)}: {same} of {agreement['voters']} "
                f"directional formulas lean {side} and {other} lean the other way."
            ),
        })
    if supporters:
        named = ", ".join(f"{name} {_fmt(formula_values.get(name, 0.0))}" for name in supporters)
        bullets.append({
            "kind": "formula-support",
            "weight": None,
            "supports": True,
            "text": f"Backing the {side}: {named}.",
        })
    if dissenters:
        named = ", ".join(f"{name} {_fmt(formula_values.get(name, 0.0))}" for name in dissenters)
        bullets.append({
            "kind": "formula-dissent",
            "weight": None,
            "supports": False,
            "text": f"Arguing against: {named}.",
        })

    # 3. Agents --------------------------------------------------------------------
    contributions = fusion.get("contributions", {})
    agents = [
        f"{name} {contributions[name].get('decision')} "
        f"({float(contributions[name].get('confidence') or 0.0):.0%})"
        for name in ("gemini", "local", "github")
        if name in contributions
    ]
    if agents:
        bullets.append({
            "kind": "agents",
            "weight": None,
            "supports": True,
            "text": "AI agents: " + ", ".join(agents) + ".",
        })
    else:
        bullets.append({
            "kind": "agents",
            "weight": None,
            "supports": True,
            "text": "No AI agent answered this window — the fly's read-out and the "
                    "formula consensus carried the decision alone.",
        })

    # 4. Hedge ---------------------------------------------------------------------
    hsi = hedge.get("hsi")
    if hsi is not None:
        try:
            hsi_value = float(hsi)
            working = hsi_value < 0.80
            bullets.append({
                "kind": "hedge",
                "weight": None,
                "supports": working,
                "text": (
                    f"Hedge stress (HSI) {hsi_value:.2f} — the BTC/PAXG hedge is "
                    f"{'working' if working else 'breaking down'}"
                    + ("" if working else "; position size is dampened for it") + "."
                ),
            })
        except (TypeError, ValueError):
            pass
    shrp = formula_values.get("SHRP")
    if shrp is not None:
        try:
            shrp_value = float(shrp)
            if abs(shrp_value) < 0.05:
                # A reading this close to zero is "no rotation to report"; it
                # must not be dressed up as risk-on or risk-off.
                raise ValueError
            bullets.append({
                "kind": "rotation",
                "weight": None,
                "supports": (shrp_value > 0) == (side == "BUY"),
                "text": (
                    f"Safe-haven rotation (SHRP) {_fmt(shrp_value)} — flow is "
                    f"{'leaving gold for BTC (risk-on)' if shrp_value > 0 else 'leaving BTC for gold (risk-off)'}."
                ),
            })
        except (TypeError, ValueError):
            pass

    # 5. News ----------------------------------------------------------------------
    headline = news.get("headline") or news.get("top_headline")
    sentiment = news.get("sentiment", news.get("smd"))
    if headline:
        leans = "supports" if (float(sentiment or 0.0) > 0) == (side == "BUY") else "cuts against"
        bullets.append({
            "kind": "news",
            "weight": None,
            "supports": leans == "supports",
            "text": (
                f"News {_fmt(float(sentiment or 0.0))} — “{str(headline)[:110]}” "
                f"{leans} the {side} side."
            ),
        })
    elif sentiment is not None:
        bullets.append({
            "kind": "news",
            "weight": None,
            "supports": True,
            "text": f"News sentiment {_fmt(float(sentiment or 0.0))} — no headline moved the tape.",
        })

    # 6. Levels --------------------------------------------------------------------
    if risk.get("tradeable") and risk.get("take_profit"):
        bullets.append({
            "kind": "levels",
            "weight": None,
            "supports": True,
            "text": (
                f"Levels: entry {risk.get('entry')}, target {risk.get('take_profit')} and stop "
                f"{risk.get('stop_loss')} — {float(risk.get('tp_bps') or 0.0):.0f} bps each way, "
                f"a 1:1 reward:risk on {float(risk.get('volatility_bps') or 0.0):.1f} bps realised "
                f"volatility."
            ),
        })

    # 7. Measured accuracy ---------------------------------------------------------
    if accuracy.get("evaluated"):
        bullets.append({
            "kind": "accuracy",
            "weight": None,
            "supports": float(accuracy.get("win_rate") or 0.0) >= 0.5,
            "text": (
                f"Recent form: {float(accuracy.get('win_rate') or 0.0):.0%} of the last "
                f"{accuracy.get('evaluated')} evaluated windows closed in the predicted direction."
            ),
        })

    for text in extra_against or []:
        bullets.append({"kind": "caution", "weight": None, "supports": False, "text": text})

    supports = [b["text"] for b in bullets if b["supports"]]
    against = [b["text"] for b in bullets if not b["supports"]]

    majority = ""
    if agreement["voters"]:
        majority = (
            f"; {max(agreement['up'], agreement['down'])} of {agreement['voters']} directional "
            f"formulas agree"
        )
    summary = (
        f"{side} at {conviction.lower()} conviction ({confidence:.0%})"
        f"{majority}."
    )
    if against:
        summary += (
            f" {len(against)} counterpoint{'s' if len(against) != 1 else ''} below."
        )
    summary += f" Levels are 1:1 on realised {float(risk.get('volatility_bps') or 0.0):.0f} bps volatility."

    return {
        "summary": summary,
        "bullets": bullets,
        "supports": supports,
        "against": against,
        "consensus": agreement,
        "generated_at": _iso(time.time()),
        "window_seconds": round(float(window_seconds), 1),
    }


def build(
    *,
    side: str,
    confidence: float,
    conviction: str,
    computed_wall: float,
    max_age: float,
    risk: dict,
    reasoning: dict,
    accuracy: dict,
    window_seconds: float,
    weak: bool = False,
    emergency: bool = False,
    now: float | None = None,
) -> dict:
    """The complete prediction object embedded in every payload."""
    age = freshness(computed_wall, max_age, now=now)
    return {
        "side": side,
        "confidence": round(float(confidence or 0.0), 4),
        "conviction": conviction,
        "weak": bool(weak),
        "emergency": bool(emergency),
        "entry": risk.get("entry"),
        "take_profit": risk.get("take_profit"),
        "stop_loss": risk.get("stop_loss"),
        "tp_bps": risk.get("tp_bps"),
        "sl_bps": risk.get("sl_bps"),
        "rr": risk.get("rr"),
        "rr_target": risk.get("rr_target", 1.0),
        "volatility_bps": risk.get("volatility_bps"),
        "horizon_seconds": risk.get("horizon_seconds", window_seconds),
        "reasoning": reasoning,
        "accuracy": accuracy,
        **age,
    }
