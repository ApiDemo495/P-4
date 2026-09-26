"""How the Drosophila circuit is wired into the trading decision.

This module answers, in plain language and in one payload, the question
*"where exactly is the fly brain used?"*  It is the data source behind the
"Brain" panel in the dashboard, the `Brain` card in the Flutter client and the
`/api/brain/explain` endpoint.

The chain, once per window:

    20 formulas ──▶ 20 projection neurons (0-19)
                        │  signed, excitatory + inhibitory
                        ▼
                   50 Kenyon Cell clusters (20-69)      ReLU + top-10 % sparsity
                        │  modulated by the dopamine gates
                        ▼
    DRG ──▶ PAM / PPL1 (70, 71)     HSI ──▶ OA (72)
                        │
                        ▼
        MBONs (73-76): α3 approach · γ5β'2a avoid · β2β'2a neutral · α'2 confidence
                        │  3-layer graph convolution, gain 3.2802
                        ▼
        Lateral horn (77-79) ──▶ CCSv2 = tanh(LH_approach − LH_avoid)     [formula 22]
                             └─▶ KCAE  = 1 − H_KC / ln(50)                [formula 21]
                        │
                        ▼
        fusion: drosophila weight 0.40 of the final score (Section 8.1)
"""

from __future__ import annotations

from backend.brain import graph_convolution as gc
from backend.formulas.engine import ALL_FORMULAS

#: The five stages of the circuit, as the UI narrates them.
STAGES = (
    {
        "key": "pn",
        "name": "Projection neurons (0-19)",
        "role": "One neuron per formula. Each formula's value is written straight "
                "onto its PN, scaled to [-1, +1].",
        "detail": "ORN / thermosensory / mechanosensory analogues of the 20 signals.",
    },
    {
        "key": "kc",
        "name": "Kenyon Cell clusters (20-69)",
        "role": "The sparse code. ReLU on the PN→KC fan-in, then only the strongest "
                "10 % of the 50 clusters survive.",
        "detail": "50 spectral clusters of ~2000 real Kenyon Cells (hemibrain v1.2.1).",
    },
    {
        "key": "dan",
        "name": "Dopamine + octopamine gates (70-72)",
        "role": "PAM is driven by the DRG reward learner, PPL1 by its mirror image, "
                "OA by the Hedge Stress Index.",
        "detail": "This is how learning and hedge stress reach the circuit.",
    },
    {
        "key": "mbon",
        "name": "MBON outputs (73-76)",
        "role": "Four behavioural read-outs: α3 approach, γ5β'2a avoid, β2β'2a "
                "neutral, α'2 confidence.",
        "detail": "Three graph-convolution layers with gain 3.2802.",
    },
    {
        "key": "lh",
        "name": "Lateral horn (77-79)",
        "role": "CCSv2 = tanh(LH_approach − LH_avoid). KCAE measures the sparsity of "
                "the Kenyon Cell code and becomes the brain's confidence.",
        "detail": "Formula 22 (CCSv2) and Formula 21 (KCAE).",
    },
)


def wiring() -> dict:
    """The static map: which formula drives which brain structure."""
    pn_rows: list[dict] = []
    for spec in ALL_FORMULAS:
        if spec.category == "H":
            continue
        pn_rows.append(
            {
                "pn": spec.order - 1,
                "formula": spec.name,
                "title": spec.title,
                "category": spec.category,
                "brain_node": spec.brain_node,
                "directional": spec.directional,
                "latency_ms": spec.latency_ms,
            }
        )
    return {
        "stages": list(STAGES),
        "projection_neurons": pn_rows,
        "layout": {
            "pn": [0, 19],
            "kenyon_cells": [gc.KC_START, gc.KC_END - 1],
            "dans": [gc.PAM, gc.OA],
            "mbons": [gc.MBON_APPROACH, gc.MBON_CONFIDENCE],
            "lateral_horn": [gc.LH_APPROACH, gc.LH_NEUTRAL],
        },
        "sparsity": gc.SPARSITY,
        "fusion_weight": 0.40,
        "outputs": {
            "CCSv2": "formula 22 - the directional brain signal, weight 0.40 in fusion",
            "KCAE": "formula 21 - sparsity of the KC code, the brain's confidence",
        },
    }


def explain(manager) -> dict:
    """The story of the brain for the window that is currently locked."""
    result = manager.last_formula_result
    if result is None or not result.brain_trace:
        return {
            "available": False,
            "detail": "no completed cycle yet - the first window is still being computed",
        }

    trace = dict(result.brain_trace)
    signal = manager.lock.try_get_current()
    fusion = manager.last_fusion or {}
    contribution = (fusion.get("contributions") or {}).get("drosophila") or {}

    values = result.values
    dominant = trace.get("dominant_pns") or []
    for row in dominant:
        row["category"] = next(
            (spec.category for spec in ALL_FORMULAS if spec.name == row.get("formula")),
            "",
        )

    lh = trace.get("lateral_horn") or {}
    mbons = trace.get("mbons") or {}
    kcs = trace.get("kenyon_cells") or {}
    dan = trace.get("dan") or {}

    lean = (
        "BUY (approach)"
        if (lh.get("approach", 0) or 0) > (lh.get("avoid", 0) or 0)
        else "SELL (avoid)"
    )
    verdict = (
        f"LH approach {lh.get('approach', 0):+.3f} vs avoid {lh.get('avoid', 0):+.3f} "
        f"→ {lean}; the sparse KC code was {kcs.get('active', 0)}/{kcs.get('of', 50)} "
        f"clusters (KCAE {trace.get('kcae', 0):.2f}) → CCSv2 "
        f"{trace.get('ccs', 0):+.3f} at {trace.get('confidence', 0) * 100:.0f}% confidence."
    )

    status = manager.brain.status_dict()
    return {
        "available": True,
        "cycle_number": result.values.get("_cycle", manager.stats.cycle_number),
        "asset": manager.asset,
        "computed_at": getattr(signal, "computed_at", "") if signal else "",
        "valid_from": getattr(signal, "valid_from", "") if signal else "",
        "valid_until": getattr(signal, "valid_until", "") if signal else "",
        "signal": signal.signal if signal else None,
        "ccs_value": round(float(values.get("CCSv2", 0.0)), 4),
        "ccs_confidence": round(float(values.get("KCAE", 0.0)), 4),
        "stage": trace,
        "dominant_inputs": dominant,
        "all_inputs": trace.get("pn") or {},
        "kenyon_cells": kcs,
        "mbons": mbons,
        "lateral_horn": lh,
        "dopamine": dan,
        "weights": {
            "in_fusion": 0.40,
            "applied": contribution.get("weight", round(contribution.get("weight", 0.0), 4)),
            "value": contribution.get("value"),
            "decision": contribution.get("decision"),
            "share_of_score": (
                round(abs(float(contribution.get("value") or 0.0)) *
                      float((fusion.get("weights_used") or {}).get("drosophila", 0.0)), 4)
                if contribution
                else None
            ),
        },
        "verdict": verdict,
        "source": {
            "status": status.get("status"),
            "message": status.get("message"),
            "dataset": status.get("dataset"),
            "checksum": (status.get("matrix") or {}).get("checksum"),
            "gain": status.get("gain"),
            "is_live": status.get("is_live"),
            "steps": status.get("steps") or [],
        },
    }
