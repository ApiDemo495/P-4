"""FORMULA 22 - Connectome Consensus Score v2 (CCSv2).

The master aggregation.  All 20 directional/contextual formula outputs are
mapped onto the 20 projection neurons of the mushroom body, propagated through
three graph-convolution layers over the 80-node circuit, and read out at the
lateral horn:

    CCSv2      = tanh(LH_approach - LH_avoid)
    confidence = KCAE * (1 - |LH_neutral| / (|appr| + |avoid| + |neut| + eps))

The circuit is 80 nodes (20 PN + 50 KC clusters + 3 DAN + 4 MBON + 3 LH) and the
propagation is the one defined in ``backend.brain.graph_convolution``.

Because KCAE is *derived from* the Kenyon Cell activations produced by this very
formula, the engine calls :func:`prepare` first and then :func:`compute` - see
``backend/formulas/engine.py``.  ``prepare`` is idempotent for a given cycle.

Brain mapping: this IS the brain computation - every formula converges here.

Latency budget: < 0.8 ms.
"""

from __future__ import annotations

import numpy as np

from backend.brain import graph_convolution as gc
from backend.formulas._util import EPS, finite
from backend.formulas._util import trace as trace_row
from backend.formulas.category_h_brain import kcae as kcae_formula

NAME = "CCSv2"
CATEGORY = "H"
TITLE = "Connectome Consensus Score v2"
BRAIN_NODE = "Whole mushroom body circuit"
DIRECTIONAL = True
LATENCY_MS = 0.8
DESCRIPTION = "3-layer graph convolution over the 80-node mushroom body, read at the lateral horn."

FORMULA_ORDER = gc.PN_NAMES


class State:
    __slots__ = ("trace", "confidence", "kcae", "ready", "drg", "hsi")

    def __init__(self) -> None:
        self.trace: gc.ActivationTrace | None = None
        self.confidence = 0.0
        self.kcae = 0.0
        self.ready = False
        self.drg = 0.0
        self.hsi = 0.0

    def to_dict(self) -> dict:
        return {"confidence": self.confidence, "kcae": self.kcae, "drg": self.drg, "hsi": self.hsi}

    @classmethod
    def from_dict(cls, payload: dict) -> "State":
        obj = cls()
        obj.confidence = float(payload.get("confidence", 0.0))
        obj.kcae = float(payload.get("kcae", 0.0))
        obj.drg = float(payload.get("drg", 0.0))
        obj.hsi = float(payload.get("hsi", 0.0))
        return obj


#: Formulas whose output has no side (``DIRECTIONAL = False``): HSI scales
#: conviction and already enters the circuit through the octopamine node, ERC
#: describes the regime.  Feeding their (always positive) magnitudes into the
#: projection neurons read as a permanent *bullish* drive - on a flat tape the
#: circuit voted +0.42 and the fused signal repeated BUY (Deviation 22-C).
NON_DIRECTIONAL_PN = frozenset({"HSI", "ERC"})


def _vector_from_ctx(ctx: dict) -> np.ndarray:
    return np.asarray(
        [0.0 if name in NON_DIRECTIONAL_PN else float(ctx.get(name, 0.0)) for name in FORMULA_ORDER],
        dtype=np.float64,
    )


def prepare(snapshot, asset: str, state: State, params: dict, ctx: dict) -> gc.ActivationTrace | None:
    """Run the graph convolution and hand the KC activations to Formula 21."""
    brain = ctx.get("_brain")
    conv = getattr(brain, "conv", None)
    drg = float(ctx.get("_drg", 0.0))
    hsi = float(ctx.get("HSI", 0.0))
    state.drg, state.hsi = drg, hsi
    vector = _vector_from_ctx(ctx)

    # --- Round AQ: the real mushroom body ---------------------------------
    # When the FlyWire v783 connectome is loaded, the pass runs through the
    # 8 353-neuron mushroom-body / lateral-horn circuit (every synapse real;
    # push-pull and dopamine gating inside ``whole_brain.ConnectomeGraph``).
    # The 80x80 convolution below is only the fallback until it is.
    mb_pass = getattr(brain, "mushroom_body_pass", None)
    if callable(mb_pass):
        readout = mb_pass(vector, drg, hsi)
        if readout is not None:
            trace = brain.mb.to_activation_trace(readout)
            trace.diagnostics["balance"] = float(readout.balance)
            # KCAE on the real Kenyon-cell *code*: the rectified, APL-sparsened
            # activity over all 5 177 cells.  With ON/OFF glomerular channels a
            # bearish reading excites its own KCs, so - unlike the 80-node
            # stand-in - the rectified code is never empty for a SELL and the
            # drive work-around is not needed.  1 - H/log N is the sparseness
            # of the code: ~0.3 for a ~5 % ensemble, 0 on a silent tape.
            ensemble_drive = np.asarray(trace.kc_activations, dtype=np.float64)
            kcae_value = kcae_formula.from_activations(ensemble_drive) if ensemble_drive.size else 0.0
            trace.kcae = kcae_value
            confidence = kcae_value * float(readout.confidence)
            trace.confidence = confidence
            trace.diagnostics.update({
                "confidence_spec": confidence, "decisiveness": float(readout.confidence),
                "neutral_share": abs(readout.neutral) / (abs(readout.approach) + abs(readout.avoid) + abs(readout.neutral) + EPS),
                "resting_balance": 0.0, "balance_used": float(readout.balance),
            })
            trace.resting_balance = 0.0
            state.trace = trace
            state.kcae = kcae_value
            state.confidence = confidence
            state.ready = True
            ctx["_kc_activations"] = trace.kc_activations
            ctx["_kc_drive"] = ensemble_drive          # Formula 21 scores the ensemble that fired
            ctx["KCAE"] = kcae_value
            ctx["_ccsv2_confidence"] = confidence
            ctx["_brain_trace"] = trace.to_dict()
            return trace

    if conv is None:
        state.ready = False
        return None
    trace = conv.propagate(vector, drg=drg, hsi=hsi)

    # --- push-pull read-out (Deviation 22-D, SPEC_NOTES.md) --------------
    # The PN -> KC fan-in is rectified, so a *negative* formula value barely
    # reaches the Kenyon Cells: NIV = +0.7 read +0.38 at the lateral horn,
    # NIV = -0.7 read -0.02, and LCS = -0.75 changed nothing at all.  The
    # zero-vector baseline of Deviation 22-A could not repair that - it is a
    # bias in the *gain*, not in the offset - and the circuit voted BUY on
    # tapes whose formulas were mostly bearish.  The fly solves the same
    # problem with ON/OFF channels: the mirrored ensemble is propagated through
    # the same wiring and the two read-outs are combined antisymmetrically,
    #     approach = (approach(x) + avoid(-x)) / 2,   avoid = (avoid(x) + approach(-x)) / 2
    # which makes the score an odd function of the input (f(-x) = -f(x)),
    # removes the resting offset exactly, and leaves every bullish response
    # unchanged in magnitude.
    mirror = conv.propagate(-vector, drg=drg, hsi=hsi)
    approach = 0.5 * (float(trace.lh_approach) + float(mirror.lh_avoid))
    avoid = 0.5 * (float(trace.lh_avoid) + float(mirror.lh_approach))
    neutral = 0.5 * (float(trace.lh_neutral) + float(mirror.lh_neutral))
    trace.lh_approach, trace.lh_avoid, trace.lh_neutral = approach, avoid, neutral
    trace.active_kcs = max(int(trace.active_kcs), int(mirror.active_kcs))
    resting_balance = 0.0  # the odd read-out has no resting offset by construction
    trace.resting_balance = resting_balance  # surfaced through the brain trace

    # KCAE is measured on the Kenyon Cell *drive* (see the |a|^2 note below).
    kcae_value = kcae_formula.from_activations(trace.kc_drive)
    trace.kcae = kcae_value

    # --- the specification's literal confidence term --------------------
    denominator = (
        abs(trace.lh_approach) + abs(trace.lh_avoid) + abs(trace.lh_neutral) + EPS
    )
    confidence_spec = kcae_value * max(0.0, 1.0 - abs(trace.lh_neutral) / denominator)

    # --- operative confidence -------------------------------------------
    # `1 - |neutral| / sum` is structurally asymmetric in this circuit: the
    # Kenyon Cells drive the neutral MBON only on the rectified (bullish) side,
    # so the same ensemble conviction would be scored ~0.35 when bullish and
    # ~0.93 when bearish.  Confidence is therefore the *read-out margin* - how
    # far the approach and avoidance channels are from cancelling - which is
    # what the neutral term is trying to express and is sign-symmetric.
    margin = abs(trace.lh_approach - trace.lh_avoid) / (
        abs(trace.lh_approach) + abs(trace.lh_avoid) + EPS
    )
    confidence = kcae_value * margin
    trace.confidence = confidence
    trace.diagnostics = {
        "confidence_spec": confidence_spec,
        "decisiveness": margin,
        "neutral_share": abs(trace.lh_neutral) / denominator,
        "resting_balance": resting_balance,
        "balance_used": float(trace.lh_approach - trace.lh_avoid) - resting_balance,
    }

    state.trace = trace
    state.kcae = kcae_value
    state.confidence = confidence
    state.ready = True

    # Publish for Formula 21 (KCAE) and for the API layer.
    ctx["_kc_activations"] = trace.kc_activations
    ctx["_kc_drive"] = trace.kc_drive
    ctx["KCAE"] = kcae_value
    ctx["_ccsv2_confidence"] = confidence
    ctx["_brain_trace"] = trace.to_dict()
    return trace


def compute(snapshot, asset: str, state: State, params: dict, ctx: dict | None = None) -> float:
    ctx = ctx or {}
    if not state.ready or state.trace is None:
        if prepare(snapshot, asset, state, params, ctx) is None:
            return 0.0
    trace = state.trace
    if trace is None:
        return 0.0
    diag = trace.diagnostics or {}
    if "connectome" in diag:
        # real circuit: the odd MBON read-out is already in [-1, 1]
        balance = float(diag.get("balance", 0.0))
        score = balance
        trace_row(ctx, "connectome", str(diag.get("connectome")), f"{int(diag.get('neurons', 0)):,} neurons · {int(diag.get('connections', 0)):,} connections")
        trace_row(ctx, "approach MBON drive (ACh + GABA)", float(trace.lh_approach), "summed over hops, push-pull")
        trace_row(ctx, "avoid MBON drive (Glu)", float(trace.lh_avoid), "summed over hops, push-pull")
        trace_row(ctx, "active Kenyon cells", int(trace.active_kcs), "of the real KC population (APL-sparse)")
        trace_row(ctx, "hops · active neurons", " → ".join(str(x) for x in diag.get("hop_active", [])), "per hop")
        trace_row(ctx, "pass time", int(diag.get("elapsed_us", 0)), "µs")
        trace_row(ctx, "balance = tanh(2·(approach − avoid)/(|approach|+|avoid|))", balance, "-1..1")
    else:
        balance = float(trace.lh_approach - trace.lh_avoid) - float(trace.resting_balance)
        score = np.tanh(balance)
        trace_row(ctx, "LH approach drive", float(trace.lh_approach), "lateral horn")
        trace_row(ctx, "LH avoid drive", float(trace.lh_avoid), "lateral horn")
        trace_row(ctx, "resting balance", float(trace.resting_balance), "subtracted baseline")
        trace_row(ctx, "net balance", balance, "approach - avoid - resting")
    # The score is computed here rather than in ``prepare``, so publish it on the
    # trace as well: /api/brain/trace and the matrix viewer must show the same
    # CCSv2 value the signal was built from.
    trace.ccs = float(score)
    ctx["_brain_trace"] = trace.to_dict()
    return finite(score)


def confidence(state: State) -> float:
    return float(state.confidence)


DOUBLE_CHECK = ("value = tanh(2·(approach − avoid)/(|approach|+|avoid|)) over the real MBONs "
                "(fallback: tanh(LH approach - LH avoid - resting balance)), push-pull read-out")


def double_check(t: dict, asset: str) -> float:
    """Independent re-derivation of the output from the traced intermediates."""
    if "approach MBON drive (ACh + GABA)" in t and "avoid MBON drive (Glu)" in t:
        app = float(t["approach MBON drive (ACh + GABA)"])
        avo = float(t["avoid MBON drive (Glu)"])
        denom = abs(app) + abs(avo)
        return float(np.tanh(2.0 * (app - avo) / denom)) if denom > 1e-6 else 0.0
    if "net balance" not in t:
        return 0.0
    return float(np.tanh(float(t["net balance"])))
