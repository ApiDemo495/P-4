"""Generate the committed 80x80 fallback adjacency matrix.

The fallback file ``brain/fallback/mb_adjacency_80x80.csv`` must ALWAYS exist in
the repository (Section 6.2, step 5) so the app can never end up without a
mushroom-body circuit.  This script produces it deterministically from a fixed
seed using the wiring rules of the hemibrain MB:

  * PN -> KC      : sparse, excitatory, ~10 KCs per projection neuron
  * KC -> KC      : very sparse and *inhibitory* (the APL/GABAeric feedback)
  * KC -> MBON    : compartment-specific excitatory read-out
  * PN -> MBON    : the direct PN -> MBON-alpha3 pathway (see note below)
  * DAN -> MBON   : PAM excites approach, PPL1 excites avoidance
  * OA  -> MB/LH  : octopamine suppresses action and raises the HOLD tone
  * MBON -> LH    : excitatory to its own LH target, inhibitory to the opposite
  * LH  -> LH     : lateral inhibition inside the lateral horn

NOTE (Deviation 22-B)
---------------------
The specification normalises weights into [0, 1].  Real mushroom-body circuits
contain both excitatory and inhibitory projections, and CCSv2 cannot express
"bearish" without them: with a purely non-negative matrix and a ReLU on the
PN -> KC fan-in, an all-negative formula vector would produce zero Kenyon Cell
activity and the read-out would collapse to HSH (0 = HOLD) regardless of how
bearish the evidence was.  Weights are therefore signed and normalised by the
maximum **absolute** weight, which preserves the numerical intent of the
original rule.

NOTE (Deviation 22-A)
---------------------
The direct PN -> MBON-alpha3 pathway is included.  It exists in the real
hemibrain (MBON-alpha3 receives monosynaptic input from projection neurons) and
it is what carries the *sign* of the ensemble evidence into the read-out.

Run:
    python -m backend.brain.generate_fallback_matrix
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np

from backend.brain import graph_convolution as gc
from backend.brain import matrix_builder as mb

log = logging.getLogger("drosophila.brain.generate")

DEFAULT_SEED = 20241215

#: How many KCs each projection neuron contacts (hemibrain PNs diverge widely).
PN_KC_FANOUT = (6, 14)
KC_MBON_APPROACH_FANOUT = (18, 26)
KC_MBON_AVOID_FANOUT = (18, 26)
KC_MBON_NEUTRAL_FANOUT = (4, 8)
KC_KC_EDGES = 40

#: Formulas whose evidence should weigh most heavily on the approach pathway.
#: These are the hedge + microstructure signals that the spec calls out as the
#: primary drivers (TAI, AFPR, VSD, DGW, BAR, HRDD, SHRP, GCDV, NIV).
HIGH_WEIGHT_FORMULAS = {0, 1, 3, 4, 6, 7, 8, 9, 18}


def build_matrix(seed: int = DEFAULT_SEED) -> np.ndarray:
    rng = np.random.default_rng(seed)
    w = mb.empty_matrix()

    # ---------------- PN -> KC (excitatory, sparse) ------------------
    for pn in range(gc.N_FORMULAS):
        fanout = int(rng.integers(*PN_KC_FANOUT))
        targets = rng.choice(gc.N_KC, size=fanout, replace=False) + gc.KC_START
        weights = rng.lognormal(mean=-0.4, sigma=0.55, size=fanout)
        for kc, weight in zip(targets, weights):
            w[pn, kc] = float(min(weight, 3.0))

    # ---------------- KC -> KC (inhibitory, sparse) ------------------
    for _ in range(KC_KC_EDGES):
        src = int(rng.integers(gc.KC_START, gc.KC_END))
        dst = int(rng.integers(gc.KC_START, gc.KC_END))
        if src == dst:
            continue
        w[src, dst] -= float(rng.uniform(0.05, 0.35))

    # ---------------- KC -> MBON (excitatory compartments) -----------
    def wire_kc_to_mbon(target: int, fanout_range: tuple[int, int], scale: float) -> None:
        fanout = int(rng.integers(*fanout_range))
        sources = rng.choice(gc.N_KC, size=fanout, replace=False) + gc.KC_START
        for kc in sources:
            w[kc, target] += float(rng.lognormal(mean=-0.5, sigma=0.5)) * scale

    wire_kc_to_mbon(gc.MBON_APPROACH, KC_MBON_APPROACH_FANOUT, 1.0)
    wire_kc_to_mbon(gc.MBON_AVOID, KC_MBON_AVOID_FANOUT, 1.0)
    wire_kc_to_mbon(gc.MBON_NEUTRAL, KC_MBON_NEUTRAL_FANOUT, 0.55)
    wire_kc_to_mbon(gc.MBON_CONFIDENCE, (14, 22), 0.6)

    # ---------------- PN -> MBON (direct, SIGNED) --------------------
    # Approach gets +w, avoidance gets -w so a negative formula value excites
    # the avoidance pathway.  This is the signed evidence channel.
    for pn in range(gc.N_FORMULAS):
        base = float(rng.uniform(0.35, 0.75))
        if pn in HIGH_WEIGHT_FORMULAS:
            base *= 1.7
        # HSI / ERC / KCAE are non-directional: they must not drive direction.
        if pn in (10, 13):
            base *= 0.15
        w[pn, gc.MBON_APPROACH] += base
        w[pn, gc.MBON_AVOID] -= base
        # Mild uncertainty tone on the neutral MBON for the regime indicators.
        if pn in (10, 11, 12, 13):
            w[pn, gc.MBON_NEUTRAL] += base * 0.9

    # ---------------- DAN / OA -> MBON ------------------------------
    w[gc.PAM, gc.MBON_APPROACH] += 1.15  # appetitive: push the approach MBON
    w[gc.PAM, gc.MBON_CONFIDENCE] += 0.55
    w[gc.PAM, gc.MBON_AVOID] -= 0.35

    w[gc.PPL1, gc.MBON_AVOID] += 1.15  # aversive: push the avoidance MBON
    w[gc.PPL1, gc.MBON_APPROACH] -= 0.45
    w[gc.PPL1, gc.MBON_CONFIDENCE] += 0.25

    # Octopamine = hedge stress.  It suppresses both action pathways and raises
    # the HOLD tone, which is exactly "become cautious when the hedge breaks".
    w[gc.OA, gc.MBON_APPROACH] -= 0.55
    w[gc.OA, gc.MBON_AVOID] -= 0.55
    w[gc.OA, gc.MBON_NEUTRAL] += 1.30
    w[gc.OA, gc.MBON_CONFIDENCE] -= 0.40

    # ---------------- MBON -> LH ------------------------------------
    w[gc.MBON_APPROACH, gc.LH_APPROACH] += 1.25
    w[gc.MBON_APPROACH, gc.LH_AVOID] -= 0.45  # lateral inhibition
    w[gc.MBON_AVOID, gc.LH_AVOID] += 1.25
    w[gc.MBON_AVOID, gc.LH_APPROACH] -= 0.45
    w[gc.MBON_NEUTRAL, gc.LH_NEUTRAL] += 0.80
    w[gc.MBON_NEUTRAL, gc.LH_APPROACH] -= 0.30
    w[gc.MBON_NEUTRAL, gc.LH_AVOID] -= 0.30
    w[gc.MBON_CONFIDENCE, gc.LH_APPROACH] += 0.25
    w[gc.MBON_CONFIDENCE, gc.LH_AVOID] += 0.25

    # ---------------- LH <-> LH (lateral inhibition) ----------------
    w[gc.LH_APPROACH, gc.LH_AVOID] -= 0.40
    w[gc.LH_AVOID, gc.LH_APPROACH] -= 0.40
    w[gc.LH_APPROACH, gc.LH_NEUTRAL] -= 0.15
    w[gc.LH_AVOID, gc.LH_NEUTRAL] -= 0.15

    return mb.scale_to_unit(w)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the fallback brain matrix")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--out", type=Path, default=mb.default_fallback_path())
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    matrix = build_matrix(args.seed)
    path = mb.save_csv(matrix, args.out)
    info = mb.stats(matrix)
    print(f"wrote {path}")
    for key, value in info.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
