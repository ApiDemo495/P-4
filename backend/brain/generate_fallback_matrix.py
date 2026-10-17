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
  * OA  -> MB/LH  : octopamine suppresses action and raises the caution tone
  * MBON -> LH    : excitatory to its own LH target, inhibitory to the opposite
  * LH  -> LH     : lateral inhibition inside the lateral horn

NOTE (Deviation 22-B)
---------------------
The specification normalises weights into [0, 1].  Real mushroom-body circuits
contain both excitatory and inhibitory projections, and CCSv2 cannot express
"bearish" without them: with a purely non-negative matrix and a ReLU on the
PN -> KC fan-in, an all-negative formula vector would produce zero Kenyon Cell
activity and the read-out would collapse to HSH (0 = no decision) regardless of how
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

from backend.brain import matrix_builder as mb

log = logging.getLogger("drosophila.brain.generate")

DEFAULT_SEED = 20241215


def build_matrix(seed: int = DEFAULT_SEED) -> np.ndarray:
    """The signed 80x80 adjacency of the named connectome (Round S): every
    edge is a hemibrain pathway with a synapse count; the only seeded part is
    which Kenyon-cell cluster a given projection neuron contacts."""
    from backend.brain import connectome

    return mb.scale_to_unit(connectome.build(seed).matrix())


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
