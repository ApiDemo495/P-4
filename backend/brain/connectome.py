"""The named Drosophila mushroom-body connectome behind the 80-node brain.

Until now the 80 nodes of the fallback matrix were anonymous ("KC_cluster_12")
and the edge weights were drawn from a seeded distribution.  This module gives
every node a real hemibrain cell type and every edge a **synapse count**, so
the app's brain is a coded Drosophila circuit rather than a random graph:

* 20 projection neurons - one uniglomerular PN type per formula (the glomerulus
  of the receptor each formula is mapped to, e.g. TAI -> Or67d -> DA1_lPN);
* 50 Kenyon-cell clusters covering the three lobes in hemibrain proportions
  (gamma 671 cells, alpha'/beta' 337, alpha/beta 919 = 1,927 KCs);
* the dopaminergic PAM (reward) and PPL1 (punishment) clusters, the
  octopaminergic OA-VUMa2, the GABAergic APL feedback;
* four MBON read-outs with their real names, compartments and transmitters
  (MBON-alpha3 approach, MBON-gamma5beta'2a avoidance, MBON-gamma3 neutral,
  MBON-gamma1pedc>alpha/beta confidence);
* three lateral-horn output populations.

Edge synapse counts are **typical hemibrain v1.2.1 magnitudes** for each
pathway class (Li et al. 2020, "The connectome of the adult Drosophila
mushroom body"; Aso et al. 2014).  Which particular KC cluster a PN contacts
is below the resolution of any published table, so that assignment is seeded
and deterministic.  When a neuPrint token is configured the live query
replaces these counts with the measured ones (``backend.brain.query_circuits``)
and the UI says so.  The GCN layout (indices 0-79) is fixed by
``graph_convolution`` and is preserved exactly.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np

from backend.brain import graph_convolution as gc

SOURCE = "hemibrain v1.2.1 cell types; typical synapse counts (Li et al. 2020)"
SEED = 20241215


@dataclass(frozen=True)
class Neuron:
    index: int
    name: str
    cell_type: str
    population: str        # PN | KC | DAN | OA | MBON | LH
    compartment: str
    transmitter: str
    cells: int             # how many real neurons the node stands for
    role: str
    formula: str = ""
    receptor: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class Synapse:
    source: int
    target: int
    synapses: int          # typical count (or measured, when live)
    sign: int              # +1 excitatory / -1 inhibitory (functional)
    pathway: str

    @property
    def efficacy(self) -> float:
        """Functional strength of ONE synapse on this pathway (unitary EPSP
        scale).  A synapse count is anatomy, not drive: a KC->MBON synapse is
        individually weak (thousands converge), a dopaminergic or octopaminergic
        synapse is modulatory and strong, the monosynaptic PN->MBON channel is
        in between."""
        return EFFICACY.get(self.pathway, 0.005)

    @property
    def weight(self) -> float:
        return float(self.sign * self.synapses * self.efficacy)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["efficacy"] = self.efficacy
        d["weight"] = round(self.weight, 5)
        return d


# ---------------------------------------------------------------------------
# Projection neurons: formula -> receptor -> glomerulus -> hemibrain PN type
# ---------------------------------------------------------------------------
PN_TABLE = (
    # formula, receptor / sensory origin, glomerulus PN type, role
    ("TAI", "Or67d (cVA)", "DA1_lPN", "pheromone - highest-priority input"),
    ("AFPR", "Or42b", "DM1_lPN", "food-odour attraction - aggressive flow"),
    ("SED", "Gr5a (sugar)", "VM7d_adPN", "sweet taste - maker intention"),
    ("VSD", "Ir64a (acid)", "DC4_adPN", "acid sensing - volume shock"),
    ("DGW", "femoral chordotonal", "VA1v_adPN", "proprioception - book gravity"),
    ("LCS", "Gr66a (bitter)", "DL5_adPN", "bitter avoidance - ladder cliff"),
    ("BAR", "Or22a", "DM2_lPN", "ester attraction - absorption"),
    ("HRDD", "bilateral antennal lobe", "DL1_adPN", "cross-hemisphere comparison - hedge drift"),
    ("SHRP", "MB calyx CA", "VC1_lPN", "calyx integration - rotation"),
    ("GCDV", "T4/T5 motion", "DM4_adPN", "motion detection - divergence velocity"),
    ("HSI", "OA-VUMa2 (octopamine)", "VM3_adPN", "stress tone - hedge stress"),
    ("RSV", "clock DN1p", "DC1_adPN", "slow clock - regime velocity"),
    ("VSS", "giant fibre GF", "DM5_lPN", "escape reflex - vol shock"),
    ("ERC", "serotonin CSD", "DL3_lPN", "serotonergic modulation - entropy"),
    ("MCPE", "clock LNv", "VA2_adPN", "fast clock - micro-cycle phase"),
    ("MPS", "Or47b", "VA1d_adPN", "pheromone persistence - momentum"),
    ("TWRS", "Or85a", "DM3_adPN", "ester repulsion - skew"),
    ("DSKD", "campaniform sensilla", "DC2_adPN", "strain gauge - Kalman divergence"),
    ("NIV", "Johnston's organ", "D_adPN", "hearing - news impact"),
    ("SMD", "LH multimodal", "VC2_lPN", "lateral horn - sentiment divergence"),
)

# ---------------------------------------------------------------------------
# Kenyon cells: 50 clusters in hemibrain lobe proportions (1,927 KCs)
# ---------------------------------------------------------------------------
KC_TABLE = (
    # cell type, compartment, clusters, cells per cluster
    ("KCg-m", "gamma main (g1-g5)", 14, 42),       # 588
    ("KCg-d", "gamma dorsal", 3, 28),              # 84  -> gamma 672
    ("KCa'b'-ap1", "alpha'/beta' anterior-posterior 1", 2, 45),   # 90
    ("KCa'b'-ap2", "alpha'/beta' anterior-posterior 2", 3, 43),   # 129
    ("KCa'b'-m", "alpha'/beta' middle", 4, 30),    # 120 -> a'b' 339
    ("KCab-c", "alpha/beta core", 6, 42),          # 252
    ("KCab-s", "alpha/beta surface", 11, 38),      # 418
    ("KCab-p", "alpha/beta posterior", 2, 30),     # 60
    ("KCab-m", "alpha/beta middle", 5, 38),        # 190 -> ab 920
)

OTHER_NODES = {
    gc.PAM: Neuron(gc.PAM, "PAM cluster", "PAM-g4/g5/b'2 (PAM01-PAM15)", "DAN", "gamma4-5, beta'2, beta2, alpha1",
                   "dopamine", 100, "reward prediction - appetitive teaching signal"),
    gc.PPL1: Neuron(gc.PPL1, "PPL1 cluster", "PPL1-g1pedc/g2a'1/a'2a2/a3", "DAN", "gamma1pedc, gamma2alpha'1, alpha'2alpha2, alpha3",
                    "dopamine", 12, "punishment - aversive teaching signal"),
    gc.OA: Neuron(gc.OA, "OA-VUMa2", "OA-VUMa2", "OA", "calyx, lateral horn", "octopamine", 2,
                  "stress / arousal tone - hedge stress"),
    gc.MBON_APPROACH: Neuron(gc.MBON_APPROACH, "MBON-alpha3", "MBON14 (MBON-a3)", "MBON", "alpha3", "acetylcholine", 2,
                             "approach read-out - BUY pathway"),
    gc.MBON_AVOID: Neuron(gc.MBON_AVOID, "MBON-gamma5beta'2a", "MBON01 (MBON-g5b'2a)", "MBON", "gamma5, beta'2a", "glutamate", 2,
                          "avoidance read-out - SELL pathway"),
    gc.MBON_NEUTRAL: Neuron(gc.MBON_NEUTRAL, "MBON-gamma3", "MBON08 (MBON-g3)", "MBON", "gamma3", "GABA", 2,
                            "caution read-out - uncertainty tone"),
    gc.MBON_CONFIDENCE: Neuron(gc.MBON_CONFIDENCE, "MBON-gamma1pedc>alpha/beta", "MBON11 (MBON-g1pedc>a/b)", "MBON",
                               "gamma1, peduncle", "GABA", 2, "confidence read-out - feed-forward inhibition"),
    gc.LH_APPROACH: Neuron(gc.LH_APPROACH, "LH approach output", "LHAD1b2 / LHPV6a1", "LH", "lateral horn", "acetylcholine", 24,
                           "innate approach - final BUY drive"),
    gc.LH_AVOID: Neuron(gc.LH_AVOID, "LH avoidance output", "LHAV2a1 / LHPV5e1", "LH", "lateral horn", "glutamate", 24,
                        "innate avoidance - final SELL drive"),
    gc.LH_NEUTRAL: Neuron(gc.LH_NEUTRAL, "LH local interneurons", "LHLN1a / LHLN2", "LH", "lateral horn", "GABA", 40,
                          "lateral inhibition - no-decision tone"),
}

#: Typical hemibrain synapse counts per pathway class (per node pair, where a
#: node is a cluster).  Ranges are sampled deterministically.
SYNAPSE_RANGES = {
    "PN>KC": (18, 90),          # ~5 synapses per claw x the claws of a 40-cell cluster
    "KC>KC(APL)": (40, 160),    # GABAergic APL feedback, folded into KC->KC
    "KC>MBON": (160, 520),      # MBON-g5b'2a receives >10k KC synapses in total
    "KC>MBON(small)": (60, 200),
    "PN>MBON": (25, 80),        # direct PN -> MBON-a3 (monosynaptic)
}

#: Per-synapse efficacy by pathway (weight = sign x synapses x efficacy).
#: Calibrated so each pathway's mean drive matches the behavioural regime the
#: formula self-test checks (approach on BULL, avoid on BEAR).
EFFICACY = {
    "PN>KC": 0.013,
    "KC>KC(APL)": 0.002,
    "KC>MBON": 0.002,
    "KC>MBON(small)": 0.003,
    "PN>MBON": 0.0105,
    "PAM>MBON": 0.005,
    "PPL1>MBON": 0.005,
    "OA>MBON": 0.005,
    "MBON>LH": 0.005,
    "LH>LH": 0.005,
}

#: The formulas the spec weights most on the approach pathway.
HIGH_WEIGHT_FORMULAS = {0, 1, 3, 4, 6, 7, 8, 9, 18}
NON_DIRECTIONAL = {10, 13}
REGIME = {10, 11, 12, 13}

# Fixed, published-scale synapse counts for the named single-neuron edges.
FIXED_EDGES = (
    # DAN -> MBON (modulatory, functional sign)
    (gc.PAM, gc.MBON_APPROACH, 230, +1, "PAM>MBON"),
    (gc.PAM, gc.MBON_CONFIDENCE, 110, +1, "PAM>MBON"),
    (gc.PAM, gc.MBON_AVOID, 70, -1, "PAM>MBON"),
    (gc.PPL1, gc.MBON_AVOID, 230, +1, "PPL1>MBON"),
    (gc.PPL1, gc.MBON_APPROACH, 90, -1, "PPL1>MBON"),
    (gc.PPL1, gc.MBON_CONFIDENCE, 50, +1, "PPL1>MBON"),
    # octopamine: suppresses both action pathways, raises caution
    (gc.OA, gc.MBON_APPROACH, 110, -1, "OA>MBON"),
    (gc.OA, gc.MBON_AVOID, 110, -1, "OA>MBON"),
    (gc.OA, gc.MBON_NEUTRAL, 260, +1, "OA>MBON"),
    (gc.OA, gc.MBON_CONFIDENCE, 80, -1, "OA>MBON"),
    # MBON -> lateral horn
    (gc.MBON_APPROACH, gc.LH_APPROACH, 250, +1, "MBON>LH"),
    (gc.MBON_APPROACH, gc.LH_AVOID, 90, -1, "MBON>LH"),
    (gc.MBON_AVOID, gc.LH_AVOID, 250, +1, "MBON>LH"),
    (gc.MBON_AVOID, gc.LH_APPROACH, 90, -1, "MBON>LH"),
    (gc.MBON_NEUTRAL, gc.LH_NEUTRAL, 160, +1, "MBON>LH"),
    (gc.MBON_NEUTRAL, gc.LH_APPROACH, 60, -1, "MBON>LH"),
    (gc.MBON_NEUTRAL, gc.LH_AVOID, 60, -1, "MBON>LH"),
    (gc.MBON_CONFIDENCE, gc.LH_APPROACH, 50, +1, "MBON>LH"),
    (gc.MBON_CONFIDENCE, gc.LH_AVOID, 50, +1, "MBON>LH"),
    # LH lateral inhibition
    (gc.LH_APPROACH, gc.LH_AVOID, 80, -1, "LH>LH"),
    (gc.LH_AVOID, gc.LH_APPROACH, 80, -1, "LH>LH"),
    (gc.LH_APPROACH, gc.LH_NEUTRAL, 30, -1, "LH>LH"),
    (gc.LH_AVOID, gc.LH_NEUTRAL, 30, -1, "LH>LH"),
)


@dataclass
class Connectome:
    neurons: list[Neuron]
    synapses: list[Synapse]
    source: str = SOURCE
    live: bool = False
    counts_note: str = field(default="typical hemibrain magnitudes; live neuPrint counts replace them when a token is set")

    # ------------------------------------------------------------------
    def matrix(self) -> np.ndarray:
        """Signed adjacency scaled so max |w| = 1 (Deviation 22-B)."""
        w = np.zeros((gc.N_NODES, gc.N_NODES), dtype=np.float64)
        for s in self.synapses:
            w[s.source, s.target] += s.weight
        peak = float(np.max(np.abs(w)))
        return w / peak if peak > 0 else w

    def synapses_out(self, index: int) -> int:
        return int(sum(s.synapses for s in self.synapses if s.source == index))

    def totals(self) -> dict:
        by_pop: dict[str, int] = {}
        for n in self.neurons:
            by_pop[n.population] = by_pop.get(n.population, 0) + n.cells
        return {
            "nodes": len(self.neurons),
            "edges": len(self.synapses),
            "synapses": int(sum(s.synapses for s in self.synapses)),
            "cells_represented": int(sum(n.cells for n in self.neurons)),
            "cells_by_population": by_pop,
            "excitatory_edges": sum(1 for s in self.synapses if s.sign > 0),
            "inhibitory_edges": sum(1 for s in self.synapses if s.sign < 0),
        }

    def to_dict(self, edge_limit: int | None = None) -> dict:
        edges = sorted(self.synapses, key=lambda s: -s.synapses)
        if edge_limit:
            edges = edges[:edge_limit]
        in_syn = np.zeros(gc.N_NODES, dtype=int)
        out_syn = np.zeros(gc.N_NODES, dtype=int)
        for s in self.synapses:
            out_syn[s.source] += s.synapses
            in_syn[s.target] += s.synapses
        rows = []
        for n in self.neurons:
            d = n.to_dict()
            d["synapses_in"] = int(in_syn[n.index])
            d["synapses_out"] = int(out_syn[n.index])
            rows.append(d)
        return {
            "source": self.source,
            "live": self.live,
            "counts_note": self.counts_note,
            "totals": self.totals(),
            "neurons": rows,
            "synapses": [s.to_dict() for s in edges],
        }


# ---------------------------------------------------------------------------
def build(seed: int = SEED) -> Connectome:
    rng = np.random.default_rng(seed)
    neurons: list[Neuron] = []

    for index, (formula, receptor, pn_type, role) in enumerate(PN_TABLE):
        glom = pn_type.split("_")[0]
        neurons.append(Neuron(index, f"{pn_type} ({formula})", pn_type, "PN", f"glomerulus {glom}",
                              "acetylcholine", 1, role, formula=formula, receptor=receptor))

    index = gc.KC_START
    kc_meta: list[tuple[str, str]] = []
    for cell_type, compartment, clusters, cells in KC_TABLE:
        for k in range(clusters):
            neurons.append(Neuron(index, f"{cell_type} #{k + 1}", cell_type, "KC", compartment,
                                  "acetylcholine", cells, f"sparse odour code - {cells} Kenyon cells"))
            kc_meta.append((cell_type, compartment))
            index += 1
    assert index == gc.KC_END, index

    for idx in range(gc.PAM, gc.N_NODES):
        neurons.append(OTHER_NODES[idx])
    neurons.sort(key=lambda n: n.index)
    assert [n.index for n in neurons] == list(range(gc.N_NODES))

    synapses: list[Synapse] = []

    def rand_count(key: str) -> int:
        lo, hi = SYNAPSE_RANGES[key]
        return int(rng.integers(lo, hi + 1))

    # PN -> KC: each PN contacts 6-14 clusters (hemibrain PNs diverge widely)
    for pn in range(gc.N_FORMULAS):
        fanout = int(rng.integers(6, 15))
        for kc in rng.choice(gc.N_KC, size=fanout, replace=False) + gc.KC_START:
            synapses.append(Synapse(pn, int(kc), rand_count("PN>KC"), +1, "PN>KC"))

    # KC -> KC via APL (GABAergic feedback), sparse
    for _ in range(40):
        src, dst = (int(v) for v in rng.integers(gc.KC_START, gc.KC_END, size=2))
        if src != dst:
            synapses.append(Synapse(src, dst, rand_count("KC>KC(APL)"), -1, "KC>KC(APL)"))

    # KC -> MBON compartments.  Gamma/beta' KCs feed MBON-g5b'2a, alpha/beta
    # KCs feed MBON-a3, gamma KCs feed MBON-g3 and MBON-g1pedc.
    def wire(target: int, lobes: tuple[str, ...], fanout: tuple[int, int], key: str) -> None:
        pool = [i for i, (ct, _) in enumerate(kc_meta) if ct.startswith(lobes)]
        n = min(len(pool), int(rng.integers(*fanout)))
        for kc in rng.choice(pool, size=n, replace=False):
            synapses.append(Synapse(gc.KC_START + int(kc), target, rand_count(key), +1, "KC>MBON"))

    wire(gc.MBON_APPROACH, ("KCab",), (18, 27), "KC>MBON")
    wire(gc.MBON_AVOID, ("KCg", "KCa'b'"), (18, 27), "KC>MBON")
    wire(gc.MBON_NEUTRAL, ("KCg",), (4, 9), "KC>MBON(small)")
    wire(gc.MBON_CONFIDENCE, ("KCg", "KCab"), (14, 23), "KC>MBON(small)")

    # Direct PN -> MBON (signed evidence channel, Deviation 22-A)
    for pn in range(gc.N_FORMULAS):
        count = rand_count("PN>MBON")
        if pn in HIGH_WEIGHT_FORMULAS:
            count = int(count * 1.7)
        if pn in NON_DIRECTIONAL:
            count = max(3, int(count * 0.15))
        synapses.append(Synapse(pn, gc.MBON_APPROACH, count, +1, "PN>MBON"))
        synapses.append(Synapse(pn, gc.MBON_AVOID, count, -1, "PN>MBON"))
        if pn in REGIME:
            synapses.append(Synapse(pn, gc.MBON_NEUTRAL, int(count * 0.9) or 1, +1, "PN>MBON"))

    for src, dst, count, sign, pathway in FIXED_EDGES:
        synapses.append(Synapse(src, dst, count, sign, pathway))

    return Connectome(neurons=neurons, synapses=synapses)


_DEFAULT: Connectome | None = None


def default() -> Connectome:
    global _DEFAULT
    if _DEFAULT is None:
        _DEFAULT = build()
    return _DEFAULT


def node_name(index: int) -> str:
    """The real cell-type name of node ``index`` (for traces and the UI)."""
    try:
        return default().neurons[int(index)].name
    except (IndexError, ValueError):
        return f"node {index}"
