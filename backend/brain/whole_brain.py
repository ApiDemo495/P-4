"""Running the real fly brain on the formulas (Round AQ).

Two instances of the same machinery:

* ``MushroomBody``  - the 8 353-neuron mushroom-body / lateral-horn circuit,
  a pass in a few milliseconds.  It replaces the 80-node stand-in in the
  per-pass formulas: CCSv2 reads the real MBONs, KCAE the real Kenyon cells.
* ``WholeBrain``    - all 138 639 neurons and 15.09 M connections; a pass is
  ~0.1-0.4 s, run on a worker thread every full emotion sample (2 s) and
  consulted at the lock.  Its read-out adds the rest of the brain: what the
  descending neurons (the fly's motor command) do with the mushroom body's
  verdict.

The model (every choice stated, nothing hidden):

* **Input.** The 20 directional formulas are the fly's odour.  Each formula
  owns a group of antennal-lobe projection neurons (the 685 ALPNs sorted by
  glomerulus / cell type and cut into 20 contiguous groups, so a formula maps
  onto whole glomeruli).  Like a real olfactory channel the group is split
  glomeruli, cut into 40 contiguous groups).  Formula i owns two of them,
  like a real olfactory channel pair: a positive value excites its ON group
  with its magnitude, a negative value its OFF group - two different odours
  the circuit can tell apart.  Rate = |value| (0..1).
* **Propagation.** Rate neurons, ``h_{k+1} = ReLU(W h_k)`` with ``W`` the
  input-fraction-normalised signed connectome (inhibitory synapses subtract),
  for ``hops`` steps; the brain's response is the activity summed over hops.
  Kenyon cells are kept sparse (top 5 % per hop) - the APL feedback neuron's
  known effect - which is what makes the KC code high-dimensional and
  decorrelated.
* **Neuromodulation.** DRG (reward learning) acts as dopamine: PAM DANs scale
  the KC -> MBON synapses of the approach MBONs up for positive DRG, PPL1 DANs
  the avoidance MBONs for negative DRG (Aso et al. 2014; Hige et al. 2015).
  HSI acts as octopamine: an arousal gain on the input layer.
* **Read-out.** MBON valence by neurotransmitter (Aso et al. 2014, eLife
  3:e04580): activation of the glutamatergic MBONs drives *avoidance*, the
  cholinergic and GABAergic MBONs drive *approach*.  ``balance =
  approach - avoid`` normalised by the total MBON activity; the whole brain
  adds the descending-neuron drive (how much of the verdict reaches the
  motor command) and the activity per super-class.
* **Push-pull.** As in the 80-node version, the mirrored input (-x) is
  propagated too and the read-outs combined antisymmetrically, so a flat
  tape reads exactly 0 and f(-x) = -f(x).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from backend.brain import connectome_data as cd
from backend.brain import graph_convolution as gc

N_FORMULAS = len(gc.PN_NAMES)
KC_SPARSITY = 0.05
EPS = 1e-9

#: MBON valence by neurotransmitter (Aso et al. 2014)
APPROACH_NT = ("acetylcholine", "gaba")
AVOID_NT = ("glutamate",)


@dataclass
class Readout:
    """One propagation through a connectome graph."""

    approach: float = 0.0
    avoid: float = 0.0
    neutral: float = 0.0
    balance: float = 0.0           # odd read-out in [-1, 1]
    confidence: float = 0.0
    active_kcs: int = 0
    kc_activity: np.ndarray | None = None
    kc_drive: np.ndarray | None = None
    pn_activations: np.ndarray | None = None
    hop_active: list = field(default_factory=list)
    by_super_class: dict = field(default_factory=dict)
    top_types: list = field(default_factory=list)
    top_mbons: list = field(default_factory=list)
    descending: float = 0.0
    dan: dict = field(default_factory=dict)
    neurons: int = 0
    connections: int = 0
    hops: int = 0
    elapsed_us: int = 0
    computed_at: float = 0.0

    def to_dict(self) -> dict:
        return {
            "approach": round(self.approach, 6), "avoid": round(self.avoid, 6), "neutral": round(self.neutral, 6),
            "balance": round(self.balance, 6), "confidence": round(self.confidence, 6),
            "active_kcs": self.active_kcs, "hop_active": list(self.hop_active),
            "by_super_class": dict(self.by_super_class), "top_types": list(self.top_types),
            "top_mbons": list(self.top_mbons), "descending": round(self.descending, 6), "dan": dict(self.dan),
            "neurons": self.neurons, "connections": self.connections, "hops": self.hops,
            "elapsed_us": self.elapsed_us, "computed_at": self.computed_at,
        }


class ConnectomeGraph:
    """A signed, input-fraction-normalised connectome and the populations in it."""

    def __init__(self, npz_path: Path, name: str, hops: int) -> None:
        import scipy.sparse as sp

        z = np.load(npz_path, allow_pickle=False)
        self.name = name
        self.hops = int(hops)
        self.n = int(z["n"])
        self.W = sp.csr_matrix((z["data"], z["indices"], z["indptr"]), shape=(self.n, self.n))
        self.root_ids = z["root_ids"]
        self.super_class = z["super_class"]
        self.nt = z["nt"]
        self.cell_class = z["cell_class"]
        self.cell_type = z["cell_type"]
        self.connections = int(z["connections"])
        self.synapses_total = int(z["synapses_total"])
        nts = np.asarray(cd.NT_CODES)[self.nt]

        # --- populations ---------------------------------------------------
        self.alpn = np.flatnonzero(self.cell_class == "ALPN")
        self.kc = np.flatnonzero(self.cell_class == "Kenyon_Cell")
        self.mbon = np.flatnonzero(np.char.startswith(self.cell_type, "MBON") | (self.cell_class == "MBON"))
        self.dan = np.flatnonzero(self.cell_class == "DAN")
        self.pam = self.dan[np.char.startswith(self.cell_type[self.dan], "PAM")]
        self.ppl1 = self.dan[np.char.startswith(self.cell_type[self.dan], "PPL1")]
        self.lh = np.flatnonzero(np.isin(self.cell_class, ("LHLN", "LHCENT")))
        self.descending = np.flatnonzero(self.super_class == cd.SUPER_CLASSES.index("descending"))
        self.mbon_approach = self.mbon[np.isin(nts[self.mbon], APPROACH_NT)]
        self.mbon_avoid = self.mbon[np.isin(nts[self.mbon], AVOID_NT)]
        self.mbon_other = np.setdiff1d(self.mbon, np.concatenate([self.mbon_approach, self.mbon_avoid]))

        # --- formula -> glomerulus groups (ON/OFF halves) ------------------
        # 40 glomerular groups: formula i owns group 2i (ON, positive values)
        # and group 2i+1 (OFF, negative values) - two *different* odour
        # channels, so a bullish and a bearish reading are discriminable
        # downstream rather than the same glomerulus split in two.
        order = self.alpn[np.argsort(self.cell_type[self.alpn], kind="stable")]
        groups = np.array_split(order, 2 * N_FORMULAS) if order.size else [np.array([], dtype=int)] * (2 * N_FORMULAS)
        self.on_groups = [groups[2 * i] for i in range(N_FORMULAS)]
        self.off_groups = [groups[2 * i + 1] for i in range(N_FORMULAS)]
        self.kc_keep = max(1, int(round(KC_SPARSITY * self.kc.size))) if self.kc.size else 0

        # dopamine gating acts on the rows of the MBONs (their incoming synapses)
        self._approach_rows = np.zeros(self.n, dtype=bool)
        self._approach_rows[self.mbon_approach] = True
        self._avoid_rows = np.zeros(self.n, dtype=bool)
        self._avoid_rows[self.mbon_avoid] = True

    # ------------------------------------------------------------------
    def populations(self) -> dict:
        return {
            "neurons": self.n, "connections": self.connections, "synapses": self.synapses_total,
            "alpn": int(self.alpn.size), "kenyon_cells": int(self.kc.size), "mbon": int(self.mbon.size),
            "mbon_approach": int(self.mbon_approach.size), "mbon_avoid": int(self.mbon_avoid.size),
            "dan": int(self.dan.size), "pam": int(self.pam.size), "ppl1": int(self.ppl1.size),
            "lateral_horn": int(self.lh.size), "descending": int(self.descending.size),
            "formula_groups": [{"formula": gc.PN_NAMES[i], "on": int(self.on_groups[i].size),
                                "off": int(self.off_groups[i].size),
                                "glomeruli": sorted({str(t).split("_")[0] for t in self.cell_type[self.on_groups[i]]})[:6]}
                               for i in range(N_FORMULAS)],
        }

    def _inject(self, vector: np.ndarray, hsi: float) -> np.ndarray:
        h = np.zeros(self.n, dtype=np.float32)
        gain = 1.0 + 0.3 * float(np.clip(hsi, 0.0, 1.0))      # octopamine arousal
        for i in range(min(vector.size, N_FORMULAS)):
            v = float(vector[i])
            if v > 0:
                h[self.on_groups[i]] = v * gain
            elif v < 0:
                h[self.off_groups[i]] = -v * gain
        return h

    def _one_pass(self, vector: np.ndarray, drg: float, hsi: float) -> tuple[np.ndarray, np.ndarray, list, np.ndarray]:
        h = self._inject(vector, hsi)
        total = h.copy()
        hop_active = [int(np.count_nonzero(h))]
        kc_drive = None
        # dopamine: PAM potentiates approach-MBON input for reward, PPL1 the avoid side
        g_app = 1.0 + 0.5 * max(0.0, float(drg))
        g_avo = 1.0 + 0.5 * max(0.0, -float(drg))
        for hop in range(self.hops):
            drive = self.W @ h
            if hop == 0 and self.kc.size:
                kc_drive = drive[self.kc].copy()
            if g_app != 1.0:
                drive[self._approach_rows] *= g_app
            if g_avo != 1.0:
                drive[self._avoid_rows] *= g_avo
            h = np.maximum(drive, 0.0, dtype=np.float32)
            if self.kc_keep and self.kc.size:
                kc = h[self.kc]
                if kc.size > self.kc_keep:
                    thr = np.partition(kc, -self.kc_keep)[-self.kc_keep]
                    kc = np.where(kc >= thr, kc, 0.0) if thr > 0 else kc
                    h[self.kc] = kc
            total += h
            hop_active.append(int(np.count_nonzero(h)))
        if kc_drive is None:
            kc_drive = np.zeros(0, dtype=np.float32)
        return total, kc_drive, hop_active, h

    def propagate(self, vector: np.ndarray, drg: float = 0.0, hsi: float = 0.0, detail: bool = True) -> Readout:
        t0 = time.perf_counter()
        vec = np.asarray(vector, dtype=np.float32).ravel()
        tot, kc_drive, hop_active, _ = self._one_pass(vec, drg, hsi)
        mir, mir_drive, _, _ = self._one_pass(-vec, drg, hsi)

        def mbon_sum(total: np.ndarray, idx: np.ndarray) -> float:
            return float(total[idx].sum()) if idx.size else 0.0

        approach = 0.5 * (mbon_sum(tot, self.mbon_approach) + mbon_sum(mir, self.mbon_avoid))
        avoid = 0.5 * (mbon_sum(tot, self.mbon_avoid) + mbon_sum(mir, self.mbon_approach))
        neutral = 0.5 * (mbon_sum(tot, self.mbon_other) + mbon_sum(mir, self.mbon_other))
        denom = abs(approach) + abs(avoid) + EPS
        # odd read-out in [-1, 1]: the share of MBON drive on the approach side
        balance = float(np.tanh(2.0 * (approach - avoid) / denom)) if denom > 1e-6 else 0.0
        kc_act = tot[self.kc] if self.kc.size else np.zeros(0, dtype=np.float32)
        kc_drive_pp = (kc_drive - mir_drive) if kc_drive.size and mir_drive.size else kc_drive
        out = Readout(
            approach=approach, avoid=avoid, neutral=neutral, balance=balance,
            confidence=float(abs(approach - avoid) / denom) if denom > 1e-6 else 0.0,
            active_kcs=int(np.count_nonzero(kc_act)), kc_activity=kc_act, kc_drive=kc_drive_pp,
            pn_activations=vec[:N_FORMULAS].astype(np.float64), hop_active=hop_active,
            neurons=self.n, connections=self.connections, hops=self.hops,
            dan={"drg": float(drg), "hsi": float(hsi), "pam_gain": 1.0 + 0.5 * max(0.0, float(drg)),
                 "ppl1_gain": 1.0 + 0.5 * max(0.0, -float(drg)), "pam_neurons": int(self.pam.size),
                 "ppl1_neurons": int(self.ppl1.size)},
            computed_at=time.time(),
        )
        if self.descending.size:
            out.descending = float(tot[self.descending].mean())
        if detail:
            net = tot - mir  # odd activity: what the sign of the input changed
            classes = np.asarray(cd.SUPER_CLASSES)
            for code in np.unique(self.super_class):
                idx = self.super_class == code
                out.by_super_class[str(classes[int(code)]) or "unlabelled"] = {
                    "neurons": int(idx.sum()), "active": int(np.count_nonzero(tot[idx])),
                    "mean_activity": round(float(tot[idx].mean()), 6) if idx.any() else 0.0,
                }
            if self.mbon.size:
                order = self.mbon[np.argsort(-np.abs(net[self.mbon]))][:8]
                out.top_mbons = [{"type": str(self.cell_type[i]), "nt": cd.NT_CODES[int(self.nt[i])],
                                  "valence": ("approach" if self.nt[i] in (cd.NT_CODES.index("acetylcholine"), cd.NT_CODES.index("gaba"))
                                              else "avoid" if self.nt[i] == cd.NT_CODES.index("glutamate") else "other"),
                                  "activity": round(float(net[i]), 6), "root_id": str(self.root_ids[i])}
                                 for i in order]
            # the busiest cell types outside the inputs
            mask = np.ones(self.n, dtype=bool)
            mask[self.alpn] = False
            idx = np.flatnonzero(mask & (np.abs(net) > 0))
            if idx.size:
                top = idx[np.argsort(-np.abs(net[idx]))[:400]]
                agg: dict[str, float] = {}
                for i in top:
                    key = str(self.cell_type[i]) or str(self.cell_class[i]) or "unlabelled"
                    agg[key] = agg.get(key, 0.0) + float(net[i])
                out.top_types = [{"type": k, "net_activity": round(v, 6)}
                                 for k, v in sorted(agg.items(), key=lambda kv: -abs(kv[1]))[:8]]
        out.elapsed_us = int((time.perf_counter() - t0) * 1e6)
        return out

    def to_activation_trace(self, r: Readout) -> gc.ActivationTrace:
        """The shape the per-pass formulas and the UI already understand."""
        trace = gc.ActivationTrace(
            layer0=r.pn_activations if r.pn_activations is not None else np.zeros(N_FORMULAS),
            layer1=r.kc_activity if r.kc_activity is not None else np.zeros(0),
            layer2=np.array([r.approach, r.avoid, r.neutral], dtype=np.float64),
            layer3=np.array([r.approach, r.avoid, r.neutral], dtype=np.float64),
            kc_activations=np.asarray(r.kc_activity, dtype=np.float64) if r.kc_activity is not None else np.zeros(0),
            kc_drive=np.asarray(r.kc_drive, dtype=np.float64) if r.kc_drive is not None else np.zeros(0),
            lh_approach=float(r.approach), lh_avoid=float(r.avoid), lh_neutral=float(r.neutral),
            active_kcs=int(r.active_kcs), pn_activations=np.asarray(r.pn_activations, dtype=np.float64),
            mbon={"approach": float(r.approach), "avoid": float(r.avoid), "neutral": float(r.neutral),
                  "confidence": float(r.confidence)},
            dan=dict(r.dan),
        )
        trace.diagnostics = {
            "connectome": self.name, "neurons": self.n, "connections": self.connections,
            "hops": self.hops, "elapsed_us": r.elapsed_us, "hop_active": list(r.hop_active),
            "top_mbons": list(r.top_mbons),
        }
        return trace


def load_mushroom_body(data_dir: Path = cd.DATA_DIR) -> ConnectomeGraph:
    return ConnectomeGraph(data_dir / cd.MB_NPZ, "FlyWire v783 mushroom body + lateral horn", hops=3)


def load_whole_brain(data_dir: Path = cd.DATA_DIR) -> ConnectomeGraph:
    return ConnectomeGraph(data_dir / cd.WHOLE_NPZ, "FlyWire v783 whole brain", hops=6)
