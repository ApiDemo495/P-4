"""Three-layer graph convolution over the 80-node mushroom-body subgraph.

Node layout (0-indexed) - this mapping is fixed and is what the CSV fallback,
the live neuPrint query and CCSv2 all agree on:

===========  =====  ==========================================================
Index range  Count  Role
===========  =====  ==========================================================
0 - 19        20    Projection neurons, one per formula (PN_TAI ... PN_SMD)
20 - 69       50    Kenyon Cell clusters (spectrally clustered from ~2000 KCs)
70 - 72        3    Dopamine / octopamine neurons: PAM(+), PPL1(-), OA(stress)
73 - 76        4    MBONs: alpha3 (approach), gamma5beta'2a (avoid),
                    beta2beta'2a (neutral), alpha'2 (confidence)
77 - 79        3    Lateral horn output neurons (approach / avoid / neutral)
===========  =====  ==========================================================

Layer 1 (PN -> KC) applies ReLU and keeps only the top 10% of KC activations,
because the mushroom body sparse code is the whole point of the circuit.

Layers 2 and 3 are linear, so that *negative* evidence survives: a formula
reading of -0.8 must be able to excite the avoidance pathway.  ReLU is applied
only where biology applies it (the PN -> KC fan-in), never on the read-out.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

N_NODES = 80
N_FORMULAS = 20
N_KC = 50
KC_START, KC_END = 20, 70
PAM, PPL1, OA = 70, 71, 72
MBON_APPROACH, MBON_AVOID, MBON_NEUTRAL, MBON_CONFIDENCE = 73, 74, 75, 76
LH_APPROACH, LH_AVOID, LH_NEUTRAL = 77, 78, 79

SPARSITY = 0.10  # keep the top 10% of KC activations

#: The 20 projection neurons in formula order (index 0 = TAI, 19 = SMD).
PN_NAMES = (
    "TAI", "AFPR", "SED", "VSD", "DGW", "LCS", "BAR",
    "HRDD", "SHRP", "GCDV", "HSI", "RSV", "VSS", "ERC",
    "MCPE", "MPS", "TWRS", "DSKD", "NIV", "SMD",
)


@dataclass
class ActivationTrace:
    """Everything the UI/diagnostics need to explain a brain decision."""

    layer0: np.ndarray
    layer1: np.ndarray
    layer2: np.ndarray
    layer3: np.ndarray
    kc_activations: np.ndarray
    kc_drive: np.ndarray = None
    """Pre-ReLU Kenyon Cell input drive.  Formula 21's |a|^2 notation is
    sign-independent, and the rectified code is identically zero for a
    net-bearish ensemble, so confidence is measured on the drive."""
    kcae: float = 0.0
    lh_approach: float = 0.0
    lh_avoid: float = 0.0
    lh_neutral: float = 0.0
    ccs: float = 0.0
    confidence: float = 0.0
    active_kcs: int = 0
    diagnostics: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "kcae": round(self.kcae, 4),
            "lh_approach": round(self.lh_approach, 4),
            "lh_avoid": round(self.lh_avoid, 4),
            "lh_neutral": round(self.lh_neutral, 4),
            "ccs": round(self.ccs, 4),
            "confidence": round(self.confidence, 4),
            "confidence_spec": round(self.diagnostics.get("confidence_spec", 0.0), 4),
            "decisiveness": round(self.diagnostics.get("decisiveness", 0.0), 4),
            "active_kcs": self.active_kcs,
            "diagnostics": self.diagnostics,
        }


def normalize_adjacency(weights: np.ndarray) -> np.ndarray:
    """Symmetric normalisation ``D^-1/2 W D^-1/2`` (Section 3.8, step 2)."""
    n = weights.shape[0]
    degree = np.sum(np.abs(weights), axis=1)
    degree = np.where(degree > 1e-12, degree, 1.0)
    inv_sqrt = 1.0 / np.sqrt(degree)
    d_mat = np.diag(inv_sqrt)
    return d_mat @ weights @ d_mat


def sparsify_kc(activations: np.ndarray, keep_fraction: float = SPARSITY) -> np.ndarray:
    """Keep only the strongest KC activations (top 10% by default)."""
    out = np.array(activations, copy=True)
    kc = out[KC_START:KC_END]
    if kc.size == 0:
        return out
    k = max(1, int(round(kc.size * keep_fraction)))
    if k >= kc.size:
        return out
    cutoff = np.partition(kc, kc.size - k)[kc.size - k]
    kc[kc < cutoff] = 0.0
    out[KC_START:KC_END] = kc
    return out


class GraphConvolution:
    """Runs the 3-layer propagation used by CCSv2.

    The symmetric normalisation ``D^-1/2 W D^-1/2`` shrinks activations by the
    square root of the node degree at every layer.  Over three layers on an
    80-node graph that is roughly a 20x attenuation, which would leave CCSv2
    pinned at ~0 no matter how strong the evidence was.  A single positive
    scalar gain is therefore applied to the propagation matrix.

    Because every non-linearity in the forward pass (ReLU, top-10% sparsification)
    commutes with multiplication by a positive scalar, the forward pass is
    positively homogeneous in the gain.  That makes the calibration below exact
    rather than approximate: the gain is set so that a probe ensemble produces a
    read-out difference of ~1.0 before the final tanh.
    """

    def __init__(self, adjacency: np.ndarray, gain: float | str | None = "auto") -> None:
        if adjacency.shape != (N_NODES, N_NODES):
            raise ValueError(f"adjacency must be {(N_NODES, N_NODES)}, got {adjacency.shape}")
        # The stored matrix - and the CSV - use the intuitive edge-list layout
        # ``M[source, target]``.  A graph-convolution layer needs the opposite
        # convention (``W[receiver, sender]``) because it computes
        # ``a_out = W @ a_in``.  Normalising the transpose bridges the two
        # without forcing every other module to think in transposed indices.
        self.adjacency = np.asarray(adjacency, dtype=np.float64)
        self.normalized = normalize_adjacency(self.adjacency.T)
        if gain is None or gain == "auto":
            self.gain = self.calibrate_gain()
        else:
            self.gain = float(gain)
        self.propagation = self.normalized * self.gain

    # ------------------------------------------------------------------
    def calibrate_gain(self, target: float = 1.0, probes: int = 16, seed: int = 7) -> float:
        """Pick the gain that puts a fully coherent ensemble at ``target``.

        The forward pass multiplies by the gain once per layer, so the read-out
        scales with ``gain**3``.  The calibration therefore takes the cube root,
        otherwise the output saturates at +1/-1 for every input and CCSv2 stops
        carrying information.

        The probe set is anchored on the **all-ones vector** - the strongest
        coherent bullish ensemble the 20 formulas can produce - so a genuinely
        one-sided market reads ~0.76 after the final tanh while a half-conviction
        market lands near 0.4, which is where the 0.25 decision threshold lives.
        """
        probes_to_run = [np.ones(N_FORMULAS)]
        rng = np.random.default_rng(seed)
        for _ in range(probes):
            probes_to_run.append(rng.uniform(-1.0, 1.0, N_FORMULAS))

        peak = 0.0
        for probe in probes_to_run:
            trace = self._forward(probe, drg=0.0, hsi=0.0, matrix=self.normalized, gain=1.0)
            magnitude = abs(trace.lh_approach - trace.lh_avoid)
            peak = max(peak, magnitude)
        if peak <= 1e-12:
            return 1.0
        gain = (target / peak) ** (1.0 / 3.0)
        return float(min(max(gain, 1e-3), 1e3))

    # ------------------------------------------------------------------
    def propagate(
        self,
        formula_vector: np.ndarray,
        drg: float = 0.0,
        hsi: float = 0.0,
    ) -> ActivationTrace:
        """Full forward pass using the calibrated gain."""
        return self._forward(
            formula_vector, drg=drg, hsi=hsi, matrix=self.propagation, gain=self.gain
        )

    def _forward(
        self,
        formula_vector: np.ndarray,
        drg: float,
        hsi: float,
        matrix: np.ndarray,
        gain: float,
    ) -> ActivationTrace:
        a0 = np.zeros(N_NODES, dtype=np.float64)
        vec = np.asarray(formula_vector, dtype=np.float64).ravel()
        n = min(vec.size, N_FORMULAS)
        a0[:n] = vec[:n]
        # DANs + octopamine node enter at the input layer as specified.
        a0[PAM] = 0.0
        a0[PPL1] = 0.0
        a0[OA] = float(np.clip(hsi, 0.0, 1.0))

        # --- Layer 1: PN -> KC (+ DAN/OA overwrite) -------------------
        drive = matrix @ a0
        kc_drive = drive[KC_START:KC_END].copy()
        a1 = np.maximum(drive, 0.0)  # ReLU on the PN -> KC fan-in only
        a1[PAM] = gain * max(0.0, float(drg))
        a1[PPL1] = gain * max(0.0, -float(drg))
        a1[OA] = gain * float(np.clip(hsi, 0.0, 1.0))
        a1 = sparsify_kc(a1)

        # --- Layer 2: KC + DAN -> MBON --------------------------------
        a2 = matrix @ a1

        # --- Layer 3: MBON -> lateral horn (with lateral inhibition) ---
        a3 = matrix @ a2

        return ActivationTrace(
            layer0=a0,
            layer1=a1,
            layer2=a2,
            layer3=a3,
            kc_activations=a1[KC_START:KC_END].copy(),
            kc_drive=kc_drive,
            lh_approach=float(a3[LH_APPROACH]),
            lh_avoid=float(a3[LH_AVOID]),
            lh_neutral=float(a3[LH_NEUTRAL]),
            active_kcs=int(np.count_nonzero(a1[KC_START:KC_END])),
        )


def graph_convolution(test_input: np.ndarray, adjacency: np.ndarray) -> np.ndarray:
    """Stateless helper used by the brain health check (Section 6.6)."""
    conv = GraphConvolution(adjacency)
    vector = np.zeros(N_FORMULAS, dtype=np.float64)
    vec = np.asarray(test_input, dtype=np.float64).ravel()
    vector[: min(vec.size, N_FORMULAS)] = vec[:N_FORMULAS]
    trace = conv.propagate(vector, drg=0.1, hsi=0.2)
    return trace.layer3
