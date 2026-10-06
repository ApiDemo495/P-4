"""``FormulaSpec`` - one of the 2,100 (plus bred) formulas, and the grid that
turns a sub-category kernel into its 21 generative variants.

A kernel is a function ``kernel(frame, **params) -> (signal, raw)`` returning
two arrays aligned with the candle axis:

* ``signal`` - the directional reading for the *next* candle, roughly in
  [-1, +1] (positive = up).  Regime-type formulas translate their reading into
  a direction exactly as their interpretation says (cycling -> fade the last
  move, trending -> follow it), so every member of the pool can be scored
  against the next return with the same seven metrics.
* ``raw`` - the untranslated quantity (Lévy area, Betti number, entropy ...)
  shown in the dashboard next to the signal.

Layers (Part 4 of the specification): 2 = O(n) / O(n log n), run every
candle; 3 = O(n²), every candle; 4 = O(n³) or worse, every fifth candle and
held in between.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable

from backend.genesis.ops import NORMS, WINDOWS


class State(str, Enum):
    BIRTH = "BIRTH"
    CANDIDATE = "CANDIDATE"
    ACTIVE = "ACTIVE"
    DECAYING = "DECAYING"
    DEAD = "DEAD"
    AUTOPSY = "AUTOPSY"


DOMAINS: dict[int, tuple[str, str]] = {
    1: ("Rough path signatures", "Lyons' signature of the (price, volume) path - Lévy area, signature kernels, log-signature"),
    2: ("Persistent homology", "Vietoris-Rips Betti numbers, barcodes and landscapes of price/book point clouds"),
    3: ("Information geometry", "Fisher-Rao distance, natural gradient, α-divergences on the Gaussian manifold of returns"),
    4: ("Optimal transport", "Wasserstein shifts of the order book and of buy vs sell trade distributions"),
    5: ("Fractional stochastic calculus", "Hurst, Malliavin perturbation derivatives, rough volatility, Grünwald-Letnikov"),
    6: ("Tropical geometry", "max-plus polynomials, tropical roots, Newton polygons of OHLCV"),
    7: ("Quantum probability", "density matrix, von Neumann entropy, interference, quantum walks over indicator states"),
    8: ("p-adic analysis", "p-adic valuation of tick moves, ultrametric trees, p-adic wavelets"),
    9: ("Category theory", "natural transformations between timeframe functors, sheaf consistency, limits/colimits"),
    10: ("Algorithmic information", "Lempel-Ziv complexity, algorithmic mutual information, block entropy, logical depth"),
}


@dataclass
class FormulaSpec:
    fid: str
    name: str
    domain: int
    subcategory: str
    variant: int
    params: dict
    kernel: Callable
    layer: int = 2
    definition: str = ""
    interpretation: str = ""
    origin: str = "template"      # "template" or "bred"
    expression: str = ""          # for bred formulas: the symbolic tree
    gates: dict = field(default_factory=dict)  # autopsy-derived regime gates
    expression_tree: object = None  # bred formulas: the nested-tuple tree

    @property
    def domain_name(self) -> str:
        return DOMAINS.get(self.domain, ("bred", ""))[0]

    def describe(self) -> dict:
        return {"id": self.fid, "name": self.name, "domain": self.domain, "domain_name": self.domain_name,
                "subcategory": self.subcategory, "variant": self.variant, "params": self.params,
                "layer": self.layer, "definition": self.definition, "interpretation": self.interpretation,
                "origin": self.origin, "expression": self.expression, "gates": self.gates}


def grid(windows=WINDOWS, norms=NORMS, extra: dict | None = None) -> list[dict]:
    """The 21-variant grid: 7 look-backs x 3 normalisations by default.  When
    ``extra`` is given its keys cycle over the grid so that the sub-category's
    own axes (path type, prime, reference path ...) are covered too."""
    combos = [dict(n=n, norm=m) for n, m in itertools.product(windows, norms)]
    combos = combos[:21]
    while len(combos) < 21:
        combos.append(dict(combos[len(combos) % max(1, len(combos))]))
    if extra:
        for i, c in enumerate(combos):
            for key, values in extra.items():
                c[key] = values[i % len(values)]
    return combos


def make_variants(domain: int, sub_index: int, subcategory: str, name: str, kernel: Callable,
                  definition: str, interpretation: str, layer: int = 2,
                  params: list[dict] | None = None) -> list[FormulaSpec]:
    specs = []
    for v, p in enumerate(params or grid(), start=1):
        fid = f"D{domain}.{sub_index:02d}.v{v:02d}"
        specs.append(FormulaSpec(fid=fid, name=f"{name} v{v}", domain=domain, subcategory=subcategory,
                                 variant=v, params=dict(p), kernel=kernel, layer=layer,
                                 definition=definition, interpretation=interpretation))
    return specs
