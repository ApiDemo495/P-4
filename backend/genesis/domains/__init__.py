"""The ten domains.  ``all_variants()`` returns the 2,100 template formulas."""
from __future__ import annotations

from importlib import import_module

MODULES = ("d01_signatures", "d02_homology", "d03_infogeo", "d04_transport", "d05_fractional",
           "d06_tropical", "d07_quantum", "d08_padic", "d09_category", "d10_algorithmic")


#: execution layer per domain (Part 4): 2 = every candle, cheap; 3 = O(n²)
#: every candle; 4 = heaviest, every fifth candle live (held in between).
LAYERS = {1: 2, 2: 4, 3: 2, 4: 3, 5: 2, 6: 3, 7: 3, 8: 2, 9: 2, 10: 4}


def all_variants():
    specs = []
    for mod in MODULES:
        for sp in import_module(f"backend.genesis.domains.{mod}").variants():
            sp.layer = LAYERS.get(sp.domain, 2)
            specs.append(sp)
    return specs
