"""The ten domains.  ``all_variants()`` returns the 2,100 template formulas."""
from __future__ import annotations

from importlib import import_module

MODULES = ("d01_signatures", "d02_homology", "d03_infogeo", "d04_transport", "d05_fractional",
           "d06_tropical", "d07_quantum", "d08_padic", "d09_category", "d10_algorithmic")


def all_variants():
    specs = []
    for mod in MODULES:
        specs.extend(import_module(f"backend.genesis.domains.{mod}").variants())
    return specs
