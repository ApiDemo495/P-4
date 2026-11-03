"""Round BB - Forward Expectancy: every input to the decision is the measured
distribution of what FOLLOWED readings like the present ones, not the
present reading extrapolated."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from backend.core.forward import MIN_COVERED, ForwardExpectancy

ROOT = Path(__file__).resolve().parents[2]


def _teach(fx: ForwardExpectancy, n: int, rule) -> None:
    """Windows where a +0.5 TAI reading is followed by ``rule(i)`` (True = up)."""
    for i in range(n):
        fx.observe(i, {"TAI": 0.5, "AFPR": -0.4, "MPS": 0.2, "RBB": 0.1, "WDR": 0.3, "RAR": -0.2}, 1)
        fx.settle(i, rule(i), 3.0 if rule(i) else -3.0)


def test_a_bullish_reading_that_is_followed_by_down_windows_gets_a_negative_forward_value():
    fx = ForwardExpectancy()
    _teach(fx, 60, lambda i: i % 4 == 0)          # up only 25 % of the time after this reading
    e = fx.expect_one("TAI", 0.5, 1)
    assert e.n >= 50 and e.p_up < 0.35 and e.edge < 0 and "after an up window" in e.context
    comp = fx.composite({"TAI": 0.5, "AFPR": -0.4, "MPS": 0.2, "RBB": 0.1, "WDR": 0.3, "RAR": -0.2}, 1)
    assert comp["available"] and comp["side"] == "SELL" and comp["covered"] >= MIN_COVERED
    assert comp["inverted_vs_reading"] >= 1               # TAI points against its own reading
    edges = fx.edges_vector({"TAI": 0.5, "AFPR": -0.4}, 1)
    assert edges["TAI"] < 0 < 0.5                         # the brain is fed the forward value


def test_no_record_means_fifty_percent_and_abstention():
    fx = ForwardExpectancy()
    e = fx.expect_one("TAI", 0.9, 0)
    assert abs(e.p_up - 0.5) < 1e-9 and e.n == 0
    assert fx.composite({"TAI": 0.9, "AFPR": 0.9}, 0)["available"] is False
    assert fx.edges_vector({"TAI": 0.9}, 0) == {}


def test_context_cell_falls_back_to_pooled_and_ties_are_counted():
    fx = ForwardExpectancy()
    for i in range(20):
        fx.observe(i, {"TAI": 0.5}, -1)
        fx.settle(i, None if i % 5 == 0 else True, 0.0)
    e = fx.expect_one("TAI", 0.5, 1)                      # asked after an UP window: no such cell
    assert "pooled" in e.context and e.p_tie > 0.15 and fx.ties == 4
    tbl = fx.table("TAI")
    assert tbl["rows"] and any(r["context"] == "pooled" for r in tbl["rows"])


def test_quantile_bins_once_a_formula_has_a_record():
    fx = ForwardExpectancy()
    rng = np.random.default_rng(0)
    for i in range(80):
        fx.observe(i, {"TAI": float(rng.normal(0, 3))}, 0)
        fx.settle(i, True, 1.0)
    edges = fx.edges("TAI")
    assert edges.size == 6 and edges[0] < -1.0 and edges[-1] > 1.0   # wider than the fixed edges


def test_the_forward_layer_is_wired_everywhere():
    cm = (ROOT / "backend" / "core" / "cycle_manager.py").read_text()
    assert "self.formulas.forward_hook = " in cm and "self.forward.observe(" in cm and "self.forward.settle(" in cm
    assert "forward=forward," in cm and "forward_vote=" in cm and '"forward"' in cm.split("VAULT_OBJECTS = ")[1].split("\n")[0]
    fu = (ROOT / "backend" / "agents" / "fusion.py").read_text()
    assert 'table.mark("forward"' in fu and '"forward")' in fu.split("elif name in (")[1].split("\n")[0]
    cc = (ROOT / "backend" / "formulas" / "category_h_brain" / "ccsv2.py").read_text()
    assert '_forward_edges' in cc
    ws = (ROOT / "backend" / "core" / "warm_start.py").read_text()
    assert "forward.observe(cycle" in ws and "forward.settle(cycle" in ws
    js = (ROOT / "backend" / "web" / "app.js").read_text()
    assert "state.liveForward" in js and "AGAINST the reading" in js and "FROM THE FUTURE" in js
    rt = (ROOT / "backend" / "api" / "routes_formulas.py").read_text()
    assert '@router.get("/api/formulas/forward")' in rt
    cfg = (ROOT / "backend" / "core" / "config.py").read_text()
    assert "weight_forward" in cfg
