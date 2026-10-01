"""Round Q - every formula is double-checked on every pass.

Replay on a private copy of the state, independent re-derivation from the
traced intermediates, range - and a formula that fails any of them is zeroed
for the pass with the reason in ``errors``."""

from __future__ import annotations

import pytest

from backend.formulas import double_check as dc
from backend.formulas import self_test, synthetic
from backend.formulas.engine import ALL_FORMULAS, FormulaEngine
from backend.formulas import drg as drg_module


def _engine():
    return FormulaEngine(brain=self_test._brain_stub())


def _run_stream(key: str = "BULL", asset: str = "BTC", windows: int | None = None):
    scenario = synthetic.scenario(key)
    engine = _engine()
    result = None
    for snapshot in (scenario.snapshots if windows is None else scenario.snapshots[-windows:]):
        result = engine.run(snapshot, asset)
    return result


def test_every_formula_module_declares_its_double_check():
    for spec in ALL_FORMULAS:
        assert callable(getattr(spec.module, "double_check", None)), spec.name
        assert getattr(spec.module, "DOUBLE_CHECK", ""), spec.name
    assert callable(drg_module.double_check) and drg_module.DOUBLE_CHECK


@pytest.mark.parametrize("key", list(synthetic.SCENARIOS)[:4])
@pytest.mark.parametrize("asset", ["BTC", "PAXG"])
def test_all_formulas_verify_on_every_synthetic_tape(key, asset):
    result = _run_stream(key, asset)
    names = [spec.name for spec in ALL_FORMULAS] + ["DRG"]
    assert set(result.checks) >= set(names)
    failed = {n: c for n, c in result.checks.items() if not c["ok"]}
    assert not failed, failed
    # every formula got all three checks, not just the range
    assert all(c["verdict"] == "verified" for c in result.checks.values()), result.checks
    summary = result.check_summary()
    assert summary["verified"] == summary["formulas"] == len(result.checks)
    assert summary["failed"] == 0 and summary["check_us"] > 0
    # the verdict is on the trace, next to the work
    for name in names:
        rows = [r for r in result.traces[name] if r.get("label") == "double check"]
        assert len(rows) == 1 and "verified" in rows[0]["value"], name


def test_traces_carry_the_exact_numbers_used_by_the_re_derivation():
    result = _run_stream("BULL", "BTC", windows=5)
    raw = dc.raw_trace(result.traces["TAI"])
    assert "jerk / floor" in raw and isinstance(raw["jerk / floor"], float)
    # a rendered row never replaces the raw one
    assert any("raw" in row for row in result.traces["TAI"])


def test_a_formula_whose_intermediates_do_not_match_its_output_is_zeroed(monkeypatch):
    from backend.formulas.category_a_microstructure import tai

    real = tai.compute

    def lying_compute(snapshot, asset, state, params, ctx=None):
        value = real(snapshot, asset, state, params, ctx)
        return 0.9 if abs(value) < 0.9 else -0.9   # not what the trace says

    monkeypatch.setattr(tai, "compute", lying_compute)
    result = _run_stream("BULL", "BTC", windows=3)
    check = result.checks["TAI"]
    assert check["ok"] is False and check["verdict"] == "failed"
    assert check["rederived"] is False
    assert result.values["TAI"] == 0.0
    assert result.errors["TAI"].startswith("double check failed")
    assert "re-derive" in result.errors["TAI"]


def test_a_non_deterministic_formula_fails_the_replay(monkeypatch):
    from backend.formulas.category_e_temporal import twrs

    calls = {"n": 0}
    real = twrs.compute

    def flaky(snapshot, asset, state, params, ctx=None):
        calls["n"] += 1
        value = real(snapshot, asset, state, params, ctx)
        return value if calls["n"] % 2 else value + 0.05

    monkeypatch.setattr(twrs, "compute", flaky)
    result = _run_stream("BULL", "BTC", windows=2)
    check = result.checks["TWRS"]
    assert check["replay"] is False and result.values["TWRS"] == 0.0
    assert "replay gave" in result.errors["TWRS"]


def test_out_of_range_output_is_caught_even_when_the_trace_agrees(monkeypatch):
    from backend.formulas.category_c_hedge import hsi

    monkeypatch.setattr(hsi, "compute", lambda *a, **k: 1.4)
    monkeypatch.setattr(hsi, "double_check", lambda t, asset: 1.4)
    result = _run_stream("STRESS", "BTC", windows=2)
    check = result.checks["HSI"]
    assert check["range_ok"] is False and result.values["HSI"] == 0.0
    assert "outside [0, 1]" in result.errors["HSI"]


def test_double_check_can_be_switched_off(monkeypatch):
    monkeypatch.setenv("FORMULA_DOUBLE_CHECK", "0")
    result = _run_stream("BULL", "BTC", windows=2)
    assert result.checks == {} and result.check_summary()["enabled"] is False
    assert result.to_dict()["double_check"]["formulas"] == 0
