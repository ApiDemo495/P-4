"""The formula contract: do the 22 formulas (and the DRG) do what they claim?

Round E asked for three things and this module is the regression guard for two
of them:

* **the numbers must follow their own maths** - :func:`test_every_formula_passes_its_self_test`
  replays four synthetic tapes (bull / bear / chop / hedge-breakdown) through a
  real :class:`~backend.formulas.engine.FormulaEngine` and checks every formula's
  *sign* on the tape whose character is known by construction;
* **the logic must be visible** - the other tests assert that every formula
  publishes a logic entry (expression, steps, bands, sign) and records the
  intermediate numbers (traces) that the Formula Explorer renders.

Run with:  PYTHONPATH=. python -m pytest backend/tests -q
"""

from __future__ import annotations

import pytest

from backend.formulas import logic as logic_module
from backend.formulas import self_test as self_test_module
from backend.formulas import synthetic
from backend.formulas.engine import ALL_FORMULAS, FormulaEngine

FORMULA_NAMES = [spec.name for spec in ALL_FORMULAS]
ALL_NAMES = FORMULA_NAMES + ["DRG"]


def test_every_formula_passes_its_self_test():
    """The four synthetic tapes must reproduce every formula's declared sign."""
    report = self_test_module.run()
    failed = [v for v in report.verdicts if not v.passed]
    detail = "; ".join(
        f"{v.name}: {v.claim} -> {v.detail} | {v.observed}" for v in failed
    )
    assert report.passed, f"{len(failed)} formula(s) failed: {detail}"
    assert len(report.verdicts) == len(ALL_NAMES)


def test_every_formula_publishes_a_logic_entry():
    """Expression, steps, reading bands and the sign convention, for all 23."""
    for name in ALL_NAMES:
        entry = logic_module.get(name)
        assert entry is not None, f"{name} has no logic entry"
        assert entry.expression, f"{name} has no expression"
        assert len(entry.steps) >= 2, f"{name} has too few steps"
        assert entry.bands, f"{name} has no reading bands"
        assert entry.sign, f"{name} has no sign convention"
        # The value -> words mapping must be total over the formula's *own*
        # range: HSI is a [0, 1] indicator while ERC is signed, so the declared
        # bands - not a hard-coded [-1, 1] - are what has to cover the domain,
        # without a hole between neighbouring bands.
        bands = sorted(entry.bands, key=lambda band: band[0])
        for (_, hi, _), (lo_next, _, _) in zip(bands, bands[1:]):
            assert abs(hi - lo_next) < 1e-9, f"{name} has a gap between {hi} and {lo_next}"
        low, high = bands[0][0], bands[-1][1]
        span = high - low
        for fraction in (0.0, 0.25, 0.5, 0.75, 0.999):
            value = low + fraction * span
            assert entry.reading(value) != "outside the expected range", f"{name} at {value}"


def test_every_formula_records_the_numbers_behind_its_value():
    """Each formula leaves a trace of its own arithmetic for the current window."""
    scenario = synthetic.scenario("BULL")
    engine = FormulaEngine(brain=self_test_module._brain_stub())
    result = None
    for snapshot in scenario.snapshots[-3:]:
        result = engine.run(snapshot, "BTC")
    assert result is not None

    missing = [
        name for name in ALL_NAMES if len(result.traces.get(name, [])) < 2
    ]
    assert not missing, f"no trace recorded for {missing}"
    for name in ALL_NAMES:
        for item in result.traces[name]:
            assert item["label"], name
            assert "value" in item
    # ... and the readings the dashboard shows come from the same registry
    # (``_hsi`` is the engine's internal hand-off key for the brain, not a node)
    readings = logic_module.readings(result.values)
    assert set(readings) == {name for name in result.values if not name.startswith("_")}


@pytest.mark.parametrize("scenario_key", ["BULL", "BEAR", "STRESS"])
def test_the_tapes_are_stable_across_a_stream(scenario_key: str):
    """A tape with one character must not make a formula flip sign every window.

    The self-test reads the final window; this checks the window before it too,
    so a formula that alternates sign cannot pass by luck.  The balanced tape is
    deliberately excluded: its flow has no bias at all, so a 6-second pressure
    ratio (AFPR) really does oscillate around zero there - the level check in
    the self-test is what covers that tape.
    """
    scenario = synthetic.scenario(scenario_key)
    engine = FormulaEngine(brain=self_test_module._brain_stub())
    previous = None
    for snapshot in scenario.snapshots[-8:]:
        result = engine.run(snapshot, "BTC")
        if previous is not None:
            # Only a *meaningful* reading reversing counts: a formula that is
            # sitting on zero (AFPR on the balanced tape, where the drift really
            # is zero) is allowed to wobble across the axis.
            flipped = [
                name
                for name in ("AFPR", "BAR", "DGW", "LCS", "VSD", "NIV")
                if result.values.get(name, 0.0) * previous.get(name, 0.0) < -0.01
                and abs(result.values.get(name, 0.0)) > 0.05
                and abs(previous.get(name, 0.0)) > 0.05
            ]
            assert not flipped, f"{scenario_key}: {flipped} flipped between windows"
        previous = dict(result.values)
