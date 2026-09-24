"""Formula self-test: does every formula do what it says it does?

Each of the 22 formulas is a claim ("positive output when buyers are aggressive",
"detects a volume shock", "reads the hedge stress").  A claim like that is
testable: build a tape whose character is known by construction (see
:mod:`backend.formulas.synthetic`), run the engine over a *stream* of windows so
the stateful filters warm up, and read the sign.

Each expectation names the two tapes it is judged on (``up`` / ``down``): a
phase estimator is judged on the tape that carries a micro-cycle, an
acceleration formula on the tape that carries an impulse, a hedge formula on the
tape where the pair breaks down.  Every directional formula is *also* judged on
the flat tape, which is the anti-"random numbers" rule: a tape with no feature in
it must produce no signal.

What the report gives the user:

* ``PASS`` - the formula responded to the feature it claims to detect;
* ``FAIL`` - it did not, with the expected number next to the actual one;
* ``observed`` - every tape's value, so the verdict is auditable.

The self-test is served by ``GET /api/formulas/self-test`` and the Formula
Explorer shows the verdict next to each formula.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from backend.formulas import synthetic
from backend.formulas.engine import ALL_FORMULAS, FormulaEngine

log = logging.getLogger("drosophila.formulas.selftest")

#: Where the committed 80x80 fallback matrix lives - the self-test must work
#: with no network, exactly like the app does in fallback mode.
FALLBACK_MATRIX = Path(__file__).resolve().parents[1] / "brain" / "fallback" / "mb_adjacency_80x80.csv"


@dataclass(frozen=True)
class Expectation:
    """One testable claim about one formula."""

    name: str
    claim: str
    kind: str = "sign"  # sign | range | stress | regime | flat
    threshold: float = 0.15
    flat_max: float | None = 0.25
    lo: float | None = None
    hi: float | None = None
    margin: float = 0.15
    #: The tapes the claim is judged on.  ``up`` carries the bullish version of
    #: the feature, ``down`` the mirror image, ``flat`` the tape where the
    #: formula must stay quiet.
    up: str = "BULL"
    down: str = "BEAR"
    flat: str = "FLAT"


IMPULSE = ("IMPULSE_UP", "IMPULSE_DOWN")
CYCLE = ("CYCLE_UP", "CYCLE_DOWN")

EXPECTATIONS: tuple[Expectation, ...] = (
    Expectation("TAI", "positive on an accelerating tape, negative on the mirror",
                threshold=0.15, flat_max=0.35, up=IMPULSE[0], down=IMPULSE[1]),
    Expectation("AFPR", "aggressive buyers dominate a rising tape",
                threshold=0.4, flat_max=0.35),
    Expectation("SED", "makers widen into buying on the bull tape",
                threshold=0.10, flat_max=0.35),
    Expectation("VSD", "positive on the buy-side volume shock, negative on the sell side",
                threshold=0.3, flat_max=0.35, up=IMPULSE[0], down=IMPULSE[1]),
    Expectation("DGW", "positive when the ask ladder is the empty side",
                threshold=0.10, flat_max=0.35),
    Expectation("LCS", "positive when the ask ladder is the broken one",
                threshold=0.10, flat_max=0.30),
    Expectation("BAR", "absorption happens on the side being eaten",
                threshold=0.20, flat_max=0.60),
    Expectation("HRDD", "large when the hedge ratio drifts, quiet when it holds",
                kind="stress", threshold=0.25, margin=0.35),
    Expectation("SHRP", "flow rotating toward the traded leg",
                threshold=0.10, flat_max=0.60),
    Expectation("GCDV", "BTC outpacing PAXG on the bull tape",
                threshold=0.10, flat_max=0.35),
    Expectation("HSI", "hedge breakdown on the divergent pair, healthy otherwise",
                kind="stress", threshold=0.75, margin=0.15),
    Expectation("RSV", "regime-switch velocity has a direction",
                threshold=0.05, flat_max=0.60),
    Expectation("VSS", "volatility expansion signed by the drift",
                threshold=0.10, flat_max=0.35),
    Expectation("ERC", "a trend is more structured than chop",
                kind="regime", margin=0.05),
    Expectation("MCPE", "the micro-cycle has a phase with a sign",
                threshold=0.20, flat_max=0.40, up=CYCLE[0], down=CYCLE[1]),
    Expectation("MPS", "persistence signed by the direction of the move",
                threshold=0.10, flat_max=0.40),
    Expectation("TWRS", "skew follows the tape's tail",
                threshold=0.05, flat_max=0.35),
    Expectation("DSKD", "the fast Kalman estimate leads in the direction of the move",
                threshold=0.10, flat_max=0.35, up=IMPULSE[0], down=IMPULSE[1]),
    Expectation("NIV", "headline sentiment is signed",
                threshold=0.3, flat_max=0.20),
    Expectation("SMD", "news minus tape divergence is signed",
                threshold=0.10, flat_max=0.60),
    Expectation("KCAE", "a confidence term inside [0, 1]",
                kind="range", lo=0.0, hi=1.0),
    Expectation("CCSv2", "the whole circuit votes with the tape",
                threshold=0.15, flat_max=0.45),
    Expectation("DRG", "a reward signal inside [-1, 1]",
                kind="range", lo=-1.0, hi=1.0),
)


@dataclass
class FormulaVerdict:
    name: str
    claim: str
    passed: bool
    detail: str
    observed: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "claim": self.claim,
            "passed": bool(self.passed),
            "detail": self.detail,
            "observed": {k: round(v, 6) for k, v in self.observed.items()},
        }


@dataclass
class SelfTestReport:
    verdicts: list[FormulaVerdict] = field(default_factory=list)
    scenarios: dict[str, str] = field(default_factory=dict)
    windows: int = 0

    @property
    def passed(self) -> bool:
        return all(v.passed for v in self.verdicts)

    @property
    def failures(self) -> list[str]:
        return [v.name for v in self.verdicts if not v.passed]

    def to_dict(self) -> dict:
        return {
            "passed": self.passed,
            "checked": len(self.verdicts),
            "failed": self.failures,
            "windows_per_scenario": self.windows,
            "scenarios": self.scenarios,
            "formulas": [v.to_dict() for v in self.verdicts],
        }


# ---------------------------------------------------------------------------
# running the tapes
# ---------------------------------------------------------------------------
def _brain_stub():
    """The committed fallback matrix, wrapped like the app's Brain object."""
    from backend.brain import graph_convolution as gc
    from backend.brain import matrix_builder as mb

    class _Stub:
        def __init__(self, matrix):
            self.conv = gc.GraphConvolution(matrix)

    return _Stub(mb.load_csv(FALLBACK_MATRIX))


def run_scenario(key: str, windows: int | None = None) -> dict[str, float]:
    """Run one tape through a **fresh** engine and return the final window's values.

    A fresh engine matters: formulas carry per-(formula, asset) state, and
    reusing an engine across tapes silently feeds one scenario's history into
    the next one's Kalman filter or EMA baseline.
    """
    scenario = synthetic.scenario(key)
    snapshots = scenario.snapshots if windows is None else scenario.snapshots[-windows:]
    engine = FormulaEngine(brain=_brain_stub())
    values: dict[str, float] = {}
    for snapshot in snapshots:
        result = engine.run(snapshot, "BTC")
        values = dict(result.values)
        if result.errors:
            log.debug("self-test %s errors: %s", key, result.errors)
    return values


def _evaluate(expectation: Expectation, observed: dict[str, float]) -> FormulaVerdict:
    up = observed.get(expectation.up, 0.0)
    down = observed.get(expectation.down, 0.0)
    flat = observed.get(expectation.flat, 0.0)
    stress = observed.get("STRESS", 0.0)
    shown = {key: round(float(value), 6) for key, value in observed.items()}
    t = expectation.threshold
    where = f"{expectation.up} / {expectation.down}"

    if expectation.kind == "sign":
        ok = up >= t and down <= -t
        detail = f"on {where}: expected >= {t:+.2f} and <= {-t:+.2f}"
        if ok and expectation.flat_max is not None:
            ok = abs(flat) <= expectation.flat_max
            detail += f"; on {expectation.flat}: |value| <= {expectation.flat_max:.2f}"
        return FormulaVerdict(expectation.name, expectation.claim, ok, detail, shown)

    if expectation.kind == "range":
        lo, hi = expectation.lo, expectation.hi
        ok = all(lo <= value <= hi for value in observed.values())
        return FormulaVerdict(
            expectation.name, expectation.claim, ok,
            f"expected every tape inside [{lo}, {hi}]", shown,
        )

    if expectation.kind == "stress":
        calm = [v for k, v in observed.items() if k in ("BULL", "BEAR", "FLAT")]
        ok = stress >= expectation.threshold and stress - max(calm or [0.0]) >= expectation.margin
        return FormulaVerdict(
            expectation.name, expectation.claim, ok,
            f"expected STRESS >= {expectation.threshold:.2f} and at least "
            f"{expectation.margin:.2f} above the calm tapes",
            shown,
        )

    if expectation.kind == "regime":
        ok = flat > up + expectation.margin
        return FormulaVerdict(
            expectation.name, expectation.claim, ok,
            f"expected the choppy tape {expectation.flat} above {expectation.up} "
            f"by {expectation.margin:.2f}",
            shown,
        )

    if expectation.kind == "flat":
        ok = abs(flat) <= (expectation.flat_max or 0.25)
        return FormulaVerdict(
            expectation.name, expectation.claim, ok,
            f"expected |{expectation.flat}| <= {expectation.flat_max}", shown,
        )

    return FormulaVerdict(expectation.name, expectation.claim, False, "unknown check", shown)


def run(windows: int = 40) -> SelfTestReport:
    """Run every tape, then score every formula on the tapes it claims about."""
    observed_by_scenario: dict[str, dict[str, float]] = {}
    report = SelfTestReport(windows=windows)
    for key in synthetic.SCENARIOS:
        scenario = synthetic.scenario(key)
        report.scenarios[key] = scenario.description
        observed_by_scenario[key] = run_scenario(key, windows=windows)

    names = [spec.name for spec in ALL_FORMULAS] + ["DRG"]
    by_name = {e.name: e for e in EXPECTATIONS}
    for name in names:
        expectation = by_name.get(name)
        if expectation is None:
            continue
        observed = {key: values.get(name, 0.0) for key, values in observed_by_scenario.items()}
        report.verdicts.append(_evaluate(expectation, observed))
    return report


_cached: SelfTestReport | None = None


def cached_report(refresh: bool = False) -> SelfTestReport:
    global _cached
    if _cached is None or refresh:
        _cached = run()
    return _cached


def summary_line(report: SelfTestReport | None = None) -> str:
    report = report or cached_report()
    total = len(report.verdicts)
    failed = report.failures
    if not failed:
        return f"all {total} formulas responded correctly to the known tapes"
    return f"{total - len(failed)}/{total} formulas verified; failing: {', '.join(failed)}"


if __name__ == "__main__":  # pragma: no cover - manual run
    import json
    import sys

    logging.basicConfig(level=logging.WARNING)
    report = run()
    print(json.dumps(report.to_dict(), indent=2))
    sys.exit(0 if report.passed else 1)
