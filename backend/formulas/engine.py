"""The formula engine: runs all 22 formulas on one frozen snapshot.

Design rules that come straight from the specification:

* Every formula sees **only** the frozen snapshot - never a live buffer.  This is
  what makes the Signal Lock Protocol possible.
* Every formula is isolated: an exception inside one formula sets its value to
  0.0, records the error, and lets the other 21 continue.  If 11 or more of the
  20 input formulas are zero the caller forces HOLD (Section 10.1).
* Every formula is timed, and the per-formula latency is reported so the
  "~3 ms total" budget can be verified in production rather than assumed.
* Formula 22 needs the Kenyon Cell activations that Formula 21 measures, so the
  engine runs 1-20, then the brain pass, then 21, then 22.
"""

from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass, field

import numpy as np

from backend.core import config as cfg
from backend.core.micro import analyze as micro_analyze
from backend.core.micro import MicroState
from backend.core.timebase import format_us, perf_us
from backend.formulas import double_check as dc
from backend.formulas import drg as drg_module
from backend.formulas.category_a_microstructure import afpr, sed, tai, vsd
from backend.formulas.category_b_orderbook import bar, dgw, lcs
from backend.formulas.category_c_hedge import gcdv, hrdd, hsi, shrp
from backend.formulas.category_d_regime import erc, rsv, vss
from backend.formulas.category_e_temporal import mcpe, mps, twrs
from backend.formulas.category_f_kalman import dskd
from backend.formulas.category_g_news import niv, smd
from backend.formulas.category_h_brain import ccsv2, kcae

log = logging.getLogger("drosophila.formulas")


@dataclass(frozen=True)
class FormulaSpec:
    name: str
    module: object
    category: str
    order: int

    @property
    def title(self) -> str:
        return getattr(self.module, "TITLE", self.name)

    @property
    def brain_node(self) -> str:
        return getattr(self.module, "BRAIN_NODE", "")

    @property
    def latency_ms(self) -> float:
        return float(getattr(self.module, "LATENCY_MS", 0.0))

    @property
    def directional(self) -> bool:
        return bool(getattr(self.module, "DIRECTIONAL", True))

    @property
    def description(self) -> str:
        return str(getattr(self.module, "DESCRIPTION", ""))

    def to_dict(self) -> dict:
        return {
            "index": self.order,
            "name": self.name,
            "title": self.title,
            "category": self.category,
            "brain_node": self.brain_node,
            "latency_ms": self.latency_ms,
            "directional": self.directional,
            "description": self.description,
        }


CATEGORY_NAMES = {
    "A": "Micro-Structure",
    "B": "Order Book",
    "C": "Hedge (BTC x PAXG)",
    "D": "Volatility & Regime",
    "E": "Temporal Pattern",
    "F": "Kalman & State Estimation",
    "G": "News & Sentiment",
    "H": "Drosophila Brain Output",
    "REWARD": "Reward Learning",
}

#: Formulas 1-20 - the ones that feed the projection neurons, in PN order.
INPUT_FORMULAS: tuple[FormulaSpec, ...] = (
    FormulaSpec("TAI", tai, "A", 1),
    FormulaSpec("AFPR", afpr, "A", 2),
    FormulaSpec("SED", sed, "A", 3),
    FormulaSpec("VSD", vsd, "A", 4),
    FormulaSpec("DGW", dgw, "B", 5),
    FormulaSpec("LCS", lcs, "B", 6),
    FormulaSpec("BAR", bar, "B", 7),
    FormulaSpec("HRDD", hrdd, "C", 8),
    FormulaSpec("SHRP", shrp, "C", 9),
    FormulaSpec("GCDV", gcdv, "C", 10),
    FormulaSpec("HSI", hsi, "C", 11),
    FormulaSpec("RSV", rsv, "D", 12),
    FormulaSpec("VSS", vss, "D", 13),
    FormulaSpec("ERC", erc, "D", 14),
    FormulaSpec("MCPE", mcpe, "E", 15),
    FormulaSpec("MPS", mps, "E", 16),
    FormulaSpec("TWRS", twrs, "E", 17),
    FormulaSpec("DSKD", dskd, "F", 18),
    FormulaSpec("NIV", niv, "G", 19),
    FormulaSpec("SMD", smd, "G", 20),
)

BRAIN_FORMULAS: tuple[FormulaSpec, ...] = (
    FormulaSpec("KCAE", kcae, "H", 21),
    FormulaSpec("CCSv2", ccsv2, "H", 22),
)

ALL_FORMULAS: tuple[FormulaSpec, ...] = INPUT_FORMULAS + BRAIN_FORMULAS


def logic_entry(name: str):
    """The logic registry entry for a formula, or None (never raises)."""
    try:
        from backend.formulas import logic as logic_module

        return logic_module.get(name)
    except Exception:  # pragma: no cover - defensive
        return None


def _replay_ctx(ctx: dict) -> dict:
    """A shallow copy of the formula context for the replay: the second pass
    may read everything the first one saw but must not write into it."""
    replay = dict(ctx)
    replay["_trace"] = []
    return replay


def _tick_times(engine, result: FormulaResult, asset: str):
    """The tick timestamps of the last frozen snapshot, for the span note."""
    snapshot = getattr(engine, "last_snapshot", None)
    if snapshot is None:
        return np.zeros(0)
    try:
        ticks = snapshot.ticks(asset)
    except Exception:  # noqa: BLE001 - never let tracing break a pass
        return np.zeros(0)
    return ticks[:, 0] if ticks is not None and len(ticks) else np.zeros(0)


def _readings(values: dict) -> dict:
    """Readings lookup that never raises - the UI must always get a payload."""
    try:
        from backend.formulas import logic as _logic

        return _logic.readings(values)
    except Exception:  # pragma: no cover - defensive
        return {}


#: How many samples of each formula the engine remembers, per asset.
#: 360 windows is six minutes at the 60-second cadence - exactly the 6x the
#: dashboard used to show - and it is what makes the per-formula statistics
#: (mean, sigma, z-score, trend, zero rate) meaningful instead of decorative.
FORMULA_HISTORY = 360


@dataclass
class FormulaStats:
    """What one formula has been reading, and where this window sits in it."""

    samples: int = 0
    mean: float = 0.0
    std: float = 0.0
    minimum: float = 0.0
    maximum: float = 0.0
    last: float = 0.0
    zscore: float = 0.0
    nonzero_rate: float = 0.0
    trend: float = 0.0
    percentile: float = 0.0

    def to_dict(self) -> dict:
        return {
            "samples": self.samples,
            "mean": round(self.mean, 6),
            "std": round(self.std, 6),
            "min": round(self.minimum, 6),
            "max": round(self.maximum, 6),
            "last": round(self.last, 6),
            "zscore": round(self.zscore, 4),
            "nonzero_rate": round(self.nonzero_rate, 4),
            "trend": round(self.trend, 6),
            "percentile": round(self.percentile, 4),
        }


@dataclass
class FormulaResult:
    """The output of one full formula pass."""

    values: dict[str, float] = field(default_factory=dict)
    errors: dict[str, str] = field(default_factory=dict)
    timings_ms: dict[str, float] = field(default_factory=dict)
    #: The same timings at the resolution the engine actually works at.  A
    #: formula costs tens of microseconds, so ``timings_ms`` rounds most of them
    #: to 0.03 or 0.00; these are exact.
    timings_us: dict[str, int] = field(default_factory=dict)
    #: Per-formula statistics over the retained history (mean, sigma, z-score,
    #: trend, zero rate, percentile) - the "is this number normal?" context.
    stats: dict[str, dict] = field(default_factory=dict)
    #: ``{formula: [{"label", "value", "unit"}, ...]}`` - the intermediate
    #: numbers each formula recorded while it ran.  This is what turns the
    #: Formula Explorer from a list of numbers into an audit trail.
    traces: dict[str, list] = field(default_factory=dict)
    #: ``{formula: {ok, verdict, replay, rederived, range_ok, ...}}`` - the
    #: double check every formula passed (or failed) on this pass.
    checks: dict[str, dict] = field(default_factory=dict)
    #: Microseconds spent double-checking (kept apart from the formula timings).
    check_us: int = 0
    ccs_confidence: float = 0.0
    kcae: float = 0.0
    brain_trace: dict = field(default_factory=dict)
    total_ms: float = 0.0
    total_us: int = 0
    asset: str = "BTC"
    timestamp: float = 0.0
    fatal: str = ""
    #: The microsecond picture of the tape this pass was computed on.
    micro: dict = field(default_factory=dict)
    #: How many past windows the statistics above are computed over.
    history_window: int = 0

    def check_summary(self) -> dict:
        verdicts = [c.get("verdict") for c in self.checks.values()]
        return {
            "enabled": dc.enabled(),
            "formulas": len(verdicts),
            "verified": sum(1 for v in verdicts if v == "verified"),
            "partial": sum(1 for v in verdicts if v == "partial"),
            "failed": sum(1 for v in verdicts if v == "failed"),
            "failed_names": [k for k, c in self.checks.items() if c.get("verdict") == "failed"],
            "check_us": int(self.check_us),
            "rule": (
                "every formula is run twice (replay on a private copy of its state), "
                "re-derived from its own traced intermediates and range-checked; "
                "a formula that fails any of the three is zeroed for the pass"
            ),
        }

    @property
    def zero_count(self) -> int:
        return sum(1 for spec in INPUT_FORMULAS if abs(self.values.get(spec.name, 0.0)) < 1e-12)

    @property
    def failed_count(self) -> int:
        return len(self.errors)

    @property
    def insufficient_evidence(self) -> bool:
        """Section 10.1: >= 11 of the 20 input formulas are zero -> forced HOLD."""
        return self.zero_count >= (cfg.SETTINGS.max_failed_formulas + 1)

    def to_dict(self) -> dict:
        return {
            "asset": self.asset,
            "timestamp": self.timestamp,
            "formulas": {k: round(v, 6) for k, v in self.values.items()},
            "errors": self.errors,
            # The one-line interpretation of every value above, straight from
            # the logic registry: the UI prints this instead of making the user
            # read the raw number.
            "readings": _readings(self.values),
            "timings_ms": {k: round(v, 4) for k, v in self.timings_ms.items()},
            "timings_us": {k: int(v) for k, v in self.timings_us.items()},
            "stats": {k: v for k, v in self.stats.items()},
            "micro": dict(self.micro),
            "traces": {k: v for k, v in self.traces.items()},
            "checks": {k: v for k, v in self.checks.items()},
            "double_check": self.check_summary(),
            "zero_count": self.zero_count,
            "failed_count": self.failed_count,
            "ccs_confidence": round(self.ccs_confidence, 6),
            "kcae": round(self.kcae, 6),
            "brain_trace": self.brain_trace,
            "total_ms": round(self.total_ms, 4),
            "total_us": int(self.total_us),
            "history_window": FORMULA_HISTORY,
        }


class FormulaEngine:
    """Owns the per-formula state and executes the full pass."""

    def __init__(self, brain=None) -> None:
        self.brain = brain
        self._state: dict[tuple[str, str], object] = {}
        #: Rolling sample history per (formula, asset): the context that turns a
        #: bare number into "normal for this tape, or not".
        self._history: dict[tuple[str, str], deque[float]] = {}
        self._micro = MicroState()
        self.runs = 0
        self.last: FormulaResult | None = None

    # ------------------------------------------------------------------
    def history(self, name: str, asset: str) -> deque[float]:
        key = (name, asset.upper())
        series = self._history.get(key)
        if series is None:
            series = deque(maxlen=FORMULA_HISTORY)
            self._history[key] = series
        return series

    def _record(self, name: str, asset: str, value: float) -> FormulaStats:
        """Append the value and describe where it sits in its own history."""
        series = self.history(name, asset)
        series.append(float(value))
        samples = np.fromiter(series, dtype=float, count=len(series))
        stats = FormulaStats(samples=int(samples.size), last=round(float(value), 9))
        if samples.size:
            stats.mean = float(np.mean(samples))
            stats.std = float(np.std(samples))
            stats.minimum = float(np.min(samples))
            stats.maximum = float(np.max(samples))
            stats.nonzero_rate = float(np.count_nonzero(np.abs(samples) > 1e-12)) / samples.size
            if stats.std > 0:
                stats.zscore = float((value - stats.mean) / stats.std)
            stats.percentile = float(np.count_nonzero(samples <= value)) / samples.size
            if samples.size >= 3:
                recent = samples[-20:]
                index = np.arange(recent.size, dtype=float)
                try:
                    slope = float(np.polyfit(index, recent, 1)[0])
                except Exception:  # noqa: BLE001 - degenerate series
                    slope = 0.0
                stats.trend = slope
        return stats

    # ------------------------------------------------------------------
    def state_for(self, spec: FormulaSpec, asset: str):
        key = (spec.name, asset.upper())
        state = self._state.get(key)
        if state is None:
            state = spec.module.State()
            self._state[key] = state
        return state

    def drg_state(self, asset: str) -> drg_module.State:
        key = ("DRG", asset.upper())
        state = self._state.get(key)
        if state is None:
            state = drg_module.State()
            self._state[key] = state
        return state

    # ------------------------------------------------------------------
    def run(self, snapshot, asset: str) -> FormulaResult:
        """Execute all 22 formulas + DRG against ``snapshot``."""
        asset = asset.upper()
        started_us = perf_us()
        params = cfg.asset_params(asset)
        result = FormulaResult(asset=asset, timestamp=snapshot.timestamp)

        ctx: dict = {"_brain": self.brain, "asset": asset}

        # --- Reward meta-parameter first: DRG gates the dopamine nodes -----
        ctx["_trace"] = []
        drg_us = perf_us()
        drg_state = self.drg_state(asset)
        drg_copy = dc.copy_state(drg_state) if dc.enabled() else None
        try:
            value = drg_module.compute(snapshot, drg_state, params, ctx=ctx)
            result.values["DRG"] = float(value)
            ctx["_drg"] = float(value)
        except Exception as exc:  # noqa: BLE001
            result.errors["DRG"] = str(exc)
            result.values["DRG"] = 0.0
            ctx["_drg"] = 0.0
        finally:
            result.timings_us["DRG"] = perf_us() - drg_us
            result.traces["DRG"] = ctx.pop("_trace", [])
        if "DRG" not in result.errors:
            self._double_check(
                "DRG", drg_module, snapshot, asset, params, ctx, result,
                replay=(lambda: drg_module.compute(snapshot, drg_copy, params, ctx=_replay_ctx(ctx)))
                if drg_copy is not None else None,
            )
            ctx["_drg"] = result.values["DRG"]

        # --- Formulas 1-20 ------------------------------------------------
        for spec in INPUT_FORMULAS:
            self._run_one(spec, snapshot, asset, params, ctx, result)

        # --- Brain pass: KC activations must exist before KCAE -------------
        ccs_spec = BRAIN_FORMULAS[1]
        ccs_state = self.state_for(ccs_spec, asset)
        try:
            t0 = time.perf_counter()
            t0_us = perf_us()
            ccsv2.prepare(snapshot, asset, ccs_state, params, ctx)
            result.timings_ms["CCSv2:prepare"] = (time.perf_counter() - t0) * 1000.0
            result.timings_us["CCSv2:prepare"] = perf_us() - t0_us
        except Exception as exc:  # noqa: BLE001
            result.errors["CCSv2"] = f"prepare failed: {exc}"
            log.warning("CCSv2 prepare failed: %s", exc)

        # --- Formula 21 (KCAE), then 22 (CCSv2) ---------------------------
        self._run_one(BRAIN_FORMULAS[0], snapshot, asset, params, ctx, result)
        self._run_one(ccs_spec, snapshot, asset, params, ctx, result)

        result.ccs_confidence = float(ctx.get("_ccsv2_confidence", 0.0))
        result.kcae = float(ctx.get("KCAE", 0.0))
        result.brain_trace = dict(ctx.get("_brain_trace", {}))
        result.total_us = perf_us() - started_us
        result.total_ms = result.total_us / 1000.0
        result.history_window = FORMULA_HISTORY
        result.values.setdefault("_hsi", result.values.get("HSI", 0.0))

        # --- the microsecond picture of the tape this pass ran on -----------
        try:
            report = micro_analyze(snapshot, asset, state=self._micro)
            result.micro = report.to_dict()
        except Exception as exc:  # noqa: BLE001 - analysis must never break a pass
            log.debug("micro analysis failed: %s", exc)
            result.micro = {"available": False, "reason": str(exc)}

        # --- per-formula history and statistics ----------------------------
        for name, value in result.values.items():
            if name.startswith("_"):
                continue
            try:
                stats = self._record(name, asset, float(value))
                result.stats[name] = stats.to_dict()
            except Exception as exc:  # noqa: BLE001
                log.debug("stats for %s failed: %s", name, exc)
        self._annotate_traces(result, asset)

        self.runs += 1
        self.last = result
        return result

    # ------------------------------------------------------------------
    def _run_one(
        self,
        spec: FormulaSpec,
        snapshot,
        asset: str,
        params: dict,
        ctx: dict,
        result: FormulaResult,
    ) -> None:
        t0 = time.perf_counter()
        t0_us = perf_us()
        ctx["_trace"] = []
        state = self.state_for(spec, asset)
        # The replay runs on a private copy taken *before* the first pass, so
        # both passes see the same state and the same frozen snapshot.
        state_copy = dc.copy_state(state) if dc.enabled() else None
        try:
            value = spec.module.compute(snapshot, asset, state, params, ctx)
            value = float(value)
            if not np.isfinite(value):
                raise ValueError("non-finite formula output")
            result.values[spec.name] = value
            ctx[spec.name] = value
        except Exception as exc:  # noqa: BLE001 - one formula must never kill the pass
            result.values[spec.name] = 0.0
            ctx[spec.name] = 0.0
            result.errors[spec.name] = str(exc)
            log.warning("formula %s failed for %s: %s", spec.name, asset, exc)
        finally:
            result.timings_ms[spec.name] = (time.perf_counter() - t0) * 1000.0
            result.timings_us[spec.name] = perf_us() - t0_us
            result.traces[spec.name] = ctx.pop("_trace", [])

        if spec.name not in result.errors:
            self._double_check(
                spec.name, spec.module, snapshot, asset, params, ctx, result,
                replay=(lambda: spec.module.compute(snapshot, asset, state_copy, params, _replay_ctx(ctx)))
                if state_copy is not None else None,
            )
            ctx[spec.name] = result.values[spec.name]

            # Every formula records how many ticks it actually saw: the first
        # question when a value looks wrong is "did it have data?".
        result.traces[spec.name].insert(
            0,
            {
                "label": "ticks seen",
                "value": f"{int(snapshot.tick_count(asset))}",
                "unit": f"asset {asset}, window {int(snapshot.timestamp)}",
            },
        )
        try:
            from backend.formulas import logic as logic_module

            entry = logic_module.get(spec.name)
            if entry is not None:
                result.traces[spec.name].append(
                    {"label": "reading", "value": entry.reading(float(result.values[spec.name])), "unit": ""}
                )
        except Exception:  # noqa: BLE001
            pass

    # ------------------------------------------------------------------
    def _double_check(self, name: str, module, snapshot, asset: str, params: dict,
                      ctx: dict, result: FormulaResult, *, replay) -> None:
        """Replay + re-derive + range-check one formula; zero it if any fails."""
        if not dc.enabled():
            return
        started = perf_us()
        try:
            report = dc.verify(
                module, name, result.values[name], result.traces.get(name, []), asset, replay=replay
            )
        except Exception as exc:  # noqa: BLE001 - the check itself must never kill the pass
            log.debug("double check of %s crashed: %s", name, exc)
            return
        result.checks[name] = report.to_dict()
        result.traces.setdefault(name, []).append(report.trace_row())
        result.check_us += perf_us() - started
        if not report.ok:
            result.errors[name] = "double check failed: " + "; ".join(report.notes)
            result.values[name] = 0.0
            log.warning("formula %s failed its double check for %s: %s", name, asset, "; ".join(report.notes))

    # ------------------------------------------------------------------
    def _annotate_traces(self, result: FormulaResult, asset: str) -> None:
        """Add the microsecond, history and timing context to every trace.

        A formula records what *it* computed (4-8 intermediate numbers).  Those
        numbers are only trustworthy next to three more facts: how much data the
        pass actually saw, how long the formula took at microsecond resolution,
        and whether the value it produced is normal for this tape.  This is what
        turns the Formula Explorer from a list of results into an audit trail.
        """
        ticks = result.micro.get("ticks") if isinstance(result.micro, dict) else None
        for name, trace in result.traces.items():
            if not isinstance(trace, list):
                continue
            stats = result.stats.get(name, {})
            value = float(result.values.get(name, 0.0))
            entry = logic_entry(name)
            context = [
                {
                    "label": "value · "
                    + (entry.range_label() if entry is not None else "raw"),
                    "value": f"{value:+.6f}",
                    "unit": (entry.units if entry is not None else ""),
                },
                {
                    "label": "computed in",
                    "value": format_us(result.timings_us.get(name, 0)),
                    "unit": f"{result.timings_us.get(name, 0)} µs · resolution {result.micro.get('resolution_label', '—')}",
                },
                {
                    "label": "data window",
                    "value": (result.micro.get("span_label") or "—"),
                    "unit": f"{int(ticks or 0)} ticks analysed in microseconds",
                },
            ]
            if stats:
                context += [
                    {
                        "label": f"history over {stats.get('samples', 0)} windows",
                        "value": f"mean {stats.get('mean', 0):+.4f} · sigma {stats.get('std', 0):.4f}",
                        "unit": f"range {stats.get('min', 0):+.4f} … {stats.get('max', 0):+.4f}",
                    },
                    {
                        "label": "z-score vs own history",
                        "value": f"{stats.get('zscore', 0):+.2f}σ",
                        "unit": f"percentile {float(stats.get('percentile', 0)) * 100:.0f}% · "
                        f"non-zero {float(stats.get('nonzero_rate', 0)) * 100:.0f}% of windows",
                    },
                    {
                        "label": "trend over the last 20 windows",
                        "value": f"{stats.get('trend', 0):+.6f}",
                        "unit": "per window" + (" · rising" if stats.get("trend", 0) > 0 else " · falling" if stats.get("trend", 0) < 0 else ""),
                    },
                ]
            if entry is not None and entry.units:
                context.append(
                    {
                        "label": "units · sensitivity",
                        "value": entry.units,
                        "unit": entry.sensitivity,
                    }
                )
            trace.extend(context)

    # ------------------------------------------------------------------
    # Metadata / persistence
    # ------------------------------------------------------------------
    @staticmethod
    def metadata() -> dict:
        from backend.formulas import logic as logic_module

        by_category: dict[str, list[dict]] = {}
        for spec in ALL_FORMULAS:
            entry = spec.to_dict()
            logic = logic_module.get(spec.name)
            if logic is not None:
                entry["logic"] = logic.to_dict()
            by_category.setdefault(spec.category, []).append(entry)
        return {
            "categories": [
                {"key": key, "name": CATEGORY_NAMES.get(key, key), "formulas": formulas}
                for key, formulas in by_category.items()
            ],
            "count": len(ALL_FORMULAS),
            "reward": {
                "name": drg_module.NAME,
                "title": drg_module.TITLE,
                "description": drg_module.DESCRIPTION,
                "brain_node": drg_module.BRAIN_NODE,
                "latency_ms": drg_module.LATENCY_MS,
                "directional": False,
                "logic": logic_module.get("DRG").to_dict() if logic_module.get("DRG") else None,
            },
        }

    def export_state(self) -> dict:
        payload: dict = {}
        for (name, asset), state in self._state.items():
            serializer = getattr(state, "to_dict", None)
            if callable(serializer):
                payload[f"{name}:{asset}"] = serializer()
        return payload

    def import_state(self, payload: dict) -> int:
        """Restore persisted formula state (used by the Redis cache)."""
        restored = 0
        lookup = {spec.name: spec for spec in ALL_FORMULAS}
        lookup["DRG"] = FormulaSpec("DRG", drg_module, "REWARD", 0)
        for key, data in (payload or {}).items():
            if ":" not in key:
                continue
            name, _, asset = key.partition(":")
            spec = lookup.get(name)
            if spec is None:
                continue
            try:
                self._state[(name, asset)] = spec.module.State.from_dict(data)
                restored += 1
            except Exception:  # noqa: BLE001 - stale cache is not fatal
                continue
        return restored
