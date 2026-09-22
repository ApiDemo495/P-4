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
from dataclasses import dataclass, field

import numpy as np

from backend.core import config as cfg
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


@dataclass
class FormulaResult:
    """The output of one full formula pass."""

    values: dict[str, float] = field(default_factory=dict)
    errors: dict[str, str] = field(default_factory=dict)
    timings_ms: dict[str, float] = field(default_factory=dict)
    ccs_confidence: float = 0.0
    kcae: float = 0.0
    brain_trace: dict = field(default_factory=dict)
    total_ms: float = 0.0
    asset: str = "BTC"
    timestamp: float = 0.0
    fatal: str = ""

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

    def directional_values(self) -> dict[str, float]:
        return {
            spec.name: self.values.get(spec.name, 0.0)
            for spec in INPUT_FORMULAS
            if spec.directional
        }

    def to_dict(self) -> dict:
        return {
            "asset": self.asset,
            "timestamp": self.timestamp,
            "formulas": {k: round(v, 6) for k, v in self.values.items()},
            "errors": self.errors,
            "timings_ms": {k: round(v, 4) for k, v in self.timings_ms.items()},
            "zero_count": self.zero_count,
            "failed_count": self.failed_count,
            "ccs_confidence": round(self.ccs_confidence, 6),
            "kcae": round(self.kcae, 6),
            "brain_trace": self.brain_trace,
            "total_ms": round(self.total_ms, 4),
        }


class FormulaEngine:
    """Owns the per-formula state and executes the full pass."""

    def __init__(self, brain=None) -> None:
        self.brain = brain
        self._state: dict[tuple[str, str], object] = {}
        self.runs = 0
        self.last: FormulaResult | None = None

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
        started = time.perf_counter()
        params = cfg.asset_params(asset)
        result = FormulaResult(asset=asset, timestamp=snapshot.timestamp)

        ctx: dict = {"_brain": self.brain, "asset": asset}

        # --- Reward meta-parameter first: DRG gates the dopamine nodes -----
        try:
            drg_state = self.drg_state(asset)
            value = drg_module.compute(snapshot, drg_state, params)
            result.values["DRG"] = float(value)
            ctx["_drg"] = float(value)
        except Exception as exc:  # noqa: BLE001
            result.errors["DRG"] = str(exc)
            result.values["DRG"] = 0.0
            ctx["_drg"] = 0.0

        # --- Formulas 1-20 ------------------------------------------------
        for spec in INPUT_FORMULAS:
            self._run_one(spec, snapshot, asset, params, ctx, result)

        # --- Brain pass: KC activations must exist before KCAE -------------
        ccs_spec = BRAIN_FORMULAS[1]
        ccs_state = self.state_for(ccs_spec, asset)
        try:
            t0 = time.perf_counter()
            ccsv2.prepare(snapshot, asset, ccs_state, params, ctx)
            result.timings_ms["CCSv2:prepare"] = (time.perf_counter() - t0) * 1000.0
        except Exception as exc:  # noqa: BLE001
            result.errors["CCSv2"] = f"prepare failed: {exc}"
            log.warning("CCSv2 prepare failed: %s", exc)

        # --- Formula 21 (KCAE), then 22 (CCSv2) ---------------------------
        self._run_one(BRAIN_FORMULAS[0], snapshot, asset, params, ctx, result)
        self._run_one(ccs_spec, snapshot, asset, params, ctx, result)

        result.ccs_confidence = float(ctx.get("_ccsv2_confidence", 0.0))
        result.kcae = float(ctx.get("KCAE", 0.0))
        result.brain_trace = dict(ctx.get("_brain_trace", {}))
        result.total_ms = (time.perf_counter() - started) * 1000.0
        result.values.setdefault("_hsi", result.values.get("HSI", 0.0))

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
        try:
            state = self.state_for(spec, asset)
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

    # ------------------------------------------------------------------
    # Metadata / persistence
    # ------------------------------------------------------------------
    @staticmethod
    def metadata() -> dict:
        by_category: dict[str, list[dict]] = {}
        for spec in ALL_FORMULAS:
            by_category.setdefault(spec.category, []).append(spec.to_dict())
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
