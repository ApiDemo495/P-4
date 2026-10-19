"""The double check every formula gets on every pass.

A formula's number is only trusted after three independent checks agree:

1. **Replay** - the formula is run a second time on a private copy of its
   state and the same frozen snapshot.  A different answer means hidden
   state, a non-deterministic step or a mutation bug; the number is unsafe.
2. **Re-derivation** - every formula module carries ``double_check(trace,
   asset)``: an independent one-line re-computation of its output from the
   intermediate numbers it traced while running (``DOUBLE_CHECK`` states the
   rule in words).  If the printed intermediates do not reproduce the printed
   result, the audit trail is lying.
3. **Range** - finite and inside the formula's declared range (``RANGE``,
   default [-1, 1]).

A formula that fails any check is **zeroed for this pass** and the failure is
recorded in ``errors`` - a wrong number must never reach the fusion; a
missing one is handled by the degradation ladder (Section 10.1).  The
verdict, the deltas and the rule are appended to the formula's trace so the
Formula Explorer shows the double check next to the work.
"""

from __future__ import annotations

import copy
import math
import os
from dataclasses import dataclass, field

#: Replay must reproduce the value to floating-point noise.
REPLAY_TOLERANCE = 1e-9
#: Re-derivation uses the exact traced numbers, so it is just as tight; the
#: small absolute slack covers EPS-padded denominators.
REDERIVE_TOLERANCE = 1e-6
DEFAULT_RANGE = (-1.0, 1.0)


def enabled() -> bool:
    return os.environ.get("FORMULA_DOUBLE_CHECK", "1") not in ("0", "false", "no")


@dataclass
class CheckReport:
    name: str
    value: float
    replay_value: float | None = None
    replay_ok: bool | None = None
    rederived_value: float | None = None
    rederive_ok: bool | None = None
    range_ok: bool = True
    range: tuple[float, float] = DEFAULT_RANGE
    rule: str = ""
    notes: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.range_ok and self.replay_ok is not False and self.rederive_ok is not False

    @property
    def verdict(self) -> str:
        if not self.ok:
            return "failed"
        checks = sum(1 for flag in (self.replay_ok, self.rederive_ok) if flag) + 1
        return "verified" if checks >= 3 else "partial"

    def to_dict(self) -> dict:
        return {
            "ok": self.ok,
            "verdict": self.verdict,
            "value": round(self.value, 9),
            "replay": self.replay_ok,
            "replay_value": None if self.replay_value is None else round(self.replay_value, 9),
            "replay_delta": None if self.replay_value is None else abs(self.replay_value - self.value),
            "rederived": self.rederive_ok,
            "rederived_value": None if self.rederived_value is None else round(self.rederived_value, 9),
            "rederived_delta": None if self.rederived_value is None else abs(self.rederived_value - self.value),
            "range_ok": self.range_ok,
            "range": list(self.range),
            "rule": self.rule,
            "notes": list(self.notes),
        }

    def trace_row(self) -> dict:
        parts = []
        if self.replay_ok is not None:
            parts.append("replay " + ("✓" if self.replay_ok else f"✗ ({self.replay_value:+.6g})"))
        if self.rederive_ok is not None:
            parts.append("re-derived " + ("✓" if self.rederive_ok else f"✗ ({self.rederived_value:+.6g})"))
        parts.append("range " + ("✓" if self.range_ok else "✗"))
        icon = "✓✓" if self.verdict == "verified" else "✓" if self.verdict == "partial" else "✗"
        return {
            "label": "double check",
            "value": f"{icon} {self.verdict}: " + " · ".join(parts),
            "unit": self.rule,
        }


def raw_trace(rows: list) -> dict:
    """``{label: raw number}`` from the trace rows (last occurrence wins)."""
    out: dict = {}
    for row in rows or ():
        if isinstance(row, dict) and "raw" in row:
            out[str(row.get("label"))] = float(row["raw"])
    return out


def _close(a: float, b: float, tol: float) -> bool:
    return abs(a - b) <= tol + tol * max(abs(a), abs(b))


def verify(
    module,
    name: str,
    value: float,
    trace_rows: list,
    asset: str,
    *,
    replay=None,
) -> CheckReport:
    """Run the three checks.  ``replay`` is a zero-argument callable that
    recomputes the formula on a private copy of its state (or None)."""
    report = CheckReport(name=name, value=float(value), rule=str(getattr(module, "DOUBLE_CHECK", "")))
    lo, hi = getattr(module, "RANGE", DEFAULT_RANGE)
    report.range = (float(lo), float(hi))

    # 3. range / finiteness
    if not math.isfinite(report.value):
        report.range_ok = False
        report.notes.append("non-finite output")
    elif report.value < lo - 1e-9 or report.value > hi + 1e-9:
        report.range_ok = False
        report.notes.append(f"{report.value:+.6g} outside [{lo:g}, {hi:g}]")

    # 1. replay on a private copy of the state
    if replay is not None:
        try:
            second = float(replay())
            report.replay_value = second
            report.replay_ok = math.isfinite(second) and _close(second, report.value, REPLAY_TOLERANCE)
            if not report.replay_ok:
                report.notes.append(f"replay gave {second:+.9g}, first pass {report.value:+.9g}")
        except Exception as exc:  # noqa: BLE001 - a crashing replay is a failed check
            report.replay_ok = False
            report.notes.append(f"replay raised {exc}")

    # 2. independent re-derivation from the traced intermediates
    check = getattr(module, "double_check", None)
    if callable(check):
        try:
            expected = check(raw_trace(trace_rows), asset)
            if expected is None:
                report.rederive_ok = None
            else:
                expected = float(expected)
                report.rederived_value = expected
                report.rederive_ok = _close(expected, report.value, REDERIVE_TOLERANCE)
                if not report.rederive_ok:
                    report.notes.append(
                        f"intermediates re-derive to {expected:+.9g}, formula returned {report.value:+.9g}"
                    )
        except Exception as exc:  # noqa: BLE001
            report.rederive_ok = False
            report.notes.append(f"re-derivation raised {exc}")
    return report


def copy_state(state):
    """A private copy of a formula state for the replay (never raises)."""
    try:
        return copy.deepcopy(state)
    except Exception:  # noqa: BLE001
        return None
