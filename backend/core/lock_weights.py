"""Round AA lock weighting - ONE table that says how much each voter weighs.

Before this module the weights of the fused voters (brain, formulas,
physics, news, the three agents) were spread over `fusion.py` as literal
settings lookups and ad-hoc multipliers.  Now every lock is weighed from a
single table built here:

    final(v) = base(v) * reliability(v) * availability(v)

* **base** - the published prior of the source (settings `weight_*`).
* **reliability** - earned from the evidence ledger: each fusion voter is
  mapped to the ledger sources that back it (`formulas` <- every `f:*`,
  `drosophila` <- `brain:CCSv2`, ...).  The shrunk hit-rate of those sources
  (n / (n + 10) towards 0.5) becomes a multiplier in [0.4, 1.6]:
  a 50 % source keeps its base weight, a 70 % source weighs 1.4x, a source
  the ledger has caught being wrong weighs less than half.
* **availability** - how alive the input is this window (physics liveness,
  news mass, agent answered or not) - a quiet input cannot outweigh a loud
  one just because its prior is large.

The table is attached to every fusion result as ``lock_weights`` so the
dashboard can print exactly why a voter weighed what it weighed, and the
`+0.00` weights of the old display cannot happen: a voter that is absent is
listed as absent, not as zero.
"""
from __future__ import annotations

from dataclasses import dataclass, field

SHRINK_N = 10.0
MULT_FLOOR = 0.4
MULT_CEIL = 1.6

# fusion voter -> ledger source prefixes / names that back it
BACKERS = {
    "drosophila": ("brain:CCSv2",),
    "formulas": ("f:",),
    "physics": ("physics:layer",),
    "genesis": ("genesis:composite",),
    "news": ("news:NIV",),
    "gemini": ("agent:gemini",),
    "local": ("agent:local",),
    "github": ("agent:github",),
}


@dataclass
class LockWeight:
    voter: str
    base: float
    reliability: float = 1.0
    availability: float = 1.0
    hit_rate: float | None = None
    samples: float = 0.0
    present: bool = False

    @property
    def final(self) -> float:
        return self.base * self.reliability * self.availability if self.present else 0.0

    def as_dict(self) -> dict:
        return {
            "base": round(self.base, 4),
            "reliability": round(self.reliability, 3),
            "availability": round(self.availability, 3),
            "hit_rate": None if self.hit_rate is None else round(self.hit_rate, 3),
            "samples": round(self.samples, 1),
            "present": self.present,
            "final": round(self.final, 4),
        }


@dataclass
class LockWeightTable:
    rows: dict[str, LockWeight] = field(default_factory=dict)

    def mark(self, voter: str, availability: float = 1.0) -> float:
        row = self.rows[voter]
        row.present = True
        row.availability = max(0.0, min(1.0, float(availability)))
        return row.final

    def normalised(self) -> dict[str, float]:
        total = sum(r.final for r in self.rows.values() if r.present) or 1e-9
        return {k: r.final / total for k, r in self.rows.items() if r.present}

    def as_dict(self) -> dict:
        shares = self.normalised()
        out = {k: dict(r.as_dict(), share=round(shares.get(k, 0.0), 4)) for k, r in self.rows.items()}
        out["rule"] = (
            "weight = base prior x ledger reliability (shrunk hit rate -> x0.4..x1.6) x availability; "
            "shares renormalise over the voters present this window"
        )
        return out


def reliability_multiplier(hit_rate: float | None, samples: float) -> float:
    """Shrunk hit rate -> multiplier: 0.5 -> 1.0, 0.7 -> 1.4, 0.3 -> 0.6."""
    if hit_rate is None or samples <= 0:
        return 1.0
    shrink = samples / (samples + SHRINK_N)
    p = 0.5 + shrink * (float(hit_rate) - 0.5)
    return max(MULT_FLOOR, min(MULT_CEIL, 1.0 + 2.0 * (p - 0.5)))


def build(settings, ledger_sources: list[dict] | None = None) -> LockWeightTable:
    """Build the table from settings priors and the ledger's source rows."""
    bases = {
        "drosophila": float(getattr(settings, "weight_drosophila", 0.4) or 0.0),
        "formulas": float(getattr(settings, "weight_formulas", 0.0) or 0.0),
        "physics": float(getattr(settings, "weight_physics", 0.0) or 0.0),
        "genesis": float(getattr(settings, "weight_genesis", 0.0) or 0.0),
        "news": float(getattr(settings, "weight_news", 0.0) or 0.0),
        "gemini": float(getattr(settings, "weight_gemini", 0.0) or 0.0),
        "local": float(getattr(settings, "weight_local", 0.0) or 0.0),
        "github": float(getattr(settings, "weight_github", 0.0) or 0.0),
    }
    table = LockWeightTable()
    rows = ledger_sources or []
    for voter, base in bases.items():
        hits = 0.0
        n = 0.0
        for row in rows:
            name = str(row.get("source") or "")
            if not any(name == b or (b.endswith(":") and name.startswith(b)) for b in BACKERS[voter]):
                continue
            rn = float(row.get("n") or 0.0)
            rr = row.get("reliability")
            if rn <= 0 or rr is None:
                continue
            hits += float(rr) * rn
            n += rn
        hit_rate = (hits / n) if n > 0 else None
        table.rows[voter] = LockWeight(
            voter=voter,
            base=base,
            reliability=reliability_multiplier(hit_rate, n),
            hit_rate=hit_rate,
            samples=n,
        )
    return table
