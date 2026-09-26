"""Common types shared by every AI agent."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum


class AgentStatus(str, Enum):
    ACTIVE = "ACTIVE"  # healthy and answering
    DISABLED = "DISABLED"  # not configured by the user
    INACTIVE = "INACTIVE"  # configured but skipped this cycle
    ERROR = "ERROR"  # configured, failed this cycle
    RATE_LIMITED = "RATE_LIMITED"
    TIMEOUT = "TIMEOUT"
    LOADING = "LOADING"
    STUB = "STUB"  # development stand-in, clearly not a real model


@dataclass
class AgentResult:
    """One agent's answer for one cycle."""

    agent: str
    decision: str | None = None  # "BUY" | "SELL" | None (HOLD is not a vote)
    confidence: float | None = None
    reasoning: str = ""
    status: AgentStatus = AgentStatus.INACTIVE
    error: str = ""
    latency_ms: float = 0.0
    model: str = ""
    weight: float = 0.0
    raw: str = ""

    @property
    def available(self) -> bool:
        return self.decision is not None and self.status in (
            AgentStatus.ACTIVE,
            AgentStatus.STUB,
        )

    @property
    def directional(self) -> int:
        return {"BUY": 1, "SELL": -1}.get(self.decision or "", 0)

    def to_dict(self) -> dict:
        return {
            "agent": self.agent,
            "decision": self.decision,
            "confidence": round(self.confidence, 4) if self.confidence is not None else None,
            "reasoning": self.reasoning,
            "status": self.status.value,
            "error": self.error,
            "latency_ms": round(self.latency_ms, 2),
            "model": self.model,
            "weight": self.weight,
        }


@dataclass
class AgentHealth:
    name: str
    status: AgentStatus = AgentStatus.DISABLED
    detail: str = ""
    model: str = ""
    extra: dict = field(default_factory=dict)
    updated_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        payload = {
            "name": self.name,
            "status": self.status.value,
            "detail": self.detail,
            "model": self.model,
            "updated_at": self.updated_at,
        }
        if self.extra:
            payload.update(self.extra)
        return payload


SYSTEM_PROMPT = (
    "You are DROSOPHILA TRADER v2, an expert AI for 1-minute scalp trading on BTC and PAXG. "
    "You receive 22 proprietary formula outputs spanning micro-structure, order book, hedge "
    "dynamics, regime analysis, temporal patterns, Kalman filtering, news sentiment, and a "
    "biological neural network consensus from the Drosophila melanogaster brain connectome.\n"
    "\n"
    "RULES:\n"
    "1. Respond with ONLY JSON: {\"decision\": \"BUY\"|\"SELL\", "
    "\"confidence\": 0.0-1.0, \"reasoning\": \"<one sentence>\"}\n"
    "1b. HOLD is not an option. You must always name a side; if the evidence is "
    "mixed, pick the side the evidence leans to and say so in the reasoning, with "
    "a LOW confidence.\n"
    "2. If HSI > 0.7 (hedge stress high), keep the direction but cut confidence.\n"
    "3. If ERC > 0.5 (choppy regime), cut confidence and note the chop.\n"
    "4. If DRG < -0.3 (recent losses), reduce confidence by 20%.\n"
    "5. If NIV and TAI agree in direction, increase confidence.\n"
    "6. If NIV and TAI disagree (SMD far from 0), note the divergence.\n"
    "7. Never output confidence > 0.95.\n"
    "8. Reference at least 2 specific formula names in reasoning.\n"
    "9. Pay special attention to HEDGE formulas (HRDD, SHRP, GCDV, HSI) when analyzing PAXG."
)

#: Gemini's structured-output schema (Section 7.1, Fix 1) - guarantees parseable JSON.
RESPONSE_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "decision": {"type": "STRING", "enum": ["BUY", "SELL"]},
        "confidence": {"type": "NUMBER"},
        "reasoning": {"type": "STRING"},
    },
    "required": ["decision", "confidence", "reasoning"],
}


def build_cycle_prompt(context: dict) -> str:
    """The per-cycle message template from Appendix D."""
    f = context.get("formulas", {})
    return (
        f"Asset: {context.get('asset', 'BTC')}/USDT | Price: ${context.get('price', 0):.2f} "
        f"| Time: {context.get('timestamp', '')}\n"
        "\n"
        f"MICRO-STRUCTURE: TAI={f.get('TAI', 0):+.3f} AFPR={f.get('AFPR', 0):+.3f} "
        f"SED={f.get('SED', 0):+.3f} VSD={f.get('VSD', 0):+.3f}\n"
        f"ORDER BOOK: DGW={f.get('DGW', 0):+.3f} LCS={f.get('LCS', 0):+.3f} "
        f"BAR={f.get('BAR', 0):+.3f}\n"
        f"HEDGE: HRDD={f.get('HRDD', 0):+.3f} SHRP={f.get('SHRP', 0):+.3f} "
        f"GCDV={f.get('GCDV', 0):+.3f} HSI={f.get('HSI', 0):.3f}\n"
        f"REGIME: RSV={f.get('RSV', 0):+.3f} VSS={f.get('VSS', 0):+.3f} ERC={f.get('ERC', 0):+.3f}\n"
        f"TEMPORAL: MCPE={f.get('MCPE', 0):+.3f} MPS={f.get('MPS', 0):+.3f} "
        f"TWRS={f.get('TWRS', 0):+.3f}\n"
        f"KALMAN: DSKD={f.get('DSKD', 0):+.3f}\n"
        f"NEWS: NIV={f.get('NIV', 0):+.3f} SMD={f.get('SMD', 0):+.3f} "
        f"| Latest: \"{context.get('headline', '')}\"\n"
        f"BRAIN: KCAE={f.get('KCAE', 0):.3f} CCSv2={f.get('CCSv2', 0):+.3f} "
        f"(Confidence={context.get('ccs_confidence', 0):.0%})\n"
        f"DRG: {context.get('drg', 0):+.3f} | Recent win rate: {context.get('win_rate', 0):.0%}\n"
        "\n"
        f"Brain status: {context.get('brain_status', 'UNKNOWN')}\n"
        "\n"
        "Provide your analysis."
    )


def parse_agent_json(text: str) -> tuple[str | None, float | None, str]:
    """Tolerant JSON extraction - models sometimes wrap JSON in prose."""
    import json
    import re

    if not text:
        return None, None, ""
    candidate = text.strip()
    if candidate.startswith("```"):
        candidate = re.sub(r"^```[a-zA-Z]*\n?|```$", "", candidate).strip()
    try:
        payload = json.loads(candidate)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", candidate, re.DOTALL)
        if not match:
            return None, None, ""
        try:
            payload = json.loads(match.group(0))
        except json.JSONDecodeError:
            return None, None, ""

    decision = str(payload.get("decision", "")).upper().strip()
    if decision == "HOLD":
        # HOLD was removed from the protocol.  A model that still answers HOLD is
        # abstaining, not voting for "zero": returning None makes the fusion
        # layer renormalise its weight away instead of dragging the score toward
        # the middle.  The reasoning is kept so the UI can show what it said.
        reasoning = str(payload.get("reasoning", ""))
        return None, None, f"[abstained: answered HOLD, which is not a signal] {reasoning}"[:400]
    if decision not in ("BUY", "SELL"):
        return None, None, str(payload.get("reasoning", ""))
    try:
        confidence = float(payload.get("confidence", 0.5))
    except (TypeError, ValueError):
        confidence = 0.5
    confidence = max(0.0, min(0.95, confidence))  # Rule 7: never above 0.95
    return decision, confidence, str(payload.get("reasoning", ""))[:400]
