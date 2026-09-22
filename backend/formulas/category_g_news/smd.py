"""FORMULA 20 - Sentiment Momentum Divergence (SMD)  [NEW in v2.0].

Compares what the news says (NIV) with what the tape is doing (TAI).

    SMD = tanh(NIV - TAI)

    SMD > 0 -> news is more bullish than price: either the news has not been
               priced in yet (contrarian buy) or the market is ignoring it
    SMD < 0 -> news is more bearish than price: price is showing strength, or
               bad news is about to land
    SMD ~ 0 -> news and price agree: confirmation

Brain mapping: multimodal integration neuron in the lateral horn, comparing
olfactory (price) and mechanosensory (news) channels.

Latency budget: < 0.01 ms.
"""

from __future__ import annotations

from backend.formulas._util import finite, tanh

NAME = "SMD"
CATEGORY = "G"
TITLE = "Sentiment Momentum Divergence"
BRAIN_NODE = "LH multimodal"
DIRECTIONAL = True
LATENCY_MS = 0.01
DESCRIPTION = "Divergence between news sentiment (NIV) and price momentum (TAI)."


class State:
    __slots__ = ("last_smd",)

    def __init__(self) -> None:
        self.last_smd = 0.0

    def to_dict(self) -> dict:
        return {"last_smd": self.last_smd}

    @classmethod
    def from_dict(cls, payload: dict) -> "State":
        obj = cls()
        obj.last_smd = float(payload.get("last_smd", 0.0))
        return obj


def compute(snapshot, asset: str, state: State, params: dict, ctx: dict | None = None) -> float:
    ctx = ctx or {}
    niv = float(ctx.get("NIV", 0.0))
    tai = float(ctx.get("TAI", 0.0))
    score = tanh(niv - tai)
    state.last_smd = score
    return finite(score)
