"""The physics layer's per-cycle pass (Round AJ rebuild).

``PhysicsEngine.compute(snapshot, asset)`` runs every kinetic mechanism on the
frozen snapshot and returns one JSON-ready report: the signed vote for
``asset`` (BTC +, PAXG the mirror), a confidence, every mechanism with its
inputs and printed logic, the multi-mechanism Kelly blend, the temperature
drag and the cost gate.

What changed in Round AJ and why
--------------------------------
The old layer centred on three *planetary* quantities - the Landauer thermal
valve, the solar-flux opportunity cost and the E = mc² work-to-rest-mass
ratio.  All three depend on the global hashrate, which moves by fractions of
a percent per day: inside a 60-second window they were constants, so their
drift terms printed 0.00, the phase angle was always "balanced", and the
AMM surface needed a DEX host that is unreachable from most networks.  They
are gone.  Every mechanism that remains or was added reads the tape of the
window itself, carries an ``active`` flag with the reason when it cannot be
estimated (so it is excluded from the blend instead of voting 0.00), and the
blend is gated by the cost of actually crossing the spread.

Deterministic, allocation-light (< 3 ms) and lock-safe: it reads nothing
live except the telemetry cache (venue quotes, gold spot) and that only to
*enable* the two mechanisms that need it.
"""
from __future__ import annotations

import time

from backend.core import config as cfg
from backend.physics import constants as K
from backend.physics import kinetics, micro, unified, venues
from backend.physics.telemetry import get_telemetry


class PhysicsEngine:
    def __init__(self, delta_w: float | None = None) -> None:
        # Kept for configuration compatibility (PHYSICS_DELTA_W); it now bounds
        # how far the Kelly tilt may push w_micro from ½ in one window.
        self.delta_w = float(delta_w if delta_w is not None else
                             getattr(cfg.SETTINGS, "physics_delta_w", K.DELTA_W_DEFAULT))
        self.edge_hist: dict[str, list[float]] = {m: [] for m in unified.MECHANISMS}
        self.last_report: dict | None = None
        self.passes = 0

    # ------------------------------------------------------------------
    def compute(self, snapshot, asset: str, seconds_left: float = 60.0) -> dict:
        started = time.perf_counter()
        now = time.time()
        tele = get_telemetry()
        btc_usd = float(snapshot.last_price("BTC"))
        paxg_usd = float(snapshot.last_price("PAXG"))
        market_ratio = btc_usd / paxg_usd if paxg_usd > 0 else 0.0
        source = str(getattr(snapshot, "source", "") or "")
        real_tape = source not in ("", "simulator", "none")

        mechanisms = {
            "vpin": micro.vpin(snapshot),
            "ou": micro.ornstein_uhlenbeck(snapshot),
            "pendulum": micro.pendulum(snapshot),
            "hawkes": kinetics.hawkes(snapshot),
            "kinetic": kinetics.kinetic(snapshot),
            "entropy": kinetics.entropy(snapshot),
            "diffusion": kinetics.diffusion(snapshot),
            "temperature": kinetics.temperature(snapshot),
            "as_spread": micro.avellaneda_stoikov(snapshot, seconds_left),
            "fragmentation": micro.fragmentation(snapshot, tele.venues.value or {}, []),
            "peg": venues.peg_drift(paxg_usd, tele.xau.value, tele.xau.provider, tele.wbtc.value, btc_usd,
                                    tele.wbtc.provider),
        }
        for name, m in mechanisms.items():
            m.setdefault("active", True)
            m.setdefault("direction", 0)
            m.setdefault("edge_bps", 0.0)
            self.edge_hist[name].append(float(m.get("direction", 0)) * float(m.get("edge_bps", 0.0)) if m["active"] else 0.0)
            del self.edge_hist[name][:-240]

        active = {k: m for k, m in mechanisms.items() if m.get("active")}
        directional = [m for m in active.values() if m.get("direction")]

        # --- Kelly blend over the active mechanisms, temperature drag, cost gate
        kel = unified.kelly(active, self.edge_hist)
        drag = float(mechanisms["temperature"].get("drag", 0.0) or 0.0)
        w_micro = kel["w_micro"]
        w_final, clamped = unified.clamp_weight(w_micro, 0.5, self.delta_w)
        raw_vote = 2.0 * (w_micro - 0.5) * (1.0 - drag)
        gross_edge = sum(float(m.get("edge_bps", 0.0)) for m in directional)
        cost_bps = float(mechanisms["as_spread"].get("cost_bps", 0.0) or 0.0)
        net_edge = gross_edge - cost_bps
        below_cost = bool(directional) and net_edge <= 0.0
        vote_btc = raw_vote * (0.5 if below_cost else 1.0)
        vote = vote_btc if asset.upper() == "BTC" else -vote_btc

        agree = (abs(sum(m["direction"] for m in directional)) / len(directional)) if directional else 0.0
        live_inputs = (len(active) if real_tape else 0) + sum(
            1 for m in active.values() if str(m.get("source", "")).startswith("live"))
        confidence = max(0.05, min(0.95, 0.30 + 0.35 * agree + 0.03 * min(8, live_inputs)
                                   + 0.2 * min(1.0, abs(vote_btc) / 0.3) - 0.15 * drag - (0.1 if below_cost else 0.0)))

        self.passes += 1
        w_logic = (f"w_micro = ½ + ½·tanh(Σf/0.6) = {w_micro:.4f} from {len(directional)} voting of {len(active)} active mechanisms; "
                   f"vote = 2(w_micro − ½)·(1 − drag {drag:.2f}) = {raw_vote:+.4f}; gross edge {gross_edge:.2f} bp − spread cost "
                   f"{cost_bps:.2f} bp = {net_edge:+.2f} bp" + (" ≤ 0 ⇒ vote halved" if below_cost else "") + f" ⇒ {vote_btc:+.4f}")
        report = {
            "asset": asset.upper(), "pair": "BTC/PAXG", "vote": round(vote, 4), "vote_btc": round(vote_btc, 4),
            "side": ("BUY" if vote >= 0 else "SELL"), "confidence": round(confidence, 4),
            "weights": {
                "w_micro": round(w_micro, 4), "w_final": round(w_final, 4), "w_paxg": round(1.0 - w_final, 4),
                "delta_w": self.delta_w, "clamped": clamped, "drag": round(drag, 4),
                "below_cost": below_cost, "logic": w_logic,
            },
            "mechanisms": [mechanisms[m] for m in unified.MECHANISMS],
            "active": sorted(active), "inactive": {k: m.get("logic", "") for k, m in mechanisms.items() if not m.get("active")},
            "kelly": kel,
            "composite": {
                "gross_edge_bps": round(gross_edge, 3), "cost_bps": round(cost_bps, 3), "net_edge_bps": round(net_edge, 3),
                "below_cost": below_cost, "temperature": round(float(mechanisms["temperature"].get("value", 1.0)), 3),
                "hawkes_n": round(float(mechanisms["hawkes"].get("value", 0.0)), 3),
                "entropy": round(float(mechanisms["entropy"].get("value", 0.0)), 3),
                "voting": len(directional), "active": len(active), "agreement": round(agree, 3),
                "note": ("expected edges are model estimates for this window, never a guaranteed yield; "
                         "planetary sections (Landauer, solar, E=mc², AMM) were retired in Round AJ because they "
                         "are constant inside a minute; Sections 3, 6, 7 and 9 of the source document are not implemented"),
            },
            "market": {"btc_usd": round(btc_usd, 2), "paxg_usd": round(paxg_usd, 2), "ratio": round(market_ratio, 4)},
            "telemetry": tele.status(),
            "live_inputs": live_inputs, "agreement": round(agree, 3),
            "elapsed_us": int((time.perf_counter() - started) * 1e6), "passes": self.passes,
            "computed_at": now,
        }
        self.last_report = report
        return report
