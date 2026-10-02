"""The thermodynamic layer's per-cycle pass.

``PhysicsEngine.compute(snapshot, asset)`` runs every implemented section on
the frozen snapshot plus the cached telemetry and returns one JSON-ready
report: the signed vote for ``asset`` (BTC +, PAXG the mirror), a confidence,
the physical and microstructural weights, every mechanism with its inputs and
its printed logic, the Kelly blend, TSR and the phase angle.

Deterministic, allocation-light (< 2 ms) and lock-safe: it reads nothing
live except the telemetry cache, which the fusion stage snapshots with it.
"""
from __future__ import annotations

import time
from collections import deque

from backend.core import config as cfg
from backend.physics import constants as K
from backend.physics import micro, physical, unified, venues
from backend.physics.telemetry import get_telemetry


class PhysicsEngine:
    def __init__(self, delta_w: float | None = None) -> None:
        self.delta_w = float(delta_w if delta_w is not None else
                             getattr(cfg.SETTINGS, "physics_delta_w", K.DELTA_W_DEFAULT))
        self.theta_hist: deque[tuple[float, float]] = deque(maxlen=720)      # (t, Θ) ~ 1 h at 5 s
        self.omega_hist: deque[tuple[float, float]] = deque(maxlen=720)
        self.ratio_hist: deque[tuple[float, float, float]] = deque(maxlen=720)  # (t, R, market)
        self.edge_hist: dict[str, list[float]] = {m: [] for m in unified.MECHANISMS}
        self.last_report: dict | None = None
        self.passes = 0

    # ------------------------------------------------------------------
    def _remember(self, store: deque, now: float, *values: float, min_gap: float = 5.0) -> None:
        if store and now - store[-1][0] < min_gap:
            return
        store.append((now, *values))
        while store and now - store[0][0] > 3600.0:
            store.popleft()

    # ------------------------------------------------------------------
    def compute(self, snapshot, asset: str, seconds_left: float = 60.0) -> dict:
        started = time.perf_counter()
        now = time.time()
        tele = get_telemetry()
        btc_usd = float(snapshot.last_price("BTC"))
        paxg_usd = float(snapshot.last_price("PAXG"))
        market_ratio = btc_usd / paxg_usd if paxg_usd > 0 else 0.0

        hashrate = float(tele.hashrate.value or K.MODEL_HASHRATE_H_PER_S)
        hr_source = ("live (" + tele.hashrate.provider + ")") if tele.hashrate.source == "live" else "model (hashrate constant)"
        series = list(tele.hashrate.detail.get("series") or [])

        # --- Sections 1, 2, 4 ---------------------------------------------
        th = physical.landauer(hashrate, series, now, hr_source)
        so = physical.solar(hashrate, now)
        self._remember(self.theta_hist, now, th["theta"])
        self._remember(self.omega_hist, now, so["omega"])
        alpha = physical.blend_alpha([v for _, v in self.theta_hist], [v for _, v in self.omega_hist])
        w_composite = alpha * th["value"] + (1.0 - alpha) * so["value"]
        em = physical.energy_mass(hashrate, market_ratio, list(self.ratio_hist), now, hr_source)
        self._remember(self.ratio_hist, now, em["ratio_oz_per_btc"], market_ratio, min_gap=30.0)

        # --- Sections 5, 8, 10 --------------------------------------------
        pools = list(tele.pools.value or [])
        pool_source = "live (dexscreener)" if tele.pools.source == "live" else "unavailable (DexScreener unreachable here)"
        mechanisms = {
            "amm": venues.amm_surface(pools, btc_usd, paxg_usd, pool_source),
            "vpin": micro.vpin(snapshot),
            "fragmentation": micro.fragmentation(snapshot, tele.venues.value or {}, pools),
            "ou": micro.ornstein_uhlenbeck(snapshot),
            "pendulum": micro.pendulum(snapshot),
            "as_spread": micro.avellaneda_stoikov(snapshot, seconds_left),
            "peg": venues.peg_drift(paxg_usd, tele.xau.value, tele.xau.provider, tele.wbtc.value, btc_usd,
                                    tele.wbtc.provider),
            "energy_mass": {**em, "direction": int(em["value"] > 0.05) - int(em["value"] < -0.05),
                            "edge_bps": min(4.0, abs(em["value"]) * 4.0)},
        }
        for name, m in mechanisms.items():
            self.edge_hist[name].append(float(m.get("direction", 0)) * float(m.get("edge_bps", 0.0)))
            del self.edge_hist[name][:-240]

        # --- Sections 11-12 -----------------------------------------------
        kel = unified.kelly(mechanisms, self.edge_hist)
        w_final, clamped = unified.clamp_weight(kel["w_micro"], w_composite, self.delta_w)
        vote_btc, drag = unified.window_vote(kel["w_micro"], w_composite)
        vote = vote_btc if asset.upper() == "BTC" else -vote_btc
        directional = [m for m in mechanisms.values() if m.get("direction")]
        agree = (abs(sum(m["direction"] for m in directional)) / len(directional)) if directional else 0.0
        live_inputs = sum(1 for m in mechanisms.values() if str(m.get("source", "")).startswith("live"))
        confidence = max(0.05, min(0.95, 0.35 + 0.35 * agree + 0.05 * live_inputs + 0.2 * min(1.0, abs(vote_btc) / 0.3)))
        total_edge = sum(float(m.get("edge_bps", 0.0)) for m in mechanisms.values())
        floor_edge = float(mechanisms["fragmentation"].get("edge_bps", 0.0)) + 0.5 * float(mechanisms["as_spread"].get("edge_bps", 0.0))
        tsr = unified.thermodynamic_sharpe(total_edge, th["theta"])

        self.passes += 1
        report = {
            "asset": asset.upper(), "pair": "BTC/PAXG", "vote": round(vote, 4), "vote_btc": round(vote_btc, 4),
            "side": ("BUY" if vote >= 0 else "SELL"), "confidence": round(confidence, 4),
            "weights": {
                "w_thermal": round(th["value"], 4), "w_solar": round(so["value"], 4), "alpha": round(alpha, 4),
                "w_composite": round(w_composite, 4), "w_micro": round(kel["w_micro"], 4),
                "delta_w": self.delta_w, "w_final": round(w_final, 4), "w_paxg": round(1.0 - w_final, 4),
                "clamped": clamped, "drag": round(drag, 4),
                "logic": (f"w_composite = α·w_thermal + (1−α)·w_solar = {alpha:.3f}·{th['value']:.3f} + {1 - alpha:.3f}·{so['value']:.3f} = {w_composite:.4f} (portfolio target); "
                          f"w_final = clamp(w_micro = {kel['w_micro']:.4f}, w_composite ± {self.delta_w}) = {w_final:.4f}"
                          f"{' (clamped)' if clamped else ''}; "
                          f"60 s vote = 2(w_micro − ½)·(1 − drag), drag = {drag:.3f} against the physical lean ⇒ {vote_btc:+.4f}"),
            },
            "physical": {"landauer": th, "solar": so, "energy_mass": em},
            "mechanisms": [mechanisms[m] for m in unified.MECHANISMS],
            "kelly": kel,
            "composite": {
                "tsr": round(tsr, 5), "theta": round(th["theta"], 4),
                "expected_edge_bps": round(total_edge, 3), "floor_edge_bps": round(floor_edge, 3),
                "phase_angle_deg": round(physical.math.degrees(em["phase_angle_rad"]), 3),
                "phase": unified.phase_label(em["phase_angle_rad"]),
                "note": ("expected edges are model estimates for this window, never a guaranteed yield; "
                         "Sections 3, 7 and 9 of the source document are not implemented"),
            },
            "market": {"btc_usd": round(btc_usd, 2), "paxg_usd": round(paxg_usd, 2), "ratio": round(market_ratio, 4)},
            "telemetry": tele.status(),
            "live_inputs": live_inputs, "agreement": round(agree, 3),
            "elapsed_us": int((time.perf_counter() - started) * 1e6), "passes": self.passes,
            "computed_at": now,
        }
        self.last_report = report
        return report

