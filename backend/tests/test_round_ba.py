"""Round BA - "right about the past, not the future": the warm start replays
real history through the formulas and scores every source on the NEXT
window; the competition ledger audits whether picks merely follow the last
move; both are on the dashboard."""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from backend.core import warm_start as ws
from backend.core.calibration import EvidenceLedger
from backend.core.competition import CompetitionLedger

ROOT = Path(__file__).resolve().parents[2]


def _bars(seconds: int, seed: int = 3) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t0 = (int(time.time() * 1000) // 1000) * 1000 - seconds * 1000
    ret = rng.normal(0, 2e-5, seconds)
    px = 60000 * np.exp(np.cumsum(ret))
    vol = rng.gamma(2, 0.3, seconds)
    buy = vol * np.clip(0.5 + np.sign(ret) * 0.2, 0, 1)
    return np.column_stack([t0 + np.arange(seconds) * 1000 + 999, px, px, px, px, vol, buy])


def test_bars_become_the_live_tape_shape():
    ticks = ws.bars_to_ticks(_bars(100))
    assert ticks.shape == (100, 4) and set(np.unique(ticks[:, 3])) <= {-1.0, 0.0, 1.0}
    closes = ws.minute_closes(_bars(600), _bars(600)[-1, 0])
    assert 9 <= closes.size <= 11


def test_boundaries_follow_the_epoch_grid():
    b = ws.window_boundaries(1_000_000, 1_000_000 + 10 * 53_000, 53.0)
    assert all(int(x) % 53_000 == 0 for x in b) and b.size >= 9


def test_replay_scores_every_source_on_the_next_window_and_activates_the_ledger():
    ledger = EvidenceLedger(path=None, enabled=True, min_samples=30)
    rep = ws.replay("BTC", _bars(3600 + 700), None, ledger, brain_mb=None, period_s=53.0)
    assert rep.phase == "done", rep.error
    assert rep.scored >= 50 and rep.recipe_windows <= rep.scored
    assert rep.continuation_rate is not None and 0.0 <= rep.continuation_rate <= 1.0
    assert any(name.startswith("f:") for name in rep.sources)      # the formulas were scored
    assert "micro:ret_1m" in rep.replayed_sources
    verdict = ledger.evaluate("BTC", {"f:TAI": 1, "micro:ret_1m": 1})
    assert verdict["scored"] == rep.scored and verdict["active"] is True
    # replay windows never collide with live cycle numbers
    assert all(k[1] < 0 for k in ledger._pending)


def test_warm_starter_fails_soft_without_the_network(monkeypatch):
    ledger = EvidenceLedger(path=None, enabled=True)
    starter = ws.WarmStarter(ledger, asset="BTC", hours=1.0, brain=None)
    monkeypatch.setattr(ws, "fetch_klines", lambda *a, **k: (_ for _ in ()).throw(OSError("offline")))
    rep = starter.run()
    assert rep.phase == "failed" and "offline" in rep.error and rep.elapsed_s >= 0


def test_lag_audit_measures_following_the_last_move():
    led = CompetitionLedger()
    px = [100.0, 101.0, 100.0, 101.0, 100.0, 101.0]
    # a picker that always repeats the last move on an alternating tape
    for i in range(1, len(px) - 1):
        side = "BUY" if px[i] > px[i - 1] else "SELL"
        led.enter(i, side, px[i])
        led.settle(i, px[i + 1])
    audit = led.lag_audit()
    assert audit["pairs"] >= 3 and audit["follows_last_window"] == 1.0 and audit["win_when_following"] == 0.0
    assert "lag_audit" in led.to_dict()


def test_dashboard_prints_past_vs_future():
    html = (ROOT / "backend" / "web" / "index.html").read_text()
    js = (ROOT / "backend" / "web" / "app.js").read_text()
    assert 'id="c-lag"' in html and "function lagLine" in js and "warm_start" in js
    src = (ROOT / "backend" / "api" / "routes_signals.py").read_text()
    assert '@router.get("/api/learning/warm-start")' in src
    cm = (ROOT / "backend" / "core" / "cycle_manager.py").read_text()
    assert "self._maybe_warm_start()" in cm and 'out["warm_start"] = self.warm_starter.report.to_dict()' in cm
