"""Round AS - the competition rule, the ladder and the boundary pick."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pytest

from backend.core.competition import (
    BoundaryPicker, CompetitionLedger, boundary_features, expected_windows_to_streak, streak_probability,
)

ROOT = Path(__file__).resolve().parents[2]


def test_the_rule_is_last_trade_vs_last_trade_and_a_tie_loses():
    led = CompetitionLedger()
    led.enter(1, "BUY", 83000.0)
    assert led.settle(1, 83000.0).tie is True and led.streak == 0 and led.ties == 1
    led.enter(2, "BUY", 83000.0)
    assert led.settle(2, 83000.01).won is True and led.streak == 1
    led.enter(3, "SELL", 83000.0)
    assert led.settle(3, 82999.99).won is True and led.streak == 2
    led.enter(4, "SELL", 83000.0)
    assert led.settle(4, 83000.01).won is False and led.streak == 0 and led.best_streak == 2
    assert led.streak_history[-1] == 2
    d = led.to_dict()
    assert d["scored"] == 4 and d["wins"] == 2 and d["win_rate"] == 0.5
    assert "tie is a loss" in d["rule"]


def test_the_ladder_is_the_users_ladder_and_the_arithmetic_is_honest():
    led = CompetitionLedger()
    for i in range(1, 21):
        led.enter(i, "BUY", 100.0)
        led.settle(i, 101.0)
    d = led.to_dict()
    assert [r["streak"] for r in d["ladder"]] == [5, 20, 50, 72]
    assert d["ladder"][0]["hit"] == 1 and d["ladder"][1]["hit"] == 1 and d["ladder"][1]["reached"]
    assert "$30" in d["ladder"][2]["prize"] and d["next_rung"] == 50 and d["to_next_rung"] == 30
    # 20/20 wins is still shrunk toward 0.5 - no "50-streak every afternoon"
    assert d["p_estimate"] < 0.9
    assert streak_probability(0.8, 10) == pytest.approx(0.1074, abs=1e-3)
    assert expected_windows_to_streak(0.5, 5) == pytest.approx(62.0)
    assert expected_windows_to_streak(0.0, 5) is None and expected_windows_to_streak(1.0, 5) == 5.0


def _tape(entry=83000.0, last_side=1.0, drift_ticks=0.0, n=120, now_ms=1_000_000.0):
    rows = []
    for i in range(n):
        t = now_ms - (n - i) * 100.0
        px = entry - drift_ticks * 0.01 * (1 - i / n)
        rows.append((t, round(px, 2), 0.05, 1.0 if i % 2 else -1.0))
    rows[-1] = (now_ms - 50.0, entry, 0.05, last_side)
    return np.asarray(rows, dtype=np.float64)


def test_boundary_features_read_the_entry_print_microprice_and_short_flow():
    f = boundary_features(_tape(last_side=1.0), bid=82999.99, ask=83000.01, bid_qty=9.0, ask_qty=1.0,
                          now_ms=1_000_000.0, pin_rate=0.0, tick=0.01)
    assert f.entry == 83000.0 and f.entry_side == 1.0
    # heavy bid -> microprice above the entry (ask*bid_qty dominates)
    assert f.microprice > f.entry and f.gap_ticks > 0 and f.book_imbalance == pytest.approx(0.8)
    assert f.prints_5s == 50 and abs(f.ofi5) <= 1.0
    assert boundary_features(np.zeros((0, 4)), 1, 2, 1, 1, 0.0, 0.0) is None


def test_the_picker_keeps_a_confident_recipe_and_flips_a_coin_flip_on_the_print():
    pk = BoundaryPicker()
    strong = boundary_features(_tape(last_side=1.0), 82999.99, 83000.01, 1.0, 9.0, 1_000_000.0, 0.0, 0.01)
    keep = pk.pick(1, strong, "BUY", 0.80)
    assert keep.side == "BUY" and not keep.flipped and "recipe kept" in keep.note
    # a 51 % BUY whose entry print sat at the ask with a heavy ask -> SELL
    flip = pk.pick(2, strong, "BUY", 0.51)
    assert flip.side == "SELL" and flip.flipped and flip.p_up < 0.5
    d = flip.as_dict()
    assert d["p_side"] > 0.5 and set(d["contributions"]) == set(BoundaryPicker.PRIOR)


def test_a_pinned_tape_breaks_the_tie_toward_the_microprice():
    pk = BoundaryPicker()
    f = boundary_features(_tape(last_side=0.0), 82999.99, 83000.01, 5.0, 5.0, 1_000_000.0, 0.4, 0.01)
    f.entry_side, f.ofi5, f.book_imbalance, f.gap_ticks = 0.0, 0.0, 0.0, 0.3   # balanced, microprice a shade above
    pick = pk.pick(3, f, "SELL", 0.5)
    assert pick.side == "BUY" and "tie broken toward the microprice" in pick.note


def test_the_picker_learns_online_and_ties_teach_nothing():
    pk = BoundaryPicker()
    f = boundary_features(_tape(last_side=1.0), 82999.99, 83000.01, 1.0, 9.0, 1_000_000.0, 0.0, 0.01)
    w0 = dict(pk.w)
    pk.pick(1, f, "BUY", 0.6)
    pk.learn(1, None)
    assert pk.updates == 0 and pk.w == w0
    for i in range(2, 40):
        pk.pick(i, f, "BUY", 0.6)
        pk.learn(i, False)                   # the recipe keeps saying up; the tape keeps going down
    assert pk.updates == 38 and pk.w["recipe"] < w0["recipe"] and pk.w["recipe"] >= 0.2
    assert pk.w["bounce"] < w0["bounce"]     # buy-print -> down got reinforced
    d = pk.to_dict()
    assert d["updates"] == 38 and d["log_loss_200"] is not None


def test_the_engine_enters_on_the_last_trade_and_settles_the_competition():
    src = (ROOT / "backend/core/cycle_manager.py").read_text()
    assert "self.market.last_trade(self.asset)" in src and "_boundary_pick(draft, cycle, price)" in src
    assert "self.competition.enter(cycle, locked.signal" in src
    assert "self.competition.settle(signal.cycle_number, settle_px)" in src
    assert "self.picker.learn(signal.cycle_number" in src
    assert 'payload["competition"]' in src and 'payload["boundary_pick"]' in src
    hub = (ROOT / "backend/data/market_hub.py").read_text()
    assert "def last_trade(self, asset: str)" in hub and "def l1(self, asset: str)" in hub
    index = (ROOT / "backend/web/index.html").read_text()
    app = (ROOT / "backend/web/app.js").read_text()
    assert 'id="competition-cell"' in index and 'id="c-ladder"' in index and 'id="c-pick"' in index
    assert "function renderCompetition" in app and "msg.data.competition" in app
    assert "/api/competition" in (ROOT / "backend/api/routes_brain.py").read_text()


# ---------------------------------------------------------- key states fix
def test_a_key_in_any_slot_counts_as_configured_everywhere():
    """User report: 'UI says api key not set even though I saved it'."""
    from backend.agents.keyring import KeyRing

    ring = KeyRing("gemini")
    for slot in ring.slots:
        slot.key = ""
    ring.set_slot(2, "AQ.Ab-some-new-style-key")
    assert ring.configured and ring.primary == "AQ.Ab-some-new-style-key"
    main_src = (ROOT / "backend/api/main.py").read_text()
    assert 'settings.rings["gemini"].configured' in main_src
    app_js = (ROOT / "backend/web/app.js").read_text()
    # the dashboard re-reads the engine's config after a save instead of
    # repainting the copy it took at page load
    assert "async function refreshKeyStates" in app_js
    assert app_js.count("await refreshKeyStates()") >= 1
    assert len(re.findall(r"setInterval\(", app_js)) == 1


def test_local_runtime_is_built_in_the_background_and_documented():
    run_sh = (ROOT / "run.sh").read_text()
    assert "llama-cpp-python" in run_sh and "LOCAL_RUNTIME_AUTOINSTALL" in run_sh
    assert "setsid nohup" in run_sh and "llama_runtime.log" in run_sh
    doc = (ROOT / "docs/LOCAL_MODELS.md").read_text()
    for needle in ("9 s", "Q4_K_M", "GGUF", "n_ctx", "Codespaces", "Qwen2.5-0.5B-Instruct"):
        assert needle in doc, needle
    from backend.agents.local_model_agent import LocalModelAgent
    assert "LOCAL_MODELS.md" in LocalModelAgent._runtime_build_hint() or "background" in LocalModelAgent._runtime_build_hint()


# --------------------------------------------------------- Round AU: fetch
def test_model_urls_are_normalised_to_direct_file_links():
    from backend.agents.model_fetch import normalise_url

    direct = "https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF/resolve/main/qwen2.5-1.5b-instruct-q4_k_m.gguf"
    assert normalise_url("hf://Qwen/Qwen2.5-1.5B-Instruct-GGUF/qwen2.5-1.5b-instruct-q4_k_m.gguf") == direct
    assert normalise_url("https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF/blob/main/qwen2.5-1.5b-instruct-q4_k_m.gguf") == direct
    assert normalise_url(direct + "?download=true") == direct + "?download=true"
    with pytest.raises(ValueError):
        normalise_url("https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF")      # the repo page, not a file
    with pytest.raises(ValueError):
        normalise_url("Qwen2.5-1.5B")


def test_big_models_bypass_the_browser(tmp_path):
    """User report: HTTP 5xx from the Codespaces proxy on a 1 GB upload."""
    from backend.agents.local_pool import LocalModelPool

    pool = LocalModelPool(store_dir=tmp_path)
    (tmp_path / "tiny.gguf").write_bytes(b"GGUF" + b"\0" * 64)
    (tmp_path / "notes.txt").write_text("x")
    files = pool.fetcher.files_on_disk()
    assert [f["file"] for f in files] == ["tiny.gguf"]
    assert pool.fetcher.status()["phase"] == "idle" and not pool.fetcher.active
    routes = (ROOT / "backend/api/routes_agents.py").read_text()
    for needle in ("/api/agents/local/fetch", "/api/agents/local/load-path", "await file.read(1 << 20)"):
        assert needle in routes, needle
    html = (ROOT / "backend/web/settings.html").read_text()
    assert 'id="fetch-url"' in html and 'id="disk-files"' in html
    js = (ROOT / "backend/web/settings.js").read_text()
    assert "/api/agents/local/fetch" in js and "/api/agents/local/load-path" in js


@pytest.mark.asyncio
async def test_load_path_refuses_missing_and_invalid_files(tmp_path):
    from backend.agents.local_pool import LocalModelPool

    pool = LocalModelPool(store_dir=tmp_path)
    missing = await pool.load_path("nope.gguf", 1)
    assert not missing["success"] and "not in" in missing["error"]
    (tmp_path / "bad.gguf").write_bytes(b"not a model")
    bad = await pool.load_path("bad.gguf", 2)
    assert not bad["success"] and "GGUF" in bad["error"]


# ------------------------------------------------------ Round AV: manifest
def test_manifest_models_arrive_by_themselves(tmp_path):
    from backend.agents.local_pool import LocalModelPool
    from backend.agents.model_fetch import ManifestAutoloader, read_manifest

    entries = read_manifest(ROOT / "models.json")
    assert entries and entries[0]["file"].endswith(".gguf") and entries[0]["url"].startswith("https://huggingface.co/")
    bad = tmp_path / "bad.json"
    bad.write_text('{"models": [{"url": "nonsense"}, {"url": "hf://a/b/c.gguf", "enabled": false}]}')
    assert read_manifest(bad) == []
    # the image seed is linked into the store without a copy
    seed = tmp_path / "seed"; seed.mkdir()
    (seed / "m.gguf").write_bytes(b"GGUF" + b"\0" * 32)
    pool = LocalModelPool(store_dir=tmp_path / "store")
    good = tmp_path / "good.json"
    good.write_text('{"models": [{"url": "hf://a/b/m.gguf", "slot": 2}]}')
    auto = ManifestAutoloader(pool, good, seed_dir=seed)
    assert auto.entries[0]["slot"] == 2
    linked = auto._seed("m.gguf")
    assert linked is not None and linked.is_symlink() and linked.resolve() == (seed / "m.gguf").resolve()
    assert auto.status()["phase"] == "idle"


def test_image_bakes_runtime_and_connectome_but_not_the_weights():
    docker = (ROOT / ".devcontainer/Dockerfile").read_text()
    # Round BG: the GGUF weights left the image (1.4 GB that does not compress
    # kept "Setting up remote connection" busy); the autoloader fetches them.
    assert "tools/fetch_models.py" not in docker and "COPY models.json" not in docker
    for needle in ("llama-cpp-python", "abetlen.github.io/llama-cpp-python/whl/cpu", "/opt/drosophila/connectome",
                   "/opt/drosophila/models", "ConnectomeData"):
        assert needle in docker, needle
    wf = (ROOT / ".github/workflows/devcontainer-image.yml").read_text()
    assert "connectome_data.py" in wf
    run_sh = (ROOT / "run.sh").read_text()
    assert "adopt_image_assets()" in run_sh and "/opt/drosophila" in run_sh
    assert "abetlen.github.io/llama-cpp-python/whl/cpu" in run_sh
    cm = (ROOT / "backend/core/cycle_manager.py").read_text()
    assert "self.agents.local.autoloader.start()" in cm and "MODEL_AUTOLOAD" in cm
    html = (ROOT / "backend/web/settings.html").read_text()
    assert 'id="autoload-status"' in html


# -------------------------------------------- Round AW: violent != calm
def test_a_market_that_stays_violent_is_not_complacent():
    """User: 'even when the market was too violent it was continuously showing
    complacency'.  The tape's own last-minute scale normalises violence away;
    the hourly candle anchor must not."""
    import numpy as np
    from backend.core import emotions as E
    from backend.core.frozen_snapshot import FrozenMarketSnapshot

    rng = np.random.default_rng(3)
    n, rate = 180 * 10, 10
    t = np.arange(n) * (1000.0 / rate)
    # a tape that has been violent for 3 minutes: 90 bps per minute
    rets = rng.normal(0.0, 90.0 / 1e4 / np.sqrt(60 * rate), size=n)
    px = 60_000.0 * np.exp(np.cumsum(rets))
    side = np.where(rng.random(n) > 0.5, 1.0, -1.0)
    ticks = np.column_stack([t, px, np.ones(n), side])
    # the hour before was quiet: 1-minute closes moving ~8 bps
    calm_closes = 60_000.0 * np.exp(np.cumsum(rng.normal(0, 8.0 / 1e4, size=60)))
    book = np.zeros((2, 20, 2)); book[0, 0] = (px[-1] - 1, 5); book[1, 0] = (px[-1] + 1, 5)
    snap = FrozenMarketSnapshot(timestamp=0.0, btc_ticks=ticks, paxg_ticks=ticks, btc_book=book, paxg_book=book,
                                btc_book_prev=book, paxg_book_prev=book, btc_candles=calm_closes, paxg_candles=calm_closes,
                                news_items=(), drg_outcomes=np.zeros((0, 2)), btc_tick_count=n, paxg_tick_count=n)
    f = E.features(snap, "BTC")
    assert f["sigma_window_bps"] > 3.0 * f["sigma_hour_bps"] > 0, f
    # a stateless read has no memory to judge by: neutral
    assert f["turbulence"] == 1.0
    # the tracker remembers the quiet hour (seeded from exchange history)
    tracker = E.EmotionTracker()
    tracker.restore_anchor({"sigma_anchor_bps": f["sigma_hour_bps"], "sigma_anchor_samples": 0,
                            "sigma_anchor_seeded": True})
    report = E.analyze(snap, "BTC", tracker=tracker).to_dict()
    assert report["features"]["turbulence"] > 3.0, report["features"]["turbulence"]
    assert report["dominant"]["name"] != "COMPLACENCY", report["dominant"]
    comp = [e for e in report["emotions"] if e.get("name") == "COMPLACENCY"]
    assert not comp or comp[0]["intensity"] < 0.2, comp
    # the same violent tape in a market whose norm IS violent is just normal
    wild = E.EmotionTracker()
    wild.restore_anchor({"sigma_anchor_bps": f["sigma_window_bps"], "sigma_anchor_seeded": True})
    r2 = E.analyze(snap, "BTC", tracker=wild).to_dict()
    assert r2["features"]["turbulence"] < 1.6
    # an unseeded tracker stays silent until it has learned (~10 min)
    fresh = E.EmotionTracker()
    assert fresh.anchor_turbulence(50.0, 0.0, 0) == 1.0
    # the anchor survives a restart through its state dict
    assert E.EmotionTracker().restore_anchor(tracker.anchor_state()) is None
    # ...and real exchange history (>= 2 h of candles) seeds it at once
    seeded = E.EmotionTracker()
    assert seeded.anchor_turbulence(90.0, 8.0, 120) > 3.0


# -------------------------------------------- Round AW: memory survives a restart
def test_everything_learned_survives_a_restart(tmp_path, monkeypatch):
    """User: the app must 'permanently run and save history and really learn'
    and a reconnect must never lose history.  Two engines, one vault."""
    monkeypatch.setenv("STATE_VAULT_DIR", str(tmp_path / "vault"))
    from backend.core.cycle_manager import CycleManager
    from backend.core.state_vault import StateVault

    first = CycleManager(local_stub=True)
    first.competition.enter(41, "BUY", 100.0)
    first.competition.settle(41, 100.5)
    first.competition.enter(42, "SELL", 100.5)
    first.competition.settle(42, 100.1)
    first.stats.cycle_number = 42
    first.outcomes.append(1.0, 5.0)
    first.outcomes.append(1.0, 4.0)
    first.dopamine.observe(5.0, 1.0, 0.6)
    first._raw_outcomes.extend([1, 1])
    first.emotions.tracker.restore_anchor({"sigma_anchor_bps": 9.5, "sigma_anchor_seeded": True})
    written = first._persist_learned()
    assert all(written[k] for k in ("competition", "picker", "dopamine", "triune", "outcomes", "stats", "guard", "emotions"))

    second = CycleManager(local_stub=True)
    assert second.competition.streak == 0 and len(second.outcomes) == 0
    restored = second._restore_learned()
    assert {"competition", "outcomes", "stats", "guard", "emotions", "dopamine"} <= set(restored)
    assert second.competition.streak == 2
    assert second.stats.cycle_number == 42          # window ids keep climbing
    assert len(second.outcomes) == 2
    assert list(second._raw_outcomes) == [1, 1]
    assert second.emotions.tracker.sigma_anchor_bps == 9.5
    # the live references are kept (state is adopted, objects not swapped)
    assert second.competition is not first.competition

    # a stale vault (older than the max age) is not a live streak
    old = StateVault(tmp_path / "old")
    old.save("competition", first.competition)
    assert old.load("competition", max_age=10.0) is not None
    import os, time
    stamp = time.time() - 4 * 86_400
    os.utime(old._path("competition"), (stamp, stamp))
    # the age check reads the record's own clock, not the file's: rewrite it old
    import pickle
    with open(old._path("competition"), "rb") as fh:
        rec = pickle.load(fh)
    rec["saved_at"] = stamp
    with open(old._path("competition"), "wb") as fh:
        pickle.dump(rec, fh)
    assert old.load("competition", max_age=3 * 86_400) is None
    # the settlement path writes the vault, and the status payload reports it
    import inspect
    src = inspect.getsource(CycleManager)
    assert "await asyncio.to_thread(self._persist_learned)" in src
    assert '"vault":' in src
    js = (ROOT / "backend/web/app.js").read_text()
    assert "function vaultLabel" in js


# -------------------------------------------- Round AW: the window reaches the agents and the formulas
def test_agents_and_formulas_speak_the_70_second_window():
    from backend.agents import base, local_model_agent
    from backend.core import config as cfg

    w = int(round(cfg.window_seconds()))
    assert f"{w}-second" in base.SYSTEM_PROMPT and "unchanged price is a loss" in base.SYSTEM_PROMPT
    assert "1-minute" not in base.SYSTEM_PROMPT and "1-minute" not in local_model_agent.SYSTEM_PROMPT_LOCAL
    assert f"next {w} seconds" in base.build_cycle_prompt({"formulas": {}})
    from backend.formulas.category_c_hedge import shrp
    assert shrp.FLOW_SECONDS == cfg.window_seconds()
    from backend.core.signal_lock import FrozenSignal
    assert FrozenSignal._field_defaults["window_seconds"] == 70.0
    # DRG no longer pins at -0.98 on one ordinary loss (magnitudes are normalised)
    import numpy as np
    from backend.formulas import drg

    class Snap:
        drg_outcomes = np.array([[1.0, 8.0], [-1.0, 9.0]])
    value = drg.compute(Snap(), drg.State(), None, {})
    assert -0.5 < value < 0.0, value


# -------------------------------------------- Round AX: 31 formulas, the tie problem
def test_the_nine_window_formulas_are_distinct_and_pin_drives_the_pick():
    import numpy as np
    from backend.formulas.engine import ALL_FORMULAS, WINDOW_FORMULAS, CATEGORY_NAMES
    from backend.formulas import logic as logic_module
    from backend.core.competition import BoundaryFeatures, BoundaryPicker

    assert len(ALL_FORMULAS) == 31 and len(WINDOW_FORMULAS) == 9
    names = [s.name for s in WINDOW_FORMULAS]
    assert names == ["PIN", "MPL", "RBB", "RAR", "WDR", "PRT", "FIH", "LAI", "ANA"]
    assert CATEGORY_NAMES["I"] == "The 70-second Window"
    # every one has its own mathematics, logic entry, detail and double-check
    rules = set()
    for spec in WINDOW_FORMULAS:
        assert spec.module.DOUBLE_CHECK not in rules
        rules.add(spec.module.DOUBLE_CHECK)
        assert logic_module.get(spec.name) is not None
        assert spec.name in logic_module.DETAIL
        assert callable(spec.module.double_check)
    assert not WINDOW_FORMULAS[0].directional   # PIN is a risk, not a side
    assert all(s.directional for s in WINDOW_FORMULAS[1:])

    # PIN on a tape that does not move reads near 1; on a live tape near 0
    from backend.core.frozen_snapshot import FrozenMarketSnapshot
    from backend.formulas.category_i_window import pin
    n = 2400
    t = np.arange(n) * 100.0
    flat = np.full(n, 60000.0); flat[::7] += 0.01          # one-tick flicker
    rng = np.random.default_rng(1)
    live = 60000.0 * np.exp(np.cumsum(rng.normal(0, 3e-5, n)))
    book = np.zeros((2, 20, 2)); book[0, 0] = (59999.99, 5); book[1, 0] = (60000.01, 5)
    def snap(px):
        ticks = np.column_stack([t, px, np.ones(n), np.where(rng.random(n) > 0.5, 1.0, -1.0)])
        return FrozenMarketSnapshot(timestamp=0.0, btc_ticks=ticks, paxg_ticks=ticks, btc_book=book, paxg_book=book,
                                    btc_book_prev=book, paxg_book_prev=book, btc_candles=np.zeros(0), paxg_candles=np.zeros(0),
                                    news_items=(), drg_outcomes=np.zeros((0, 2)), btc_tick_count=n, paxg_tick_count=n)
    p_flat = pin.compute(snap(flat), "BTC", pin.State(), {}, {})
    p_live = pin.compute(snap(live), "BTC", pin.State(), {}, {})
    assert p_flat > 0.6 > 0.2 > p_live, (p_flat, p_live)

    # with a high tie risk the pick follows the microstructure, not the recipe
    picker = BoundaryPicker()
    f = BoundaryFeatures(entry=60000.0, entry_side=0.0, microprice=59999.997, gap_ticks=-0.3, r3_ticks=0.0,
                         ofi5=-0.1, book_imbalance=-0.1, pin_rate=0.8, tick=0.01, prints_5s=3)
    pick = picker.pick(1, f, "BUY", 0.62)
    assert pick.side == "SELL" and "tie risk 80%" in pick.note
    f2 = f.__class__(**{**f.__dict__, "pin_rate": 0.0})
    assert picker.pick(2, f2, "BUY", 0.62).side == "BUY"


def test_paralysis_reads_a_frozen_tape_and_the_emotion_ledger_scores_windows():
    from backend.core import emotions as E

    # a tape that has not moved: every window ties -> PARALYSIS dominates
    frozen = {"z_1s": 0.0, "z_5s": 0.0, "z_win": 0.0, "aggression_5s": 0.0, "vol_ratio": 0.6,
              "tick_surge": 0.2, "spread_blowout": 1.0, "spread_bps": 0.3, "range_position": 0.5,
              "pin_prob": 0.7, "flat_share": 0.6, "turbulence": 0.8}
    scores = {s.name: s for s in E.score_emotions(frozen)}
    assert "PARALYSIS" in scores and scores["PARALYSIS"].family == "freeze"
    assert scores["PARALYSIS"].intensity > 0.6
    assert scores["PARALYSIS"].intensity > scores["COMPLACENCY"].intensity
    assert 3 <= len(scores["PARALYSIS"].drivers) <= 4
    # the same calm numbers on a live tape (no pin) are complacency, not paralysis
    live = dict(frozen, pin_prob=0.01, flat_share=0.0, tick_surge=1.0)
    live_scores = {s.name: s for s in E.score_emotions(live)}
    assert live_scores["COMPLACENCY"].intensity > live_scores["PARALYSIS"].intensity
    # and a surprise kills it
    moving = dict(frozen, z_1s=2.5, z_5s=2.0)
    assert {s.name: s for s in E.score_emotions(moving)}["PARALYSIS"].intensity < 0.25

    # the pin statistics on a real tick array: a flat tape vs a drifting one
    t = np.arange(0, 400_000, 100, dtype=float)
    flat = np.full(t.size, 60000.0); flat[::9] += 0.01
    stats = E._pin_statistics(t, flat)
    assert stats["tick_size"] == pytest.approx(0.01) and stats["pin_prob"] > 0.5
    drift = 60000.0 + np.cumsum(np.random.default_rng(3).normal(0, 0.5, t.size)).round(2)
    assert E._pin_statistics(t, drift)["pin_prob"] < 0.1

    # the learned ledger: dominant at the boundary -> outcome of the window
    ledger = E.EmotionOutcomes()
    for i in range(10):
        ledger.record(i, "FEAR", 0.5)
        ledger.settle(i, 100.0, 99.0 if i < 9 else 101.0)
    for i in range(10, 20):
        ledger.record(i, "PARALYSIS", 0.7)
        ledger.settle(i, 100.0, 100.0)
    summary = ledger.summary()
    assert summary["windows_settled"] == 20
    assert summary["by_emotion"]["FEAR"]["verdict"].startswith("leans DOWN")
    assert summary["by_emotion"]["PARALYSIS"]["tie_rate"] == 1.0
    assert summary["by_emotion"]["PARALYSIS"]["verdict"].startswith("ties")
    assert summary["by_emotion"]["HOPE"]["verdict"] == "learning"
