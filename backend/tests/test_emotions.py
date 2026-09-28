"""Round J: the crowd's emotions, from microseconds to minutes.

The user's premise: *"60 second timeframe market easily get manipulated by
retailers emotions"* - so the engine should read the crowd's emotions
continuously (µs to seconds), name the dominant one live in the UI, and let it
shape the prediction.  Each requirement has a test here:

1.  **Eight named emotions**, each an intensity in [0, 1], each decomposed over
    five timescales (micro / seconds / window / minutes / news).
2.  **The right emotion on the right tape**: euphoria or FOMO on a melt-up, fear
    or panic on an impulse down, complacency on a flat tape.
3.  **A manipulation read** in [0, 1] with a named mechanism.
4.  **Continuity**: a tracker that smooths, remembers how long the dominant
    emotion has held, and counts switches.
5.  **It shapes the prediction**: the fusion dampens confidence on a crowded
    minute but never flips the side; the reasoning names the crowd.
6.  **It is live**: the manager samples the tape on its own cadence, streams an
    ``EMOTION`` message, and every payload carries the block the UI draws.
7.  **The UI draws it**: the dashboard has the panel and the stream handler, and
    the Flutter client has the model.
"""

from __future__ import annotations

import asyncio
import re
import time
from pathlib import Path

import pytest
import pytest_asyncio

from backend.agents import fusion as fusion_module
from backend.core import config as cfg
from backend.core import emotions as E
from backend.core.cycle_manager import CycleManager
from backend.formulas import synthetic

ROOT = Path(__file__).resolve().parents[2]
APP_JS = ROOT / "backend" / "web" / "app.js"
INDEX = ROOT / "backend" / "web" / "index.html"
STYLES = ROOT / "backend" / "web" / "styles.css"
CYCLE_PY = ROOT / "backend" / "core" / "cycle_manager.py"
DART_MODEL = ROOT / "frontend" / "lib" / "models" / "signal.dart"
DART_STATE = ROOT / "frontend" / "lib" / "state" / "app_state.dart"
DART_PANEL = ROOT / "frontend" / "lib" / "widgets" / "emotion_panel.dart"

SETTINGS = cfg.Settings(time_scale=1.0)


def _last(key: str):
    return synthetic.scenario(key).snapshots[-1]


def _report(key: str, tracker: E.EmotionTracker | None = None) -> dict:
    return E.analyze(_last(key), "BTC", tracker=tracker).to_dict()


# ---------------------------------------------------------------------------
# 1. the vocabulary
# ---------------------------------------------------------------------------
def test_there_are_eight_emotions_over_five_timescales() -> None:
    assert len(E.EMOTIONS) == 8
    assert set(E.EMOTIONS) == {
        "FEAR", "PANIC", "CAPITULATION", "DENIAL", "HOPE", "EUPHORIA", "FOMO", "COMPLACENCY",
    }
    assert E.TIMESCALES == ("micro", "seconds", "window", "minutes", "news")
    # the families the user named: fear, happiness, withdrawal - and the rest
    families = set(E.FAMILIES.values())
    assert {"fear", "happiness", "withdrawal"} <= families
    assert set(E.CONVICTION_HINT) == set(E.EMOTIONS)


def test_a_reading_scores_every_emotion_on_every_timescale() -> None:
    payload = _report("BULL")
    assert payload["available"] is True
    names = [item["name"] for item in payload["emotions"]]
    assert sorted(names) == sorted(E.EMOTIONS)
    for item in payload["emotions"]:
        assert 0.0 <= item["intensity"] <= 1.0
        assert item["percent"] == pytest.approx(item["intensity"] * 100.0, abs=0.06)
        assert set(item["by_timescale"]) == set(E.TIMESCALES)
        assert all(0.0 <= v <= 1.0 for v in item["by_timescale"].values())
        assert item["dominant_timescale"] in E.TIMESCALES
        assert item["tone"] in ("negative", "positive", "neutral")
        assert len(item["drivers"]) == 3, "every emotion prints the numbers behind it"
    # ranked, dominant first
    intensities = [item["intensity"] for item in payload["emotions"]]
    assert intensities == sorted(intensities, reverse=True)
    assert payload["dominant"]["name"] == names[0]


def test_the_reading_is_measured_in_microseconds() -> None:
    payload = _report("BULL")
    assert payload["at_us"] > 1_600_000_000_000_000
    assert payload["at"].endswith("Z") and re.search(r"\.\d{6}Z$", payload["at"])
    assert payload["resolution_us"] > 0
    assert payload["resolution_label"].endswith(("µs", "ms", "s"))
    assert payload["ticks"] >= 15
    assert payload["span_us"] > 0


# ---------------------------------------------------------------------------
# 2. the right emotion on the right tape
# ---------------------------------------------------------------------------
def test_a_melt_up_reads_as_a_chase() -> None:
    for key in ("BULL", "IMPULSE_UP", "STRESS"):
        payload = _report(key)
        top = payload["dominant"]
        assert top["name"] in ("EUPHORIA", "FOMO", "HOPE"), (key, top)
        assert top["tone"] == "positive"
        assert payload["tone_bias"] > 0.2, (key, payload["tone_bias"])


def test_an_impulse_down_reads_as_fear() -> None:
    payload = _report("IMPULSE_DOWN")
    top = payload["dominant"]
    assert top["name"] in ("FEAR", "PANIC", "CAPITULATION"), top
    assert top["tone"] == "negative"
    assert payload["tone_bias"] < -0.2
    intensity = {item["name"]: item["intensity"] for item in payload["emotions"]}
    # panic is the acute form: it is loud on a violent tape ...
    assert intensity["PANIC"] > 0.4
    # ... and the chase family is quiet
    assert intensity["EUPHORIA"] < 0.2


def test_a_bear_tape_reads_as_fear() -> None:
    payload = _report("BEAR")
    assert payload["dominant"]["tone"] == "negative"
    assert payload["tone_bias"] < 0


def test_a_flat_tape_reads_as_complacency() -> None:
    for key in ("FLAT", "CYCLE_UP", "CYCLE_DOWN"):
        payload = _report(key)
        assert payload["dominant"]["name"] == "COMPLACENCY", (key, payload["dominant"])
        assert abs(payload["tone_bias"]) < 0.35, (key, payload["tone_bias"])


def test_the_scales_are_the_tapes_own_not_absolute() -> None:
    """A move counts in multiples of the tape's typical move, so the same
    reading cannot be bought by simply being a more volatile asset."""
    f = E.features(_last("IMPULSE_DOWN"), "BTC")
    assert f["scale_1s_bps"] > 0 and f["scale_5s_bps"] > 0 and f["scale_60s_bps"] > 0
    assert f["z_1s"] == pytest.approx(f["return_1s_bps"] / f["scale_1s_bps"], rel=1e-6)
    assert f["z_5s"] < -2.0, "an impulse is several typical moves"
    flat = E.features(_last("FLAT"), "BTC")
    assert abs(flat["z_5s"]) < 1.0


def test_the_read_sentence_names_the_dominant_emotion_and_a_hint() -> None:
    payload = _report("IMPULSE_UP")
    assert payload["dominant"]["label"] in payload["read"]
    assert "timescale" in payload["read"]
    assert payload["hint"] == E.CONVICTION_HINT[payload["dominant"]["name"]]


# ---------------------------------------------------------------------------
# 3. manipulation
# ---------------------------------------------------------------------------
def test_the_manipulation_read_is_bounded_and_named() -> None:
    for key in ("FLAT", "BULL", "BEAR", "STRESS", "IMPULSE_DOWN", "IMPULSE_UP"):
        block = _report(key)["manipulation"]
        assert 0.0 <= block["score"] <= 1.0, (key, block)
        assert block["kind"] in (
            "none", "retail chase", "stop hunt", "whipsaw", "book imbalance",
            "momentum ignition", "toxic flow", "quote stuffing", "spoofing",
        )
        assert set(block["components"]) == {
            "herding", "whipsaw", "stop_hunt", "thin_book", "volume_climax",
            "ignition", "toxicity", "stuffing", "spoofing", "pushable",
        }
        assert all(0.0 <= v <= 1.0 for v in block["components"].values()), (key, block)
        assert len(block["evidence"]) == 7
        assert block["note"]


def test_a_one_sided_impulse_is_more_crowded_than_a_flat_tape() -> None:
    impulse = _report("IMPULSE_DOWN")["manipulation"]
    flat = _report("CYCLE_UP")["manipulation"]
    assert impulse["score"] > flat["score"]
    assert impulse["components"]["herding"] > 0.8, "every print on the same side is a crowd"
    assert flat["kind"] == "none"


# ---------------------------------------------------------------------------
# 4. continuity
# ---------------------------------------------------------------------------
def test_the_tracker_smooths_holds_and_counts_switches() -> None:
    tracker = E.EmotionTracker(asset="BTC")
    base = 1_700_000_000_000_000
    flat = synthetic.scenario("FLAT").snapshots[-4:]
    # a *sustained* impulse: the same violent tape, sampled six times (3 s)
    impulse = [synthetic.scenario("IMPULSE_DOWN").snapshots[-1]] * 6
    seen = []
    for index, snap in enumerate(flat + impulse):
        report = E.analyze(snap, "BTC", tracker=tracker, at_us=base + index * 500_000)
        seen.append(report.to_dict())
    first, last = seen[0], seen[-1]
    assert first["held_seconds"] == 0.0 and first["samples"] == 1
    assert last["samples"] == len(seen)
    assert first["dominant"]["name"] == "COMPLACENCY"
    assert last["dominant"]["tone"] == "negative"
    # the dominant emotion switched exactly once - hysteresis, not flicker -
    # and within two seconds of the impulse, and the panel can say when
    assert last["churn_per_minute"] == 1
    switched_at = next(i for i, s in enumerate(seen) if s["dominant"]["tone"] == "negative")
    assert len(flat) <= switched_at <= len(flat) + 4, switched_at
    assert last["held_seconds"] == pytest.approx(0.5 * (len(seen) - 1 - switched_at), abs=1e-6)
    assert len(last["history"]) == len(seen)
    assert all(0.0 <= s["manipulation"] <= 1.0 for s in last["history"])
    # smoothing: the state is an EMA, so no bar can jump the full distance
    intensities = [{i["name"]: i["intensity"] for i in s["emotions"]} for s in seen]
    jumps = [abs(b["FEAR"] - a["FEAR"]) for a, b in zip(intensities, intensities[1:])]
    assert max(jumps) < 0.5


def test_the_headline_waits_for_the_crowd_to_mean_it() -> None:
    """Hysteresis: a challenger one point ahead does not take the headline."""
    tracker = E.EmotionTracker(asset="BTC")
    tracker.emotions = {name: 0.10 for name in E.EMOTIONS}
    tracker.emotions["COMPLACENCY"] = 0.50
    tracker.last_dominant = "COMPLACENCY"
    # one point ahead, once: still complacency
    tracker.emotions["FEAR"] = 0.51
    assert tracker._elect(sorted(tracker.emotions.items(), key=lambda kv: kv[1], reverse=True)) == "COMPLACENCY"
    # clearly ahead, but only for one sample: still complacency
    tracker.emotions["FEAR"] = 0.60
    ranked = sorted(tracker.emotions.items(), key=lambda kv: kv[1], reverse=True)
    assert tracker._elect(ranked) == "COMPLACENCY"
    # clearly ahead for a second consecutive sample: fear takes the headline
    assert tracker._elect(ranked) == "FEAR"


def test_the_streamed_form_is_compact_but_complete() -> None:
    payload = _report("BULL", tracker=E.EmotionTracker(asset="BTC"))
    small = E.compact(payload)
    for key in ("emotions", "dominant", "runner_up", "tone_bias", "manipulation", "read", "hint", "held_seconds", "at_us"):
        assert key in small, key
    assert "features" not in small and "history" not in small
    assert set(small["manipulation"]) <= {"score", "percent", "kind", "note", "components"}
    assert len(str(small)) < len(str(payload))


# ---------------------------------------------------------------------------
# 5. it shapes the prediction
# ---------------------------------------------------------------------------
def _fuse(crowd: dict | None):
    return fusion_module.fuse(
        agents={},
        ccs_value=0.6,
        ccs_confidence=0.8,
        hsi=0.1,
        settings=SETTINGS,
        formula_consensus=0.5,
        consensus_voters=10,
        crowd=crowd,
    )


def test_a_crowded_minute_dampens_confidence_but_never_flips_the_side() -> None:
    calm = _fuse({"manipulation": {"score": 0.10, "kind": "none"}, "dominant": {"label": "Complacency"}})
    hot = _fuse({"manipulation": {"score": 0.90, "kind": "retail chase"}, "dominant": {"label": "Panic"}})
    none = _fuse(None)
    assert calm.decision == hot.decision == none.decision == "BUY"
    assert calm.crowd_adjustment == 1.0 and none.crowd_adjustment == 1.0
    assert hot.crowd_adjustment < 1.0
    assert hot.confidence < calm.confidence
    # bounded by the configured maximum
    assert hot.crowd_adjustment >= 1.0 - SETTINGS.emotion_dampen_max - 1e-9
    assert "crowd:" in hot.reasoning and "panic" in hot.reasoning.lower()
    assert "crowd:" not in calm.reasoning
    payload = hot.to_dict()
    assert payload["crowd_adjustment"] == pytest.approx(hot.crowd_adjustment)
    assert "retail chase" in payload["crowd_note"]


def test_the_dampener_starts_at_the_threshold_and_grows_with_the_score() -> None:
    threshold = SETTINGS.emotion_dampen_threshold
    at = _fuse({"manipulation": {"score": threshold, "kind": "whipsaw"}})
    above = _fuse({"manipulation": {"score": (threshold + 1.0) / 2, "kind": "whipsaw"}})
    top = _fuse({"manipulation": {"score": 1.0, "kind": "whipsaw"}})
    assert at.crowd_adjustment == pytest.approx(1.0)
    assert 1.0 > above.crowd_adjustment > top.crowd_adjustment
    assert top.crowd_adjustment == pytest.approx(1.0 - SETTINGS.emotion_dampen_max)


def test_the_reasoning_names_the_crowd() -> None:
    from backend.core.prediction import build_reasoning

    crowd = _report("IMPULSE_DOWN")
    reasoning = build_reasoning(
        side="SELL", confidence=0.7, conviction="HIGH", formula_values={}, crowd=crowd,
    )
    bullets = [b for b in reasoning["bullets"] if b["kind"] == "crowd"]
    assert len(bullets) == 1
    text = bullets[0]["text"]
    assert crowd["dominant"]["label"].lower() in text.lower()
    assert "timescale" in text
    # a fearful crowd leans with a SELL ...
    calm_crowd = dict(crowd, manipulation={"score": 0.1, "kind": "none"})
    reasoning = build_reasoning(
        side="SELL", confidence=0.7, conviction="HIGH", formula_values={}, crowd=calm_crowd,
    )
    assert [b for b in reasoning["bullets"] if b["kind"] == "crowd"][0]["supports"] is True
    # ... and against a BUY
    reasoning = build_reasoning(
        side="BUY", confidence=0.7, conviction="HIGH", formula_values={}, crowd=calm_crowd,
    )
    assert [b for b in reasoning["bullets"] if b["kind"] == "crowd"][0]["supports"] is False


def test_the_prediction_detail_carries_the_crowd() -> None:
    from backend.core.prediction import detail

    block = detail(side="BUY", crowd={"available": True, "dominant": "Euphoria", "percent": 71.0})
    assert block["crowd"]["dominant"] == "Euphoria"
    assert detail(side="BUY")["crowd"] == {}


# ---------------------------------------------------------------------------
# 6. it is live
# ---------------------------------------------------------------------------
@pytest_asyncio.fixture(loop_scope="module", scope="module")
async def manager():
    instance = CycleManager(cfg.Settings(), local_stub=True)
    await instance.start()
    instance.mark_started()
    deadline = time.time() + 8.0
    while time.time() < deadline and instance.lock.try_get_current() is None:
        await asyncio.sleep(0.1)
    try:
        yield instance
    finally:
        await instance.stop()


async def test_the_manager_samples_the_live_tape_on_its_own_cadence(manager: CycleManager):
    deadline = time.time() + 4.0
    while time.time() < deadline and manager.emotions.samples < 3:
        await asyncio.sleep(0.1)
    assert manager.emotions.samples >= 3, "the emotion loop must keep sampling between PULSE marks"
    assert manager.emotions.errors == 0
    payload = manager.emotions_payload()
    assert payload["available"] is True
    assert payload["asset"] == manager.asset
    assert payload["dominant"]["name"] in E.EMOTIONS
    assert payload["interval_seconds"] == manager.settings.emotion_interval_seconds
    assert "locked" in payload and "dampening" in payload
    assert set(payload["dampening"]) == {"threshold", "max", "applied", "note"}
    # the frozen reading behind the locked signal is there too, and it is the
    # same measurement (same vocabulary) taken on the immutable snapshot
    locked = payload["locked"]
    assert locked and locked["available"] is True
    assert {item["name"] for item in locked["emotions"]} == set(E.EMOTIONS)


async def test_the_emotion_stream_arrives_between_pulses(manager: CycleManager):
    queue = manager.subscribe()
    try:
        emotions, pulses = [], 0
        deadline = time.time() + 4.0
        while time.time() < deadline and len(emotions) < 3:
            try:
                message = await asyncio.wait_for(queue.get(), timeout=1.0)
            except asyncio.TimeoutError:
                continue
            if message.get("type") == "EMOTION":
                emotions.append(message["data"])
            elif message.get("type") == "PULSE":
                pulses += 1
    finally:
        manager.unsubscribe(queue)
    assert len(emotions) >= 3, "the crowd's mood must stream on its own cadence"
    data = emotions[0]
    assert set(data) >= {"cycle_number", "asset", "locked_side", "emotions", "dampening"}
    assert "signal" not in data, "an emotion message must not impersonate a signal payload"
    block = data["emotions"]
    assert block["available"] is True
    assert "features" not in block and "history" not in block, "the stream is the compact form"
    assert len(block["emotions"]) == 8
    assert block["dominant"]["label"]
    # consecutive readings are half a second apart, in microseconds
    gaps = [b["emotions"]["at_us"] - a["emotions"]["at_us"] for a, b in zip(emotions, emotions[1:])]
    assert all(g > 0 for g in gaps)
    assert min(gaps) >= manager.settings.emotion_interval_seconds * 1e6 * 0.8


async def test_every_snapshot_and_pulse_carries_the_emotion_block(manager: CycleManager):
    snapshot = manager.snapshot_payload()
    assert "emotions" in snapshot
    assert snapshot["emotions"]["available"] is True
    prediction = snapshot["prediction"]
    assert "crowd" in prediction["detail"]
    kinds = {b["kind"] for b in prediction["reasoning"]["bullets"]}
    assert "crowd" in kinds
    assert "crowd_dampening" in prediction["detail"]["confidence_parts"]
    # the PULSE builder lists it next to the other panels
    source = CYCLE_PY.read_text()
    loop = source.split("async def _heartbeat_loop", 1)[1].split("async def ", 1)[0]
    assert '"emotions": self.emotions_payload()' in loop


async def test_the_locked_reading_is_taken_on_the_frozen_snapshot(manager: CycleManager):
    assert manager.emotion_locked, "lock time records the crowd"
    again = E.analyze(manager.last_snapshot, manager.asset).to_dict()
    # same immutable inputs -> same emotions (only the wall-clock stamp differs)
    before = {i["name"]: i["intensity"] for i in manager.emotion_locked["emotions"]}
    after = {i["name"]: i["intensity"] for i in again["emotions"]}
    assert before == pytest.approx(after, abs=1e-9)


def test_the_route_exists() -> None:
    from backend.api import main as api_main

    paths = {getattr(route, "path", "") for route in api_main.app.routes}
    from backend.api import routes_emotions

    paths |= {getattr(route, "path", "") for route in routes_emotions.router.routes}
    assert "/api/emotions" in paths
    assert "/api/emotions/history" in paths


# ---------------------------------------------------------------------------
# 7. the UI draws it
# ---------------------------------------------------------------------------
def test_the_dashboard_has_the_emotion_panel_and_the_stream_handler() -> None:
    html = INDEX.read_text()
    for element_id in (
        "emotion-card", "emotion-dominant", "emotion-bars", "emotion-tone-marker",
        "emotion-manip-bar", "emotion-timescales", "emotion-locked", "emotion-held",
        "w-crowd-line", "w-crowd-text",
    ):
        assert f'id="{element_id}"' in html, element_id
    js = APP_JS.read_text()
    assert 'case "EMOTION"' in js
    assert "function renderEmotions(" in js
    assert "function adoptEmotions(" in js
    # the stream repaints only its own panel - not the whole page
    handler = js.split('case "EMOTION"', 1)[1].split("break;", 1)[0]
    assert "renderEmotions()" in handler and "renderAll()" not in handler
    # ... and the client still has exactly one timer (the safety net)
    assert len(re.findall(r"setInterval\(([^,]+),", js)) == 1
    css = STYLES.read_text()
    for cls in (".emotion-card", ".emotion-row", ".tone-gauge", ".timescale-cell", ".crowd-dot"):
        assert cls in css, cls


def test_the_flutter_client_has_the_model_and_the_panel() -> None:
    model = DART_MODEL.read_text()
    assert "class EmotionReading" in model and "class EmotionScore" in model
    assert "class CrowdEmotions" in model
    state = DART_STATE.read_text()
    assert "'EMOTION'" in state and "emotions" in state
    panel = DART_PANEL.read_text()
    assert "class EmotionPanel" in panel
    assert "dominant" in panel and "manipulation" in panel


# ---------------------------------------------------------------------------
# Round L: the emotions are printed formulas and are cross-checked against the
# 22 formulas, which keep the vote
# ---------------------------------------------------------------------------


def test_every_emotion_carries_its_formula_and_live_terms():
    from backend.core import emotions as E
    from backend.formulas import synthetic

    snap = synthetic.scenario("IMPULSE_DOWN").snapshots[-1]
    report = E.analyze(snap, "BTC")
    assert report.available
    for score in report.scores:
        assert score.formula.startswith(score.name + " =")
        assert score.terms, score.name
        weights = sum(t["weight"] for t in score.terms)
        assert weights == pytest.approx(1.0, abs=1e-6)
        # the printed terms reproduce the ramp reading (before gate + band lift)
        raw = sum(t["contribution"] for t in score.terms)
        assert 0.0 <= raw <= 1.0
        assert 0.0 <= score.gate <= 1.0
        for term in score.terms:
            assert term["term"]
            assert 0.0 <= term["value"] <= 1.0 + 1e-9
    payload = report.to_dict()
    assert set(payload["formula_glossary"]) >= {"sell", "buy", "down_1s", "top", "bottom"}
    assert payload["emotions"][0]["formula"]


def test_formula_agreement_names_the_verdict_and_who_has_the_vote():
    from backend.core.emotions import formula_agreement

    aligned = formula_agreement({"score": 0.5, "voters": 10, "up": 8, "down": 2}, 0.4)
    assert aligned["verdict"] == "aligned" and aligned["alignment"] == pytest.approx(0.4)
    conflict = formula_agreement({"score": 0.5, "voters": 10, "up": 8, "down": 2}, -0.3)
    assert conflict["verdict"] == "conflict" and conflict["alignment"] == pytest.approx(-0.3)
    assert "formulas keep the vote" in conflict["note"]
    silent = formula_agreement({"score": 0.9, "voters": 2}, 0.9)
    assert silent["verdict"] == "formulas silent" and not silent["available"]
    flat = formula_agreement({"score": -0.4, "voters": 6}, 0.0)
    assert flat["verdict"] == "crowd flat" and flat["formula_side"] == "SELL"
    assert aligned["weights"]["formulas"] == 0.40
    assert aligned["weights"]["crowd_max_confidence_cut"] == 0.25


def test_the_live_reading_is_checked_against_the_formulas():
    from backend.core import emotions as E
    from backend.formulas import synthetic

    snap = synthetic.scenario("IMPULSE_UP").snapshots[-1]
    directional = {"CCSv2": "core", "MPS": "momentum", "DRG": "drosophila", "TQD": "flow"}
    report = E.analyze(
        snap, "BTC",
        formulas={"CCSv2": 0.7, "MPS": 0.5, "DRG": 0.6, "TQD": 0.4},
        directional=directional,
    )
    agree = report.formula_agreement
    assert agree["available"] and agree["voters"] == 4 and agree["formula_side"] == "BUY"
    assert report.deep["formula_voters"] == 4 and report.deep["formula_consensus"] > 0
    assert report.deep["evidence"]["formula_up"] > 0
    assert report.deep["chain"][-1]["name"] == "22-formula cross-check"
    assert agree["note"] in report.read()
    streamed = E.compact(report.to_dict())
    assert streamed["formula_agreement"]["verdict"] == agree["verdict"]
