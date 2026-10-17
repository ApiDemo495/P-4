"""Round BC - 70-second windows everywhere, and the superposition: every
source is carried as |up>/|down>/|tie> of the NEXT window from its measured
and projected reading; the engine acts on the collapsed value."""
from __future__ import annotations

from pathlib import Path

from backend.core import config as cfg
from backend.core.forward import ForwardExpectancy
from backend.core.superposition import Superposition, project

ROOT = Path(__file__).resolve().parents[2]


def test_the_window_is_seventy_seconds_everywhere():
    assert cfg.Settings(time_scale=1.0).cycle_seconds == 70.0 and cfg.SYNC_WINDOW_SECONDS == 70
    text = (ROOT / "README.md").read_text()
    assert "70 s" in text or "70-second" in text
    env = (ROOT / ".env.example").read_text()
    assert "CYCLE_SECONDS=70.0" in env


def test_projection_reads_a_rising_source_where_it_is_going():
    hist = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    projected, phi = project(hist, 0.7)
    assert projected > 0.7 and phi > 0.5
    flat, phi0 = project([0.3, 0.3, 0.3], 0.3)
    assert flat == 0.3 and phi0 == 0.0            # too short a history: no projection


def test_superposition_collapses_to_the_next_window_not_the_reading():
    fx = ForwardExpectancy()
    sp = Superposition(fx)
    for i in range(60):                             # a +0.5 reading followed by DOWN 75 % of the time
        fx.observe(i, {"TAI": 0.5}, 1)
        fx.settle(i, i % 4 == 0, 2.0 if i % 4 == 0 else -2.0)
    st = sp.state("TAI", 0.5, 1)
    assert st.covered and st.p_down > st.p_up and st.collapsed < 0 < st.measured
    assert 0.0 < st.coherence <= 1.0 and abs(st.p_up + st.p_down + st.p_tie - 1.0) < 1e-6
    acted = sp.collapse({"TAI": 0.5, "AFPR": 0.2}, 1)
    assert acted["TAI"] == st.collapsed and acted["AFPR"] == 0.2    # uncovered: measured
    summary = sp.summary()
    assert "TAI" in summary["against_their_reading"] and summary["covered"] == 1


def test_everything_is_wired_through_the_superposition():
    cm = (ROOT / "backend" / "core" / "cycle_manager.py").read_text()
    assert "future_values = self.superposition.collapse(" in cm
    assert "agreement = consensus(future_values" in cm and "formula_values=future_values" in cm
    assert "self.formulas.forward_hook = self._brain_future_vector" in cm
    assert "physics_report = self._physics_future(physics_report, last_dir)" in cm
    assert "self.forward.observe(signal.cycle_number, self._all_sources(" in cm
    assert '"phys:vote"' in cm and 'f"emo:{e.get(' in cm
    js = (ROOT / "backend" / "web" / "app.js").read_text()
    assert "state.liveFuture" in js and "|up⟩" in js and "measured now" in js
