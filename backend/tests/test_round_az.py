"""Round AZ - the real brain everywhere the user looks.

* the Brain Matrix page is FlyWire-first (block matrices of the measured
  connectome); the 80x80 stand-in is a labelled fallback section only;
* the connectome build is memory-lean and runs in a worker process so an
  OOM kill cannot take the engine down (the reason a 4 GB Codespace only
  ever showed the stand-in);
* a key written to .env wins over an empty variable of the same name, and
  the dashboard re-reads the key states periodically.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from backend.brain import whole_brain as wb
from backend.tests.test_round_aq import _tiny_connectome

ROOT = Path(__file__).resolve().parents[2]
WEB = ROOT / "backend" / "web"


def test_block_matrix_is_a_population_matrix_of_the_real_graph(tmp_path):
    g = wb.ConnectomeGraph(_tiny_connectome(tmp_path), "tiny test connectome", hops=3)
    block = g.block_matrix()
    k = len(block["labels"])
    assert k >= 28 + 2 and len(block["matrix"]) == k and all(len(row) == k for row in block["matrix"])
    assert sum(block["sizes"]) == g.n                      # every neuron sits in exactly one population
    assert block["labels"][0].startswith("PN ") and "Kenyon cells" in block["labels"]
    # the sum of all blocks equals the sum of the whole matrix (nothing lost)
    assert abs(float(np.sum(block["matrix"])) - float(g.W.sum())) < 1e-3
    assert block is g.block_matrix()                       # cached
    whole = wb.ConnectomeGraph(_tiny_connectome(tmp_path), "tiny whole brain", hops=6).block_matrix()
    assert all(label in ("optic", "central", "sensory", "visual_projection", "ascending", "descending",
                         "sensory_ascending", "visual_centrifugal", "motor", "endocrine", "unclassified")
               for label in whole["labels"])


def test_matrix_page_is_flywire_first_and_the_stand_in_is_a_fallback_section():
    html = (WEB / "matrix.html").read_text()
    js = (WEB / "matrix.js").read_text()
    assert html.index('id="flywire-card"') < html.index('id="mb-grid"') < html.index('id="fallback-card"')
    assert "stand-in (fallback only)" in html and 'id="wb-readout"' in html
    assert "/api/brain/matrix/flywire" in js and "function buildBlockGrid" in js
    assert "neuPrint" not in js                            # the old "not connected" note is gone


def test_flywire_matrix_route_exists():
    src = (ROOT / "backend" / "api" / "routes_brain.py").read_text()
    assert '@router.get("/api/brain/matrix/flywire")' in src


def test_connectome_build_runs_in_a_worker_and_is_memory_lean():
    src = (ROOT / "backend" / "brain" / "connectome_data.py").read_text()
    assert "def build_in_worker" in src and "build_in_worker(self.data_dir, self.status)" in src
    assert "subprocess.run([sys.executable" in src
    assert 'column("Presynaptic_Index", np.int32)' in src and "whole.data *= scale[row_of]" in src


def test_dotenv_value_beats_an_empty_environment_variable(tmp_path, monkeypatch):
    from backend.core import config as cfg

    env_file = tmp_path / ".env"
    env_file.write_text("ROUND_AZ_TEST_KEY=from-dotenv\nROUND_AZ_TEST_KEEP=from-dotenv\n")
    monkeypatch.setattr(cfg, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(cfg, "BACKEND_DIR", tmp_path / "backend")
    monkeypatch.setenv("ROUND_AZ_TEST_KEY", "")
    monkeypatch.setenv("ROUND_AZ_TEST_KEEP", "explicit")
    cfg._load_dotenv()
    assert os.environ["ROUND_AZ_TEST_KEY"] == "from-dotenv"
    assert os.environ["ROUND_AZ_TEST_KEEP"] == "explicit"


def test_dashboard_rereads_key_states():
    js = (WEB / "app.js").read_text()
    assert "refreshKeyStates();" in js.split("async function safetyNet()")[1].split("\n\n")[0] or \
        "if (updateTick % 6 === 2) refreshKeyStates();" in js
    assert 'document.addEventListener("visibilitychange"' in js
