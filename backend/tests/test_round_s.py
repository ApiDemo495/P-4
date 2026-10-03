"""Round S - named connectome, token-only neuPrint client, in-app docs,
autostart visibility."""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

from backend.brain import connectome, graph_convolution as gc, matrix_builder as mb
from backend.brain.generate_fallback_matrix import build_matrix
from backend.brain import query_circuits as qc

ROOT = Path(__file__).resolve().parents[2]


def test_every_node_is_a_named_hemibrain_cell_type():
    c = connectome.default()
    assert len(c.neurons) == gc.N_NODES
    names = [n.name for n in c.neurons]
    assert not any(re.match(r"^(KC_cluster_\d+|node \d+)$", n) for n in names)
    assert all(n.cell_type and n.transmitter and n.compartment for n in c.neurons)
    # layout matches the GCN constants
    assert all(n.population == "PN" for n in c.neurons[:20])
    assert all(n.population == "KC" for n in c.neurons[gc.KC_START : gc.KC_END])
    assert c.neurons[gc.PAM].population == "DAN" and c.neurons[gc.OA].population == "OA"
    assert c.neurons[gc.MBON_APPROACH].population == "MBON"
    assert c.neurons[gc.LH_NEUTRAL].population == "LH"


def test_synapses_carry_counts_sign_and_pathway():
    c = connectome.default()
    assert c.synapses
    for s in c.synapses:
        assert s.synapses > 0 and s.sign in (-1, 1) and s.pathway
        assert 0 <= s.source < gc.N_NODES and 0 <= s.target < gc.N_NODES
    t = c.totals()
    assert t["inhibitory_edges"] > 0 and t["excitatory_edges"] > t["inhibitory_edges"]
    assert t["cells_represented"] > 2000  # ~2000 Kenyon cells are really represented


def test_fallback_csv_is_the_named_connectome():
    csv_matrix = mb.load_csv(mb.default_fallback_path())
    assert np.allclose(csv_matrix, build_matrix(), atol=1e-5)
    text = (ROOT / "backend/brain/fallback/mb_adjacency_80x80.csv").read_text()
    assert "KC_cluster_" not in text and "KCg" in text and "MBON" in text
    assert mb.NODE_TYPES == [n.name for n in connectome.default().neurons]


def test_offline_counts_are_never_presented_as_live():
    d = connectome.default().to_dict(edge_limit=5)
    assert d["live"] is False and "typical" in d["counts_note"]
    assert len(d["synapses"]) == 5 and d["neurons"][0]["synapses_out"] > 0


def test_http_neuprint_client_needs_no_optional_package(monkeypatch):
    import httpx

    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        assert request.headers["authorization"] == "Bearer tok"
        if request.url.path == "/api/version":
            return httpx.Response(200, json={"Version": "1.7.0"})
        body = json.loads(request.content)
        assert body["dataset"] == "hemibrain:v1.2.1" and "$ids" not in body["cypher"]
        return httpx.Response(200, json={"columns": ["a", "b"], "data": [[1, 2], [3, 4]]})

    client = qc.HttpNeuprintClient("https://neuprint.janelia.org", "hemibrain:v1.2.1", "tok")
    client._client = httpx.Client(transport=httpx.MockTransport(handler), base_url="https://neuprint.janelia.org",
                                  headers={"Authorization": "Bearer tok"})
    assert client.fetch_version() == "1.7.0"
    rows = client.fetch_custom("MATCH (n) WHERE n.bodyId IN $ids RETURN n", {"ids": [1, 2]})
    assert rows == [{"a": 1, "b": 2}, {"a": 3, "b": 4}]
    assert len(calls) == 2


def test_http_neuprint_client_reports_a_bad_token():
    import httpx
    import pytest

    client = qc.HttpNeuprintClient("neuprint.janelia.org", "hemibrain:v1.2.1", "bad")
    client._client = httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(401)),
                                  base_url="https://neuprint.janelia.org")
    with pytest.raises(PermissionError):
        client.fetch_version()


def test_readme_is_served_in_app_and_markdown_is_required():
    from backend.api import docs_pages

    html, status = docs_pages.render("README")
    assert status == 200 and "<h1" in html and "DROSOPHILA" in html
    assert docs_pages.render("../etc/passwd")[1] == 404
    assert re.search(r"^markdown>=", (ROOT / "requirements.txt").read_text(), re.M)
    main_src = (ROOT / "backend/api/main.py").read_text()
    assert '"/readme"' in main_src and "/api/system/autostart" in main_src


def test_autostart_journals_every_line_for_the_app():
    script = (ROOT / "tools/codespace_autostart.sh").read_text()
    assert "autostart.journal" in script
    for fn in ("log()", "ok()", "warn()", "bad()"):
        assert "journal" in script[script.index(fn): script.index(fn) + 120]
    from backend.api import autostart_status

    payload = autostart_status.status(lines=5)
    assert set(payload) >= {"verdict", "healthy", "provisioned", "failures", "journal", "logs", "retry"}


def test_flutter_installer_prefers_the_archive_and_never_hits_already_exists():
    script = (ROOT / "frontend/run_web.sh").read_text()
    archive = script.index("route A (preferred): the official release archive")
    clone = script.index("route B: shallow git clone")
    assert archive < clone                                        # archive first, git fallback
    assert "git clone" in script[clone:]
    assert 'git clone --depth 1 --single-branch -b stable \\\n           https://github.com/flutter/flutter.git "$staging"' in script  # fresh dir, then mv
    assert "removing an incomplete Flutter SDK" in script and "flutter-sdk" in script
    assert "-C -" in script                                       # resumable download
    assert "flutter_home" in script
    setup = (ROOT / ".devcontainer/setup.sh").read_text()
    assert "skipping apt (fast path)" in setup and "--prefer-binary" in setup
    assert '"waitFor": "postCreateCommand"' in (ROOT / ".devcontainer/devcontainer.json").read_text()
    auto = (ROOT / "tools/codespace_autostart.sh").read_text()
    assert "start_placeholder" in auto and "stop_placeholder" in auto
    assert "still waiting" in auto                              # the lock wait talks
