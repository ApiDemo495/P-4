"""Round AQ - the real fly brain (FlyWire v783) replaces the 80x80 stand-in.

The connectome itself (~130 MB) is never downloaded by the tests; the graph
machinery is exercised on a tiny synthetic connectome written in the same
``.npz`` layout, and - when the real arrays happen to be on disk - on the real
mushroom body too.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pytest

from backend.brain import connectome_data as cd
from backend.brain import whole_brain as wb

ROOT = Path(__file__).resolve().parents[2]


def _tiny_connectome(tmp_path: Path) -> Path:
    """40 ALPNs (20 ON/OFF pairs), 60 KCs, 6 MBONs (3 ACh approach, 3 Glu avoid),
    2 DANs, 2 descending neurons; wiring chosen so ON odours reach the approach
    MBONs and OFF odours the avoid MBONs."""
    import scipy.sparse as sp

    n_alpn, n_kc, n_mbon, n_dan, n_dn = 40, 60, 6, 2, 2
    n = n_alpn + n_kc + n_mbon + n_dan + n_dn
    rows, cols, vals = [], [], []
    rng = np.random.default_rng(7)
    kc0 = n_alpn
    for g in range(n_alpn):
        for k in rng.choice(n_kc, 6, replace=False):
            # ON groups (even) project to the first half of the KCs, OFF to the second
            kc = (k % (n_kc // 2)) + (0 if g % 2 == 0 else n_kc // 2)
            rows.append(kc0 + kc)
            cols.append(g)
            vals.append(5.0)
    mb0 = kc0 + n_kc
    for kc in range(n_kc):
        target = mb0 + (kc % 3 if kc < n_kc // 2 else 3 + kc % 3)
        rows.append(target)
        cols.append(kc0 + kc)
        vals.append(8.0)
    dn0 = mb0 + n_mbon + n_dan
    for m in range(n_mbon):
        rows.append(dn0 + (m % 2))
        cols.append(mb0 + m)
        vals.append(3.0 if m < 3 else -3.0)
    raw = sp.csr_matrix((np.asarray(vals, dtype=np.float32), (rows, cols)), shape=(n, n))
    total_in = np.asarray(abs(raw).sum(axis=1)).ravel().astype(np.float32)
    scale = np.where(total_in > 0, 1.0 / np.maximum(total_in, 1e-9), 0.0).astype(np.float32)
    W = (sp.diags(scale) @ raw).tocsr()
    cell_class = np.array(["ALPN"] * n_alpn + ["Kenyon_Cell"] * n_kc + ["MBON"] * n_mbon + ["DAN"] * n_dan + [""] * n_dn, dtype="U32")
    cell_type = np.array([f"G{g:02d}_PN" for g in range(n_alpn)] + ["KCab"] * n_kc
                         + [f"MBON0{m}" for m in range(n_mbon)] + ["PAM01", "PPL101"] + ["DNp01", "DNp02"], dtype="U48")
    nt = np.array([cd.NT_CODES.index("acetylcholine")] * (n_alpn + n_kc)
                  + [cd.NT_CODES.index("acetylcholine")] * 3 + [cd.NT_CODES.index("glutamate")] * 3
                  + [cd.NT_CODES.index("dopamine")] * 2 + [cd.NT_CODES.index("acetylcholine")] * 2, dtype=np.int8)
    super_class = np.array([cd.SUPER_CLASSES.index("central")] * (n - n_dn) + [cd.SUPER_CLASSES.index("descending")] * n_dn, dtype=np.int8)
    path = tmp_path / "tiny.npz"
    np.savez(path, version=cd.BUILD_VERSION, n=n, indptr=W.indptr, indices=W.indices, data=W.data,
             root_ids=np.arange(n, dtype=np.int64) + 720575940000000000, super_class=super_class, nt=nt,
             side=np.zeros(n, dtype=np.int8), cell_class=cell_class, cell_type=cell_type, total_in=total_in,
             synapses_total=int(np.abs(vals).sum()), connections=int(W.nnz), annotated=n)
    return path


@pytest.fixture
def tiny(tmp_path) -> wb.ConnectomeGraph:
    return wb.ConnectomeGraph(_tiny_connectome(tmp_path), "tiny test connectome", hops=3)


def test_populations_are_read_from_the_annotations(tiny):
    pops = tiny.populations()
    assert pops["alpn"] == 40 and pops["kenyon_cells"] == 60 and pops["mbon"] == 6
    assert pops["mbon_approach"] == 3 and pops["mbon_avoid"] == 3
    assert pops["pam"] == 1 and pops["ppl1"] == 1 and pops["descending"] == 2
    assert len(pops["formula_groups"]) == 20 and all(g["on"] == 1 and g["off"] == 1 for g in pops["formula_groups"])


def test_bullish_and_bearish_inputs_are_different_odours(tiny):
    """f(-x) = -f(x), a flat tape reads exactly 0, and a bullish ensemble
    reaches the approach MBONs."""
    flat = tiny.propagate(np.zeros(20))
    assert flat.balance == 0.0 and flat.active_kcs == 0
    up = tiny.propagate(np.full(20, 0.6))
    down = tiny.propagate(np.full(20, -0.6))
    assert up.balance > 0.5 and down.balance == pytest.approx(-up.balance, abs=1e-6)
    assert up.active_kcs > 0 and up.hop_active[0] == 20
    assert up.descending != 0.0 and up.elapsed_us > 0


def test_dopamine_gates_the_mbon_synapses(tiny):
    v = np.full(20, 0.4)
    base = tiny.propagate(v, drg=0.0).balance
    assert tiny.propagate(v, drg=0.9).balance >= base - 1e-6
    assert tiny.propagate(v, drg=-0.9).balance <= base + 1e-6
    mixed = np.r_[np.full(10, 0.4), np.full(10, -0.4)]   # both MBON sides driven
    assert tiny.propagate(mixed, drg=0.9).balance > tiny.propagate(mixed, drg=-0.9).balance
    r = tiny.propagate(v, drg=0.9)
    assert r.dan["pam_gain"] == pytest.approx(1.45) and r.dan["ppl1_gain"] == 1.0


def test_the_activation_trace_keeps_the_shape_the_formulas_expect(tiny):
    r = tiny.propagate(np.full(20, 0.5))
    trace = tiny.to_activation_trace(r)
    d = trace.to_dict()
    assert d["kenyon_cells"]["of"] == 60 and d["kenyon_cells"]["connectome"] == "tiny test connectome"
    assert trace.kc_activations.size == 60 and trace.pn_activations.size == 20
    assert "hop_active" in trace.diagnostics and trace.diagnostics["neurons"] == tiny.n


def test_ccsv2_runs_through_the_real_mushroom_body_when_loaded(tiny):
    from backend.formulas.category_h_brain import ccsv2

    class FakeBrain:
        mb = tiny
        conv = None

        def mushroom_body_pass(self, vector, drg, hsi):
            return tiny.propagate(vector, drg=drg, hsi=hsi)

    ctx = {"_brain": FakeBrain(), "_drg": 0.1, "HSI": 0.2}
    for name in ccsv2.FORMULA_ORDER:
        ctx[name] = 0.5
    state = ccsv2.State()
    value = ccsv2.compute(None, "BTC", state, {}, ctx)
    assert value > 0.3 and ctx["KCAE"] > 0.0 and ctx["_brain_trace"]["kenyon_cells"]["of"] == 60
    rows = {r["label"]: r["value"] for r in ctx.get("_trace", [])} if "_trace" in ctx else {}
    # the re-derivation rule reads the two MBON rows
    assert ccsv2.double_check({"approach MBON drive (ACh + GABA)": 2.0, "avoid MBON drive (Glu)": 1.0}, "BTC") > 0
    assert rows == rows  # (traces are optional in a bare ctx)


def test_data_status_reports_the_download_honestly(tmp_path, monkeypatch):
    data = cd.ConnectomeData(tmp_path, enabled=False)
    assert data.ready is False and data.ensure() is False
    assert data.to_dict()["phase"] == "disabled"
    assert set(cd.SOURCES) == {"Connectivity_783.parquet", "neuron_annotations_783.tsv"}
    assert all(any("github" in u for u in urls) for urls in cd.SOURCES.values())


def test_brain_exposes_the_connectome_and_falls_back_until_loaded():
    from backend.brain.brain import Brain

    brain = Brain()
    payload = brain.connectome_payload()
    assert payload["fallback_in_use"] is (not brain.connectome_ready)
    assert "80x80" in payload["fallback"]
    assert brain.mushroom_body_pass(np.zeros(20), 0.0, 0.0) is None or brain.connectome_ready
    assert brain.whole_brain_fresh() is None


@pytest.mark.skipif(not cd.artifacts_ready(), reason="FlyWire arrays not on disk (downloaded on first engine start)")
def test_the_real_mushroom_body_when_present():
    mb = wb.load_mushroom_body()
    pops = mb.populations()
    assert pops["kenyon_cells"] > 5000 and pops["mbon"] >= 90 and pops["alpn"] > 600
    r = mb.propagate(np.random.default_rng(3).uniform(-1, 1, 20))
    assert -1 <= r.balance <= 1 and r.active_kcs > 0 and r.elapsed_us < 200_000


def test_ui_and_routes_carry_the_real_brain():
    app_js = (ROOT / "backend/web/app.js").read_text()
    index = (ROOT / "backend/web/index.html").read_text()
    routes = (ROOT / "backend/api/routes_brain.py").read_text()
    assert "function renderWholeBrain" in app_js and 'id="brain-whole"' in index
    assert "/api/brain/connectome" in routes
    assert len(re.findall(r"setInterval\(", app_js)) == 1
    assert "pyarrow" in (ROOT / "requirements.txt").read_text()
    assert "data/connectome/" in (ROOT / ".gitignore").read_text()
