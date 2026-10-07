"""The real fly brain on disk: download, build and cache the FlyWire v783
whole-brain connectome (Round AQ).

Sources - public, no login, all on GitHub (reachable from a Codespace and
from the sandbox alike):

* ``philshiu/Drosophila_brain_model`` - ``Connectivity_783.parquet``: the
  15.09 M neuron-to-neuron connections of the proofread FlyWire v783 release
  (Dorkenwald et al. 2024, Nature) with the sign of each connection from the
  neurotransmitter prediction (Eckstein et al. 2024), exactly the table
  Shiu et al. 2024 ("A Drosophila computational brain model") simulate.
* ``flyconnectome/flywire_annotations`` - ``Supplemental_file1_neuron_annotations.tsv``
  (Schlegel et al. 2024): the cell class / cell type / neurotransmitter /
  side of every one of the 139 248 neurons.

What is built (once, cached as ``.npz`` under ``data/connectome/``):

* ``whole``  - CSR matrix ``W[post, pre]`` = sign x synapse count, divided by
  the postsynaptic neuron's total input (the *input fraction*), so a unit of
  presynaptic activity contributes its share of the target's input and
  activity stays bounded through any number of hops.  138 639 neurons,
  15.09 M connections.
* ``mb``     - the mushroom-body / lateral-horn sub-circuit of the same matrix:
  antennal-lobe projection neurons (ALPN, the input), Kenyon cells (5 177),
  dopaminergic neurons (DAN), the mushroom-body output neurons (MBON, 96),
  APL / MBIN and the lateral-horn neurons (LHLN / LHCENT) plus their local
  partners - 8 331 neurons, 836 k connections.  Small enough for a pass in
  milliseconds, so it drives the per-pass formulas CCSv2 / KCAE.

Nothing here is invented: every edge is a counted synapse, every role is the
published annotation.  The only modelling choices are in ``whole_brain.py``.
"""

from __future__ import annotations

import csv
import logging
import os
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from backend.core import config as cfg

log = logging.getLogger("drosophila.brain.data")

DATASET = "FlyWire FAFB v783"
DATA_DIR = Path(os.environ.get("CONNECTOME_DATA_DIR", str(cfg.REPO_ROOT / "data" / "connectome")))

#: (file name, candidate URLs in order).  The api.github.com form works from
#: hosts where raw.githubusercontent.com is blocked; the raw form is the
#: usual fast path.
_RAW = "application/vnd.github.raw"
SOURCES = {
    "Connectivity_783.parquet": (
        "https://raw.githubusercontent.com/philshiu/Drosophila_brain_model/main/Connectivity_783.parquet",
        "https://api.github.com/repos/philshiu/Drosophila_brain_model/contents/Connectivity_783.parquet",
    ),
    "neuron_annotations_783.tsv": (
        "https://raw.githubusercontent.com/flyconnectome/flywire_annotations/main/supplemental_files/Supplemental_file1_neuron_annotations.tsv",
        "https://api.github.com/repos/flyconnectome/flywire_annotations/contents/supplemental_files/Supplemental_file1_neuron_annotations.tsv",
    ),
}
EXPECTED_MIN_BYTES = {"Connectivity_783.parquet": 90_000_000, "neuron_annotations_783.tsv": 25_000_000}
WHOLE_NPZ = "flywire_783_whole.npz"
MB_NPZ = "flywire_783_mushroom_body.npz"
BUILD_VERSION = 3

#: cell classes that make up the mushroom-body / lateral-horn sub-circuit
MB_CLASSES = ("ALPN", "Kenyon_Cell", "DAN", "MBON", "MBIN", "LHLN", "LHCENT", "ALLN", "ALIN")
MB_TYPE_PREFIXES = ("MBON", "APL", "MBIN", "DPM", "LHPV", "LHAV", "LHAD", "LHPD")

#: coded columns kept per neuron (small ints; the string tables ride along)
SUPER_CLASSES = ("optic", "central", "sensory", "visual_projection", "ascending", "descending",
                 "sensory_ascending", "visual_centrifugal", "motor", "endocrine", "")
NT_CODES = ("acetylcholine", "glutamate", "gaba", "dopamine", "octopamine", "serotonin", "")
SIDES = ("left", "right", "center", "na", "")


@dataclass
class DataStatus:
    phase: str = "idle"          # idle | downloading | building | ready | failed | disabled
    detail: str = ""
    file: str = ""
    bytes_done: int = 0
    bytes_total: int = 0
    started_at: float = 0.0
    finished_at: float = 0.0
    error: str = ""
    files: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        pct = (100.0 * self.bytes_done / self.bytes_total) if self.bytes_total else 0.0
        return {
            "dataset": DATASET,
            "phase": self.phase,
            "detail": self.detail,
            "file": self.file,
            "bytes_done": self.bytes_done,
            "bytes_total": self.bytes_total,
            "percent": round(pct, 1),
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "error": self.error,
            "files": dict(self.files),
            "data_dir": str(DATA_DIR),
        }


def artifacts_ready(data_dir: Path = DATA_DIR) -> bool:
    return (data_dir / WHOLE_NPZ).exists() and (data_dir / MB_NPZ).exists()


# ---------------------------------------------------------------------------
# download
# ---------------------------------------------------------------------------
def _stream_httpx(url: str, headers: dict, tmp: Path, status: DataStatus, timeout: float) -> None:
    import httpx

    with httpx.stream("GET", url, headers=headers, follow_redirects=True, timeout=timeout) as r:
        if r.status_code != 200:
            raise RuntimeError(f"HTTP {r.status_code}")
        status.bytes_total = int(r.headers.get("content-length") or 0)
        with tmp.open("wb") as fh:
            for chunk in r.iter_bytes(1 << 20):
                fh.write(chunk)
                status.bytes_done += len(chunk)


def _stream_urllib(url: str, headers: dict, tmp: Path, status: DataStatus, timeout: float) -> None:
    """Same download through the standard library and the *system* trust
    store (corporate / sandbox proxies often sign with a CA that certifi -
    which httpx uses - does not carry)."""
    import ssl
    import urllib.request

    ctx = ssl.create_default_context()
    try:
        ctx.load_default_certs()
    except Exception:  # noqa: BLE001
        pass
    req = urllib.request.Request(url, headers={**headers, "User-Agent": "drosophila-trader"})
    with urllib.request.urlopen(req, timeout=timeout, context=ctx) as r:
        if getattr(r, "status", 200) != 200:
            raise RuntimeError(f"HTTP {getattr(r, 'status', '?')}")
        status.bytes_total = int(r.headers.get("content-length") or 0)
        with tmp.open("wb") as fh:
            while True:
                chunk = r.read(1 << 20)
                if not chunk:
                    break
                fh.write(chunk)
                status.bytes_done += len(chunk)


def _download(name: str, dest: Path, status: DataStatus, timeout: float = 600.0) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    errors = []
    token = os.environ.get("GITHUB_TOKEN") or getattr(cfg.SETTINGS, "github_models_token", "")
    for url in SOURCES[name]:
        host = url.split("/")[2]
        headers = {"Accept": _RAW} if "api.github.com" in url else {}
        if token and "api.github.com" in url:
            headers["Authorization"] = f"Bearer {token}"
        for label, streamer in (("httpx", _stream_httpx), ("urllib", _stream_urllib)):
            try:
                status.file, status.bytes_done, status.bytes_total = name, 0, 0
                streamer(url, headers, tmp, status, timeout)
                if tmp.stat().st_size < EXPECTED_MIN_BYTES.get(name, 1):
                    raise RuntimeError(f"only {tmp.stat().st_size} bytes")
                tmp.replace(dest)
                status.files[name] = {"bytes": dest.stat().st_size, "from": f"{host} ({label})"}
                return
            except Exception as exc:  # noqa: BLE001 - try the next transport / mirror
                errors.append(f"{host}/{label}: {type(exc).__name__}: {exc}"[:160])
                tmp.unlink(missing_ok=True)
    raise RuntimeError(f"{name}: " + " | ".join(errors))


# ---------------------------------------------------------------------------
# build
# ---------------------------------------------------------------------------
def _read_annotations(path: Path) -> dict[int, dict]:
    out: dict[int, dict] = {}
    with path.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh, delimiter="\t"):
            try:
                out[int(row["root_id"])] = row
            except (KeyError, ValueError):
                continue
    return out


def _code(table: tuple, value: str) -> int:
    value = (value or "").strip()
    return table.index(value) if value in table else len(table) - 1


def build_arrays(data_dir: Path = DATA_DIR, status: DataStatus | None = None) -> dict:
    """Parquet + TSV -> the two ``.npz`` artifacts.  Returns build facts."""
    import pyarrow.parquet as pq
    import scipy.sparse as sp

    status = status or DataStatus()
    t0 = time.perf_counter()
    status.phase, status.detail = "building", "reading 15 M connections"
    table = pq.read_table(
        data_dir / "Connectivity_783.parquet",
        columns=["Presynaptic_ID", "Presynaptic_Index", "Postsynaptic_ID", "Postsynaptic_Index",
                 "Excitatory x Connectivity"],
    )
    pre = table["Presynaptic_Index"].to_numpy().astype(np.int64)
    post = table["Postsynaptic_Index"].to_numpy().astype(np.int64)
    signed = table["Excitatory x Connectivity"].to_numpy().astype(np.float32)
    pre_id = table["Presynaptic_ID"].to_numpy()
    post_id = table["Postsynaptic_ID"].to_numpy()
    del table
    n = int(max(pre.max(), post.max())) + 1
    root_ids = np.zeros(n, dtype=np.int64)
    root_ids[pre] = pre_id
    root_ids[post] = post_id
    del pre_id, post_id

    status.detail = "input-fraction normalisation"
    # W[post, pre]: activity flows pre -> post with h_next = W @ h
    raw = sp.csr_matrix((signed, (post, pre)), shape=(n, n), dtype=np.float32)
    raw.sum_duplicates()
    total_in = np.asarray(abs(raw).sum(axis=1)).ravel().astype(np.float32)
    scale = np.where(total_in > 0, 1.0 / np.maximum(total_in, 1e-9), 0.0).astype(np.float32)
    whole = sp.diags(scale) @ raw
    whole = whole.tocsr()
    whole.sort_indices()
    synapses_total = int(np.abs(signed).sum())
    del raw, signed, pre, post

    status.detail = "annotating 139 k neurons"
    ann = _read_annotations(data_dir / "neuron_annotations_783.tsv")
    super_class = np.full(n, len(SUPER_CLASSES) - 1, dtype=np.int8)
    nt = np.full(n, len(NT_CODES) - 1, dtype=np.int8)
    side = np.full(n, len(SIDES) - 1, dtype=np.int8)
    cell_class = np.empty(n, dtype=object)
    cell_type = np.empty(n, dtype=object)
    cell_class[:] = ""
    cell_type[:] = ""
    annotated = 0
    for i in range(n):
        row = ann.get(int(root_ids[i]))
        if row is None:
            continue
        annotated += 1
        super_class[i] = _code(SUPER_CLASSES, row.get("super_class", ""))
        nt[i] = _code(NT_CODES, row.get("top_nt", ""))
        side[i] = _code(SIDES, row.get("side", ""))
        cell_class[i] = (row.get("cell_class") or "").strip()
        cell_type[i] = (row.get("cell_type") or row.get("hemibrain_type") or "").strip()
    cell_class_s = cell_class.astype("U32")
    cell_type_s = cell_type.astype("U48")

    np.savez(
        data_dir / WHOLE_NPZ, version=BUILD_VERSION, n=n, indptr=whole.indptr, indices=whole.indices,
        data=whole.data, root_ids=root_ids, super_class=super_class, nt=nt, side=side,
        cell_class=cell_class_s, cell_type=cell_type_s, total_in=total_in,
        synapses_total=synapses_total, connections=int(whole.nnz), annotated=annotated,
    )

    status.detail = "cutting the mushroom-body sub-circuit"
    in_mb = np.isin(cell_class_s, MB_CLASSES)
    for prefix in MB_TYPE_PREFIXES:
        in_mb |= np.char.startswith(cell_type_s, prefix)
    mb_nodes = np.flatnonzero(in_mb)
    sub = whole[mb_nodes][:, mb_nodes].tocsr()
    sub.sort_indices()
    np.savez(
        data_dir / MB_NPZ, version=BUILD_VERSION, n=int(mb_nodes.size), indptr=sub.indptr, indices=sub.indices,
        data=sub.data, nodes=mb_nodes, root_ids=root_ids[mb_nodes], super_class=super_class[mb_nodes],
        nt=nt[mb_nodes], side=side[mb_nodes], cell_class=cell_class_s[mb_nodes], cell_type=cell_type_s[mb_nodes],
        total_in=total_in[mb_nodes], synapses_total=synapses_total, connections=int(sub.nnz), annotated=annotated,
    )
    facts = {
        "neurons": n, "connections": int(whole.nnz), "synapses": synapses_total, "annotated": annotated,
        "mb_neurons": int(mb_nodes.size), "mb_connections": int(sub.nnz),
        "build_seconds": round(time.perf_counter() - t0, 1),
    }
    log.info("connectome built: %s", facts)
    return facts


# ---------------------------------------------------------------------------
# orchestration (runs on a worker thread; never blocks the engine loop)
# ---------------------------------------------------------------------------
class ConnectomeData:
    """Owns the download/build state; ``ensure()`` is idempotent and resumable."""

    def __init__(self, data_dir: Path = DATA_DIR, enabled: bool | None = None) -> None:
        self.data_dir = Path(data_dir)
        env = os.environ.get("CONNECTOME_AUTODOWNLOAD", "1").strip().lower()
        self.enabled = (env not in ("0", "false", "no")) if enabled is None else bool(enabled)
        self.status = DataStatus(phase="ready" if artifacts_ready(self.data_dir) else ("idle" if self.enabled else "disabled"))
        self.facts: dict = {}
        self._lock = threading.Lock()

    @property
    def ready(self) -> bool:
        return artifacts_ready(self.data_dir)

    def ensure(self) -> bool:
        """Download what is missing, build what is missing.  Blocking; call
        from a thread.  Returns True when both artifacts exist."""
        with self._lock:
            if self.ready:
                self.status.phase = "ready"
                return True
            if not self.enabled:
                self.status.phase, self.status.detail = "disabled", "CONNECTOME_AUTODOWNLOAD=0"
                return False
            self.status.started_at = time.time()
            self.status.error = ""
            try:
                for name in SOURCES:
                    dest = self.data_dir / name
                    if dest.exists() and dest.stat().st_size >= EXPECTED_MIN_BYTES[name]:
                        self.status.files[name] = {"bytes": dest.stat().st_size, "from": "cache"}
                        continue
                    self.status.phase = "downloading"
                    self.status.detail = f"{name} from GitHub"
                    _download(name, dest, self.status)
                self.status.phase = "building"
                self.facts = build_arrays(self.data_dir, self.status)
                self.status.phase, self.status.detail = "ready", "built"
                self.status.finished_at = time.time()
                return True
            except Exception as exc:  # noqa: BLE001 - reported, retried later
                self.status.phase = "failed"
                self.status.error = f"{type(exc).__name__}: {exc}"[:400]
                self.status.detail = "will retry"
                log.warning("connectome data: %s", self.status.error)
                return False

    def to_dict(self) -> dict:
        out = self.status.to_dict()
        out["ready"] = self.ready
        out["facts"] = dict(self.facts)
        out["enabled"] = self.enabled
        if self.ready:
            try:
                out["artifacts"] = {
                    WHOLE_NPZ: {"bytes": (self.data_dir / WHOLE_NPZ).stat().st_size},
                    MB_NPZ: {"bytes": (self.data_dir / MB_NPZ).stat().st_size},
                }
            except OSError:
                pass
        return out
