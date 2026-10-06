"""Domain 2 - persistent homology of price clouds (210 formulas).

Point clouds are the last n candles embedded in 3-D (one of three coordinate
systems), min-max normalised per window.  The Vietoris-Rips complex at scale
ε connects points closer than ε; β₀ is counted by graph search, β₁ by the
Euler formula on the 1-skeleton (E − V + C, the cycle rank), H₀ persistence
from the single-linkage merge tree (MST edge lengths = death times).
"""
from __future__ import annotations

import numpy as np

from backend.genesis import ops
from backend.genesis.domains._common import components, finish, momentum_sign, mst_lengths, pdist_sq, tail_windows
from backend.genesis.spec import grid, make_variants

D = 2
N_WINDOWS = (10, 15, 20, 30, 40, 50, 60)
EPSILONS = (0.15, 0.25, 0.35)
CLOUDS = ("close_vol_trades", "close_range_vol", "ret_vol_skew")


def _cloud(frame, kind):
    if kind == "close_range_vol":
        return [frame.close, frame.rng, frame.volume]
    if kind == "ret_vol_skew":
        return [frame.ret, frame.roll("ret_std", 5), frame.roll("ret_skew", 8)]
    if kind == "book":
        return [frame.imbalance, frame.spread_bps, frame.bid_depth + frame.ask_depth]
    return [frame.close, frame.volume, frame.trades]


def _betti(W, eps, Dm=None):
    Dm = pdist_sq(W) if Dm is None else Dm
    adj = (Dm < eps * eps)
    np.fill_diagonal(adj, False)
    V = len(W)
    E = int(adj.sum() // 2)
    C = components(adj)
    return V, E, C, E - V + C, Dm, adj


def _clouds(frame, n, cloud):
    """Shared per-(cloud, n) cache of (t, W, D²) so the ten sub-categories do
    not recompute the same distance matrices; layer-4 kernels stride by 5."""
    key = ("d2cloud", cloud, n)
    hit = frame.cache.get(key)
    if hit is None:
        hit = [(t, W, pdist_sq(W)) for t, W in tail_windows(frame, n, _cloud(frame, cloud))]
        frame.cache[key] = hit
    return hit


def _msts(frame, n, cloud):
    key = ("d2mst", cloud, n)
    hit = frame.cache.get(key)
    if hit is None:
        hit = [(t, mst_lengths(Dm)) for t, _W, Dm in _clouds(frame, n, cloud)]
        frame.cache[key] = hit
    return hit


def _run(frame, n, cloud, fn, stride: int = 1):
    raw = ops.nan(frame.n)
    items = _clouds(frame, n, cloud)
    last = np.nan
    for i, (t, W, Dm) in enumerate(items):
        if i % stride == 0 or i == len(items) - 1:
            last = fn(W, Dm)
        raw[t] = last
    return raw


def _run_mst(frame, n, cloud, fn):
    raw = ops.nan(frame.n)
    for t, L in _msts(frame, n, cloud):
        raw[t] = fn(L)
    return raw


def k_b0(frame, n, norm, eps=0.25, cloud="close_vol_trades", **_):
    raw = _run(frame, n, cloud, lambda W, Dm: _betti(W, eps, Dm)[2] / len(W))
    # many components = the market keeps jumping to new states -> follow the last move
    sig = (raw - 0.3) * 3.0 * momentum_sign(frame)
    return finish(frame, sig, raw, norm, n, 1.0)


def k_b1(frame, n, norm, eps=0.25, cloud="close_vol_trades", **_):
    def f(W, Dm):
        V, E, C, b1, *_ = _betti(W, eps, Dm)
        return b1 / max(1, (V - 1) * (V - 2) / 2)  # max cycle rank of a complete graph
    raw = _run(frame, n, cloud, f)
    # B1M > 0.3 cycling -> fade the last move; < 0.1 trending -> follow it
    sig = np.where(raw > 0.3, -1.0, np.where(raw < 0.1, 1.0, (0.2 - raw) / 0.1)) * momentum_sign(frame)
    return finish(frame, sig, raw, norm, n, 1.0)


def k_b2(frame, n, norm, eps=0.35, cloud="close_vol_trades", **_):
    """β₂ via the Euler characteristic of the clique complex truncated at
    tetrahedra: χ = V − E + F − T, β₂ = χ − β₀ + β₁ (exact when β₃ = 0)."""
    def f(W, Dm):
        V, E, C, b1, Dm, adj = _betti(W, eps, Dm)
        A = adj.astype(np.int64)
        F = int(np.trace(A @ A @ A) // 6)
        T = 0
        idx = np.arange(V)
        for i in range(V):
            nb = idx[(A[i] > 0) & (idx > i)]
            if len(nb) >= 3:
                sub = A[np.ix_(nb, nb)]
                T += int(np.trace(sub @ sub @ sub) // 6)
        chi = V - E + F - T
        return max(0, chi - C + b1) / max(1, V // 4)
    raw = _run(frame, n, cloud, f, stride=5)
    sig = raw * -momentum_sign(frame)  # enclosed voids = states surrounded on all sides: fade
    return finish(frame, sig, raw, norm, n, 1.0)


def k_barcode(frame, n, norm, cloud="close_vol_trades", **_):
    raw = _run_mst(frame, n, cloud, lambda L: float(L.sum()) / np.sqrt(3.0))
    sig = (raw / n - 0.2) * 5.0 * momentum_sign(frame)  # spread-out cloud = trending
    return finish(frame, sig, raw, norm, n, 1.0)


def k_landscape(frame, n, norm, cloud="close_vol_trades", **_):
    raw = _run_mst(frame, n, cloud, lambda L: float(L.max()) / 2.0)
    sig = (raw - 0.15) * 6.0 * momentum_sign(frame)  # a big H0 gap = a jump: breakout follow
    return finish(frame, sig, raw, norm, n, 1.0)


def k_pentropy(frame, n, norm, cloud="close_vol_trades", **_):
    def f(L):
        p = L / (L.sum() + 1e-12)
        return float(ops.entropy_bits(p)) / np.log2(max(2, len(L)))
    raw = _run_mst(frame, n, cloud, f)
    sig = (0.8 - raw) * 4.0 * momentum_sign(frame)  # low persistence entropy = structure: trade it
    return finish(frame, sig, raw, norm, n, 1.0)


def k_wasserstein_pd(frame, n, norm, cloud="close_vol_trades", **_):
    bars = ops.nan(frame.n)
    store = {t: np.sort(L) for t, L in _msts(frame, n, cloud)}
    for t, L in store.items():
        prev = store.get(t - max(1, n // 2))
        if prev is not None:
            bars[t] = ops.wasserstein1_samples(L, prev)
    sig = bars * 10.0 * momentum_sign(frame)  # the diagram moved: shape change in the move's direction
    return finish(frame, sig, bars, norm, n, 1.0)


def k_persistent_b1(frame, n, norm, cloud="close_vol_trades", **_):
    scales = np.linspace(0.1, 0.6, 11)

    # single-linkage fact: β₀(ε) = 1 + #{MST edges longer than ε}, so every
    # scale costs O(n²) for the edge count and O(n) for the components.
    msts = dict(_msts(frame, n, cloud))

    def f(W, Dm):
        alive = 0
        V = len(W)
        L = msts.get(f.t)
        iu = np.triu_indices(V, 1)
        d2 = Dm[iu]
        for e in scales:
            E = int(np.count_nonzero(d2 < e * e))
            if E < V:
                continue
            C = 1 + int(np.count_nonzero(L > e)) if L is not None else components(Dm < e * e)
            alive += int(E - V + C > 0)
        return alive / len(scales)
    raw = ops.nan(frame.n)
    for t, W, Dm in _clouds(frame, n, cloud):
        f.t = t
        raw[t] = f(W, Dm)
    sig = -(raw - 0.5) * 2.0 * momentum_sign(frame)  # loops that persist across scales: cycling
    return finish(frame, sig, raw, norm, n, 1.0)


def k_cech(frame, n, norm, eps=0.25, cloud="close_vol_trades", **_):
    """Čech vs Rips: a Rips triangle is Čech-filled iff its minimal enclosing
    ball has radius ≤ ε/2.  The unfilled fraction counts 'thin' loops."""
    def f(W, Dm):
        V, E, C, b1, Dm, adj = _betti(W, eps, Dm)
        tri = 0
        thin = 0
        for i in range(V):
            for j in range(i + 1, V):
                if not adj[i, j]:
                    continue
                ks = np.nonzero(adj[i] & adj[j])[0]
                ks = ks[ks > j]
                for k in ks:
                    tri += 1
                    a, b, c = np.sqrt(Dm[i, j]), np.sqrt(Dm[j, k]), np.sqrt(Dm[i, k])
                    s = max(a, b, c)
                    if s * s >= a * a + b * b + c * c - s * s:  # obtuse / right: R = longest/2
                        R = s / 2.0
                    else:
                        area = max(1e-12, 0.25 * np.sqrt(max(0.0, (a + b + c) * (-a + b + c) * (a - b + c) * (a + b - c))))
                        R = a * b * c / (4.0 * area)
                    thin += int(R > eps / 2.0)
        return thin / tri if tri else 0.0
    raw = _run(frame, n, cloud, f, stride=5)
    sig = -(raw - 0.3) * 3.0 * momentum_sign(frame)
    return finish(frame, sig, raw, norm, n, 1.0)


def k_alpha(frame, n, norm, eps=0.25, **_):
    """Alpha complex on the 2-D (return, volume-return) cloud: Delaunay edges
    shorter than 2α, cycle rank of that 1-skeleton."""
    try:
        from scipy.spatial import Delaunay
    except Exception:  # noqa: BLE001
        return finish(frame, ops.nan(frame.n), ops.nan(frame.n), norm, n, 1.0)

    def f(W):
        W2 = W[:, :2]
        try:
            tri = Delaunay(W2)
        except Exception:  # noqa: BLE001
            return 0.0
        edges = set()
        for s in tri.simplices:
            for a, b in ((0, 1), (1, 2), (0, 2)):
                i, j = sorted((int(s[a]), int(s[b])))
                if np.linalg.norm(W2[i] - W2[j]) <= 2 * eps:
                    edges.add((i, j))
        V = len(W2)
        adj = np.zeros((V, V), dtype=bool)
        for i, j in edges:
            adj[i, j] = adj[j, i] = True
        return (len(edges) - V + components(adj)) / max(1, V)
    raw = ops.nan(frame.n)
    last = np.nan
    for i, (t, W) in enumerate(tail_windows(frame, n, [frame.ret, frame.vol_ret, frame.rng])):
        if i % 5 == 0:
            last = f(W)
        raw[t] = last
    sig = -(raw - 0.2) * 4.0 * momentum_sign(frame)
    return finish(frame, sig, raw, norm, n, 1.0)


def variants():
    g = grid(windows=N_WINDOWS, extra=dict(eps=EPSILONS, cloud=CLOUDS))
    g_small = grid(windows=(10, 12, 15, 18, 20, 25, 30), extra=dict(eps=EPSILONS, cloud=CLOUDS))
    return (
        make_variants(D, 1, "Betti-0", "Betti-0 fragmentation", k_b0,
                      "β₀(ε) = connected components of the Vietoris-Rips graph on the normalised 3-D cloud, / n",
                      "many components: the market keeps visiting new states - follow the last move", 3, g)
        + make_variants(D, 2, "Betti-1", "Betti-1 momentum (B1M)", k_b1,
                        "β₁ = E − V + C of the ε-graph (Euler formula), / C(n−1, 2)",
                        "B1M > 0.3 cycling (fade extremes); < 0.1 trending (ride it); falling B1M = breakout forming", 3, g)
        + make_variants(D, 3, "Betti-2", "Betti-2 voids", k_b2,
                        "χ = V − E + F − T over the clique complex up to tetrahedra; β₂ = χ − β₀ + β₁",
                        "enclosed voids: the current state is surrounded by visited states on all sides - fade", 4, g_small)
        + make_variants(D, 4, "persistence barcode", "H₀ total persistence", k_barcode,
                        "Σ death times of H₀ bars = total MST length of the cloud (single-linkage filtration)",
                        "a long barcode = spread-out, exploring cloud = trend; short = compact = chop", 3, g)
        + make_variants(D, 5, "persistence landscape", "H₀ landscape peak λ₁", k_landscape,
                        "λ₁ peak = (longest H₀ bar)/2 = half the longest MST edge",
                        "a tall peak means two clusters separated by a jump - breakout, follow it", 3, g)
        + make_variants(D, 6, "persistence entropy", "Persistence entropy", k_pentropy,
                        "Shannon entropy of normalised H₀ lifetimes / log₂(bars)",
                        "low entropy = a few dominant structures (exploitable); high = uniform noise", 3, g)
        + make_variants(D, 7, "Wasserstein persistence", "W₁ between persistence diagrams", k_wasserstein_pd,
                        "1-Wasserstein distance between the sorted H₀ lifetimes now and n/2 candles ago",
                        "the diagram moved: the shape of the market changed, in the direction of the last move", 3, g)
        + make_variants(D, 8, "Vietoris-Rips persistence", "Persistent β₁ across scales", k_persistent_b1,
                        "fraction of 11 scales ε ∈ [0.1, 0.6] at which the Rips graph has a cycle",
                        "loops that survive many scales = genuine cycling - fade the last move", 3, g)
        + make_variants(D, 9, "Čech complex", "Čech-unfilled triangle ratio", k_cech,
                        "share of Rips triangles whose minimal enclosing ball radius exceeds ε/2 (not Čech-filled)",
                        "thin, unfilled loops = cycling corridors; filled = solid moves", 4, g_small)
        + make_variants(D, 10, "alpha complex", "Alpha-complex cycle rank", k_alpha,
                        "Delaunay edges of the (return, Δvolume) cloud shorter than 2α; cycle rank E − V + C, / n",
                        "many alpha cycles = returns revisiting the same states = mean reversion", 4, g_small)
    )
