"""Developer benchmark: evaluate the whole template pool on a synthetic tape."""
import numpy as np, time, warnings
warnings.simplefilter("ignore")
from backend.genesis.candles import CandleStore
from backend.genesis.features import build_frame
from backend.genesis.domains import all_variants


def make(seed, trend=0.0, minutes=620):
    st = CandleStore("BTC"); rng = np.random.default_rng(seed); t0 = 1_700_000_000_000; price = 60000.0
    for m in range(minutes):
        for k in range(20):
            price *= np.exp(rng.normal(trend, 2e-4))
            book = np.zeros((2, 20, 2)); book[0, :, 0] = price - np.arange(1, 21) * 0.5; book[1, :, 0] = price + np.arange(1, 21) * 0.5
            book[:, :, 1] = rng.uniform(0.1, 2.0, (2, 20))
            st.on_tape([(t0 + m * 60000 + k * 3000, price, rng.uniform(0.001, 0.5), rng.choice([-1.0, 1.0]))], book if k % 3 == 0 else None)
    return st


if __name__ == "__main__":
    a, b = make(1), make(2)
    fr = build_frame("BTC", a.rows(), b.rows())
    specs = all_variants()
    print("pool", len(specs), "ids unique", len({s.fid for s in specs}))
    from collections import defaultdict
    tm = defaultdict(float); allnan = []
    T = time.perf_counter()
    for sp in specs:
        t = time.perf_counter()
        try:
            sig, raw = sp.kernel(fr, **sp.params)
        except Exception as e:
            print("FAIL", sp.fid, type(e).__name__, e); continue
        tm[sp.domain] += time.perf_counter() - t
        assert sig.shape == (fr.n,), sp.fid
        if np.isnan(sig[-60:]).all(): allnan.append(sp.fid)
        elif np.nanstd(sig[-300:]) == 0: allnan.append(sp.fid + "(const)")
    print("total %.1f s" % (time.perf_counter() - T))
    for d, v in sorted(tm.items()): print("D%d %.1f s" % (d, v))
    print("all-nan/const:", len(allnan), allnan[:60])
