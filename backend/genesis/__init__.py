"""Formula Genesis Engine (v3.0, Round AK).

A pool of 2,100 generated formulas across ten mathematical domains, scored on
rolling history, bred and culled by a genetic programme in the background,
with the best 200 (correlation-pruned, regime-gated) voting into the lock.

Package map
-----------
candles     minute OHLCV + microstructure store per asset (Layer 1 inputs)
features    one vectorised pre-computation pass -> ``Frame`` (Layer 1)
spec        ``FormulaSpec`` / lifecycle states / the generative grid
domains/    the ten domains, each exporting ``variants()`` (210 specs each)
fitness     the seven metrics, overfit penalty, greedy correlation selection
regime      trending / mean-reverting / high-vol / low-vol / cascade gate
lifecycle   BIRTH -> CANDIDATE -> ACTIVE -> DECAYING -> DEAD -> AUTOPSY
symbolic    genetic programming over expression trees (background breeding)
engine      ``GenesisEngine`` - owns everything, feeds fusion, serves the API
"""
import warnings as _warnings

# NaN-aware rolling statistics on a warming-up tape raise "mean of empty
# slice" style warnings by design; they are not errors.
_warnings.filterwarnings("ignore", category=RuntimeWarning, module=r"backend\.genesis.*")
