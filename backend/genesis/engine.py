"""``GenesisEngine`` - owns the candle stores, the formula pool, scoring,
lifecycle, breeding and the live composite vote that fusion consumes.

Everything heavy runs on one background worker thread (NumPy only - the
Codespace has no GPU and no Rust toolchain, so the pool is vectorised and
strided rather than compiled):

* per asset, when a minute closes       -> evaluate the regime-gated active set
* every ``RESCORE_EVERY`` closed candles -> score the whole pool, reselect the 200
* every ``GENESIS_EVERY_S`` seconds      -> breed a generation from the survivors

The engine never blocks the feed: the hub's listener only appends ticks to a
``CandleStore``; the worker notices the new minute on its next tick.
"""
from __future__ import annotations

import json
import logging
import os
import random
import threading
import time
import warnings
from collections import Counter, defaultdict

import numpy as np

from backend.genesis import fitness as fit
from backend.genesis import lifecycle as life
from backend.genesis import regime as reg
from backend.genesis import symbolic
from backend.genesis.candles import CandleStore, bootstrap_history
from backend.genesis.domains import all_variants
from backend.genesis.features import build_frame
from backend.genesis.spec import DOMAINS, FormulaSpec, State

log = logging.getLogger("drosophila.genesis")

ASSETS = ("BTC", "PAXG")
ACTIVE_SIZE = 200
MIN_HISTORY = 240          # candles before the first full scoring
RESCORE_EVERY = 100        # candles between full re-evaluations
GENESIS_EVERY_S = 4 * 3600
BREED_COUNT = 50
HISTORY_ROWS = 1500
TICK_SIZE = {"BTC": 0.01, "PAXG": 0.01}
_RNG = random.Random(2101)


class GenesisEngine:
    def __init__(self, state_dir: str | None = None, assets=ASSETS) -> None:
        self.assets = tuple(assets)
        self.state_dir = state_dir
        if state_dir:
            os.makedirs(state_dir, exist_ok=True)
        self.stores = {a: CandleStore(a, state_dir) for a in self.assets}
        self.templates: list[FormulaSpec] = all_variants()
        self.pools: dict[str, dict[str, life.Formula]] = {
            a: {sp.fid: life.Formula(spec=sp) for sp in self.templates} for a in self.assets}
        self.active: dict[str, list[life.Formula]] = {a: [] for a in self.assets}
        self.signals: dict[str, dict[str, np.ndarray]] = {a: {} for a in self.assets}
        self.latest: dict[str, dict] = {a: self._empty_payload(a) for a in self.assets}
        self.generation: dict[str, int] = {a: 0 for a in self.assets}
        self.scored_at_len: dict[str, int] = {a: 0 for a in self.assets}
        self.last_genesis: dict[str, float] = {a: 0.0 for a in self.assets}
        self.last_rescore: dict[str, float] = {a: 0.0 for a in self.assets}
        self.rescore_seconds: dict[str, float] = {a: 0.0 for a in self.assets}
        self.candle_count: dict[str, int] = {a: 0 for a in self.assets}
        self.graveyard: dict[str, list] = {a: [] for a in self.assets}
        self.events: list[dict] = []
        self._lock = threading.RLock()
        self._busy = ""
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._seen_len: dict[str, int] = {a: -1 for a in self.assets}
        self._held: dict[str, dict[str, tuple[float, float]]] = {a: {} for a in self.assets}
        for a in self.assets:
            self._load(a)

    # ------------------------------------------------------------ wiring
    def attach(self, hub) -> None:
        """Register as a tape listener on the market hub."""
        if self._on_tape not in hub.listeners:
            hub.listeners.append(self._on_tape)

    def _on_tape(self, asset: str, ticks, book) -> None:
        store = self.stores.get(asset)
        if store is not None:
            store.on_tape(ticks, book)

    async def bootstrap(self) -> dict:
        out = {}
        for a in self.assets:
            try:
                rows, src = await bootstrap_history(a)
                n = self.stores[a].bootstrap(rows, src) if len(rows) else 0
                out[a] = {"rows": int(n), "source": src}
            except Exception as exc:  # noqa: BLE001
                out[a] = {"rows": 0, "source": "", "error": str(exc)}
        self._event("bootstrap", str(out))
        return out

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name="genesis-worker", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        for a in self.assets:
            try:
                self.stores[a].save()
                self._save(a)
            except Exception:  # noqa: BLE001
                pass

    # ------------------------------------------------------------ worker
    def _loop(self) -> None:
        warnings.simplefilter("ignore", RuntimeWarning)
        while not self._stop.is_set():
            try:
                for a in self.assets:
                    self._tick(a)
            except Exception as exc:  # noqa: BLE001
                log.exception("genesis worker: %s", exc)
                self._event("error", str(exc))
            self._stop.wait(1.0)

    def _tick(self, asset: str) -> None:
        store = self.stores[asset]
        n = len(store)
        if n == self._seen_len[asset]:
            return
        self._seen_len[asset] = n
        if n < MIN_HISTORY:
            self.latest[asset] = self._empty_payload(asset, f"warming up: {n}/{MIN_HISTORY} candles")
            return
        if self.scored_at_len[asset] == 0 or n - self.scored_at_len[asset] >= RESCORE_EVERY:
            self.rescore(asset)
        if self.scored_at_len[asset] and time.time() - self.last_genesis[asset] >= GENESIS_EVERY_S:
            self.genesis(asset)
        self.candle_count[asset] += 1
        self.evaluate_live(asset)

    # ------------------------------------------------------------ frames
    def _frame(self, asset: str, include_open: bool = False, eval_tail: int = 700):
        other = [a for a in self.assets if a != asset]
        store = self.stores[asset]
        rows = store.rows(limit=HISTORY_ROWS, include_open=include_open)
        other_rows = self.stores[other[0]].rows(limit=HISTORY_ROWS, include_open=include_open) if other else None
        fr = build_frame(asset, rows, other_rows, books=store.recent_books(8), tick_size=TICK_SIZE.get(asset, 0.01))
        fr.eval_tail = eval_tail
        return fr

    @staticmethod
    def _run(formula: life.Formula, frame, signals: dict) -> tuple[np.ndarray, np.ndarray] | None:
        sp = formula.spec
        try:
            frame.cache["__signals__"] = signals
            with np.errstate(all="ignore"):
                sig, raw = sp.kernel(frame, **sp.params)
            sig = np.asarray(sig, dtype=np.float64)
            raw = np.asarray(raw, dtype=np.float64)
            if sig.shape != (frame.n,):
                raise ValueError(f"signal shape {sig.shape} != ({frame.n},)")
            if raw.shape != (frame.n,):
                raw = np.full(frame.n, np.nan) if raw.ndim == 0 or len(raw) != frame.n else raw
            formula.error = ""
            return np.clip(np.nan_to_num(sig, nan=0.0), -1.0, 1.0), raw
        except Exception as exc:  # noqa: BLE001
            formula.error = f"{type(exc).__name__}: {exc}"[:160]
            return None

    # ------------------------------------------------------------ scoring
    def rescore(self, asset: str) -> dict:
        """Score every living formula on the last ``HISTORY_ROWS`` candles,
        reselect the active 200, apply lifecycle transitions and autopsies."""
        t0 = time.perf_counter()
        self._busy = f"scoring {asset}"
        frame = self._frame(asset, include_open=False, eval_tail=700)
        labels = reg.regime_labels(frame)
        pool = self.pools[asset]
        signals: dict[str, np.ndarray] = {}
        # templates first, bred formulas after (they may read template signals)
        order = sorted(pool.values(), key=lambda f: (f.spec.origin != "template", f.fid))
        for f in order:
            if f.state in (State.DEAD, State.AUTOPSY):
                continue
            res = self._run(f, frame, signals)
            if res is None:
                f.record(fit.Score(fitness=-1.0, note=f.error))
                continue
            sig, raw = res
            signals[f.fid] = sig
            f.record(fit.score(sig, frame.ret, labels, eval_tail=700))
            f.last_signal = float(sig[-1])
            f.last_raw = float(raw[-1]) if np.isfinite(raw[-1]) else 0.0
        alive = [f for f in pool.values() if f.state not in (State.DEAD, State.AUTOPSY)]
        ranked = sorted(alive, key=lambda f: f.fitness, reverse=True)
        eligible = [f for f in ranked if f.fitness >= life.CANDIDATE_FLOOR]
        chosen = fit.select(eligible, signals, k=ACTIVE_SIZE)
        chosen_ids = {f.fid for f in chosen}
        newly_dead = []
        for f in alive:
            before = f.state
            after = life.transition(f, f.fid in chosen_ids)
            if after == State.DEAD and before != State.DEAD:
                newly_dead.append(f)
        for f in newly_dead:
            sig = signals.get(f.fid)
            if sig is None:
                f.state = State.AUTOPSY
                f.autopsy = {"verdict": "archived", "cause": f.error or "no signal"}
            else:
                life.autopsy(f, sig, frame.ret, labels)
            if f.state == State.AUTOPSY:
                self.graveyard[asset].append({"id": f.fid, "name": f.spec.name, "fitness": round(f.fitness, 4),
                                              "cause": f.autopsy.get("cause"), "at": time.time()})
                del self.graveyard[asset][:-300]
        with self._lock:
            self.active[asset] = [f for f in chosen if f.state == State.ACTIVE]
            self.signals[asset] = signals
            self.scored_at_len[asset] = len(self.stores[asset])
            self.last_rescore[asset] = time.time()
            self.rescore_seconds[asset] = round(time.perf_counter() - t0, 1)
        self._busy = ""
        self._save(asset)
        summary = {"asset": asset, "scored": len(signals), "active": len(self.active[asset]),
                   "dead": len(newly_dead), "seconds": self.rescore_seconds[asset],
                   "best": [(f.fid, round(f.fitness, 3)) for f in ranked[:5]]}
        self._event("rescore", json.dumps(summary))
        log.info("genesis rescore %s", summary)
        return summary

    # ------------------------------------------------------------ breeding
    def genesis(self, asset: str) -> dict:
        """Breed a generation from the fittest survivors, score the children on
        the same history and let the lifecycle decide."""
        pool = self.pools[asset]
        survivors = sorted((f for f in pool.values() if f.state == State.ACTIVE), key=lambda f: f.fitness, reverse=True)
        parents = [f.spec for f in survivors if f.spec.origin == "bred"][:40]
        sig_ids = [f.fid for f in survivors[:60]]
        self.generation[asset] += 1
        gen = self.generation[asset]
        children = symbolic.breed(_RNG, parents, sig_ids, gen, BREED_COUNT)
        # cap the bred population so the pool stays ~2,100 + a few hundred
        bred_alive = [f for f in pool.values() if f.spec.origin == "bred" and f.state not in (State.DEAD, State.AUTOPSY)]
        if len(bred_alive) > 400:
            for f in sorted(bred_alive, key=lambda f: f.fitness)[: len(bred_alive) - 400]:
                f.state = State.AUTOPSY
                f.autopsy = {"verdict": "archived", "cause": "population cap"}
        for sp in children:
            pool[sp.fid] = life.Formula(spec=sp)
        self.last_genesis[asset] = time.time()
        self._event("genesis", f"{asset} generation {gen}: {len(children)} children from {len(parents)} bred parents "
                               f"and {len(sig_ids)} survivor signals")
        self.rescore(asset)
        born = [pool[sp.fid] for sp in children]
        alive = sum(1 for f in born if f.state != State.DEAD)
        return {"generation": gen, "children": len(children), "alive_after_scoring": alive}

    # ------------------------------------------------------------ live
    def evaluate_live(self, asset: str) -> dict:
        """Run the regime-gated active set on the latest closed candle and
        publish the composite vote."""
        with self._lock:
            active = list(self.active[asset])
        if not active:
            self.latest[asset] = self._empty_payload(asset, "no active formulas yet")
            return self.latest[asset]
        self._busy = f"live {asset}"
        frame = self._frame(asset, include_open=False, eval_tail=8)
        regime_info = reg.classify(frame)
        regime = regime_info["regime"]
        gated = reg.gate(active, regime)
        count = self.candle_count[asset]
        held = self._held[asset]
        signals: dict[str, np.ndarray] = {}
        readings = []
        for f in gated:
            if f.spec.layer >= 4 and count % 5 and f.fid in held:
                s_val, r_val = held[f.fid]
            else:
                res = self._run(f, frame, signals)
                if res is None:
                    continue
                sig, raw = res
                signals[f.fid] = sig
                s_val = float(sig[-1])
                r_val = float(raw[-1]) if np.isfinite(raw[-1]) else 0.0
                held[f.fid] = (s_val, r_val)
            s_val = life.gated_signal(f, s_val, regime)
            f.last_signal, f.last_raw = s_val, r_val
            readings.append((f, s_val, r_val))
        self._busy = ""
        payload = self._compose(asset, readings, regime_info, len(active), frame)
        self.latest[asset] = payload
        return payload

    def _compose(self, asset: str, readings, regime_info: dict, n_active: int, frame) -> dict:
        w_sum = 0.0
        v_sum = 0.0
        votes_up = votes_down = 0
        by_domain: dict[int, list] = defaultdict(list)
        rows = []
        for f, s, r in readings:
            w = max(0.0, f.fitness - 0.30)
            if s != 0.0:
                w_sum += w
                v_sum += w * s
                if s > 0:
                    votes_up += 1
                else:
                    votes_down += 1
            by_domain[f.spec.domain].append(s)
            rows.append({"id": f.fid, "name": f.spec.name, "domain": f.spec.domain, "signal": round(s, 4),
                         "raw": round(r, 6), "fitness": round(f.fitness, 4), "weight": round(w, 4),
                         "layer": f.spec.layer, "origin": f.spec.origin, "state": f.state.value})
        vote = v_sum / w_sum if w_sum > 0 else 0.0
        firing = votes_up + votes_down
        agreement = (max(votes_up, votes_down) / firing) if firing else 0.0
        coverage = firing / max(1, len(readings))
        confidence = min(1.0, abs(vote) * (0.5 + 0.5 * agreement) * (0.5 + 0.5 * min(1.0, coverage / 0.5)))
        rows.sort(key=lambda d: abs(d["signal"]) * d["weight"], reverse=True)
        domains = []
        for d in sorted(by_domain):
            vals = by_domain[d]
            domains.append({"domain": d, "name": DOMAINS[d][0] if d in DOMAINS else "bred",
                            "count": len(vals), "mean": round(float(np.mean(vals)), 4),
                            "firing": int(sum(1 for v in vals if v != 0))})
        return {"status": "live", "asset": asset, "vote": round(float(vote), 4),
                "decision": "BUY" if vote > 0 else ("SELL" if vote < 0 else "NEUTRAL"),
                "confidence": round(float(confidence), 4), "agreement": round(float(agreement), 4),
                "votes_up": votes_up, "votes_down": votes_down, "firing": firing,
                "regime": regime_info, "active": n_active, "gated": len(readings),
                "domains": domains, "top": rows[:12], "readings": rows,
                "candle_ts_ms": float(frame.ts_ms[-1]) if frame.n else 0.0,
                "close": float(frame.close[-1]) if frame.n else 0.0,
                "generation": self.generation[asset], "computed_at": time.time(),
                "note": f"{firing}/{len(readings)} gated formulas firing in {regime_info['regime']} · "
                        f"{votes_up} up / {votes_down} down"}

    # ------------------------------------------------------------ API
    def _empty_payload(self, asset: str, note: str = "not started") -> dict:
        return {"status": "warming", "asset": asset, "vote": 0.0, "decision": "NEUTRAL", "confidence": 0.0,
                "agreement": 0.0, "votes_up": 0, "votes_down": 0, "firing": 0, "regime": {"regime": "low_vol"},
                "active": 0, "gated": 0, "domains": [], "top": [], "readings": [], "generation": 0,
                "computed_at": time.time(), "note": note}

    def vote(self, asset: str) -> dict:
        """What fusion consumes: the latest composite for ``asset``."""
        return dict(self.latest.get(asset) or self._empty_payload(asset))

    def status(self, asset: str | None = None) -> dict:
        assets = [asset] if asset else list(self.assets)
        out = {"compute": "NumPy, single worker thread (no GPU / Rust in the Codespace)",
               "pool_templates": len(self.templates), "busy": self._busy, "assets": {}}
        for a in assets:
            counts = Counter(f.state.value for f in self.pools[a].values())
            bred = sum(1 for f in self.pools[a].values() if f.spec.origin == "bred")
            since = len(self.stores[a]) - self.scored_at_len[a] if self.scored_at_len[a] else 0
            out["assets"][a] = {
                "candles": self.stores[a].status(), "states": dict(counts), "pool": len(self.pools[a]),
                "bred": bred, "active": len(self.active[a]), "generation": self.generation[a],
                "last_rescore": self.last_rescore[a], "rescore_seconds": self.rescore_seconds[a],
                "next_rescore_in_candles": max(0, RESCORE_EVERY - since) if self.scored_at_len[a] else None,
                "next_genesis_in_s": (max(0.0, GENESIS_EVERY_S - (time.time() - self.last_genesis[a]))
                                      if self.scored_at_len[a] else None),
                "graveyard": len(self.graveyard[a]), "latest": {k: v for k, v in self.latest[a].items()
                                                                 if k not in ("readings", "top", "domains")}}
        out["events"] = self.events[-20:]
        return out

    def list_formulas(self, asset: str, state: str | None = None, domain: int | None = None,
                      limit: int = 200, offset: int = 0) -> dict:
        pool = self.pools.get(asset) or {}
        items = list(pool.values())
        if state:
            items = [f for f in items if f.state.value == state.upper()]
        if domain is not None:
            items = [f for f in items if f.spec.domain == domain]
        items.sort(key=lambda f: f.fitness, reverse=True)
        total = len(items)
        return {"asset": asset, "total": total, "offset": offset,
                "items": [f.as_dict() for f in items[offset: offset + limit]]}

    def formula(self, asset: str, fid: str) -> dict | None:
        f = (self.pools.get(asset) or {}).get(fid)
        if f is None:
            return None
        d = f.as_dict(verbose=True)
        sig = self.signals.get(asset, {}).get(fid)
        if sig is not None:
            d["signal_tail"] = [round(float(x), 4) for x in sig[-120:]]
        return d

    def domains(self) -> list[dict]:
        out = []
        for d, (name, desc) in DOMAINS.items():
            subs = []
            seen = set()
            for sp in self.templates:
                if sp.domain == d and sp.subcategory not in seen:
                    seen.add(sp.subcategory)
                    subs.append({"subcategory": sp.subcategory, "name": sp.name.rsplit(" v", 1)[0],
                                 "definition": sp.definition, "interpretation": sp.interpretation, "layer": sp.layer})
            out.append({"domain": d, "name": name, "description": desc, "subcategories": subs})
        return out

    # ------------------------------------------------------------ persistence
    def _path(self, asset: str) -> str | None:
        return os.path.join(self.state_dir, f"genesis_{asset}.json") if self.state_dir else None

    def _save(self, asset: str) -> None:
        path = self._path(asset)
        if not path:
            return
        pool = self.pools[asset]
        data = {"generation": self.generation[asset], "last_genesis": self.last_genesis[asset],
                "graveyard": self.graveyard[asset][-100:],
                "formulas": {fid: {"state": f.state.value, "fitness": round(f.fitness, 4), "strikes": f.strikes,
                                   "lives": f.lives, "gates": f.spec.gates, "history": f.history[-20:],
                                   "tree": f.spec.expression_tree if f.spec.origin == "bred" else None,
                                   "autopsy": f.autopsy if f.state == State.AUTOPSY else {}}
                             for fid, f in pool.items()
                             if f.state != State.BIRTH or f.spec.origin == "bred"}}
        try:
            tmp = path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as fh:
                json.dump(data, fh, default=_json_default)
            os.replace(tmp, path)
        except OSError as exc:
            log.debug("genesis save failed: %s", exc)

    def _load(self, asset: str) -> None:
        path = self._path(asset)
        if not path or not os.path.exists(path):
            return
        try:
            with open(path, encoding="utf-8") as fh:
                data = json.load(fh)
        except (OSError, ValueError):
            return
        pool = self.pools[asset]
        self.generation[asset] = int(data.get("generation") or 0)
        self.last_genesis[asset] = float(data.get("last_genesis") or 0.0)
        self.graveyard[asset] = list(data.get("graveyard") or [])
        for fid, rec in (data.get("formulas") or {}).items():
            f = pool.get(fid)
            if f is None and rec.get("tree") is not None:
                tree = _tuplify(rec["tree"])
                sp = FormulaSpec(fid=fid, name=f"bred {fid}", domain=0, subcategory="bred", variant=0,
                                 params={}, kernel=symbolic.make_kernel(tree), layer=2,
                                 definition=symbolic.to_text(tree), origin="bred",
                                 expression=symbolic.to_text(tree))
                sp.expression_tree = tree
                f = pool[fid] = life.Formula(spec=sp)
            if f is None:
                continue
            try:
                f.state = State(rec.get("state") or "BIRTH")
            except ValueError:
                f.state = State.BIRTH
            f.score = fit.Score(fitness=float(rec.get("fitness") or -1.0))
            f.strikes = int(rec.get("strikes") or 0)
            f.lives = int(rec.get("lives") or 0)
            f.spec.gates = dict(rec.get("gates") or {})
            f.history = [tuple(h) for h in rec.get("history") or []]
            f.autopsy = dict(rec.get("autopsy") or {})
        # ACTIVE membership is recomputed by the first rescore; until then the
        # saved ACTIVE set votes so a restart does not go dark for 100 candles.
        self.active[asset] = sorted((f for f in pool.values() if f.state == State.ACTIVE),
                                    key=lambda f: f.fitness, reverse=True)[:ACTIVE_SIZE]

    def _event(self, kind: str, text: str) -> None:
        self.events.append({"at": time.time(), "kind": kind, "text": text[:400]})
        del self.events[:-100]


def _json_default(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, tuple):
        return list(o)
    return str(o)


def _tuplify(x):
    if isinstance(x, list):
        return tuple(_tuplify(c) for c in x)
    return x
