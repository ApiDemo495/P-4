"""Part 7 - symbolic breeding.  Genetic programming over expression trees whose
leaves are frame columns, rolling statistics and the *signals of surviving
formulas*, so a bred formula can combine a Lévy-area reading with a
Wasserstein shift and a p-adic valuation in one expression.

Trees are nested tuples ``(op, child, ...)`` / ``("col", name)`` /
``("sig", fid)`` / ``("k", value)``; they are evaluated by walking the tree,
never by ``eval``.  Depth is capped at 4 and every generation is scored by the
same seven metrics as the templates, so bred formulas compete on equal terms.
"""
from __future__ import annotations

import random
from typing import Callable

import numpy as np

from backend.genesis import ops
from backend.genesis.spec import FormulaSpec

UNARY = ("tanh", "sign", "neg", "abs", "z21", "z55", "ema5", "ema13", "diff1", "diff3", "lag1", "rank34")
BINARY = ("add", "sub", "mul", "div", "max", "min")
COLS = ("ret", "bar_ret", "vol_ret", "imbalance", "other_ret", "rng", "spread_bps",
        "dxy_ret", "social_ret", "exch_flow")   # the last three are NaN->0 until a keyed provider is configured
MAX_DEPTH = 4


def _leaf(rng: random.Random, sig_ids: list[str]):
    r = rng.random()
    if sig_ids and r < 0.6:
        return ("sig", rng.choice(sig_ids))
    if r < 0.9:
        return ("col", rng.choice(COLS))
    return ("k", round(rng.uniform(-1.5, 1.5), 2))


def random_tree(rng: random.Random, sig_ids: list[str], depth: int = 0):
    if depth >= MAX_DEPTH or (depth > 0 and rng.random() < 0.3):
        return _leaf(rng, sig_ids)
    if rng.random() < 0.5:
        return (rng.choice(UNARY), random_tree(rng, sig_ids, depth + 1))
    return (rng.choice(BINARY), random_tree(rng, sig_ids, depth + 1), random_tree(rng, sig_ids, depth + 1))


def _nodes(tree, path=()):
    yield path, tree
    if tree[0] in UNARY or tree[0] in BINARY:
        for i, child in enumerate(tree[1:], start=1):
            yield from _nodes(child, path + (i,))


def _replace(tree, path, new):
    if not path:
        return new
    i = path[0]
    lst = list(tree)
    lst[i] = _replace(tree[i], path[1:], new)
    return tuple(lst)


def _depth(tree) -> int:
    if tree[0] in UNARY or tree[0] in BINARY:
        return 1 + max(_depth(c) for c in tree[1:])
    return 1


def crossover(rng: random.Random, a, b):
    pa = rng.choice([p for p, _ in _nodes(a)])
    pb, sub = rng.choice(list(_nodes(b)))
    child = _replace(a, pa, sub)
    return child if _depth(child) <= MAX_DEPTH + 1 else a


def mutate(rng: random.Random, tree, sig_ids: list[str]):
    path, node = rng.choice(list(_nodes(tree)))
    r = rng.random()
    if node[0] == "k":
        return _replace(tree, path, ("k", round(node[1] * rng.uniform(0.5, 1.5) + rng.uniform(-0.2, 0.2), 2)))
    if r < 0.4:
        return _replace(tree, path, random_tree(rng, sig_ids, len(path)))
    if node[0] in UNARY:
        return _replace(tree, path, (rng.choice(UNARY),) + node[1:])
    if node[0] in BINARY:
        return _replace(tree, path, (rng.choice(BINARY),) + node[1:])
    return _replace(tree, path, _leaf(rng, sig_ids))


def to_text(tree) -> str:
    op = tree[0]
    if op == "col":
        return tree[1]
    if op == "sig":
        return f"S[{tree[1]}]"
    if op == "k":
        return f"{tree[1]:g}"
    if op in UNARY:
        return f"{op}({to_text(tree[1])})"
    sym = {"add": "+", "sub": "-", "mul": "*", "div": "/"}.get(op)
    if sym:
        return f"({to_text(tree[1])} {sym} {to_text(tree[2])})"
    return f"{op}({to_text(tree[1])}, {to_text(tree[2])})"


def evaluate(tree, frame, signals: dict) -> np.ndarray:
    n = frame.n
    op = tree[0]
    with np.errstate(all="ignore"):
        if op == "col":
            return np.nan_to_num(np.asarray(frame.col(tree[1]), dtype=np.float64))
        if op == "sig":
            s = signals.get(tree[1])
            if s is None or len(s) != n:
                return np.zeros(n)
            return np.nan_to_num(np.asarray(s, dtype=np.float64))
        if op == "k":
            return np.full(n, float(tree[1]))
        if op in UNARY:
            x = evaluate(tree[1], frame, signals)
            if op == "tanh":
                return np.tanh(x)
            if op == "sign":
                return np.sign(x)
            if op == "neg":
                return -x
            if op == "abs":
                return np.abs(x)
            if op == "z21":
                return np.nan_to_num(ops.zscore(x, 21))
            if op == "z55":
                return np.nan_to_num(ops.zscore(x, 55))
            if op == "ema5":
                return np.nan_to_num(ops.ema(x, 5))
            if op == "ema13":
                return np.nan_to_num(ops.ema(x, 13))
            if op == "diff1":
                return np.nan_to_num(ops.diff(x, 1))
            if op == "diff3":
                return np.nan_to_num(ops.diff(x, 3))
            if op == "lag1":
                return np.nan_to_num(ops.shift(x, 1))
            if op == "rank34":
                return np.nan_to_num(ops.rrank(x, 34))
        a = evaluate(tree[1], frame, signals)
        b = evaluate(tree[2], frame, signals)
        if op == "add":
            return a + b
        if op == "sub":
            return a - b
        if op == "mul":
            return a * b
        if op == "div":
            return np.where(np.abs(b) > 1e-9, a / np.where(np.abs(b) > 1e-9, b, 1.0), 0.0)
        if op == "max":
            return np.maximum(a, b)
        if op == "min":
            return np.minimum(a, b)
    return np.zeros(n)


def make_kernel(tree) -> Callable:
    def kernel(frame, **params):
        signals = frame.cache.get("__signals__", {})
        raw = evaluate(tree, frame, signals)
        sig = ops.normalize(raw, "tanh", 55)
        return np.nan_to_num(sig), raw
    return kernel


def breed(rng: random.Random, parents: list[FormulaSpec], sig_ids: list[str], generation: int,
          count: int = 50) -> list[FormulaSpec]:
    """Produce ``count`` new specs: 40 % crossover of two parents, 40 % mutation
    of one parent, 20 % fresh random trees."""
    trees = [p.expression_tree for p in parents if getattr(p, "expression_tree", None) is not None]
    out = []
    for i in range(count):
        r = rng.random()
        if trees and len(trees) >= 2 and r < 0.4:
            t = crossover(rng, rng.choice(trees), rng.choice(trees))
            how = "crossover"
        elif trees and r < 0.8:
            t = mutate(rng, rng.choice(trees), sig_ids)
            how = "mutation"
        else:
            t = random_tree(rng, sig_ids)
            how = "random"
        fid = f"G{generation:03d}.{i + 1:03d}"
        spec = FormulaSpec(fid=fid, name=f"bred {how} {fid}", domain=0, subcategory=how, variant=i + 1,
                           params={"generation": generation}, kernel=make_kernel(t), layer=2,
                           definition=to_text(t),
                           interpretation="Genetically bred expression over frame columns and surviving formula signals; "
                                          "tanh-normalised over 55 candles, positive = up.",
                           origin="bred", expression=to_text(t))
        spec.expression_tree = t
        out.append(spec)
    return out

