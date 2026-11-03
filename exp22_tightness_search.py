"""Hill-climbing search for large Exc*/s* on small folded components.

Exact Fractions; Exc* by enumeration of spanning in-trees.
Run: python simulations/exp22_tightness_search.py [seconds_per_lambda]
"""
from __future__ import annotations

import os
import random
import sys
import time
from fractions import Fraction

sys.path.insert(0, os.path.dirname(__file__))

from folded_uniform import INF, Folded, bottleneck, exact_exc_star, held_karp  # noqa: E402

RESULTS = os.path.join(os.path.dirname(__file__), "results")


def random_instance(rng: random.Random, n: int):
    arcs = []
    for x in range(n):
        targets = rng.sample([y for y in range(n) if y != x], k=min(n - 1, rng.randint(1, 3)))
        for y in targets:
            arcs.append([x, y, Fraction(rng.randint(1, 40), rng.randint(1, 8))])
    exits = [Fraction(rng.randint(2, 80), rng.randint(1, 4)) if rng.random() < 0.5 else None
             for _ in range(n)]
    if all(e is None for e in exits):
        exits[rng.randrange(n)] = Fraction(rng.randint(2, 80))
    return arcs, exits


def evaluate(n, arcs, exits, lam):
    comp = Folded(n, [tuple(a) for a in arcs], [INF if e is None else e for e in exits], lam)
    J = held_karp(comp)
    if any(j >= INF / 2 for j in J):
        return None
    try:
        exc, _ = exact_exc_star(comp, J)
    except RuntimeError:
        return None
    s, h, _ = bottleneck(comp, J)
    if s <= 0:
        return None
    return exc / s, exc, s, h


def mutate(rng, arcs, exits):
    arcs = [list(a) for a in arcs]
    exits = list(exits)
    r = rng.random()
    if r < 0.6 and arcs:
        a = rng.choice(arcs)
        a[2] = max(Fraction(1, 64), a[2] * Fraction(rng.randint(3, 13), 8))
    elif r < 0.9:
        i = rng.randrange(len(exits))
        if exits[i] is not None:
            exits[i] = max(Fraction(1, 8), exits[i] * Fraction(rng.randint(3, 13), 8))
    else:
        i = rng.randrange(len(exits))
        exits[i] = None if exits[i] is not None else Fraction(rng.randint(2, 80))
        if all(e is None for e in exits):
            exits[i] = Fraction(10)
    return arcs, exits


def search(lam: Fraction, budget_s: float, seed: int):
    rng = random.Random(seed)
    best = None
    t_end = time.time() + budget_s
    while time.time() < t_end:
        n = rng.randint(4, 6)
        arcs, exits = random_instance(rng, n)
        cur = evaluate(n, arcs, exits, lam)
        if cur is None:
            continue
        for _ in range(150):
            if time.time() >= t_end:
                break
            a2, e2 = mutate(rng, arcs, exits)
            val = evaluate(n, a2, e2, lam)
            if val is not None and val[0] >= cur[0]:
                arcs, exits, cur = a2, e2, val
        if best is None or cur[0] > best[0][0]:
            best = (cur, n, arcs, exits)
    return best


def main() -> int:
    budget = float(sys.argv[1]) if len(sys.argv) > 1 else 120.0
    out = ["E22: hill-climbing for Exc*/s* (exact)"]
    for lam in (Fraction(1, 2), Fraction(3, 4), Fraction(9, 10)):
        best = search(lam, budget, seed=int(lam * 1000))
        if best is None:
            out.append(f"  lam={lam}: no instance with s*>0")
            continue
        (ratio, exc, s, h), n, arcs, exits = best
        fac = (1 - lam ** h) / (1 - lam)
        out.append(f"  lam={lam}: best Exc*/s*={ratio} ≈ {float(ratio):.4f}; Exc*={exc} s*={s} "
                   f"h*={h} factor={fac} ≈ {float(fac):.4f}; n={n}")
        out.append(f"    arcs={[(a[0], a[1], str(a[2])) for a in arcs]}")
        out.append(f"    exits={[None if e is None else str(e) for e in exits]}")
        print("\n".join(out[-3:]), flush=True)
    with open(os.path.join(RESULTS, "exp22_tightness_search.txt"), "w") as fh:
        fh.write("\n".join(out) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
