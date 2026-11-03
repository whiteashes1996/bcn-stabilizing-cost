"""Exact checks of the four reductions used in the paper.

  R1  two-cost class (internal a, exits b, a < (1-lam) b):
      J*(x) = c(m(x)) with m(x) the longest simple path from x;
  R2  Theorem (deciding Exc*(S)=0 for a two-state upstream component):
      Exc*({p,q}) = 0  iff  G' has a Hamiltonian path from s;
  R3  coNP gadget (chain of n-2 arcs from s):
      V^K(s) = J*(s)  iff  G' has no Hamiltonian path from s;
  R4  aggregate: min_T sum_x cost_T(x) <= n A + B (1-lam^n)/(1-lam)
      iff G' has a Hamiltonian path (any endpoints).

Run: python simulations/verify_reductions.py
"""
from __future__ import annotations

import itertools
import random
import sys
from fractions import Fraction
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from folded_uniform import Folded, enumerate_intrees, exact_exc_star, held_karp, tree_cost  # noqa: E402

LAM = Fraction(1, 2)
A_COST = Fraction(1)
B_EXIT = Fraction(3)          # (1-lam) b = 3/2 > a = 1


def c_of(k: int, lam=LAM, a=A_COST, b=B_EXIT) -> Fraction:
    return a * (1 - lam ** k) / (1 - lam) + lam ** k * b


def longest_simple_from(n, adj, s) -> int:
    best = 0
    stack = [(s, frozenset([s]), 0)]
    while stack:
        v, seen, d = stack.pop()
        best = max(best, d)
        for w in adj[v]:
            if w not in seen:
                stack.append((w, seen | {w}, d + 1))
    return best


def has_ham_path_from(n, adj, s) -> bool:
    return longest_simple_from(n, adj, s) == n - 1


def has_ham_path_any(n, adj) -> bool:
    return any(has_ham_path_from(n, adj, s) for s in range(n))


def random_digraph(rng, n, p):
    adj = {v: [w for w in range(n) if w != v and rng.random() < p] for v in range(n)}
    return adj


def main() -> int:
    rng = random.Random(7)
    fails = 0
    n_r1 = n_r2 = n_r3 = n_r4 = 0
    ham2 = {True: 0, False: 0}
    for trial in range(160):
        n = rng.randint(3, 5)
        adj = random_digraph(rng, n, rng.choice([0.3, 0.45, 0.6]))
        internal = [(v, w, A_COST) for v in range(n) for w in adj[v]]
        comp = Folded(n, internal, [B_EXIT] * n, LAM)
        J = held_karp(comp)
        # R1
        for x in range(n):
            if J[x] != c_of(longest_simple_from(n, adj, x)):
                fails += 1
                print("R1 FAIL", trial, x)
        n_r1 += 1
        s = 0
        ham = has_ham_path_from(n, adj, s)
        # R2: upstream two-state gadget p,q
        g0 = Fraction(1)
        E = g0 + LAM * J[s]
        c0 = (1 - LAM) * (g0 + LAM * c_of(n - 1))
        up = Folded(2, [(0, 1, c0), (1, 0, c0)], [E, E], LAM)
        Jup = held_karp(up)
        exc_up, _ = exact_exc_star(up, Jup)
        if (exc_up == 0) != ham:
            fails += 1
            print("R2 FAIL", trial, exc_up, ham)
        n_r2 += 1
        ham2[ham] += 1
        # R3: chain z_1..z_{n-2} from s (new states n..2n-3), exits b
        if n >= 3:
            m = n - 2
            chain = list(range(n, n + m))
            internal3 = list(internal) + [(s, chain[0], A_COST)]
            internal3 += [(chain[i], chain[i + 1], A_COST) for i in range(m - 1)]
            internal3 += [(chain[-1], s, A_COST)]
            comp3 = Folded(n + m, internal3, [B_EXIT] * (n + m), LAM)
            J3 = held_karp(comp3)
            vk = c_of(m)
            if (vk == J3[s]) != (not ham):
                fails += 1
                print("R3 FAIL", trial, vk, J3[s], ham)
            n_r3 += 1
        # R4: aggregate over in-trees of comp
        A = A_COST / (1 - LAM)
        B = B_EXIT - A
        theta = n * A + B * (1 - LAM ** n) / (1 - LAM)
        best = min(sum(tree_cost(comp, mu, x) for x in range(n)) for mu in enumerate_intrees(comp))
        if (best <= theta) != has_ham_path_any(n, adj):
            fails += 1
            print("R4 FAIL", trial, best, theta)
        n_r4 += 1
    print(f"R1 checked {n_r1} digraphs; R2 {n_r2} (ham={ham2[True]}, no-ham={ham2[False]}); "
          f"R3 {n_r3}; R4 {n_r4}; failures={fails}")
    return 0 if fails == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
