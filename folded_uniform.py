"""Exact-arithmetic utilities on a folded component S ∪ {ω}.

Used by verify_uniform.py and exp20_uniform.py.  All arithmetic is
fractions.Fraction unless a caller passes floats.
"""
from __future__ import annotations

from collections import defaultdict, deque
from fractions import Fraction
from itertools import product
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

INF = Fraction(10**18)
Num = Fraction

Action = Tuple[Optional[int], Num]  # (dst or None for ω, cost)


def F(x) -> Fraction:
    return x if isinstance(x, Fraction) else Fraction(x)


class Folded:
    def __init__(
        self,
        n: int,
        internal: Sequence[Tuple[int, int, object]],
        exits: Sequence[object],
        lam: object,
    ):
        self.n = n
        self.S = list(range(n))
        self.lam = F(lam)
        self.internal: Dict[int, List[Tuple[int, Num]]] = {i: [] for i in self.S}
        for x, y, g in internal:
            self.internal[int(x)].append((int(y), F(g)))
        self.e = [F(v) if v is not None and v != float("inf") else INF for v in exits]

    def outgoing(self, x: int) -> List[Action]:
        out: List[Action] = []
        if self.e[x] < INF:
            out.append((None, self.e[x]))
        for y, g in self.internal[x]:
            out.append((y, g))
        return out


def held_karp(comp: Folded) -> List[Num]:
    n = comp.n
    H: Dict[Tuple[int, int], Num] = {}
    for v in range(n):
        H[(v, 1 << v)] = comp.e[v]
    for size in range(2, n + 1):
        # iterate subsets with given size
        mask = (1 << size) - 1
        while mask < (1 << n):
            for v in range(n):
                if not (mask & (1 << v)):
                    continue
                best = comp.e[v]
                rest = mask ^ (1 << v)
                for y, g in comp.internal[v]:
                    if rest & (1 << y):
                        cand = g + comp.lam * H[(y, rest)]
                        if cand < best:
                            best = cand
                H[(v, mask)] = best
            # next combination of the same size (Gosper)
            c = mask & -mask
            r = mask + c
            mask = (((r ^ mask) >> 2) // c) | r
    full = (1 << n) - 1
    return [H[(v, full)] for v in range(n)]


def value_star(comp: Folded, iters: int = 80) -> List[Num]:
    """Discounted Bellman value by simultaneous Gauss–Seidel-style iteration.

    With positive costs the contraction is λ, so iters ≥ 80 is ample for
    exact-rational instances of moderate depth after a final exact PI.
    """
    n = comp.n
    V = [comp.e[x] if comp.e[x] < INF else Fraction(0) for x in range(n)]
    for _ in range(max(iters, 4 * n + 8)):
        nxt = []
        changed = False
        for x in range(n):
            best = comp.e[x]
            for y, g in comp.internal[x]:
                cand = g + comp.lam * V[y]
                if cand < best:
                    best = cand
            nxt.append(best)
            if best != V[x]:
                changed = True
        V = nxt
        if not changed:
            break
    return V


def local_excess(comp: Folded, J: Sequence[Num], x: int, y: Optional[int], cost: Num) -> Num:
    if y is None:
        return cost - J[x]
    return cost + comp.lam * J[y] - J[x]


def tree_cost(comp: Folded, mu: Sequence[Action], x: int) -> Num:
    """Discounted cost of the unique path of an in-tree; INF if a cycle is met."""
    seen = set()
    tot = Fraction(0)
    disc = Fraction(1)
    cur = x
    while cur is not None:
        if cur in seen:
            return INF
        seen.add(cur)
        y, g = mu[cur]
        tot += disc * g
        disc *= comp.lam
        cur = y
    return tot


def is_spanning_intree(comp: Folded, mu: Sequence[Action]) -> bool:
    for x in range(comp.n):
        if tree_cost(comp, mu, x) >= INF:
            return False
    return True


def enumerate_intrees(comp: Folded):
    choices: List[List[Action]] = []
    for x in range(comp.n):
        opts = list(comp.outgoing(x))
        if not opts:
            return
        choices.append(opts)
    for combo in product(*choices):
        if is_spanning_intree(comp, combo):
            yield combo


def excess_of_tree(comp: Folded, J: Sequence[Num], mu: Sequence[Action]) -> Num:
    return max(tree_cost(comp, mu, x) - J[x] for x in range(comp.n))


def exact_exc_star(comp: Folded, J: Sequence[Num]) -> Tuple[Num, Tuple[Optional[int], ...]]:
    best = INF
    best_mu = None
    for mu in enumerate_intrees(comp):
        ex = excess_of_tree(comp, J, mu)
        if ex < best:
            best = ex
            best_mu = mu
    if best_mu is None:
        raise RuntimeError("no spanning in-tree")
    return best, best_mu


def identity_holds(comp: Folded, J: Sequence[Num], mu: Sequence[Action], x: int) -> bool:
    seen = set()
    tot = Fraction(0)
    disc = Fraction(1)
    cur = x
    rhs = Fraction(0)
    while cur is not None:
        if cur in seen:
            return False
        seen.add(cur)
        y, cost = mu[cur]
        rhs += disc * local_excess(comp, J, cur, y, cost)
        tot += disc * cost
        disc *= comp.lam
        cur = y
    return tot - J[x] == rhs


def tight_arcs(comp: Folded, J: Sequence[Num]) -> Dict[int, List[Action]]:
    acts: Dict[int, List[Action]] = {x: [] for x in range(comp.n)}
    for x in range(comp.n):
        for y, cost in comp.outgoing(x):
            if local_excess(comp, J, x, y, cost) == 0:
                acts[x].append((y, cost))
    return acts


def spanning_from_acts(comp: Folded, acts: Dict[int, List[Action]]) -> bool:
    pred: Dict[Optional[int], List[int]] = defaultdict(list)
    for x, alist in acts.items():
        for y, _c in alist:
            pred[y].append(x)
    good: Set[int] = set()
    dq: deque = deque()
    for x in pred.get(None, []):
        good.add(x)
        dq.append(x)
    while dq:
        y = dq.popleft()
        for x in pred.get(y, []):
            if x not in good:
                good.add(x)
                dq.append(x)
    return len(good) == comp.n


def bottleneck(comp: Folded, J: Sequence[Num]) -> Tuple[Num, int, Dict[int, Action]]:
    """s*, h* (BFS hops in G_ε(s*)), and a reverse-BFS in-tree in G_ε(s*)."""
    cands: List[Tuple[Num, int, Action]] = []
    for x in range(comp.n):
        for y, cost in comp.outgoing(x):
            cands.append((local_excess(comp, J, x, y, cost), x, (y, cost)))
    cands.sort(key=lambda t: t[0])

    def reachable(theta: Num) -> Tuple[bool, int, Dict[int, Action]]:
        pred: Dict[Optional[int], List[Tuple[int, Action]]] = defaultdict(list)
        for eps, x, act in cands:
            if eps <= theta:
                pred[act[0]].append((x, act))
        good: Set[int] = set()
        dist: Dict[int, int] = {}
        tree: Dict[int, Action] = {}
        dq: deque = deque()
        for x, act in pred.get(None, []):
            if x not in good:
                good.add(x)
                dist[x] = 1
                tree[x] = act
                dq.append(x)
        while dq:
            y = dq.popleft()
            for x, act in pred.get(y, []):
                if x not in good:
                    good.add(x)
                    dist[x] = dist[y] + 1
                    tree[x] = act
                    dq.append(x)
        ok = len(good) == comp.n
        h = max(dist.values()) if dist else 0
        return ok, h, tree

    thetas = sorted({eps for eps, _, _ in cands if eps >= 0})
    if not thetas:
        thetas = [Fraction(0)]
    sstar = thetas[-1]
    hstar = comp.n
    tree: Dict[int, Action] = {}
    for theta in thetas:
        ok, h, tr = reachable(theta)
        if ok:
            sstar = theta
            hstar = h
            tree = tr
            break
    return sstar, hstar, tree


def greedy_vstar_tree(comp: Folded) -> Tuple[List[Action], List[Num]]:
    """Cycle-break greedy of the Sep 28 report: attach by min ρ_V + λ δ."""
    V = value_star(comp)
    n = comp.n
    A = {-1}  # ω
    delta = {-1: Fraction(0)}
    mu: Dict[int, Action] = {}
    remaining = set(range(n))

    def rho(x: int, y: Optional[int], cost: Num) -> Num:
        if y is None:
            return cost - V[x]
        return cost + comp.lam * V[y] - V[x]

    while remaining:
        best = None
        for x in remaining:
            for y, cost in comp.outgoing(x):
                v = -1 if y is None else y
                if v not in A:
                    continue
                cand = rho(x, y, cost) + comp.lam * delta[v]
                if best is None or cand < best[0]:
                    best = (cand, x, (y, cost))
        if best is None:
            raise RuntimeError("greedy could not attach")
        dlt, x, act = best
        mu[x] = act
        delta[x] = dlt
        A.add(x)
        remaining.remove(x)
    mud = [mu[x] for x in range(n)]
    cert = [delta[x] for x in range(n)]
    return mud, cert
