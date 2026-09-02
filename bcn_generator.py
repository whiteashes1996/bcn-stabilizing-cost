"""Synthetic BCN generators for large-scale experiments."""
from __future__ import annotations

import random
from typing import Callable, Dict, List, Tuple

from ara_io import control_fixpoints, hamming_cost_factory, next_state
from core_solver import WeightedGraph, build_graph


def random_bcn(
    n: int,
    m: int = 2,
    seed: int = 0,
    ensure_fixpoint: bool = True,
    acyclic: bool = False,
) -> Tuple[int, int, int, int, List[int], int, int, Callable[[int, int], int]]:
    """
    Random logical network L with N=2^n, M=2^m.
    Target xe=N (all-ones state), ue=1 by default when ensure_fixpoint.
    If acyclic=True, transitions strictly increase a random rank (DAG toward xe).
    """
    rng = random.Random(seed)
    N, M = 2**n, 2**m
    xe, ue = N, 1
    L = [rng.randint(1, N) for _ in range(M * N)]
    if acyclic:
        order = [s for s in range(1, N + 1) if s != xe]
        rng.shuffle(order)
        order.append(xe)
        rank = {s: i for i, s in enumerate(order)}
        for x in range(1, N + 1):
            for u in range(1, M + 1):
                if x == xe and u == ue:
                    continue
                cands = [y for y in range(1, N + 1) if rank[y] > rank[x]]
                if not cands:
                    cands = [xe]
                L[(u - 1) * N + (x - 1)] = rng.choice(cands)
    if ensure_fixpoint:
        L[(ue - 1) * N + (xe - 1)] = xe
    f = lambda x, u, _L=L, _N=N: next_state(_L, _N, x, u)
    return n, m, N, M, L, xe, ue, f


def make_system_from_L(
    L: List[int],
    n: int,
    m: int,
    xe: int,
    ue: int,
) -> Tuple[
    int,
    int,
    int,
    int,
    List[int],
    int,
    int,
    Callable[[int, int], int],
    Callable[[int, int], float],
    WeightedGraph,
]:
    """Build f, g, graph for a given logical matrix."""
    N, M = 2**n, 2**m
    f = lambda x, u, _L=L, _N=N: next_state(_L, _N, x, u)
    g = hamming_cost_factory(xe, ue, n)
    states = list(range(1, N + 1))
    inputs = list(range(1, M + 1))
    G = build_graph(f, g, states, inputs, xe=xe)
    return n, m, N, M, L, xe, ue, f, g, G


def random_bcn_with_beneficial_cycle(
    n: int,
    m: int = 2,
    lam: float = 0.5,
    seed: int = 0,
    decoy_offset: int = 1,
) -> Tuple[
    int,
    int,
    int,
    int,
    List[int],
    int,
    int,
    Callable[[int, int], int],
    Callable[[int, int], float],
    WeightedGraph,
    int,
]:
    """
    Random BCN plus a cheap decoy fixpoint to create beneficial-cycle structure.
    Returns decoy state index.
    """
    n, m, N, M, L, xe, ue, f = random_bcn(n, m, seed, ensure_fixpoint=True)
    decoy = min(decoy_offset + 1, max(1, N - 1))
    if decoy == xe:
        decoy = max(1, xe - 1)
    # force decoy self-loop under u=1
    L[(0) * N + (decoy - 1)] = decoy
    eps = 0.01

    def adv_g(x: int, u: int) -> float:
        if x == xe and u == ue:
            return 0.0
        if x == decoy:
            return eps
        return hamming_cost_factory(xe, ue, n)(x, u)

    states = list(range(1, N + 1))
    inputs = list(range(1, M + 1))
    G = build_graph(f, adv_g, states, inputs, xe=xe)
    return n, m, N, M, L, xe, ue, f, adv_g, G, decoy


def list_fixpoints(L: List[int], N: int, M: int) -> Dict[int, List[int]]:
    return control_fixpoints(L, N, M)
