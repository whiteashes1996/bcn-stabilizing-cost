"""Load ara operon BCN network and STP helpers (from scripts/verify_W_ara.py)."""
from __future__ import annotations

import os
from collections import deque
from typing import Callable, Dict, List, Set, Tuple

WORKSPACE_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".."))
ARA_PATH = os.path.join(
    WORKSPACE_ROOT,
    "literature",
    "04_optimal_cost_stabilization",
    "Gao2020_bcn_opt_dc_repo",
    "code",
    "networks",
    "ara_operon.txt",
)
LAMBDA_SWITCH_PATH = os.path.join(
    WORKSPACE_ROOT,
    "literature",
    "04_optimal_cost_stabilization",
    "Gao2020_bcn_opt_dc_repo",
    "code",
    "networks",
    "lambda_switch.txt",
)


def read_network(path: str | None = None) -> Tuple[int, int, int, int, List[int]]:
    path = path or ARA_PATH
    with open(path) as f:
        n = int(f.readline().strip())
        m = int(f.readline().strip())
        nums = f.readline().strip().split()
        L = [int(x) for x in nums]
    N, M = 2**n, 2**m
    assert len(L) == M * N, (len(L), M * N)
    return n, m, N, M, L


def next_state(L: List[int], N: int, x: int, u: int) -> int:
    """1-indexed state x, input u -> 1-indexed successor."""
    return L[(u - 1) * N + (x - 1)]


def inverse_map(i: int, n: int) -> List[int]:
    """Canonical index i -> 0/1 logic vector (Gao convention)."""
    r: List[int] = []
    while n > 0:
        if i % 2 == 0:
            r.append(0)
            i = i // 2
        else:
            r.append(1)
            i = (i + 1) // 2
        n -= 1
    r.reverse()
    return r


def build_ara_dynamics(
    path: str | None = None,
) -> Tuple[int, int, int, int, List[int], Callable[[int, int], int]]:
    n, m, N, M, L = read_network(path)
    f = lambda x, u: next_state(L, N, x, u)
    return n, m, N, M, L, f


def hamming_cost_factory(
    xe: int, ue: int, n: int, w_state: float = 1.0, w_ctrl: float = 1.0
) -> Callable[[int, int], float]:
    bx_e = inverse_map(xe, n)

    def g(x: int, u: int) -> float:
        bx = inverse_map(x, n)
        ds = sum(1 for a, b in zip(bx, bx_e) if a != b)
        dc = 0.0 if u == ue else w_ctrl
        return w_state * ds + dc

    return g


def control_fixpoints(
    L: List[int], N: int, M: int
) -> Dict[int, List[int]]:
    fps: Dict[int, List[int]] = {}
    for x in range(1, N + 1):
        us = [u for u in range(1, M + 1) if next_state(L, N, x, u) == x]
        if us:
            fps[x] = us
    return fps


def reverse_reachable_states(
    rev: List[Set[int]], xe: int
) -> Tuple[Set[int], int]:
    dist = {xe: 0}
    dq = deque([xe])
    while dq:
        v = dq.popleft()
        for p in rev[v]:
            if p not in dist:
                dist[p] = dist[v] + 1
                dq.append(p)
    return set(dist), max(dist.values())
