"""Loader and cost factories for the Example 2 real BCN datasets (plan §5.3).

Reads the annotated network files written by gen_networks.py (plus the two
Gao-repo files for ara operon / lambda switch), and assembles weighted STGs
exclusively through the methods.py public API.  No numeric decisions are made
here: cost factories return plain ints (or whatever eps object the caller
passes), and every scalar is injected by methods.build_stg via the caller's
NumericContext.
"""
from __future__ import annotations

import os
import sys
from collections import deque
from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple

sys.path.insert(0, os.path.dirname(__file__))

from ara_io import inverse_map  # bit convention (single source)
from methods import STG, NumericContext, build_stg

SIM_DIR = os.path.dirname(os.path.abspath(__file__))
NETWORKS_DIR = os.path.join(SIM_DIR, "networks")
REPO_ROOT = os.path.normpath(os.path.join(SIM_DIR, "..", ".."))
GAO_NET_DIR = os.path.join(
    REPO_ROOT, "literature", "04_optimal_cost_stabilization",
    "Gao2020_bcn_opt_dc_repo", "code", "networks",
)

# The two Gao-repo files carry no annotation header; their metadata is pinned
# here (node orders established by ara_rules.verify and gen_networks A-lam-diff).
_EXTERNAL: Dict[str, dict] = {
    "ara_operon": {
        "path": os.path.join(GAO_NET_DIR, "ara_operon.txt"),
        "nodes": ["A", "Am", "Ara+", "C", "E", "L", "MS", "MT", "T"],
        "inputs": ["Ae", "Aem", "Ara-", "Ge"],
        "origin": "Jenkins & Macauley 2017 (jenkins2017araoperon)",
        "bcn_usage": "Gao, Xiang & Lee 2020 ICCA (gaoshuhua2020icca)",
    },
    "lambda_switch": {
        "path": os.path.join(GAO_NET_DIR, "lambda_switch.txt"),
        "nodes": ["N", "cI", "cII", "cIII", "cro"],
        "inputs": ["u"],
        "origin": "Thieffry & Thomas 1995 (thieffry1995dynamical)",
        "bcn_usage": ("Laschov & Margaliot 2013 SICON (margaliot2013siam2869); "
                      "matrix: Zhu et al. 2018 SICON Example 2 via Gao repo"),
    },
}


@dataclass(frozen=True)
class NetworkSpec:
    name: str
    n: int
    m: int
    N: int
    M: int
    L: Tuple[int, ...]
    node_names: Tuple[str, ...]
    input_names: Tuple[str, ...]
    valid_inputs: Optional[Tuple[int, ...]]        # None = all M inputs
    provenance: Dict[str, str] = field(default_factory=dict)

    # -- raw dynamics -------------------------------------------------------
    def next_state(self, x: int, u: int) -> int:
        return self.L[(u - 1) * self.N + (x - 1)]

    def inputs(self) -> List[int]:
        return list(self.valid_inputs) if self.valid_inputs else list(range(1, self.M + 1))


def bits_of(spec: NetworkSpec, i: int) -> Tuple[int, ...]:
    return tuple(inverse_map(i, spec.n))


def input_bits_of(spec: NetworkSpec, k: int) -> Tuple[int, ...]:
    return tuple(inverse_map(k, spec.m))


def state_index(spec: NetworkSpec, bits: Sequence[int]) -> int:
    i = 1
    for k, b in enumerate(bits):
        if not b:
            i += 2 ** (spec.n - 1 - k)
    return i


def input_index(spec: NetworkSpec, bits: Sequence[int]) -> int:
    i = 1
    for k, b in enumerate(bits):
        if not b:
            i += 2 ** (spec.m - 1 - k)
    return i


def load_network(name_or_path: str) -> NetworkSpec:
    """Load by short name (networks dir / pinned external files) or by path."""
    meta: Dict[str, str] = {}
    nodes: List[str] = []
    inputs: List[str] = []
    valid: Optional[List[int]] = None
    if name_or_path in _EXTERNAL:
        ext = _EXTERNAL[name_or_path]
        path = ext["path"]
        nodes, inputs = list(ext["nodes"]), list(ext["inputs"])
        meta = {"origin": ext["origin"], "bcn_usage": ext["bcn_usage"]}
        name = name_or_path
    else:
        path = (name_or_path if os.path.sep in name_or_path
                else os.path.join(NETWORKS_DIR, f"{name_or_path}.txt"))
        name = os.path.splitext(os.path.basename(path))[0]
    lines: List[str] = []
    with open(path) as fh:
        for raw in fh:
            s = raw.strip()
            if not s:
                continue
            if s.startswith("#"):
                body = s.lstrip("# ").strip()
                if ":" in body:
                    key, val = body.split(":", 1)
                    key, val = key.strip(), val.strip()
                    if key == "nodes":
                        nodes = val.split()
                    elif key == "inputs":
                        inputs = val.split()
                    elif key == "valid_inputs":
                        valid = [int(t) for t in val.split()]
                    elif key == "name":
                        name = val
                    else:
                        meta[key] = val
                continue
            lines.append(s)
    n, m = int(lines[0]), int(lines[1])
    N, M = 2 ** n, 2 ** m
    body_lines = lines[2:]
    if body_lines and len(body_lines[0].split()) == 1 and N * M > 1:
        # sbcn_mmc 4-line format: a sub-network count precedes L
        assert body_lines[0] == "1", "multi-subnetwork files are not supported"
        body_lines = body_lines[1:]
    # collect exactly N*M integers, ignoring any trailing sections
    toks: List[int] = []
    stop = False
    for bl in body_lines:
        for t in bl.split():
            if not (t.isdigit() or (t[:1] == "-" and t[1:].isdigit())):
                stop = True
                break
            toks.append(int(t))
            if len(toks) == N * M:
                stop = True
                break
        if stop:
            break
    L = tuple(toks)
    assert len(L) == N * M, (path, len(L), N * M)
    if not nodes:
        nodes = [f"x{i}" for i in range(1, n + 1)]
    if not inputs:
        inputs = [f"u{i}" for i in range(1, m + 1)]
    return NetworkSpec(
        name=name, n=n, m=m, N=N, M=M, L=L,
        node_names=tuple(nodes), input_names=tuple(inputs),
        valid_inputs=tuple(valid) if valid else None, provenance=meta,
    )


# ---------------------------------------------------------------------------
# structure probes (raw L, no STG needed)
# ---------------------------------------------------------------------------

def control_fixpoints(spec: NetworkSpec) -> Dict[int, List[int]]:
    """States fixed under at least one admissible constant input."""
    out: Dict[int, List[int]] = {}
    for x in range(1, spec.N + 1):
        us = [u for u in spec.inputs() if spec.next_state(x, u) == x]
        if us:
            out[x] = us
    return out


def reach_depth(spec: NetworkSpec, xe: int,
                Cu: Optional[Callable[[int], Iterable[int]]] = None) -> Tuple[Set[int], int]:
    """Reverse-reachable set of xe and its BFS depth D_reach (raw L walk)."""
    rev: Dict[int, List[int]] = {}
    for x in range(1, spec.N + 1):
        for u in (Cu(x) if Cu is not None else spec.inputs()):
            rev.setdefault(spec.next_state(x, u), []).append(x)
    dist = {xe: 0}
    dq = deque([xe])
    while dq:
        v = dq.popleft()
        for p in rev.get(v, []):
            if p not in dist:
                dist[p] = dist[v] + 1
                dq.append(p)
    return set(dist), max(dist.values())


# ---------------------------------------------------------------------------
# cost factories (pure callbacks; numeric injection happens in build_stg)
# ---------------------------------------------------------------------------

def hamming_cost(spec: NetworkSpec, xe: int, ue: int) -> Callable[[int, int], object]:
    """g(x,u) = d_H(x, x_e) + 1[u != u_e]  (eq. toy-cost), precomputed bits."""
    target = bits_of(spec, xe)
    dh = [0] * (spec.N + 1)
    for x in range(1, spec.N + 1):
        b = inverse_map(x, spec.n)
        dh[x] = sum(1 for a, c in zip(b, target) if a != c)

    def g(x: int, u: int) -> int:
        return dh[x] + (0 if u == ue else 1)

    return g


def decoy_cost(base: Callable[[int, int], object],
               decoy_pairs: Set[Tuple[int, int]], eps) -> Callable[[int, int], object]:
    """g_eps: price every (x,u) in decoy_pairs at eps, base elsewhere."""

    def g(x: int, u: int):
        return eps if (x, u) in decoy_pairs else base(x, u)

    return g


# ---------------------------------------------------------------------------
# STG assembly (the only seam to methods.py)
# ---------------------------------------------------------------------------

def make_stg(spec: NetworkSpec, cost: Callable[[int, int], object],
             ctx: NumericContext,
             Cu: Optional[Callable[[int], Iterable[int]]] = None,
             states: Optional[Iterable[int]] = None) -> STG:
    """Weighted STG over `states` (default: all N) with admissible inputs.

    Cu defaults to the spec's valid_inputs (input constraints of the model);
    `states` allows building only on a reverse-reachable subset for large N
    (plan §7.4 first fallback)."""
    if Cu is None:
        allowed = spec.inputs()
        Cu = lambda _x, _a=tuple(allowed): _a
    return build_stg(
        spec.next_state, cost,
        states if states is not None else range(1, spec.N + 1),
        range(1, spec.M + 1), ctx, Cu=Cu,
    )
