"""E14: the four-node BCN of the new Example 1 (N=16, M=2, lambda=1/2).

Implements the design plan notes/rewrite_experiments/step1_example1_design_plan.md
(modules A-F, acceptance items A1-A16), entirely in exact rational arithmetic.

The instance exercises every stage of Algorithm master at once:
  - two unstabilizable states (15, 16) make reverse reachability nontrivial;
  - a benign 3-state SCC B = {6,7,10} needs two strict label-correcting sweeps;
  - exactly one beneficial-cycle SCC H = {2,3,5} defeats VI/PI/Floyd (improper
    at 6 of 14 stabilizable states, including upstream states 4, 13, 14);
  - the condensation DAG has 10 components with a nontrivial folding order
    (B's exit weight e(7) needs the SolveCore output J(3) as boundary data).

Label-correcting detection deliberately goes through core_solver.local_lc_paper,
i.e. the synchronous (Jacobi) sweeps of the paper's operator T; no Gauss-Seidel.
"""
from __future__ import annotations

import math
import os
import sys
from collections import defaultdict
from fractions import Fraction
from typing import Dict, List, Optional, Set, Tuple

sys.path.insert(0, os.path.dirname(__file__))

from core_solver import (  # noqa: E402
    FR_INF,
    INF,
    WeightedGraph,
    augment_component,
    condense,
    enumerate_beneficial_cycles,
    local_lc_paper,
    master,
    policy_iteration_exact,
    reverse_reachable,
    simulate_closed_loop,
    solve_core_heldkarp,
    sweep_T,
    terminal_to_augmented,
)

RESULTS = os.path.join(os.path.dirname(__file__), "results")

# ---------------------------------------------------------------------------
# Module A: instance definition
# ---------------------------------------------------------------------------

N_NODES = 4
N = 16
M = 2
XE = 1  # delta_16^1 = (1,1,1,1)
UE = 1
LAM = Fraction(1, 2)

# Column-major network transition matrix L = delta_16[...]: entry (u-1)*N+(x-1).
# The dominated u=2 arcs of states 8, 9, 11 were retargeted from the design
# plan's first draft (8->8, 9->1, 11->11) to (8->1, 9->9, 11->1): value-neutral
# by exhaustive check, and the Quine-McCluskey form drops from 30 to 26
# conjuncts (design plan, sections 5.1 and 7).
L = [
    1, 3, 5, 2, 2, 7, 10, 1, 1, 6, 9, 10, 5, 13, 16, 15,   # u = delta_2^1
    1, 8, 5, 15, 2, 8, 3, 1, 9, 9, 1, 6, 16, 12, 16, 15,   # u = delta_2^2
]

H_SCC = {2, 3, 5}      # beneficial-cycle component
B_SCC = {6, 7, 10}     # benign component

# Hand-derived exact tables (design plan section 1.4).
EXPECTED_J: Dict[int, Optional[Fraction]] = {
    1: Fraction(0), 2: Fraction(7, 2), 3: Fraction(19, 8), 4: Fraction(15, 4),
    5: Fraction(11, 4), 6: Fraction(31, 8), 7: Fraction(15, 4), 8: Fraction(3),
    9: Fraction(1), 10: Fraction(7, 2), 11: Fraction(5, 2), 12: Fraction(19, 4),
    13: Fraction(27, 8), 14: Fraction(75, 16), 15: None, 16: None,
}
EXPECTED_V: Dict[int, Fraction] = dict(EXPECTED_J)  # type: ignore[arg-type]
EXPECTED_V.update({
    2: Fraction(2), 3: Fraction(2), 5: Fraction(2),
    4: Fraction(3), 13: Fraction(3), 14: Fraction(9, 2),
})
del EXPECTED_V[15], EXPECTED_V[16]

EXPECTED_GAP = {2, 3, 4, 5, 13, 14}
EXPECTED_IMPROPER = {2, 3, 4, 5, 13, 14}
EXPECTED_E_H = {2: Fraction(7, 2)}                    # e(3)=e(5)=inf
EXPECTED_E_B = {6: Fraction(9, 2), 7: Fraction(67, 16), 10: Fraction(7, 2)}
EXPECTED_THR_H = {2: Fraction(49, 16), 3: Fraction(133, 64), 5: Fraction(77, 32)}
G_LOOP_H = Fraction(7, 4)                             # 1 + lam + lam^2 at lam=1/2
LAM_STAR = (1 + math.sqrt(13)) / 6                    # root of 3*lam^2 - lam - 1


def bits(i: int) -> Tuple[int, int, int, int]:
    """delta_16^i -> logical vector, with delta_16^1 = (1,1,1,1)."""
    v = N - i
    return ((v >> 3) & 1, (v >> 2) & 1, (v >> 1) & 1, v & 1)


def hamming_to_target(i: int) -> int:
    return sum(1 for a, b in zip(bits(i), bits(XE)) if a != b)


def f(x: int, u: int) -> int:
    return L[(u - 1) * N + (x - 1)]


def g(x: int, u: int) -> Fraction:
    return Fraction(hamming_to_target(x) + (0 if u == UE else 1))


def build() -> WeightedGraph:
    """Weighted STG with exact Fraction arc costs."""
    G = WeightedGraph(states=set(), xe=XE)
    for x in range(1, N + 1):
        for u in range(1, M + 1):
            G.add_arc(x, u, f(x, u), g(x, u))
    return G


def _isinf(v) -> bool:
    return v == INF or (isinstance(v, Fraction) and v >= FR_INF)


# --- Quine-McCluskey over (u, x1..x4); same routine as exp10 (copied, 5 vars) --


def _combine(a: Tuple[int, int], b: Tuple[int, int]) -> Optional[Tuple[int, int]]:
    if a[1] != b[1]:
        return None
    diff = a[0] ^ b[0]
    if diff and diff & (diff - 1) == 0:
        return (a[0] & ~diff, a[1] | diff)
    return None


def _covers(term: Tuple[int, int], m: int) -> bool:
    return term[0] == m & ~term[1]


def quine_mccluskey(minterms: List[int], nvars: int) -> List[Tuple[int, int]]:
    """Prime implicants covering the minterms, reduced by a greedy cover."""
    if not minterms:
        return []
    if len(minterms) == 1 << nvars:
        return [(0, (1 << nvars) - 1)]
    terms = {(m, 0) for m in minterms}
    primes: Set[Tuple[int, int]] = set()
    while terms:
        nxt: Set[Tuple[int, int]] = set()
        used: Set[Tuple[int, int]] = set()
        tl = sorted(terms)
        for i in range(len(tl)):
            for j in range(i + 1, len(tl)):
                c = _combine(tl[i], tl[j])
                if c is not None:
                    nxt.add(c)
                    used.add(tl[i])
                    used.add(tl[j])
        primes |= terms - used
        terms = nxt

    uncovered = set(minterms)
    chosen: List[Tuple[int, int]] = []
    while uncovered:
        best = max(primes, key=lambda p: (sum(1 for m in uncovered if _covers(p, m)), -p[0]))
        gain = [m for m in uncovered if _covers(best, m)]
        if not gain:
            break
        chosen.append(best)
        uncovered -= set(gain)
        primes.discard(best)
    return chosen


def render_dnf(terms: List[Tuple[int, int]], names: List[str]) -> str:
    if not terms:
        return "0"
    nvars = len(names)
    if terms == [(0, (1 << nvars) - 1)]:
        return "1"
    parts = []
    for bitsval, mask in terms:
        lits = []
        for k, nm in enumerate(names):
            bit = 1 << (nvars - 1 - k)
            if mask & bit:
                continue
            lits.append(nm if bitsval & bit else f"~{nm}")
        parts.append(" & ".join(lits) if lits else "1")
    return " | ".join(f"({p})" if " & " in p else p for p in parts)


def logical_form() -> Tuple[List[str], List[int]]:
    """x_i(t+1) as Boolean functions of (u, x1..x4); returns (DNFs, term counts)."""
    names = ["u", "x1", "x2", "x3", "x4"]
    minterms: List[List[int]] = [[], [], [], []]
    for u in (1, 2):
        U = 1 if u == UE else 0
        for x in range(1, N + 1):
            a, b, c, d = bits(x)
            m = (U << 4) | (a << 3) | (b << 2) | (c << 1) | d
            nxt = bits(f(x, u))
            for k in range(4):
                if nxt[k]:
                    minterms[k].append(m)
    dnfs, counts = [], []
    for k in range(4):
        terms = quine_mccluskey(sorted(minterms[k]), 5)
        dnfs.append(f"x{k + 1}(t+1) = " + render_dnf(terms, names))
        counts.append(len(terms))
    return dnfs, counts


# ---------------------------------------------------------------------------
# Module B: exhaustive simple-path oracle (full graph, all 16 states)
# ---------------------------------------------------------------------------


def oracle_simple_paths(
    G: WeightedGraph, xe: int, lam: Fraction
) -> Dict[int, Optional[Fraction]]:
    """Exact min discounted cost over all simple paths x -> xe; None if none."""

    def dfs(x: int, visited: Set[int]) -> Optional[Fraction]:
        if x == xe:
            return Fraction(0)
        best: Optional[Fraction] = None
        for u, y, w in G.by_state.get(x, []):
            if y in visited:
                continue
            sub = dfs(y, visited | {y})
            if sub is None:
                continue
            cand = Fraction(w) + lam * sub
            if best is None or cand < best:
                best = cand
        return best

    return {x: dfs(x, {x}) for x in sorted(G.states)}


# ---------------------------------------------------------------------------
# Module C: literature baselines on the restricted graph (exact)
# ---------------------------------------------------------------------------


def restricted_adjacency(
    G: WeightedGraph, R: Set[int]
) -> Dict[int, List[Tuple[int, int, Fraction]]]:
    return {
        x: [(u, y, Fraction(w)) for u, y, w in G.by_state.get(x, []) if y in R]
        for x in R
    }


def baseline_vi(
    adjR: Dict[int, List[Tuple[int, int, Fraction]]],
    xe: int,
    lam: Fraction,
    sweeps: int,
) -> List[Dict[int, Optional[Fraction]]]:
    """Synchronous Bellman iteration from V_0 = +inf (None), V_0(xe) = 0.

    Returns the whole sequence [V_0, ..., V_sweeps]; each V_k is the minimum
    discounted cost over walks of at most k arcs that end at xe, hence the
    sequence is pointwise non-increasing and lower-bounded by V_lambda*.
    """
    V: Dict[int, Optional[Fraction]] = {x: None for x in adjR}
    V[xe] = Fraction(0)
    seq = [dict(V)]
    for _ in range(sweeps):
        nxt: Dict[int, Optional[Fraction]] = {}
        for x in adjR:
            if x == xe:
                nxt[x] = Fraction(0)
                continue
            best: Optional[Fraction] = None
            for u, y, w in adjR[x]:
                vy = V[y]
                if vy is None:
                    continue
                cand = w + lam * vy
                if best is None or cand < best:
                    best = cand
            nxt[x] = best
        V = nxt
        seq.append(dict(V))
    return seq


def baseline_pi(
    G: WeightedGraph, R: Set[int], xe: int, lam: Fraction
) -> Tuple[Dict[int, Fraction], Dict[int, Tuple[int, int]]]:
    """Exact policy iteration for V_lambda* on the restricted graph.

    Folds R\\{xe} with xe absorbing and calls core_solver.policy_iteration_exact
    (finite exact steps, no tolerance), then maps the policy back to (u, y).
    """
    S = {x for x in R if x != xe}
    internal: Dict[int, List[Tuple[int, int, Fraction]]] = defaultdict(list)
    e: Dict[int, Fraction] = {}
    for x in S:
        for u, y, w in G.by_state.get(x, []):
            if y not in R:
                continue
            if y == xe:
                w = Fraction(w)
                if x not in e or w < e[x]:
                    e[x] = w
            else:
                internal[x].append((u, y, Fraction(w)))
    e_full = {x: e.get(x, INF) for x in S}
    V, pol, _steps = policy_iteration_exact(S, dict(internal), e_full, lam)
    V_out: Dict[int, Fraction] = {xe: Fraction(0)}
    V_out.update({x: Fraction(v) for x, v in V.items()})
    mu: Dict[int, Tuple[int, int]] = {xe: (UE, xe)}
    for x in S:
        act = pol.get(x)
        if act is None:
            continue
        if act[0] == "exit":
            best_u = min(
                (u for u, y, _w in G.by_state.get(x, []) if y == xe),
                key=lambda u: g(x, u),
            )
            mu[x] = (best_u, xe)
        else:
            mu[x] = (act[1], act[2])
    return V_out, mu


def baseline_floyd(
    adjR: Dict[int, List[Tuple[int, int, Fraction]]],
    xe: int,
    lam: Fraction,
    K: int,
) -> List[Dict[int, Optional[Fraction]]]:
    """Cheng/Zhu-type discounted (min,+) matrix powers on the restricted graph.

    W1[x][y] = min one-step cost x -> y; W_{k+1} = W1 (+) (W1 (x)_lam W_k),
    so W_k[x][y] is the min discounted cost of a walk of 1..k arcs. Returns the
    xe-column sequence [V_1, ..., V_K] with V_k(x) = W_k[x][xe], V_k(xe) = 0.
    """
    states = sorted(adjR)
    W1: Dict[int, Dict[int, Fraction]] = {x: {} for x in states}
    for x in states:
        for _u, y, w in adjR[x]:
            if y not in W1[x] or w < W1[x][y]:
                W1[x][y] = w

    def column(W: Dict[int, Dict[int, Fraction]]) -> Dict[int, Optional[Fraction]]:
        col: Dict[int, Optional[Fraction]] = {x: W[x].get(xe) for x in states}
        col[xe] = Fraction(0)
        return col

    Wk = {x: dict(W1[x]) for x in states}
    cols = [column(Wk)]
    for _ in range(K - 1):
        nxt: Dict[int, Dict[int, Fraction]] = {}
        for x in states:
            row: Dict[int, Fraction] = dict(W1[x])
            for z, wxz in W1[x].items():
                for y, wzy in Wk[z].items():
                    cand = wxz + lam * wzy
                    if y not in row or cand < row[y]:
                        row[y] = cand
            nxt[x] = row
        Wk = nxt
        cols.append(column(Wk))
    return cols


def greedy_actions(
    V: Dict[int, Optional[Fraction]],
    adjR: Dict[int, List[Tuple[int, int, Fraction]]],
    xe: int,
    lam: Fraction,
) -> Dict[int, frozenset]:
    """All exact Bellman argmin inputs at each state (tol = 0)."""
    acts: Dict[int, frozenset] = {}
    for x in adjR:
        if x == xe:
            continue
        cands = []
        for u, y, w in adjR[x]:
            vy = V.get(y)
            if vy is None:
                continue
            cands.append((w + lam * vy, u))
        if not cands:
            acts[x] = frozenset()
            continue
        best = min(v for v, _u in cands)
        acts[x] = frozenset(u for v, u in cands if v == best)
    return acts


def greedy_mu(
    V: Dict[int, Optional[Fraction]],
    adjR: Dict[int, List[Tuple[int, int, Fraction]]],
    xe: int,
    lam: Fraction,
) -> Dict[int, Tuple[int, int]]:
    """One greedy policy (smallest minimizing u, matching the tie-break of VI)."""
    mu: Dict[int, Tuple[int, int]] = {xe: (UE, xe)}
    for x, us in greedy_actions(V, adjR, xe, lam).items():
        if us:
            u = min(us)
            mu[x] = (u, f(x, u))
    return mu


def improper_states(mu: Dict[int, Tuple[int, int]], states: Set[int]) -> Set[int]:
    bad = set()
    for x in states:
        if x == XE:
            continue
        proper, _steps, _traj = simulate_closed_loop(f, mu, x, XE, max_steps=4 * N)
        if not proper:
            bad.add(x)
    return bad


def closed_loop_cost(
    mu: Dict[int, Tuple[int, int]], x: int, lam: Fraction
) -> Optional[Fraction]:
    """Exact discounted cost of the closed loop from x; None if it never reaches xe."""
    total, disc, cur, seen = Fraction(0), Fraction(1), x, set()
    while cur != XE:
        if cur in seen or cur not in mu:
            return None
        seen.add(cur)
        u, _y = mu[cur]
        total += disc * g(cur, u)
        disc *= lam
        cur = f(cur, u)
    return total


# ---------------------------------------------------------------------------
# Module D: stagewise pipeline (manual mirror of Algorithm master, exact)
# ---------------------------------------------------------------------------


def sweep_sequence(S, internal, e, lam, cap: int):
    """[V_0=e, V_1, ..., V_cap] under the synchronous operator T of eq. local-lc."""
    seq = [{x: e.get(x, INF) for x in S}]
    for _ in range(cap):
        nxt, _pol = sweep_T(S, internal, e, lam, seq[-1])
        seq.append(nxt)
    return seq


def lt_somewhere(A, B) -> bool:
    """Exact test whether A < B at some state (None/inf handled)."""
    for x, av in A.items():
        bv = B.get(x, INF)
        if _isinf(bv):
            if not _isinf(av):
                return True
        elif not _isinf(av) and av < bv:
            return True
    return False


def manual_master(G: WeightedGraph, lam: Fraction):
    """Reverse topo pass mirroring Algorithm master, with per-component records.

    Detection strictly via core_solver.local_lc_paper (synchronous sweeps,
    tol=0); beneficial components re-solved by solve_core_heldkarp.
    """
    R = reverse_reachable(G, XE)
    scc = condense(G, R, XE)
    J: Dict[int, Fraction] = {XE: Fraction(0)}
    records = []
    for cid in scc.reverse_topo:
        S = set(scc.components[cid])
        if XE in S:
            if len(S) == 1:
                records.append(dict(cid=cid, S=S, terminal=True))
                continue
            S, internal, e = terminal_to_augmented(S, _component_internal(G, S), XE, lam)
        else:
            internal, e = augment_component(G, S, J, lam)
        Jc, _pol, has_cycle, sweeps = local_lc_paper(S, internal, e, lam, tol=0)
        seq = sweep_sequence(S, internal, e, lam, cap=len(S))
        strict = sum(1 for a, b in zip(seq[1:], seq[:-1]) if lt_somewhere(a, b))
        pol_core = None
        if has_cycle:
            Jc, pol_core, _cnt = solve_core_heldkarp(S, internal, e, lam)
        J.update({x: Fraction(v) for x, v in Jc.items() if not _isinf(v)})
        records.append(dict(
            cid=cid, S=S, terminal=False, internal=internal, e=e,
            has_cycle=has_cycle, sweeps=sweeps, strict=strict, seq=seq,
            J={x: (None if _isinf(v) else Fraction(v)) for x, v in Jc.items()},
            pol_core=pol_core,
        ))
    return R, scc, J, records


def _component_internal(G: WeightedGraph, S: Set[int]):
    internal: Dict[int, List[Tuple[int, int, Fraction]]] = defaultdict(list)
    for x in S:
        for u, y, w in G.by_state.get(x, []):
            if y in S:
                internal[x].append((u, y, Fraction(w)))
    return dict(internal)


# ---------------------------------------------------------------------------
# Module E: exact lambda* threshold analysis
# ---------------------------------------------------------------------------


def fold_H_at(lam: Fraction):
    """Folded H = {2,3,5} at discount lam: J(8) = 3 for every lam."""
    internal = {
        2: [(1, 3, Fraction(1))],
        3: [(1, 5, Fraction(1)), (2, 5, Fraction(2))],
        5: [(1, 2, Fraction(1)), (2, 2, Fraction(2))],
    }
    e = {2: Fraction(2) + lam * Fraction(3), 3: INF, 5: INF}
    return internal, e


def beneficial_at(lam: Fraction) -> bool:
    """Exact beneficial-cycle predicate on H at rational lam.

    Cross-checks three exact routes and asserts they agree:
    (i) local_lc_paper (synchronous sweeps, tol=0) on the folded component;
    (ii) the criterion g_loop < (1-lam^3) * J_lam(2) with closed-form J;
    (iii) the cycle oracle enumerate_beneficial_cycles on SolveCore's J.
    """
    internal, e = fold_H_at(lam)
    _J, _pol, has_cycle, _sw = local_lc_paper(H_SCC, internal, e, lam, tol=0)
    J2 = Fraction(2) + 3 * lam
    crit = (1 + lam + lam**2) < (1 - lam**3) * J2
    J_core, _p, _c = solve_core_heldkarp(H_SCC, internal, e, lam)
    J_q = {x: Fraction(v) for x, v in J_core.items()}
    cyc = enumerate_beneficial_cycles(H_SCC, internal, e, lam, J_q)
    assert has_cycle == crit == bool(cyc), (lam, has_cycle, crit, len(cyc))
    return has_cycle


def lambda_star_bisect(
    lo: Fraction = Fraction(1, 2), hi: Fraction = Fraction(7, 8), iters: int = 60
) -> Tuple[Fraction, Fraction]:
    """Rational bracket around lambda*; endpoints stay dyadic."""
    assert beneficial_at(lo) and not beneficial_at(hi)
    for _ in range(iters):
        mid = (lo + hi) / 2
        if beneficial_at(mid):
            lo = mid
        else:
            hi = mid
    return lo, hi


def poly_sign(lam: Fraction) -> Fraction:
    """p(lam) = 3 lam^2 - lam - 1; negative exactly on the beneficial side."""
    return 3 * lam**2 - lam - 1


# ---------------------------------------------------------------------------
# Module F: outputs (table rows, figures, results file)
# ---------------------------------------------------------------------------


def fr_dec(v: Optional[Fraction]) -> str:
    """Exact decimal string of a dyadic rational (None -> infinity symbol)."""
    if v is None or _isinf(v):
        return "$\\infty$"
    v = Fraction(v)
    if v.denominator == 1:
        return str(v.numerator)
    a = v.denominator.bit_length() - 1
    assert v.denominator == 1 << a, v
    scaled = v.numerator * 5**a
    s = str(abs(scaled)).rjust(a + 1, "0")
    return ("-" if scaled < 0 else "") + s[:-a] + "." + s[-a:]


COMP_LABEL = {
    frozenset({1}): "$\\{x_e\\}$",
    frozenset(H_SCC): "$S^{\\mathrm{bc}}$",
    frozenset(B_SCC): "$S^{\\mathrm{lc}}$",
}


def make_table_rows(
    V: Dict[int, Fraction],
    J: Dict[int, Fraction],
    mu_vi, mu_pi, mu_master,
    scc,
    n_improper: Dict[str, int],
) -> str:
    lines = []
    for x in range(1, N + 1):
        b = "".join(str(t) for t in bits(x))
        if x in (15, 16):
            # row template wraps each field in $...$; bare \infty avoids $$\infty$$
            comp, vs, js, kvi, kpi, kmm = "--", "--", "\\infty", "--", "--", "--"
        else:
            comp = COMP_LABEL.get(
                frozenset(scc.components[scc.comp_of[x]]), "sgl"
            )
            vs, js = fr_dec(V[x]), fr_dec(J[x])
            kvi = str(mu_vi[x][0]) if x in mu_vi else "--"
            kpi = str(mu_pi[x][0]) if x in mu_pi else "--"
            kmm = str(mu_master[x][0]) if x in mu_master else "--"
        lines.append(
            f"$\\delta_{{16}}^{{{x}}}$ & ${b}$ & ${hamming_to_target(x)}$ & {comp}"
            f" & ${vs}$ & ${js}$ & ${kvi}$ & ${kpi}$ & ${kmm}$ \\\\"
        )
    lines.append("\\midrule")
    lines.append(
        "\\multicolumn{6}{@{}l}{improper states} & "
        f"${n_improper['vi']}$ & ${n_improper['pi']}$ & ${n_improper['master']}$ \\\\"
    )
    return "\n".join(lines)


POS = {
    1: (5.6, 0.0),
    9: (4.2, 1.1),
    8: (4.2, -1.3),
    2: (2.6, -0.7),
    3: (1.5, -1.7),
    5: (2.9, -2.0),
    4: (0.6, -0.9),
    13: (1.0, -3.0),
    6: (1.6, 1.7),
    7: (0.4, 0.9),
    10: (2.7, 1.5),
    11: (4.2, 2.3),
    12: (-1.0, 1.6),
    14: (-2.0, 0.3),
    15: (-2.0, -2.2),
    16: (-0.9, -2.8),
}

TRAP = "#d1495b"
GOOD = "#1a7f37"
FAINT = "#c8ccd0"
DEAD = "#9aa0a6"


def _draw_panel(ax, G, mu, title, colour, trapped, highlight=None):
    import networkx as nx

    base = nx.DiGraph()
    base.add_nodes_from(POS)
    all_edges = {(x, y) for x, _u, y, _w in G.arcs if x != y}
    loop_edges = {(x, y) for x, (_u, y) in mu.items() if x != y}
    trap_edges = {(x, y) for x, y in loop_edges if x in trapped and y in trapped}
    key_edges = set(highlight or ()) & loop_edges
    plain_edges = loop_edges - trap_edges - key_edges
    base.add_edges_from(all_edges | loop_edges)

    node_fill, node_edge = [], []
    for nd in base.nodes():
        if nd == XE:
            node_fill.append("#eaf5ec")
            node_edge.append(GOOD)
        elif nd in trapped:
            node_fill.append("#fdeced")
            node_edge.append(TRAP)
        elif nd in (15, 16):
            node_fill.append("#f1f3f4")
            node_edge.append(DEAD)
        else:
            node_fill.append("white")
            node_edge.append("#444444")
    nx.draw_networkx_nodes(
        base, POS, ax=ax, node_size=760, node_color=node_fill,
        edgecolors=node_edge, linewidths=1.5,
    )
    nx.draw_networkx_labels(
        base, POS, ax=ax, font_size=9,
        labels={i: rf"$\delta_{{16}}^{{{i}}}$" for i in POS},
    )

    def edges(lst, col, width, alpha=1.0):
        if not lst:
            return
        nx.draw_networkx_edges(
            base, POS, ax=ax, edgelist=sorted(lst), edge_color=col, width=width,
            alpha=alpha, arrows=True, arrowsize=13, arrowstyle="-|>",
            node_size=760, min_source_margin=11, min_target_margin=11,
            connectionstyle="arc3,rad=0.12",
        )

    edges(all_edges - loop_edges, FAINT, 0.9)
    edges(plain_edges, "#5b6167", 1.5)
    edges(trap_edges, TRAP, 2.4)
    edges(key_edges, colour, 2.4)
    ax.set_title(title, fontsize=10.5)
    ax.set_xlim(-3.0, 6.6)
    ax.set_ylim(-3.6, 3.0)
    ax.axis("off")


def figure_stg(G, mu_vi, mu_master, trapped: Set[int]) -> str:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.2))
    _draw_panel(
        axes[0], G, mu_vi,
        r"(a) closed loop of the $V_\lambda^*$-induced policy: six states"
        r" never reach $x_e$",
        TRAP, trapped,
    )
    _draw_panel(
        axes[1], G, mu_master,
        r"(b) closed loop of $\mu^*$ from Algorithm 1: every state of"
        r" $\mathcal{V}_s$ reaches $x_e$",
        GOOD, set(), highlight={(2, 8)},
    )
    axes[0].text(4.5, -2.6, "beneficial cycle", fontsize=9.5, color=TRAP, ha="center")
    axes[0].text(-1.5, -3.35, "unstabilizable", fontsize=9.5, color=DEAD, ha="center")
    axes[1].annotate(
        "escape arc paid by $\\mu^*$", xy=(3.5, -1.05), xytext=(4.6, -2.6),
        fontsize=9.5, color=GOOD, ha="center",
        arrowprops=dict(arrowstyle="-", color=GOOD, lw=0.8),
    )
    fig.tight_layout(pad=0.3)
    out = os.path.abspath(os.path.join(RESULTS, "ex1_n4_stg.pdf"))
    fig.savefig(out, bbox_inches="tight", transparent=True)
    fig.savefig(out.replace(".pdf", ".png"), bbox_inches="tight", dpi=200)
    plt.close(fig)
    return out


def figure_threshold(G: WeightedGraph, adjR, R: Set[int]) -> str:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    lams, Vs, Js = [], [], []
    for i in range(1, 100):
        lam = Fraction(i, 100)
        _R, _scc, J, _rec = manual_master(G, lam)
        V, _mu = baseline_pi(G, R, XE, lam)
        lams.append(float(lam))
        Js.append(float(J[2]))
        Vs.append(float(V[2]))

    fig, ax = plt.subplots(figsize=(5.4, 3.2))
    ax.plot(lams, Js, color=GOOD, lw=2.0,
            label=r"$J_\lambda^*(\delta_{16}^{2})$  (proper, Algorithm 1)")
    ax.plot(lams, Vs, color=TRAP, lw=2.0, ls="--",
            label=r"$V_\lambda^*(\delta_{16}^{2})$  (value/policy iteration)")
    ax.fill_between(lams, Vs, Js, where=[j > v + 1e-12 for v, j in zip(Vs, Js)],
                    color=TRAP, alpha=0.10, lw=0)
    ax.axvline(LAM_STAR, color="#555555", lw=1.0, ls=":")
    ax.annotate(
        rf"$\lambda^\star=\frac{{1+\sqrt{{13}}}}{{6}}\approx{LAM_STAR:.4f}$",
        xy=(LAM_STAR, 1.0), xytext=(LAM_STAR - 0.05, 4.6),
        fontsize=9.5, ha="right", color="#333333",
    )
    ax.text(0.33, 2.45, "beneficial cycle:\nVI/PI/Floyd improper",
            fontsize=9, color=TRAP, ha="center")
    ax.set_xlabel(r"discount factor $\lambda$")
    ax.set_ylabel("discounted cost at $\\delta_{16}^{2}$")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 6.0)
    ax.legend(fontsize=9, loc="upper left", frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(pad=0.3)
    out = os.path.abspath(os.path.join(RESULTS, "ex1_n4_threshold.pdf"))
    fig.savefig(out, bbox_inches="tight", transparent=True)
    fig.savefig(out.replace(".pdf", ".png"), bbox_inches="tight", dpi=200)
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# Run: acceptance items A1-A16 (design plan section 3)
# ---------------------------------------------------------------------------


def run() -> dict:
    os.makedirs(RESULTS, exist_ok=True)
    passed: List[str] = []

    def check(tag: str, cond: bool, detail: str = "") -> None:
        assert cond, f"{tag} FAILED {detail}"
        passed.append(f"{tag} PASS  {detail}".rstrip())

    G = build()
    K_ITERS = 64

    # A1 target is a fixed point of the constant-input system, zero stage cost
    check("A01", f(XE, UE) == XE and g(XE, UE) == 0, "f(1,u_e)=1, g(1,u_e)=0")

    # A2 stabilizable set
    R = reverse_reachable(G, XE)
    check("A02", R == set(range(1, 15)), f"V_s = 1..14, unstabilizable = {sorted(set(range(1, N + 1)) - R)}")

    # A3 restriction prunes arcs of stabilizable states
    check("A03", f(4, 2) == 15 and f(13, 2) == 16, "arcs 4->15, 13->16 pruned by Step 1")

    # A4 condensation: 10 components, the two nontrivial SCCs, folding order
    scc = condense(G, R, XE)
    nontrivial = sorted((c for c in scc.components if len(c) > 1), key=min)
    check(
        "A04",
        len(scc.components) == 10
        and nontrivial == [H_SCC, B_SCC]
        and scc.comp_of[8] < scc.comp_of[2] < scc.comp_of[6],
        "10 components; nontrivial = {2,3,5},{6,7,10}; order {8} < H < B",
    )

    # manual pipeline (local_lc_paper detection, exact)
    R2, scc2, J, records = manual_master(G, LAM)
    rec_by_min = {min(r["S"]): r for r in records if not r.get("terminal")}
    recH, recB = rec_by_min[2], rec_by_min[6]

    # A5 exit weights
    eH, eB = recH["e"], recB["e"]
    check(
        "A05",
        eH[2] == EXPECTED_E_H[2] and _isinf(eH[3]) and _isinf(eH[5])
        and all(eB[x] == EXPECTED_E_B[x] for x in B_SCC),
        "e(2)=3.5, e(3)=e(5)=inf; e(6)=4.5, e(7)=4.1875, e(10)=3.5",
    )

    # A6 detection verdicts: exactly one beneficial component
    flags = {min(r["S"]): r["has_cycle"] for r in records if not r.get("terminal")}
    check(
        "A06",
        flags[2] is True and sum(1 for v in flags.values() if v) == 1,
        "hasCycle(H)=True, all other components False",
    )

    # A7 sweep counts (synchronous local_lc_paper): B needs 2 strict sweeps
    seqB = recB["seq"]
    check(
        "A07",
        recB["strict"] == 2 and recB["sweeps"] == 3
        and lt_somewhere(seqB[1], seqB[0]) and lt_somewhere(seqB[2], seqB[1])
        and not lt_somewhere(seqB[3], seqB[2])
        and recH["sweeps"] == 3 and recH["strict"] == 3,
        "B: V0!=V1!=V2=V3 (2 strict sweeps, 3 total = D+1); H never settles",
    )

    # A8 cycle oracle: beneficial cycles exactly in H, with the hand thresholds
    J_H = {x: Fraction(v) for x, v in recH["J"].items()}
    cycles_H = enumerate_beneficial_cycles(H_SCC, recH["internal"], eH, LAM, J_H)
    thr = {y: (1 - LAM**3) * J_H[y] for y in H_SCC}
    others_empty = all(
        not enumerate_beneficial_cycles(
            r["S"], r["internal"], r["e"], LAM,
            {x: (FR_INF if v is None else v) for x, v in r["J"].items()},
        )
        for r in records
        if not r.get("terminal") and min(r["S"]) != 2
    )
    check(
        "A08",
        bool(cycles_H)
        and all(sorted(set(c[0])) == [2, 3, 5] for c in cycles_H)
        and min(c[1] for c in cycles_H) == G_LOOP_H
        and thr == EXPECTED_THR_H
        and others_empty,
        f"H: {len(cycles_H)} beneficial cycles on {{2,3,5}}, min g_loop=7/4;"
        " thresholds 49/16,133/64,77/32; all other components clean",
    )

    # A9 J table: manual pipeline == hand table == DFS oracle == core master
    oracle = oracle_simple_paths(G, XE, LAM)
    J_master, mu_master = master(G, XE, LAM, ue=UE)
    ok9 = True
    for x in range(1, N + 1):
        exp = EXPECTED_J[x]
        if exp is None:
            ok9 &= oracle[x] is None and x not in J and x not in J_master
        else:
            ok9 &= (
                J[x] == exp and oracle[x] == exp
                and Fraction(J_master[x]) == exp
            )
    check("A09", ok9, "J table: pipeline == hand == DFS oracle == master, inf at 15,16")

    # A10 SolveCore vs in-component simple-path enumeration on H
    J_core, _pol, _cnt = solve_core_heldkarp(H_SCC, recH["internal"], eH, LAM)
    brute = {}
    for x0 in H_SCC:
        def dfs(x, vis, depth):
            best = None if _isinf(eH.get(x, INF)) else (LAM**depth) * eH[x]
            for u, y, w in recH["internal"].get(x, []):
                if y in vis:
                    continue
                sub = dfs(y, vis | {y}, depth + 1)
                if sub is None:
                    continue
                cand = (LAM**depth) * w + sub
                if best is None or cand < best:
                    best = cand
            return best
        brute[x0] = dfs(x0, {x0}, 0)
    check(
        "A10",
        all(Fraction(J_core[x]) == brute[x] == EXPECTED_J[x] for x in H_SCC),
        "SolveCore == brute force == (3.5, 2.375, 2.75) on H",
    )

    # A11 exact V table and the gap set
    V_pi, mu_pi = baseline_pi(G, R, XE, LAM)
    gap = {x for x in R if V_pi[x] < J[x]}
    check(
        "A11",
        all(V_pi[x] == EXPECTED_V[x] for x in R) and gap == EXPECTED_GAP,
        "V table exact; gap set = {2,3,4,5,13,14}",
    )

    # A12 improper sets: VI == PI == Floyd == {2,3,4,5,13,14}; master == {}
    adjR = restricted_adjacency(G, R)
    vi_seq = baseline_vi(adjR, XE, LAM, K_ITERS)
    fl_seq = baseline_floyd(adjR, XE, LAM, K_ITERS)
    mu_vi = greedy_mu(vi_seq[-1], adjR, XE, LAM)
    mu_fl = greedy_mu(fl_seq[-1], adjR, XE, LAM)
    bad_vi, bad_pi = improper_states(mu_vi, R), improper_states(mu_pi, R)
    bad_fl, bad_master = improper_states(mu_fl, R), improper_states(mu_master, R)
    check(
        "A12",
        bad_vi == bad_pi == bad_fl == EXPECTED_IMPROPER and bad_master == set(),
        f"VI/PI/Floyd improper = {sorted(bad_vi)} (6 states); master improper = 0",
    )

    # A13 VI/Floyd monotone, lower-bounded by V*, greedy action set stabilizes
    acts_star = greedy_actions(V_pi, adjR, XE, LAM)

    def monotone_and_bounded(seq) -> bool:
        for A, B in zip(seq[1:], seq[:-1]):
            for x in A:
                a, b = A[x], B.get(x)
                if b is not None and (a is None or a > b):
                    return False
                if a is not None and a < V_pi[x]:
                    return False
        return True

    def stabilization_k(seq) -> Optional[int]:
        k0 = None
        for k, Vk in enumerate(seq):
            if greedy_actions(Vk, adjR, XE, LAM) == acts_star:
                if k0 is None:
                    k0 = k
            else:
                k0 = None
        return k0

    k0_vi, k0_fl = stabilization_k(vi_seq), stabilization_k(fl_seq)
    check(
        "A13",
        monotone_and_bounded(vi_seq) and monotone_and_bounded(fl_seq)
        and k0_vi is not None and k0_vi <= K_ITERS
        and k0_fl is not None and k0_fl <= K_ITERS,
        f"monotone nonincreasing, >= V*; greedy sets stable from k0(VI)={k0_vi},"
        f" k0(Floyd)={k0_fl}",
    )

    # A14 master feedback attains J pointwise (single-exit H, no no-joint clash)
    check(
        "A14",
        all(closed_loop_cost(mu_master, x, LAM) == J[x] for x in R),
        "closed-loop cost of mu* equals J_lambda* at all 14 states",
    )

    # A15 lambda*: exact rational tests, dyadic bisection bracket, poly sign
    b34, b79 = beneficial_at(Fraction(3, 4)), beneficial_at(Fraction(7, 9))
    lo, hi = lambda_star_bisect()
    check(
        "A15",
        b34 is True and b79 is False
        and float(hi - lo) < 1e-15
        and poly_sign(lo) < 0 < poly_sign(hi)
        and abs(float(lo) - LAM_STAR) < 1e-9,
        f"beneficial at 3/4, not at 7/9; bracket [{float(lo):.12f}, {float(hi):.12f}]"
        " contains (1+sqrt(13))/6",
    )

    # A16 every finite value is dyadic (denominator a power of two)
    pool: List[Fraction] = [v for v in J.values()]
    pool += [V_pi[x] for x in R]
    pool += [Fraction(v) for e_ in (eH, eB) for v in e_.values() if not _isinf(v)]
    pool += list(thr.values()) + [G_LOOP_H]
    pool += [closed_loop_cost(mu_master, x, LAM) for x in R]
    check(
        "A16",
        all(Fraction(v).denominator & (Fraction(v).denominator - 1) == 0 for v in pool),
        f"{len(pool)} finite values, all denominators powers of 2",
    )

    # ---------------- outputs ----------------
    dnfs, term_counts = logical_form()
    n_improper = {"vi": len(bad_vi), "pi": len(bad_pi), "master": len(bad_master)}
    table = make_table_rows(V_pi, J, mu_vi, mu_pi, mu_master, scc, n_improper)
    table_path = os.path.join(RESULTS, "exp14_ex1_n4_table.tex")
    with open(table_path, "w") as fh:
        fh.write(table + "\n")

    fig1 = figure_stg(G, mu_vi, mu_master, EXPECTED_IMPROPER)
    fig2 = figure_threshold(G, adjR, R)

    lines = [
        f"E14: four-node BCN, N={N}, M={M}, xe=delta_16^{XE}, ue=delta_2^{UE},"
        f" lambda={LAM}",
        "  L = delta_16[" + " ".join(str(v) for v in L[:16]) + "",
        "               " + " ".join(str(v) for v in L[16:]) + "]",
        "  g(x,u) = Hamming(x, x_e) + [u != u_e]   (exact Fractions throughout)",
        "",
        "  logical form (Quine-McCluskey over u, x1..x4):",
    ]
    lines += [f"    {s}" for s in dnfs]
    lines += [
        f"    conjunct counts per update function: {term_counts}"
        f" (total {sum(term_counts)})",
        "",
        f"  |V_s| = {len(R)} / {N}, unstabilizable = {sorted(set(range(1, N + 1)) - R)}",
        f"  components = {len(scc.components)}, nontrivial = "
        f"{[sorted(c) for c in nontrivial]}",
        f"  folding: e(2)={fr_dec(eH[2])}, e(3)=e(5)=inf | "
        f"e(6)={fr_dec(eB[6])}, e(7)={fr_dec(eB[7])}, e(10)={fr_dec(eB[10])}",
        f"  detection: hasCycle(H)=True (sweeps={recH['sweeps']}),"
        f" hasCycle(B)=False (sweeps={recB['sweeps']}, strict={recB['strict']}, D(B)=2)",
        f"  cycle oracle on H: {len(cycles_H)} beneficial cycle(s), g_loop min = "
        f"{fr_dec(G_LOOP_H)}; thresholds {fr_dec(thr[2])}, {fr_dec(thr[3])}, {fr_dec(thr[5])}",
        "",
        "  state  bits   d  V_lambda*   J_lambda*   VI    PI    Floyd  master",
    ]
    for x in range(1, N + 1):
        b = "".join(str(t) for t in bits(x))
        if x in R:
            vs, js = fr_dec(V_pi[x]), fr_dec(J[x])
            kv = mu_vi.get(x, ("-",))[0]
            kp = mu_pi.get(x, ("-",))[0]
            kf = mu_fl.get(x, ("-",))[0]
            km = mu_master.get(x, ("-",))[0]
        else:
            vs, js, kv, kp, kf, km = "--", "inf", "-", "-", "-", "-"
        lines.append(
            f"  d16^{x:<3}({b}) {hamming_to_target(x)}  {vs:>9}  {js:>9}"
            f"   u={kv}   u={kp}   u={kf}    u={km}"
        )
    lines += [
        "",
        f"  gap states (V < J): {sorted(gap)}",
        f"  improper: VI={sorted(bad_vi)}, PI={sorted(bad_pi)},"
        f" Floyd={sorted(bad_fl)}, master={sorted(bad_master)}",
        f"  greedy action sets stable from k0(VI)={k0_vi}, k0(Floyd)={k0_fl}"
        f" (of {K_ITERS} exact sweeps)",
        f"  lambda*: bisection bracket [{float(lo):.12f}, {float(hi):.12f}],"
        f" closed form (1+sqrt(13))/6 = {LAM_STAR:.12f}",
        "",
        "  acceptance items:",
    ]
    lines += [f"    {p}" for p in passed]
    lines += [
        "",
        f"  outputs: {os.path.basename(table_path)}, {os.path.basename(fig1)},"
        f" {os.path.basename(fig2)}",
        "  ALL 16 ASSERTIONS PASSED",
    ]
    text = "\n".join(lines)
    with open(os.path.join(RESULTS, "exp14_ex1_n4.txt"), "w") as fh:
        fh.write(text + "\n")
    print(text)
    return {
        "passed": passed,
        "term_counts": term_counts,
        "k0_vi": k0_vi,
        "k0_floyd": k0_fl,
        "improper": sorted(bad_vi),
        "lambda_star_bracket": (lo, hi),
    }


if __name__ == "__main__":
    run()
