"""E12: the depth-parametrized route on the NP-complete kernel.

Algorithm core is exponential in the component size |S|.  Section main also
offers a second route for the case where the optimal stabilizing simple path
inside the component is shallow: restrict the search to simple paths of at most
D internal arcs.  That costs O(|S| M^D), polynomial in |S| and M for every
fixed D, and returns

    J^(D) >= J_lambda^*,

with equality once D reaches the depth of the optimal simple path.  Since J^(D)
is non-increasing in D and reaches J_lambda^* at D = |S| - 1, watching where it
stops moving locates that depth.

Part A validates the solver against the exact subset solver on random
beneficial-cycle components small enough for both.  Part B runs it on the
108-state ara operon component of E9 under the side-effect-aware cost, the one
instance for which the subset solver is out of reach (2^107 subsets).
"""
from __future__ import annotations

import os
import sys
import time
from typing import Dict, List, Optional, Set, Tuple

sys.path.insert(0, os.path.dirname(__file__))

from ara_io import build_ara_dynamics, hamming_cost_factory  # noqa: E402
from core_solver import (  # noqa: E402
    INF,
    _internal_arcs,
    augment_component,
    build_graph,
    condense,
    detect_beneficial_cycle_v2,
    random_beneficial_scc,
    reverse_reachable,
    solve_core,
    solve_core_depth,
    terminal_to_augmented,
)

RESULTS = os.path.join(os.path.dirname(__file__), "results")

XE, UE, LAM = 512, 11, 0.5
XD, UD = 9, 2
EPS = 0.05

TOL = 1e-9


def e12a_validate(sizes=(5, 6, 7, 8), seeds=range(5, 13)) -> dict:
    """J^(D) is non-increasing, above the exact value, and meets it by D=|S|-1."""
    rows = []
    mismatches = 0
    monotone_breaks = 0
    below_exact = 0
    depths: List[int] = []

    for seed in seeds:
        for n in sizes:
            S, internal, e = random_beneficial_scc(n, LAM, seed=seed)
            J_exact, _, _ = solve_core(S, internal, e, LAM)
            prev: Optional[Dict[int, float]] = None
            hit: Optional[int] = None
            for D in range(1, n):
                J_d, _, _ = solve_core_depth(S, internal, e, LAM, D)
                for x in S:
                    if J_d[x] < J_exact[x] - TOL:
                        below_exact += 1
                    if prev is not None and J_d[x] > prev[x] + TOL:
                        monotone_breaks += 1
                if hit is None and all(
                    abs(J_d[x] - J_exact[x]) <= TOL for x in S
                ):
                    hit = D
                prev = J_d
            # prev is J^(|S|-1): must equal the exact optimum
            assert prev is not None
            if any(abs(prev[x] - J_exact[x]) > TOL for x in S):
                mismatches += 1
            if hit is not None:
                depths.append(hit)
            rows.append((seed, n, hit))

    assert mismatches == 0, f"{mismatches} components where J^(|S|-1) != exact"
    assert below_exact == 0, f"{below_exact} values below the exact optimum"
    assert monotone_breaks == 0, f"{monotone_breaks} monotonicity violations"

    return {
        "components": len(rows),
        "mismatches": mismatches,
        "below_exact": below_exact,
        "monotone_breaks": monotone_breaks,
        "depth_min": min(depths),
        "depth_max": max(depths),
        "depth_mean": sum(depths) / len(depths),
    }


def _policy_failures(
    S: Set[int], policy: Dict[int, Tuple], e: Dict[int, float], D: int
) -> int:
    """States from which the stationary feedback fails to leave S within D arcs.

    A component feedback that leaves S is proper: the exit weight already
    carries the finite cost of the downstream states, which the reverse
    topological order of Algorithm master has settled.  Reading the first arc
    of each state's own optimal depth-D path gives a stationary feedback, and
    this counts the states on which that reading does not close.
    """
    bad = 0
    for x in S:
        v, steps, ok = x, 0, False
        while steps <= D:
            move = policy.get(v)
            if move is None:
                break
            if move[0] == "exit":
                ok = e.get(v, INF) < 1e14
                break
            v = move[2]
            steps += 1
        if not ok:
            bad += 1
    return bad


def _ara_carrier() -> Tuple[Set[int], Dict, Dict, Set[int]]:
    """The beneficial-cycle component of E9 with all sixteen inputs available."""
    n, _m, N, M, _L, f = build_ara_dynamics()
    base = hamming_cost_factory(XE, UE, n)

    def g(x: int, u: int) -> float:
        return EPS if (x == XD and u == UD) else base(x, u)

    G = build_graph(f, g, list(range(1, N + 1)), list(range(1, M + 1)),
                    Cu=None, xe=XE)
    R = reverse_reachable(G, XE)
    scc = condense(G, R, XE)

    J_known: Dict[int, float] = {XE: 0.0}
    carrier = None
    for cid in scc.reverse_topo:
        S = scc.components[cid]
        if XE in S:
            if len(S) == 1:
                continue
            S2, internal, e = terminal_to_augmented(
                S, _internal_arcs(G, S), XE, LAM)
        else:
            S2, (internal, e) = S, augment_component(G, S, J_known, LAM)
        if not S2:
            continue
        flag, trapped, _ = detect_beneficial_cycle_v2(S2, internal, e, LAM)
        if flag and carrier is None:
            carrier = (S2, internal, e, trapped)
        for x in S2:
            J_known.setdefault(x, 0.0)

    assert carrier is not None, "no beneficial-cycle component found"
    return carrier


def e12b_ara(max_depth: int = 5) -> dict:
    """The instance Algorithm core declines: 2^107 subsets, solved by depth."""
    S, internal, e, trapped = _ara_carrier()
    deg = [len(internal.get(v, [])) for v in S]

    rows = []
    proper_from: Optional[int] = None
    for D in range(1, max_depth + 1):
        t0 = time.perf_counter()
        J_d, pol, expansions = solve_core_depth(S, internal, e, LAM, D)
        dt = time.perf_counter() - t0
        finite = [v for v in J_d.values() if v < 1e14]
        unreached = len(J_d) - len(finite)
        bad = _policy_failures(S, pol, e, D)
        if bad == 0 and unreached == 0 and proper_from is None:
            proper_from = D
        rows.append((D, expansions, dt, J_d.get(XD),
                     max(finite) if finite else None, unreached, bad))

    # The stationary feedback that reads off the first arc of each state's own
    # optimal depth-D path need not close at the smallest D for which every
    # value is finite: two states may disagree about a shared successor.  One
    # extra arc of budget resolves it here.
    assert proper_from is not None and proper_from <= 3, (
        f"no proper feedback by D=3 (first at {proper_from})")

    # J^(D) is non-increasing and settles; the settled value is the one the
    # restricted-actuation regime of E9 obtains exactly from SolveCore.
    decoy_vals = [r[3] for r in rows if r[3] is not None and r[3] < 1e14]  # noqa: E501
    assert all(
        b <= a + TOL for a, b in zip(decoy_vals, decoy_vals[1:])
    ), "J^(D) at the decoy is not non-increasing"
    settled = decoy_vals[-1]
    assert abs(settled - 11.375) < 1e-9, f"decoy value {settled} != 11.375"
    first_settled = next(
        D for D, _, _, v, *_ in rows if v is not None and abs(v - settled) < TOL
    )
    assert first_settled <= 3, f"settles only at D={first_settled}"

    with open(os.path.join(RESULTS, "exp12_depth_ara.csv"), "w") as fh:
        fh.write("D,expansions,time_s,J_decoy,J_worst,unreached,policy_failures\n")
        for D, ex, dt, jd, jw, un, bad in rows:
            fh.write(f"{D},{ex},{dt:.4f},{jd},{jw},{un},{bad}\n")

    return {
        "size": len(S),
        "out_degree_max": max(deg),
        "out_degree_mean": sum(deg) / len(deg),
        "trapped": sorted(trapped),
        "rows": rows,
        "settled": settled,
        "first_settled": first_settled,
        "proper_from": proper_from,
    }


def run() -> dict:
    os.makedirs(RESULTS, exist_ok=True)
    print("E12a: depth solver against the exact subset solver...")
    r12a = e12a_validate()
    print("E12b: the 107-state ara component...")
    r12b = e12b_ara()

    lines = [
        "E12: depth-parametrized route",
        f"  E12a: {r12a['components']} random components, "
        f"mismatches={r12a['mismatches']}, "
        f"below-exact={r12a['below_exact']}, "
        f"monotonicity breaks={r12a['monotone_breaks']}",
        f"        depth at which J^(D) meets the exact optimum: "
        f"min={r12a['depth_min']}, max={r12a['depth_max']}, "
        f"mean={r12a['depth_mean']:.2f}",
        f"  E12b: ara carrier |S|={r12b['size']}, "
        f"internal out-degree max={r12b['out_degree_max']}, "
        f"mean={r12b['out_degree_mean']:.2f}",
        f"        trapped states: {r12b['trapped']}",
        "         D   expansions      time (s)   J(decoy)     J(worst)"
        "  unreached  improper",
    ]
    for D, ex, dt, jd, jw, un, bad in r12b["rows"]:
        jd_s = "inf" if jd is None or jd >= 1e14 else f"{jd:.6f}"
        jw_s = "-" if jw is None else f"{jw:.6f}"
        lines.append(
            f"        {D:2d}   {ex:>11,}   {dt:9.3f}   {jd_s:>10}   {jw_s:>10}"
            f"   {un:>8d}   {bad:>7d}"
        )
    lines.append(
        f"        proper feedback at all {r12b['size']} states from "
        f"D={r12b['proper_from']}"
    )
    lines.append(
        f"        settles at J^(D)={r12b['settled']} from D={r12b['first_settled']},"
        f" matching the exact SolveCore value of the restricted-actuation regime"
    )
    lines.append("  ASSERTIONS PASSED")

    text = "\n".join(lines)
    with open(os.path.join(RESULTS, "exp12_depth.txt"), "w") as fh:
        fh.write(text + "\n")
    print(text)
    return {"e12a": r12a, "e12b": r12b}


if __name__ == "__main__":
    run()
