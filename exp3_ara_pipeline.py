"""E3: ara operon pipeline + SCC histogram (C3, C4)."""
from __future__ import annotations

import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(__file__))

from ara_rules import verify as verify_rules
from ara_io import (
    build_ara_dynamics,
    control_fixpoints,
    hamming_cost_factory,
    reverse_reachable_states,
)
from core_solver import (
    _internal_arcs,
    augment_component,
    build_graph,
    condense,
    detect_beneficial_cycle_v2,
    master,
    reverse_reachable,
    simulate_closed_loop,
    terminal_to_augmented,
    vi_policy,
    vi_unconstrained,
)

RESULTS = os.path.join(os.path.dirname(__file__), "results")


def run(xe: int = 512, ue: int = 11, lam: float = 0.5) -> dict:
    os.makedirs(RESULTS, exist_ok=True)
    rules = verify_rules()
    n, m, N, M, L, f = build_ara_dynamics()
    g = hamming_cost_factory(xe, ue, n)
    states = list(range(1, N + 1))
    inputs = list(range(1, M + 1))
    G = build_graph(f, g, states, inputs, xe=xe)

    # reverse adjacency for depth
    rev = [set() for _ in range(N + 1)]
    for x in range(1, N + 1):
        for u in inputs:
            y = f(x, u)
            rev[y].add(x)

    R, D_reach = reverse_reachable_states(rev, xe)
    J, mu = master(G, xe, lam, f=f, ue=ue)
    V = vi_unconstrained(G, lam, R, xe)
    pol_vi = vi_policy(G, V, lam, R)

    scc = condense(G, R, xe)
    bc_sccs = []
    for cid in scc.reverse_topo:
        S = scc.components[cid]
        if len(S) <= 1:
            continue
        if xe in S:
            S2, internal, e = terminal_to_augmented(S, _internal_arcs(G, S), xe, lam)
        else:
            S2, (internal, e) = S, augment_component(G, S, J, lam)
        flag, _trapped, _V = detect_beneficial_cycle_v2(S2, internal, e, lam)
        if flag:
            bc_sccs.append((cid, len(S2)))

    # VI non-proper count
    vi_improper = 0
    master_improper = 0
    gap_states = 0
    for x in R:
        if abs(V[x] - J[x]) > 1e-6:
            gap_states += 1
        ok_vi, _, _ = simulate_closed_loop(f, pol_vi, x, xe, max_steps=N + 5)
        ok_m, _, _ = simulate_closed_loop(f, mu, x, xe, max_steps=N + 5)
        if not ok_vi:
            vi_improper += 1
        if not ok_m:
            master_improper += 1

    sizes = [len(c) for c in scc.components]
    hist = sorted(Counter(sizes).items())

    fps = control_fixpoints(L, N, M)
    lines = [
        f"E3: ara operon pipeline (N={N}, M={M}, xe={xe}, ue={ue}, lambda={lam})",
        f"  Boolean rules of Table ara-rules verified against L:"
        f" {rules['checked']} node evaluations, {rules['nodes']} nodes",
        f"  control fixpoints: {len(fps)}",
        f"  |R| = {len(R)} / {N}, D_reach = {D_reach}",
        f"  SCC count = {len(scc.components)}, max SCC size = {max(sizes)}",
        f"  SCC size distribution (size: count) = {dict(hist)}",
        f"  beneficial-cycle SCCs: {len(bc_sccs)} {bc_sccs[:5]}",
        f"  V!=J states: {gap_states} / {len(R)}",
        f"  VI non-proper states: {vi_improper} / {len(R)}",
        f"  master non-proper states: {master_improper} / {len(R)}",
        f"  sample J(1)={J.get(1, 'NA'):.4f}, V(1)={V.get(1, 'NA'):.4f}",
        "  ASSERTIONS: master all proper expected",
    ]
    assert master_improper == 0, f"master improper count {master_improper}"
    assert gap_states == 0, f"gap {gap_states}"  # natural Hamming cost: (W) holds

    text = "\n".join(lines) + "\n  ASSERTIONS PASSED\n"
    out = os.path.join(RESULTS, "exp3_ara_pipeline.txt")
    with open(out, "w") as f:
        f.write(text)
    print(text)
    return {
        "|R|": len(R),
        "D_reach": D_reach,
        "scc_count": len(scc.components),
        "vi_improper": vi_improper,
        "gap_states": gap_states,
    }


if __name__ == "__main__":
    run()
