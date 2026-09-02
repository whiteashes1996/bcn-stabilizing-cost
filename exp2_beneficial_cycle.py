"""E2: Beneficial cycle => V_lambda* < J_lambda* (C2)."""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from core_solver import (
    brute_force_simple_paths,
    make_beneficial_cycle_graph,
    master,
    reverse_reachable,
    simulate_closed_loop,
    solve_core_heldkarp,
    vi_policy,
    vi_unconstrained,
    augment_component,
    detect_beneficial_cycle,
)

RESULTS = os.path.join(os.path.dirname(__file__), "results")
Y = 1


def run() -> dict:
    os.makedirs(RESULTS, exist_ok=True)
    G, xe, lam = make_beneficial_cycle_graph()
    R = reverse_reachable(G, xe)
    cycle = {1, 2, 3}

    V = vi_unconstrained(G, lam, R, xe)
    J_master, mu = master(G, xe, lam)

    internal, e = augment_component(G, cycle, {xe: 0.0}, lam)
    assert detect_beneficial_cycle(cycle, internal, e, lam)
    J_core, _, _ = solve_core_heldkarp(cycle, internal, e, lam)
    J_brute = brute_force_simple_paths(cycle, internal, e, lam)

    g_loop = 1 + lam + lam**2
    V_loop = g_loop / (1 - lam**3)
    J_tail = J_master[Y]

    pol_vi = vi_policy(G, V, lam, R)

    def f(x, u):
        for uu, y, _w in G.by_state.get(x, []):
            if uu == u:
                return y
        return x

    proper_vi, steps_vi, traj_vi = simulate_closed_loop(f, pol_vi, Y, xe)

    # Hand-checked
    assert abs(V[Y] - 2.0) < 1e-9, V[Y]
    assert abs(J_master[Y] - 26.5) < 1e-9, J_master[Y]
    assert abs(J_core[Y] - 26.5) < 1e-9
    assert abs(J_brute[Y] - 26.5) < 1e-9
    assert V[Y] < J_master[Y]
    assert g_loop < (1 - lam**3) * J_tail
    assert not proper_vi or steps_vi > 100  # VI traps in cycle

    lines = [
        "E2: Beneficial cycle (lambda=0.5, 3-cycle + exit e=100)",
        f"  g_loop(C) = {g_loop:.4f}",
        f"  V_lambda*(y) = {V[Y]:.4f}  (expected 2, VI lasso)",
        f"  J_lambda*(y) = {J_master[Y]:.4f}  (expected 26.5, proper)",
        f"  SolveCore J(y) = {J_core[Y]:.4f}",
        f"  brute-force J(y) = {J_brute[Y]:.4f}",
        f"  beneficial: {g_loop:.4f} < {(1-lam**3)*J_tail:.4f} = {(1-lam**3)*J_tail:.4f}",
        f"  VI greedy proper at y? {proper_vi}, traj length={len(traj_vi)}",
        "  ASSERTIONS PASSED",
    ]
    text = "\n".join(lines)
    out = os.path.join(RESULTS, "exp2_beneficial_cycle.txt")
    with open(out, "w") as f:
        f.write(text + "\n")
    print(text)
    return {"V_y": V[Y], "J_y": J_master[Y]}


if __name__ == "__main__":
    run()
