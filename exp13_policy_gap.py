"""E13: the feedback returned by SolveCore is pointwise optimal, not jointly.

SolveCore stores, at every table entry (v, W), the arc attaining the minimum of
the Held-Karp recursion, and reads the feedback off the entries (v, S). Each
such arc is the first arc of an optimal simple path from its own state, so the
values are correct; but the arcs of different states need not fit together into
one stationary feedback that leaves S. This experiment shows that they do not,
and that on a beneficial-cycle component no stationary feedback whatsoever can
attain the pointwise optima simultaneously.
"""
from __future__ import annotations

import os
import sys
from itertools import product
from typing import Dict, List, Set, Tuple

sys.path.insert(0, os.path.dirname(__file__))

from core_solver import (
    INF,
    assemble_proper_in_tree,
    brute_force_simple_paths,
    component_closed_loop_value,
    component_trapped_states,
    random_beneficial_scc,
    solve_core,
    solve_core_bitmask,
    solve_core_heldkarp,
)

RESULTS = os.path.join(os.path.dirname(__file__), "results")

Internal = Dict[int, List[Tuple[int, int, float]]]


def two_state_component(
    lam: float = 0.5, eps: float = 0.01, exit_weight: float = 1.5
) -> Tuple[Set[int], Internal, Dict[int, float]]:
    """Two mutually reachable states, each cheaper to leave through the other.

    Both states can exit at cost exit_weight, and the internal arcs are almost
    free, so each state prefers one detour through its neighbour: eps + lam *
    exit_weight beats exit_weight whenever exit_weight * (1 - lam) > eps. The
    two preferences point at each other.
    """
    S = {1, 2}
    internal: Internal = {1: [(0, 2, eps)], 2: [(0, 1, eps)]}
    e = {1: exit_weight, 2: exit_weight}
    return S, internal, e


def improper_states(
    S: Set[int],
    internal: Internal,
    e: Dict[int, float],
    lam: float,
    policy: Dict[int, Tuple],
) -> List[int]:
    """States whose closed loop under the given feedback never leaves S."""
    return sorted(component_trapped_states(S, internal, e, lam, policy))


def all_stationary_policies(S: Set[int], internal: Internal):
    """Every stationary feedback on the folded component: exit or one arc."""
    states = sorted(S)
    per_state = [
        [("exit", v)] + [("internal", u, y) for u, y, _w in internal.get(v, []) if y in S]
        for v in states
    ]
    for combo in product(*per_state):
        yield dict(zip(states, combo))


def best_joint_shortfall(
    S: Set[int],
    internal: Internal,
    e: Dict[int, float],
    lam: float,
    J: Dict[int, float],
) -> Tuple[int, float]:
    """Over all stationary feedbacks, the fewest states left above the optimum.

    Returns (states above optimum, worst excess there) for the best feedback.
    A pair (0, 0.0) means some single stationary feedback is pointwise optimal
    everywhere; anything else means no such feedback exists.
    """
    best_count, best_excess = len(S) + 1, INF
    for mu in all_stationary_policies(S, internal):
        vals = {
            x: component_closed_loop_value(S, internal, e, lam, mu, x) for x in S
        }
        above = [x for x in S if vals[x] > J[x] + 1e-9]
        excess = max((vals[x] - J[x] for x in above), default=0.0)
        if (len(above), excess) < (best_count, best_excess):
            best_count, best_excess = len(above), excess
    return best_count, best_excess


def e13a_two_state(lam: float = 0.5) -> dict:
    """The minimal instance: correct values, first-arc feedback trapped in S."""
    S, internal, e = two_state_component(lam=lam)
    J_bf = brute_force_simple_paths(S, internal, e, lam)
    out = {"lam": lam, "e": dict(e), "J": {}, "policy": {}, "impl": {}}
    for name, solver in (("dict", solve_core_heldkarp), ("bitmask", solve_core_bitmask)):
        J, policy, _ = solver(S, internal, e, lam)
        bad = improper_states(S, internal, e, lam, policy)
        out["impl"][name] = {
            "J": dict(J),
            "policy": {x: policy[x] for x in sorted(S)},
            "improper": bad,
        }
        for x in S:
            assert abs(J[x] - J_bf[x]) < 1e-12, f"{name}: value wrong at {x}"
        assert bad == sorted(S), f"{name}: expected both states trapped, got {bad}"
    out["J"] = dict(J_bf)
    out["policy"] = out["impl"]["dict"]["policy"]

    n_above, excess = best_joint_shortfall(S, internal, e, lam, J_bf)
    assert n_above >= 1, "a pointwise optimal stationary feedback should not exist"
    out["joint"] = {"states_above_optimum": n_above, "excess": excess}

    V_tree, tree = assemble_proper_in_tree(S, internal, e, lam)
    assert not improper_states(S, internal, e, lam, tree), "in-tree must be proper"
    out["in_tree"] = {
        "policy": {x: tree[x] for x in sorted(S)},
        "V": {x: V_tree[x] for x in sorted(S)},
        "excess": {x: V_tree[x] - J_bf[x] for x in sorted(S)},
    }
    return out


def e13b_random(
    sizes: Tuple[int, ...] = (4, 5, 6, 7, 8), seeds: range = range(20), lam: float = 0.5
) -> dict:
    """How often the first-arc reading is trapped, and whether the in-tree saves it.

    The in-tree fallback is checked on every instance: it must be proper
    everywhere, and its value must sit at or above the pointwise optimum.
    """
    rows = []
    trapped_instances = 0
    worst_excess = 0.0
    for n in sizes:
        n_bad, n_states_bad = 0, 0
        for seed in seeds:
            S, internal, e = random_beneficial_scc(n, lam, seed=seed)
            J, policy, _ = solve_core(S, internal, e, lam)
            bad = improper_states(S, internal, e, lam, policy)
            if bad:
                n_bad += 1
                n_states_bad += len(bad)
            V_tree, tree = assemble_proper_in_tree(S, internal, e, lam)
            assert not improper_states(
                S, internal, e, lam, tree
            ), f"in-tree improper at n={n} seed={seed}"
            assert all(
                V_tree[x] >= J[x] - 1e-9 for x in S
            ), f"in-tree below the optimum at n={n} seed={seed}"
            worst_excess = max(worst_excess, max(V_tree[x] - J[x] for x in S))
        rows.append((n, len(list(seeds)), n_bad, n_states_bad))
        trapped_instances += n_bad
    assert trapped_instances > 0, "expected the reading to fail on some component"
    return {
        "rows": rows,
        "trapped_instances": trapped_instances,
        "worst_in_tree_excess": worst_excess,
    }


def e13c_no_joint_optimum(
    sizes: Tuple[int, ...] = (4, 5), seeds: range = range(6), lam: float = 0.5
) -> dict:
    """Exhaustive check that no stationary feedback attains every optimum."""
    rows = []
    for n in sizes:
        for seed in seeds:
            S, internal, e = random_beneficial_scc(n, lam, seed=seed)
            J = brute_force_simple_paths(S, internal, e, lam)
            n_above, excess = best_joint_shortfall(S, internal, e, lam, J)
            rows.append((n, seed, n_above, excess))
    return {"rows": rows, "max_states_above": max(r[2] for r in rows)}


def run() -> dict:
    os.makedirs(RESULTS, exist_ok=True)
    print("E13a: two-state counterexample...")
    r13a = e13a_two_state()
    print("E13b: how often the first-arc reading is trapped...")
    r13b = e13b_random()
    print("E13c: exhaustive search for a jointly optimal feedback...")
    r13c = e13c_no_joint_optimum()

    lines = ["E13: pointwise optimal values, no jointly optimal feedback", ""]
    lines.append("  E13a: two-state component, lam=0.5, internal arcs 0.01, e=1.5")
    for x in sorted(r13a["J"]):
        act = r13a["policy"][x]
        lines.append(
            f"    x={x}: J*={r13a['J'][x]:.4f}  stored arc={act}"
        )
    lines.append(
        f"    both implementations agree with brute force; states trapped in S:"
        f" {r13a['impl']['dict']['improper']}"
    )
    lines.append(
        f"    best stationary feedback still leaves"
        f" {r13a['joint']['states_above_optimum']} state(s) above the optimum,"
        f" by {r13a['joint']['excess']:.4f}"
    )
    lines.append(
        "    in-tree fallback is proper everywhere, at values "
        + ", ".join(
            f"x={x}: {r13a['in_tree']['V'][x]:.4f} (+{r13a['in_tree']['excess'][x]:.4f})"
            for x in sorted(r13a["in_tree"]["V"])
        )
    )
    lines.append("")
    lines.append("  E13b: random beneficial-cycle components (20 seeds each)")
    lines.append("    |S|   instances   first-arc reading trapped   trapped states")
    for n, tot, bad, states in r13b["rows"]:
        lines.append(f"    {n:<5d} {tot:<11d} {bad:<26d} {states}")
    lines.append(
        "    the in-tree fallback is proper on all of them, worst excess over"
        f" the optimum {r13b['worst_in_tree_excess']:.4f}"
    )
    lines.append("")
    lines.append("  E13c: all stationary feedbacks enumerated")
    lines.append("    |S|   seed   states above optimum   worst excess")
    for n, seed, above, excess in r13c["rows"]:
        lines.append(f"    {n:<5d} {seed:<6d} {above:<22d} {excess:.4f}")
    lines.append("  ASSERTIONS PASSED")

    text = "\n".join(lines)
    with open(os.path.join(RESULTS, "exp13_policy_gap.txt"), "w") as fh:
        fh.write(text + "\n")
    print(text)
    return {"e13a": r13a, "e13b": r13b, "e13c": r13c}


if __name__ == "__main__":
    run()
