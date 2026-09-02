"""E1: Reverse Dijkstra fails under discounting (C1)."""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from core_solver import (
    local_lc,
    make_toy_dijkstra_graph,
    master,
    reverse_dijkstra_discounted_wrong,
    reverse_reachable,
    augment_component,
)

RESULTS = os.path.join(os.path.dirname(__file__), "results")


def run() -> dict:
    os.makedirs(RESULTS, exist_ok=True)
    G, xe, lam = make_toy_dijkstra_graph()
    R = reverse_reachable(G, xe)

    # Hand-checked values (main.tex Fig. dijkstra-ce)
    A, B = 2, 1

    J_master, _mu = master(G, xe, lam)

    # label-correcting on upstream component after downstream J* known
    internal, e = augment_component(G, {B}, {A: 100.0, xe: 0.0}, lam)
    J_lc, _, _ = local_lc({B}, internal, e, lam)

    J_dijk_wrong = reverse_dijkstra_discounted_wrong(G, xe, lam, R)
    assert abs(J_master[A] - 100.0) < 1e-9, J_master[A]
    assert abs(J_master[B] - 51.0) < 1e-9, J_master[B]
    assert abs(J_lc[B] - 51.0) < 1e-9
    assert abs(J_dijk_wrong[B] - 60.0) < 1e-9, J_dijk_wrong[B]
    assert J_dijk_wrong[B] > J_master[B]

    lines = [
        "E1: Dijkstra counterexample (lambda=0.5)",
        f"  J*(A) = {J_master[A]:.4f}  (expected 100)",
        f"  J*(B) = {J_master[B]:.4f}  (expected 51, via expensive A)",
        f"  reverse Dijkstra J(B) = {J_dijk_wrong[B]:.4f}  (wrong: 60)",
        f"  label-correcting J(B) = {J_lc[B]:.4f}  (correct: 51)",
        "  ASSERTIONS PASSED",
    ]
    text = "\n".join(lines)
    out = os.path.join(RESULTS, "exp1_dijkstra.txt")
    with open(out, "w") as f:
        f.write(text + "\n")
    print(text)
    return {"J_A": J_master[A], "J_B": J_master[B], "dijkstra_B": J_dijk_wrong[B]}


if __name__ == "__main__":
    run()
