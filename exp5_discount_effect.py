"""E5: Discount shifts optimal path — defer expensive actions (C6)."""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from core_solver import (
    make_discount_effect_graph,
    master,
    undiscounted_shortest_path,
)

RESULTS = os.path.join(os.path.dirname(__file__), "results")


def path_choice(J_B: float, lam: float) -> str:
    """P_rush: expensive early; P_defer: expensive late."""
    c_rush = 50 + lam * 1
    c_defer = 1 + lam * 50
    if abs(J_B - c_rush) < 1e-6:
        return "P_rush_early_expensive"
    if abs(J_B - c_defer) < 1e-6:
        return "P2_late_expensive"
    return f"other({J_B:.4f})"


def run() -> dict:
    os.makedirs(RESULTS, exist_ok=True)
    G, xe, _ = make_discount_effect_graph()
    B = 1  # start

    lams = np.linspace(0.05, 0.995, 40)
    choices = []
    J_vals = []
    for lam in lams:
        J, _ = master(G, xe, float(lam))
        J_vals.append(J[B])
        choices.append(path_choice(J[B], float(lam)))

    J_undisc = undiscounted_shortest_path(G, xe)
    undisc_choice = path_choice(J_undisc[B], 1.0)

    # Discount defers expensive action: cheap-then-expensive beats expensive-then-cheap for all lambda<1
    # P_defer: 1->3 (1) ->4 (50): 1 + lam*50
    # P_rush:  1->2 (50)->4 (1): 50 + lam*1
    first = choices[0]
    last = choices[-1]
    assert first.startswith("P2") or "defer" in first.lower() or first.startswith("P2_late"), first
    assert last.startswith("P2") or last.startswith("P2_late"), last
    switch_lam = None  # P2 (defer expensive) dominates for lambda in (0,1) on this toy graph

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(lams, J_vals, "b-", label="J*(B)")
    ax.axhline(50 + 1, color="g", ls="--", alpha=0.6, label="P1 cost at lam=1")
    ax.axhline(1 + 50, color="r", ls="--", alpha=0.6, label="P2 cost at lam=1")
    if switch_lam:
        ax.axvline(switch_lam, color="k", ls=":", label=f"switch ~{switch_lam:.2f}")
    ax.set_xlabel("lambda")
    ax.set_ylabel("J*(B)")
    ax.set_title("Discount effect: defer expensive action")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS, "exp5_discount_effect.png"), dpi=150)
    plt.close(fig)

    lines = [
        "E5: Discount trajectory effect",
        f"  undiscounted optimal from B: {undisc_choice}, J={J_undisc[B]:.4f}",
        f"  lambda=0.05: {choices[0]}, J={J_vals[0]:.4f}  (defer expensive wins)",
        f"  lambda=0.995: {choices[-1]}, J={J_vals[-1]:.4f}",
        "  Note: on this toy graph P_defer (1+50*lambda) beats P_rush for all lambda in (0,1)",
        "  ASSERTIONS PASSED",
    ]
    text = "\n".join(lines)
    out = os.path.join(RESULTS, "exp5_discount_effect.txt")
    with open(out, "w") as f:
        f.write(text + "\n")
    print(text)
    return {"switch_lam": switch_lam, "undisc": undisc_choice}


if __name__ == "__main__":
    run()
