"""Uniform-optimality certificates on theory examples, Example 1, and
recorded real-network kernels.

Run: python simulations/exp20_uniform.py
"""
from __future__ import annotations

import csv
import os
import sys
import time
from fractions import Fraction
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))

from folded_uniform import (  # noqa: E402
    INF,
    Folded,
    bottleneck,
    exact_exc_star,
    greedy_vstar_tree,
    held_karp,
    spanning_from_acts,
    tight_arcs,
    tree_cost,
    value_star,
)
from verify_uniform import five_state_ratio, n_ring, two_state  # noqa: E402

RESULTS = Path(__file__).resolve().parent / "results"
F = Fraction


def analyze(name: str, comp: Folded, gap_states=None, kernel=None) -> dict:
    t0 = time.perf_counter()
    J = held_karp(comp)
    V = value_star(comp)
    joint = spanning_from_acts(comp, tight_arcs(comp, J))
    try:
        Exc, mu = exact_exc_star(comp, J)
    except RuntimeError:
        Exc, mu = None, None
    sstar, hstar, tree = bottleneck(comp, J)
    mu_list = [tree[x] for x in range(comp.n)]
    tree_exc = max(tree_cost(comp, mu_list, x) - J[x] for x in range(comp.n))
    ub = sstar * (1 - comp.lam ** hstar) / (1 - comp.lam) if sstar > 0 else F(0)
    n_gap = sum(1 for x in range(comp.n) if V[x] < J[x])
    gmu, cert = greedy_vstar_tree(comp)
    cert_max = max(cert)
    dt = time.perf_counter() - t0
    row = {
        "name": name,
        "|S|": comp.n,
        "joint": joint,
        "Exc*": str(Exc) if Exc is not None else "n/a",
        "s*": str(sstar),
        "h*": hstar,
        "tree_excess": str(tree_exc),
        "ub": str(ub),
        "n_gap": n_gap,
        "|I_VI|": gap_states if gap_states is not None else n_gap,
        "kernel": kernel if kernel is not None else comp.n,
        "greedy_cert": str(cert_max),
        "time_s": f"{dt:.4f}",
    }
    return row


def example1_H() -> Folded:
    # H = {2,3,5} after folding; e(2)=7/2, e(3)=e(5)=∞, λ=1/2.
    # Internal: 2→3 (g=1), 3→5 (g=1 and 2), 5→2 (g=1 and 2).
    internal = [
        (0, 1, 1),
        (1, 2, 1),
        (1, 2, 2),
        (2, 0, 1),
        (2, 0, 2),
    ]
    return Folded(3, internal, [F(7, 2), INF, INF], F(1, 2))


def write_rows(path: Path, rows: list):
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(rows[0].keys())
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def milp_large():
    """Replay existing MILP logs and try the two detect-only kernels."""
    lines = []
    milp_csv = RESULTS / "exp16_milp.csv"
    if milp_csv.exists():
        with milp_csv.open() as f:
            for row in csv.DictReader(f):
                lines.append(
                    f"  existing {row['tag']}: |S|={row['size']} "
                    f"coverage={row['coverage']} t={row['t_milp_s']}s "
                    f"certified={row['certified']}"
                )
    # Attempt large kernels only if methods + pulp are importable.
    try:
        import methods as MM  # noqa: F401
        import pulp  # noqa: F401
        pulp_ok = True
    except Exception as exc:
        pulp_ok = False
        lines.append(f"  skip live MILP ({exc})")
    lines.append(
        "  kernels beyond the Held–Karp cap (|S|=34, 105, 107) are run by "
        "exp21_uniform_real.py; see exp21_uniform_real_{small,large}.txt."
    )
    return lines, pulp_ok


def main():
    rows = [
        analyze("two-state (prop:no-joint)", two_state()),
        analyze("3-ring", n_ring(3)),
        analyze("5-state ratio", five_state_ratio()),
        analyze("Example1-H", example1_H(), gap_states=6, kernel=3),
    ]
    write_rows(RESULTS / "exp20_uniform.csv", rows)
    milp_lines, pulp_ok = milp_large()
    out = RESULTS / "exp20_uniform.txt"
    with out.open("w") as f:
        f.write("E20: uniform optimality certificates\n")
        f.write("  exact Fractions throughout on folded components\n\n")
        for r in rows:
            f.write(
                f"  {r['name']}: |S|={r['|S|']} joint={r['joint']} "
                f"Exc*={r['Exc*']} s*={r['s*']} h*={r['h*']} "
                f"tree_excess={r['tree_excess']} n_gap={r['n_gap']}\n"
            )
        f.write("\nMILP / large kernels:\n")
        f.write("\n".join(milp_lines) + "\n")
        f.write(
            "\nNotes:\n"
            "  Example 1 (global): exhaustive 2^16 enumeration (exp18) shows a "
            "proper feedback attaining J* on all 14 states of V_s, so Exc*(global)=0 "
            "even though H carries a beneficial cycle (V*<J* on 6 states).\n"
            "  Upper sandwich factor tightness: open; 5-state ratio "
            "Exc*/s* = 144/83 ≈ 1.735 vs (1-λ^5)/(1-λ)=31/16=1.9375 at λ=1/2.\n"
            "  Earlier searches reported Exc*/s* up to 949/375 ≈ 2.53 (lambda=9/10, "
            "Sep 22) and, in an independent rerun, at most ≈ 1.86 (lambda=9/10, "
            "Sep 28); both are J*-based. See exp22_tightness_search.txt for the "
            "current exact search.\n"
            f"  pulp available: {pulp_ok}\n"
        )
    print(out.read_text())
    print("wrote", RESULTS / "exp20_uniform.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
