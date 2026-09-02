"""E4: SolveCore validation against exhaustive enumeration, and its scaling."""
from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from core_solver import (
    brute_force_simple_paths,
    random_beneficial_scc,
    solve_core_bitmask,
    solve_core_heldkarp,
)

RESULTS = os.path.join(os.path.dirname(__file__), "results")


def e4a_crosscheck(seeds: range = range(5, 12)) -> dict:
    """Both SolveCore implementations against brute-force simple-path search."""
    lam = 0.5
    mismatches = 0
    tests = 0
    for seed in seeds:
        for n in (4, 5, 6, 7):
            S, internal, e = random_beneficial_scc(n, lam, seed=seed)
            J_hk, _, _ = solve_core_heldkarp(S, internal, e, lam)
            J_bm, _, _ = solve_core_bitmask(S, internal, e, lam)
            J_bf = brute_force_simple_paths(S, internal, e, lam)
            for x in S:
                tests += 2
                if abs(J_hk[x] - J_bf[x]) > 1e-6:
                    mismatches += 1
                    print(f"  mismatch (dict) seed={seed} n={n} x={x}:"
                          f" hk={J_hk[x]} bf={J_bf[x]}")
                if abs(J_bm[x] - J_bf[x]) > 1e-6:
                    mismatches += 1
                    print(f"  mismatch (bitmask) seed={seed} n={n} x={x}:"
                          f" bm={J_bm[x]} bf={J_bf[x]}")
    assert mismatches == 0, f"{mismatches} mismatches"
    return {"tests": tests, "mismatches": mismatches}


def e4b_scaling(max_n: int = 20, dict_max_n: int = 16) -> dict:
    """Runtime of SolveCore against |S|, with the two implementations compared.

    The dictionary recursion is only run while it stays cheap; beyond that the
    vectorized bitmask sweep carries the curve to |S| = 20.
    """
    lam = 0.5
    rows = []
    for n in range(5, max_n + 1):
        S, internal, e = random_beneficial_scc(n, lam, seed=42)

        t0 = time.perf_counter()
        J_bm, _, evals = solve_core_bitmask(S, internal, e, lam)
        t_bm = time.perf_counter() - t0

        t_dict = None
        agree = None
        if n <= dict_max_n:
            t0 = time.perf_counter()
            J_hk, _, _ = solve_core_heldkarp(S, internal, e, lam)
            t_dict = time.perf_counter() - t0
            agree = max(abs(J_bm[x] - J_hk[x]) for x in S) < 1e-9
        rows.append((n, evals, t_dict, t_bm, agree))

    assert all(a is not False for _, _, _, _, a in rows)

    ns = [r[0] for r in rows]
    t_bm = [r[3] for r in rows]
    nd = [r[0] for r in rows if r[2] is not None]
    td = [r[2] for r in rows if r[2] is not None]

    fig, ax = plt.subplots(figsize=(5.6, 3.4))
    ax.plot(nd, td, "s--", color="#8a8f94", ms=4.5, lw=1.4,
            label="dictionary recursion")
    ax.plot(ns, t_bm, "o-", color="#1f4e79", ms=4.5, lw=1.8,
            label="bitmask sweep")
    ref = [t_bm[-1] * 2.0 ** (n - ns[-1]) for n in ns]
    ax.plot(ns, ref, ":", color="#d1495b", lw=1.4, label=r"$\propto 2^{|S|}$")
    ax.set_xlabel(r"$|S|$")
    ax.set_ylabel("runtime (s)")
    ax.set_yscale("log")
    ax.set_xticks(ns[::2])
    ax.legend(fontsize=8.5, frameon=False, loc="upper left")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(True, alpha=0.25, lw=0.5)
    fig.tight_layout(pad=0.3)
    figdir = os.path.join(os.path.dirname(__file__), "..", "figures")
    os.makedirs(figdir, exist_ok=True)
    out_pdf = os.path.abspath(os.path.join(figdir, "solvecore_scaling.pdf"))
    fig.savefig(out_pdf, bbox_inches="tight", transparent=True)
    fig.savefig(out_pdf.replace(".pdf", ".png"), bbox_inches="tight", dpi=200)
    fig.savefig(os.path.join(RESULTS, "exp4_scaling.png"), dpi=150)
    plt.close(fig)

    with open(os.path.join(RESULTS, "exp4_scaling.csv"), "w") as fh:
        fh.write("n,subset_state_pairs,t_dict_s,t_bitmask_s,agree\n")
        for n, ev, td_, tb, ag in rows:
            fh.write(f"{n},{ev},{'' if td_ is None else f'{td_:.4f}'},"
                     f"{tb:.4f},{'' if ag is None else ag}\n")

    assert t_bm[-1] > t_bm[0] * 10
    return {"n_max": max_n, "time_max": t_bm[-1], "rows": rows}


def run() -> dict:
    os.makedirs(RESULTS, exist_ok=True)
    print("E4a: Held-Karp vs brute-force...")
    r4a = e4a_crosscheck()
    print("E4b: scaling...")
    r4b = e4b_scaling()

    lines = [
        "E4: SolveCore validation",
        f"  E4a: {r4a['tests']} point checks, mismatches={r4a['mismatches']}",
        f"  E4b: scaling to |S|={r4b['n_max']}, max time={r4b['time_max']:.4f}s",
        "    |S|   (state,subset) pairs   dict (s)   bitmask (s)   agree",
    ]
    for n, ev, td, tb, ag in r4b["rows"]:
        lines.append(
            f"    {n:<5d} {ev:<21d} {'-' if td is None else f'{td:8.4f}'}"
            f"   {tb:9.4f}   {'-' if ag is None else ag}"
        )
    lines.append("  ASSERTIONS PASSED")

    text = "\n".join(lines)
    out = os.path.join(RESULTS, "exp4_solvecore.txt")
    with open(out, "w") as f:
        f.write(text + "\n")
    print(text)
    return {"e4a": r4a, "e4b": r4b}


if __name__ == "__main__":
    run()
