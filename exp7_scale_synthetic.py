"""E7: the polynomial branch of Algorithm master on easy synthetic BCNs.

Proposition complexity says that when no component carries a beneficial cycle
the whole pipeline costs O(|E|) for the graph operations plus O(M |X_s| D),
where D is the largest number of policy-iteration steps spent on any single
component.  This experiment measures the running time against that bound on
random BCNs with n = 4..13 state nodes and m = 2 control nodes, three seeds
each, so that N = 2^n ranges over three orders of magnitude.

Every instance is checked to be easy, so the measured time is the polynomial
branch alone; the exponential branch is the subject of E4 and E12.
"""
from __future__ import annotations

import csv
import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import numpy as np  # noqa: E402

from bcn_generator import make_system_from_L, random_bcn  # noqa: E402
from core_solver import (  # noqa: E402
    augment_component,
    condense,
    detect_beneficial_cycle_v2,
    master,
    policy_iteration_exact,
    reverse_reachable,
    simulate_closed_loop,
    vi_unconstrained,
)

RESULTS = os.path.join(os.path.dirname(__file__), "results")
FIGURES = os.path.join(os.path.dirname(__file__), "..", "figures")


def run(n_max: int = 13, seeds: tuple[int, ...] = (0, 1, 2),
        lam: float = 0.5) -> dict:
    os.makedirs(RESULTS, exist_ok=True)
    rows: list[dict] = []
    carriers = 0

    for n0 in range(4, n_max + 1):
        for seed in seeds:
            n, m, N, M, L, xe, ue, f = random_bcn(n0, m=2, seed=seed,
                                                  acyclic=True)
            _n, m, N, M, L, xe, ue, f, g, G = make_system_from_L(
                L, n, m, xe, ue)
            R = reverse_reachable(G, xe)
            scc = condense(G, R, xe)
            max_scc = max((len(c) for c in scc.components), default=0)

            # D of Proposition complexity: the largest policy-iteration step
            # count on any one component, not the sum over components.
            max_steps = 0
            J_known = {xe: 0.0}
            for cid in scc.reverse_topo:
                S = scc.components[cid]
                if xe in S and len(S) == 1:
                    continue
                internal, e = augment_component(G, S, J_known, lam)
                flag, _, _ = detect_beneficial_cycle_v2(S, internal, e, lam)
                if flag:
                    carriers += 1
                    continue
                _V, _pol, steps = policy_iteration_exact(S, internal, e, lam)
                max_steps = max(max_steps, steps)
                for x in S:
                    J_known.setdefault(x, 0.0)

            t0 = time.perf_counter()
            J, mu = master(G, xe, lam, f=f, ue=ue)
            t_master = time.perf_counter() - t0

            t0 = time.perf_counter()
            V = vi_unconstrained(G, lam, R, xe)
            t_vi = time.perf_counter() - t0

            improper = sum(
                1 for x in R
                if not simulate_closed_loop(f, mu, x, xe,
                                            max_steps=4 * len(R))[0]
            )
            gap = sum(1 for x in R if V[x] + 1e-9 < J[x])

            rows.append(
                {
                    "n": n,
                    "N": N,
                    "M": M,
                    "seed": seed,
                    "lambda": lam,
                    "|X_s|": len(R),
                    "|E|": G.num_arcs(),
                    "scc_count": len(scc.components),
                    "max_scc": max_scc,
                    "D_steps": max_steps,
                    "bound": G.num_arcs() + M * len(R) * max_steps,
                    "t_master_s": round(t_master, 5),
                    "t_vi_s": round(t_vi, 5),
                    "improper": improper,
                    "gap_states": gap,
                }
            )

    assert carriers == 0, f"{carriers} components carry a beneficial cycle"
    assert all(r["improper"] == 0 for r in rows), "master returned improper"
    assert all(r["gap_states"] == 0 for r in rows), "V != J on an easy instance"

    out_csv = os.path.join(RESULTS, "exp7_scale.csv")
    with open(out_csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    fit = _figure(rows)

    text = "\n".join(
        [
            "E7: polynomial branch on easy synthetic BCNs",
            f"  n=4..{n_max}, m=2, seeds={seeds}, lambda={lam}",
            f"  {len(rows)} instances, all easy, all feedbacks proper,"
            f" V=J everywhere",
            f"  N from {min(r['N'] for r in rows)} to"
            f" {max(r['N'] for r in rows)}",
            f"  bound |E|+M|X_s|D from {min(r['bound'] for r in rows):,} to"
            f" {max(r['bound'] for r in rows):,}",
            f"  runtime from {min(r['t_master_s'] for r in rows):.4f}s to"
            f" {max(r['t_master_s'] for r in rows):.4f}s",
            f"  log-log slope of runtime against the bound: {fit:.2f}",
            "  ASSERTIONS PASSED",
        ]
    )
    with open(os.path.join(RESULTS, "exp7_scale.txt"), "w") as fh:
        fh.write(text + "\n")
    print(text)
    return {"rows": len(rows), "csv": out_csv, "slope": fit}


def _figure(rows: list[dict]) -> float:
    """Runtime against |E| + M|X_s|D, with the least-squares log-log slope."""
    by_n: dict[int, list[tuple[int, float]]] = {}
    for r in rows:
        by_n.setdefault(r["N"], []).append((r["bound"], r["t_master_s"]))

    xs = np.array([r["bound"] for r in rows], dtype=float)
    ys = np.array([r["t_master_s"] for r in rows], dtype=float)
    slope, _intercept = np.polyfit(np.log(xs), np.log(ys), 1)

    Ns = sorted(by_n)
    mx = [float(np.mean([b for b, _ in by_n[N]])) for N in Ns]
    my = [float(np.mean([t for _, t in by_n[N]])) for N in Ns]

    # A slope-one line through the centroid of the data in log-log coordinates.
    order = np.argsort(xs)
    ref = np.exp(float(np.mean(np.log(ys) - np.log(xs)))) * xs

    fig, ax = plt.subplots(figsize=(5.6, 3.4))
    ax.plot(xs, ys, "o", color="#1f4e79", ms=4.0, alpha=0.55,
            label="individual instances")
    ax.plot(mx, my, "-", color="#1f4e79", lw=1.8, label="mean over seeds")
    ax.plot(xs[order], ref[order], ":", color="#d1495b", lw=1.4,
            label="linear reference")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$|E| + M\,|X_s|\,D$")
    ax.set_ylabel("runtime (s)")
    ax.legend(fontsize=8.5, frameon=False, loc="upper left")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(True, alpha=0.25, lw=0.5, which="both")
    fig.tight_layout(pad=0.3)

    os.makedirs(FIGURES, exist_ok=True)
    out_pdf = os.path.abspath(os.path.join(FIGURES, "easy_scaling.pdf"))
    fig.savefig(out_pdf, bbox_inches="tight", transparent=True)
    fig.savefig(out_pdf.replace(".pdf", ".png"), bbox_inches="tight", dpi=200)
    fig.savefig(os.path.join(RESULTS, "exp7_scale.png"), dpi=150)
    plt.close(fig)
    return float(slope)


if __name__ == "__main__":
    run()
