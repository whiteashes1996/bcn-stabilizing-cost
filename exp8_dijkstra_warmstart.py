"""E8: Dijkstra warm-start acceleration (LC init + SolveCore bounding)."""
from __future__ import annotations

import csv
import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ara_io import ARA_PATH, hamming_cost_factory, next_state, read_network
from bcn_generator import make_system_from_L, random_bcn, random_bcn_with_beneficial_cycle
from core_solver import (
    augment_component,
    build_graph,
    discounted_cost_sp_tree,
    local_lc,
    local_lc_warm,
    random_beneficial_scc,
    solve_core_heldkarp,
    solve_core_heldkarp_bounded,
    undiscounted_sp_to_omega,
)

RESULTS = os.path.join(os.path.dirname(__file__), "results")


def path_a_lc_warmstart() -> list[dict]:
    """Label-correcting: cold J=e vs warm J=undiscounted Dijkstra."""
    rows: list[dict] = []
    cases: list[tuple[str, object]] = [("ara_operon", ARA_PATH)]

    for n in (8, 9, 10, 11, 12):
        cases.append((f"synthetic_n{n}", ("syn", n)))

    lam = 0.5
    for name, spec in cases:
        if name == "ara_operon":
            n, m, N, M, L = read_network(ARA_PATH)
            xe, ue = 512, 11
            g = hamming_cost_factory(xe, ue, n)
            f = lambda x, u: next_state(L, N, x, u)
            G = build_graph(f, g, list(range(1, N + 1)), list(range(1, M + 1)), xe=xe)
            from core_solver import condense, reverse_reachable

            R = reverse_reachable(G, xe)
            scc = condense(G, R, xe)
            components = scc.components
            cid_order = scc.reverse_topo
        else:
            _, n = spec
            n, m, N, M, L, xe, ue, f = random_bcn(n, m=2, seed=42, acyclic=True)
            _n, m, N, M, L, xe, ue, f, g, G = make_system_from_L(L, n, m, xe, ue)
            from core_solver import condense, reverse_reachable

            R = reverse_reachable(G, xe)
            scc = condense(G, R, xe)
            components = scc.components
            cid_order = scc.reverse_topo

        J_known = {xe: 0.0}
        cold_sw = warm_sw = 0
        t_cold = t_warm = 0.0
        for cid in cid_order:
            S = components[cid]
            if xe in S and len(S) == 1:
                continue
            internal, e = augment_component(G, S, J_known, lam)
            if not internal and all(v >= 1e30 for v in e.values()):
                continue
            J_u = undiscounted_sp_to_omega(S, internal, e)
            J_warm_init = discounted_cost_sp_tree(S, internal, e, lam, J_u)

            t0 = time.perf_counter()
            Jc, _, swc = local_lc(S, internal, e, lam)
            t_cold += time.perf_counter() - t0
            cold_sw += swc

            t0 = time.perf_counter()
            Jw, _, sww = local_lc_warm(S, internal, e, lam, J_warm_init)
            t_warm += time.perf_counter() - t0
            warm_sw += sww

            for x in S:
                if abs(Jc.get(x, 0) - Jw.get(x, 0)) > 1e-6:
                    raise AssertionError(f"LC mismatch {name} x={x}: {Jc.get(x)} vs {Jw.get(x)}")
            J_known.update(Jc)

        speedup = t_cold / t_warm if t_warm > 0 else float("inf")
        rows.append(
            {
                "case": name,
                "N": N,
                "path": "LC_warmstart",
                "t_cold_s": round(t_cold, 5),
                "t_warm_s": round(t_warm, 5),
                "speedup": round(speedup, 3),
                "sweeps_cold": cold_sw,
                "sweeps_warm": warm_sw,
            }
        )
    return rows


def path_b_core_bound() -> list[dict]:
    """SolveCore: bare Held-Karp vs Dijkstra upper-bound pruning."""
    rows: list[dict] = []
    lam = 0.5
    for n in range(8, 17):
        S, internal, e = random_beneficial_scc(n, lam, seed=42)
        J_u = undiscounted_sp_to_omega(S, internal, e)
        ub = discounted_cost_sp_tree(S, internal, e, lam, J_u)

        t0 = time.perf_counter()
        Jc, _, ec = solve_core_heldkarp(S, internal, e, lam)
        t_cold = time.perf_counter() - t0

        t0 = time.perf_counter()
        Jw, _, ew = solve_core_heldkarp_bounded(S, internal, e, lam, ub)
        t_warm = time.perf_counter() - t0

        for x in S:
            if abs(Jc[x] - Jw[x]) > 1e-6:
                raise AssertionError(f"HK mismatch |S|={n} x={x}")

        rows.append(
            {
                "case": f"beneficial_S{n}",
                "N": n,
                "path": "SolveCore_bound",
                "t_cold_s": round(t_cold, 5),
                "t_warm_s": round(t_warm, 5),
                "speedup": round(t_cold / t_warm if t_warm > 0 else 0, 3),
                "eval_cold": ec,
                "eval_warm": ew,
            }
        )
    return rows


def run() -> dict:
    os.makedirs(RESULTS, exist_ok=True)
    rows = path_a_lc_warmstart() + path_b_core_bound()

    out_csv = os.path.join(RESULTS, "exp8_warmstart.csv")
    fieldnames = [
        "case",
        "N",
        "path",
        "t_cold_s",
        "t_warm_s",
        "speedup",
        "sweeps_cold",
        "sweeps_warm",
        "eval_cold",
        "eval_warm",
    ]
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    lc_rows = [r for r in rows if r["path"] == "LC_warmstart"]
    fig, ax = plt.subplots(figsize=(7, 4))
    x = range(len(lc_rows))
    ax.bar([i - 0.15 for i in x], [r["t_cold_s"] for r in lc_rows], width=0.3, label="cold")
    ax.bar([i + 0.15 for i in x], [r["t_warm_s"] for r in lc_rows], width=0.3, label="Dijkstra warm")
    ax.set_xticks(list(x))
    ax.set_xticklabels([r["case"] for r in lc_rows], rotation=30, ha="right")
    ax.set_ylabel("LC time (s)")
    ax.set_title("Path A: label-correcting warm start")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS, "exp8_warmstart_lc.png"), dpi=150)
    plt.close(fig)

    hk_rows = [r for r in rows if r["path"] == "SolveCore_bound"]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot([r["N"] for r in hk_rows], [r["t_cold_s"] for r in hk_rows], "o-", label="Held-Karp")
    ax.plot([r["N"] for r in hk_rows], [r["t_warm_s"] for r in hk_rows], "s-", label="bounded")
    ax.set_yscale("log")
    ax.set_xlabel("|S|")
    ax.set_ylabel("time (s)")
    ax.set_title("Path B: SolveCore with Dijkstra UB")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS, "exp8_warmstart_hk.png"), dpi=150)
    plt.close(fig)

    lines = ["E8: Dijkstra warm-start", f"  rows: {len(rows)}"]
    for r in rows:
        lines.append(
            f"  {r['case']} {r['path']}: cold={r['t_cold_s']}s warm={r['t_warm_s']}s speedup={r['speedup']}"
        )
    lines.append("  ASSERTIONS PASSED")
    text = "\n".join(lines)
    with open(os.path.join(RESULTS, "exp8_warmstart.txt"), "w") as f:
        f.write(text + "\n")
    print(text)
    return {"rows": len(rows)}


if __name__ == "__main__":
    run()
