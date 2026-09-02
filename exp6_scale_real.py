"""E6: Real BCN networks scale scan (ara operon + lambda switch)."""
from __future__ import annotations

import csv
import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))

from ara_io import (
    LAMBDA_SWITCH_PATH,
    ARA_PATH,
    control_fixpoints,
    hamming_cost_factory,
    next_state,
    read_network,
    reverse_reachable_states,
)
from core_solver import (
    augment_component,
    build_graph,
    condense,
    detect_beneficial_cycle,
    master,
    reverse_reachable,
    vi_unconstrained,
)

RESULTS = os.path.join(os.path.dirname(__file__), "results")


def _scan_network(name: str, path: str, lambdas: tuple[float, ...]) -> list[dict]:
    n, m, N, M, L = read_network(path)
    fps = control_fixpoints(L, N, M)
    rows: list[dict] = []

    rev = [set() for _ in range(N + 1)]
    for x in range(1, N + 1):
        for u in range(1, M + 1):
            rev[next_state(L, N, x, u)].add(x)

    for xe, us in sorted(fps.items()):
        ue = us[0]
        g = hamming_cost_factory(xe, ue, n)
        f = lambda x, u, _L=L, _N=N: next_state(_L, _N, x, u)
        G = build_graph(f, g, list(range(1, N + 1)), list(range(1, M + 1)), xe=xe)
        R = reverse_reachable(G, xe)
        _, D_reach = reverse_reachable_states(rev, xe)
        scc = condense(G, R, xe)
        max_scc = max(len(c) for c in scc.components) if scc.components else 0
        bc_count = 0
        for cid in scc.reverse_topo:
            S = scc.components[cid]
            if len(S) <= 1 or xe in S:
                continue
            internal, e = augment_component(G, S, {xe: 0.0}, 0.5)
            if detect_beneficial_cycle(S, internal, e, 0.5):
                bc_count += 1

        for lam in lambdas:
            t0 = time.perf_counter()
            J, _ = master(G, xe, lam, f=f, ue=ue)
            t_master = time.perf_counter() - t0

            t0 = time.perf_counter()
            V = vi_unconstrained(G, lam, R, xe)
            t_vi = time.perf_counter() - t0
            gap = sum(1 for x in R if abs(V[x] - J[x]) > 1e-6)

            rows.append(
                {
                    "network": name,
                    "n": n,
                    "N": N,
                    "M": M,
                    "xe": xe,
                    "ue": ue,
                    "lambda": lam,
                    "|R|": len(R),
                    "D_reach": D_reach,
                    "scc_count": len(scc.components),
                    "max_scc": max_scc,
                    "bc_scc_count": bc_count,
                    "gap_states": gap,
                    "t_master_s": round(t_master, 4),
                    "t_vi_s": round(t_vi, 4),
                }
            )
    return rows


def run() -> dict:
    os.makedirs(RESULTS, exist_ok=True)
    rows = []
    rows.extend(_scan_network("ara_operon", ARA_PATH, (0.5, 0.9)))
    rows.extend(_scan_network("lambda_switch", LAMBDA_SWITCH_PATH, (0.5, 0.9)))

    out = os.path.join(RESULTS, "exp6_real_networks.csv")
    if rows:
        with open(out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

    ara512 = [r for r in rows if r["network"] == "ara_operon" and r["xe"] == 512 and r["lambda"] == 0.5]
    if ara512:
        assert ara512[0]["|R|"] == 512
        assert ara512[0]["D_reach"] == 3

    lines = [
        "E6: Real BCN network scan",
        f"  networks: ara_operon (N=512), lambda_switch (N=32)",
        f"  rows: {len(rows)}",
    ]
    for r in rows[:4]:
        lines.append(
            f"  {r['network']} xe={r['xe']} lam={r['lambda']}: |R|={r['|R|']}, "
            f"t_master={r['t_master_s']}s, t_vi={r['t_vi_s']}s, gap={r['gap_states']}"
        )
    lines.append("  ASSERTIONS PASSED")
    text = "\n".join(lines)
    with open(os.path.join(RESULTS, "exp6_real_networks.txt"), "w") as f:
        f.write(text + "\n")
    print(text)
    return {"rows": len(rows), "csv": out}


if __name__ == "__main__":
    run()
