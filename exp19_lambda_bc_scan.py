"""Natural-cost BC count vs lambda for the six literature networks (+ restricted).

Detection only: fold + detect_via_pi / solve_easy; skip SolveCore when |S| > CORE_CAP
so large kernels at small lambda remain countable. Improper counts use VI greedy
only when N <= 1024 or lambda in DETAIL_LAMBDAS for T-LGL.
"""
from __future__ import annotations

import csv
import os
import sys
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(__file__))

import bcn_data as BD  # noqa: E402
import methods as MM  # noqa: E402
from exp16_ex2_real import _mk_row_config, vi_until  # noqa: E402

RESULTS = os.path.join(os.path.dirname(__file__), "results")
LAMBDAS = [0.1, 0.25, 0.5, 0.75, 0.9]
CORE_CAP = 20
DETAIL_LAMBDAS = {0.5}


def detect_bc(
    G: MM.STG, xe: int, lam: float, ctx: MM.NumericContext, ue: int
) -> Tuple[int, List[int], int]:
    """Return (bc_count, bc_sizes, n_large_skipped_cores)."""
    R = MM.reverse_reachable(G, xe)
    if len(R) <= 1:
        return 0, [], 0
    Gr = G.restrict(R)
    J: Dict[int, float] = {xe: ctx.num(0)}
    cond = MM.condense(Gr, R, xe)
    bc_sizes: List[int] = []
    skipped = 0
    for cid in cond.reverse_topo:
        S = set(cond.components[cid])
        if xe in S:
            if len(S) == 1:
                continue
            S, internal, e = MM.fold_terminal(Gr, S, xe, lam, ctx)
        else:
            internal, e = MM.fold_component(Gr, S, J, lam, ctx)
        det = MM.detect_via_pi(S, internal, e, lam, ctx)
        if det.has_cycle:
            bc_sizes.append(len(S))
            if len(S) > CORE_CAP:
                skipped += 1
                for x, v in det.V.items():
                    if not ctx.is_inf(v):
                        J[x] = v
                continue
            core = MM.solve_core(S, internal, e, lam, ctx)
            for x, v in core.J.items():
                if not ctx.is_inf(v):
                    J[x] = v
        else:
            for x, v in det.V.items():
                if not ctx.is_inf(v):
                    J[x] = v
    return len(bc_sizes), sorted(bc_sizes), skipped


def improper_vi(
    G: MM.STG, xe: int, lam: float, ctx: MM.NumericContext, ue: int, spec
) -> int:
    R = MM.reverse_reachable(G, xe)
    seq, _k = vi_until(G, R, xe, lam, ctx)
    adjR = MM.restricted_adjacency(G, R)
    mu = MM.greedy_mu(seq[-1], adjR, xe, lam, ctx, spec.next_state, ue=ue)
    return len(MM.improper_states(spec.next_state, mu, R, xe))


def main() -> None:
    os.makedirs(RESULTS, exist_ok=True)
    ctx = MM.floating(1e-9)
    rows_out: List[dict] = []
    configs = _mk_row_config()

    for cfg in configs:
        spec, xe, ue = cfg["spec"], cfg["xe"], cfg["ue"]
        Cu_t = cfg["Cu"]
        Cu = (lambda _x, _a=Cu_t: _a) if Cu_t else None
        g = BD.hamming_cost(spec, xe, ue)
        G = BD.make_stg(spec, g, ctx, Cu=Cu)
        print(f"=== {cfg['key']}  N={spec.N} ===", flush=True)
        for lam in LAMBDAS:
            bc, sizes, skipped = detect_bc(G, xe, lam, ctx, ue)
            do_imp = spec.N <= 1024 or lam in DETAIL_LAMBDAS
            imp: Optional[int] = None
            if do_imp:
                try:
                    imp = improper_vi(G, xe, lam, ctx, ue, spec)
                except Exception as exc:  # noqa: BLE001
                    print(f"  lam={lam}: improper failed: {exc}", flush=True)
                    imp = None
            rec = {
                "network": cfg["key"],
                "N": spec.N,
                "lambda": lam,
                "BC": bc,
                "bc_sizes": ";".join(map(str, sizes)),
                "cores_skipped_gt20": skipped,
                "improper_VI": imp if imp is not None else "",
            }
            rows_out.append(rec)
            print(
                f"  lam={lam}: BC={bc} sizes={sizes} skipped={skipped} "
                f"imp_VI={imp}",
                flush=True,
            )

    out = os.path.join(RESULTS, "exp19_lambda_bc_scan.csv")
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows_out[0].keys()))
        w.writeheader()
        w.writerows(rows_out)
    print(f"wrote {out} ({len(rows_out)} rows)", flush=True)


if __name__ == "__main__":
    main()
