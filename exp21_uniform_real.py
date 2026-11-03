"""E21: uniform-optimality certificates on the benchmark networks.

For every beneficial-cycle component S of each configuration this reports
  |S|, joint optimality (folded tight-arc graph spans S), s*(S), h*(S),
  the excess of the reverse-BFS tree T_0 of G_eps(s*) (an upper bound on
  Exc*(S)), and globally |gap set| (V* < J*), |union S_B|, and the realized
  uniform excess of the composed feedback (tight trees on benign components,
  T_0 on beneficial-cycle components).

Components beyond the Held-Karp cap are solved by the time-expanded MILP of
methods.milp_oracle (per-source CBC time limit below); uncertified sources
are reported as coverage < 1.

Run:  python simulations/exp21_uniform_real.py small
      python simulations/exp21_uniform_real.py large   (|S| = 105, 107; slow)
"""
from __future__ import annotations

import csv
import os
import sys
import time
from collections import defaultdict, deque
from typing import Dict, List, Optional, Set, Tuple

sys.path.insert(0, os.path.dirname(__file__))

import bcn_data as BD  # noqa: E402
import exp16_ex2_real as E16  # noqa: E402
import methods as MM  # noqa: E402

RESULTS = os.path.join(os.path.dirname(__file__), "results")
INF = MM.INF
LAM = 0.5
CORE_CAP = 20
TOL = 1e-7


def _scale(v: float) -> float:
    return max(1.0, abs(v))


def walk_with_milp(G, xe, ctx, ue, milp_limit: float) -> dict:
    """pipeline_walk, but beneficial-cycle components above the Held-Karp cap
    are handed to the MILP oracle; a component is solved iff every source is
    certified."""
    lam = ctx.lam(LAM)
    R = MM.reverse_reachable(G, xe)
    Gr = G.restrict(R)
    J: Dict[int, float] = {xe: ctx.num(0)}
    cond = MM.condense(Gr, R, xe)
    comps: List[dict] = []
    complete = True
    for cid in cond.reverse_topo:
        S0 = set(cond.components[cid])
        if xe in S0:
            if len(S0) == 1:
                continue
            S, internal, e = MM.fold_terminal(Gr, S0, xe, lam, ctx)
        else:
            S = S0
            internal, e = MM.fold_component(Gr, S, J, lam, ctx)
        det = MM.detect_via_pi(S, internal, e, lam, ctx)
        verdict = det.has_cycle
        solved, how, t_core, coverage = True, "easy", 0.0, 1.0
        if verdict:
            t0 = time.perf_counter()
            if len(S) <= CORE_CAP:
                core = MM.solve_core(S, internal, e, lam, ctx, max_core=CORE_CAP)
                Jc = core.J
                how = "held-karp"
            else:
                MM.MILP_TIME_LIMIT_S = milp_limit
                Jc = MM.milp_oracle(S, internal, e, lam, ctx)
                how = "milp"
                coverage = len(Jc) / len(S)
                if len(Jc) < len(S):
                    solved = False
                    complete = False
            t_core = time.perf_counter() - t0
            for x, v in Jc.items():
                if not ctx.is_inf(v):
                    J[x] = v
        else:
            for x, v in det.V.items():
                if not ctx.is_inf(v):
                    J[x] = v
        comps.append({"cid": cid, "S": S, "internal": internal, "e": e,
                      "verdict": verdict, "solved": solved, "how": how,
                      "t_core": t_core, "coverage": coverage})
    return {"R": R, "Gr": Gr, "cond": cond, "J": J, "comps": comps, "complete": complete}


def component_certificate(S, internal, e, J, lam) -> dict:
    """Tight-arc test, s*, h*, and the reverse-BFS tree T_0 of G_eps(s*)."""
    arcs: List[Tuple[float, int, Optional[int], float, object]] = []
    for x in S:
        ex = e.get(x, INF)
        if ex < 1e300:
            arcs.append((float(ex) - float(J[x]), x, None, float(ex), None))
        for u, y, w in internal.get(x, []):
            arcs.append((float(w) + lam * float(J[y]) - float(J[x]), x, y, float(w), u))

    def bfs(theta: float):
        pred: Dict[Optional[int], List[Tuple[int, tuple]]] = defaultdict(list)
        for eps, x, y, w, u in arcs:
            if eps <= theta + TOL * _scale(float(J[x])):
                pred[y].append((x, (y, w, u)))
        depth: Dict[int, int] = {}
        tree: Dict[int, tuple] = {}
        dq: deque = deque()
        for x, act in pred.get(None, []):
            if x not in depth:
                depth[x] = 1
                tree[x] = act
                dq.append(x)
        while dq:
            y = dq.popleft()
            for x, act in pred.get(y, []):
                if x not in depth:
                    depth[x] = depth[y] + 1
                    tree[x] = act
                    dq.append(x)
        return len(depth) == len(S), (max(depth.values()) if depth else 0), tree

    joint, _h0, _t0 = bfs(0.0)
    thetas = sorted({max(0.0, a[0]) for a in arcs})
    lo, hi = 0, len(thetas) - 1
    while lo < hi:
        mid = (lo + hi) // 2
        if bfs(thetas[mid])[0]:
            hi = mid
        else:
            lo = mid + 1
    sstar = thetas[lo]
    ok, hstar, tree = bfs(sstar)
    assert ok
    cost: Dict[int, float] = {}

    def c(x: int) -> float:
        if x in cost:
            return cost[x]
        stack, cur = [], x
        while cur not in cost:
            stack.append(cur)
            y, _w, _u = tree[cur]
            if y is None:
                break
            cur = y
        for v in reversed(stack):
            y, w, _u = tree[v]
            cost[v] = w if y is None else w + lam * cost[y]
        return cost[x]

    tree_exc = max(c(x) - float(J[x]) for x in S)
    ub = sstar * (1 - lam ** hstar) / (1 - lam)
    return {"joint": joint, "sstar": sstar, "hstar": hstar, "tree_exc": max(0.0, tree_exc),
            "ub": ub, "tree": tree}


def configs(which: str) -> List[dict]:
    rows = {c["key"]: c for c in E16._mk_row_config()}
    tlgl, lac, ara, mcc = (rows["t_lgl"]["spec"], rows["lac_operon"]["spec"],
                           rows["ara_operon"]["spec"], rows["mammalian_cc"]["spec"])
    xe_lac, ue_lac = rows["lac_operon"]["xe"], rows["lac_operon"]["ue"]
    xe_m = rows["mammalian_cc"]["xe"]
    ue_t = BD.input_index(tlgl, (1, 0, 1))
    u_d_lac = BD.input_index(lac, (0, 1, 1))
    fps_lac = [x for x, us in BD.control_fixpoints(lac).items() if u_d_lac in us]
    x_d_lac = [x for x in fps_lac if BD.bits_of(lac, x)[:3] == (1, 1, 1)][0]
    out: List[dict] = []
    if which == "small":
        out.append(dict(tag="T-LGL (natural cost)", spec=tlgl, xe=65535, ue=ue_t,
                        Cu=None, decoy=None))
        out.append(dict(tag="T-LGL (decoy)", spec=tlgl, xe=65535, ue=ue_t, Cu=None,
                        decoy={(58834, ue_t)}))
        import gen_networks as GN
        att1 = GN.synchronous_attractors(lambda b: GN.mcc_step(b, (1,)), 9)
        cyc = {BD.state_index(mcc, b) for b in att1[0]}
        out.append(dict(tag="mammalian cell cycle (decoy)", spec=mcc, xe=xe_m, ue=2,
                        Cu=None, decoy={(x, 1) for x in cyc}))
        out.append(dict(tag="ara, restricted inputs (decoy)", spec=ara, xe=512, ue=11,
                        Cu=(2, 11), decoy={(9, 2)}))
        out.append(dict(tag="lac, restricted inputs (decoy)", spec=lac, xe=xe_lac, ue=ue_lac,
                        Cu=(ue_lac, u_d_lac), decoy={(x_d_lac, u_d_lac)}))
    else:
        out.append(dict(tag="lac, all inputs (decoy)", spec=lac, xe=xe_lac, ue=ue_lac,
                        Cu=None, decoy={(x_d_lac, u_d_lac)}))
        out.append(dict(tag="ara, all inputs (decoy)", spec=ara, xe=512, ue=11,
                        Cu=None, decoy={(9, 2)}))
    return out


def realized_excess(walk: dict, certs: Dict[int, dict], xe: int, lam) -> float:
    """Uniform excess of the composed feedback: tight trees on benign
    components (Lemma recover-tree), T_0 on beneficial-cycle components."""
    J = walk["J"]
    Gr = walk["Gr"]
    V: Dict[int, float] = {xe: 0.0}
    for comp in walk["comps"]:
        S, internal, e = comp["S"], comp["internal"], comp["e"]
        if comp["cid"] in certs:
            tree = certs[comp["cid"]]["tree"]
        else:
            tree = component_certificate(S, internal, e, J, lam)["tree"]
        # exit actions: argmin of g + lam*V(b) over arcs leaving S
        order: List[int] = []
        depth = {}
        children = defaultdict(list)
        roots = []
        for x, (y, _w, _u) in tree.items():
            if y is None:
                roots.append(x)
            else:
                children[y].append(x)
        dq = deque(roots)
        while dq:
            v = dq.popleft()
            order.append(v)
            dq.extend(children[v])
        for x in order:
            y, w, _u = tree[x]
            if y is None:
                best = INF
                for _u2, b, g in Gr.out.get(x, []):
                    if b in S or b not in V:
                        continue
                    cand = float(g) + lam * V[b]
                    if cand < best:
                        best = cand
                V[x] = best
            else:
                V[x] = float(w) + lam * V[y]
    exc = 0.0
    for x, jv in J.items():
        if x in V:
            exc = max(exc, V[x] - float(jv))
    return exc


def run_config(cfg: dict, milp_limit: float) -> Tuple[List[dict], dict]:
    ctx = MM.floating(1e-9)
    spec = cfg["spec"]
    base = BD.hamming_cost(spec, cfg["xe"], cfg["ue"])
    g = BD.decoy_cost(base, cfg["decoy"], E16.DECOY_EPS) if cfg["decoy"] else base
    Cu = (lambda _x, _a=cfg["Cu"]: _a) if cfg["Cu"] else None
    G = BD.make_stg(spec, g, ctx, Cu=Cu)
    t0 = time.perf_counter()
    walk = walk_with_milp(G, cfg["xe"], ctx, cfg["ue"], milp_limit)
    t_walk = time.perf_counter() - t0
    lam = float(LAM)
    R, J = walk["R"], walk["J"]
    V_pi, _ = MM.pi(G, R, cfg["xe"], LAM, ctx)
    gap = [x for x in R if x in J and V_pi.get(x, INF) < float(J[x]) - 1e-6 * _scale(float(J[x]))]
    rows: List[dict] = []
    certs: Dict[int, dict] = {}
    union_b = 0
    for comp in walk["comps"]:
        if not comp["verdict"]:
            continue
        union_b += len(comp["S"])
        row = {"config": cfg["tag"], "|S|": len(comp["S"]), "solver": comp["how"],
               "coverage": round(comp["coverage"], 3), "t_core_s": round(comp["t_core"], 3)}
        if comp["solved"]:
            cert = component_certificate(comp["S"], comp["internal"], comp["e"], J, lam)
            certs[comp["cid"]] = cert
            row.update({"joint": cert["joint"], "s*": round(cert["sstar"], 6),
                        "h*": cert["hstar"], "T0_excess": round(cert["tree_exc"], 6),
                        "sandwich_ub": round(cert["ub"], 6)})
        else:
            row.update({"joint": "n/a", "s*": "n/a", "h*": "n/a", "T0_excess": "n/a",
                        "sandwich_ub": "n/a"})
        rows.append(row)
    summary = {"config": cfg["tag"], "|V_s|": len(R), "gap": len(gap), "union_SB": union_b,
               "complete": walk["complete"], "t_walk_s": round(t_walk, 2)}
    if walk["complete"]:
        summary["realized_excess"] = round(realized_excess(walk, certs, cfg["xe"], lam), 6)
        summary["max_s*"] = round(max([c["sstar"] for c in certs.values()] or [0.0]), 6)
        summary["joint_global"] = all(c["joint"] for c in certs.values())
    return rows, summary


def main() -> int:
    which = sys.argv[1] if len(sys.argv) > 1 else "small"
    milp_limit = float(sys.argv[2]) if len(sys.argv) > 2 else 60.0
    os.makedirs(RESULTS, exist_ok=True)
    all_rows: List[dict] = []
    summaries: List[dict] = []
    for cfg in configs(which):
        print(f"[{cfg['tag']}] ...", flush=True)
        rows, summ = run_config(cfg, milp_limit)
        for r in rows:
            print("   ", r, flush=True)
        print("   ", summ, flush=True)
        all_rows.extend(rows)
        summaries.append(summ)
    with open(os.path.join(RESULTS, f"exp21_uniform_real_{which}.csv"), "w", newline="") as fh:
        if all_rows:
            w = csv.DictWriter(fh, fieldnames=list(all_rows[0].keys()))
            w.writeheader()
            w.writerows(all_rows)
    with open(os.path.join(RESULTS, f"exp21_uniform_real_{which}.txt"), "w") as fh:
        fh.write(f"E21 ({which}): lambda={LAM}, floating(1e-9), MILP limit {milp_limit}s/source\n")
        for r in all_rows:
            fh.write(f"  {r}\n")
        for s in summaries:
            fh.write(f"  SUMMARY {s}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
