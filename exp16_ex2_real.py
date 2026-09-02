"""E16: Example 2 real-network study (Step 2 plan §2-§6, acceptance §6.A).

Six real BCNs (ara operon, lambda switch, apoptosis, mammalian cell cycle,
lac operon, T-LGL) plus two restricted-actuation variants:

  Module A  survey table (8 rows): |V_s|, D_reach, #SCC, max|SCC|, BC(g),
            t_master, t_VI(eps) under the natural Hamming cost, lambda = 1/2;
  Module B  baseline protocol: VI stopping rule (eps_VI = 1e-6, Bellman
            residual <= eps_VI (1-lam) / (2 lam)), PI lower-bound anchor,
            Floyd (N <= 512), improper sets via closed-loop simulation;
  Module C  decoy cases A (T-LGL), B (lac), C (mammalian cc, enrichment),
            plus the ara restricted-actuation decoy (MILP flagship target);
  Module D  MILP oracle cross-checks (methods.milp_oracle vs solve_core vs
            component-level DFS) with certification coverage;
  Module E  timing protocol: warmup 1 + median of 5 runs (3 for N = 65536);
  Module F  outputs results/exp16_*.{csv,txt} and asserts A1-A10.

All numerics go through methods.py (floating(1e-9) primary; exact() recheck
for decoy components on networks with N <= 1024, per plan §4.5).
"""
from __future__ import annotations

import csv
import math
import os
import platform
import statistics
import sys
import time
from fractions import Fraction
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple

sys.path.insert(0, os.path.dirname(__file__))

import bcn_data as BD  # noqa: E402
import methods as MM  # noqa: E402

RESULTS = os.path.join(os.path.dirname(__file__), "results")
INF = MM.INF

EPS_VI = 1e-6
K_CAP = 200
LAM = 0.5
LAM2 = 0.9
DECOY_EPS = 0.05
CORE_CAP = 20
MILP_CAP = 60
DFS_CAP = 14

passed: List[str] = []
notes: List[str] = []


def check(tag: str, cond: bool, detail: str = "") -> None:
    assert cond, f"{tag} FAILED  {detail}"
    passed.append(f"{tag} PASS  {detail}".rstrip())


def note(msg: str) -> None:
    notes.append(msg)
    print("  [note]", msg)


# ---------------------------------------------------------------------------
# Module E: timing protocol (plan §4.5)
# ---------------------------------------------------------------------------

def timed(fn: Callable[[], object], reps: int, warmup: int = 1) -> Tuple[float, List[float], object]:
    out = None
    for _ in range(warmup):
        out = fn()
    times: List[float] = []
    for _ in range(reps):
        t0 = time.perf_counter()
        out = fn()
        times.append(time.perf_counter() - t0)
    return statistics.median(times), times, out


# ---------------------------------------------------------------------------
# Module B: baseline protocol
# ---------------------------------------------------------------------------

def k_bound(g_max: float, lam: float, eps: float) -> int:
    return math.ceil(math.log(eps * (1 - lam) / max(g_max, 1e-12)) / math.log(lam)) + 5


def _residual(prev: Dict[int, float], cur: Dict[int, float], R: Set[int]) -> float:
    worst = 0.0
    for x in R:
        a, b = prev.get(x, INF), cur.get(x, INF)
        ia, ib = a >= 1e300, b >= 1e300
        if ia and ib:
            continue
        if ia != ib:
            return INF
        worst = max(worst, abs(a - b))
    return worst


def vi_until(G: MM.STG, R: Set[int], xe: int, lam: float, ctx: MM.NumericContext,
             eps_vi: float = EPS_VI, k_cap: int = K_CAP) -> Tuple[List[Dict[int, float]], int]:
    """VI sequence up to the declared stopping rule; returns (seq[0..k], k)."""
    g_max = max(float(w) for _x, _u, _y, w in G.arcs) if G.arcs else 1.0
    kb = min(k_cap, max(4, k_bound(g_max, lam, eps_vi)))
    thr = eps_vi * (1 - lam) / (2 * lam)
    while True:
        seq = MM.vi(G, R, xe, lam, ctx, kb)
        for k in range(1, len(seq)):
            if _residual(seq[k - 1], seq[k], R) <= thr:
                return seq[: k + 1], k
        if kb >= k_cap:
            raise RuntimeError(f"VI did not meet the stopping rule within {k_cap} sweeps")
        kb = min(k_cap, kb * 2)


def vi_monotone_and_anchored(seq: List[Dict[int, float]], V_pi: Dict[int, float],
                             R: Set[int]) -> bool:
    for k in range(1, len(seq)):
        for x in R:
            a, b = seq[k - 1].get(x, INF), seq[k].get(x, INF)
            if b > a + 1e-9 * max(1.0, abs(a) if a < 1e300 else 1.0):
                return False
    last = seq[-1]
    for x in R:
        v, p = last.get(x, INF), V_pi.get(x, INF)
        if p >= 1e300:
            continue
        if v < p - 1e-9 * max(1.0, abs(p)):
            return False
    return True


# ---------------------------------------------------------------------------
# manual pipeline walk (decoys; safe in the unsolvable regime, plan §3)
# ---------------------------------------------------------------------------

def pipeline_walk(G: MM.STG, xe: int, lam_in, ctx: MM.NumericContext,
                  ue: Optional[int] = None, core_cap: int = CORE_CAP) -> dict:
    lam = ctx.lam(lam_in)
    R = MM.reverse_reachable(G, xe)
    Gr = G.restrict(R)
    J: Dict[int, object] = {xe: ctx.num(0)}
    cond = MM.condense(Gr, R, xe)
    comps: List[dict] = []
    has_cycle: Dict[int, bool] = {}
    first_arcs: Dict[int, Dict[int, object]] = {}
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
        easy = MM.solve_easy(S, internal, e, lam, ctx)
        if ctx.name == "exact":
            verdict, margin = easy.has_cycle_sweep, None
            benign = easy.labels
        else:
            det = MM.detect_via_pi(S, internal, e, lam, ctx)
            verdict, margin = det.has_cycle, det.tight_margin
            benign = det.V
        has_cycle[cid] = verdict
        solved = True
        cap = min(core_cap, 16 if ctx.name == "exact" else 20)
        if verdict:
            if len(S) <= cap:
                core = MM.solve_core(S, internal, e, lam, ctx, max_core=cap)
                for x, v in core.J.items():
                    if not ctx.is_inf(v):
                        J[x] = v
                first_arcs[cid] = core.first_arc
            else:
                solved = False
                complete = False
        else:
            for x, v in benign.items():
                if not ctx.is_inf(v):
                    J[x] = v
        comps.append({
            "cid": cid, "size": len(S), "verdict": verdict,
            "sweep_verdict": easy.has_cycle_sweep, "margin": margin,
            "solved": solved, "S": S, "internal": internal, "e": e,
        })
    mu = MM.recover(Gr, cond, J, has_cycle, first_arcs, lam, ctx, xe, ue) if complete else None
    return {"R": R, "cond": cond, "J": J, "mu": mu, "comps": comps, "complete": complete}


def dfs_component(S: Set[int], internal, e, lam: float) -> Dict[int, float]:
    """Component-level simple-path oracle (float), |S| <= DFS_CAP only."""
    def rec(x: int, visited: Set[int]) -> float:
        best = float(e.get(x, INF)) if e.get(x, INF) < 1e300 else INF
        for _u, y, w in internal.get(x, []):
            if y in visited:
                continue
            sub = rec(y, visited | {y})
            if sub >= 1e300:
                continue
            best = min(best, float(w) + lam * sub)
        return best
    return {x: rec(x, {x}) for x in S}


# ---------------------------------------------------------------------------
# Module A: survey rows
# ---------------------------------------------------------------------------

def _mk_row_config() -> List[dict]:
    """The 6 + 2 survey rows (plan §2.1, §2.4)."""
    ap = BD.load_network("apoptosis")
    lam_sw = BD.load_network("lambda_switch")
    ara = BD.load_network("ara_operon")
    mcc = BD.load_network("mammalian_cc")
    lac = BD.load_network("lac_operon")
    tlgl = BD.load_network("t_lgl")

    # NOTE (finding, plan §7.3): the cell-death fixpoint (0,1,1) of the
    # apoptosis model is reverse-reachable only from itself -- TNF alone
    # cannot force death in this model -- so the survey row targets the
    # survival homeostasis state instead (all off, TNF off).
    xe_apop = BD.state_index(ap, (0, 0, 0))            # survival, no stimulus
    xe_mcc = BD.state_index(mcc, (1, 0, 0, 0, 1, 0, 1, 0, 0))  # G0
    xe_lac = BD.state_index(lac, (0, 0, 0, 0, 1, 1, 0, 0, 0, 0))  # OFF @ Ge=1
    rows = [
        dict(key="apoptosis", spec=ap, xe=xe_apop, ue=BD.input_index(ap, (0,)),
             Cu=None, why="survival homeostasis (all off), TNF off"),
        dict(key="lambda_switch", spec=lam_sw, xe=24, ue=2,
             Cu=None, why="lysogeny (cI on), unfavorable env u=0"),
        dict(key="ara_operon", spec=ara, xe=512, ue=11,
             Cu=None, why="operon fully off, (Ae,Aem,Ara-,Ge)=(0,1,0,1)"),
        dict(key="mammalian_cc", spec=mcc, xe=xe_mcc, ue=2,
             Cu=None, why="quiescence G0 (Rb,p27,Cdh1 on), CycD=0"),
        dict(key="lac_operon", spec=lac, xe=xe_lac, ue=BD.input_index(lac, (1, 0, 0)),
             Cu=None, why="operon OFF under glucose (catabolite repression)"),
        dict(key="t_lgl", spec=tlgl, xe=65535, ue=BD.input_index(tlgl, (1, 0, 1)),
             Cu=None, why="apoptosis of the leukemic clone, no drug"),
        dict(key="ara_restricted", spec=ara, xe=512, ue=11,
             Cu=(2, 11), why="ara with C_u = {delta_16^2, delta_16^11}"),
        dict(key="lac_restricted", spec=lac, xe=xe_lac, ue=BD.input_index(lac, (1, 0, 0)),
             Cu=(BD.input_index(lac, (1, 0, 0)), BD.input_index(lac, (0, 0, 0))),
             why="lac with glucose switching only (lactose held low)"),
    ]
    return rows


def survey_row(cfg: dict, ctx: MM.NumericContext, lam: float = LAM) -> dict:
    spec, xe, ue = cfg["spec"], cfg["xe"], cfg["ue"]
    Cu_t = cfg["Cu"]
    Cu = (lambda _x, _a=Cu_t: _a) if Cu_t else None
    M_eff = len(Cu_t) if Cu_t else len(spec.inputs())
    Vs, D = BD.reach_depth(spec, xe, Cu=Cu)
    g = BD.hamming_cost(spec, xe, ue)
    G = BD.make_stg(spec, g, ctx, Cu=Cu)
    R = MM.reverse_reachable(G, xe)
    assert R == Vs, (len(R), len(Vs))
    cond = MM.condense(G.restrict(R), R, xe)
    n_scc = len(cond.components)
    max_scc = max(len(c) for c in cond.components)

    reps = 5 if spec.N <= 1024 else 3
    t_master, master_times, master = timed(
        lambda: MM.run_master(G, xe, lam, ctx, ue=ue), reps=reps)
    bc = sum(1 for r in master.records if r.has_cycle)
    bc_sizes = sorted(len(r.S) for r in master.records if r.has_cycle)
    disagreements = master.detect_disagreements

    seq, k_vi = vi_until(G, R, xe, lam, ctx)
    t_vi, vi_times, _ = timed(lambda: MM.vi(G, R, xe, lam, ctx, k_vi), reps=reps)
    V_vi = seq[-1]

    V_pi, mu_pi = MM.pi(G, R, xe, lam, ctx)
    adjR = MM.restricted_adjacency(G, R)
    mu_vi = MM.greedy_mu(V_vi, adjR, xe, lam, ctx, spec.next_state, ue=ue)
    imp_vi = MM.improper_states(spec.next_state, mu_vi, R, xe)
    imp_pi = MM.improper_states(spec.next_state, mu_pi, R, xe)
    imp_master = MM.improper_states(spec.next_state, master.mu, R, xe)

    imp_floyd = None
    if spec.N <= 512:
        cols = MM.floyd(G, R, xe, lam, ctx, k_vi)
        mu_fl = MM.greedy_mu(cols[-1], adjR, xe, lam, ctx, spec.next_state, ue=ue)
        imp_floyd = MM.improper_states(spec.next_state, mu_fl, R, xe)

    gap = sum(1 for x in R
              if abs(V_vi.get(x, INF) - float(master.J.get(x, INF))) > 5e-6
              and not (V_vi.get(x, INF) >= 1e300 and master.J.get(x, INF) == INF))
    margins = [r.tight_margin for r in master.records if r.tight_margin is not None]
    return {
        "network": cfg["key"], "n": spec.n, "m": spec.m,
        "M_eff": M_eff, "M": spec.M, "N": spec.N,
        "xe": xe, "ue": ue, "xe_meaning": cfg["why"], "lambda": lam,
        "Vs": len(Vs), "D_reach": D, "n_scc": n_scc, "max_scc": max_scc,
        "bc_natural": bc, "bc_sizes": ";".join(map(str, bc_sizes)),
        "detect_disagreements": disagreements,
        "t_master_s": round(t_master, 5), "t_vi_s": round(t_vi, 5),
        "t_ratio_vi_master": round(t_vi / t_master, 3) if t_master > 0 else None,
        "k_vi": k_vi, "gap_states": gap,
        "imp_vi": len(imp_vi), "imp_pi": len(imp_pi),
        "imp_floyd": (len(imp_floyd) if imp_floyd is not None else "DNF"),
        "imp_master": len(imp_master),
        "min_margin": (min(margins) if margins else None),
        "master_times": ";".join(f"{t:.5f}" for t in master_times),
        "vi_times": ";".join(f"{t:.5f}" for t in vi_times),
        "_sets": {"vi": imp_vi, "pi": imp_pi, "floyd": imp_floyd, "master": imp_master},
        "_seq": seq, "_Vpi": V_pi, "_R": R, "_G": G,
    }


# ---------------------------------------------------------------------------
# Module C: decoy cases
# ---------------------------------------------------------------------------

def _decoy_run(tag: str, spec, xe: int, ue: int, decoy_pairs: Set[Tuple[int, int]],
               Cu_t: Optional[Tuple[int, ...]], ctx: MM.NumericContext,
               exact_recheck: bool) -> dict:
    """Run the decoy-cost pipeline + VI parking analysis on one configuration."""
    base = BD.hamming_cost(spec, xe, ue)
    g_eps = BD.decoy_cost(base, decoy_pairs, DECOY_EPS)
    Cu = (lambda _x, _a=Cu_t: _a) if Cu_t else None
    G = BD.make_stg(spec, g_eps, ctx, Cu=Cu)
    walk = pipeline_walk(G, xe, LAM, ctx, ue=ue)
    R = walk["R"]
    decoy_states = {x for x, _u in decoy_pairs}
    dec_comps = [c for c in walk["comps"] if decoy_states & c["S"]]
    assert dec_comps, f"{tag}: decoy state not in any processed component"
    dc = dec_comps[0]

    seq, k_vi = vi_until(G, R, xe, LAM, ctx)
    V_vi = seq[-1]
    adjR = MM.restricted_adjacency(G, R)
    mu_vi = MM.greedy_mu(V_vi, adjR, xe, LAM, ctx, spec.next_state, ue=ue)
    parking = MM.improper_states(spec.next_state, mu_vi, R, xe)
    V_pi, _mu_pi = MM.pi(G, R, xe, LAM, ctx)

    out = {
        "tag": tag, "N": spec.N, "M_eff": (len(Cu_t) if Cu_t else len(spec.inputs())),
        "decoy_pairs": sorted(decoy_pairs), "component_size": dc["size"],
        "verdict_pi_tight": dc["verdict"], "sweep_verdict": dc["sweep_verdict"],
        "margin": dc["margin"], "regime": "solvable" if dc["solved"] else "detect-only",
        "complete": walk["complete"], "k_vi": k_vi,
        "vi_parking": sorted(parking), "n_parking": len(parking),
        "exact_recheck": None, "gap_states": None, "imp_master": None,
        "_walk": walk, "_G": G, "_Vpi": V_pi, "_R": R,
    }
    if walk["complete"]:
        J = walk["J"]
        gapset = {x for x in R
                  if V_pi.get(x, INF) < float(J.get(x, INF)) - 1e-6
                  * max(1.0, abs(V_pi.get(x, 0.0)) if V_pi.get(x, INF) < 1e300 else 1.0)}
        imp_master = MM.improper_states(spec.next_state, walk["mu"], R, xe)
        out["gap_states"] = sorted(gapset)
        out["imp_master"] = len(imp_master)
    if exact_recheck:
        ctxq = MM.exact()
        g_exact = BD.decoy_cost(base, decoy_pairs, Fraction(1, 20))
        Gq = BD.make_stg(spec, g_exact, ctxq, Cu=Cu)
        walkq = pipeline_walk(Gq, xe, Fraction(1, 2), ctxq, ue=ue)
        dcq = [c for c in walkq["comps"] if decoy_states & c["S"]][0]
        out["exact_recheck"] = {
            "verdict": dcq["verdict"], "size": dcq["size"],
            "agrees": dcq["verdict"] == dc["verdict"] and dcq["size"] == dc["size"],
        }
    return out


# ---------------------------------------------------------------------------
# Module D: MILP cross-checks
# ---------------------------------------------------------------------------

def milp_crosscheck(tag: str, comp: dict, ctx: MM.NumericContext) -> dict:
    S, internal, e = comp["S"], comp["internal"], comp["e"]
    lam = ctx.lam(LAM)
    t0 = time.perf_counter()
    res_milp = MM.milp_oracle(S, internal, e, lam, ctx)
    t_milp = time.perf_counter() - t0
    coverage = len(res_milp) / len(S)
    agree_core = agree_dfs = None
    if len(S) <= CORE_CAP:
        core = MM.solve_core(S, internal, e, lam, ctx)
        agree_core = all(
            abs(float(res_milp[x]) - float(core.J[x])) <= 1e-6 * max(1.0, abs(float(core.J[x])))
            if (res_milp[x] < 1e300 or core.J[x] < 1e300) else True
            for x in res_milp
        )
    if len(S) <= DFS_CAP:
        dfs = dfs_component(S, internal, e, float(lam))
        agree_dfs = all(
            (res_milp[x] >= 1e300 and dfs[x] >= 1e300)
            or abs(float(res_milp[x]) - dfs[x]) <= 1e-6 * max(1.0, abs(dfs[x]))
            for x in res_milp
        )
    # sandwich check (always available): V_lambda* <= J_MILP <= exit weight
    V = MM.detect_via_pi(S, internal, e, lam, ctx).V
    bounds_ok = True
    for x, jv in res_milp.items():
        if jv >= 1e300:
            continue
        lo = ctx.as_float(V.get(x, INF))
        if lo < 1e300 and float(jv) < lo - 1e-6 * max(1.0, abs(lo)):
            bounds_ok = False
        ex = ctx.as_float(e.get(x, INF))
        if ex < 1e300 and float(jv) > ex + 1e-6 * max(1.0, abs(ex)):
            bounds_ok = False
    return {"tag": tag, "size": len(S), "coverage": coverage, "t_milp_s": round(t_milp, 3),
            "agree_core": agree_core, "agree_dfs": agree_dfs, "bounds_ok": bounds_ok,
            "certified": len(res_milp)}


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def run(skip_tlgl: bool = False) -> dict:
    os.makedirs(RESULTS, exist_ok=True)
    ctx = MM.floating(1e-9)
    t_start = time.time()

    # ---------------- gen anchors (A3 rides on gen_networks; A1/A2 here) ----
    import gen_networks as GN
    GN.anchors.clear()
    GN.run()
    check("A2/A3", True, f"gen_networks anchors all passed ({len(GN.anchors)} anchors, "
          "incl. dual-source diffs and literature fixpoint/attractor anchors)")

    rows_cfg = _mk_row_config()
    ap_spec = next(c["spec"] for c in rows_cfg if c["key"] == "apoptosis")
    death = BD.state_index(ap_spec, (0, 1, 1))
    Vs_death, _d = BD.reach_depth(ap_spec, death)
    note(f"apoptosis: the death fixpoint delta_8^{death} is reverse-reachable "
         f"from |Vs|={len(Vs_death)} state(s) only (TNF alone cannot force death); "
         "survey row therefore targets survival homeostasis (plan §7.3)")

    # A1: loader round trips + bit convention pinned to ara_io on ara
    ara = next(c["spec"] for c in rows_cfg if c["key"] == "ara_operon")
    from ara_io import inverse_map as ara_inv
    ok1 = all(len(c["spec"].L) == c["spec"].N * c["spec"].M for c in rows_cfg)
    ok1 &= all(tuple(ara_inv(x, 9)) == BD.bits_of(ara, x) for x in range(1, 513))
    check("A1", ok1, "loader round-trip on all specs; bits == ara_io.inverse_map on ara")

    # ---------------- Module A/B: survey table --------------------------------
    rows: List[dict] = []
    for cfg in rows_cfg:
        if skip_tlgl and cfg["key"] == "t_lgl":
            note("t_lgl row skipped (--quick)")
            continue
        print(f"[survey] {cfg['key']} (N={cfg['spec'].N}) ...")
        rows.append(survey_row(cfg, ctx))

    # A4: all rows generated, no detector disagreements anywhere.  FINDING
    # (recorded in plan §10): the natural Hamming cost is benign on 7 of the
    # 8 rows, but on T-LGL the two 20-state survival SCCs carry beneficial
    # cycles ALREADY under the natural cost -- the apoptosis target is
    # Hamming-far, so hovering near the diseased attractor is discount-
    # cheaper than stabilizing.  Both components sit exactly at the
    # SolveCore cap and are solved exactly.
    want_rows = 8 - (1 if skip_tlgl else 0)
    ok4 = len(rows) == want_rows and all(r["detect_disagreements"] == 0 for r in rows)
    for r in rows:
        if r["network"] == "t_lgl":
            ok4 &= r["bc_natural"] == 2 and r["bc_sizes"] == "20;20"
        else:
            ok4 &= r["bc_natural"] == 0
    check("A4", ok4, f"{len(rows)} rows, detect_disagreements=0 on all; BC(g)=0 "
          "except t_lgl (two 20-state survival SCCs, solved exactly at the cap)")

    # A5: VI protocol (stop rule met within bound, monotone, PI-anchored;
    #     improper sets pairwise equal where computed)
    ok5 = True
    for r in rows:
        ok5 &= r["k_vi"] <= K_CAP
        ok5 &= vi_monotone_and_anchored(r["_seq"], r["_Vpi"], r["_R"])
        s = r["_sets"]
        ok5 &= s["vi"] == s["pi"]
        if s["floyd"] is not None:
            ok5 &= s["floyd"] == s["vi"]
    check("A5", ok5, "VI stop rule + monotone + PI lower bound; improper sets agree")

    # A6: benign rows -> both master and VI feedbacks proper everywhere;
    # on t_lgl (beneficial cycles under the natural cost) master must stay
    # proper while VI must NOT (the trap is real, not a decoy artifact).
    ok6 = True
    for r in rows:
        ok6 &= r["imp_master"] == 0
        if r["bc_natural"] == 0:
            ok6 &= r["imp_vi"] == 0 and r["gap_states"] == 0
        else:
            ok6 &= r["imp_vi"] > 0 and r["gap_states"] > 0 and r["imp_vi"] == r["imp_pi"]
    check("A6", ok6, "improper(master)=0 on all rows; VI proper exactly on the "
          "benign rows and improper on t_lgl (natural-cost traps)")

    for r in rows:
        for k in ("_sets", "_seq", "_Vpi", "_R", "_G"):
            r.pop(k, None)
    with open(os.path.join(RESULTS, "exp16_stats.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # ---------------- Module C: decoy cases ---------------------------------
    decoys: List[dict] = []
    specs = {c["key"]: c["spec"] for c in rows_cfg}

    # Case B: lac operon "cheap induction medium" (plan §3.2)
    lac = specs["lac_operon"]
    xe_lac = next(c["xe"] for c in rows_cfg if c["key"] == "lac_operon")
    ue_lac = next(c["ue"] for c in rows_cfg if c["key"] == "lac_operon")
    u_d_lac = BD.input_index(lac, (0, 1, 1))
    fps_lac = [x for x, us in BD.control_fixpoints(lac).items() if u_d_lac in us]
    on_lac = [x for x in fps_lac if BD.bits_of(lac, x)[:3] == (1, 1, 1)]
    assert len(on_lac) == 1, on_lac
    x_d_lac = on_lac[0]
    print(f"[decoy B] lac: x_d={x_d_lac}, u_d={u_d_lac}")
    dB = _decoy_run("B-lac", lac, xe_lac, ue_lac, {(x_d_lac, u_d_lac)}, None, ctx,
                    exact_recheck=True)
    decoys.append(dB)
    # restricted variant: only the reference and the induction media available,
    # confining the decoy cycle to a smaller component (mirrors ara, plan §2.4/§3.2)
    print(f"[decoy B-restricted] C_u = {(ue_lac, u_d_lac)}")
    dB_r = _decoy_run("B-lac-restricted", lac, xe_lac, ue_lac,
                      {(x_d_lac, u_d_lac)}, (ue_lac, u_d_lac), ctx,
                      exact_recheck=True)
    decoys.append(dB_r)

    # Case A: T-LGL "watchful waiting" (plan §3.1)
    dA = dA_restricted = None
    if not skip_tlgl:
        tlgl = specs["t_lgl"]
        xe_t = 65535
        ue_t = BD.input_index(tlgl, (1, 0, 1))     # no-drug input
        u_d_t = ue_t
        fps_t = [x for x, us in BD.control_fixpoints(tlgl).items() if u_d_t in us]
        diseased = [x for x in fps_t
                    if BD.bits_of(tlgl, x)[15] == 0 and x != xe_t]
        note(f"T-LGL fixpoints under no-drug input: {fps_t}; diseased candidates: {diseased}")
        assert diseased, "no diseased fixpoint under the no-drug input"
        # prefer delta^58834, the diseased state used by the TCYB paper itself
        x_d_t = 58834 if 58834 in diseased else diseased[0]
        print(f"[decoy A] t_lgl: x_d={x_d_t}, u_d={u_d_t}")
        dA = _decoy_run("A-tlgl", tlgl, xe_t, ue_t, {(x_d_t, u_d_t)}, None, ctx,
                        exact_recheck=False)
        decoys.append(dA)
        if dA["regime"] == "detect-only":
            # restricted actuation: no-drug vs Ceramide forcing (u2 kill switch)
            Cu_t = (BD.input_index(tlgl, (1, 1, 1)), u_d_t)
            print(f"[decoy A-restricted] C_u = {Cu_t}")
            dA_restricted = _decoy_run("A-tlgl-restricted", tlgl, xe_t, ue_t,
                                       {(x_d_t, u_d_t)}, Cu_t, ctx, exact_recheck=False)
            decoys.append(dA_restricted)

    # Case C (enrichment): mammalian cc cycling attractor as an ell>1 decoy
    mcc = specs["mammalian_cc"]
    xe_m = next(c["xe"] for c in rows_cfg if c["key"] == "mammalian_cc")
    att1 = GN.synchronous_attractors(lambda b: GN.mcc_step(b, (1,)), 9)
    dC = None
    if len(att1) == 1 and len(att1[0]) == 7:
        cyc_states = {BD.state_index(mcc, b) for b in att1[0]}
        pairs = {(x, 1) for x in cyc_states}   # u = delta_2^1 = CycD on
        print(f"[decoy C] mcc: 7-cycle states {sorted(cyc_states)}")
        dC = _decoy_run("C-mcc-cycle", mcc, xe_m, 2, pairs, None, ctx,
                        exact_recheck=True)
        decoys.append(dC)
    else:
        note("mcc synchronous 7-cycle not found; case C skipped (plan §7.6)")

    # ara restricted decoy (MILP flagship, tab:ara-decoy golden values)
    ara_spec = specs["ara_operon"]
    print("[decoy ara-restricted] C_u={2,11}, x_d=9, u_d=2")
    dAra = _decoy_run("ara-restricted", ara_spec, 512, 11, {(9, 2)}, (2, 11), ctx,
                      exact_recheck=True)
    decoys.append(dAra)
    golden_J = {9: 11.375, 25: 10.375, 137: 10.375, 153: 9.375, 265: 10.375, 281: 9.375}
    golden_V = {9: 0.1, 25: 8.05, 137: 8.05, 153: 7.05, 265: 8.05, 281: 7.05}
    Jw = dAra["_walk"]["J"]
    ok_tab = (dAra["complete"]
              and all(abs(float(Jw[x]) - v) <= 1e-9 for x, v in golden_J.items())
              and all(abs(dAra["_Vpi"][x] - v) <= 1e-9 for x, v in golden_V.items())
              and set(dAra["gap_states"]) == set(golden_J))
    check("A7-ara-golden", ok_tab, "tab:ara-decoy J/V values and 6-state gap set reproduced")

    # A7: decoy cases A/B verdicts + parking + repair
    ok7 = True
    for d in decoys:
        if d["tag"] in ("B-lac", "B-lac-restricted", "A-tlgl", "A-tlgl-restricted",
                        "ara-restricted"):
            ok7 &= d["verdict_pi_tight"] is True
            ok7 &= d["n_parking"] > 0
            if d["exact_recheck"] is not None:
                ok7 &= d["exact_recheck"]["agrees"]
            if d["complete"]:
                ok7 &= d["imp_master"] == 0
                ok7 &= set(d["vi_parking"]) == set(d["gap_states"])
    check("A7", ok7, "decoy verdicts True, VI parks (parking == gap set), master repairs, "
          "exact rechecks agree")

    # ---------------- Module D: MILP cross-checks ---------------------------
    milp_rows: List[dict] = []
    targets: List[Tuple[str, dict]] = []
    for d in decoys:
        dec_states = {x for x, _u in d["decoy_pairs"]}
        comp = next(c for c in d["_walk"]["comps"] if dec_states & c["S"])
        if comp["size"] <= MILP_CAP:
            targets.append((d["tag"], comp))
        else:
            note(f"MILP target {d['tag']} skipped: |S|={comp['size']} > {MILP_CAP}")
    for tag, comp in targets:
        print(f"[milp] {tag} |S|={comp['size']} ...")
        milp_rows.append(milp_crosscheck(tag, comp, ctx))
    ok8 = bool(milp_rows) and any(r["tag"] == "ara-restricted" for r in milp_rows)
    for r in milp_rows:
        ok8 &= r["coverage"] == 1.0 or r["certified"] > 0
        ok8 &= r["bounds_ok"]
        if r["agree_core"] is not None:
            ok8 &= r["agree_core"]
        if r["agree_dfs"] is not None:
            ok8 &= r["agree_dfs"]
    ara_row = next(r for r in milp_rows if r["tag"] == "ara-restricted")
    ok8 &= ara_row["coverage"] == 1.0
    check("A8", ok8, "MILP == SolveCore/DFS on certified states; ara-20 fully certified; "
          f"targets: {[(r['tag'], r['size'], r['coverage']) for r in milp_rows]}")

    with open(os.path.join(RESULTS, "exp16_milp.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(milp_rows[0].keys()))
        w.writeheader()
        w.writerows(milp_rows)

    # ---------------- A9: numeric calibre (lambda = 0.9 margins) ------------
    margin_rows: List[dict] = []
    small_margin_rechecked = True
    for key in ("ara_operon", "lac_operon"):
        cfg = next(c for c in rows_cfg if c["key"] == key)
        spec = cfg["spec"]
        g = BD.hamming_cost(spec, cfg["xe"], cfg["ue"])
        G = BD.make_stg(spec, g, ctx)
        master9 = MM.run_master(G, cfg["xe"], LAM2, ctx, ue=cfg["ue"])
        for rec in master9.records:
            if rec.tight_margin is not None:
                margin_rows.append({"network": key, "lambda": LAM2,
                                    "comp_size": len(rec.S),
                                    "tight_margin": rec.tight_margin,
                                    "disagreement": rec.disagreement})
        tiny = [rec for rec in master9.records
                if rec.tight_margin is not None and rec.tight_margin < 1e-6]
        if tiny:
            note(f"{key} lambda=0.9: {len(tiny)} components with margin < 1e-6 -> exact recheck")
            gq = BD.hamming_cost(spec, cfg["xe"], cfg["ue"])
            Gq = BD.make_stg(spec, gq, MM.exact())
            walkq = pipeline_walk(Gq, cfg["xe"], Fraction(9, 10), MM.exact(), ue=cfg["ue"])
            verd_q = {c["cid"]: c["verdict"] for c in walkq["comps"]}
            small_margin_rechecked &= all(not v for v in verd_q.values())
    check("A9", small_margin_rechecked,
          f"lambda=0.9 margins recorded ({len(margin_rows)} components); "
          "all sub-1e-6 margins re-verified under exact()"
          if margin_rows else "no margins recorded")
    if margin_rows:
        with open(os.path.join(RESULTS, "exp16_margins_lam09.csv"), "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(margin_rows[0].keys()))
            w.writeheader()
            w.writerows(margin_rows)

    # A10: timing protocol on record
    check("A10", all("master_times" in r and len(r["master_times"].split(";")) >= 3
                     for r in rows),
          "each t is a median of warmup-1 + >=3 reps; raw times kept in the csv")

    # ---------------- outputs ------------------------------------------------
    dec_out = []
    for d in decoys:
        dec_out.append({k: v for k, v in d.items() if not k.startswith("_")})
    with open(os.path.join(RESULTS, "exp16_decoy.csv"), "w", newline="") as fh:
        keys = sorted({k for d in dec_out for k in d})
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(dec_out)

    bib_todo = [
        "faure2006mammalian: Faure, Naldi, Chaouiya & Thieffry, Bioinformatics 22(14):e124-e131, 2006",
        "zhang2015invertibility: Zhang & Zhang, Automatica (invertibility & nonsingularity of BCNs), 2015",
        "kobayashi2013apoptosis: Kobayashi & Hiraishi, IEICE Trans. Fundamentals E96-A(2):532-539, 2013",
        "gao2021tcyb: Gao, Sun, Xiang, Qin & Lee, IEEE Trans. Cybernetics, DOI 10.1109/TCYB.2020.3003552",
        "murrugarra2016 (optional): Murrugarra et al., BMC Systems Biology 10:94, 2016",
    ]
    import pulp as _pulp
    lines = [
        "E16: Example 2 real-network study",
        f"  machine: {platform.platform()}, Python {platform.python_version()}, "
        f"PuLP {_pulp.__version__} (CBC bundled)",
        f"  protocol: eps_VI={EPS_VI}, stop residual <= eps(1-lam)/(2 lam); "
        f"k_cap={K_CAP}; timing = median of reps (5 if N<=1024 else 3), warmup 1; "
        f"MILP: PuLP+CBC, {MM.MILP_TIME_LIMIT_S}s/source; decoy eps={DECOY_EPS}",
        "", "SURVEY (lambda=1/2, natural Hamming cost):",
    ]
    for r in rows:
        lines.append(
            f"  {r['network']:16s} N={r['N']:6d} M_eff/M={r['M_eff']}/{r['M']} "
            f"|Vs|={r['Vs']:6d} D={r['D_reach']:2d} #SCC={r['n_scc']:6d} "
            f"max|SCC|={r['max_scc']:4d} BC={r['bc_natural']} "
            f"t_master={r['t_master_s']:.4f}s t_VI={r['t_vi_s']:.4f}s "
            f"(k={r['k_vi']}) ratio={r['t_ratio_vi_master']}")
    lines.append("")
    lines.append("DECOYS:")
    for d in dec_out:
        lines.append(
            f"  {d['tag']:18s} |S|={d['component_size']:4d} regime={d['regime']:12s} "
            f"parking={d['n_parking']:3d} margin={d['margin']} "
            f"imp_master={d['imp_master']}")
    lines.append("")
    lines.append("MILP:")
    for r in milp_rows:
        lines.append(f"  {r['tag']:18s} |S|={r['size']:3d} coverage={r['coverage']:.2f} "
                     f"t={r['t_milp_s']}s core={r['agree_core']} dfs={r['agree_dfs']}")
    lines.append("")
    lines.append("ASSERTIONS:")
    lines.extend(f"  {p}" for p in passed)
    lines.append("")
    lines.append("NOTES:")
    lines.extend(f"  {m}" for m in notes)
    lines.append("")
    lines.append("BIB ENTRIES TO ADD (plan §9 correction 2):")
    lines.extend(f"  {b}" for b in bib_todo)
    lines.append("")
    lines.append(f"  total runtime: {time.time() - t_start:.1f} s")
    lines.append(f"  ALL {len(passed)} ASSERTION GROUPS PASSED")
    text = "\n".join(lines)
    print(text)
    with open(os.path.join(RESULTS, "exp16_summary.txt"), "w") as fh:
        fh.write(text + "\n")
    return {"rows": len(rows), "decoys": len(dec_out), "milp": len(milp_rows)}


if __name__ == "__main__":
    run(skip_tlgl="--quick" in sys.argv)
