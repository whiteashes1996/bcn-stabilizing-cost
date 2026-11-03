"""E23: one static feedback for every stabilizable state (Problem joint-feedback).

The test is the one in the manuscript, not the older first-arc fallback:
on a folded component S, an arc is kept in G_0(S) when its local excess
(Definition local-excess) is zero, and the test passes when every state of
S reaches omega in G_0(S) (Theorem joint). Algorithm recover then returns
an in-tree of G_0(S); otherwise it returns an in-tree of the folded
digraph (Lemma in-tree: a backward search from omega).

Parts
  ex1    Example 1, component H, exact fractions, plus the lambda sweep
  bench  T-LGL natural, T-LGL decoy, mammalian decoy, ara restricted decoy
  sweep  lambda sweep of Example 1 and of the mammalian decoy kernel
  all    ex1, then bench, then sweep (sweep's Example 1 half is inside ex1)

Run:  python simulations/exp23_joint_feedback.py [all|ex1|bench|sweep]
"""
from __future__ import annotations

import csv
import os
import sys
import time
from collections import defaultdict, deque
from fractions import Fraction
from typing import Dict, List, Optional, Sequence, Set, Tuple

sys.path.insert(0, os.path.dirname(__file__))

import bcn_data as BD  # noqa: E402
import exp16_ex2_real as E16  # noqa: E402
import methods as MM  # noqa: E402
from exp14_ex1_n4 import (  # noqa: E402
    EXPECTED_GAP,
    EXPECTED_J,
    H_SCC,
    L as L_EX1,
    M as M_EX1,
    N as N_EX1,
    UE,
    XE,
    g as g_ex1,
)
from folded_uniform import (  # noqa: E402
    INF as FOLD_INF,
    Folded,
    enumerate_intrees,
    excess_of_tree,
    held_karp,
    tree_cost,
)

RESULTS = os.path.join(os.path.dirname(__file__), "results")
INF = MM.INF
LAM = Fraction(1, 2)
DECOY_EPS = Fraction(1, 20)

fails: List[str] = []
_CFG_CACHE: Optional[List[dict]] = None


def check(tag: str, cond: bool, detail: str = "") -> None:
    if cond:
        print(f"  PASS  {tag}", flush=True)
    else:
        msg = f"{tag}" + (f" -- {detail}" if detail else "")
        fails.append(msg)
        print(f"  FAIL  {msg}", flush=True)


def fmt(v) -> str:
    """Exact decimal when the value is a terminating decimal, else a/b.

    Fraction's format spec rounds to six places, which turned 3087/640
    into 4.823438. Build the decimal from the factors of the denominator.
    """
    if v is None:
        return "n/a"
    if isinstance(v, float):
        if v == INF or v >= 1e300:
            return "inf"
        return f"{v:.8g}"
    if not isinstance(v, Fraction):
        return str(v)
    if v.denominator == 1:
        return str(v.numerator)
    sign = "-" if v.numerator < 0 else ""
    n, d = abs(v.numerator), v.denominator
    twos = fives = 0
    dd = d
    while dd % 2 == 0:
        dd //= 2
        twos += 1
    while dd % 5 == 0:
        dd //= 5
        fives += 1
    if dd != 1:
        return f"{v.numerator}/{v.denominator}"
    scale = max(twos, fives)
    num = n * (2 ** (scale - twos)) * (5 ** (scale - fives))
    digits = str(num)
    if scale == 0:
        return sign + digits
    if len(digits) <= scale:
        digits = digits.zfill(scale + 1)
    ip, fp = digits[:-scale], digits[-scale:]
    fp = fp.rstrip("0")
    return sign + (ip if ip else "0") + ("." + fp if fp else "")


def fmt_tex(v) -> str:
    s = fmt(v)
    if s == "n/a":
        return "n/a"
    if "/" in s:
        a, b = s.split("/")
        return f"{a}/{b}"
    return s


# ---------------------------------------------------------------------------
# Pipeline (Algorithm master) keeping the folded arcs SolveCore saw
# ---------------------------------------------------------------------------


def walk(G: MM.STG, xe: int, lam_in, ctx: MM.NumericContext) -> dict:
    """Restriction, condensation, folding, detection, per-component J*.

    Beneficial-cycle components are solved by SolveCore. A component larger
    than the context cap is not handed to an MILP; the caller is expected
    to stay inside the cap (this experiment excludes the |S|>20 kernels).
    """
    MM._check_ctx(G, ctx)
    lam = ctx.lam(lam_in)
    R = MM.reverse_reachable(G, xe)
    Gr = G.restrict(R)
    J: Dict[int, MM.Num] = {xe: ctx.num(0)}
    comps: List[dict] = []
    if len(R) == 1:
        return {"R": R, "Gr": Gr, "J": J, "comps": comps, "lam": lam}
    cond = MM.condense(Gr, R, xe)
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
            verdict = easy.has_cycle_sweep
            benign = easy.labels
        else:
            det = MM.detect_via_pi(S, internal, e, lam, ctx)
            verdict = det.has_cycle
            benign = det.V
        t_core = 0.0
        if verdict:
            t0 = time.perf_counter()
            core = MM.solve_core(S, internal, e, lam, ctx)
            t_core = time.perf_counter() - t0
            Jc = core.J
        else:
            Jc = benign
        for x, v in Jc.items():
            if not ctx.is_inf(v):
                J[x] = v
        comps.append({
            "cid": cid, "S": set(S), "internal": internal, "e": e,
            "verdict": bool(verdict), "t_core": t_core,
        })
    return {"R": R, "Gr": Gr, "J": J, "comps": comps, "lam": lam, "cond": cond}


# ---------------------------------------------------------------------------
# G_0(S): local excess zero, backward search from omega
# ---------------------------------------------------------------------------


Arc = Tuple[str, int, Optional[int], Optional[int], MM.Num, MM.Num]
# (kind, x, y or None, u or None, stage weight, excess)


def component_arcs(comp: dict, J: Dict[int, MM.Num], lam, ctx: MM.NumericContext) -> List[Arc]:
    S, internal, e = comp["S"], comp["internal"], comp["e"]
    arcs: List[Arc] = []
    for x in sorted(S):
        jx = J[x]
        ex = e.get(x, INF)
        if not ctx.is_inf(ex):
            arcs.append(("exit", x, None, None, ex, ex - jx))
        for u, y, w in internal.get(x, []):
            arcs.append(("internal", x, y, u, w, w + lam * J[y] - jx))
    return arcs


def _backward_tree(S: Set[int], arcs: Sequence[Arc]) -> Tuple[Set[int], Dict[int, Arc]]:
    """Lemma in-tree: walk backward from omega; the first arc that reaches a
    state is a shortest-path arc, so the recorded selection is an in-tree."""
    pred: Dict[Optional[int], List[Arc]] = defaultdict(list)
    for arc in arcs:
        pred[arc[2]].append(arc)
    good: Set[int] = set()
    tree: Dict[int, Arc] = {}
    dq: deque = deque()
    for arc in pred.get(None, []):
        x = arc[1]
        if x not in good:
            good.add(x)
            tree[x] = arc
            dq.append(x)
    while dq:
        y = dq.popleft()
        for arc in pred.get(y, []):
            x = arc[1]
            if x not in good and x in S:
                good.add(x)
                tree[x] = arc
                dq.append(x)
    return good, tree


def g0_test(comp: dict, J: Dict[int, MM.Num], lam, ctx: MM.NumericContext,
            repeat: bool = False) -> dict:
    """Build G_0(S). The repeated timing is only for a beneficial-cycle
    component that will be reported against its SolveCore time."""
    def once():
        arcs = component_arcs(comp, J, lam, ctx)
        zero = [a for a in arcs if ctx.close(a[5], ctx.num(0))]
        good, tree = _backward_tree(comp["S"], zero)
        return arcs, zero, good, tree

    arcs, zero, good, tree = once()
    t_test = 0.0
    if repeat:
        t0 = time.perf_counter()
        reps = 40
        for _ in range(reps):
            once()
        t_test = (time.perf_counter() - t0) / reps
    fallback_good, fallback = _backward_tree(comp["S"], arcs)
    return {
        "arcs": arcs, "zero": zero, "good": good, "tree": tree,
        "fallback": fallback, "fallback_spans": fallback_good == comp["S"],
        "joint": good == comp["S"], "t_test": t_test,
    }


def folded_path_cost(tree: Dict[int, Arc], x: int, lam, ctx: MM.NumericContext):
    """Discounted cost of the folded in-tree path. The exit weight already
    contains lam * J*(boundary), so this equals the original closed-loop
    cost only when every downstream component attains J*."""
    seen: Set[int] = set()
    tot = ctx.num(0)
    disc = ctx.num(1)
    cur: Optional[int] = x
    while cur is not None:
        if cur in seen:
            return INF
        seen.add(cur)
        kind, _x, y, _u, w, _eps = tree[cur]
        tot = tot + disc * w
        if kind == "exit":
            return tot
        disc = disc * lam
        cur = y
    return tot


def component_excess(comp: dict, tree: Dict[int, Arc], J, lam, ctx: MM.NumericContext):
    """Max folded excess, and the number of states whose excess is positive."""
    worst = ctx.num(0)
    n_pos = 0
    per: Dict[int, MM.Num] = {}
    for x in comp["S"]:
        cost = folded_path_cost(tree, x, lam, ctx)
        if cost == INF or ctx.is_inf(cost):
            return INF, len(comp["S"]), {}
        diff = cost - J[x]
        per[x] = diff
        if ctx.lt(ctx.num(0), diff):
            n_pos += 1
            if ctx.lt(worst, diff):
                worst = diff
    return worst, n_pos, per


def margins(arcs: Sequence[Arc], zero: Sequence[Arc], ctx: MM.NumericContext):
    """Largest |excess| kept in G_0, and smallest |excess| left outside it."""
    zset = {id(a) for a in zero}
    in_max = ctx.num(0)
    out_min = None
    for a in arcs:
        mag = abs(a[5])
        if id(a) in zset:
            if ctx.lt(in_max, mag):
                in_max = mag
        else:
            if out_min is None or ctx.lt(mag, out_min):
                out_min = mag
    return in_max, out_min


# ---------------------------------------------------------------------------
# Global feedback: Algorithm recover, then the true closed-loop cost
# ---------------------------------------------------------------------------


def recover_feedback(G: MM.STG, xe: int, ue: int, walked: dict, tests: Dict[int, dict],
                     ctx: MM.NumericContext) -> Dict[int, Tuple[int, int]]:
    lam = walked["lam"]
    J = walked["J"]
    mu: Dict[int, Tuple[int, int]] = {xe: (ue, xe)}
    for comp in walked["comps"]:
        test = tests[comp["cid"]]
        tree = test["tree"] if test["joint"] else test["fallback"]
        S = comp["S"]
        for x, arc in tree.items():
            kind, _x, y, u, _w, _eps = arc
            if kind == "internal":
                mu[x] = (u, y)
                continue
            best, bu, by = INF, None, None
            for uu, yy, w in G.out.get(x, []):
                if yy in S:
                    continue
                jy = J.get(yy, INF)
                if ctx.is_inf(jy):
                    continue
                val = w + lam * jy
                if ctx.lt(val, best):
                    best, bu, by = val, uu, yy
            if bu is None:
                raise RuntimeError(f"no concrete exit from {x}")
            mu[x] = (bu, by)
    return mu


def closed_loop_cost(G: MM.STG, mu: Dict[int, Tuple[int, int]], x0: int, xe: int,
                     lam, ctx: MM.NumericContext):
    seen: Set[int] = set()
    tot = ctx.num(0)
    disc = ctx.num(1)
    x = x0
    while x != xe:
        if x in seen or x not in mu:
            return INF
        seen.add(x)
        u, y = mu[x]
        w = None
        for uu, yy, ww in G.out.get(x, []):
            if uu == u and yy == y:
                w = ww
                break
        if w is None:
            return INF
        tot = tot + disc * w
        disc = disc * lam
        x = y
    return tot


def global_excess(G, mu, walked, xe, ctx: MM.NumericContext):
    J, lam = walked["J"], walked["lam"]
    worst = ctx.num(0)
    n_pos = 0
    missing = []
    for x in walked["R"]:
        if x == xe:
            continue
        if x not in mu or x not in J:
            missing.append(x)
            continue
        cost = closed_loop_cost(G, mu, x, xe, lam, ctx)
        if cost == INF or ctx.is_inf(cost):
            return INF, len(walked["R"]) - 1
        diff = cost - J[x]
        if ctx.lt(ctx.num(0), diff):
            n_pos += 1
            if ctx.lt(worst, diff):
                worst = diff
    if missing:
        raise RuntimeError(f"feedback missing at {missing[:8]}")
    return worst, n_pos


def gap_set(G, walked, xe, ctx: MM.NumericContext) -> List[int]:
    """States of V_s where the unconstrained value is strictly below J*."""
    V, _mu = MM.pi(G, walked["R"], xe, walked["lam"], ctx)
    out = []
    for x in walked["R"]:
        if x not in walked["J"]:
            continue
        if ctx.name == "exact":
            if V[x] < walked["J"][x]:
                out.append(x)
        else:
            scale = max(1.0, abs(float(walked["J"][x])))
            if float(V[x]) < float(walked["J"][x]) - 1e-6 * scale:
                out.append(x)
    return out


# ---------------------------------------------------------------------------
# Enumeration of spanning in-trees (folded component, exact arithmetic)
# ---------------------------------------------------------------------------


def to_folded(comp: dict, lam) -> Tuple[Folded, List[int]]:
    S = sorted(comp["S"])
    idx = {x: i for i, x in enumerate(S)}
    internal = []
    for x in S:
        for _u, y, w in comp["internal"].get(x, []):
            internal.append((idx[x], idx[y], Fraction(w)))
    exits = []
    for x in S:
        ex = comp["e"].get(x, INF)
        exits.append(None if ex == INF or (isinstance(ex, float) and ex >= 1e300) else Fraction(ex))
    return Folded(len(S), internal, exits, Fraction(lam)), S


def enumerate_exc_star(comp: dict, J: Dict[int, MM.Num], lam):
    """Min over spanning in-trees of the max folded excess. Caps the search
    so a thick component cannot explode; the mammalian kernel is 3^14."""
    folded, S = to_folded(comp, lam)
    choices = 1
    for x in range(folded.n):
        choices *= max(1, len(folded.outgoing(x)))
    if choices > 6_000_000:
        return {"exc_star": None, "n_trees": None, "n_pos_best": None,
                "choices": choices, "hk_agree": None}
    Jf = [Fraction(J[x]) for x in S]
    hk = held_karp(folded)
    agree = all(hk[i] == Jf[i] for i in range(folded.n))
    best = FOLD_INF
    best_mu = None
    n_trees = 0
    for mu in enumerate_intrees(folded):
        n_trees += 1
        ex = excess_of_tree(folded, Jf, mu)
        if ex < best:
            best = ex
            best_mu = mu
    n_pos = None
    if best_mu is not None:
        n_pos = sum(1 for i in range(folded.n) if tree_cost(folded, best_mu, i) > Jf[i])
    return {"exc_star": None if best_mu is None else best, "n_trees": n_trees,
            "n_pos_best": n_pos, "choices": choices, "hk_agree": agree,
            "J_hk": hk, "S": S}


# ---------------------------------------------------------------------------
# Part A — Example 1
# ---------------------------------------------------------------------------


def build_example1(ctx: MM.NumericContext) -> MM.STG:
    return MM.stg_from_L(L_EX1, N_EX1, M_EX1, g_ex1, ctx)


def analyze(G, xe, ue, lam, ctx: MM.NumericContext, anchor: Optional[int] = None,
            time_test: bool = True) -> dict:
    walked = walk(G, xe, lam, ctx)
    tests: Dict[int, dict] = {}
    for comp in walked["comps"]:
        tests[comp["cid"]] = g0_test(
            comp, walked["J"], walked["lam"], ctx,
            repeat=time_test and comp["verdict"],
        )
    mu = recover_feedback(G, xe, ue, walked, tests, ctx)
    max_g, n_g = global_excess(G, mu, walked, xe, ctx)
    gap = gap_set(G, walked, xe, ctx)
    tracked = None
    if anchor is not None:
        for comp in walked["comps"]:
            if anchor in comp["S"]:
                tracked = comp
                break
    return {"walked": walked, "tests": tests, "mu": mu, "max_global": max_g,
            "n_global": n_g, "gap": gap, "tracked": tracked}


def part_ex1() -> dict:
    print("\n== Example 1, exact, lambda = 1/2 ==", flush=True)
    ctx = MM.exact()
    G = build_example1(ctx)
    out = analyze(G, XE, UE, LAM, ctx, anchor=2)
    walked, tests = out["walked"], out["tests"]
    J = walked["J"]
    for x, expected in EXPECTED_J.items():
        if expected is None:
            check(f"ex1 J({x}) absent", x not in J or x not in out["walked"]["R"])
        else:
            check(f"ex1 J({x})={fmt(expected)}", x in J and J[x] == expected,
                  f"got {fmt(J.get(x))}")
    check("ex1 gap set", set(out["gap"]) == EXPECTED_GAP, f"got {sorted(out['gap'])}")
    comp = out["tracked"]
    check("ex1 tracked component is H", comp is not None and comp["S"] == H_SCC,
          f"got {None if comp is None else sorted(comp['S'])}")
    test = tests[comp["cid"]]
    check("ex1 H is a beneficial-cycle component", comp["verdict"])
    check("ex1 G_0(H) spans H", test["joint"])
    check("ex1 fallback would also span", test["fallback_spans"])
    worst, n_pos, _per = component_excess(comp, test["tree"], J, walked["lam"], ctx)
    check("ex1 in-tree excess is 0", worst == 0 and n_pos == 0, f"worst={fmt(worst)} n={n_pos}")
    check("ex1 global excess is 0", out["max_global"] == 0 and out["n_global"] == 0,
          f"max={fmt(out['max_global'])} n={out['n_global']}")
    # The recovered in-tree on H: exit at 2 (the u=2 arc into 8), u=1 at 3 and 5.
    check("ex1 mu(2) leaves H", out["mu"][2][1] not in H_SCC, f"mu={out['mu'].get(2)}")
    check("ex1 mu(3)=(1,5)", out["mu"].get(3) == (1, 5), f"mu={out['mu'].get(3)}")
    check("ex1 mu(5)=(1,2)", out["mu"].get(5) == (1, 2), f"mu={out['mu'].get(5)}")
    in_max, out_min = margins(test["arcs"], test["zero"], ctx)
    check("ex1 zero excesses are exact", in_max == 0, f"in_max={fmt(in_max)}")
    enum = enumerate_exc_star(comp, J, walked["lam"])
    check("ex1 Held-Karp agrees", enum["hk_agree"] is True)
    check("ex1 enumeration Exc*=0", enum["exc_star"] == 0, f"got {enum['exc_star']}")

    lines = ["E23 Example 1, lambda=1/2, exact Fractions",
             f"  |V_s|={len(walked['R'])} gap={sorted(out['gap'])} joint_global=True",
             f"  H={sorted(comp['S'])} |arcs|={len(test['arcs'])} |G_0|={len(test['zero'])} "
             f"reach={len(test['good'])} joint={test['joint']}",
             f"  t_test_s={test['t_test']:.3e} t_core_s={comp['t_core']:.3e}",
             f"  enumeration: n_trees={enum['n_trees']} Exc*={fmt(enum['exc_star'])}",
             "  arcs (kind, x, y, u, weight, excess, in G_0):"]
    zero_ids = {id(a) for a in test["zero"]}
    for a in test["arcs"]:
        kind, x, y, u, w, eps = a
        lines.append(f"    {kind:8} x={x} y={y} u={u} w={fmt(w)} eps={fmt(eps)} "
                     f"G0={id(a) in zero_ids}")
    lines.append("  recovered mu on H: " +
                 ", ".join(f"{x}->{out['mu'][x]}" for x in sorted(comp["S"])))
    text = "\n".join(lines)
    print(text, flush=True)
    _write("exp23_ex1_G0.txt", text + "\n")
    return out


# ---------------------------------------------------------------------------
# Part B — benchmark kernels at lambda = 1/2
# ---------------------------------------------------------------------------


def _configs() -> List[dict]:
    global _CFG_CACHE
    if _CFG_CACHE is not None:
        return _CFG_CACHE
    rows = {c["key"]: c for c in E16._mk_row_config()}
    tlgl = rows["t_lgl"]["spec"]
    ara = rows["ara_operon"]["spec"]
    mcc = rows["mammalian_cc"]["spec"]
    ue_t = BD.input_index(tlgl, (1, 0, 1))
    import gen_networks as GN
    att = GN.synchronous_attractors(lambda b: GN.mcc_step(b, (1,)), 9)
    cyc = {BD.state_index(mcc, b) for b in att[0]}
    out = [
        dict(tag="T-LGL (natural cost)", spec=tlgl, xe=65535, ue=ue_t,
             Cu=None, decoy=None, anchor=58834, exact=False),
        dict(tag="T-LGL (decoy)", spec=tlgl, xe=65535, ue=ue_t,
             Cu=None, decoy={(58834, ue_t)}, anchor=58834, exact=False),
        dict(tag="mammalian cell cycle (decoy)", spec=mcc,
             xe=rows["mammalian_cc"]["xe"], ue=2, Cu=None,
             decoy={(x, 1) for x in cyc}, anchor=min(cyc), exact=True),
        dict(tag="ara, restricted inputs (decoy)", spec=ara, xe=512, ue=11,
             Cu=(2, 11), decoy={(9, 2)}, anchor=9, exact=False),
    ]
    _CFG_CACHE = out
    return out


def _build_cfg(cfg: dict, ctx: MM.NumericContext) -> MM.STG:
    base = BD.hamming_cost(cfg["spec"], cfg["xe"], cfg["ue"])
    eps = DECOY_EPS if ctx.name == "exact" else float(DECOY_EPS)
    g = BD.decoy_cost(base, cfg["decoy"], eps) if cfg["decoy"] else base
    Cu = (lambda _x, _a=cfg["Cu"]: _a) if cfg["Cu"] else None
    return BD.make_stg(cfg["spec"], g, ctx, Cu=Cu)


def _row_for_component(cfg, comp, test, J, lam, ctx, gap_n, enum) -> dict:
    in_max, out_min = margins(test["arcs"], test["zero"], ctx)
    tree = test["tree"] if test["joint"] else test["fallback"]
    worst, n_pos, _per = component_excess(comp, tree, J, lam, ctx)
    return {
        "config": cfg["tag"],
        "states": " ".join(str(x) for x in sorted(comp["S"])),
        "contains_58834": 58834 in comp["S"],
        "contains_anchor": cfg["anchor"] in comp["S"],
        "|S|": len(comp["S"]),
        "n_arcs": len(test["arcs"]),
        "|G_0|": len(test["zero"]),
        "n_reach": len(test["good"]),
        "joint": test["joint"],
        "has_cycle": comp["verdict"],
        "gap": gap_n,
        "max_excess_S": fmt(worst),
        "n_excess_S": n_pos,
        "exc_star": fmt(enum["exc_star"]) if enum else "n/a",
        "n_trees": enum["n_trees"] if enum else "n/a",
        "n_pos_best": enum["n_pos_best"] if enum else "n/a",
        "t_test_s": f"{test['t_test']:.3e}",
        "t_core_s": f"{comp['t_core']:.6f}",
        "margin_in": fmt(in_max),
        "margin_out": fmt(out_min) if out_min is not None else "inf",
    }


def part_bench() -> List[dict]:
    print("\n== Benchmark kernels, lambda = 1/2 ==", flush=True)
    rows: List[dict] = []
    summaries: List[str] = []
    mcc_exact_lines: List[str] = []
    for cfg in _configs():
        ctx = MM.exact() if cfg["exact"] else MM.floating(1e-9)
        print(f"[{cfg['tag']}] context={ctx.name}", flush=True)
        t0 = time.perf_counter()
        G = _build_cfg(cfg, ctx)
        out = analyze(G, cfg["xe"], cfg["ue"], LAM, ctx, anchor=cfg["anchor"])
        elapsed = time.perf_counter() - t0
        walked = out["walked"]
        bc = [c for c in walked["comps"] if c["verdict"]]
        check(f"{cfg['tag']} has a beneficial cycle", len(bc) >= 1, f"n={len(bc)}")
        joint_global = all(out["tests"][c["cid"]]["joint"] for c in bc)
        print(f"   |Vs|={len(walked['R'])} BC={len(bc)} gap={len(out['gap'])} "
              f"joint_global={joint_global} max_excess={fmt(out['max_global'])} "
              f"n_excess={out['n_global']} t={elapsed:.2f}s", flush=True)
        if joint_global:
            check(f"{cfg['tag']} global excess 0",
                  out["max_global"] == 0 or (isinstance(out["max_global"], float)
                                            and abs(out["max_global"]) <= 1e-8),
                  f"max={fmt(out['max_global'])} n={out['n_global']}")
        else:
            check(f"{cfg['tag']} global excess positive",
                  out["n_global"] > 0, f"max={fmt(out['max_global'])}")
        for comp in bc:
            test = out["tests"][comp["cid"]]
            in_max, out_min = margins(test["arcs"], test["zero"], ctx)
            check(f"{cfg['tag']} |S|={len(comp['S'])} G0 excesses ~ 0",
                  (in_max == 0) if ctx.name == "exact" else float(in_max) <= 1e-8,
                  f"margin_in={fmt(in_max)}")
            if out_min is not None:
                check(f"{cfg['tag']} |S|={len(comp['S'])} outside gap",
                      float(out_min) > 1e-6, f"margin_out={fmt(out_min)}")
            enum = None
            if ctx.name == "exact" and (not test["joint"] or len(comp["S"]) <= 8):
                print(f"   enumerating in-trees on |S|={len(comp['S'])}", flush=True)
                enum = enumerate_exc_star(comp, walked["J"], walked["lam"])
                check(f"{cfg['tag']} Held-Karp agrees", enum["hk_agree"] is True)
                if test["joint"]:
                    check(f"{cfg['tag']} Exc*=0", enum["exc_star"] == 0, f"{enum['exc_star']}")
                else:
                    check(f"{cfg['tag']} Exc*>0", enum["exc_star"] is not None and enum["exc_star"] > 0,
                          f"{enum['exc_star']}")
                    tree = test["fallback"]
                    worst, _n, _p = component_excess(comp, tree, walked["J"], walked["lam"], ctx)
                    check(f"{cfg['tag']} fallback >= Exc*",
                          worst + Fraction(0) >= enum["exc_star"],
                          f"fallback={fmt(worst)} Exc*={fmt(enum['exc_star'])}")
            row = _row_for_component(cfg, comp, test, walked["J"], walked["lam"], ctx,
                                    len(out["gap"]), enum)
            row["max_excess_global"] = fmt(out["max_global"])
            row["n_excess_global"] = out["n_global"]
            row["joint_global"] = joint_global
            row["|V_s|"] = len(walked["R"])
            rows.append(row)
            print(f"   |S|={row['|S|']} |G0|={row['|G_0|']} reach={row['n_reach']}/"
                  f"{row['|S|']} joint={row['joint']} excess_S={row['max_excess_S']} "
                  f"t_test={row['t_test_s']} t_core={row['t_core_s']}", flush=True)
            if cfg["exact"] and not test["joint"]:
                mcc_exact_lines.append(
                    f"config={cfg['tag']} |S|={len(comp['S'])} joint={test['joint']} "
                    f"|arcs|={len(test['arcs'])} |G_0|={len(test['zero'])} "
                    f"reach={len(test['good'])} fallback_excess={row['max_excess_S']} "
                    f"n_excess_S={row['n_excess_S']} Exc*={row['exc_star']} "
                    f"n_trees={row['n_trees']} n_pos_best={row['n_pos_best']} "
                    f"gap={len(out['gap'])} max_excess_global={fmt(out['max_global'])} "
                    f"n_excess_global={out['n_global']}"
                )
                _write_mcc_arcs(comp, test, walked, ctx)
        # Float cross-check of the exact mammalian run: same verdict.
        if cfg["exact"]:
            print("   float cross-check", flush=True)
            Gf = _build_cfg(cfg, MM.floating(1e-9))
            out_f = analyze(Gf, cfg["xe"], cfg["ue"], LAM, MM.floating(1e-9),
                            anchor=cfg["anchor"], time_test=False)
            bc_f = [c for c in out_f["walked"]["comps"] if c["verdict"]]
            joints = [out_f["tests"][c["cid"]]["joint"] for c in bc_f]
            check(f"{cfg['tag']} float verdict matches exact",
                  len(bc_f) == len(bc) and joints == [False],
                  f"float BC={len(bc_f)} joints={joints}")
            check(f"{cfg['tag']} float gap matches",
                  len(out_f["gap"]) == len(out["gap"]),
                  f"float {len(out_f['gap'])} exact {len(out['gap'])}")
        summaries.append(
            f"SUMMARY {cfg['tag']}: |Vs|={len(walked['R'])} gap={len(out['gap'])} "
            f"BC={len(bc)} joint_global={joint_global} "
            f"max_excess_global={fmt(out['max_global'])} n_excess_global={out['n_global']}"
        )
    # Anchors from the earlier exp21 runs. These are pipeline facts, not G_0 facts;
    # a mismatch means the walk diverged, not that the test is wrong.
    anchors = {
        "T-LGL (natural cost)": (1216, 2, True),
        "T-LGL (decoy)": (1240, 2, True),
        "mammalian cell cycle (decoy)": (234, 1, False),
        "ara, restricted inputs (decoy)": (6, 1, True),
    }
    by_cfg: Dict[str, List[dict]] = defaultdict(list)
    for r in rows:
        by_cfg[r["config"]].append(r)
    for tag, (gap, n_bc, joint) in anchors.items():
        got = by_cfg.get(tag, [])
        check(f"anchor {tag} gap", got and got[0]["gap"] == gap, f"got {got[0]['gap'] if got else None}")
        check(f"anchor {tag} n_bc", len(got) == n_bc, f"got {len(got)}")
        check(f"anchor {tag} joint", all(r["joint"] == joint for r in got),
              f"got {[r['joint'] for r in got]}")
    _write_bench(rows, summaries)
    if mcc_exact_lines:
        _write("exp23_mcc_exact.txt",
               "E23 mammalian decoy, lambda=1/2, eps=1/20, exact Fractions\n  "
               + "\n  ".join(mcc_exact_lines) + "\n")
    return rows


def _write_mcc_arcs(comp, test, walked, ctx) -> None:
    J, lam = walked["J"], walked["lam"]
    tree = test["fallback"]
    _worst, _n, per = component_excess(comp, tree, J, lam, ctx)
    zero_ids = {id(a) for a in test["zero"]}
    arc_path = os.path.join(RESULTS, "exp23_mcc_G0_arcs.csv")
    with open(arc_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["x", "y", "u", "g", "eps", "in_G0", "sign"])
        for a in test["arcs"]:
            kind, x, y, u, weight, eps = a
            sign = 0 if id(a) in zero_ids else (-1 if eps < 0 else 1)
            w.writerow([x, "omega" if kind == "exit" else y,
                        "" if u is None else u, fmt(weight), fmt(eps),
                        int(id(a) in zero_ids), sign])
    node_path = os.path.join(RESULTS, "exp23_mcc_G0_nodes.csv")
    with open(node_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["x", "reachable_in_G0", "J", "recovered_successor", "recovered_excess"])
        for x in sorted(comp["S"]):
            arc = tree[x]
            succ = "omega" if arc[0] == "exit" else arc[2]
            w.writerow([x, int(x in test["good"]), fmt(J[x]), succ, fmt(per.get(x, 0))])


def _write_bench(rows: List[dict], summaries: List[str]) -> None:
    csv_path = os.path.join(RESULTS, "exp23_bench.csv")
    with open(csv_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for r in rows:
            out = dict(r)
            out["states"] = out["states"]  # already a string
            writer.writerow(out)
    # Table body consumed by sections/examples/03_joint_feedback.tex.
    # Columns: network, |S|, |G_0|, reach, Problem 2, |gap|, recovered excess,
    # enumeration min-max excess, t_test, t_core.
    body = []
    for r in rows:
        name = _tex_name(r)
        body.append(
            f"{name} & {r['|S|']} & {r['|G_0|']} & {r['n_reach']} & "
            f"{'yes' if r['joint'] else 'no'} & {r['gap']} & "
            f"${r['max_excess_S']}$ & ${r['exc_star']}$ & "
            f"{r['t_test_s']} & {r['t_core_s']} \\\\"
        )
    _write("exp23_bench_table.tex", "\n".join(body) + "\n")
    text = "E23 benchmark, lambda=1/2\n" + "\n".join(summaries) + "\n"
    for r in rows:
        text += f"  { {k: r[k] for k in r if k != 'states'} }\n"
    _write("exp23_bench.txt", text)


def _tex_name(r: dict) -> str:
    tag = r["config"]
    if tag.startswith("T-LGL (natural"):
        base = "T-LGL, natural"
    elif tag.startswith("T-LGL (decoy"):
        base = "T-LGL, decoy"
    elif tag.startswith("mammalian"):
        base = "cell cycle, decoy"
    else:
        base = "ara, restricted, decoy"
    # Two kernels of one network: mark the one that contains the literature
    # survival state, and number the other.
    if "T-LGL" in base:
        base += ", with $\\delta^{58834}$" if r["contains_58834"] else ", other"
    return base


# ---------------------------------------------------------------------------
# Part C — lambda sweep
# ---------------------------------------------------------------------------


def _sweep_grid(tail: Sequence[Fraction] = ()) -> List[Fraction]:
    vals = [Fraction(1, 100)]
    vals += [Fraction(k, 20) for k in range(1, 20)]
    vals.append(Fraction(99, 100))
    vals.extend(tail)
    return vals


def _tracked_record(out: dict, ctx: MM.NumericContext) -> dict:
    comp = out["tracked"]
    test = out["tests"][comp["cid"]]
    tree = test["tree"] if test["joint"] else test["fallback"]
    worst, n_pos, _per = component_excess(comp, tree, out["walked"]["J"],
                                         out["walked"]["lam"], ctx)
    bc = [c for c in out["walked"]["comps"] if c["verdict"]]
    joint_global = all(out["tests"][c["cid"]]["joint"] for c in bc)
    return {
        "has_cycle": int(comp["verdict"]),
        "gap": len(out["gap"]),
        "n_S": len(comp["S"]),
        "n_reach": len(test["good"]),
        "n_unreach": len(comp["S"]) - len(test["good"]),
        "joint": int(test["joint"]),
        "joint_global": int(joint_global),
        "max_excess_S": worst,
        "n_excess_S": n_pos,
        "max_excess_global": out["max_global"],
        "n_excess_global": out["n_global"],
    }


def _eval_lambda(build, xe, ue, anchor, lam: Fraction, ctx: MM.NumericContext) -> dict:
    G = build(ctx)
    out = analyze(G, xe, ue, lam, ctx, anchor=anchor, time_test=False)
    rec = _tracked_record(out, ctx)
    rec["lam"] = lam
    return rec


def _crossing(records: List[dict], key: str, high_is_one: bool):
    """Bracket a monotone 0/1 switch. high_is_one: the predicate is 0 on the
    left of the bracket and 1 on the right (lambda increasing)."""
    seq = [r[key] for r in records]
    if all(v == seq[0] for v in seq):
        return {"monotone": True, "crossing": False, "value": seq[0]}
    flips = []
    for a, b in zip(records, records[1:]):
        if a[key] != b[key]:
            flips.append((a, b))
    monotone = len(flips) == 1
    return {"monotone": monotone, "crossing": True, "flips": flips}


def _bisect(build, xe, ue, anchor, ctx, lo: Fraction, hi: Fraction,
            key: str, want_at_hi: int) -> Tuple[Fraction, Fraction]:
    """Shrink [lo, hi] until the width is below 1e-4. The predicate equals
    1 - want_at_hi at lo and want_at_hi at hi, lambda increasing."""
    for _ in range(24):
        if hi - lo < Fraction(1, 1000000):
            break
        mid = (lo + hi) / 2
        rec = _eval_lambda(build, xe, ue, anchor, mid, ctx)
        if rec[key] == want_at_hi:
            hi = mid
        else:
            lo = mid
    return lo, hi


def _sweep_instance(name: str, build, xe, ue, anchor, ctx: MM.NumericContext,
                    grid: Optional[Sequence[Fraction]] = None) -> List[dict]:
    print(f"\n== lambda sweep: {name} ({ctx.name}) ==", flush=True)
    records = []
    for lam in (grid if grid is not None else _sweep_grid()):
        rec = _eval_lambda(build, xe, ue, anchor, lam, ctx)
        records.append(rec)
        print(f"  lam={fmt(lam):8} bc={rec['has_cycle']} joint={rec['joint']} "
              f"gap={rec['gap']:4} unreach={rec['n_unreach']:3} "
              f"excess_S={fmt(rec['max_excess_S'])} "
              f"excess_global={fmt(rec['max_excess_global'])}", flush=True)
    return records


def _refine(name: str, records, build, xe, ue, anchor, ctx, lines: List[str]):
    """Returns (extra endpoint records, brackets). A bracket is the last
    evaluated lambda where the predicate still has its left-hand value and
    the first where it has its right-hand value."""
    extra = []
    brackets = {}
    for key, label in (("has_cycle", "lambda_bc"), ("joint", "lambda_joint")):
        info = _crossing(records, key, high_is_one=(key == "joint"))
        if not info["monotone"]:
            lines.append(f"{name} {label}: NOT monotone; grid flips "
                         + ", ".join(f"({fmt(a['lam'])}->{fmt(b['lam'])}: {a[key]}->{b[key]})"
                                     for a, b in info["flips"]))
            continue
        if not info["crossing"]:
            lines.append(f"{name} {label}: no crossing on the grid; "
                         f"{key}={info['value']} from lam={fmt(records[0]['lam'])} "
                         f"to lam={fmt(records[-1]['lam'])}")
            continue
        left, right = info["flips"][0]
        want_hi = right[key]
        lo, hi = _bisect(build, xe, ue, anchor, ctx, left["lam"], right["lam"], key, want_hi)
        lines.append(f"{name} {label}: {key}={left[key]} at lam={fmt(lo)}; "
                     f"{key}={right[key]} at lam={fmt(hi)}; width={fmt(hi - lo)}")
        brackets[label] = (lo, hi)
        extra.append(_eval_lambda(build, xe, ue, anchor, lo, ctx) | {"note": f"{label}-lo"})
        extra.append(_eval_lambda(build, xe, ue, anchor, hi, ctx) | {"note": f"{label}-hi"})
    return extra, brackets


def part_sweep(ex1_done: bool = False) -> None:
    ctx = MM.exact()
    lines = ["E23 lambda sweep, exact Fractions. "
             "lambda_bc: beneficial cycle on the tracked component disappears. "
             "lambda_joint: G_0 starts to span."]

    def build_ex1(_ctx):
        return build_example1(_ctx)

    rec_ex1 = _sweep_instance("example1-H", build_ex1, XE, UE, 2, ctx)
    extra_ex1, brackets_ex1 = _refine("example1-H", rec_ex1, build_ex1, XE, UE, 2, ctx, lines)
    # Component B = {6,7,10} is benign at lambda=1/2 and is the one that
    # crosses both thresholds as lambda grows.
    rec_b = _sweep_instance("example1-B", build_ex1, XE, UE, 6, ctx)
    extra_b, _brackets_b = _refine("example1-B", rec_b, build_ex1, XE, UE, 6, ctx, lines)

    def build_mcc(_ctx):
        cfg = next(c for c in _configs() if c["tag"].startswith("mammalian"))
        return _build_cfg(cfg, _ctx)

    cfg_m = next(c for c in _configs() if c["tag"].startswith("mammalian"))
    grid_m = _sweep_grid((Fraction(999, 1000), Fraction(9999, 10000)))
    rec_m = _sweep_instance("mammalian-decoy", build_mcc, cfg_m["xe"], cfg_m["ue"],
                            cfg_m["anchor"], ctx, grid_m)
    extra_m, _brackets_m = _refine(
        "mammalian-decoy", rec_m, build_mcc, cfg_m["xe"], cfg_m["ue"],
        cfg_m["anchor"], ctx, lines)

    # The cycle on H of Example 1 is beneficial exactly while 3*lam^2 - lam - 1 < 0,
    # i.e. while lam is below (1 + sqrt(13))/6. The bracket has to contain that root.
    root = (1 + _sqrt_frac(13)) / 6
    lines.append(f"example1-H algebraic cycle root (1+sqrt(13))/6 ≈ {float(root):.12f}")
    if "lambda_bc" in brackets_ex1:
        lo, hi = brackets_ex1["lambda_bc"]
        check("example1 cycle root inside the has_cycle bracket",
              lo < root < hi, f"bracket ({float(lo):.8f}, {float(hi):.8f}) root={float(root):.12f}")
    _write("exp23_thresholds.txt", "\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)

    rows = []
    for instance, recs, extras in (
        ("example1-H", rec_ex1, extra_ex1),
        ("example1-B", rec_b, extra_b),
        ("mammalian-decoy", rec_m, extra_m),
    ):
        for rec in recs:
            rows.append(_sweep_row(instance, rec, "grid"))
        for rec in extras:
            if "has_cycle" not in rec:
                continue
            rows.append(_sweep_row(instance, rec, rec.get("note", "refined")))
    path = os.path.join(RESULTS, "exp23_lambda_sweep.csv")
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    check("example1 sweep wrote a has_cycle crossing or a constant",
          any("lambda_bc" in ln for ln in lines))


def _sqrt_frac(n: int) -> Fraction:
    """Integer square root is not enough; keep a rational enclosure only for
    the printed algebraic value. The threshold test uses the float of the
    true root solely as a comment. Returned value is a high-precision
    rational approximation (error < 5e-13) for the log line."""
    x = Fraction(n)
    # Newton on rationals, a few steps from a float seed.
    y = Fraction(float(n) ** 0.5)
    for _ in range(6):
        y = (y + x / y) / 2
    return y


def _sweep_row(instance: str, rec: dict, note: str) -> dict:
    lam = rec["lam"]
    return {
        "instance": instance,
        "note": note,
        "lam_exact": fmt(lam),
        "lam": f"{float(lam):.10f}",
        "has_cycle": rec["has_cycle"],
        "joint": rec["joint"],
        "gap": rec["gap"],
        "n_S": rec["n_S"],
        "n_reach": rec["n_reach"],
        "n_unreach": rec["n_unreach"],
        "joint_global": rec["joint_global"],
        "max_excess_S": fmt(rec["max_excess_S"]),
        "max_excess_S_float": f"{float(rec['max_excess_S']):.8g}",
        "n_excess_S": rec["n_excess_S"],
        "max_excess_global": fmt(rec["max_excess_global"]),
        "max_excess_global_float": f"{float(rec['max_excess_global']):.8g}",
        "n_excess_global": rec["n_excess_global"],
    }


def _write(name: str, text: str) -> None:
    os.makedirs(RESULTS, exist_ok=True)
    with open(os.path.join(RESULTS, name), "w") as fh:
        fh.write(text)


def main() -> int:
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    os.makedirs(RESULTS, exist_ok=True)
    if which in ("all", "ex1", "sweep"):
        part_ex1()
    if which in ("all", "bench"):
        part_bench()
    if which in ("all", "sweep"):
        part_sweep()
    print("\n== assertions ==")
    if fails:
        print(f"{len(fails)} FAILED")
        for f in fails:
            print(" ", f)
        return 1
    print("ALL PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
