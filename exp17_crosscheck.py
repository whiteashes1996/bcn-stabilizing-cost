#!/usr/bin/env python3
"""exp17 crosscheck: C++ solver (floating 1e-9 semantics) vs methods.py exact().

Three layers (plan section 5, tolerance tiers per the section-8b amendment):
  1. Example 1 fixed point (exp14 golden values, three-way agreement);
  2. random small grid: n in {4,5,6} x m in {1,2} x core in {0,3,5,8}
     x lambda in {1/2, 9/10} x rep in {0..4}  (240 instances);
  3. curve-(c) generator-path sample: n in {4,5,6}, rep 0..4 (15 instances).

Value comparison tiers:
  dyadic instance (lambda and all decoy costs dyadic): if the exact value is
  representable in float64 (Fraction(float(v)) == v) -> bitwise equality (==);
  otherwise relative error <= 4 ulp, instance recorded (expected: zero hits).
  non-dyadic instance (lambda = 9/10): relative error <= 1e-9.
Integer counters (sweeps, strict_sweeps, k_vi, core_evals): equal on dyadic
instances; |delta| <= 1 tolerated and recorded on non-dyadic ones (plan
section 7 risk 5).  Verdict-class quantities are compared with zero tolerance
everywhere.

Also runs B1 (determinism: regenerate/resolve byte-identical) and C3 (debug
ASan+UBSan binary reproduces every release result file byte for byte).
"""
from __future__ import annotations

import math
import os
import subprocess
import sys
import time
from fractions import Fraction
from typing import Dict, FrozenSet, List, Optional, Set, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import methods as MM  # noqa: E402
import exp14_ex1_n4 as E1  # noqa: E402

CPP_DIR = os.path.join(HERE, "cpp")
BCN = os.path.join(CPP_DIR, "build", "bcn")
BCN_DBG = os.path.join(CPP_DIR, "build", "bcn_debug")
XDIR = os.path.join(HERE, "results", "exp17_crosscheck")
SUMMARY = os.path.join(HERE, "results", "exp17_crosscheck.txt")
INF = math.inf

passed: List[str] = []
failed: List[str] = []
notes: List[str] = []
tier2_hits: List[str] = []
int_drift: List[str] = []


def check(tag: str, cond: bool, detail: str = "") -> None:
    (passed if cond else failed).append(f"{tag}{' ' + detail if detail and not cond else ''}")
    if not cond:
        print(f"FAIL {tag}: {detail}")


def note(s: str) -> None:
    notes.append(s)


def run_cpp(args: List[str], binary: str = BCN) -> str:
    r = subprocess.run([binary] + args, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"cpp failed: {args}\n{r.stdout}\n{r.stderr}")
    return r.stdout


# ---------------------------------------------------------------------------
# instance / result file parsing
# ---------------------------------------------------------------------------
def parse_instance(path: str) -> dict:
    toks: List[str] = []
    with open(path) as f:
        for line in f:
            if line.startswith("#"):
                continue
            toks.extend(line.split())
    it = iter(toks)
    out: dict = {"cores": [], "benigns": [], "decoys": {}}
    while True:
        try:
            t = next(it)
        except StopIteration:
            break
        if t == "config":
            out["config"] = next(it)
        elif t == "seed":
            out["seed"] = int(next(it))
        elif t in ("n", "m", "xe", "ue"):
            out[t] = int(next(it))
        elif t == "lam":
            out["lam"] = Fraction(int(next(it)), int(next(it)))
        elif t in ("cores", "benigns"):
            k = int(next(it))
            sets = []
            for _ in range(k):
                sz = int(next(it))
                sets.append([int(next(it)) for _ in range(sz)])
            out[t] = sets
        elif t == "decoys":
            k = int(next(it))
            for _ in range(k):
                x, u, p, q = (int(next(it)) for _ in range(4))
                out["decoys"][(x, u)] = Fraction(p, q)
        elif t == "L":
            N, M = 1 << out["n"], 1 << out["m"]
            out["L"] = [int(next(it)) for _ in range(N * M)]
            out["N"], out["M"] = N, M
    return out


def parse_result(path: str) -> dict:
    out: dict = {"comps": [], "J": {}, "mu": {}, "VI": {}, "PIV": {}}
    with open(path) as f:
        for line in f:
            p = line.split()
            if not p:
                continue
            if p[0] == "META":
                d = dict(zip(p[1::2], p[2::2]))
                out["meta"] = {k: int(v) for k, v in d.items()}
            elif p[0] == "COMP":
                sep = p.index(":")
                out["comps"].append(dict(
                    size=int(p[1]), has_cycle=bool(int(p[2])),
                    sweep_verdict=bool(int(p[3])), disagreement=bool(int(p[4])),
                    sweeps=int(p[5]), strict=int(p[6]),
                    margin=None if p[7] == "none" else float(p[7]),
                    evals=int(p[8]),
                    members=frozenset(int(v) for v in p[sep + 1:])))
            elif p[0] == "S":
                x = int(p[1])
                out["J"][x] = float(p[2])
                if int(p[3]) > 0:
                    out["mu"][x] = (int(p[3]), int(p[4]))
            elif p[0] == "V":
                out["VI"][int(p[1])] = float(p[2])
            elif p[0] == "P":
                out["PIV"][int(p[1])] = float(p[2])
    return out


# ---------------------------------------------------------------------------
# exact side
# ---------------------------------------------------------------------------
def cost_factory(N: int, xe: int, ue: int, decoys: Dict[Tuple[int, int], Fraction]):
    xb = N - xe

    def g(x: int, u: int):
        d = decoys.get((x, u))
        if d is not None:
            return d
        return bin((N - x) ^ xb).count("1") + (0 if u == ue else 1)

    return g


def is_pow2(q: int) -> bool:
    return q > 0 and (q & (q - 1)) == 0


def instance_dyadic(inst: dict) -> bool:
    if not is_pow2(inst["lam"].denominator):
        return False
    return all(is_pow2(v.denominator) for v in inst["decoys"].values())


def cmp_value(tag: str, cpp: float, frac, dyadic: bool) -> bool:
    """Tiered value comparison (plan section 8b amendment)."""
    inf_c = math.isinf(cpp) or cpp >= 1e300
    inf_p = frac is None or (isinstance(frac, float) and math.isinf(frac))
    if inf_c or inf_p:
        return inf_c and inf_p
    v = float(frac)
    if dyadic:
        if Fraction(v) == frac:            # exactly representable -> bitwise
            return cpp == v
        # 4-ulp tier: fires on values with non-dyadic exact form (e.g. the
        # unconstrained V* on a cycle carries a 1/(1-lam^ell) factor even
        # under dyadic data); record the observed deviation in ulps.
        d = abs(cpp - v)
        tier2_hits.append(f"{tag} delta={d / math.ulp(abs(v)):.1f}ulp")
        return d <= 4 * math.ulp(abs(v))
    return abs(cpp - v) <= 1e-9 * max(1.0, abs(cpp), abs(v))


def cmp_int(tag: str, a: int, b: int, dyadic: bool) -> bool:
    if a == b:
        return True
    if not dyadic and abs(a - b) <= 1:
        int_drift.append(f"{tag}: cpp={a} py={b}")
        return True
    return False


def vi_until_exact(G, R, xe, lam, ctx, k_cap: int = 4096):
    """exp16.vi_until replica in exact arithmetic; the threshold COMPARISON
    mirrors the C++ float threshold bitwise (thr computed in float64, residual
    converted through float, exact on dyadic data)."""
    lam_f = float(lam)
    thr = 1e-6 * (1.0 - lam_f) / (2.0 * lam_f)
    kb = 64
    while True:
        seq = MM.vi(G, R, xe, lam, ctx, kb)
        for k in range(1, len(seq)):
            worst = 0.0
            bad = False
            for x in R:
                a, b = seq[k - 1].get(x, INF), seq[k].get(x, INF)
                ia, ib = a == INF, b == INF
                if ia and ib:
                    continue
                if ia != ib:
                    bad = True
                    break
                worst = max(worst, abs(float(a - b)))
            if not bad and worst <= thr:
                return seq, k
        if kb >= k_cap:
            raise RuntimeError("exact VI did not stop")
        kb *= 2


# ---------------------------------------------------------------------------
# one instance crosscheck
# ---------------------------------------------------------------------------
def crosscheck(tag: str, inst_path: str, res_path: str,
               golden: bool = False) -> None:
    inst = parse_instance(inst_path)
    res = parse_result(res_path)
    N, M, xe, ue, lam = inst["N"], inst["M"], inst["xe"], inst["ue"], inst["lam"]
    dyadic = instance_dyadic(inst)
    ctx = MM.exact()
    g = cost_factory(N, xe, ue, inst["decoys"])
    G = MM.stg_from_L(inst["L"], N, M, g, ctx)
    ns = lambda x, u: inst["L"][(u - 1) * N + (x - 1)]  # noqa: E731

    mr = MM.run_master(G, xe, lam, ctx, ue=ue)
    R = mr.Vs

    # --- Vs / J table -----------------------------------------------------
    check(f"{tag}/Vs", set(res["J"].keys()) == R and res["meta"]["vs"] == len(R))
    okJ = all(cmp_value(f"{tag}/J[{x}]", res["J"][x], mr.J.get(x, INF), dyadic)
              for x in R)
    check(f"{tag}/J", okJ)

    # --- components: partition, verdicts, sweep counters, evals ------------
    py_comps = {frozenset(r.S): r for r in mr.records}
    cpp_comps = {c["members"]: c for c in res["comps"]}
    check(f"{tag}/comp-partition", set(py_comps) == set(cpp_comps))
    ok_v = ok_s = ok_e = True
    dis = 0
    for mset, pr in py_comps.items():
        cr = cpp_comps.get(mset)
        if cr is None:
            ok_v = False
            continue
        # exact verdict (sweep, theorem-backed) == C++ pi-tight verdict
        ok_v &= (cr["has_cycle"] == pr.has_cycle)
        ok_v &= (cr["sweep_verdict"] == pr.sweep_verdict)
        ok_s &= cmp_int(f"{tag}/sweeps{sorted(mset)[:2]}", cr["sweeps"], pr.sweeps, dyadic)
        ok_s &= cmp_int(f"{tag}/strict{sorted(mset)[:2]}", cr["strict"], pr.strict_sweeps, dyadic)
        if cr["has_cycle"]:
            n_ = len(mset)
            ok_e &= (cr["evals"] == n_ * (1 << (n_ - 1)))
        dis += 1 if cr["disagreement"] else 0
    check(f"{tag}/verdicts", ok_v)
    check(f"{tag}/sweep-counters", ok_s)
    check(f"{tag}/core-evals", ok_e)
    check(f"{tag}/meta-disagree", res["meta"]["disagree"] == dis)
    if dyadic:
        check(f"{tag}/dyadic-no-disagree", dis == 0)
    elif dis:
        note(f"{tag}: {dis} sweep-vs-pi-tight disagreement(s) on non-dyadic data")

    # --- planted structure (B4) --------------------------------------------
    planted_core = {frozenset(s) for s in inst["cores"]}
    planted_all = planted_core | {frozenset(s) for s in inst["benigns"]}
    nontrivial = {m for m in cpp_comps if len(m) > 1}
    check(f"{tag}/planted-sccs", nontrivial == planted_all)
    check(f"{tag}/planted-verdicts",
          {m for m, c in cpp_comps.items() if c["has_cycle"]} == planted_core)
    check(f"{tag}/bc-count", res["meta"]["bc"] == len(planted_core))

    # --- C++ feedback ------------------------------------------------------
    # Strongest form: the C++ mu is IDENTICAL to the Python exact-context mu
    # (same tie-breaks throughout, including the in-tree fallback).  Theory
    # checks on top: proper everywhere; closed-loop cost >= J pointwise
    # (Fraction-exact); == J whenever the closed loop avoids beneficial-core
    # states (inside/through a core the in-tree fallback may exceed J and the
    # excess propagates upstream along the loop -- Prop. no-joint; the plan's
    # blanket "== J off-core" was an overstatement, amended in section 10).
    mu_cpp = dict(res["mu"])
    check(f"{tag}/mu-domain", set(mu_cpp) == set(mr.mu))
    imp = MM.improper_states(ns, mu_cpp, R, xe)
    check(f"{tag}/mu-proper", imp == set(), f"improper={sorted(imp)[:6]}")
    check(f"{tag}/meta-improper-master", res["meta"]["improper_master"] == 0)
    core_states: Set[int] = set()
    for m, c in cpp_comps.items():
        if c["has_cycle"]:
            core_states |= set(m)

    def visits_core(x0: int) -> bool:
        cur = x0
        while cur != xe:
            if cur in core_states:
                return True
            cur = ns(cur, mu_cpp[cur][0])
        return False

    ok_ge = ok_eq = True
    excess = 0
    for x in sorted(R):
        if x == xe:
            continue
        cost = MM.closed_loop_cost(ns, g, mu_cpp, x, xe, lam, ctx)
        jx = mr.J.get(x, INF)
        ok_ge &= not (cost < jx)
        if visits_core(x):
            if cost != jx:
                excess += 1
        else:
            ok_eq &= (cost == jx)
    check(f"{tag}/mu-cost-ge-J", ok_ge)
    check(f"{tag}/mu-cost-eq-off-core-paths", ok_eq)
    if excess:
        note(f"{tag}: mu exceeds J at {excess} state(s) whose loop crosses a "
             f"beneficial core (in-tree fallback, Prop. no-joint)")
    # mu arc identity: equal up to ties.  Where the two selections differ
    # outside beneficial comps, BOTH arcs must be Bellman-tight at x in exact
    # arithmetic (equal-value in-tree branches; BFS-order dependent).
    ties = core_ties = 0
    ok_tie = True
    for x in set(mu_cpp) | set(mr.mu):
        if x == xe or mu_cpp.get(x) == mr.mu.get(x):
            continue
        if x in core_states:
            core_ties += 1  # in-tree fallback order difference; cost>=J holds
            continue
        ties += 1
        jx = mr.J.get(x, INF)
        for (u, y) in (mu_cpp[x], mr.mu[x]):
            jy = mr.J.get(y, INF)
            ok_tie &= (jy != INF) and (g(x, u) + lam * jy == jx)
    check(f"{tag}/mu-ties-tight", ok_tie)
    if ties or core_ties:
        note(f"{tag}: mu differs from Python at {ties} benign-tie state(s) "
             f"and {core_ties} core in-tree state(s) (order-dependent ties)")

    # --- VI: stop index and final table ------------------------------------
    seq, k_py = vi_until_exact(G, R, xe, lam, ctx)
    check(f"{tag}/vi-k", cmp_int(f"{tag}/vi-k", res["meta"]["vi_k"], k_py, dyadic))
    k_use = min(res["meta"]["vi_k"], len(seq) - 1)
    Vk = seq[k_use]
    check(f"{tag}/vi-table",
          all(cmp_value(f"{tag}/VI[{x}]", res["VI"][x], Vk.get(x, INF), dyadic)
              for x in R))
    adjR = MM.restricted_adjacency(G, R)
    mu_vi = MM.greedy_mu(Vk, adjR, xe, lam, ctx, ns, ue)
    imp_vi = MM.improper_states(ns, mu_vi, R, xe)
    check(f"{tag}/vi-improper", res["meta"]["improper_vi"] == len(imp_vi))

    # --- PI table -----------------------------------------------------------
    V_pi, _mu_pi = MM.pi(G, R, xe, lam, ctx)
    check(f"{tag}/pi-table",
          all(cmp_value(f"{tag}/PI[{x}]", res["PIV"][x], V_pi.get(x, INF), dyadic)
              for x in R))

    # --- margins: non-dyadic branch audit (B3: every comp IS exact-checked) -
    if not dyadic:
        small = [c for c in res["comps"]
                 if c["margin"] is not None and c["margin"] < 1e-6]
        if small:
            note(f"{tag}: {len(small)} component(s) with tight margin < 1e-6 "
                 f"(exact recheck = this very comparison, verdicts agreed)")

    # --- Example 1 golden layer (B2) ----------------------------------------
    if golden:
        for x in range(1, 17):
            exp = E1.EXPECTED_J.get(x)
            got = res["J"].get(x)
            if exp is None:
                check(f"{tag}/golden-J[{x}]-inf", got is None or math.isinf(got))
            else:
                check(f"{tag}/golden-J[{x}]", got is not None and got == float(exp))
        check(f"{tag}/golden-comps", len(res["comps"]) == 9)  # 10 minus {xe}
        hb = {frozenset(E1.H_SCC): (3, 3, True), frozenset(E1.B_SCC): (3, 2, False)}
        for mset, (sw, st, cyc) in hb.items():
            c = cpp_comps[mset]
            check(f"{tag}/golden-sweeps{sorted(mset)}",
                  (c["sweeps"], c["strict"], c["has_cycle"]) == (sw, st, cyc))
        check(f"{tag}/golden-improper-vi",
              res["meta"]["improper_vi"] == len(E1.EXPECTED_IMPROPER)
              and imp_vi == set(E1.EXPECTED_IMPROPER))
        for x in range(1, 17):  # E1 special case: mu attains J everywhere
            if x in R and x != xe:
                cost = MM.closed_loop_cost(ns, g, mu_cpp, x, xe, lam, ctx)
                check(f"{tag}/golden-mu-cost[{x}]", cost == mr.J[x])
        for (x, u), exp_g in [((3, 1), E1.g(3, 1)), ((8, 2), E1.g(8, 2))]:
            check(f"{tag}/golden-cost({x},{u})", g(x, u) == exp_g)


# ---------------------------------------------------------------------------
# drivers
# ---------------------------------------------------------------------------
def gen_and_solve(name: str, gen_args: List[str], binary: str = BCN
                  ) -> Tuple[str, str]:
    inst = os.path.join(XDIR, f"inst_{name}.txt")
    res = os.path.join(XDIR, f"res_{name}.txt")
    run_cpp(["gen"] + gen_args + ["--out", inst], binary)
    run_cpp(["solve", "--inst", inst, "--out", res], binary)
    return inst, res


def e1_instance() -> Tuple[str, str]:
    inst = os.path.join(XDIR, "inst_e1.txt")
    with open(inst, "w") as f:
        f.write("# Example 1 fixed point (from exp14_ex1_n4, single source of truth)\n")
        f.write("config e1\nseed 0\nn 4\nm 1\nxe 1\nue 1\nlam 1 2\n")
        f.write("cores 1\n3 2 3 5\nbenigns 1\n3 6 7 10\ndecoys 0\nL")
        for v in E1.L:
            f.write(f" {v}")
        f.write("\n")
    res = os.path.join(XDIR, "res_e1.txt")
    run_cpp(["solve", "--inst", inst, "--out", res])
    return inst, res


GRID: List[Tuple[str, List[str]]] = []
for n in (4, 5, 6):
    for m in (1, 2):
        for core in (0, 3, 5, 8):
            for lamp, lamq in ((1, 2), (9, 10)):
                for rep in range(5):
                    name = f"g_n{n}m{m}c{core}l{lamp}_{lamq}r{rep}"
                    GRID.append((name, [
                        "--n", str(n), "--m", str(m),
                        "--cores", str(core) if core else "",
                        "--benign", "1", "--bsize", "4",
                        "--lamp", str(lamp), "--lamq", str(lamq),
                        "--rep", str(rep)]))

CURVE_C: List[Tuple[str, List[str]]] = []
for n in (4, 5, 6):
    for rep in range(5):
        name = f"c_n{n}r{rep}"
        CURVE_C.append((name, [
            "--n", str(n), "--m", "2", "--cores", "8",
            "--benign", "1", "--bsize", "8",
            "--lamp", "1", "--lamq", "2", "--rep", str(rep)]))


def main() -> int:
    os.makedirs(XDIR, exist_ok=True)
    t0 = time.time()
    subprocess.run(["make", "-C", CPP_DIR], check=True, capture_output=True)
    subprocess.run(["make", "-C", CPP_DIR, "debug"], check=True,
                   capture_output=True)

    # layer 1: Example 1 fixed point
    inst, res = e1_instance()
    crosscheck("e1", inst, res, golden=True)
    print(f"e1 done ({time.time() - t0:.1f}s)")

    # layers 2+3: random grid + curve-(c) generator path
    all_named = GRID + CURVE_C
    for i, (name, args) in enumerate(all_named):
        inst, res = gen_and_solve(name, args)
        crosscheck(name, inst, res)
        if (i + 1) % 40 == 0:
            print(f"{i + 1}/{len(all_named)} instances ({time.time() - t0:.1f}s)")

    # B1 determinism: regenerate + resolve, byte-compare
    for name, args in [GRID[7], GRID[123], CURVE_C[9]]:
        i1 = os.path.join(XDIR, f"det_{name}_1.txt")
        i2 = os.path.join(XDIR, f"det_{name}_2.txt")
        run_cpp(["gen"] + args + ["--out", i1])
        run_cpp(["gen"] + args + ["--out", i2])
        check(f"B1/gen/{name}", open(i1).read() == open(i2).read())
        r1 = os.path.join(XDIR, f"det_{name}_r1.txt")
        r2 = os.path.join(XDIR, f"det_{name}_r2.txt")
        run_cpp(["solve", "--inst", i1, "--out", r1])
        run_cpp(["solve", "--inst", i1, "--out", r2])
        check(f"B1/solve/{name}", open(r1).read() == open(r2).read())

    # C3: ASan+UBSan debug binary reproduces every release result bytewise
    dbg_fail = 0
    for name, _args in [("e1", None)] + all_named:  # type: ignore[list-item]
        inst = os.path.join(XDIR, f"inst_{name}.txt")
        res = os.path.join(XDIR, f"res_{name}.txt")
        resd = os.path.join(XDIR, f"resdbg_{name}.txt")
        try:
            run_cpp(["solve", "--inst", inst, "--out", resd], BCN_DBG)
        except RuntimeError as exc:
            dbg_fail += 1
            note(f"C3 sanitizer failure on {name}: {exc}")
            continue
        if open(res).read() != open(resd).read():
            dbg_fail += 1
            note(f"C3 debug/release result mismatch on {name}")
        os.remove(resd)
    check("C3/asan-ubsan-grid", dbg_fail == 0, f"{dbg_fail} failures")

    # summary
    with open(SUMMARY, "w") as f:
        f.write("exp17 crosscheck summary\n")
        f.write(f"instances: {1 + len(all_named)}  "
                f"(e1 + {len(GRID)} grid + {len(CURVE_C)} curve-c)\n")
        f.write(f"checks passed: {len(passed)}  failed: {len(failed)}\n")
        f.write(f"tier-2 (4 ulp) hits: {len(tier2_hits)}\n")
        for t in tier2_hits:
            f.write(f"  {t}\n")
        f.write(f"integer-counter drift (non-dyadic, |d|<=1): {len(int_drift)}\n")
        for t in int_drift:
            f.write(f"  {t}\n")
        f.write("notes:\n")
        for s in notes:
            f.write(f"  {s}\n")
        if failed:
            f.write("FAILED CHECKS:\n")
            for s in failed:
                f.write(f"  {s}\n")
        f.write(f"runtime: {time.time() - t0:.1f} s\n")
        f.write("RESULT: " + ("PASS" if not failed else "FAIL") + "\n")
    print(f"\npassed={len(passed)} failed={len(failed)} "
          f"tier2={len(tier2_hits)} drift={len(int_drift)} "
          f"({time.time() - t0:.1f}s)  -> {SUMMARY}")
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
