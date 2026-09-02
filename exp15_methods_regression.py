"""E15: regression and cross-check harness for methods.py (Task 0, plan §5-§6).

Three-way validation on the Example 1 instance (single source of truth:
exp14_ex1_n4, imported for L, the stage cost and every golden value):

  R1  methods + exact()  reproduces exp14's sixteen acceptance items (M02-M17);
  R2  methods + exact()  agrees pointwise with the core_solver route (M18);
  R3  methods + floating(1e-9) agrees with exact: values within 1e-9, same
      improper sets, same per-component beneficial-cycle verdicts, zero
      detector disagreements, healthy tight margins (M19-M21);
  R4  (optional, report-only) fuzz on random small folded components:
      exact verdict vs float verdicts agreement statistics;
  R5  smoke: exp10 and exp14 still pass as subprocesses (M24), plus source /
      git hygiene checks (M01, M22, M23, M25 interleaved below).

No figures; writes results/exp15_methods_regression.txt.
"""
from __future__ import annotations

import math
import os
import random
import re
import subprocess
import sys
import time
from fractions import Fraction
from typing import Dict, List, Set

sys.path.insert(0, os.path.dirname(__file__))

import exp14_ex1_n4 as E1  # noqa: E402  (constants + instance; import is side-effect free)
import methods as MM  # noqa: E402

RESULTS = os.path.join(os.path.dirname(__file__), "results")
SIM_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SIM_DIR)

XE, UE = E1.XE, E1.UE
LAM = Fraction(1, 2)
H, B = set(E1.H_SCC), set(E1.B_SCC)

passed: List[str] = []


def check(tag: str, cond: bool, detail: str = "") -> None:
    assert cond, f"{tag} FAILED {detail}"
    passed.append(f"{tag} PASS  {detail}".rstrip())


def build_graph(ctx: MM.NumericContext) -> MM.STG:
    return MM.stg_from_L(E1.L, E1.N, E1.M, E1.g, ctx)


# ---------------------------------------------------------------------------
# lambda* predicate through methods.py (folded H at rational lambda; J(8)=3)
# ---------------------------------------------------------------------------


def fold_H_at(lam: Fraction):
    internal = {
        2: [(1, 3, Fraction(1))],
        3: [(1, 5, Fraction(1)), (2, 5, Fraction(2))],
        5: [(1, 2, Fraction(1)), (2, 2, Fraction(2))],
    }
    e = {2: Fraction(2) + lam * 3, 3: MM.INF, 5: MM.INF}
    return internal, e


def beneficial_at_m(lam: Fraction) -> bool:
    """Exact predicate; asserts sweep verdict, pi-tight verdict and the
    closed-form criterion agree at every sample."""
    ctx = MM.exact()
    internal, e = fold_H_at(lam)
    easy = MM.solve_easy(H, internal, e, lam, ctx)
    det = MM.detect_via_pi(H, internal, e, lam, ctx)
    crit = (1 + lam + lam**2) < (1 - lam**3) * (2 + 3 * lam)
    assert easy.has_cycle_sweep == det.has_cycle == crit, (lam, easy, det, crit)
    return crit


# ---------------------------------------------------------------------------
# R4: fuzz on random small folded components (report-only)
# ---------------------------------------------------------------------------


def fuzz_small_components(n_cases: int = 150, seed: int = 0) -> dict:
    """Random strongly connected folded components with non-dyadic costs and
    lambda = 7/10, so float64 genuinely rounds.  Ground truth: exact sweep
    verdict (== exact pi-tight verdict, asserted).  Reports agreement of the
    float pi-tight (primary) and float sweep (audit) verdicts."""
    rng = random.Random(seed)
    ctxq, ctxf = MM.exact(), MM.floating(1e-9)
    lamq = Fraction(7, 10)
    lamf = float(lamq)
    stats = {"cases": 0, "beneficial": 0, "pi_float_agree": 0, "sweep_float_agree": 0}
    for _ in range(n_cases):
        n = rng.randint(2, 7)
        S = set(range(1, n + 1))
        internal_q: Dict[int, list] = {i: [] for i in S}
        for i in range(1, n + 1):
            internal_q[i].append((1, i % n + 1, Fraction(rng.randint(1, 99), 100)))
        for _extra in range(rng.randint(0, 2 * n)):
            a, b = rng.randint(1, n), rng.randint(1, n)
            if a != b:
                internal_q[a].append((2, b, Fraction(rng.randint(1, 99), 100)))
        e_q: Dict[int, object] = {
            i: (Fraction(rng.randint(50, 400), 100) if rng.random() < 0.7 else MM.INF)
            for i in S
        }
        if all(MM.exact().is_inf(v) for v in e_q.values()):
            e_q[1] = Fraction(rng.randint(50, 400), 100)
        truth = MM.solve_easy(S, internal_q, e_q, lamq, ctxq).has_cycle_sweep
        det_q = MM.detect_via_pi(S, internal_q, e_q, lamq, ctxq)
        assert det_q.has_cycle == truth, "exact detectors disagree (bug)"
        internal_f = {
            i: [(u, y, float(w)) for u, y, w in arcs] for i, arcs in internal_q.items()
        }
        e_f = {i: (MM.INF if MM.exact().is_inf(v) else float(v)) for i, v in e_q.items()}
        pi_f = MM.detect_via_pi(S, internal_f, e_f, lamf, ctxf).has_cycle
        sweep_f = MM.solve_easy(S, internal_f, e_f, lamf, ctxf).has_cycle_sweep
        stats["cases"] += 1
        stats["beneficial"] += int(truth)
        stats["pi_float_agree"] += int(pi_f == truth)
        stats["sweep_float_agree"] += int(sweep_f == truth)
    return stats


# ---------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------


def run() -> dict:
    os.makedirs(RESULTS, exist_ok=True)
    t0 = time.time()
    ctxq, ctxf = MM.exact(), MM.floating(1e-9)
    Gq, Gf = build_graph(ctxq), build_graph(ctxf)

    # M01 -- arc-level agreement with exp14's instance
    arcs_m = sorted((x, u, y, Fraction(w)) for x, u, y, w in Gq.arcs)
    arcs_e = sorted((x, u, y, Fraction(w)) for x, u, y, w in E1.build().arcs)
    check("M01", arcs_m == arcs_e, "stg_from_L == exp14 graph, arc by arc (32 arcs)")

    # ---------------- R1: B01-B16 == exp14's A01-A16 via methods+exact ------
    # M02 (B01) fixed point
    check("M02", E1.f(XE, UE) == XE and E1.g(XE, UE) == 0, "B01: f(1,u_e)=1, g=0")

    # M03 (B02) stabilizable set
    Rq = MM.reverse_reachable(Gq, XE)
    check("M03", Rq == set(range(1, 15)), "B02: V_s = 1..14; 15,16 unstabilizable")

    # M04 (B03) pruning
    check("M04", E1.f(4, 2) == 15 and E1.f(13, 2) == 16, "B03: arcs 4->15, 13->16 pruned")

    # M05 (B04) condensation
    cond = MM.condense(Gq, Rq, XE)
    nontrivial = sorted((c for c in cond.components if len(c) > 1), key=min)
    check(
        "M05",
        len(cond.components) == 10 and nontrivial == [H, B]
        and cond.comp_of[8] < cond.comp_of[2] < cond.comp_of[6],
        "B04: 10 components; nontrivial {2,3,5},{6,7,10}; order {8} < H < B",
    )

    # master pipeline (exact)
    res_q = MM.run_master(Gq, XE, LAM, ctxq, ue=UE)
    rec_q = {frozenset(r.S): r for r in res_q.records}
    recH, recB = rec_q[frozenset(H)], rec_q[frozenset(B)]

    # M06 (B05) exit weights
    check(
        "M06",
        recH.e[2] == Fraction(7, 2) and ctxq.is_inf(recH.e[3]) and ctxq.is_inf(recH.e[5])
        and all(recB.e[x] == E1.EXPECTED_E_B[x] for x in B),
        "B05: e(2)=3.5, e(3)=e(5)=inf; e(6)=4.5, e(7)=4.1875, e(10)=3.5",
    )

    # M07 (B06) exactly one beneficial component
    check(
        "M07",
        len(res_q.records) == 9 and recH.has_cycle
        and sum(1 for r in res_q.records if r.has_cycle) == 1,
        "B06: hasCycle(H) only, 9 folded components",
    )

    # M08 (B07) sweep counts (synchronous Jacobi, solve_easy)
    check(
        "M08",
        (recB.sweeps, recB.strict_sweeps) == (3, 2)
        and (recH.sweeps, recH.strict_sweeps) == (3, 3)
        and all(r.detector == "sweep" and not r.disagreement for r in res_q.records),
        "B07: B (sweeps,strict)=(3,2)=D+1; H=(3,3) never settles; detector=sweep",
    )

    # M09 (B08) beneficial-cycle criterion, thresholds, non-beneficial cycles
    Jq = res_q.J
    g_loop = 1 + LAM + LAM**2
    thr = {y: (1 - LAM**3) * Jq[y] for y in H}
    crit_ok = all(thr[y] == E1.EXPECTED_THR_H[y] and g_loop < thr[y] for y in H)
    b_loop = 2 * (1 + LAM + LAM**2)
    non_beneficial = (
        not (b_loop < (1 - LAM**3) * Jq[6])
        and not (Fraction(2) < (1 - LAM) * Jq[9])   # self-loop at 9, cost 2
    )
    check(
        "M09",
        crit_ok and non_beneficial and g_loop == Fraction(7, 4),
        "B08: g_loop=7/4 < thresholds 49/16,133/64,77/32; B cycle & 9-selfloop not beneficial",
    )

    # M10 (B09) J table == hand table == DFS oracle; inf at 15,16
    oracle = MM.dfs_simple_paths(Gq, XE, LAM, ctxq)
    ok10 = all(
        (E1.EXPECTED_J[x] is None and x not in Jq and ctxq.is_inf(oracle[x]))
        or (Jq.get(x) == E1.EXPECTED_J[x] == oracle[x])
        for x in range(1, 17)
    )
    check("M10", ok10, "B09: J == hand table == DFS oracle on all 16 states")

    # M11 (B10) SolveCore on H
    Gr = Gq.restrict(Rq)
    internal_H, e_H = MM.fold_component(Gr, H, Jq, LAM, ctxq)
    core = MM.solve_core(H, internal_H, e_H, LAM, ctxq)
    check(
        "M11",
        all(core.J[x] == E1.EXPECTED_J[x] for x in H),
        "B10: SolveCore == (3.5, 2.375, 2.75) on H",
    )

    # M12 (B11) V table and gap set
    Vq, mu_pi_q = MM.pi(Gq, Rq, XE, LAM, ctxq)
    gap = {x for x in Rq if Vq[x] < Jq[x]}
    check(
        "M12",
        all(Vq[x] == E1.EXPECTED_V[x] for x in Rq) and gap == E1.EXPECTED_GAP,
        "B11: V exact; gap = {2,3,4,5,13,14}",
    )

    # M13 (B12) improper sets
    adjR = MM.restricted_adjacency(Gq, Rq)
    vi_seq = MM.vi(Gq, Rq, XE, LAM, ctxq, 64)
    fl_seq = MM.floyd(Gq, Rq, XE, LAM, ctxq, 64)
    mu_vi_q = MM.greedy_mu(vi_seq[-1], adjR, XE, LAM, ctxq, E1.f, ue=UE)
    mu_fl_q = MM.greedy_mu(fl_seq[-1], adjR, XE, LAM, ctxq, E1.f, ue=UE)
    mu_pi_q[XE] = (UE, XE)
    bad = {
        name: MM.improper_states(E1.f, mu, Rq, XE)
        for name, mu in (("vi", mu_vi_q), ("pi", mu_pi_q), ("floyd", mu_fl_q),
                         ("master", res_q.mu))
    }
    check(
        "M13",
        bad["vi"] == bad["pi"] == bad["floyd"] == E1.EXPECTED_IMPROPER
        and bad["master"] == set(),
        f"B12: VI/PI/Floyd improper = {sorted(bad['vi'])}; master = 0",
    )

    # M14 (B13) monotone, bounded below by V*, greedy action sets stabilize
    acts_star = MM.greedy_actions(Vq, adjR, XE, LAM, ctxq)

    def monotone_and_bounded(seq) -> bool:
        for A, Bv in zip(seq[1:], seq[:-1]):
            for x in A:
                a, b = A[x], Bv[x]
                if ctxq.is_inf(a) and not ctxq.is_inf(b):
                    return False
                if not ctxq.is_inf(a) and not ctxq.is_inf(b) and a > b:
                    return False
                if not ctxq.is_inf(a) and a < Vq[x]:
                    return False
        return True

    def k0_of(seq):
        k0 = None
        for k, Vk in enumerate(seq):
            if MM.greedy_actions(Vk, adjR, XE, LAM, ctxq) == acts_star:
                if k0 is None:
                    k0 = k
            else:
                k0 = None
        return k0

    k0_vi, k0_fl = k0_of(vi_seq), k0_of(fl_seq)
    check(
        "M14",
        monotone_and_bounded(vi_seq) and monotone_and_bounded(fl_seq)
        and k0_vi is not None and k0_fl is not None,
        f"B13: monotone >= V*; greedy sets stable from k0(VI)={k0_vi}, k0(Floyd)={k0_fl}",
    )

    # M15 (B14) master feedback attains J pointwise
    cl = {x: MM.closed_loop_cost(E1.f, E1.g, res_q.mu, x, XE, LAM, ctxq) for x in Rq}
    check("M15", all(cl[x] == Jq[x] for x in Rq),
          "B14: closed-loop cost of mu* == J at all 14 states")

    # M16 (B15) lambda*: exact bisection; floating() entry refused
    b34, b79 = beneficial_at_m(Fraction(3, 4)), beneficial_at_m(Fraction(7, 9))
    lo, hi = MM.bisect_rational_threshold(
        beneficial_at_m, Fraction(1, 2), Fraction(7, 8), ctxq, iters=60
    )
    poly = lambda l: 3 * l**2 - l - 1  # noqa: E731
    raised = False
    try:
        MM.bisect_rational_threshold(
            beneficial_at_m, Fraction(1, 2), Fraction(7, 8), ctxf
        )
    except ValueError:
        raised = True
    check(
        "M16",
        b34 is True and b79 is False and float(hi - lo) < 1e-15
        and poly(lo) < 0 < poly(hi)
        and abs(float(lo) - E1.LAM_STAR) < 1e-9 and raised,
        f"B15: bracket [{float(lo):.12f},{float(hi):.12f}] ~ (1+sqrt(13))/6;"
        " floating() entry raises ValueError",
    )

    # M17 (B16) every finite value dyadic
    pool: List[Fraction] = list(Jq.values()) + [Vq[x] for x in Rq]
    pool += [v for r in res_q.records for v in r.e.values() if not ctxq.is_inf(v)]
    pool += list(thr.values()) + [g_loop] + [cl[x] for x in Rq]
    check(
        "M17",
        all(Fraction(v).denominator & (Fraction(v).denominator - 1) == 0 for v in pool),
        f"B16: {len(pool)} finite values, all denominators powers of 2",
    )

    # ---------------- R2: three-way against the core_solver route -----------
    E1_G = E1.build()
    _R1, _scc1, J_cs, _rec_cs = E1.manual_master(E1_G, LAM)
    V_cs, _mu_cs = E1.baseline_pi(E1_G, Rq, XE, LAM)
    check(
        "M18",
        all(J_cs[x] == Jq[x] for x in Rq) and all(V_cs[x] == Vq[x] for x in Rq),
        "R2: core_solver route == methods+exact on J and V, pointwise",
    )

    # ---------------- R3: float context cross-check -------------------------
    res_f = MM.run_master(Gf, XE, LAM, ctxf, ue=UE)
    Rf = MM.reverse_reachable(Gf, XE)
    check(
        "M19",
        Rf == Rq
        and all(abs(float(res_f.J[x]) - float(Jq[x])) < 1e-9 for x in Rq)
        and all(x not in res_f.J for x in (15, 16)),
        "R3: |J_float - J_exact| < 1e-9 on the 14 finite states; inf at 15,16",
    )

    adjRf = MM.restricted_adjacency(Gf, Rf)
    Vf, mu_pi_f = MM.pi(Gf, Rf, XE, LAM, ctxf)
    mu_pi_f[XE] = (UE, XE)
    mu_vi_f = MM.greedy_mu(MM.vi(Gf, Rf, XE, LAM, ctxf, 64)[-1], adjRf, XE, LAM,
                           ctxf, E1.f, ue=UE)
    mu_fl_f = MM.greedy_mu(MM.floyd(Gf, Rf, XE, LAM, ctxf, 64)[-1], adjRf, XE, LAM,
                           ctxf, E1.f, ue=UE)
    bad_f = {
        name: MM.improper_states(E1.f, mu, Rf, XE)
        for name, mu in (("vi", mu_vi_f), ("pi", mu_pi_f), ("floyd", mu_fl_f),
                         ("master", res_f.mu))
    }
    check(
        "M20",
        all(bad_f[k] == bad[k] for k in bad),
        "R3: float improper sets identical to exact (VI/PI/Floyd/master)",
    )

    rec_f = {frozenset(r.S): r for r in res_f.records}
    margins_ok = all(
        r.tight_margin is None or r.tight_margin > 1e-8 for r in res_f.records
    )
    check(
        "M21",
        set(rec_f) == set(rec_q)
        and all(rec_f[k].has_cycle == rec_q[k].has_cycle for k in rec_q)
        and all(r.detector == "pi-tight" for r in res_f.records)
        and res_f.detect_disagreements == 0 and margins_ok,
        "R3: per-component verdicts match; 0 detector disagreements; margins > 1e-8",
    )

    # M22 -- exact: sweep detector == pi-tight detector on every folded component
    ok22 = True
    for cid in cond.reverse_topo:
        S = set(cond.components[cid])
        if XE in S:
            if len(S) == 1:
                continue
            S, internal, e = MM.fold_terminal(Gr, S, XE, LAM, ctxq)
        else:
            internal, e = MM.fold_component(Gr, S, Jq, LAM, ctxq)
        easy = MM.solve_easy(S, internal, e, LAM, ctxq)
        det = MM.detect_via_pi(S, internal, e, LAM, ctxq)
        ok22 &= easy.has_cycle_sweep == det.has_cycle
    check("M22", ok22, "exact: Lemma lc-cap verdict == Prop gap-W(b) verdict, all 9 components")

    # M23 -- MILP oracle (function body implemented by the Example 2 task,
    # superseding the Task-0 placeholder "hook raises NotImplementedError"):
    # it must reproduce the exact SolveCore values on the beneficial-cycle
    # component H, per the contract in its docstring.
    res23 = MM.milp_oracle(H, internal_H, e_H, LAM, ctxq)
    ok23 = set(res23) == set(H) and all(
        abs(float(res23[x]) - float(Jq[x])) <= 1e-6 for x in H
    )
    check("M23", ok23, "milp_oracle == exact SolveCore values on H (CBC)")

    # M24 -- smoke: existing scripts unaffected
    r10 = subprocess.run([sys.executable, "exp10_toy_bcn.py"], cwd=SIM_DIR,
                         capture_output=True, text=True)
    r14 = subprocess.run([sys.executable, "exp14_ex1_n4.py"], cwd=SIM_DIR,
                         capture_output=True, text=True)
    check(
        "M24",
        r10.returncode == 0 and "ASSERTIONS PASSED" in r10.stdout
        and r14.returncode == 0 and "ALL 16 ASSERTIONS PASSED" in r14.stdout,
        "R5: exp10 and exp14 subprocesses pass",
    )

    # M25 -- hygiene: no core_solver import; protected files untouched by this
    # task.  core_solver.py / exp10 are tracked and must be clean in git;
    # exp14 (untracked) is pinned to the sha256 of its accepted version --
    # STRONGER than a git check.  Two tex files were modified by a concurrent
    # paper-editing session on 2026-08-31 14:54 (notation J_tail ->
    # J_lambda^tail; not by this task -- plan section 10): their exact content
    # is pinned below, so any FURTHER tex modification still fails here.
    with open(os.path.join(SIM_DIR, "methods.py")) as fh:
        src = fh.read()
    no_import = not re.search(r"^\s*(import|from)\s+core_solver\b", src, re.M)

    def sha256_of(rel: str) -> str:
        import hashlib
        with open(os.path.join(REPO_ROOT, rel), "rb") as fh2:
            return hashlib.sha256(fh2.read()).hexdigest()

    EXP14_SHA = "e1ab16be8d30a21d272277ee4b7294e458ce1dcab8e3c09cc395aae0002af2a2"
    EXTERNAL_TEX_EDITS = {
        "sections/main/01_vi_inapplicable.tex":
            "cdb87fa942a6b46b93ea2a2333bdf670dc4b8dad07aa66f2f9a8a30b1f9732a2",
        "sections/main/05_master_algorithm.tex":
            "82cc2fd4e568b942cbe8eb2c91c49a68bb0ec6bd22306b1f0a69dac4cfa5569e",
    }
    porcelain = subprocess.run(
        ["git", "status", "--porcelain"], cwd=REPO_ROOT, capture_output=True, text=True
    ).stdout.splitlines()
    modified = [
        ln[3:] for ln in porcelain if ln[:2].strip() in ("M", "MM", "D", "AM")
    ]
    code_clean = not any(
        p.endswith(("core_solver.py", "exp10_toy_bcn.py")) for p in modified
    )
    tex_modified = [p for p in modified if p.endswith(".tex")]
    tex_ok = all(
        p in EXTERNAL_TEX_EDITS and sha256_of(p) == EXTERNAL_TEX_EDITS[p]
        for p in tex_modified
    )
    exp14_ok = sha256_of("simulations/exp14_ex1_n4.py") == EXP14_SHA
    check(
        "M25",
        no_import and code_clean and tex_ok and exp14_ok,
        "no core_solver import; core_solver/exp10 clean in git; exp14 pinned by"
        " sha256; tex changes limited to the pinned external edits",
    )

    # ---------------- R4: fuzz (report-only) --------------------------------
    fuzz = fuzz_small_components(150, seed=0)

    elapsed = time.time() - t0

    lines = [
        "E15: regression & cross-check for methods.py (Task 0)",
        f"  instance: Example 1 via import exp14_ex1_n4 (N={E1.N}, M={E1.M},"
        f" lambda={LAM})",
        f"  contexts: exact() vs floating(1e-9)",
        "",
        "  acceptance items:",
    ]
    lines += [f"    {p}" for p in passed]
    lines += [
        "",
        f"  R4 fuzz (|S|<=7, lambda=7/10, non-dyadic costs, {fuzz['cases']} cases,"
        f" {fuzz['beneficial']} with a beneficial cycle):",
        f"    float pi-tight  agrees with exact: {fuzz['pi_float_agree']}/{fuzz['cases']}",
        f"    float sweep     agrees with exact: {fuzz['sweep_float_agree']}/{fuzz['cases']}",
        "    (sweep flips are expected only for large |S|, where the residual"
        " lambda^|S| falls below eps; see methods.py header)",
        "",
        f"  runtime: {elapsed:.2f} s",
        "  ALL 25 ACCEPTANCE ITEMS PASSED",
    ]
    text = "\n".join(lines)
    with open(os.path.join(RESULTS, "exp15_methods_regression.txt"), "w") as fh:
        fh.write(text + "\n")
    print(text)
    return {"passed": passed, "fuzz": fuzz, "elapsed": elapsed}


if __name__ == "__main__":
    run()
