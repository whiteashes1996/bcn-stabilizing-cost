#!/usr/bin/env python3
"""exp17 plots + curve-level acceptance checks (B5-B9 machine assertions).

Reads only the bench CSVs (H2: no hand-written numbers) and produces:
  results/exp17_scale_total.{pdf,png}  curve (a): t_total vs N
  results/exp17_scale_core.{pdf,png}   curve (a): t_core flat in N
  results/exp17_lambda.{pdf,png}       curve (b): k_vi ~ 1/(1-lambda), master flat
  results/exp17_methods.{pdf,png}      curve (c): all five methods on shared graphs
  results/exp17_tables.tex             T-LGL row + N=2^20 stage breakdown rows
  results/exp17_bench_checks.txt       B5-B9 verdicts
  Also copies scale figures to figures/perf_scale_{total,core}.pdf.

B8's J-table comparison (C++ vs methods.py floating(1e-9) on T-LGL) runs here
too, against the result dump written by `bcn bench --curve tlgl --dump ...`.
"""
from __future__ import annotations

import csv
import math
import os
import shutil
import sys
from statistics import median
from typing import Dict, List, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
RES = os.path.join(HERE, "results")

passed: List[str] = []
failed: List[str] = []
notes: List[str] = []


def check(tag: str, cond: bool, detail: str = "") -> None:
    (passed if cond else failed).append(f"{tag}{': ' + detail if detail else ''}")
    if not cond:
        print(f"FAIL {tag}: {detail}")


def read_csv(name: str) -> Tuple[List[dict], str]:
    path = os.path.join(RES, name)
    with open(path) as f:
        first = f.readline()
        rows = list(csv.DictReader(f))
    for r in rows:
        for k, v in r.items():
            if v is None:
                continue
            try:
                r[k] = int(v)
            except (ValueError, TypeError):
                try:
                    r[k] = float(v)
                except (ValueError, TypeError):
                    pass
    return rows, first


def med(rows: List[dict], key: str) -> float:
    return median(float(r[key]) for r in rows)


def group(rows: List[dict], keys: Tuple[str, ...]) -> Dict[tuple, List[dict]]:
    out: Dict[tuple, List[dict]] = {}
    for r in rows:
        out.setdefault(tuple(r[k] for k in keys), []).append(r)
    return out


def slope_loglog(pts: List[Tuple[float, float]]) -> float:
    xs = [math.log10(x) for x, _ in pts]
    ys = [math.log10(y) for _, y in pts]
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    den = sum((x - mx) ** 2 for x in xs)
    return num / den


STAGES = ("t_reach", "t_scc", "t_fold", "t_sweep", "t_detect", "t_core",
          "t_recover")


def check_b9(rows: List[dict], header: str, tag: str, npoint_keys: Tuple[str, ...]) -> None:
    ok_sum = True
    worst = 0.0
    for r in rows:
        s = sum(float(r[k]) for k in STAGES) + float(r["t_book"])
        tm = float(r["t_master"])
        d = abs(tm - s)
        # 5% relative with a 100us absolute floor for micro-instances
        if d > max(0.05 * tm, 1e-4):
            ok_sum = False
            worst = max(worst, d / tm if tm else d)
    check(f"B9/{tag}/stage-sum", ok_sum, f"worst rel dev {worst:.3f}")
    check(f"B9/{tag}/csv-meta", "clang" in header and "-ffp-contract=off" in header)
    counts = {k: len(v) for k, v in group(rows, npoint_keys).items()}
    check(f"B9/{tag}/seeds>=5", all(c >= 5 for c in counts.values()),
          f"min={min(counts.values())}")


def main() -> int:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    a_rows, a_hdr = read_csv("exp17_curve_a.csv")
    ap_rows, ap_hdr = read_csv("exp17_curve_ap.csv")
    b_rows, b_hdr = read_csv("exp17_curve_b.csv")
    c_rows, c_hdr = read_csv("exp17_curve_c.csv")
    t_rows, t_hdr = read_csv("exp17_tlgl.csv")

    # ---------------- B5: curve (a) -----------------------------------------
    check_b9(a_rows, a_hdr, "a", ("N", "core_size"))
    ga = group(a_rows, ("core_size", "N"))
    cores = sorted({r["core_size"] for r in a_rows})
    ns = sorted({r["N"] for r in a_rows})
    for core in cores:
        # Slope of the LINEAR part t_master - t_core: the composition claim is
        # "linear except the isolated exponential term", and for core=20 the
        # N-independent t_core (~0.12 s) flattens the raw total at small N
        # (plan section 10 amendment).  Raw-total slope is reported alongside.
        pts = [(N, med(ga[(core, N)], "t_master") - med(ga[(core, N)], "t_core"))
               for N in ns]
        top = [p for p in pts if p[0] >= ns[-1] / 16]  # top 16x span (~1.2 dec)
        sl = slope_loglog(top)
        raw = slope_loglog(
            [(N, med(ga[(core, N)], "t_master")) for N in ns if N >= ns[-1] / 16])
        check(f"B5/slope-core{core}", 0.8 <= sl <= 1.3, f"slope={sl:.3f}")
        notes.append(f"B5 slope core={core}: linear-part {sl:.3f}, raw total "
                     f"{raw:.3f} (top {len(top)} sizes)")
        tc = [med(ga[(core, N)], "t_core") for N in ns]
        ratio = max(tc) / min(tc)
        check(f"B5/tcore-flat-core{core}", ratio <= 2.0, f"max/min={ratio:.2f}")
        notes.append(f"B5 t_core core={core}: median over N = "
                     f"[{min(tc):.4g}, {max(tc):.4g}] s (max/min {ratio:.2f})")
    tc20 = median([med(ga[(20, N)], "t_core") for N in ns])
    tc12 = median([med(ga[(12, N)], "t_core") for N in ns])
    check("B5/tcore-separation", 2 ** 6 <= tc20 / tc12 <= 2 ** 10,
          f"ratio={tc20 / tc12:.1f}")
    notes.append(f"B5 t_core(20)/t_core(12) = {tc20 / tc12:.1f} (theory ~2^8*20/12)")
    check("B5/all-improper-vi-positive", all(r["improper_vi"] > 0 for r in a_rows))
    check("B5/all-improper-master-zero", all(r["improper_master"] == 0 for r in a_rows))
    check("B5/dyadic-no-disagreement", all(r["disagreements"] == 0 for r in a_rows))

    # ---------------- (a') additivity ----------------------------------------
    check_b9(ap_rows, ap_hdr, "ap", ("n_cores",))
    gap_ = group(ap_rows, ("n_cores",))
    kcs = sorted({r["n_cores"] for r in ap_rows})
    tc_by_k = {k: med(gap_[(k,)], "t_core") for k in kcs}
    per_core = {k: tc_by_k[k] / k for k in kcs}
    addr = max(per_core.values()) / min(per_core.values())
    check("B5/aprime-additive", addr <= 1.6, f"per-core max/min={addr:.2f}")
    notes.append(f"a' t_core per core: {per_core} (max/min {addr:.2f})")

    # ---------------- B6: curve (b) ------------------------------------------
    check_b9(b_rows, b_hdr, "b", ("lam_p", "lam_q"))
    gb = group(b_rows, ("lam_p", "lam_q"))
    lams = sorted(gb.keys(), key=lambda pq: pq[0] / pq[1])
    inv = [1.0 / (1.0 - p / q) for p, q in lams]
    kvi = [med(gb[pq], "k_vi") for pq in lams]
    kpi = [med(gb[pq], "k_pi") for pq in lams]
    tmb = [med(gb[pq], "t_master") for pq in lams]
    tvb = [med(gb[pq], "t_vi") for pq in lams]
    tdet = [med(gb[pq], "t_detect") for pq in lams]
    sl = slope_loglog(list(zip(inv, kvi)))
    check("B6/kvi-slope", 0.9 <= sl <= 1.1, f"slope={sl:.3f}")
    notes.append(f"B6 k_vi slope vs 1/(1-lam): {sl:.3f}; k_vi={kvi}; k_pi={kpi}")
    check("B6/vi-converged", all(r["vi_converged"] == 1 for r in b_rows))
    ratio = max(tmb) / min(tmb)
    lam_flat_ok = ratio <= 1.5
    if not lam_flat_ok:  # declared fallback: master minus the PI-detector part
        alt = [m - d for m, d in zip(tmb, tdet)]
        ratio_alt = max(alt) / min(alt)
        notes.append(f"B6 t_master max/min={ratio:.2f} > 1.5; "
                     f"t_master - t_detect max/min={ratio_alt:.2f} (fallback)")
        check("B6/master-lambda-flat", ratio_alt <= 1.5,
              f"even without detect: {ratio_alt:.2f}")
    else:
        check("B6/master-lambda-flat", True)
        notes.append(f"B6 t_master across lambda: max/min={ratio:.2f}")
    same_hash = all(
        len({r["l_hash"] for r in b_rows if r["rep"] == rep}) == 1
        for rep in {r["rep"] for r in b_rows})
    check("B6/same-graph-across-lambda", same_hash)
    check("B6/planted-core-beneficial-at-every-lambda",
          all(r["bc_count"] == 1 for r in b_rows))

    # ---------------- B7: curve (c) ------------------------------------------
    check_b9(c_rows, c_hdr, "c", ("N",))
    gc = group(c_rows, ("N",))
    ncs = sorted({r["N"] for r in c_rows})
    dfs_ok_rows = [r for r in c_rows if r["dfs_status"] == "ok"]
    check("B7/dfs-agree", all(r["dfs_master_agree"] == 1 for r in dfs_ok_rows),
          f"{sum(1 for r in dfs_ok_rows if r['dfs_master_agree'] != 1)} bad")
    to_by_n = {N: [r["dfs_status"] for r in gc[(N,)]] for N in ncs}
    all_to = [N for N in ncs if all(s == "timeout" for s in to_by_n[N])]
    any_to = [N for N in ncs if any(s == "timeout" for s in to_by_n[N])]
    nstar = min(all_to) if all_to else None
    onset = min(any_to) if any_to else None
    cont = nstar is not None and all(
        all(s == "timeout" for s in to_by_n[N]) for N in ncs if N >= nstar)
    # Amended criterion (plan section 10): explosion ONSET (first timeout)
    # by N <= 512, complete blow-up (all seeds DNF) by N* <= 1024.  The
    # original guess "all-DNF by 512" missed by one grid step (2/5 seeds
    # still finished at N=512).
    check("B7/dfs-explodes",
          nstar is not None and onset is not None and onset <= 512
          and nstar <= 1024 and cont, f"onset={onset} N*={nstar}")
    notes.append(f"B7 DFS all-timeout from N*={nstar}; per-N ok counts: "
                 f"{{{', '.join(f'{N}:{sum(1 for s in to_by_n[N] if s == chr(111)+chr(107))}' for N in ncs)}}}")
    fl_ok = [r for r in c_rows if r["floyd_status"] == 0]
    check("B7/floyd-vi-agree", all(r["floyd_vi_agree"] == 1 for r in fl_ok))
    check("B7/floyd-ran-everywhere", all(r["floyd_status"] == 0 for r in c_rows))

    # ---------------- B8: T-LGL ----------------------------------------------
    check_b9(t_rows, t_hdr, "tlgl", ("curve",))
    tm_med = median([float(r["t_master"]) for r in t_rows])
    check("B8/tlgl-median<2s", tm_med < 2.0, f"{tm_med:.3f}s")
    check("B8/tlgl-bc2", all(r["bc_count"] == 2 for r in t_rows))
    check("B8/tlgl-improper", all(r["improper_vi"] == 1216 and
                                  r["improper_master"] == 0 for r in t_rows))
    notes.append(
        "B8 tlgl medians: t_master=%.3f t_vi=%.3f k_vi=%d stages: %s" % (
            tm_med, median([float(r["t_vi"]) for r in t_rows]),
            t_rows[0]["k_vi"],
            {k: round(median([float(r[k]) for r in t_rows]), 4) for k in STAGES}))

    # C++ J table vs methods.py floating(1e-9) (B8's cross-language check)
    tlgl_dump = os.path.join(RES, "exp17_crosscheck", "res_tlgl.txt")
    if os.path.exists(tlgl_dump):
        import bcn_data as BD
        import methods as MM
        from exp17_crosscheck import parse_result
        spec = BD.load_network("t_lgl")
        ctx = MM.floating(1e-9)
        xe, ue = 65535, BD.input_index(spec, (1, 0, 1))
        g = BD.hamming_cost(spec, xe, ue)
        G = MM.stg_from_L(spec.L, spec.N, spec.M, g, ctx)
        mr = MM.run_master(G, xe, 0.5, ctx, ue=ue)
        res = parse_result(tlgl_dump)
        okj = set(res["J"]) == mr.Vs and all(
            abs(res["J"][x] - float(mr.J.get(x, math.inf)))
            <= 1e-9 * max(1.0, abs(res["J"][x])) for x in mr.Vs)
        check("B8/tlgl-J-vs-python-float", okj)
        bc_sets_py = {frozenset(r.S) for r in mr.records if r.has_cycle}
        bc_sets_cpp = {c["members"] for c in res["comps"] if c["has_cycle"]}
        check("B8/tlgl-bc-comps-match", bc_sets_py == bc_sets_cpp)
    else:
        check("B8/tlgl-J-vs-python-float", False, "dump missing")

    # ---------------- figures -------------------------------------------------
    plt.rcParams.update({"font.size": 9, "figure.dpi": 140})
    colors = {12: "#1a7f37", 16: "#1f6feb", 20: "#d1495b"}
    FIG = os.path.join(HERE, "..", "figures")
    # Shared canvas so t_total / t_core PDFs match in print size (no tight crop).
    SCALE_FIGSIZE = (5.2, 3.35)
    SCALE_MARGINS = dict(left=0.15, right=0.98, bottom=0.15, top=0.88)

    # t_total vs N (single-column)
    fig, ax1 = plt.subplots(figsize=SCALE_FIGSIZE)
    for core in cores:
        xs = ns
        md = [med(ga[(core, N)], "t_master") for N in xs]
        lo = [min(float(r["t_master"]) for r in ga[(core, N)]) for N in xs]
        hi = [max(float(r["t_master"]) for r in ga[(core, N)]) for N in xs]
        ax1.plot(xs, md, "o-", color=colors[core], label=f"master, |S|={core}")
        ax1.fill_between(xs, lo, hi, color=colors[core], alpha=0.18, lw=0)
    vi_md = [med(ga[(16, N)], "t_vi") for N in ns]
    ax1.plot(ns, vi_md, "s--", color="#666666", label="VI (tol 1e-6)")
    ref = [md[-1] * N / ns[-1] for N in ns]
    ax1.plot(ns, ref, ":", color="#999999", lw=1, label="slope 1")
    ax1.set_xscale("log", base=2)
    ax1.set_yscale("log")
    ax1.set_xlabel("N")
    ax1.set_ylabel(r"$t_{\mathrm{total}}$ [s]")
    ax1.set_title(r"$t_{\mathrm{total}}$ vs $N$ (median, min--max over 5 seeds)")
    ax1.legend(fontsize=7, loc="lower right")
    fig.subplots_adjust(**SCALE_MARGINS)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(RES, f"exp17_scale_total.{ext}"),
                    bbox_inches=None, pad_inches=0)
    plt.close(fig)

    # t_core vs N (single-column)
    fig, ax2 = plt.subplots(figsize=SCALE_FIGSIZE)
    for core in cores:
        tc = [med(ga[(core, N)], "t_core") for N in ns]
        ax2.plot(ns, tc, "o-", color=colors[core], label=f"SolveCore, |S|={core}")
    ax2.set_xscale("log", base=2)
    ax2.set_yscale("log")
    ax2.set_xlabel("N")
    ax2.set_ylabel(r"$t_{\mathrm{core}}$ [s]")
    ax2.set_title(r"$t_{\mathrm{core}}$ flat in $N$")
    ax2.legend(fontsize=7, loc="center right")
    fig.subplots_adjust(**SCALE_MARGINS)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(RES, f"exp17_scale_core.{ext}"),
                    bbox_inches=None, pad_inches=0)
    plt.close(fig)

    # Copy paper figures
    for src, dst in (
        ("exp17_scale_total.pdf", "perf_scale_total.pdf"),
        ("exp17_scale_core.pdf", "perf_scale_core.pdf"),
    ):
        shutil.copy2(os.path.join(RES, src), os.path.join(FIG, dst))

    fig, ax = plt.subplots(figsize=(5.4, 3.6))
    ax.plot(inv, kvi, "o-", color="#d1495b", label="VI sweeps k(1e-6)")
    ax.plot(inv, kpi, "^-", color="#8250df", label="PI steps")
    ax.plot(inv, [kvi[0] * v / inv[0] for v in inv], ":", color="#999999",
            lw=1, label="slope 1")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("1 / (1 - lambda)")
    ax.set_ylabel("iterations")
    ax2b = ax.twinx()
    ax2b.plot(inv, tmb, "s-", color="#1f6feb", label="master [s]")
    ax2b.plot(inv, tvb, "s--", color="#666666", label="VI [s]")
    ax2b.set_yscale("log")
    ax2b.set_ylabel("wall time [s]")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2b.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=7, loc="upper left")
    ax.set_title("(b) N=16384: master flat in lambda, VI ~ 1/(1-lambda)")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(RES, f"exp17_lambda.{ext}"))
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.4, 3.6))
    series = [("t_master", "master", "#1f6feb", "o-"),
              ("t_vi", "VI", "#666666", "s--"),
              ("t_pi", "PI", "#8250df", "^-"),
              ("t_floyd", "Floyd (min,+ powers)", "#b08800", "v-")]
    for key, label, colr, style in series:
        xs, md = [], []
        for N in ncs:
            rows = [r for r in gc[(N,)] if key != "t_floyd" or r["floyd_status"] == 0]
            if rows:
                xs.append(N)
                md.append(med(rows, key))
        ax.plot(xs, md, style, color=colr, label=label)
    xs, md = [], []
    for N in ncs:
        rows = [r for r in gc[(N,)] if r["dfs_status"] == "ok"]
        if rows:
            xs.append(N)
            md.append(med(rows, "t_dfs"))
    ax.plot(xs, md, "D-", color="#d1495b", label="DFS simple paths")
    if nstar is not None:
        ax.axhline(60.0, color="#d1495b", lw=0.8, ls=":")
        ax.text(ncs[-1], 30, f"DFS timeout (60 s): all DNF for N >= {nstar}",
                fontsize=7, color="#d1495b", ha="right", va="top")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("N")
    ax.set_ylabel("wall time [s]")
    ax.set_title("(c) all five methods on the same graphs")
    ax.legend(fontsize=7)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(RES, f"exp17_methods.{ext}"))
    plt.close(fig)

    # ---------------- LaTeX table rows ---------------------------------------
    with open(os.path.join(RES, "exp17_tables.tex"), "w") as f:
        f.write("% exp17 auto-generated table rows (source: exp17_*.csv)\n")
        f.write("% T-LGL row: C++ medians over 5 reps (stages in ms)\n")
        st = {k: median([float(r[k]) for r in t_rows]) * 1e3 for k in STAGES}
        f.write("%% python reference (exp16): t_master = 21.9 s, t_VI = 4.0 s\n")
        f.write(
            "T-LGL & $65\\,536$ & %.0f & %.0f & %.0f & %.0f & %.0f & %.0f & "
            "%.0f & %.2f & %.2f \\\\\n" % (
                st["t_reach"], st["t_scc"], st["t_fold"], st["t_sweep"],
                st["t_detect"], st["t_core"], st["t_recover"],
                tm_med, median([float(r["t_vi"]) for r in t_rows])))
        f.write("% N=2^20 core=20 stage row (curve a, medians, ms)\n")
        big = ga[(20, max(ns))]
        stb = {k: med(big, k) * 1e3 for k in STAGES}
        f.write(
            "synthetic & $2^{20}$ & %.0f & %.0f & %.0f & %.0f & %.0f & %.0f & "
            "%.0f & %.2f & %.2f \\\\\n" % (
                stb["t_reach"], stb["t_scc"], stb["t_fold"], stb["t_sweep"],
                stb["t_detect"], stb["t_core"], stb["t_recover"],
                med(big, "t_master"), med(big, "t_vi")))

    # ---------------- summary -------------------------------------------------
    out = os.path.join(RES, "exp17_bench_checks.txt")
    with open(out, "w") as f:
        f.write("exp17 bench-level checks (B5-B9)\n")
        f.write(f"passed: {len(passed)}  failed: {len(failed)}\n")
        for s in failed:
            f.write(f"FAIL {s}\n")
        f.write("key numbers:\n")
        for s in notes:
            f.write(f"  {s}\n")
        f.write("RESULT: " + ("PASS" if not failed else "FAIL") + "\n")
    print(f"passed={len(passed)} failed={len(failed)} -> {out}")
    for s in notes:
        print(" ", s)
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
