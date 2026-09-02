"""Generate summary figures for large-scale experiments E6-E8."""
from __future__ import annotations

import csv
import os
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RESULTS = os.path.join(os.path.dirname(__file__), "results")


def load_exp7():
    path = os.path.join(RESULTS, "exp7_scale.csv")
    by_n: dict[int, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            N = int(row["N"])
            by_n[N]["t_master"].append(float(row["t_master_s"]))
            by_n[N]["t_vi"].append(float(row["t_vi_s"]))
    Ns = sorted(by_n)
    return Ns, by_n


def load_exp8():
    path = os.path.join(RESULTS, "exp8_warmstart.csv")
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def load_exp6():
    path = os.path.join(RESULTS, "exp6_real_networks.csv")
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def plot_summary():
    os.makedirs(RESULTS, exist_ok=True)
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "legend.fontsize": 9,
        }
    )

    fig = plt.figure(figsize=(12, 9))
    gs = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.28)

    # --- Panel A: E7 scale curve ---
    ax1 = fig.add_subplot(gs[0, 0])
    Ns, by_n = load_exp7()
    t_m = [np.mean(by_n[N]["t_master"]) for N in Ns]
    t_v = [np.mean(by_n[N]["t_vi"]) for N in Ns]
    t_m_std = [np.std(by_n[N]["t_master"]) for N in Ns]
    t_v_std = [np.std(by_n[N]["t_vi"]) for N in Ns]

    ax1.errorbar(Ns, t_m, yerr=t_m_std, fmt="o-", capsize=3, label="master", color="#2563eb")
    ax1.errorbar(Ns, t_v, yerr=t_v_std, fmt="s--", capsize=3, label="VI", color="#dc2626")
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.set_xlabel("N = 2^n")
    ax1.set_ylabel("time (s)")
    ax1.set_title("(A) Synthetic BCN: time vs scale")
    ax1.grid(True, alpha=0.3, which="both")
    ax1.legend()

    # fit slope on log-log for master
    logN = np.log10(Ns)
    logT = np.log10(t_m)
    slope, intercept = np.polyfit(logN, logT, 1)
    ax1.text(
        0.05,
        0.05,
        f"master slope ~ {slope:.2f}",
        transform=ax1.transAxes,
        fontsize=9,
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    # --- Panel B: E6 real networks ---
    ax2 = fig.add_subplot(gs[0, 1])
    rows6 = load_exp6()
    labels, t_master, t_vi = [], [], []
    for r in rows6:
        if r["network"] == "ara_operon" and r["xe"] == "512" and r["lambda"] == "0.5":
            labels.append("ara\nxe=512")
        elif r["network"] == "lambda_switch" and r["xe"] == "24":
            labels.append("lambda\nxe=24")
        else:
            continue
        t_master.append(float(r["t_master_s"]) * 1000)
        t_vi.append(float(r["t_vi_s"]) * 1000)

    # show all ara xe=512 + lambda cases grouped
    ara = [r for r in rows6 if r["network"] == "ara_operon" and r["xe"] == "512"]
    lam = [r for r in rows6 if r["network"] == "lambda_switch"]
    groups = [
        ("ara operon\nN=512", np.mean([float(r["t_master_s"]) for r in ara]) * 1000,
         np.mean([float(r["t_vi_s"]) for r in ara]) * 1000),
        ("lambda switch\nN=32", np.mean([float(r["t_master_s"]) for r in lam]) * 1000,
         np.mean([float(r["t_vi_s"]) for r in lam]) * 1000),
    ]
    x = np.arange(len(groups))
    w = 0.35
    ax2.bar(x - w / 2, [g[1] for g in groups], w, label="master", color="#2563eb")
    ax2.bar(x + w / 2, [g[2] for g in groups], w, label="VI", color="#dc2626")
    ax2.set_xticks(x)
    ax2.set_xticklabels([g[0] for g in groups])
    ax2.set_ylabel("time (ms)")
    ax2.set_title("(B) Real BCN: mean runtime")
    ax2.legend()
    ax2.grid(True, axis="y", alpha=0.3)

    # --- Panel C: E8 path A speedup ---
    ax3 = fig.add_subplot(gs[1, 0])
    rows8 = load_exp8()
    lc = [r for r in rows8 if r["path"] == "LC_warmstart"]
    names = [r["case"].replace("synthetic_", "syn ") for r in lc]
    speedups = [float(r["speedup"]) for r in lc]
    colors = ["#16a34a" if s >= 1 else "#f97316" for s in speedups]
    ax3.barh(names, speedups, color=colors)
    ax3.axvline(1.0, color="gray", ls="--", lw=1)
    ax3.set_xlabel("speedup (cold / warm)")
    ax3.set_title("(C) Path A: LC Dijkstra warm-start")
    ax3.grid(True, axis="x", alpha=0.3)

    # --- Panel D: E8 path B ---
    ax4 = fig.add_subplot(gs[1, 1])
    hk = [r for r in rows8 if r["path"] == "SolveCore_bound"]
    Ss = [int(r["N"]) for r in hk]
    cold = [float(r["t_cold_s"]) for r in hk]
    warm = [float(r["t_warm_s"]) for r in hk]
    ax4.plot(Ss, cold, "o-", label="Held-Karp", color="#2563eb")
    ax4.plot(Ss, warm, "s-", label="bounded (warm)", color="#16a34a")
    ax4.set_yscale("log")
    ax4.set_xlabel("|S| (beneficial SCC size)")
    ax4.set_ylabel("time (s)")
    ax4.set_title("(D) Path B: SolveCore with Dijkstra UB")
    ax4.grid(True, alpha=0.3, which="both")
    ax4.legend()

    fig.suptitle("Large-scale experiments E6–E8 summary", fontsize=13, y=0.98)
    out = os.path.join(RESULTS, "large_experiments_summary.png")
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)

    # also regenerate exp7 standalone
    fig2, ax = plt.subplots(figsize=(7, 4.5))
    ax.errorbar(Ns, t_m, yerr=t_m_std, fmt="o-", capsize=4, label="master")
    ax.errorbar(Ns, t_v, yerr=t_v_std, fmt="s--", capsize=4, label="VI")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("N = 2^n")
    ax.set_ylabel("time (s)")
    ax.set_title("E7: Synthetic acyclic BCN (lambda=0.5, Hamming cost)")
    ax.grid(True, alpha=0.3, which="both")
    ax.legend()
    fig2.tight_layout()
    fig2.savefig(os.path.join(RESULTS, "exp7_scale.png"), dpi=150)
    plt.close(fig2)

    print(f"Saved: {out}")
    return out


if __name__ == "__main__":
    plot_summary()
