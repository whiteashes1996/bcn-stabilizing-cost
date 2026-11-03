#!/usr/bin/env python3
"""Runtime of synchronous SolveEasy versus the heap version at λ = 1/2.

Four families:
  path         no reopen, depth n-1
  star         no reopen, depth 1
  selfloop     beneficial self-loops, heap aborts immediately, then full sync
  path_gadget  one appendix pair forces a reopen after the path is settled
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from experiments import m_of, path_graph, solve_heap, solve_sync, star_graph

ROOT = Path(__file__).resolve().parent
FIGDIR = ROOT / "figures"
POINTS = ROOT / "discount_points.jsonl"
LAM = 0.5


def path_with_gadget(n):
    adj = [[] for _ in range(n)]
    exit_cost = [None] * n
    for i in range(n - 3):
        adj[i].append((i + 1, 1))
    exit_cost[n - 3] = 1
    adj[n - 2].append((n - 1, 1))
    exit_cost[n - 2] = 60
    exit_cost[n - 1] = 100
    return adj, exit_cost


def gadgets(n):
    adj = []
    exit_cost = []
    for _ in range(n // 2):
        base = len(adj)
        adj.append([(base + 1, 1)])
        adj.append([])
        exit_cost.extend([60.0, 100.0])
    return adj, exit_cost


def time_median(fn, repeats):
    walls = []
    cpus = []
    result = None
    for _ in range(repeats):
        wall0 = time.perf_counter()
        cpu0 = time.process_time()
        result = fn()
        walls.append(time.perf_counter() - wall0)
        cpus.append(time.process_time() - cpu0)
    walls.sort()
    cpus.sort()
    mid = len(walls) // 2
    return result, walls[mid], cpus[mid]


def repeats_for(family, n):
    if family in ("star",):
        return 2 if n <= 32000 else 1
    if n <= 4000:
        return 3
    if n <= 8000:
        return 2
    return 1


def close_enough(left, right):
    if len(left) != len(right):
        return False
    for a, b in zip(left, right):
        if a is None or b is None:
            if a is not b:
                return False
            continue
        scale = max(abs(a), abs(b), 1.0)
        if abs(a - b) > 1e-9 * scale:
            return False
    return True


def check(family, n, sync, heap):
    if family == "path":
        if heap["fallback"] or sync["has_cycle"] or sync["sweeps"] != n:
            raise AssertionError(f"path flags n={n} {heap['fallback']} {sync}")
    elif family == "star":
        if heap["fallback"] or sync["has_cycle"] or sync["sweeps"] != 2:
            raise AssertionError(f"star flags n={n}")
    elif family == "gadgets":
        if not heap["fallback"] or sync["has_cycle"] or sync["sweeps"] != 2:
            raise AssertionError(f"gadgets flags n={n} sweeps={sync['sweeps']} fallback={heap['fallback']}")
    elif family == "path_gadget":
        if not heap["fallback"] or sync["has_cycle"]:
            raise AssertionError(f"path_gadget flags n={n}")
    if not close_enough(heap["V"], sync["V"]):
        raise AssertionError(f"values differ {family} n={n}")


def measure(family, builder, n):
    adj, exit_cost = builder(n)
    repeats = repeats_for(family, n)
    print(f"start {family} n={n} repeats={repeats}", flush=True)
    sync, sync_t, sync_cpu = time_median(lambda: solve_sync(adj, exit_cost, LAM), repeats)
    heap, heap_t, heap_cpu = time_median(lambda: solve_heap(adj, exit_cost, LAM), repeats)
    check(family, n, sync, heap)
    row = {
        "family": family,
        "n": n,
        "m": m_of(adj),
        "fallback": heap["fallback"],
        "has_cycle": sync["has_cycle"],
        "sync_sweeps": sync["sweeps"],
        "sync_sec": sync_t,
        "heap_sec": heap_t,
        "sync_cpu": sync_cpu,
        "heap_cpu": heap_cpu,
        "heap_extracts": heap["extracts"],
        "repeats": repeats,
    }
    with POINTS.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row) + "\n")
    print(
        f"done {family} n={n} wall {sync_t:.3f}/{heap_t:.3f}s "
        f"cpu {sync_cpu:.3f}/{heap_cpu:.3f}s "
        f"heap/sync={heap_cpu / sync_cpu:.3f} fallback={heap['fallback']} sweeps={sync['sweeps']}",
        flush=True,
    )
    return row


def load_rows():
    if not POINTS.exists():
        return []
    return [json.loads(line) for line in POINTS.read_text(encoding="utf-8").splitlines() if line.strip()]


def already(rows, family, n):
    return any(row["family"] == family and row["n"] == n for row in rows)


def draw(rows):
    FIGDIR.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["PingFang SC", "Heiti SC", "Arial Unicode MS"],
            "axes.unicode_minus": False,
            "font.size": 9,
            "pdf.fonttype": 42,
        }
    )
    panels = (
        ("path", "λ = 1/2，长路径\n不重开"),
        ("star", "λ = 1/2，星形\n不重开"),
        ("gadgets", "λ = 1/2，许多重开对\n很快重开"),
        ("path_gadget", "λ = 1/2，长路径加一个重开对\n然后重跑同步"),
    )
    fig, axes = plt.subplots(2, 2, figsize=(7.4, 6.2))
    for ax, (family, title) in zip(axes.ravel(), panels):
        subset = sorted((row for row in rows if row["family"] == family), key=lambda row: row["n"])
        ns = [row["n"] for row in subset]
        ax.plot(ns, [row["sync_cpu"] for row in subset], "o-", color="#b33a3a", label="同步 SolveEasy", markersize=4.5)
        ax.plot(ns, [row["heap_cpu"] for row in subset], "s-", color="#1f4e79", label="堆 SolveEasy", markersize=4.5)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(title)
        ax.set_xlabel("状态数 n")
        ax.set_ylabel("CPU 时间（秒）")
        ax.grid(True, which="major", color="0.85", linewidth=0.6)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.legend(frameon=False, fontsize=8)
        if family == "path":
            ax.set_xticks(ns)
            ax.set_xticklabels([str(n) for n in ns], rotation=90, ha="center", va="top")
    fig.tight_layout()
    fig.savefig(FIGDIR / "discount_compare.pdf")
    fig.savefig(FIGDIR / "discount_compare.png", dpi=200)
    plt.close(fig)


def write_tex(rows):
    def fmt(seconds):
        if seconds < 0.001:
            return f"{seconds * 1e6:.0f} $\\mu$s"
        if seconds < 1:
            return f"{seconds * 1e3:.2f} ms"
        return f"{seconds:.2f} s"

    names = {
        "path": "长路径，不重开",
        "star": "星形，不重开",
        "gadgets": "许多重开对，很快重开",
        "path_gadget": "长路径加一个重开对",
    }
    lines = [
        "\\begin{center}",
        "\\scriptsize",
        "\\begin{tabular}{llrrrr}",
        "\\toprule",
        "图 & $n$ & 同步 CPU & 堆 CPU & 堆/同步 & 重开 \\\\",
        "\\midrule",
    ]
    for family in ("path", "star", "gadgets", "path_gadget"):
        for row in sorted((item for item in rows if item["family"] == family), key=lambda item: item["n"]):
            ratio = row["heap_cpu"] / row["sync_cpu"]
            ratio_text = f"{ratio:.2f}" if ratio < 10 else f"{ratio:.1f}"
            lines.append(
                f"{names[family]} & {row['n']} & {fmt(row['sync_cpu'])} & {fmt(row['heap_cpu'])} "
                f"& {ratio_text} & {'是' if row['fallback'] else '否'} \\\\"
            )
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{center}", ""])
    (ROOT / "discount_table.tex").write_text("\n".join(lines), encoding="utf-8")


def main():
    rows = [row for row in load_rows() if "sync_cpu" in row]
    POINTS.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    jobs = []
    for n in (2000, 4000, 8000, 16000, 32000):
        jobs.append(("path", path_graph, n))
        jobs.append(("path_gadget", path_with_gadget, n))
    for n in (8000, 32000, 128000, 256000):
        jobs.append(("star", star_graph, n))
        jobs.append(("gadgets", gadgets, n))
    for family, builder, n in jobs:
        if already(rows, family, n):
            print(f"skip {family} n={n}", flush=True)
            continue
        rows.append(measure(family, builder, n))
    draw(rows)
    write_tex(rows)
    print("wrote figures/discount_compare.pdf", flush=True)


if __name__ == "__main__":
    main()
