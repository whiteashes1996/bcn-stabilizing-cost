#!/usr/bin/env python3
"""Wall-clock comparison of synchronous SolveEasy and the heap version.

Path graphs are the deep case. Star graphs have depth 1. Both use λ = 1.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter

from experiments import m_of, path_graph, solve_heap, solve_sync, star_graph

ROOT = Path(__file__).resolve().parent
FIGDIR = ROOT / "figures"
POINTS = ROOT / "runtime_points.jsonl"


def time_median(fn, repeats):
    samples = []
    result = None
    for _ in range(repeats):
        t0 = time.perf_counter()
        result = fn()
        samples.append(time.perf_counter() - t0)
    samples.sort()
    return result, samples[len(samples) // 2]


def repeats_for(n):
    if n <= 4000:
        return 3
    if n <= 16000:
        return 2
    return 1


def measure(family, builder, n):
    adj, exit_cost = builder(n)
    m = m_of(adj)
    repeats = repeats_for(n)
    print(f"start {family} n={n} repeats={repeats}", flush=True)
    sync, sync_t = time_median(lambda: solve_sync(adj, exit_cost, 1), repeats)
    heap, heap_t = time_median(lambda: solve_heap(adj, exit_cost, 1), repeats)
    if sync["has_cycle"] or heap["fallback"] or heap["has_cycle"]:
        raise AssertionError(f"{family} n={n} flags")
    if heap["extracts"] != n or heap["scans"] != m or heap["V"] != sync["V"]:
        raise AssertionError(
            f"{family} n={n} extracts={heap['extracts']} scans={heap['scans']} m={m}"
        )
    row = {
        "family": family,
        "n": n,
        "m": m,
        "sync_sweeps": sync["sweeps"],
        "sync_scans": sync["scans"],
        "heap_scans": heap["scans"],
        "heap_cmp": heap["cmp"],
        "sync_sec": sync_t,
        "heap_sec": heap_t,
        "repeats": repeats,
    }
    with POINTS.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row) + "\n")
    print(
        f"done {family} n={n} sync={sync_t:.4f}s heap={heap_t:.4f}s "
        f"ratio={sync_t / heap_t:.1f} sweeps={sync['sweeps']}",
        flush=True,
    )
    return row


def load_rows():
    rows = []
    if not POINTS.exists():
        return rows
    for line in POINTS.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def already(rows, family, n):
    return any(row["family"] == family and row["n"] == n for row in rows)


def style_axis(ax):
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, which="major", color="0.85", linewidth=0.6)
    ax.grid(True, which="minor", color="0.93", linewidth=0.4)


def draw(rows):
    FIGDIR.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["PingFang SC", "Heiti SC", "Arial Unicode MS"],
            "axes.unicode_minus": False,
            "font.size": 10,
            "pdf.fonttype": 42,
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.7))
    panels = (
        ("path", "λ = 1，长路径", axes[0]),
        ("star", "星形，深度为 1", axes[1]),
    )
    for family, title, ax in panels:
        subset = sorted((row for row in rows if row["family"] == family), key=lambda row: row["n"])
        ns = [row["n"] for row in subset]
        ax.plot(ns, [row["sync_sec"] for row in subset], "o-", color="#b33a3a", label="同步 SolveEasy", markersize=4.5)
        ax.plot(ns, [row["heap_sec"] for row in subset], "s-", color="#1f4e79", label="堆 SolveEasy", markersize=4.5)
        last = subset[-1]
        ratio = last["sync_sec"] / last["heap_sec"]
        ratio_text = f"{ratio:.0f}" if ratio >= 10 else f"{ratio:.2f}"
        ax.text(
            0.97,
            0.50 if family == "path" else 0.42,
            f"n={last['n']} 时\n同步/堆 = {ratio_text}",
            transform=ax.transAxes,
            ha="right",
            va="center",
            fontsize=8,
            color="0.25",
        )
        style_axis(ax)
        ax.set_title(title)
        ax.set_xlabel("状态数 n")
        ax.set_ylabel("运行时间（秒）")
        ax.legend(frameon=False, loc="upper left")
        if family == "path":
            ax.xaxis.set_major_locator(FixedLocator(ns))
            ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _pos: f"{int(round(value))}"))
            ax.xaxis.set_minor_formatter(NullFormatter())
            ax.tick_params(axis="x", which="minor", labelbottom=False)
    fig.tight_layout()
    fig.canvas.draw()
    for label in axes[0].get_xticklabels():
        label.set_rotation(90)
        label.set_rotation_mode("anchor")
        label.set_ha("center")
        label.set_va("top")
    fig.subplots_adjust(bottom=0.30)
    fig.savefig(FIGDIR / "runtime_compare.pdf")
    fig.savefig(FIGDIR / "runtime_compare.png", dpi=200)
    plt.close(fig)


def write_tex(rows):
    def fmt(seconds):
        if seconds < 0.001:
            return f"{seconds * 1e6:.0f} $\\mu$s"
        if seconds < 1:
            return f"{seconds * 1e3:.2f} ms"
        return f"{seconds:.2f} s"

    lines = [
        "\\begin{center}",
        "\\scriptsize",
        "\\begin{tabular}{llrrrr}",
        "\\toprule",
        "图 & $n$ & 同步 & 堆 & 同步/堆 & 同步轮数 \\\\",
        "\\midrule",
    ]
    order = {"path": "长路径", "star": "星形"}
    for family in ("path", "star"):
        for row in sorted((item for item in rows if item["family"] == family), key=lambda item: item["n"]):
            ratio = row["sync_sec"] / row["heap_sec"]
            ratio_text = f"{ratio:.0f}" if ratio >= 10 else f"{ratio:.2f}"
            lines.append(
                f"{order[family]} & {row['n']} & {fmt(row['sync_sec'])} & {fmt(row['heap_sec'])} "
                f"& {ratio_text} & {row['sync_sweeps']} \\\\"
            )
    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}",
            "\\end{center}",
            "",
        ]
    )
    (ROOT / "runtime_table.tex").write_text("\n".join(lines), encoding="utf-8")


def main():
    rows = load_rows()
    jobs = [("path", path_graph, n) for n in (2000, 4000, 8000, 16000, 32000, 64000)]
    jobs += [("star", star_graph, n) for n in (2000, 8000, 32000, 128000, 256000)]
    for family, builder, n in jobs:
        if already(rows, family, n):
            print(f"skip {family} n={n}", flush=True)
            continue
        rows.append(measure(family, builder, n))
    draw(rows)
    write_tex(rows)
    print("wrote figures/runtime_compare.pdf", flush=True)


if __name__ == "__main__":
    main()
