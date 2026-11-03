#!/usr/bin/env python3
"""Correctness and complexity checks for heap-ordered SolveEasy.

Synchronous sweeps are the paper's SolveEasy. The heap pops the minimum
label and relaxes incoming arcs. A second real pop of any vertex discards
the heap labels and restarts the synchronous routine. At discount 1 with
positive arc costs that second pop does not occur, so the heap run is
Dijkstra.

    python3 experiments.py
"""

from __future__ import annotations

import heapq
import json
import math
import random
import time
from fractions import Fraction
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
FIGDIR = ROOT / "figures"
SEED = 20261027


class HeapKey:
    __slots__ = ("value", "node", "ver", "counter")

    def __init__(self, value, node, ver, counter):
        self.value = value
        self.node = node
        self.ver = ver
        self.counter = counter

    def __lt__(self, other):
        self.counter[0] += 1
        if self.value != other.value:
            return self.value < other.value
        if self.node != other.node:
            return self.node < other.node
        return self.ver < other.ver


def better(a, b):
    if a is None:
        return False
    if b is None:
        return True
    return a < b


def combine(g, lam, v):
    if v is None:
        return None
    return g + lam * v


def solve_sync(adj, exit_cost, lam):
    """Simultaneous Bellman sweeps, capped at n, as in SolveEasy."""
    n = len(exit_cost)
    current = list(exit_cost)
    scans = 0
    for sweep in range(1, n + 1):
        updated = list(exit_cost)
        changed = False
        for u, nbrs in enumerate(adj):
            best = updated[u]
            for v, g in nbrs:
                scans += 1
                alt = combine(g, lam, current[v])
                if better(alt, best):
                    best = alt
            if best != current[u]:
                changed = True
            updated[u] = best
        current = updated
        if not changed:
            return {"has_cycle": False, "V": current, "sweeps": sweep, "scans": scans}
    return {"has_cycle": True, "V": current, "sweeps": n, "scans": scans}


def solve_heap(adj, exit_cost, lam):
    """Minimum-label correction. The second real pop falls back to solve_sync."""
    n = len(exit_cost)
    incoming = [[] for _ in range(n)]
    for u, nbrs in enumerate(adj):
        for v, g in nbrs:
            incoming[v].append((u, g))

    labels = list(exit_cost)
    version = [0] * n
    real_pops = [0] * n
    counter = [0]
    heap = []
    for i, value in enumerate(labels):
        if value is not None:
            heapq.heappush(heap, HeapKey(value, i, 0, counter))

    extracts = 0
    scans = 0
    while heap:
        item = heapq.heappop(heap)
        if item.ver != version[item.node]:
            continue
        z = item.node
        real_pops[z] += 1
        extracts += 1
        if real_pops[z] == 2:
            sync = solve_sync(adj, exit_cost, lam)
            return {
                "has_cycle": sync["has_cycle"],
                "V": sync["V"],
                "fallback": True,
                "extracts": extracts,
                "scans": scans,
                "cmp": counter[0],
                "sync_sweeps": sync["sweeps"],
                "sync_scans": sync["scans"],
            }
        for u, g in incoming[z]:
            scans += 1
            alt = combine(g, lam, labels[z])
            if better(alt, labels[u]):
                labels[u] = alt
                version[u] += 1
                heapq.heappush(heap, HeapKey(labels[u], u, version[u], counter))
    return {
        "has_cycle": False,
        "V": labels,
        "fallback": False,
        "extracts": extracts,
        "scans": scans,
        "cmp": counter[0],
        "sync_sweeps": 0,
        "sync_scans": 0,
    }


def settle_forever(adj, exit_cost, lam):
    """Dijkstra that never reopens a popped vertex."""
    n = len(exit_cost)
    incoming = [[] for _ in range(n)]
    for u, nbrs in enumerate(adj):
        for v, g in nbrs:
            incoming[v].append((u, g))
    labels = list(exit_cost)
    version = [0] * n
    settled = [False] * n
    heap = []
    for i, value in enumerate(labels):
        if value is not None:
            heapq.heappush(heap, HeapKey(value, i, 0, [0]))
    while heap:
        item = heapq.heappop(heap)
        if item.ver != version[item.node] or settled[item.node]:
            continue
        z = item.node
        settled[z] = True
        for u, g in incoming[z]:
            if settled[u]:
                continue
            alt = combine(g, lam, labels[z])
            if better(alt, labels[u]):
                labels[u] = alt
                version[u] += 1
                heapq.heappush(heap, HeapKey(labels[u], u, version[u], [0]))
    return labels


def simple_costs(adj, exit_cost, lam):
    n = len(exit_cost)
    costs = [None] * n

    def dfs(u, seen, acc, disc, best):
        if exit_cost[u] is not None:
            cand = acc + disc * exit_cost[u]
            if better(cand, best[0]):
                best[0] = cand
        for v, g in adj[u]:
            if v in seen:
                continue
            seen.add(v)
            dfs(v, seen, acc + disc * g, disc * lam, best)
            seen.remove(v)

    for s in range(n):
        best = [None]
        dfs(s, {s}, 0, 1, best)
        costs[s] = best[0]
    return costs


def has_beneficial(adj, lam, costs):
    def dfs(start, u, seen, acc, disc, ell):
        for v, g in adj[u]:
            g_loop = acc + disc * g
            ell2 = ell + 1
            if v == start:
                base = costs[start]
                if base is None or g_loop < (1 - lam**ell2) * base:
                    return True
            elif v not in seen:
                seen.add(v)
                if dfs(start, v, seen, g_loop, disc * lam, ell2):
                    return True
                seen.remove(v)
        return False

    for s in range(len(adj)):
        if dfs(s, s, {s}, 0, 1, 0):
            return True
    return False


def reaches_exit(adj, exit_cost):
    n = len(exit_cost)
    incoming = [[] for _ in range(n)]
    for u, nbrs in enumerate(adj):
        for v, _g in nbrs:
            incoming[v].append(u)
    good = [value is not None for value in exit_cost]
    stack = [i for i, ok in enumerate(good) if ok]
    while stack:
        z = stack.pop()
        for u in incoming[z]:
            if not good[u]:
                good[u] = True
                stack.append(u)
    return all(good)


def m_of(adj):
    return sum(len(nbrs) for nbrs in adj)


def path_graph(n):
    adj = [[] for _ in range(n)]
    exit_cost = [None] * n
    for i in range(n - 1):
        adj[i].append((i + 1, 1))
    exit_cost[n - 1] = 1
    return adj, exit_cost


def path_noise(n, rng):
    adj, exit_cost = path_graph(n)
    penalty = n + 5
    for _ in range(n):
        adj[rng.randrange(n)].append((rng.randrange(n), penalty))
    return adj, exit_cost


def star_graph(n):
    center = n - 1
    adj = [[] for _ in range(n)]
    exit_cost = [None] * n
    exit_cost[center] = 1
    for i in range(center):
        adj[i].append((center, 1))
    return adj, exit_cost


def random_graph(n, extra, rng, self_loop_p=0.0):
    order = list(range(n))
    rng.shuffle(order)
    root = order[-1]
    exit_cost = [None] * n
    exit_cost[root] = rng.randint(1, 12)
    adj = [[] for _ in range(n)]
    for i, u in enumerate(order[:-1]):
        v = order[rng.randrange(i + 1, n)]
        adj[u].append((v, rng.randint(1, 8)))
    for _ in range(extra):
        adj[rng.randrange(n)].append((rng.randrange(n), rng.randint(1, 8)))
    if rng.random() < self_loop_p:
        u = rng.randrange(n)
        adj[u].append((u, 1))
        if exit_cost[u] is None:
            exit_cost[u] = rng.randint(8, 20)
    if not reaches_exit(adj, exit_cost):
        raise RuntimeError("generator produced a vertex that cannot exit")
    return adj, exit_cost


def fail(msg):
    raise AssertionError(msg)


def check_unit():
    half = Fraction(1, 2)
    adj = [[(1, 1)], []]
    exit_cost = [60, 100]

    heap_half = solve_heap(adj, exit_cost, half)
    sync_half = solve_sync(adj, exit_cost, half)
    if not heap_half["fallback"] or heap_half["has_cycle"] or sync_half["has_cycle"]:
        fail("gadget λ=1/2 should reopen and still report no cycle")
    if heap_half["V"] != [51, 100] or sync_half["V"] != heap_half["V"]:
        fail(f"gadget values {heap_half['V']} vs {sync_half['V']}")
    frozen = settle_forever(adj, exit_cost, half)
    if frozen != [60, 100]:
        fail(f"settle-forever should keep the direct exit, got {frozen}")

    heap_one = solve_heap(adj, exit_cost, 1)
    if heap_one["fallback"] or heap_one["V"] != [60, 100]:
        fail(f"gadget λ=1 {heap_one}")

    # The initial exit 100 is still in the heap after the label falls to 6.
    # That stale pop must not be counted as a reopen.
    adj = [[], [(0, 1)]]
    got = solve_heap(adj, [5, 100], 1)
    if got["fallback"] or got["V"] != [5, 6] or got["extracts"] != 2:
        fail(f"stale exit pop {got}")

    adj, exit_cost = path_graph(4)
    sync = solve_sync(adj, exit_cost, 1)
    heap = solve_heap(adj, exit_cost, 1)
    if sync["sweeps"] != 4 or sync["scans"] != 12 or sync["V"] != [4, 3, 2, 1]:
        fail(f"path sync {sync}")
    if heap["fallback"] or heap["extracts"] != 4 or heap["scans"] != 3 or heap["V"] != sync["V"]:
        fail(f"path heap {heap}")

    adj = [[(1, 1)], [(2, 1)], [(0, 1)]]
    exit_cost = [100, 100, 100]
    costs = simple_costs(adj, exit_cost, half)
    if not has_beneficial(adj, half, costs):
        fail("triangle should be beneficial")
    sync = solve_sync(adj, exit_cost, half)
    heap = solve_heap(adj, exit_cost, half)
    if not sync["has_cycle"] or not heap["has_cycle"] or not heap["fallback"]:
        fail("triangle detection")
    if heap["V"] != sync["V"]:
        fail("triangle values")

    adj = [[(0, 1)]]
    one = solve_sync(adj, [10], 1)
    loop = solve_sync(adj, [10], half)
    if one["has_cycle"] or not loop["has_cycle"]:
        fail("self-loop detection")
    if not has_beneficial(adj, half, simple_costs(adj, [10], half)):
        fail("brute missed the self-loop")

    adj, exit_cost = star_graph(5)
    # leaves 0..3 → center 4
    sync = solve_sync(adj, exit_cost, 1)
    heap = solve_heap(adj, exit_cost, 1)
    if sync["sweeps"] != 2 or sync["scans"] != 8 or heap["scans"] != 4 or heap["extracts"] != 5:
        fail(f"star {sync['sweeps']} {sync['scans']} {heap}")
    if heap["V"] != [2, 2, 2, 2, 1]:
        fail(f"star values {heap['V']}")

    adj, exit_cost = path_graph(12)
    heap = solve_heap(adj, exit_cost, half)
    sync = solve_sync(adj, exit_cost, half)
    if heap["fallback"] or heap["V"] != sync["V"]:
        fail("discounted path reopened or disagreed")
    for i, value in enumerate(heap["V"]):
        expect = (1 - half ** (12 - i)) / (1 - half)
        if value != expect:
            fail(f"closed form at {i}: {value} != {expect}")


def check_brute(trials_per_cell=120):
    lambdas = [
        Fraction(1),
        Fraction(1, 2),
        Fraction(2, 3),
        Fraction(4, 5),
        Fraction(9, 10),
    ]
    stats = {"trials": 0, "beneficial": 0, "clean": 0, "fallback_without_cycle": 0}
    for n in range(2, 8):
        print(f"  brute n={n}", flush=True)
        for lam in lambdas:
            loop_p = 0.0 if lam == 1 else 0.5
            for trial in range(trials_per_cell):
                rng = random.Random(SEED + 100000 * n + 1000 * lam.numerator + trial)
                adj, exit_cost = random_graph(n, extra=n, rng=rng, self_loop_p=loop_p)
                costs = simple_costs(adj, exit_cost, lam)
                cycle = has_beneficial(adj, lam, costs)
                sync = solve_sync(adj, exit_cost, lam)
                heap = solve_heap(adj, exit_cost, lam)
                stats["trials"] += 1
                if cycle:
                    stats["beneficial"] += 1
                else:
                    stats["clean"] += 1
                if sync["has_cycle"] != cycle or heap["has_cycle"] != cycle:
                    fail(
                        f"flags n={n} λ={lam} trial={trial} "
                        f"brute={cycle} sync={sync['has_cycle']} heap={heap['has_cycle']}"
                    )
                if heap["V"] != sync["V"]:
                    fail(f"values n={n} λ={lam} trial={trial}")
                if cycle and not heap["fallback"]:
                    fail(f"beneficial cycle finished in one pass n={n} trial={trial}")
                if lam == 1 and heap["fallback"]:
                    fail(f"λ=1 reopened n={n} trial={trial}")
                if (not cycle) and heap["fallback"]:
                    stats["fallback_without_cycle"] += 1
                if not cycle:
                    if sync["V"] != costs:
                        fail(f"sync != enumerated J n={n} λ={lam} trial={trial}")
                else:
                    for got, truth in zip(sync["V"], costs):
                        if truth is not None and better(truth, got):
                            fail(f"sync exceeded J* n={n} λ={lam} trial={trial}")

    for n in (5, 10, 15):
        for lam in lambdas:
            for trial in range(40):
                rng = random.Random(SEED + 900000 + n * 500 + trial + 17 * lam.numerator)
                adj, _exit = random_graph(n, extra=2 * n, rng=rng)
                adj = [[(v, 100) for v, _g in nbrs] for nbrs in adj]
                exit_cost = [1] * n
                sync = solve_sync(adj, exit_cost, lam)
                heap = solve_heap(adj, exit_cost, lam)
                stats["trials"] += 1
                stats["clean"] += 1
                if sync["has_cycle"] or heap["has_cycle"] or heap["fallback"]:
                    fail(f"expensive internal arcs n={n} λ={lam}")
                if sync["V"] != exit_cost or heap["V"] != exit_cost:
                    fail(f"expensive internal arcs changed labels n={n}")
    return stats


def check_lambda1_random():
    rows = []
    for n, trials, extra in ((40, 200, 80), (120, 80, 240), (400, 30, 800)):
        print(f"  λ=1 random n={n} x {trials}", flush=True)
        max_sweeps = 0
        for trial in range(trials):
            rng = random.Random(SEED + n * 10000 + trial)
            adj, exit_cost = random_graph(n, extra=extra, rng=rng, self_loop_p=0.3)
            m = m_of(adj)
            sync = solve_sync(adj, exit_cost, 1)
            heap = solve_heap(adj, exit_cost, 1)
            frozen = settle_forever(adj, exit_cost, 1)
            if sync["has_cycle"] or heap["has_cycle"] or heap["fallback"]:
                fail(f"λ=1 cycle n={n} trial={trial}")
            if heap["extracts"] != n or heap["scans"] != m or heap["V"] != sync["V"]:
                fail(
                    f"λ=1 shape n={n} extracts={heap['extracts']} "
                    f"scans={heap['scans']} m={m}"
                )
            if frozen != heap["V"]:
                fail(f"settle-forever disagreed n={n} trial={trial}")
            max_sweeps = max(max_sweeps, sync["sweeps"])
        rows.append(
            {
                "n": n,
                "trials": trials,
                "extra": extra,
                "max_sweeps": max_sweeps,
            }
        )
    return rows


def time_call(fn, repeats):
    samples = []
    result = None
    for _ in range(repeats):
        t0 = time.perf_counter()
        result = fn()
        samples.append(time.perf_counter() - t0)
    samples.sort()
    return result, samples[len(samples) // 2]


def scaling_family(name, builder, sizes):
    rows = []
    for n in sizes:
        rng = random.Random(SEED + 333 * n + len(name))
        adj, exit_cost = builder(n, rng)
        m = m_of(adj)
        repeats = 5 if n <= 400 else 3 if n <= 1600 else 2
        sync, sync_t = time_call(lambda: solve_sync(adj, exit_cost, 1), repeats)
        heap, heap_t = time_call(lambda: solve_heap(adj, exit_cost, 1), repeats)
        if sync["has_cycle"] or heap["fallback"] or heap["V"] != sync["V"]:
            fail(f"{name} n={n} disagreed")
        if heap["extracts"] != n or heap["scans"] != m:
            fail(f"{name} n={n} extracts={heap['extracts']} scans={heap['scans']} m={m}")
        if name in ("path", "path_noise") and sync["sweeps"] != n:
            fail(f"{name} n={n} sweeps={sync['sweeps']}")
        if name == "star" and sync["sweeps"] != 2:
            fail(f"star sweeps {sync['sweeps']}")
        rows.append(
            {
                "family": name,
                "n": n,
                "m": m,
                "sync_sweeps": sync["sweeps"],
                "sync_scans": sync["scans"],
                "heap_extracts": heap["extracts"],
                "heap_scans": heap["scans"],
                "heap_cmp": heap["cmp"],
                "sync_sec": sync_t,
                "heap_sec": heap_t,
            }
        )
        print(
            f"{name:12} n={n:6d} sweeps={sync['sweeps']:5d} "
            f"sync_scans={sync['scans']:10d} heap_scans={heap['scans']:8d} "
            f"cmp={heap['cmp']:10d} sync={sync_t:.4f}s heap={heap_t:.4f}s",
            flush=True,
        )
    return rows


def heap_only_large(builder, label, check):
    rows = []
    for n in (16000, 32000, 64000, 128000, 256000):
        adj, exit_cost = builder(n)
        heap, heap_t = time_call(lambda: solve_heap(adj, exit_cost, 1), 3)
        if heap["fallback"] or heap["extracts"] != n:
            fail(f"large {label} n={n}")
        check(n, heap)
        rows.append({"n": n, "sec": heap_t, "cmp": heap["cmp"], "scans": heap["scans"]})
        print(
            f"heap-only {label} n={n} {heap_t:.4f}s cmp={heap['cmp']} scans={heap['scans']}",
            flush=True,
        )
    return rows


def discounted_controls():
    half = Fraction(1, 2)
    out = []
    adj = [[(1, 1)], []]
    heap = solve_heap(adj, [60, 100], half)
    out.append({"name": "appendix gadget", "n": 2, "fallback": True, "has_cycle": False})
    if not heap["fallback"] or heap["has_cycle"] or heap["V"] != [51, 100]:
        fail(f"gadget control {heap}")

    for n in (50, 400, 2000):
        adj, exit_cost = path_graph(n)
        # Float is exact for the control flow: each vertex is relaxed once, from the sink backward.
        heap = solve_heap(adj, exit_cost, 0.5)
        if heap["fallback"] or heap["extracts"] != n or heap["scans"] != n - 1:
            fail(f"float path reopened n={n} extracts={heap['extracts']}")
        out.append({"name": "path toward the sink", "n": n, "fallback": False, "has_cycle": False})

    for copies in (10, 100):
        adj = []
        exit_cost = []
        for _ in range(copies):
            base = len(exit_cost)
            adj.append([(base + 1, 1)])
            adj.append([])
            exit_cost.extend([60, 100])
        heap = solve_heap(adj, exit_cost, half)
        sync = solve_sync(adj, exit_cost, half)
        if not heap["fallback"] or heap["has_cycle"] or sync["has_cycle"] or heap["V"] != sync["V"]:
            fail(f"gadget copies n={2 * copies}")
        out.append(
            {
                "name": "disjoint gadgets",
                "n": 2 * copies,
                "fallback": True,
                "has_cycle": False,
            }
        )
    return out


def loglog_slope(ns, ys):
    x = np.log(np.asarray(ns, dtype=float))
    y = np.log(np.asarray(ys, dtype=float))
    slope, intercept = np.polyfit(x, y, 1)
    pred = slope * x + intercept
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - float(y.mean())) ** 2))
    r2 = 1.0 if ss_tot == 0 else 1.0 - ss_res / ss_tot
    return float(slope), float(r2)


def fmt_int(x):
    return f"{x:,}"


def fmt_sec(x):
    if x < 0.001:
        return f"{x * 1e6:.0f} $\\mu$s"
    if x < 1:
        return f"{x * 1e3:.2f} ms"
    return f"{x:.2f} s"


def write_table(rows, headers, getters):
    spec = "l" + "r" * (len(headers) - 1)
    lines = [
        f"\\begin{{tabular}}{{{spec}}}",
        "\\toprule",
        " & ".join(headers) + " \\\\",
        "\\midrule",
    ]
    for row in rows:
        lines.append(" & ".join(get(row) for get in getters) + " \\\\")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    return "\n".join(lines)


def family_table(rows):
    return write_table(
        rows,
        [
            "$n$",
            "sweeps",
            "sync scans",
            "heap scans",
            "heap cmp",
            "sync",
            "heap",
            "ratio",
        ],
        [
            lambda r: str(r["n"]),
            lambda r: str(r["sync_sweeps"]),
            lambda r: fmt_int(r["sync_scans"]),
            lambda r: fmt_int(r["heap_scans"]),
            lambda r: fmt_int(r["heap_cmp"]),
            lambda r: fmt_sec(r["sync_sec"]),
            lambda r: fmt_sec(r["heap_sec"]),
            lambda r: f"{r['sync_scans'] / r['heap_scans']:.1f}",
        ],
    )


def write_report(payload, summary):
    path_rows = payload["scaling"]["path"]
    noise_rows = payload["scaling"]["path_noise"]
    star_rows = payload["scaling"]["star"]
    brute = payload["brute"]
    path_ratio = [r["sync_scans"] / r["heap_scans"] for r in path_rows]
    star_ratio = [r["sync_scans"] / r["heap_scans"] for r in star_rows]
    cmp_ratios = [r["heap_cmp"] / (r["n"] * math.log2(r["n"])) for r in star_rows]
    lines = []
    a = lines.append
    a("\\section{实验结果}")
    a(f"随机种子是 \\texttt{{{SEED}}}。重跑命令是 \\texttt{{python3 experiments.py}}。")
    a(
        "正确性有三套对照：枚举全部简单路和简单圈、同步 SolveEasy，以及弹出后拒绝重开的 Dijkstra。"
        f"枚举和专用构造一共 {fmt_int(brute['trials'])} 次，"
        f"其中有 beneficial cycle 的 {fmt_int(brute['beneficial'])} 次，"
        f"没有的 {fmt_int(brute['clean'])} 次，"
        f"没有圈但堆仍然重开的 {fmt_int(brute['fallback_without_cycle'])} 次。"
        "圈的标志、两边的数值，以及“有圈则堆一定退回”，全部一致。"
        "没有圈的那些里面，有 600 次把内部弧固定为代价 100、出口固定为代价 1；"
        "五个折扣下堆都不重开，数值等于出口代价。"
    )
    a("\\paragraph{$\\lambda=1$ 的随机图。}")
    a("每条弧的代价为正。每个点恰好弹出一次，弧检查次数恰好等于内部弧数。")
    a("\\begin{center}")
    a(
        write_table(
            payload["lambda1_random"],
            ["$n$", "trials", "extra arcs", "extracts$/n$", "scans$/m$", "max sweeps"],
            [
                lambda r: str(r["n"]),
                lambda r: str(r["trials"]),
                lambda r: str(r["extra"]),
                lambda r: "1",
                lambda r: "1",
                lambda r: str(r["max_sweeps"]),
            ],
        )
    )
    a("\\end{center}")
    a(
        "同步扫描的轮数取决于最优路的跳数，这批图里最深的一次是 "
        f"{max(r['max_sweeps'] for r in payload['lambda1_random'])} 轮。"
        "堆的弹出次数不随这个深度增加。"
    )
    a("\\paragraph{路径。}")
    a(
        "唯一的最优路有 $n-1$ 条内部弧。同步算法恰好扫 $n$ 轮，弧检查次数恰好是 $n(n-1)$。"
        "堆算法的弧检查次数恰好是 $n-1$，二者之比就是 $n$。"
    )
    a("\\begin{center}")
    a("\\scriptsize")
    a(family_table(path_rows))
    a("\\end{center}")
    a(
        "弧检查次数对 $n$ 的对数斜率：同步 "
        f"{summary['slope_sync']:.3f}（$R^2={summary['r2_sync']:.4f}$），堆 "
        f"{summary['slope_heap_scans']:.3f}（$R^2={summary['r2_heap_scans']:.4f}$）。"
        "这条路径上堆的比较次数全是 0：待弹出的点始终只有汇的方向上的那一个，"
        "二叉堆没有发生比较，运行比 $n\\log n$ 更小，是线性的。"
        "墙钟斜率因此也接近 1，而不是 $1$ 再加一个对数："
        f"同步 {summary['slope_sync_t']:.3f}，堆 {summary['slope_heap_t']:.3f}，"
        f"把路径加到 $n=256000$ 之后是 {summary['slope_path_large']:.3f}"
        f"（$R^2={summary['r2_path_large']:.3f}$）。"
    )
    a("\\paragraph{路径加上打不赢主干的噪声弧。}")
    a(
        "额外 $n$ 条代价为 $n+5$ 的弧不改变最短路，同步算法仍扫 $n$ 轮，但每轮要看这些弧。"
        f"扫描比从 {noise_rows[0]['sync_scans'] / noise_rows[0]['heap_scans']:.0f} "
        f"增到 {noise_rows[-1]['sync_scans'] / noise_rows[-1]['heap_scans']:.0f}，与 $n$ 同阶。"
        "堆算法仍然每条弧只检查一次。"
    )
    a("\\begin{center}")
    a("\\scriptsize")
    a(family_table(noise_rows))
    a("\\end{center}")
    a("\\paragraph{星形。}")
    a(
        "每个叶子一步到达中心，同步算法扫 2 轮。"
        f"扫描比停在 {min(star_ratio):.1f} 到 {max(star_ratio):.1f}，不随 $n$ 增长。"
        "深度是常数时，两边的弧检查都是线性的；二叉堆还要付 $\\log n$ 次比较，墙钟不一定更快。"
        "这条算法去掉的是深度因子，不是 $D=1$ 时的对数因子。"
        "星形图把全部叶子一次放进堆，比较次数才达到二叉堆的上界。"
        f"比较次数对 $n$ 的对数斜率是 {summary['slope_cmp']:.3f}"
        f"（$R^2={summary['r2_cmp']:.4f}$），"
        "除以 $n\\log_2 n$ 的最小、中位、最大值是 "
        f"{min(cmp_ratios):.2f}、{float(np.median(cmp_ratios)):.2f}、{max(cmp_ratios):.2f}。"
        f"把星形图加到 $n=256000$，墙钟斜率是 {summary['slope_star_large']:.3f}"
        f"（$R^2={summary['r2_star_large']:.3f}$），和 $n\\log n$ 而不是 $n^2$ 同阶。"
    )
    a("\\begin{center}")
    a("\\scriptsize")
    a(family_table(star_rows))
    a("\\end{center}")
    a("\\paragraph{$\\lambda=1/2$。}")
    a("\\begin{center}")
    a(
        write_table(
            payload["discounted"],
            ["instance", "$n$", "fallback", "has cycle"],
            [
                lambda r: r["name"],
                lambda r: str(r["n"]),
                lambda r: "yes" if r["fallback"] else "no",
                lambda r: "yes" if r["has_cycle"] else "no",
            ],
        )
    )
    a("\\end{center}")
    a(
        "附录里的两个点会重开，同步检验仍然判定没有 beneficial cycle。"
        "若弹出后拒绝重开，Dijkstra 把 $B$ 留在 60，正确值是 $1+\\tfrac12\\cdot 100=51$。"
        "指向汇的一条路径在 $\\lambda=1/2$ 时不会重开，因为最优顺序本来就是从汇往回弹。"
        "所以“每个点一次”在 $\\lambda=1$ 且代价为正时对所有这样的图成立；"
        "$\\lambda<1$ 时只在一部分图上碰巧成立。"
    )
    a(
        f"路径上同步扫描与堆扫描之比从 {path_ratio[0]:.0f} 增到 {path_ratio[-1]:.0f}。"
    )
    a("\\begin{figure}[t]")
    a("\\centering")
    a("\\includegraphics[width=\\textwidth]{figures/scaling.pdf}")
    a(
        "\\caption{左：路径上的弧检查次数，同步为 $n(n-1)$，堆为 $n-1$。这条路上堆的比较次数是 $0$。"
        "中：星形图的堆比较次数，与 $n\\log_2 n$ 同阶。"
        "右：路径的墙钟；堆在这条路上是线性的，虚线延续到 $n=256000$。}"
    )
    a("\\label{fig:scaling}")
    a("\\end{figure}")
    (ROOT / "results_generated.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (ROOT / "results.json").write_text(
        json.dumps(payload, indent=2, default=str), encoding="utf-8"
    )


def make_figure(path_rows, star_rows, path_large):
    FIGDIR.mkdir(parents=True, exist_ok=True)
    ns = [r["n"] for r in path_rows]
    fig, axes = plt.subplots(1, 3, figsize=(9.2, 3.05))

    axes[0].loglog(ns, [r["sync_scans"] for r in path_rows], "o-", color="#b33a3a", label="synchronous")
    axes[0].loglog(ns, [r["heap_scans"] for r in path_rows], "s-", color="#1f4e79", label="heap")
    axes[0].set_xlabel("n")
    axes[0].set_ylabel("arc scans")
    axes[0].set_title("path")
    axes[0].legend(frameon=False)

    star_n = [r["n"] for r in star_rows]
    guide_n = np.asarray(star_n, dtype=float)
    guide = guide_n * np.log2(guide_n)
    cmp = np.asarray([r["heap_cmp"] for r in star_rows], dtype=float)
    axes[1].loglog(star_n, cmp, "s-", color="#1f4e79", label="heap comparisons")
    axes[1].loglog(star_n, float(np.median(cmp / guide)) * guide, "--", color="0.45", label=r"$c\,n\log_2 n$")
    axes[1].set_xlabel("n")
    axes[1].set_ylabel("comparisons")
    axes[1].set_title("star")
    axes[1].legend(frameon=False)

    axes[2].loglog(ns, [r["sync_sec"] for r in path_rows], "o-", color="#b33a3a", label="synchronous")
    axes[2].loglog(ns, [max(r["heap_sec"], 1e-6) for r in path_rows], "s-", color="#1f4e79", label="heap")
    axes[2].loglog(
        [r["n"] for r in path_large],
        [r["sec"] for r in path_large],
        "s--",
        color="#1f4e79",
        alpha=0.75,
        label="heap, larger $n$",
    )
    axes[2].set_xlabel("n")
    axes[2].set_ylabel("seconds")
    axes[2].set_title("path")
    axes[2].legend(frameon=False, fontsize=8)

    for axis in axes:
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIGDIR / "scaling.pdf")
    plt.close(fig)


def summarize(payload):
    path_rows = payload["scaling"]["path"]
    star_rows = payload["scaling"]["star"]
    path_large = payload["path_large"]
    star_large = payload["star_large"]
    positive_heap = [r for r in path_rows if r["heap_sec"] > 0]
    slope_sync, r2_sync = loglog_slope([r["n"] for r in path_rows], [r["sync_scans"] for r in path_rows])
    slope_heap_scans, r2_heap_scans = loglog_slope(
        [r["n"] for r in path_rows], [r["heap_scans"] for r in path_rows]
    )
    slope_cmp, r2_cmp = loglog_slope([r["n"] for r in star_rows], [r["heap_cmp"] for r in star_rows])
    slope_sync_t, _r2_sync_t = loglog_slope(
        [r["n"] for r in path_rows], [r["sync_sec"] for r in path_rows]
    )
    slope_heap_t, _r2_heap_t = loglog_slope(
        [r["n"] for r in positive_heap], [r["heap_sec"] for r in positive_heap]
    )
    slope_path_large, r2_path_large = loglog_slope(
        [r["n"] for r in path_large], [r["sec"] for r in path_large]
    )
    slope_star_large, r2_star_large = loglog_slope(
        [r["n"] for r in star_large], [r["sec"] for r in star_large]
    )
    summary = {
        "slope_sync": slope_sync,
        "r2_sync": r2_sync,
        "slope_heap_scans": slope_heap_scans,
        "r2_heap_scans": r2_heap_scans,
        "slope_cmp": slope_cmp,
        "r2_cmp": r2_cmp,
        "slope_sync_t": slope_sync_t,
        "slope_heap_t": slope_heap_t,
        "slope_path_large": slope_path_large,
        "r2_path_large": r2_path_large,
        "slope_star_large": slope_star_large,
        "r2_star_large": r2_star_large,
    }
    if any(r["heap_cmp"] != 0 for r in path_rows):
        fail("path heap unexpectedly compared keys")
    if abs(slope_sync - 2) > 0.05 or abs(slope_heap_scans - 1) > 0.05:
        fail(f"scan slopes {slope_sync} {slope_heap_scans}")
    if not 1.0 <= slope_cmp <= 1.35:
        fail(f"star comparison slope {slope_cmp}")
    ratios = [r["heap_cmp"] / (r["n"] * math.log2(r["n"])) for r in star_rows]
    if min(ratios) < 0.5 or max(ratios) > 2.5:
        fail(f"star comparison ratio {min(ratios)} {max(ratios)}")
    if not 0.8 <= slope_path_large <= 1.35:
        fail(f"large path time slope {slope_path_large}")
    if not 0.9 <= slope_star_large <= 1.45:
        fail(f"large star time slope {slope_star_large}")
    return summary


def main():
    print("unit checks", flush=True)
    check_unit()
    print("brute and agreement", flush=True)
    brute = check_brute()
    print(brute, flush=True)
    print("lambda=1 random", flush=True)
    lam_rows = check_lambda1_random()
    print("discounted controls", flush=True)
    discounted = discounted_controls()
    sizes = [50, 100, 200, 400, 800, 1600, 3200, 6400]
    print("scaling", flush=True)
    scaling = {
        "path": scaling_family("path", lambda n, rng: path_graph(n), sizes),
        "path_noise": scaling_family("path_noise", path_noise, sizes),
        "star": scaling_family("star", lambda n, rng: star_graph(n), sizes),
    }
    print("large heap", flush=True)

    def check_path(n, heap):
        if heap["scans"] != n - 1 or heap["cmp"] != 0:
            fail(f"large path scans={heap['scans']} cmp={heap['cmp']}")

    def check_star(n, heap):
        if heap["scans"] != n - 1 or heap["cmp"] <= 0:
            fail(f"large star scans={heap['scans']} cmp={heap['cmp']}")

    path_large = heap_only_large(path_graph, "path", check_path)
    star_large = heap_only_large(star_graph, "star", check_star)
    payload = {
        "brute": brute,
        "lambda1_random": lam_rows,
        "discounted": discounted,
        "scaling": scaling,
        "path_large": path_large,
        "star_large": star_large,
    }
    summary = summarize(payload)
    write_report(payload, summary)
    make_figure(scaling["path"], scaling["star"], path_large)
    print("SUMMARY", json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
