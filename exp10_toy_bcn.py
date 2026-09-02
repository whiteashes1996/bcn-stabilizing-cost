"""E10: a three-node BCN on which value and policy iteration both fail.

The BCN has n=3 state nodes and m=1 control node, so N=8 and M=2.  With the
natural stage cost g(x,u) = Hamming(x, x_e) + [u != u_e] the state transition
graph contains a two-state cycle whose per-step cost is 1, while the only way
out of it first pays 2 and then lands on the state farthest from the target.
Under discounting the cycle is beneficial, so the unconstrained optimum
V_lambda* is attained by an improper policy: both value iteration and policy
iteration leave two of the eight states unstabilized, whereas Algorithm master
returns a proper feedback on all of them.

The example exercises every stage of the master algorithm (reverse
reachability, condensation, downstream folding, label correcting on the
easy components, SolveCore on the single hard one) on a graph small enough
to verify by hand.
"""
from __future__ import annotations

import math
import os
import sys
from typing import Dict, List, Optional, Set, Tuple

sys.path.insert(0, os.path.dirname(__file__))

from core_solver import (  # noqa: E402
    WeightedGraph,
    augment_component,
    brute_force_simple_paths,
    build_graph,
    condense,
    detect_beneficial_cycle_v2,
    enumerate_beneficial_cycles,
    master,
    policy_iteration_graph,
    reverse_reachable,
    simulate_closed_loop,
    solve_core_heldkarp,
    vi_policy,
    vi_unconstrained,
)

RESULTS = os.path.join(os.path.dirname(__file__), "results")
FIGURES = os.path.join(os.path.dirname(__file__), "..", "figures")

N_NODES = 3
N = 8
M = 2
XE = 1  # delta_8^1 = (1,1,1)
UE = 1
LAM = 0.5

# Column-major network transition matrix L = delta_8[...]: entry (u-1)*N+(x-1).
L = [
    1, 3, 2, 5, 1, 5, 5, 1,   # u = delta_2^1
    1, 8, 2, 5, 1, 5, 5, 1,   # u = delta_2^2
]

# Analytic threshold: the cycle is beneficial iff 1/(1-lam) < 2 + 3*lam, i.e.
# 3*lam^2 - lam - 1 < 0.
LAM_STAR = (1 + math.sqrt(13)) / 6

CYCLE = {2, 3}


def bits(i: int) -> Tuple[int, int, int]:
    """delta_8^i -> logical vector, with delta_8^1 = (1,1,1)."""
    v = N - i
    return ((v >> 2) & 1, (v >> 1) & 1, v & 1)


def hamming_to_target(i: int) -> int:
    return sum(1 for a, b in zip(bits(i), bits(XE)) if a != b)


def f(x: int, u: int) -> int:
    return L[(u - 1) * N + (x - 1)]


def g(x: int, u: int) -> float:
    return float(hamming_to_target(x) + (0 if u == UE else 1))


def build() -> WeightedGraph:
    return build_graph(f, g, list(range(1, N + 1)), list(range(1, M + 1)), xe=XE)


# ---------------------------------------------------------------------------
# Logical form of the network (Quine-McCluskey over u, x1, x2, x3)
# ---------------------------------------------------------------------------


def _combine(a: Tuple[int, int], b: Tuple[int, int]) -> Optional[Tuple[int, int]]:
    if a[1] != b[1]:
        return None
    diff = a[0] ^ b[0]
    if diff and diff & (diff - 1) == 0:
        return (a[0] & ~diff, a[1] | diff)
    return None


def _covers(term: Tuple[int, int], m: int) -> bool:
    return term[0] == m & ~term[1]


def quine_mccluskey(minterms: List[int], nvars: int) -> List[Tuple[int, int]]:
    """Prime implicants covering the minterms, reduced by a greedy cover."""
    if not minterms:
        return []
    if len(minterms) == 1 << nvars:
        return [(0, (1 << nvars) - 1)]
    terms = {(m, 0) for m in minterms}
    primes: Set[Tuple[int, int]] = set()
    while terms:
        nxt: Set[Tuple[int, int]] = set()
        used: Set[Tuple[int, int]] = set()
        tl = sorted(terms)
        for i in range(len(tl)):
            for j in range(i + 1, len(tl)):
                c = _combine(tl[i], tl[j])
                if c is not None:
                    nxt.add(c)
                    used.add(tl[i])
                    used.add(tl[j])
        primes |= terms - used
        terms = nxt

    uncovered = set(minterms)
    chosen: List[Tuple[int, int]] = []
    while uncovered:
        best = max(primes, key=lambda p: (sum(1 for m in uncovered if _covers(p, m)), -p[0]))
        gain = [m for m in uncovered if _covers(best, m)]
        if not gain:
            break
        chosen.append(best)
        uncovered -= set(gain)
        primes.discard(best)
    return chosen


def render_dnf(terms: List[Tuple[int, int]], names: List[str]) -> str:
    if not terms:
        return "0"
    nvars = len(names)
    if terms == [(0, (1 << nvars) - 1)]:
        return "1"
    parts = []
    for bitsval, mask in terms:
        lits = []
        for k, nm in enumerate(names):
            bit = 1 << (nvars - 1 - k)
            if mask & bit:
                continue
            lits.append(nm if bitsval & bit else f"~{nm}")
        parts.append(" & ".join(lits) if lits else "1")
    return " | ".join(f"({p})" if " & " in p else p for p in parts)


def logical_form() -> List[str]:
    """x_i(t+1) as Boolean functions of (u, x1, x2, x3)."""
    names = ["u", "x1", "x2", "x3"]
    minterms: List[List[int]] = [[], [], []]
    for u in (1, 2):
        U = 1 if u == UE else 0
        for x in range(1, N + 1):
            a, b, c = bits(x)
            m = (U << 3) | (a << 2) | (b << 1) | c
            nxt = bits(f(x, u))
            for k in range(3):
                if nxt[k]:
                    minterms[k].append(m)
    return [
        f"x{k+1}(t+1) = " + render_dnf(quine_mccluskey(sorted(minterms[k]), 4), names)
        for k in range(3)
    ]


# ---------------------------------------------------------------------------
# Solve
# ---------------------------------------------------------------------------


def improper_states(
    mu: Dict[int, Tuple[int, int]], states: Set[int]
) -> Set[int]:
    bad = set()
    for x in states:
        if x == XE:
            continue
        proper, _steps, _traj = simulate_closed_loop(f, mu, x, XE, max_steps=4 * N)
        if not proper:
            bad.add(x)
    return bad


def solve_at(lam: float) -> dict:
    G = build()
    R = reverse_reachable(G, XE)
    V = vi_unconstrained(G, lam, R, XE)
    mu_vi = vi_policy(G, V, lam, R)
    V_pi, mu_pi = policy_iteration_graph(G, XE, lam, R)
    J, mu_master = master(G, XE, lam, ue=UE)
    return {
        "G": G,
        "R": R,
        "V": V,
        "V_pi": V_pi,
        "mu_vi": mu_vi,
        "mu_pi": mu_pi,
        "J": J,
        "mu_master": mu_master,
    }


def lambda_threshold(lo: float = 1e-4, hi: float = 1 - 1e-4, iters: int = 60) -> float:
    """Bisect the largest lambda at which the component still has a beneficial cycle."""

    def has_bc(lam: float) -> bool:
        G = build()
        J_known = {XE: 0.0}
        # states outside the cycle are solved first; only J(delta_8^8) is needed
        J_known[8] = g(8, UE) + lam * 0.0
        internal, e = augment_component(G, CYCLE, J_known, lam)
        flag, _trapped, _V = detect_beneficial_cycle_v2(CYCLE, internal, e, lam)
        return flag

    assert has_bc(lo), "cycle should be beneficial for small lambda"
    assert not has_bc(hi), "cycle should not be beneficial for lambda near 1"
    for _ in range(iters):
        mid = (lo + hi) / 2
        if has_bc(mid):
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

POS = {
    1: (4.0, 0.0),
    5: (2.4, 0.0),
    8: (2.0, -1.5),
    2: (0.0, 0.75),
    3: (0.0, -0.75),
    4: (-2.2, 1.4),
    6: (-2.2, 0.0),
    7: (-2.2, -1.4),
}

TRAP = "#d1495b"
GOOD = "#1a7f37"
FAINT = "#c8ccd0"


def _draw_panel(ax, mu, title, colour, trapped, highlight=None):
    import networkx as nx

    G = build()
    base = nx.DiGraph()
    base.add_nodes_from(POS)
    all_edges = {(x, y) for x, _u, y, _w in G.arcs if x != y}
    loop_edges = {(x, y) for x, (_u, y) in mu.items() if x != y}
    trap_edges = {(x, y) for x, y in loop_edges if x in trapped and y in trapped}
    key_edges = set(highlight or ()) & loop_edges
    plain_edges = loop_edges - trap_edges - key_edges
    base.add_edges_from(all_edges | loop_edges)

    node_fill, node_edge = [], []
    for nd in base.nodes():
        if nd == XE:
            node_fill.append("#eaf5ec")
            node_edge.append(GOOD)
        elif nd in trapped:
            node_fill.append("#fdeced")
            node_edge.append(TRAP)
        else:
            node_fill.append("white")
            node_edge.append("#444444")
    nx.draw_networkx_nodes(
        base, POS, ax=ax, node_size=900, node_color=node_fill,
        edgecolors=node_edge, linewidths=1.6,
    )
    nx.draw_networkx_labels(
        base, POS, ax=ax, font_size=11,
        labels={i: rf"$\delta_8^{{{i}}}$" for i in POS},
    )

    def edges(lst, col, width, alpha=1.0):
        if not lst:
            return
        nx.draw_networkx_edges(
            base, POS, ax=ax, edgelist=sorted(lst), edge_color=col, width=width,
            alpha=alpha, arrows=True, arrowsize=15, arrowstyle="-|>",
            node_size=900, min_source_margin=12, min_target_margin=12,
            connectionstyle="arc3,rad=0.14",
        )

    edges(all_edges - loop_edges, FAINT, 1.0)
    edges(plain_edges, "#5b6167", 1.6)
    edges(trap_edges, TRAP, 2.6)
    edges(key_edges, colour, 2.6)
    ax.set_title(title, fontsize=10.5)
    ax.set_xlim(-3.1, 4.9)
    ax.set_ylim(-2.4, 2.2)
    ax.axis("off")


def figure_stg(sol: dict, trapped: Set[int]) -> str:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 3.4))
    _draw_panel(
        axes[0], sol["mu_vi"],
        r"(a) closed loop of the $V_\lambda^*$-induced policy: "
        r"$\delta_8^2,\delta_8^3$ never reach $x_e$",
        TRAP, trapped,
    )
    _draw_panel(
        axes[1], sol["mu_master"],
        r"(b) closed loop of $\mu^*$ from Algorithm 1: every state reaches $x_e$",
        GOOD, set(), highlight={(2, 8)},
    )
    axes[0].text(0.0, -2.1, "beneficial cycle", fontsize=9.5, color=TRAP, ha="center")
    axes[1].text(1.15, -2.1, "escape arc paid by $\\mu^*$", fontsize=9.5,
                 color=GOOD, ha="center")
    fig.tight_layout(pad=0.3)
    out = os.path.abspath(os.path.join(FIGURES, "toy_bcn_stg.pdf"))
    fig.savefig(out, bbox_inches="tight", transparent=True)
    fig.savefig(out.replace(".pdf", ".png"), bbox_inches="tight", dpi=200)
    plt.close(fig)
    return out


def figure_threshold(lam_star: float) -> str:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    lams = [i / 400 for i in range(1, 400)]
    Vs, Js = [], []
    for lam in lams:
        sol = solve_at(lam)
        Vs.append(sol["V"][2])
        Js.append(sol["J"][2])

    fig, ax = plt.subplots(figsize=(5.4, 3.2))
    ax.plot(lams, Js, color=GOOD, lw=2.0,
            label=r"$J_\lambda^*(\delta_8^2)$  (proper, Algorithm 1)")
    ax.plot(lams, Vs, color=TRAP, lw=2.0, ls="--",
            label=r"$V_\lambda^*(\delta_8^2)$  (value/policy iteration)")
    ax.fill_between(lams, Vs, Js, where=[j > v + 1e-12 for v, j in zip(Vs, Js)],
                    color=TRAP, alpha=0.10, lw=0)
    ax.axvline(lam_star, color="#555555", lw=1.0, ls=":")
    ax.annotate(
        rf"$\lambda^\star=\frac{{1+\sqrt{{13}}}}{{6}}\approx{lam_star:.4f}$",
        xy=(lam_star, 1.0), xytext=(lam_star - 0.05, 4.2),
        fontsize=9.5, ha="right", color="#333333",
    )
    ax.text(0.34, 2.35, "beneficial cycle:\nboth iterations improper",
            fontsize=9, color=TRAP, ha="center")
    ax.set_xlabel(r"discount factor $\lambda$")
    ax.set_ylabel("discounted cost at $\\delta_8^2$")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 5.6)
    ax.legend(fontsize=9, loc="upper left", frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(pad=0.3)
    out = os.path.abspath(os.path.join(FIGURES, "toy_bcn_threshold.pdf"))
    fig.savefig(out, bbox_inches="tight", transparent=True)
    fig.savefig(out.replace(".pdf", ".png"), bbox_inches="tight", dpi=200)
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------


def run() -> dict:
    os.makedirs(RESULTS, exist_ok=True)
    os.makedirs(FIGURES, exist_ok=True)

    sol = solve_at(LAM)
    G, R, V, J = sol["G"], sol["R"], sol["V"], sol["J"]
    V_pi = sol["V_pi"]

    scc = condense(G, R, XE)
    nontrivial = [c for c in scc.components if len(c) > 1]

    internal, e = augment_component(G, CYCLE, J, LAM)
    has_bc, trapped_component, V_comp = detect_beneficial_cycle_v2(
        CYCLE, internal, e, LAM
    )
    J_core, _pol_core, _ = solve_core_heldkarp(CYCLE, internal, e, LAM)
    J_brute = brute_force_simple_paths(CYCLE, internal, e, LAM)
    cycles = enumerate_beneficial_cycles(CYCLE, internal, e, LAM, J)

    bad_vi = improper_states(sol["mu_vi"], R)
    bad_pi = improper_states(sol["mu_pi"], R)
    bad_master = improper_states(sol["mu_master"], R)
    gap = {x for x in R if J[x] > V[x] + 1e-9}

    lam_star = lambda_threshold()

    # --- hand-checked values ---------------------------------------------
    # Policy iteration is exact (rational arithmetic); value iteration only
    # converges, so it is checked against the same values up to tolerance.
    assert V_pi[2] == 2.0 and V_pi[3] == 2.0, (V_pi[2], V_pi[3])
    assert abs(V[2] - 2.0) < 1e-8 and abs(V[3] - 2.0) < 1e-8, (V[2], V[3])
    assert J[2] == 3.5 and J[3] == 2.75, (J[2], J[3])
    assert J[5] == 1.0 and J[8] == 3.0 and J[4] == 2.5
    assert gap == {2, 3}, gap
    assert bad_vi == {2, 3}, bad_vi
    assert bad_pi == {2, 3}, bad_pi
    assert bad_master == set(), bad_master
    assert has_bc and trapped_component == {2, 3}, trapped_component
    assert len(nontrivial) == 1 and nontrivial[0] == CYCLE, nontrivial
    assert e[2] == 3.5 and e[3] == float("inf"), (e[2], e[3])
    assert abs(J_core[2] - 3.5) < 1e-12 and abs(J_core[3] - 2.75) < 1e-12
    assert all(abs(J_core[x] - J_brute[x]) < 1e-12 for x in CYCLE)
    # Two input choices realise the same vertex cycle delta_8^2 -> delta_8^3.
    assert cycles and all(sorted(set(c[0])) == [2, 3] for c in cycles), cycles
    assert min(c[1] for c in cycles) == 1.5, cycles
    assert abs(lam_star - LAM_STAR) < 1e-6, (lam_star, LAM_STAR)

    fig1 = figure_stg(sol, {2, 3})
    fig2 = figure_threshold(LAM_STAR)

    g_loop = g(2, UE) + LAM * g(3, UE)
    thr = (1 - LAM**2) * J[2]

    lines = [
        f"E10: three-node BCN, N={N}, M={M}, xe=delta_8^{XE}, ue=delta_2^{UE}, lambda={LAM}",
        "  L = delta_8[" + " ".join(str(v) for v in L) + "]",
        "  g(x,u) = Hamming(x, x_e) + [u != u_e]",
        "",
        "  logical form:",
    ]
    lines += [f"    {s}" for s in logical_form()]
    lines += [
        "",
        f"  |X_s| = {len(R)} / {N},  components = {len(scc.components)},"
        f"  nontrivial = {len(nontrivial)} (S* = {sorted(CYCLE)})",
        f"  downstream folding: e(delta_8^2) = {e[2]}, e(delta_8^3) = {e[3]}",
        f"  beneficial-cycle test: {has_bc}, trapped = {sorted(trapped_component)}",
        f"  criterion: g_loop(C) = {g_loop:.4f} < (1-lambda^2) J(delta_8^2) = {thr:.4f}",
        f"  cycle oracle: {len(cycles)} beneficial cycle(s) {[c[0] for c in cycles]}",
        "",
        "  state  bits     d  V_lambda*  J_lambda*   VI     PI     master",
    ]
    for x in range(1, N + 1):
        b = "".join(str(t) for t in bits(x))
        mv = sol["mu_vi"].get(x)
        mp = sol["mu_pi"].get(x)
        mm = sol["mu_master"].get(x)
        lines.append(
            f"  d_8^{x}  ({b})  {hamming_to_target(x)}   {V[x]:8.4f}  {J[x]:8.4f}"
            f"   u={mv[0] if mv else '-'}   u={mp[0] if mp else '-'}   u={mm[0] if mm else '-'}"
        )
    lines += [
        "",
        f"  V != J states: {sorted(gap)}",
        f"  value-iteration improper states:  {sorted(bad_vi)}",
        f"  policy-iteration improper states: {sorted(bad_pi)}",
        f"  master improper states:           {sorted(bad_master)}",
        f"  SolveCore J = {{2: {J_core[2]}, 3: {J_core[3]}}}, brute force agrees",
        "",
        f"  beneficial-cycle threshold: bisection {lam_star:.9f},"
        f" closed form (1+sqrt(13))/6 = {LAM_STAR:.9f}",
        f"  figures: {os.path.basename(fig1)}, {os.path.basename(fig2)}",
        "  ASSERTIONS PASSED",
    ]
    text = "\n".join(lines)
    with open(os.path.join(RESULTS, "exp10_toy_bcn.txt"), "w") as fh:
        fh.write(text + "\n")
    print(text)
    return {
        "V2": V[2], "J2": J[2], "V3": V[3], "J3": J[3],
        "vi_improper": sorted(bad_vi), "pi_improper": sorted(bad_pi),
        "master_improper": sorted(bad_master), "lambda_star": lam_star,
    }


if __name__ == "__main__":
    run()
